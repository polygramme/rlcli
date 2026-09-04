"""Trainer-vs-engine logprob replay check.

The cookbook already logs ``optim/kl_sample_train_v1``, but that is a *signed*
mean logprob difference: symmetric divergence cancels, so a badly mismatched
batch can sit near zero.  The quantity that actually catches a broken token
pipeline is the **mean absolute** difference over action tokens — the metric the
apex recipe plots as ``policy/minibatch_rollout_logprobs_abs_diff_mean``, where
"below 0.03 is usually a healthy sign".  Their two published runs sit at roughly
0.012 and 0.017; a Polygramme smoke run measured 0.004.

Where the number comes from
---------------------------
Our SkyRL backend already computes this on the policy workers
(``compute_minibatch_rollout_logprob_diff_metrics``) and returns the family —
mean, std, max, min — in ``ForwardBackwardOutput.metrics``.  It is computed on
GPU against the logprobs the loss actually optimizes against, and it is already
on the wire.

But ``tinker_cookbook.rl.train`` never reads that field.  It pulls
``loss_fn_outputs`` out of the forward-backward result and drops ``.metrics``
entirely; only the distillation recipes (``sdft.py``, ``train_off_policy.py``)
consume it.  So on the RL path the number is computed, transmitted, and thrown
away.

This module therefore does two things:

1. Harvests ``.metrics`` off each forward-backward result, which is free and
   strictly richer than anything we can compute client-side.
2. Falls back to computing the mean locally from the datums when the server
   did not supply it — a different backend, hosted Tinker, or a batch with no
   rollout logprobs.

Either way the same keys are published and the same threshold is checked, so
the guard behaves identically no matter which trainer is on the other end.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Mercor's stated healthy band. Above this, suspect the token pipeline.
DEFAULT_THRESHOLD = 0.03

ABS_DIFF_KEY = "optim/rollout_logprobs_abs_diff_mean"
MAX_DIFF_KEY = "optim/rollout_logprobs_abs_diff_max"
STD_DIFF_KEY = "optim/rollout_logprobs_abs_diff_std"
BREACH_KEY = "optim/rollout_logprobs_breach"
# 1.0 when the number came from the trainer, 0.0 when we computed it here.
SOURCE_KEY = "optim/rollout_logprobs_from_server"

# What skyrl_train_backend._extract_metrics emits. The ``:mean`` / ``:max``
# suffixes drive the SDK's cross-chunk reduction and may or may not survive into
# the client-visible dict, so both spellings are accepted.
_SERVER_STEM = "policy/rollout_train_logprobs_abs_diff"


class LogprobMismatch(RuntimeError):
    """Raised in strict mode when trainer and sampler logprobs diverge."""


# Forward-backward metrics harvested since the last training step. The RL loop
# is sequential -- every forward_backward for a step completes before
# compute_kl_sample_train runs -- so a plain list needs no locking. Bounded so
# that a code path which forward-backwards without ever reaching the drain
# cannot grow without limit or carry stale steps into a later average.
_MAX_HARVEST = 256
_harvested: list[dict[str, float]] = []


def _harvest(metrics: dict[str, float]) -> None:
    _harvested.append(metrics)
    if len(_harvested) > _MAX_HARVEST:
        del _harvested[: len(_harvested) - _MAX_HARVEST]


def _pick(metrics: dict[str, Any], suffix: str) -> float | None:
    """Read ``<stem>_<suffix>`` whether or not the reduction suffix survived."""
    for key in (f"{_SERVER_STEM}_{suffix}", f"{_SERVER_STEM}_{suffix}:{suffix}"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    # ``:mean`` is the reduction for every key in the family except max/min.
    value = metrics.get(f"{_SERVER_STEM}_{suffix}:mean")
    return float(value) if isinstance(value, (int, float)) else None


def server_metrics() -> dict[str, float]:
    """Drain the harvested forward-backward metrics into our key names.

    Several forward-backward calls make up one training step (substeps,
    minibatches), so the means are averaged and the maxima maxed -- the same
    reduction the backend applies across micro-batches and DP ranks.
    """
    batches = [m for m in _harvested if m]
    _harvested.clear()
    means = [v for m in batches if (v := _pick(m, "mean")) is not None]
    if not means:
        return {}

    out = {ABS_DIFF_KEY: sum(means) / len(means)}
    maxes = [v for m in batches if (v := _pick(m, "max")) is not None]
    if maxes:
        out[MAX_DIFF_KEY] = max(maxes)
    stds = [v for m in batches if (v := _pick(m, "std")) is not None]
    if stds:
        out[STD_DIFF_KEY] = sum(stds) / len(stds)
    return out


def abs_diff_metrics(data_D: list[Any], training_logprobs_D: list[Any]) -> dict[str, float]:
    """Client-side fallback: mean and max |diff| over action tokens."""
    import torch

    diffs: list[Any] = []
    for datum, training_logprobs in zip(data_D, training_logprobs_D):
        inputs = datum.loss_fn_inputs
        if "logprobs" not in inputs or "mask" not in inputs:
            continue
        sampling = inputs["logprobs"].to_torch()
        action = inputs["mask"].to_torch() > 0
        if action.sum() == 0:
            continue
        diffs.append((training_logprobs[action] - sampling[action]).abs())

    if not diffs:
        return {}
    joined = torch.cat(diffs)
    return {
        ABS_DIFF_KEY: float(joined.mean()),
        MAX_DIFF_KEY: float(joined.max()),
    }


def install(threshold: float = DEFAULT_THRESHOLD, *, strict: bool = False) -> bool:
    """Publish the abs-diff family on every training step and check it.

    ``strict`` decides what a breach costs.  The default is to log at ERROR and
    publish ``optim/rollout_logprobs_breach``, not to kill the run: a single
    spike can be legitimate (a large learning rate, a stale policy), and failing
    a customer's run on a diagnostic is the wrong default.  Set ``strict`` when
    bisecting a token-pipeline change, where a breach means the tokens are wrong
    and every further step is wasted compute.

    Returns ``False`` if the cookbook's internals moved and the hooks could not
    be installed, so the caller can log that rather than believe it is protected.
    """
    try:
        from tinker_cookbook.rl import train as rl_train
    except ImportError:
        logger.warning("tinker_cookbook.rl.train unavailable; logprob guard not installed")
        return False

    original = getattr(rl_train, "compute_kl_sample_train", None)
    harvest_target = getattr(rl_train, "_training_logprobs_from_fwd_bwd", None)
    if original is None or harvest_target is None:
        logger.warning("cookbook hooks not found; logprob guard not installed")
        return False
    if getattr(original, "_pg_logprob_guard", False):
        return True

    def harvesting(fwd_bwd_result):
        # The RL path reads only loss_fn_outputs and drops .metrics, which is
        # where the backend's GPU-computed logprob-gap family lives. Take a copy
        # on the way past; the wrapped function is otherwise untouched.
        try:
            metrics = getattr(fwd_bwd_result, "metrics", None)
            if metrics:
                _harvest(dict(metrics))
        except Exception:
            logger.debug("could not harvest forward-backward metrics", exc_info=True)
        return harvest_target(fwd_bwd_result)

    def guarded(data_D, training_logprobs_D):
        metrics = dict(original(data_D, training_logprobs_D))
        try:
            extra = server_metrics()
            from_server = bool(extra)
            if not from_server:
                extra = abs_diff_metrics(data_D, training_logprobs_D)
        except Exception:
            logger.warning("logprob abs-diff metric failed", exc_info=True)
            _harvested.clear()
            return metrics

        if not extra:
            return metrics

        metrics.update(extra)
        metrics[SOURCE_KEY] = 1.0 if from_server else 0.0
        mean = extra[ABS_DIFF_KEY]

        breached = mean > threshold
        metrics[BREACH_KEY] = 1.0 if breached else 0.0
        if breached:
            message = (
                f"trainer and sampler logprobs disagree: mean |diff| {mean:.4f} "
                f"over action tokens, above the {threshold} healthy threshold "
                f"({'trainer-reported' if from_server else 'computed locally'}). "
                "Suspect the token pipeline before the optimizer."
            )
            if strict:
                raise LogprobMismatch(message)
            logger.error(message)
        return metrics

    guarded._pg_logprob_guard = True  # type: ignore[attr-defined]
    rl_train._training_logprobs_from_fwd_bwd = harvesting  # type: ignore[assignment]
    rl_train.compute_kl_sample_train = guarded  # type: ignore[assignment]
    logger.info("logprob guard installed (threshold %s, strict=%s)", threshold, strict)
    return True
