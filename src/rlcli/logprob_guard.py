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

Why the server's number cannot decide the breach
-----------------------------------------------
It is averaged over the wrong tokens. ``rl/train.py`` strips ``mask`` from every
datum before ``forward_backward`` (``_remove_mask``) and sends no ``weights``, so
the tinker shim defaults weights to all-ones and the backend's ``loss_mask``
covers prompt and observation tokens too. Those carry a rollout logprob of 0.0
(``data_processing.py``), so the server's mean is |train logprob| over the
prompt — whole nats — and a perfectly healthy run breaches on every step.

So the breach is decided from the local, action-masked computation, always.
The server family is still harvested and published under ``*_server_*`` keys:
it is free, it is what the backend's own dashboards show, and the gap between
the two is itself informative.
"""

from __future__ import annotations

import json
import os

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Mercor's stated healthy band. Above this, suspect the token pipeline.
DEFAULT_THRESHOLD = 0.03

ABS_DIFF_KEY = "optim/rollout_logprobs_abs_diff_mean"
MAX_DIFF_KEY = "optim/rollout_logprobs_abs_diff_max"
STD_DIFF_KEY = "optim/rollout_logprobs_abs_diff_std"
BREACH_KEY = "optim/rollout_logprobs_breach"
# The backend's own (unmasked, see module docstring) family, for comparison only.
SERVER_PREFIX = "optim/rollout_logprobs_server_"

# What skyrl_train_backend._extract_metrics emits. The ``:mean`` / ``:max``
# suffixes drive the SDK's cross-chunk reduction and may or may not survive into
# the client-visible dict, so both spellings are accepted.
_SERVER_STEM = "policy/rollout_train_logprobs_abs_diff"


class LogprobMismatch(RuntimeError):
    """Raised in strict mode when trainer and sampler logprobs diverge."""


# Live settings, read at call time so a re-install on a reused container
# takes effect without re-wrapping.
_cfg: dict[str, Any] = {"threshold": DEFAULT_THRESHOLD, "strict": False}

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


_DUMP: dict[str, Any] = {"dir": None, "n": 0}


def configure_dump(dump_dir: str | None) -> None:
    """Write every datum's tokens, sampler logprobs, trainer logprobs and action
    mask under dump_dir (one JSON line per datum). Off when None."""
    _DUMP["dir"] = dump_dir
    _DUMP["n"] = 0
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)


def _datum_tokens(datum: Any) -> list[int] | None:
    mi = getattr(datum, "model_input", None)
    if mi is None:
        return None
    try:
        return [int(t) for t in mi.to_ints()]
    except Exception:  # noqa: BLE001
        return None


def dump_datums(data_D: list[Any], training_logprobs_D: list[Any]) -> int:
    """Token-level record per datum for the console's divergence view.
    Never raises; returns how many were written."""
    d = _DUMP["dir"]
    if not d:
        return 0
    written = 0
    path = os.path.join(d, f"lp_{_DUMP['n']:06d}.jsonl")
    try:
        with open(path, "w") as f:
            for datum, train_lp in zip(data_D, training_logprobs_D):
                inputs = getattr(datum, "loss_fn_inputs", {}) or {}
                if "logprobs" not in inputs or "mask" not in inputs:
                    continue
                tokens = _datum_tokens(datum)
                if tokens is None:
                    continue
                sample = inputs["logprobs"].to_torch().tolist()
                mask = [int(m > 0) for m in inputs["mask"].to_torch().tolist()]
                train = train_lp.tolist() if hasattr(train_lp, "tolist") else list(train_lp)
                # logprobs are for tokens[1:] (next-token); pad to token length.
                off = len(tokens) - len(sample)
                rec = {"tokens": tokens, "offset": off, "sample_lp": [round(x, 5) for x in sample],
                       "train_lp": [round(float(x), 5) for x in train], "mask": mask}
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                written += 1
        _DUMP["n"] += 1
    except Exception:  # noqa: BLE001
        logger.exception("logprob dump failed")
    return written


def install(threshold: float = DEFAULT_THRESHOLD, *, strict: bool = False, dump_dir: str | None = None) -> bool:
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
    # Re-installing (a warm Modal container serving a second run) must take the
    # new threshold/strict and must not carry the previous run's harvested
    # batches into this one's first step.
    _cfg.update(threshold=threshold, strict=strict)
    _harvested.clear()
    if getattr(original, "_pg_logprob_guard", False):
        logger.info("logprob guard re-armed (threshold %s, strict=%s)", threshold, strict)
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

    configure_dump(dump_dir)

    def guarded(data_D, training_logprobs_D):
        metrics = dict(original(data_D, training_logprobs_D))
        if _DUMP["dir"]:
            n = dump_datums(data_D, training_logprobs_D)
            metrics["optim/logprob_dump_datums"] = float(n)
        # Informational: the backend's unmasked family, under its own keys.
        try:
            for k, v in server_metrics().items():
                metrics[k.replace("optim/rollout_logprobs_", SERVER_PREFIX, 1)] = v
        except Exception:
            logger.debug("server logprob family unavailable", exc_info=True)
            _harvested.clear()
        # Decisive: the action-masked local computation.
        try:
            extra = abs_diff_metrics(data_D, training_logprobs_D)
        except Exception:
            logger.warning("logprob abs-diff metric failed", exc_info=True)
            return metrics
        if not extra:
            return metrics

        metrics.update(extra)
        mean = extra[ABS_DIFF_KEY]
        threshold, strict = _cfg["threshold"], _cfg["strict"]
        breached = mean > threshold
        metrics[BREACH_KEY] = 1.0 if breached else 0.0
        if breached:
            message = (
                f"trainer and sampler logprobs disagree: mean |diff| {mean:.4f} "
                f"over action tokens, above the {threshold} healthy threshold. "
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
