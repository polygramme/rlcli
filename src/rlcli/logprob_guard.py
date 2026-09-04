"""Trainer-vs-engine logprob replay check.

The cookbook already logs ``optim/kl_sample_train_v1``, but that is a *signed*
mean logprob difference: symmetric divergence cancels, so a badly mismatched
batch can sit near zero.  The quantity that actually catches a broken
token pipeline is the **mean absolute** difference over action tokens — the
metric the apex recipe plots as
``policy/minibatch_rollout_logprobs_abs_diff_mean``, where "below 0.03 is
usually a healthy sign".  Their two published runs sit at roughly 0.012 and
0.017; a Polygramme smoke run measured 0.004.

This is the check that would catch a bridge splicing tokens the policy never
conditioned on.  Prime's contract makes that unreachable and
``BridgingRenderer`` re-verifies the prefix anyway, but both of those check the
*tokens*; this checks the *probabilities*, which is the only thing that
notices if the trainer and the sampler disagree about what those tokens mean.

Installed by patching the name ``tinker_cookbook.rl.train`` bound at import
time, so the extra metrics ride along in the normal metrics dict.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Mercor's stated healthy band. Above this, suspect the token pipeline.
DEFAULT_THRESHOLD = 0.03

ABS_DIFF_KEY = "optim/rollout_logprobs_abs_diff_mean"
MAX_DIFF_KEY = "optim/rollout_logprobs_abs_diff_max"
BREACH_KEY = "optim/rollout_logprobs_breach"


class LogprobMismatch(RuntimeError):
    """Raised in strict mode when trainer and sampler logprobs diverge."""


def abs_diff_metrics(data_D: list[Any], training_logprobs_D: list[Any]) -> dict[str, float]:
    """Mean and max absolute logprob difference over action tokens."""
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
    """Add the abs-diff metrics to every training step's metrics dict.

    ``strict`` decides what a breach costs.  The default is to log at ERROR and
    publish ``optim/rollout_logprobs_breach``, not to kill the run: a single
    spike can be legitimate (a large learning rate, a stale policy), and failing
    a customer's run on a diagnostic is the wrong default.  Set ``strict`` when
    you are specifically bisecting a token-pipeline change, where a breach means
    the tokens are wrong and every further step is wasted compute.

    Returns ``False`` if the cookbook's internals moved and the hook could not be
    installed, so the caller can log that rather than believe it is protected.
    """
    try:
        from tinker_cookbook.rl import train as rl_train
    except ImportError:
        logger.warning("tinker_cookbook.rl.train unavailable; logprob guard not installed")
        return False

    original = getattr(rl_train, "compute_kl_sample_train", None)
    if original is None:
        logger.warning("compute_kl_sample_train not found; logprob guard not installed")
        return False
    if getattr(original, "_pg_logprob_guard", False):
        return True

    def guarded(data_D, training_logprobs_D):
        metrics = dict(original(data_D, training_logprobs_D))
        try:
            extra = abs_diff_metrics(data_D, training_logprobs_D)
        except Exception:
            logger.warning("logprob abs-diff metric failed", exc_info=True)
            return metrics

        metrics.update(extra)
        mean = extra.get(ABS_DIFF_KEY)
        if mean is None:
            return metrics

        breached = mean > threshold
        metrics[BREACH_KEY] = 1.0 if breached else 0.0
        if breached:
            message = (
                f"trainer and sampler logprobs disagree: mean |diff| {mean:.4f} "
                f"over action tokens, above the {threshold} healthy threshold "
                f"(max {extra.get(MAX_DIFF_KEY, float('nan')):.4f}). "
                "Suspect the token pipeline before the optimizer."
            )
            if strict:
                raise LogprobMismatch(message)
            logger.error(message)
        return metrics

    guarded._pg_logprob_guard = True  # type: ignore[attr-defined]
    rl_train.compute_kl_sample_train = guarded  # type: ignore[assignment]
    logger.info("logprob guard installed (threshold %s, strict=%s)", threshold, strict)
    return True
