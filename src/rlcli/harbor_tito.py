"""Harbor RL dataset builder that installs the TITO bridge on every env.

Thin wiring on top of :mod:`rlcli.tito_bridge`.  The cookbook's
``HarborEnvGroupBuilder`` builds one renderer and shares it across the group, so
the bridge is installed per env *after* ``make_envs`` returns, by swapping
``EnvFromMessageEnv.renderer``.

We wrap ``make_envs`` as a closure rather than subclassing
``HarborEnvGroupBuilder``, because subclassing would mean restating its whole
constructor signature in ``_make_env_group_builders`` and re-editing this file
every time upstream adds a field.
"""

from __future__ import annotations

import contextvars
import time

import logging
from typing import Any

import chz
from tinker_cookbook import tokenizer_utils
from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    HarborDatasetBuilder,
    HarborEnvGroupBuilder,
)

from rlcli.tito_bridge import BridgingRenderer, prime_renderer_for

logger = logging.getLogger(__name__)


# Grader (test.sh) duration per episode. HarborReward is built per env deep
# inside make_envs, so its __call__ is timed once at the class and the value
# handed back through a ContextVar that the terminal step wrapper sets.
_GRADER: contextvars.ContextVar[dict | None] = contextvars.ContextVar("pg_grader", default=None)


def patch_timed_reward(cls) -> bool:
    """Time cls.__call__ (async) and report it via _GRADER. Idempotent."""
    orig = cls.__call__
    if getattr(orig, "_pg_timed", False):
        return False

    async def timed(self, *a, **kw):
        t0 = time.monotonic()
        try:
            return await orig(self, *a, **kw)
        finally:
            holder = _GRADER.get()
            if holder is not None:
                holder["grader_s"] = time.monotonic() - t0

    timed._pg_timed = True  # type: ignore[attr-defined]
    cls.__call__ = timed
    return True


def _patch_harbor_reward() -> None:
    try:
        from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborReward

        patch_timed_reward(HarborReward)
    except Exception:  # noqa: BLE001 - optional; metrics just stay absent
        pass


def make_parse_error_policy(max_consecutive: int):
    """Recover from unparsable tool calls instead of ending the episode.

    A content parse failure (clean framing, malformed tool call) is recoverable:
    the cookbook injects a corrective message and lets the rollout continue.
    Without this, the first malformed tool call ends the episode and the
    trajectory is scored as if the model gave up, which reads as a bad model
    rather than a bad parse.

    Cost, worth stating plainly: the corrective message the cookbook injects has
    ``role="user"``, so under Qwen3.5's ``tool_cycle`` retention the bridge
    refuses that one turn and it re-renders. Bridging resumes on the next turn,
    because the bridge only inspects messages added since the last prompt. So a
    parse error costs one extra datum, not the episode's appending property.

    ``mask_error_turns`` keeps the malformed turn out of the gradient: it stays
    in the conversation as context, but contributes no training signal.
    """
    from tinker_cookbook.rl.rollout_limits import ParseErrorPolicy

    return ParseErrorPolicy(max_consecutive=max_consecutive, mask_error_turns=True)


# One Prime renderer per model for the whole run. It is stateless, and building
# one means a fresh LockedTokenizer from the registry factory (~0.9s warm) —
# per group, per batch, that silently doubled the cookbook's own per-group
# tokenizer construction (~7s a batch at groups_per_batch=8). Sharing it means
# every env serialises on one tokenizer lock rather than one per group;
# tokenization is milliseconds against sandbox and sampling time, so that is
# the right trade.
_PRIME: dict[str, Any] = {}


def _prime_for(model_name: str, renderer_name: str | None):
    key = (model_name, renderer_name)
    if key not in _PRIME:
        tokenizer = tokenizer_utils.get_tokenizer(model_name)
        _PRIME[key] = prime_renderer_for(model_name, tokenizer, renderer_name)
    return _PRIME[key]


def _surface_bridge_metrics(env) -> None:
    """Fold the bridge's counters into the episode's final StepResult.metrics.

    That is the one seam the cookbook aggregates into the run's metrics.jsonl,
    so a run that re-renders every turn (bridge refused, unknown renderer,
    contract violation) is visible in the dashboard instead of only in a
    counter nothing reads. Reported once, on the terminal step, so each
    episode contributes its totals exactly once.
    """
    renderer, orig_step = env.renderer, env.step

    async def step(action, *args, **kwargs):
        holder: dict = {}
        token = _GRADER.set(holder)
        try:
            result = await orig_step(action, *args, **kwargs)
        finally:
            _GRADER.reset(token)
        if result.episode_done:
            result.metrics.update(renderer.metrics())
            if "grader_s" in holder:
                result.metrics["grader/duration_s"] = float(holder["grader_s"])
            sb = getattr(env, "_pg_sandbox", None)
            if sb:
                result.metrics["sandbox/create_s"] = float(sb.get("create_s") or 0.0)
                result.metrics["sandbox/failed_in_group"] = float(sb.get("failed_in_group") or 0)
            rec = getattr(env, "_pg_recorder", None)
            if rec is not None:
                try:
                    result.metrics.update(rec.metrics())
                except Exception:  # noqa: BLE001 - counters must never fail a step
                    pass
            # One "went wrong" flag per episode: an unparsable tool call or a
            # TITO contract violation. The cookbook's mean = the error rate.
            bad = (result.metrics.get("errors/parse_episode") or 0) > 0 or (result.metrics.get("tito_contract_violations") or 0) > 0
            result.metrics["errors/any"] = 1.0 if bad else 0.0
        return result

    env.step = step


def time_sandboxes(builder) -> None:
    """Wrap builder.sandbox_factory to record creation time per sandbox and
    failures per group (builder._pg_sandbox_stats). Idempotent."""
    factory = getattr(builder, "sandbox_factory", None)
    if factory is None or getattr(factory, "_pg_timed", False):
        return
    stats = {"create_s": [], "failed": 0}
    builder._pg_sandbox_stats = stats

    async def timed(env_dir, timeout, *a, **kw):
        t0 = time.monotonic()
        try:
            sb = await factory(env_dir, timeout, *a, **kw)
        except Exception:
            stats["failed"] += 1
            raise
        stats["create_s"].append(time.monotonic() - t0)
        return sb

    timed._pg_timed = True  # type: ignore[attr-defined]
    builder.sandbox_factory = timed


def install_bridge(
    builder: HarborEnvGroupBuilder,
    *,
    budget_warning_ratio: float | None,
    budget_warning_text: str | None,
    parse_error_retries: int = 0,
) -> None:
    """Make ``builder.make_envs`` hand back envs with bridging renderers."""
    original_make_envs = builder.make_envs
    _patch_harbor_reward()
    time_sandboxes(builder)

    async def make_envs():
        envs = await original_make_envs()
        # ATIF trajectory per episode when a sink is configured (rlcli.atif).
        # Installed before the bridge so the recorder sees the bridge's token
        # counts on the renderer and the terminal StepResult on env.step.
        from rlcli import atif

        stats = getattr(builder, "_pg_sandbox_stats", None) or {}
        for traj_idx, env in enumerate(envs):
            atif.install_recorder(env, traj_idx=traj_idx, model_name=getattr(builder, "model_name", None))
            durations = stats.get("create_s") or []
            env._pg_sandbox = {"create_s": durations[traj_idx] if traj_idx < len(durations) else None,
                               "failed_in_group": stats.get("failed", 0)}

        if parse_error_retries > 0:
            policy = make_parse_error_policy(parse_error_retries)
            for env in envs:
                # Not a plain attribute: EnvFromMessageEnv.set_parse_error_policy
                # pushes the policy down to the inner AgentToolMessageEnv, which
                # is where malformed tool calls are actually handled. Setting the
                # outer attribute alone delivered zero retries while still
                # changing the outer env's structural-failure reward.
                env.set_parse_error_policy(policy)

        prime = _prime_for(builder.model_name, builder.renderer_name)
        if prime is None:
            # Driver log, not the cookbook's logs.log: this is the line an
            # operator needs when a run is unexpectedly 4x more expensive.
            print(f"[tito] no Prime renderer for {builder.model_name!r} "
                  f"(renderer_name={builder.renderer_name!r}); every turn will re-render. "
                  "The wrap-up nudge still applies.")
        for env in envs:
            # prime=None keeps the nudge and the fallback path; the bridge simply
            # never fires. The nudge is a tool-message append and is safe with
            # or without bridging.
            env.renderer = BridgingRenderer(
                env.renderer,
                prime,
                max_context_tokens=builder.max_trajectory_tokens,
                budget_warning_ratio=budget_warning_ratio,
                budget_warning_text=budget_warning_text,
            )
            _surface_bridge_metrics(env)
        return envs

    builder.make_envs = make_envs


def limit_eval_dataset(dataset, limit: int):
    """First ``limit`` env groups of a HarborDataset-shaped dataset (same type)."""
    builders = getattr(dataset, "env_group_builders", None)
    batch_size = getattr(dataset, "batch_size", None)
    if builders is None or batch_size is None or limit <= 0 or len(builders) <= limit:
        return dataset
    return type(dataset)(env_group_builders=list(builders)[:limit], batch_size=batch_size)


@chz.chz
class BridgingHarborDatasetBuilder(HarborDatasetBuilder):
    """``HarborDatasetBuilder`` with token-in/token-out rollouts.

    ``budget_warning_ratio`` fires the wrap-up nudge once remaining context falls
    to that fraction of the window (0.2 = at 20% remaining); ``None`` disables it.
    """

    budget_warning_ratio: float | None = 0.2
    budget_warning_text: str | None = None
    parse_error_retries: int = 2
    # The cookbook evaluates on Harbor's eval dataset = every task, one rollout
    # each, at step 0 and every ``eval_every`` steps. On a 200-task dataset that
    # is 200 sandboxes per eval; cap it to the first N tasks (None = all).
    eval_task_limit: int | None = None

    async def __call__(self):
        # Captured samples need to know which batch they came from; the cookbook
        # loop never says, so the dataset stamps it onto each batch's builders.
        from rlcli.trace_capture import StampingDataset

        dataset, test_dataset = await super().__call__()
        if test_dataset is not None and self.eval_task_limit:
            test_dataset = limit_eval_dataset(test_dataset, int(self.eval_task_limit))
        if test_dataset is not None:
            # Task identity only: the eval scope owns iteration/split.
            test_dataset = StampingDataset(test_dataset, batch_coords=False)
        return StampingDataset(dataset), test_dataset

    def _make_env_group_builders(self, group_size: int) -> list[HarborEnvGroupBuilder]:
        builders = super()._make_env_group_builders(group_size)
        for builder in builders:
            install_bridge(
                builder,
                budget_warning_ratio=self.budget_warning_ratio,
                budget_warning_text=self.budget_warning_text,
                parse_error_retries=self.parse_error_retries,
            )
        return builders
