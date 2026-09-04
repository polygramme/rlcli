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


def _prime_for(model_name: str):
    if model_name not in _PRIME:
        tokenizer = tokenizer_utils.get_tokenizer(model_name)
        _PRIME[model_name] = prime_renderer_for(model_name, tokenizer)
    return _PRIME[model_name]


def install_bridge(
    builder: HarborEnvGroupBuilder,
    *,
    budget_warning_ratio: float | None,
    budget_warning_text: str | None,
    parse_error_retries: int = 0,
) -> None:
    """Make ``builder.make_envs`` hand back envs with bridging renderers."""
    original_make_envs = builder.make_envs

    async def make_envs():
        envs = await original_make_envs()

        if parse_error_retries > 0:
            policy = make_parse_error_policy(parse_error_retries)
            for env in envs:
                # Plain attribute on EnvFromMessageEnv, same as `renderer`; it is
                # not plumbed through rl_train.Config, so we set it here.
                env.parse_error_policy = policy

        prime = _prime_for(builder.model_name)
        if prime is None:
            # No hand-written renderer for this model: leave the group exactly as
            # the cookbook built it. Every turn re-renders, which is today's
            # behaviour, and the nudge goes with it (it is only safe to inject
            # inside an observation the bridge is going to keep appending to).
            return envs
        for env in envs:
            env.renderer = BridgingRenderer(
                env.renderer,
                prime,
                max_context_tokens=builder.max_trajectory_tokens,
                budget_warning_ratio=budget_warning_ratio,
                budget_warning_text=budget_warning_text,
            )
        return envs

    builder.make_envs = make_envs


@chz.chz
class BridgingHarborDatasetBuilder(HarborDatasetBuilder):
    """``HarborDatasetBuilder`` with token-in/token-out rollouts.

    ``budget_warning_ratio`` fires the wrap-up nudge once remaining context falls
    to that fraction of the window (0.2 = at 20% remaining); ``None`` disables it.
    """

    budget_warning_ratio: float | None = 0.2
    budget_warning_text: str | None = None
    parse_error_retries: int = 2

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
