"""Token-in / token-out bridging for the cookbook's Harbor RL loop.

The problem this solves
-----------------------
``tinker_cookbook``'s ``EnvFromMessageEnv`` builds each turn's observation by
re-rendering the *entire* conversation (``rl/message_env.py:222``).  On Qwen3.5
the hand-written ``Qwen3_5Renderer`` drops ``<think>`` blocks from history when
it re-renders, so turn N+1's prompt is not an extension of turn N's
prompt+completion.  ``rl/data_processing.py``'s ``_is_prefix`` check is exact,
so every turn closes its datum and opens a new one.

That is *not* an off-policy bug — the rollout runner hands the same
``ModelInput`` to the sampler and to the ``Transition`` (``rollout_runner.py``
lines 417/430/535), so we always train on exactly what we sampled from.  It is
a throughput bug.  Measured on an 8-turn Qwen3.5-9B episode with realistic
reasoning text: 8 datums, 5388 trainer tokens, versus 1 datum and 1351 tokens
when the sequence appends cleanly.

The fix
-------
Wrap the renderer.  ``BridgingRenderer`` watches the two calls the env already
makes — ``parse_response(action)`` gives us the sampled tokens, and the next
``build_generation_prompt(messages)`` asks for the next prompt — and splices
them with PrimeIntellect ``renderers``' ``bridge_to_next_turn``, which keeps the
sampled tokens verbatim instead of re-rendering them.

Nothing about the env changes: ``EnvFromMessageEnv.renderer`` is a plain
attribute, so swapping it after construction intercepts every render path
(initial observation, normal step, truncation-continue, message injection).

Why a wrapper per env
---------------------
``HarborEnvGroupBuilder.make_envs`` creates *one* renderer and shares it across
every env in the group ("stateless, shared across envs").  Bridging state is
per-conversation, so each env gets its own wrapper; the underlying cookbook and
Prime renderers stay shared, since both are stateless.

Why we verify the contract
--------------------------
Prime documents that the returned sequence starts with
``previous_prompt + previous_completion`` unchanged, and returns ``None``
whenever it cannot prove that.  We re-check the prefix anyway before accepting a
bridge: a violated prefix is precisely the silent off-policy corruption TITO
exists to prevent, and it would otherwise be invisible until a training curve
looked wrong weeks later.
"""

from __future__ import annotations

import logging
from typing import Any

import tinker

logger = logging.getLogger(__name__)

# Matches the apex recipe's default. The ``{pct}`` placeholder is filled with the
# percentage of the context window still available.
DEFAULT_BUDGET_WARNING = (
    "[SYSTEM NOTICE] You are almost out of context budget (~{pct}% of the "
    "context window remains). Limit the number of tool calls and provide your "
    "final answer soon."
)

Message = dict[str, Any]


def append_budget_warning(new_messages: list[Message], warning: str) -> bool:
    """Append ``warning`` to the most recent tool message in ``new_messages``.

    Mutates the message dict in place.  The dicts are shared with the env's
    running history, so the warning is also what the model is trained on, not
    just what it sees.

    The warning rides *inside* a tool observation rather than arriving as its own
    user turn, and that is load-bearing rather than stylistic: Qwen3.5 resolves
    to ``thinking_retention="tool_cycle"``, under which Prime's bridge refuses to
    extend across a new user query and falls back to a full re-render.  A nudge
    written as a user message would silently switch bridging off for the rest of
    the episode.

    Only messages the caller has not yet tokenized are eligible — appending to an
    already-rendered tool message would put text into a region the bridge treats
    as opaque prior tokens, where the model would never see it.

    Returns ``True`` when a tool message was found and updated.
    """
    for msg in reversed(new_messages):
        if msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            # Structured content parts (the shape multimodal renderers use).
            content.append({"type": "text", "text": warning})
        else:
            msg["content"] = f"{content}\n\n{warning}" if content else warning
        return True
    return False


class BridgingRenderer:
    """Cookbook-renderer facade that produces a strictly appending token stream.

    Delegates everything it does not override to the wrapped cookbook renderer,
    so it is a drop-in for ``EnvFromMessageEnv.renderer``.

    Args:
        inner: The cookbook renderer to fall back to and delegate to.
        prime: A PrimeIntellect renderer for the same model, or ``None`` to
            disable bridging entirely (every turn then re-renders, i.e. exactly
            today's behaviour).
        max_context_tokens: Context budget used for the wrap-up nudge. ``None``
            disables the nudge.
        budget_warning_ratio: Fire the nudge once remaining budget falls to this
            fraction of the window (0.2 = at 20% remaining). ``None`` disables.
        budget_warning_text: Override for :data:`DEFAULT_BUDGET_WARNING`.
    """

    def __init__(
        self,
        inner: Any,
        prime: Any,
        *,
        max_context_tokens: int | None = None,
        budget_warning_ratio: float | None = None,
        budget_warning_text: str | None = None,
    ):
        self._inner = inner
        self._prime = prime
        self._max_context = max_context_tokens
        self._warn_ratio = budget_warning_ratio
        self._warn_text = budget_warning_text or DEFAULT_BUDGET_WARNING

        # Bridging state, all per-conversation.
        self._prompt_ids: list[int] | None = None
        self._completion_ids: list[int] | None = None
        self._n_messages: int = 0

        # Counters, surfaced as rollout metrics.
        self.bridged = 0
        self.fell_back = 0
        self.nudges = 0
        self.contract_violations = 0

    # ---- the two hooks the env already calls -------------------------------

    def parse_response(self, response: list[int]):
        """Record the sampled tokens, then defer to the wrapped renderer.

        ``response`` is the raw action from the sampler.  Recording it here is
        what lets the next ``build_generation_prompt`` splice rather than
        re-render.  A turn whose parse fails structurally ends the episode with
        an empty observation, so a stale completion recorded here is never used.
        """
        self._completion_ids = list(response)
        return self._inner.parse_response(response)

    def build_generation_prompt(self, messages: list[Message], **kwargs) -> tinker.ModelInput:
        """Return the next prompt, extending the prior tokens where possible."""
        have_state = self._prompt_ids is not None and self._completion_ids is not None

        # Everything after the assistant turn we just sampled. The env appends
        # exactly one assistant message at index ``_n_messages`` and then the
        # observations that follow it.
        new_messages = messages[self._n_messages + 1 :] if have_state else []

        if new_messages:
            remaining = self._remaining_after(new_messages)
            if remaining is not None and remaining <= self._warn_ratio * self._max_context:
                if append_budget_warning(new_messages, self._warning_text(remaining)):
                    self.nudges += 1

        # ``kwargs`` means a caller wants non-default rendering; don't try to
        # reproduce that through the bridge, just re-render.
        if have_state and new_messages and not kwargs:
            ids = self._try_bridge(new_messages)
            if ids is not None:
                self.bridged += 1
                self._remember(ids, len(messages))
                return tinker.ModelInput.from_ints(ids)
            self.fell_back += 1

        model_input = self._inner.build_generation_prompt(messages, **kwargs)
        self._remember(model_input.to_ints(), len(messages))
        return model_input

    # ---- internals ---------------------------------------------------------

    def _remember(self, prompt_ids: list[int], n_messages: int) -> None:
        self._prompt_ids = list(prompt_ids)
        self._completion_ids = None
        self._n_messages = n_messages

    def _try_bridge(self, new_messages: list[Message]) -> list[int] | None:
        """Splice the new messages onto the prior tokens, or ``None`` to fall back."""
        if self._prime is None:
            return None
        assert self._prompt_ids is not None and self._completion_ids is not None
        try:
            out = self._prime.bridge_to_next_turn(
                self._prompt_ids,
                self._completion_ids,
                list(new_messages),
                # Harbor puts the tool schemas in the system prompt text
                # (``_initial_messages`` -> ``create_conversation_prefix_with_tools``)
                # rather than passing them as a template kwarg, so there is no
                # tool spec to forward here.
                tools=None,
            )
        except Exception:
            logger.warning("bridge_to_next_turn raised; re-rendering", exc_info=True)
            return None

        if out is None:
            # Prime refuses when it cannot prove the prefix contract — a new user
            # query under tool_cycle retention, an assistant message in the
            # extension, an unknown template. Falling back is the correct answer.
            return None

        ids = list(out.token_ids)
        prior = self._prompt_ids + self._completion_ids
        if ids[: len(prior)] != prior:
            self.contract_violations += 1
            logger.error(
                "bridge returned a non-extending sequence (%d prior, %d returned); "
                "re-rendering. This should be unreachable.",
                len(prior),
                len(ids),
            )
            return None
        return ids

    def _observation_estimate(self, new_messages: list[Message]) -> int:
        """Tokens the observations about to be appended will add.

        The reference (``TITOAgentState.record_step``) subtracts this before
        deciding to nudge. Without it the check sees only the prior sequence:
        a 4k-token test log arriving with 7k of headroom sails past a 6.5k
        threshold, and the episode overflows before the nudge ever fires — on
        exactly the large outputs it exists for. Encoding the raw text omits
        template framing, so this runs slightly optimistic, the same direction
        as the reference.
        """
        parts = []
        for m in new_messages:
            if m.get("role") != "tool":
                continue
            c = m.get("content")
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, list):
                parts.extend(p.get("text", "") for p in c if isinstance(p, dict))
        text = "\n".join(parts)
        if not text:
            return 0
        tok = (getattr(self._inner, "tokenizer", None)
               or getattr(self._prime, "tokenizer", None)
               or getattr(self._prime, "_tokenizer", None))
        if tok is not None:
            try:
                return len(tok.encode(text, add_special_tokens=False))
            except Exception:
                pass
        return len(text) // 4  # last resort: the usual bytes-per-token rule of thumb

    def _remaining_after(self, new_messages: list[Message]) -> int | None:
        """Context left once this turn's observations land; None when the nudge is off."""
        if self._warn_ratio is None or not self._max_context or self._prompt_ids is None:
            return None
        used = (len(self._prompt_ids) + len(self._completion_ids or [])
                + self._observation_estimate(new_messages))
        return self._max_context - used

    def _warning_text(self, remaining: int) -> str:
        assert self._max_context
        pct = max(0, round(remaining / self._max_context * 100))
        return self._warn_text.replace("{pct}", str(pct))

    def metrics(self) -> dict[str, int]:
        return {
            "tito_bridged": self.bridged,
            "tito_fell_back": self.fell_back,
            "tito_nudges": self.nudges,
            "tito_contract_violations": self.contract_violations,
        }

    # ---- delegation --------------------------------------------------------

    def __getattr__(self, name: str):
        # Only reached when normal attribute lookup fails, so the fields set in
        # __init__ never come through here. Guard the sentinel so a partially
        # constructed instance raises AttributeError instead of recursing.
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


def prime_renderer_for(model_name: str, tokenizer: Any) -> Any | None:
    """Build a PrimeIntellect renderer for ``model_name``, or ``None``.

    Returns ``None`` when the package is missing or the model has no hand-written
    renderer — Prime's ``DefaultRenderer`` always refuses to bridge, so there is
    no point wrapping in that case, and saying so once at startup is clearer than
    a fallback on every turn.
    """
    try:
        from renderers import create_renderer
        from renderers.default import DefaultRenderer
    except ImportError:
        logger.info("renderers not installed; TITO bridging disabled")
        return None

    try:
        renderer = create_renderer(tokenizer)
    except Exception:
        logger.warning("could not build a Prime renderer for %s", model_name, exc_info=True)
        return None

    if isinstance(renderer, DefaultRenderer):
        logger.info(
            "%s has no hand-written Prime renderer (got DefaultRenderer, which never "
            "bridges); TITO bridging disabled",
            model_name,
        )
        return None
    logger.info("TITO bridging enabled for %s via %s", model_name, type(renderer).__name__)
    return renderer
