"""Contract tests for the TITO bridge.

The load-bearing test is
``test_a_bridged_episode_collapses_to_one_datum``: it drives the same sequence
of calls ``EnvFromMessageEnv`` makes, then runs the cookbook's real
``trajectory_to_data`` over the result. That is the whole point of the bridge,
so it is asserted end to end rather than inferred from prefix checks.
"""

from __future__ import annotations

import pytest
import tinker

from rlcli.tito_bridge import (
    DEFAULT_BUDGET_WARNING,
    BridgingRenderer,
    append_budget_warning,
    prime_renderer_for,
)

MODEL = "Qwen/Qwen3.5-9B"

SYSTEM = {"role": "system", "content": "You are a coding agent with a bash tool."}
USER = {"role": "user", "content": "The tests fail on import. Find the root cause."}
THINK = (
    "The traceback points at app.settings. I should grep the module first, "
    "then read the file and decide whether to fix the import or the definition."
)


@pytest.fixture(scope="module")
def tokenizer():
    return pytest.importorskip("tinker_cookbook.tokenizer_utils").get_tokenizer(MODEL)


@pytest.fixture(scope="module")
def inner(tokenizer):
    from tinker_cookbook.renderers import get_renderer

    return get_renderer("qwen3_5", tokenizer)


@pytest.fixture(scope="module")
def prime(tokenizer):
    renderer = prime_renderer_for(MODEL, tokenizer)
    if renderer is None:
        pytest.skip("no Prime renderer available for this model")
    return renderer


def make(inner, prime, **kwargs):
    return BridgingRenderer(inner, prime, **kwargs)


def completion_tokens(tokenizer, step: int) -> list[int]:
    """Tokens a thinking model would actually emit for one tool-calling turn."""
    text = (
        f"{THINK}</think>\n\n"
        f"Checking step {step}.\n"
        "<tool_call>\n<function=bash>\n<parameter=cmd>\n"
        "pytest -x tests/\n</parameter>\n</function>\n</tool_call><|im_end|>"
    )
    return tokenizer.encode(text, add_special_tokens=False)


def tool_message(step: int) -> dict:
    return {
        "role": "tool",
        "tool_call_id": f"c{step}",
        "name": "bash",
        "content": "ImportError: cannot import name 'Config' from 'app.settings'",
    }


def drive(renderer, tokenizer, turns: int, observation=tool_message):
    """Replay the call sequence EnvFromMessageEnv makes, returning the transcript.

    Yields ``(prompt_ids, completion_ids)`` per turn, which is exactly what a
    ``Transition``'s ``ob`` and ``ac`` carry.
    """
    messages = [SYSTEM, USER]
    transcript = []
    for step in range(turns):
        prompt = renderer.build_generation_prompt(messages)
        action = completion_tokens(tokenizer, step)
        transcript.append((prompt.to_ints(), action))
        # The env parses the action, then appends the assistant turn plus the
        # observations that follow it.
        assistant, _ = renderer.parse_response(action)
        messages = messages + [assistant, observation(step)]
    return transcript, messages


# --------------------------------------------------------------------------
# bridging
# --------------------------------------------------------------------------


def test_the_first_prompt_delegates_to_the_inner_renderer(inner, prime, tokenizer):
    renderer = make(inner, prime)
    messages = [SYSTEM, USER]
    assert renderer.build_generation_prompt(messages).to_ints() == (
        inner.build_generation_prompt(messages).to_ints()
    )
    assert renderer.bridged == 0
    assert renderer.fell_back == 0


def test_each_prompt_extends_the_previous_prompt_plus_completion(inner, prime, tokenizer):
    renderer = make(inner, prime)
    transcript, _ = drive(renderer, tokenizer, turns=5)

    for i in range(1, len(transcript)):
        prior = transcript[i - 1][0] + transcript[i - 1][1]
        this_prompt = transcript[i][0]
        assert this_prompt[: len(prior)] == prior, f"turn {i} is not an extension"

    assert renderer.bridged == 4
    assert renderer.fell_back == 0
    assert renderer.contract_violations == 0


def test_a_bridged_episode_collapses_to_one_datum(inner, prime, tokenizer):
    """The reason the bridge exists: one datum instead of one per turn."""
    from tinker_cookbook.completers import TokensWithLogprobs
    from tinker_cookbook.rl.data_processing import trajectory_to_data
    from tinker_cookbook.rl.types import Trajectory, Transition

    renderer = make(inner, prime)
    transcript, _ = drive(renderer, tokenizer, turns=6)

    transitions = [
        Transition(
            ob=tinker.ModelInput.from_ints(prompt),
            ac=TokensWithLogprobs(tokens=action, maybe_logprobs=[-0.4] * len(action)),
            reward=0.0,
            episode_done=(i == len(transcript) - 1),
            metrics={},
            logs={},
        )
        for i, (prompt, action) in enumerate(transcript)
    ]
    traj = Trajectory(
        transitions=transitions,
        final_ob=tinker.ModelInput.from_ints(transcript[-1][0] + transcript[-1][1]),
    )
    data = trajectory_to_data(traj, 1.0)
    assert len(data) == 1, f"expected one appending datum, got {len(data)}"


def test_the_unbridged_baseline_still_splinters(inner, tokenizer):
    """Guards the premise: without the bridge the same episode makes N datums."""
    from tinker_cookbook.completers import TokensWithLogprobs
    from tinker_cookbook.rl.data_processing import trajectory_to_data
    from tinker_cookbook.rl.types import Trajectory, Transition

    renderer = make(inner, None)  # bridging disabled
    transcript, _ = drive(renderer, tokenizer, turns=6)

    transitions = [
        Transition(
            ob=tinker.ModelInput.from_ints(prompt),
            ac=TokensWithLogprobs(tokens=action, maybe_logprobs=[-0.4] * len(action)),
            reward=0.0,
            episode_done=(i == len(transcript) - 1),
            metrics={},
            logs={},
        )
        for i, (prompt, action) in enumerate(transcript)
    ]
    traj = Trajectory(
        transitions=transitions,
        final_ob=tinker.ModelInput.from_ints(transcript[-1][0] + transcript[-1][1]),
    )
    assert len(trajectory_to_data(traj, 1.0)) == 6


def test_a_user_observation_falls_back_instead_of_bridging(inner, prime, tokenizer):
    """Qwen3.5 is tool_cycle: a new user query must force a full re-render."""

    def user_observation(step: int) -> dict:
        return {"role": "user", "content": f"more context {step}"}

    renderer = make(inner, prime)
    drive(renderer, tokenizer, turns=4, observation=user_observation)
    assert renderer.bridged == 0
    assert renderer.fell_back == 3


def test_bridging_resumes_after_a_one_off_user_turn(inner, prime, tokenizer):
    """A parse-error retry injects a user message; it must cost one turn, not the episode."""

    def observation(step: int):
        # turn 1 gets a corrective user message, like ParseErrorPolicy injects
        return {"role": "user", "content": "malformed tool call"} if step == 1 else tool_message(step)

    renderer = make(inner, prime)
    transcript, _ = drive(renderer, tokenizer, turns=5, observation=observation)

    assert renderer.fell_back == 1, "only the user turn should re-render"
    assert renderer.bridged == 3

    # and the turns after the interruption are extensions again
    for i in (3, 4):
        prior = transcript[i - 1][0] + transcript[i - 1][1]
        assert transcript[i][0][: len(prior)] == prior


def test_a_non_extending_bridge_is_rejected(inner, tokenizer):
    """A library that broke its own contract must not corrupt the stream."""

    class LiarRenderer:
        def bridge_to_next_turn(self, prompt_ids, completion_ids, new_messages, *, tools=None):
            class Out:
                token_ids = [1, 2, 3]  # not an extension of anything

            return Out()

    renderer = make(inner, LiarRenderer())
    transcript, _ = drive(renderer, tokenizer, turns=3)

    assert renderer.contract_violations == 2
    assert renderer.bridged == 0
    for i in range(1, len(transcript)):
        assert transcript[i][0] != [1, 2, 3]


def test_a_raising_bridge_falls_back_quietly(inner, tokenizer):
    class ExplodingRenderer:
        def bridge_to_next_turn(self, *a, **kw):
            raise RuntimeError("boom")

    renderer = make(inner, ExplodingRenderer())
    transcript, _ = drive(renderer, tokenizer, turns=3)
    assert renderer.fell_back == 2
    assert len(transcript) == 3


# --------------------------------------------------------------------------
# the context nudge
# --------------------------------------------------------------------------


def test_the_nudge_lands_inside_the_tool_message(inner, prime, tokenizer):
    """Never as a user turn: that would switch bridging off under tool_cycle."""
    renderer = make(inner, prime, max_context_tokens=400, budget_warning_ratio=0.95)
    _, messages = drive(renderer, tokenizer, turns=3)

    assert renderer.nudges > 0
    nudged = [m for m in messages if m.get("role") == "tool" and "SYSTEM NOTICE" in str(m["content"])]
    assert nudged, "the warning never reached a tool message"
    assert not any(
        m.get("role") == "user" and "SYSTEM NOTICE" in str(m.get("content", ""))
        for m in messages
    )


def test_the_nudge_keeps_bridging_alive(inner, prime, tokenizer):
    renderer = make(inner, prime, max_context_tokens=400, budget_warning_ratio=0.95)
    transcript, _ = drive(renderer, tokenizer, turns=4)
    assert renderer.nudges > 0
    assert renderer.fell_back == 0
    for i in range(1, len(transcript)):
        prior = transcript[i - 1][0] + transcript[i - 1][1]
        assert transcript[i][0][: len(prior)] == prior


def test_a_large_observation_counts_toward_the_budget(inner, prime, tokenizer):
    """The reference subtracts the incoming observation; without it the nudge
    misses exactly the big tool outputs it exists for."""
    big = {"role": "tool", "tool_call_id": "c0", "name": "bash", "content": "x " * 3000}
    obs_tokens = len(tokenizer.encode(big["content"], add_special_tokens=False))  # ~3000
    # Size the window so the observation alone crosses the 20% line while the
    # prior (~70 tokens here, no tool schema in this conversation) never would.
    renderer = make(inner, prime, max_context_tokens=obs_tokens + 500, budget_warning_ratio=0.2)
    messages = [SYSTEM, USER]
    renderer.build_generation_prompt(messages)
    action = completion_tokens(tokenizer, 0)
    assistant, _ = renderer.parse_response(action)
    renderer.build_generation_prompt(messages + [assistant, big])
    assert renderer.nudges == 1, "a ~3k-token observation must pull the nudge forward"


def test_the_nudge_fires_once_per_episode(inner, prime, tokenizer):
    """Diverges from the reference on purpose: repeated notices burn the budget."""
    renderer = make(inner, prime, max_context_tokens=400, budget_warning_ratio=0.95)
    _, messages = drive(renderer, tokenizer, turns=4)
    assert renderer.nudges == 1
    assert sum("SYSTEM NOTICE" in str(m.get("content", "")) for m in messages) == 1


def test_a_disable_thinking_renderer_name_is_honoured(tokenizer):
    """Prime must not re-open <think> on bridged turns when the tenant closed it."""
    from rlcli.tito_bridge import prime_renderer_for
    r = prime_renderer_for("Qwen/Qwen3.5-9B", tokenizer, "qwen3_5_disable_thinking")
    assert r is not None
    p = list(r.render_ids([{"role": "user", "content": "hi"}], add_generation_prompt=True))
    p = p if isinstance(p[0], int) else list(p.token_ids)
    assert tokenizer.decode(p).endswith("<think>\n\n</think>\n\n"), "closed think block expected"


def test_an_unknown_renderer_name_refuses_to_bridge(tokenizer, capsys):
    from rlcli.tito_bridge import prime_renderer_for
    assert prime_renderer_for("Qwen/Qwen3.5-9B", tokenizer, "some_custom_renderer") is None
    assert "no Prime mapping" in capsys.readouterr().out


def test_parse_policy_reaches_the_inner_env_and_metrics_reach_the_step():
    """install_bridge must use the setter (not the attribute) and fold counters into the terminal step."""
    import asyncio
    from types import SimpleNamespace
    from rlcli.harbor_tito import install_bridge

    class InnerEnv:
        parse_error_policy = None
    class Env:
        def __init__(self):
            self.message_env = InnerEnv()
            self.parse_error_policy = None
            self.renderer = SimpleNamespace(build_generation_prompt=None, parse_response=None)
        def set_parse_error_policy(self, policy):
            self.parse_error_policy = policy
            self.message_env.parse_error_policy = policy
        async def step(self, action):
            return SimpleNamespace(episode_done=True, metrics={"reward": 1.0})

    env = Env()
    async def make_envs(): return [env]
    builder = SimpleNamespace(make_envs=make_envs, model_name="Qwen/Qwen3.5-9B",
                              renderer_name=None, max_trajectory_tokens=1000)
    install_bridge(builder, budget_warning_ratio=None, budget_warning_text=None, parse_error_retries=2)
    [out] = asyncio.run(builder.make_envs())
    assert out.message_env.parse_error_policy is not None, "policy never reached the inner env"
    assert out.message_env.parse_error_policy.max_consecutive == 2
    result = asyncio.run(out.step([1, 2, 3]))
    assert "tito_bridged" in result.metrics and result.metrics["reward"] == 1.0


def test_no_nudge_while_budget_remains(inner, prime, tokenizer):
    renderer = make(inner, prime, max_context_tokens=200_000, budget_warning_ratio=0.2)
    drive(renderer, tokenizer, turns=4)
    assert renderer.nudges == 0


def test_the_nudge_is_off_by_default(inner, prime, tokenizer):
    renderer = make(inner, prime)
    drive(renderer, tokenizer, turns=3)
    assert renderer.nudges == 0


def test_append_budget_warning_targets_the_last_tool_message():
    messages = [
        {"role": "tool", "tool_call_id": "a", "content": "first"},
        {"role": "tool", "tool_call_id": "b", "content": "second"},
    ]
    assert append_budget_warning(messages, "WRAP UP") is True
    assert messages[0]["content"] == "first"
    assert messages[1]["content"] == "second\n\nWRAP UP"


def test_append_budget_warning_handles_structured_content():
    messages = [{"role": "tool", "content": [{"type": "text", "text": "out"}]}]
    assert append_budget_warning(messages, "WRAP UP") is True
    assert messages[0]["content"][-1] == {"type": "text", "text": "WRAP UP"}


def test_append_budget_warning_reports_when_there_is_no_tool_message():
    assert append_budget_warning([{"role": "user", "content": "hi"}], "WRAP UP") is False


def test_the_default_warning_carries_a_percentage():
    assert "{pct}" in DEFAULT_BUDGET_WARNING


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------


def test_unknown_attributes_delegate_to_the_inner_renderer(inner, prime):
    renderer = make(inner, prime)
    assert renderer.get_stop_sequences() == inner.get_stop_sequences()


def test_metrics_report_the_split(inner, prime, tokenizer):
    renderer = make(inner, prime)
    drive(renderer, tokenizer, turns=3)
    metrics = renderer.metrics()
    assert metrics["tito_bridged"] == 2
    assert metrics["tito_contract_violations"] == 0
