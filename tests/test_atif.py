"""rlcli.atif: ATIF trajectories from cookbook message envs."""
import asyncio
import json
import os

import pytest

from rlcli import atif


class TC:
    class FunctionBody:
        def __init__(self, name, arguments):
            self.name, self.arguments = name, arguments

    def __init__(self, id, name, arguments):
        self.id, self.function = id, TC.FunctionBody(name, arguments)


class StepOut:
    def __init__(self, next_messages, reward=0.0, episode_done=False, metrics=None):
        self.next_messages, self.reward, self.episode_done, self.metrics = next_messages, reward, episode_done, metrics or {}


class FakeMessageEnv:
    def __init__(self):
        self.history = [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "Fix the bug."}]

    async def initial_observation(self):
        return list(self.history)

    async def step(self, message):
        self.history.append(message)
        for tc in message.get("tool_calls") or []:
            self.history.append({"role": "tool", "tool_call_id": tc.id, "content": f"ran {tc.function.name}"})
        if len(self.history) >= 8:
            self.history.append({"role": "user", "content": "Please wrap up."})
        return StepOut(self.history)


class FakeRenderer:
    _prompt_ids = [1, 2, 3]
    _completion_ids = [4, 5]


class FakeEnv:
    def __init__(self):
        self.message_env = FakeMessageEnv()
        self.renderer = FakeRenderer()
        self.turn = 0

    async def step(self, action):
        # mimics EnvFromMessageEnv: parse -> message_env.step -> StepResult
        self.turn += 1
        msg = {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": f"turn {self.turn}"}],
               "tool_calls": [TC(f"c{self.turn}", "bash", json.dumps({"cmd": "ls"}))]}
        await self.message_env.step(msg)
        done = self.turn >= 3
        return StepOut([], reward=1.0 if done else 0.0, episode_done=done, metrics={"test_passed": 1.0} if done else {})


@pytest.fixture(autouse=True)
def _clear_sink():
    atif.configure(None)
    yield
    atif.configure(None)


def test_recorder_builds_atif_steps_and_finalizes(tmp_path):
    written = []
    sink = atif.FileSink(str(tmp_path), on_written=written.append)
    atif.configure(sink)
    env = FakeEnv()
    rec = atif.install_recorder(env, traj_idx=2, model_name="Qwen/Qwen3.5-9B")
    assert rec is not None

    async def run():
        await env.message_env.initial_observation()
        for _ in range(3):
            await env.step([9, 9])

    asyncio.run(run())
    assert rec.done and sink.written == 1
    rel = written[0]["path"]
    assert rel.startswith("untagged/train/") and rel.endswith("_t02.json")
    traj = json.load(open(tmp_path / rel))
    assert traj["schema_version"] == atif.SCHEMA_VERSION
    assert traj["agent"]["name"] == atif.AGENT_NAME and traj["agent"]["model_name"] == "Qwen/Qwen3.5-9B"
    srcs = [s["source"] for s in traj["steps"]]
    assert srcs[:2] == ["system", "user"] and srcs.count("agent") == 3
    first_agent = next(s for s in traj["steps"] if s["source"] == "agent")
    assert first_agent["message"] == "turn 1" and first_agent["reasoning_content"] == "hmm"
    assert first_agent["tool_calls"] == [{"tool_call_id": "c1", "function_name": "bash", "arguments": {"cmd": "ls"}}]
    assert first_agent["observation"]["results"] == [{"source_call_id": "c1", "content": "ran bash"}]
    assert first_agent["metrics"] == {"prompt_tokens": 3, "completion_tokens": 2}
    assert first_agent["llm_call_count"] == 1
    assert [s["step_id"] for s in traj["steps"]] == list(range(1, len(traj["steps"]) + 1))
    assert "Please wrap up." in [s["message"] for s in traj["steps"] if s["source"] == "user"]
    fm = traj["final_metrics"]
    assert fm["total_prompt_tokens"] == 9 and fm["total_completion_tokens"] == 6 and fm["total_steps"] == len(traj["steps"])
    assert fm["extra"]["reward"] == 1.0 and fm["extra"]["turns"] == 3 and fm["extra"]["env_metrics"] == {"test_passed": 1.0}
    assert written[0]["reward"] == 1.0 and written[0]["turns"] == 3 and written[0]["coords"]["traj_idx"] == 2


def test_recorder_uses_capture_scope_for_coordinates(tmp_path):
    capture = pytest.importorskip("tinker_cookbook.capture").capture
    written = []
    atif.configure(atif.FileSink(str(tmp_path), on_written=written.append))
    env = FakeEnv()
    atif.install_recorder(env, traj_idx=0)

    async def run():
        await env.message_env.initial_observation()
        with capture(run_id="run_x", iteration=4, group_idx=1, task="hello", task_hash="abc", split="eval/test"):
            for _ in range(3):
                await env.step([1])

    asyncio.run(run())
    assert written[0]["path"] == os.path.join("it00004", "eval_test", "g001_t00.json")
    c = written[0]["coords"]
    assert c["run_id"] == "run_x" and c["iteration"] == 4 and c["task_hash"] == "abc"
    traj = json.load(open(tmp_path / written[0]["path"]))
    assert traj["session_id"] == "run_x:4:1:0"


def test_no_sink_means_no_recorder_and_env_untouched():
    env = FakeEnv()
    assert atif.install_recorder(env, traj_idx=0) is None
    assert "step" not in vars(env) and "step" not in vars(env.message_env)  # class methods untouched


def test_sink_failure_never_breaks_the_episode(tmp_path):
    class Boom(atif.TrajectorySink):
        def write(self, trajectory):
            raise RuntimeError("disk full")

    atif.configure(Boom())
    env = FakeEnv()
    rec = atif.install_recorder(env, traj_idx=0)

    async def run():
        await env.message_env.initial_observation()
        for _ in range(3):
            await env.step([1])

    asyncio.run(run())
    assert rec.done  # finalized despite the sink raising


def test_tool_call_argument_parsing_and_unparsed():
    msg = {"role": "assistant", "content": "x", "tool_calls": [TC("a", "bash", "not json"), TC(None, "bash", {"k": 1})],
           "unparsed_tool_calls": [{"raw": "<tool>garbage"}]}
    calls = atif._tool_calls(msg)
    assert calls[0]["arguments"] == {"raw": "not json"} and calls[1]["arguments"] == {"k": 1} and calls[1]["tool_call_id"] == "call_1"
    assert calls[2]["function_name"] == "unparsed" and calls[2]["extra"] == {"parse_error": True}


def test_recorder_counts_tool_calls_and_parse_errors_for_metrics():
    env = FakeEnv()
    atif.configure(atif.FileSink(str(__import__("tempfile").mkdtemp())))
    rec = atif.install_recorder(env, traj_idx=0)
    msgs = [
        {"role": "assistant", "content": "a", "tool_calls": [TC("1", "bash", "{}"), TC("2", "bash", "{}"), TC("3", "read_file", "{}")]},
        {"role": "assistant", "content": "b", "unparsed_tool_calls": [{"raw": "<bad>"}]},
    ]

    async def run():
        await env.message_env.initial_observation()
        for m in msgs:
            await env.message_env.step(m)

    asyncio.run(run())
    assert rec.metrics() == {"tools/calls": 3.0, "tools/bash": 2.0, "tools/read_file": 1.0, "errors/parse_calls": 1.0, "errors/parse_episode": 1.0}
