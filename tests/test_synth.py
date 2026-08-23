import json
import tomllib
from pathlib import Path

import pytest

from rlcli.synth import (
    SynthError,
    build_prompt,
    parse_task_json,
    scaffold_task,
    synthesize,
)

GOOD = {
    "name": "Refund Order!",
    "instruction": "Create /app/refunds.csv containing order 123 marked refunded.",
    "test_script": "grep -q '123,refunded' /app/refunds.csv",
}


def test_build_prompt_carries_trace_and_contract():
    msgs = [{"role": "user", "content": "refund order 123"},
            {"role": "assistant", "content": "done"}]
    prompt = build_prompt(msgs)
    assert prompt[0]["role"] == "system"
    assert "FAIL on a fresh container" in prompt[0]["content"]
    assert "[user] refund order 123" in prompt[1]["content"]


def test_parse_task_json_slugifies_and_strips_fences():
    task = parse_task_json("```json\n" + json.dumps(GOOD) + "\n```")
    assert task["name"] == "refund-order"


def test_parse_task_json_fails_loud():
    with pytest.raises(SynthError, match="valid JSON"):
        parse_task_json("not json")
    with pytest.raises(SynthError, match="missing keys"):
        parse_task_json(json.dumps({"name": "x", "instruction": "y"}))


def test_scaffold_writes_harbor_layout_with_lineage(tmp_path):
    task = parse_task_json(json.dumps(GOOD))
    task_dir = scaffold_task(tmp_path, task,
                             {"source_file": "runs.jsonl", "source_line": 7,
                              "reference_reward": 0.9})
    assert (task_dir / "instruction.md").read_text().startswith("Create /app")
    assert "FROM python" in (task_dir / "environment" / "Dockerfile").read_text()
    test_sh = task_dir / "tests" / "test.sh"
    assert test_sh.read_text().startswith("#!/bin/sh")
    assert test_sh.stat().st_mode & 0o111
    toml = tomllib.loads((task_dir / "task.toml").read_text())
    assert toml["synth"]["source_line"] == 7
    assert toml["synth"]["reference_reward"] == 0.9
    # name collision → suffixed dir, not overwrite
    again = scaffold_task(tmp_path, dict(task), {"source_file": "runs.jsonl",
                                                 "source_line": 8})
    assert again.name == "refund-order-2"


def test_synthesize_reports_generation_failure_and_success(tmp_path):
    calls = []

    def generate(prompt):
        calls.append(prompt)
        if len(calls) <= 2:  # first conversation: bad JSON on both attempts
            return "garbage"
        return json.dumps(GOOD)

    convs = [({"messages": [{"role": "user", "content": "a"},
                            {"role": "assistant", "content": "b"}]}, 1),
             ({"messages": [{"role": "user", "content": "c"},
                            {"role": "assistant", "content": "d"}],
               "reward": 1.0}, 2)]
    reports = synthesize(convs, tmp_path / "tasks", generate, "runs.jsonl",
                         validate=False)
    assert reports[0]["task_dir"] is None
    assert "generation failed" in reports[0]["detail"]
    assert reports[1]["task_dir"] and Path(reports[1]["task_dir"]).is_dir()
    toml = tomllib.loads(
        (Path(reports[1]["task_dir"]) / "task.toml").read_text())
    assert toml["synth"]["reference_reward"] == 1.0
    assert len(calls) == 3  # 2 attempts for conv 1, 1 for conv 2


@pytest.mark.docker
def test_validate_task_rejects_trivially_passing_test(tmp_path):
    import asyncio

    pytest.importorskip("tinker_cookbook")  # validate_task uses sandbox_docker
    from rlcli.synth import validate_task

    trivial = dict(GOOD, test_script="true")
    task_dir = scaffold_task(tmp_path, parse_task_json(json.dumps(trivial)),
                             {"source_file": "x", "source_line": 1})
    ok, why = asyncio.run(validate_task(task_dir, timeout=120))
    assert not ok and "untouched container" in why

    honest = dict(GOOD, name="honest")
    task_dir2 = scaffold_task(tmp_path, parse_task_json(json.dumps(honest)),
                              {"source_file": "x", "source_line": 2})
    ok2, why2 = asyncio.run(validate_task(task_dir2, timeout=120))
    assert ok2, why2
