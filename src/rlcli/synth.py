"""Synthesize Harbor-format sandbox tasks from agent traces.

The continual-learning loop's missing middle: traces show what an agent was
asked to do; a task turns that into something trainable — a Docker
environment, an instruction, and a test script whose verdict is the RL
reward. `rlcli synth` drafts each task with an LLM (any OpenAI-compatible
endpoint, including a local vLLM), scaffolds the exact directory layout
tinker-cookbook's harbor_rl loader expects, and validates the result by
building the image and proving the test does NOT pass on an untouched
container (a test an idle agent passes is not a reward signal).

Lineage: each task.toml carries a [synth] table recording the source file,
line, and any imported reward — the open equivalent of a reference-trajectory
id, so every environment can be traced back to the conversation it came from.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Callable

import httpx

REQUIRED_KEYS = ("name", "instruction", "test_script")

DEFAULT_DOCKERFILE = """\
FROM python:3.11-slim
WORKDIR /app
"""

SYNTH_SYSTEM = """\
You design verifiable sandbox tasks for reinforcement learning from examples \
of real agent conversations. Given one conversation, produce ONE task that \
tests the capability the agent needed — not a transcript quiz.

Respond with a single JSON object, no markdown fence, with keys:
- "name": short kebab-case slug for the task
- "instruction": what the agent must accomplish inside the container, \
self-contained, no references to the original conversation
- "test_script": a POSIX sh script that exits 0 if and only if the task was \
completed correctly, run inside the container after the agent finishes
- "dockerfile": (optional) Dockerfile for the environment; omit for the \
default python:3.11-slim. If the task needs files pre-created, create them \
with RUN commands here.

Hard requirements:
- The test must be deterministic and self-contained: no network access, no \
external services, no randomness.
- The test must FAIL on a fresh container where the agent has done nothing.
- The instruction must not reveal the test's implementation.
"""


class SynthError(ValueError):
    pass


def build_prompt(messages: list[dict]) -> list[dict]:
    """Chat messages for the generator: system contract + the trace."""
    trace = "\n".join(f"[{m['role']}] {m['content']}" for m in messages)
    return [
        {"role": "system", "content": SYNTH_SYSTEM},
        {"role": "user", "content": f"Conversation:\n\n{trace}\n\nDesign the task."},
    ]


def parse_task_json(text: str) -> dict:
    """Parse and validate the generator's response. Fail-loud."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n|\n```$", "", text)
    try:
        task = json.loads(text)
    except json.JSONDecodeError as e:
        raise SynthError(f"generator did not return valid JSON: {e}") from e
    if not isinstance(task, dict):
        raise SynthError("generator returned JSON that is not an object")
    missing = [k for k in REQUIRED_KEYS if not isinstance(task.get(k), str) or not task[k].strip()]
    if missing:
        raise SynthError(f"generator response missing keys: {missing}")
    name = re.sub(r"[^a-z0-9-]+", "-", task["name"].lower()).strip("-")
    if not name:
        raise SynthError(f"unusable task name {task['name']!r}")
    task["name"] = name
    return task


def scaffold_task(out_dir: Path, task: dict, source: dict[str, Any]) -> Path:
    """Write the harbor_rl task layout:
    <name>/{instruction.md, task.toml, environment/Dockerfile, tests/test.sh}.
    """
    name, n = task["name"], 2
    task_dir = out_dir / name
    while task_dir.exists():
        task_dir = out_dir / f"{name}-{n}"
        n += 1
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "instruction.md").write_text(task["instruction"].rstrip() + "\n")
    dockerfile = task.get("dockerfile") or DEFAULT_DOCKERFILE
    (task_dir / "environment" / "Dockerfile").write_text(dockerfile.rstrip() + "\n")
    test = task["test_script"]
    if not test.startswith("#!"):
        test = "#!/bin/sh\n" + test
    test_path = task_dir / "tests" / "test.sh"
    test_path.write_text(test.rstrip() + "\n")
    test_path.chmod(0o755)
    lineage = "\n".join(
        f'{k} = {json.dumps(v)}' for k, v in source.items() if v is not None
    )
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n\n'
        "[metadata]\n"
        'author_name = "rlcli synth"\n'
        'category = "synthesized"\n\n'
        "[verifier]\n"
        "timeout_sec = 300.0\n\n"
        "[agent]\n"
        "timeout_sec = 600.0\n\n"
        f"[synth]\n{lineage}\n"
    )
    return task_dir


async def validate_task(task_dir: Path, timeout: int = 300) -> tuple[bool, str]:
    """Build the environment image and run tests/test.sh on an untouched
    container. Valid means: image builds AND the test exits nonzero (an idle
    agent must not be rewarded)."""
    # Deferred: sandbox_docker imports tinker_cookbook, which only the
    # [train]-extra environment has; synth's other functions stay importable
    # everywhere.
    from rlcli.sandbox_docker import _ensure_image, _run

    env_dir = task_dir / "environment"
    try:
        tag = await _ensure_image(env_dir)
    except (RuntimeError, FileNotFoundError) as e:
        return False, f"image build failed: {e}"
    tests_dir = (task_dir / "tests").resolve()
    code, out, err = await _run(
        "docker", "run", "--rm", "--network", "none",
        "-v", f"{tests_dir}:/rlcli-tests:ro",
        "--entrypoint", "sh", tag, "/rlcli-tests/test.sh",
        timeout=timeout,
    )
    if code == 0:
        return False, "test passes on an untouched container (no reward signal)"
    if code == 124:
        return False, f"test timed out after {timeout}s"
    # 125/126/127 from docker itself (bad mount, missing sh, ...) is a
    # validation ERROR, not a failing test — don't mistake it for a reward
    # signal.
    if code in (125, 126, 127):
        return False, f"docker could not run the test (exit {code}): {err.decode()[-500:]}"
    return True, f"test fails on untouched container (exit {code}) — valid"


def openai_compat_generate(
    api_base: str,
    api_key: str | None,
    model: str,
    prompt_messages: list[dict],
    timeout: float = 180.0,
) -> str:
    """One chat completion against any OpenAI-compatible endpoint."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = httpx.post(
        api_base.rstrip("/") + "/chat/completions",
        headers=headers,
        json={"model": model, "messages": prompt_messages, "temperature": 0.2},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise SynthError(f"generator endpoint returned {resp.status_code}: {resp.text[:500]}")
    try:
        return resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise SynthError(f"unexpected completion shape: {resp.text[:500]}") from e


def synthesize(
    conversations: list[tuple[dict, int]],
    out_dir: Path,
    generate: Callable[[list[dict]], str],
    source_file: str,
    validate: bool = True,
    validate_timeout: int = 300,
    retries: int = 1,
) -> list[dict]:
    """Drive generation → scaffold → validation for each conversation.

    Returns one report dict per conversation:
    {"lineno", "task_dir" | None, "valid" | None, "detail"}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for record, lineno in conversations:
        task = None
        detail = ""
        for attempt in range(retries + 1):
            try:
                task = parse_task_json(generate(build_prompt(record["messages"])))
                break
            except SynthError as e:
                detail = str(e)
                task = None
        if task is None:
            reports.append({"lineno": lineno, "task_dir": None, "valid": None,
                            "detail": f"generation failed: {detail}"})
            continue
        task_dir = scaffold_task(
            out_dir, task,
            {"source_file": source_file, "source_line": lineno,
             "reference_reward": record.get("reward")},
        )
        if not validate:
            reports.append({"lineno": lineno, "task_dir": str(task_dir),
                            "valid": None, "detail": "not validated"})
            continue
        ok, why = asyncio.run(validate_task(task_dir, timeout=validate_timeout))
        reports.append({"lineno": lineno, "task_dir": str(task_dir),
                        "valid": ok, "detail": why})
    return reports
