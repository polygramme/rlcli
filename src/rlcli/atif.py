"""Record each rollout as an ATIF trajectory (Harbor's Agent Trajectory
Interchange Format, rfcs/0001-trajectory-format.md in harbor-framework/harbor).

One JSON file per episode: system/user prompt steps, one agent step per turn
with its tool calls, token counts and the tool results it observed, and the
episode's reward in ``final_metrics``. Anything that reads ATIF — Harbor's own
viewer, ``harbor-atif2otel`` — reads these directly.

The recorder wraps the cookbook's ``EnvFromMessageEnv`` at two seams:
``message_env.initial_observation``/``message_env.step`` (message-level, where
tool calls and observations are structured) and the outer ``env.step`` (where
the episode ends and the reward is known). Nothing here is on the training
path: a recorder failure is logged and the episode continues.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

log = logging.getLogger(__name__)

SCHEMA_VERSION = "ATIF-v1.7"
AGENT_NAME = "rlcli-harbor"

_SINK: "TrajectorySink | None" = None


def configure(sink: "TrajectorySink | None") -> None:
    """Set (or clear) the process-wide sink; the TITO env installer records
    every episode while one is set."""
    global _SINK
    _SINK = sink


def current_sink() -> "TrajectorySink | None":
    return _SINK


# ---- message helpers -------------------------------------------------------


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _text_and_reasoning(content: Any) -> tuple[str, str | None]:
    """Cookbook content is a str or a list of parts (text / thinking / image)."""
    if content is None:
        return "", None
    if isinstance(content, str):
        return content, None
    texts, thoughts = [], []
    for part in content:
        kind = _get(part, "type")
        if kind == "text":
            texts.append(str(_get(part, "text") or ""))
        elif kind == "thinking":
            thoughts.append(str(_get(part, "thinking") or _get(part, "text") or ""))
        elif kind == "image":
            texts.append("[image]")
    return "".join(texts), ("".join(thoughts) or None)


def _tool_calls(message: Any) -> list[dict]:
    out = []
    for i, tc in enumerate(_get(message, "tool_calls") or []):
        fn = _get(tc, "function")
        name = _get(fn, "name") if fn is not None else _get(tc, "name")
        raw = _get(fn, "arguments") if fn is not None else _get(tc, "arguments")
        if isinstance(raw, str):
            try:
                args = json.loads(raw)
                if not isinstance(args, dict):
                    args = {"value": args}
            except ValueError:
                args = {"raw": raw}
        elif isinstance(raw, dict):
            args = raw
        else:
            args = {}
        out.append({
            "tool_call_id": str(_get(tc, "id") or f"call_{i}"),
            "function_name": str(name or "tool"),
            "arguments": args,
        })
    for i, utc in enumerate(_get(message, "unparsed_tool_calls") or []):
        out.append({
            "tool_call_id": f"unparsed_{i}",
            "function_name": "unparsed",
            "arguments": {"raw": str(_get(utc, "raw") or _get(utc, "text") or utc)},
            "extra": {"parse_error": True},
        })
    return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- recorder --------------------------------------------------------------


class TrajectoryRecorder:
    """Per-episode ATIF builder attached to one cookbook env."""

    def __init__(self, env: Any, *, traj_idx: int, model_name: str | None = None, include_token_ids: bool = False):
        self.env = env
        self.traj_idx = traj_idx
        self.model_name = model_name
        self.include_token_ids = include_token_ids
        self.steps: list[dict] = []
        self.reward: float | None = None
        self.done = False
        self._n_seen = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._started = _now()
        self._t0 = time.time()
        self._open_agent: dict | None = None

    # -- steps

    def _add(self, step: dict) -> dict:
        step["step_id"] = len(self.steps) + 1
        step.setdefault("timestamp", _now())
        self.steps.append(step)
        return step

    def _ingest_prompt(self, messages: list[Any]) -> None:
        for m in messages:
            role = _get(m, "role")
            text, _ = _text_and_reasoning(_get(m, "content"))
            if role in ("system", "user"):
                self._add({"source": role, "message": text})
        self._n_seen = len(messages)

    def _agent_step(self, message: Any) -> dict:
        text, reasoning = _text_and_reasoning(_get(message, "content"))
        step: dict = {"source": "agent", "message": text, "llm_call_count": 1}
        if self.model_name:
            step["model_name"] = self.model_name
        if reasoning:
            step["reasoning_content"] = reasoning
        calls = _tool_calls(message)
        if calls:
            step["tool_calls"] = calls
        metrics = self._bridge_metrics()
        if metrics:
            step["metrics"] = metrics
        return self._add(step)

    def _bridge_metrics(self) -> dict | None:
        r = getattr(self.env, "renderer", None)
        prompt = getattr(r, "_prompt_ids", None)
        completion = getattr(r, "_completion_ids", None)
        if prompt is None and completion is None:
            return None
        m: dict = {}
        if prompt is not None:
            m["prompt_tokens"] = len(prompt)
            self._prompt_tokens += len(prompt)
            if self.include_token_ids:
                m["prompt_token_ids"] = list(prompt)
        if completion is not None:
            m["completion_tokens"] = len(completion)
            self._completion_tokens += len(completion)
            if self.include_token_ids:
                m["completion_token_ids"] = list(completion)
        return m

    def _ingest_after_step(self, history: list[Any]) -> None:
        new = history[self._n_seen:]
        self._n_seen = len(history)
        results = []
        for m in new:
            role = _get(m, "role")
            if role == "assistant":
                continue
            text, _ = _text_and_reasoning(_get(m, "content"))
            if role == "tool":
                results.append({"source_call_id": _get(m, "tool_call_id"), "content": text})
            elif role in ("user", "system"):
                self._flush_observation(results)
                results = []
                self._add({"source": role, "message": text})
        self._flush_observation(results)

    def _flush_observation(self, results: list[dict]) -> None:
        if results and self._open_agent is not None:
            self._open_agent.setdefault("observation", {"results": []})["results"].extend(
                {k: v for k, v in r.items() if v is not None} for r in results
            )

    # -- wrappers

    def install(self) -> "TrajectoryRecorder":
        env, rec = self.env, self
        menv = getattr(env, "message_env", None)
        if menv is None:
            raise TypeError("recorder needs an EnvFromMessageEnv (no message_env)")
        orig_initial, orig_step, orig_outer = menv.initial_observation, menv.step, env.step

        async def initial_observation(*a, **kw):
            msgs = await orig_initial(*a, **kw)
            try:
                rec._ingest_prompt(list(msgs))
            except Exception:  # noqa: BLE001
                log.exception("atif: prompt ingest failed")
            return msgs

        async def step(message, *a, **kw):
            try:
                rec._open_agent = rec._agent_step(message)
            except Exception:  # noqa: BLE001
                log.exception("atif: agent step failed")
            result = await orig_step(message, *a, **kw)
            try:
                rec._ingest_after_step(list(_get(result, "next_messages") or []))
                r = _get(result, "reward")
                if isinstance(r, (int, float)):
                    rec.reward = (rec.reward or 0.0) + float(r)
            except Exception:  # noqa: BLE001
                log.exception("atif: observation ingest failed")
            return result

        async def outer_step(action, *a, **kw):
            result = await orig_outer(action, *a, **kw)
            try:
                if _get(result, "episode_done") and not rec.done:
                    r = _get(result, "reward")
                    rec.finalize(
                        reward=float(r) if isinstance(r, (int, float)) else rec.reward,
                        metrics=dict(_get(result, "metrics") or {}),
                    )
            except Exception:  # noqa: BLE001
                log.exception("atif: finalize failed")
            return result

        menv.initial_observation, menv.step, env.step = initial_observation, step, outer_step
        return self

    # -- output

    def coords(self) -> dict:
        out: dict = {"traj_idx": self.traj_idx}
        try:
            from tinker_cookbook.capture.scope import current_scope

            scope = dict(current_scope())
        except Exception:  # noqa: BLE001
            scope = {}
        for k in ("run_id", "iteration", "group_idx", "task", "task_hash", "split", "purpose"):
            if scope.get(k) is not None:
                out[k] = scope[k]
        return out

    def to_dict(self, *, reward: float | None, metrics: dict | None) -> dict:
        coords = self.coords()
        turns = sum(1 for s in self.steps if s["source"] == "agent")
        final_extra = {"reward": reward, "turns": turns, "duration_s": round(time.time() - self._t0, 3)}
        if metrics:
            final_extra["env_metrics"] = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}
        sid = ":".join(str(coords.get(k, "-")) for k in ("run_id", "iteration", "group_idx", "traj_idx"))
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": sid,
            "trajectory_id": uuid.uuid4().hex,
            "agent": {"name": AGENT_NAME, "version": _rlcli_version(), "model_name": self.model_name},
            "steps": self.steps,
            "final_metrics": {
                "total_prompt_tokens": self._prompt_tokens,
                "total_completion_tokens": self._completion_tokens,
                "total_steps": len(self.steps),
                "extra": final_extra,
            },
            "extra": {**coords, "started_at": self._started, "finished_at": _now()},
        }

    def finalize(self, *, reward: float | None, metrics: dict | None = None) -> dict | None:
        self.done = True
        traj = self.to_dict(reward=reward, metrics=metrics)
        sink = _SINK
        if sink is not None:
            try:
                sink.write(traj)
            except Exception:  # noqa: BLE001
                log.exception("atif: sink write failed")
        return traj


def _rlcli_version() -> str:
    try:
        from importlib.metadata import version

        return version("polygramme-rlcli")
    except Exception:  # noqa: BLE001
        return "0"


def install_recorder(env: Any, *, traj_idx: int, model_name: str | None = None) -> TrajectoryRecorder | None:
    """Attach a recorder when a sink is configured; None otherwise."""
    if _SINK is None:
        return None
    try:
        return TrajectoryRecorder(env, traj_idx=traj_idx, model_name=model_name).install()
    except Exception:  # noqa: BLE001
        log.exception("atif: recorder install failed")
        return None


# ---- sinks -----------------------------------------------------------------


class TrajectorySink:
    def write(self, trajectory: dict) -> None:  # pragma: no cover - protocol
        raise NotImplementedError


def _slug(s: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))[:48] or "x"


class FileSink(TrajectorySink):
    """``root/it00003/train/g001_t02.json``; ``untagged/`` when no iteration.

    ``on_written(record)`` receives {path (relative to root), coords, reward,
    turns, prompt_tokens, completion_tokens, trajectory_id} after each write.
    """

    def __init__(self, root: str, on_written: Callable[[dict], None] | None = None):
        self.root = root
        self.on_written = on_written
        self.written = 0
        self._lock = threading.Lock()
        os.makedirs(root, exist_ok=True)

    def path_for(self, trajectory: dict) -> str:
        c = trajectory.get("extra") or {}
        it = c.get("iteration")
        raw = str(c.get("split") or "train")
        if raw.startswith("eval"):
            label = raw.split("/", 1)[1] if "/" in raw else ""
            split = "eval_" + _slug(label) if label else "eval"
        else:
            split = _slug(raw)
        sub = f"it{int(it):05d}" if isinstance(it, int) and not isinstance(it, bool) else "untagged"
        g = c.get("group_idx")
        t = c.get("traj_idx", 0)
        gpart = f"g{int(g):03d}" if isinstance(g, int) else "g" + trajectory.get("trajectory_id", "x")[:6]
        return os.path.join(sub, split, f"{gpart}_t{int(t):02d}.json")

    def write(self, trajectory: dict) -> str:
        rel = self.path_for(trajectory)
        path = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with self._lock:
            with open(tmp, "w") as f:
                json.dump(trajectory, f, separators=(",", ":"))
            os.replace(tmp, path)
            self.written += 1
        if self.on_written:
            fm = trajectory.get("final_metrics") or {}
            ex = fm.get("extra") or {}
            try:
                self.on_written({
                    "path": rel,
                    "coords": dict(trajectory.get("extra") or {}),
                    "trajectory_id": trajectory.get("trajectory_id"),
                    "reward": ex.get("reward"),
                    "turns": ex.get("turns"),
                    "duration_s": ex.get("duration_s"),
                    "prompt_tokens": fm.get("total_prompt_tokens"),
                    "completion_tokens": fm.get("total_completion_tokens"),
                })
            except Exception:  # noqa: BLE001
                log.exception("atif: on_written failed")
        return rel
