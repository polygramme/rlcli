"""Normalize chat-dump JSONL into training-ready messages JSONL.

Output lines are exactly what the cookbook's FromConversationFileBuilder
consumes: {"messages": [{"role": ..., "content": ...}, ...]} — the input to
`rlcli train sl`. Imported traces feed SFT/distillation; RL needs an
environment and reward (Harbor), not raw traces.

Fail-loud: unknown formats and malformed lines raise ImportFormatError with
the line number rather than silently degrading the dataset.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

FORMATS = ("messages", "openai", "anthropic", "langsmith")

_ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "ai": "assistant",
    "model": "assistant",
    "assistant": "assistant",
    "system": "system",
}

# Roles/blocks that are tool plumbing, not conversation: dropped.
_DROP_ROLES = {"tool", "function", "tool_result"}
_TEXT_BLOCK_TYPES = {"text"}


class ImportFormatError(ValueError):
    pass


def normalize_role(role: Any) -> str | None:
    """Map provider role names onto user/assistant/system; None = drop."""
    if not isinstance(role, str):
        return None
    role = role.lower()
    if role in _DROP_ROLES:
        return None
    return _ROLE_ALIASES.get(role)


def flatten_content(content: Any) -> str:
    """Flatten string-or-blocks content to plain text.

    Text blocks are joined; tool_use / tool_result / image / thinking blocks
    are dropped. Returns "" when nothing textual remains.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in _TEXT_BLOCK_TYPES:
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(p for p in parts if p.strip())
    return ""


def _normalize_message_list(raw_messages: list, lineno: int) -> list[dict]:
    messages = []
    for m in raw_messages:
        if not isinstance(m, dict):
            raise ImportFormatError(f"line {lineno}: message is not an object: {m!r}")
        role = normalize_role(m.get("role"))
        if role is None:
            continue  # tool/function plumbing
        content = flatten_content(m.get("content"))
        if not content.strip():
            continue  # tool_calls-only assistant turns, empty blocks
        messages.append({"role": role, "content": content})
    return messages


def _parse_messages_line(record: dict, lineno: int) -> list[dict]:
    if "messages" not in record:
        raise ImportFormatError(
            f"line {lineno}: expected a 'messages' key, got keys {sorted(record)}"
        )
    if not isinstance(record["messages"], list):
        raise ImportFormatError(f"line {lineno}: 'messages' is not a list")
    return _normalize_message_list(record["messages"], lineno)


def _parse_anthropic_line(record: dict, lineno: int) -> list[dict]:
    """Anthropic Messages API request/response dumps: top-level `system` plus
    `messages` with content blocks."""
    messages = []
    system = record.get("system")
    if isinstance(system, (str, list)):
        text = flatten_content(system)
        if text.strip():
            messages.append({"role": "system", "content": text})
    messages.extend(_parse_messages_line(record, lineno))
    return messages


def _unwrap_langchain_message(m: dict) -> dict:
    """LangSmith exports may carry LangChain-serialized messages:
    {"id": [..., "HumanMessage"], "kwargs": {"content": ..., "type": "human"}}.
    Unwrap to a flat {"role", "content"} dict; plain dicts pass through."""
    if "kwargs" in m and isinstance(m["kwargs"], dict):
        kw = m["kwargs"]
        role = kw.get("type")
        if role is None and isinstance(m.get("id"), list) and m["id"]:
            cls = m["id"][-1]
            role = {"HumanMessage": "user", "AIMessage": "assistant",
                    "SystemMessage": "system"}.get(cls)
        return {"role": role, "content": kw.get("content")}
    if "role" not in m and "type" in m:
        return {"role": m["type"], "content": m.get("content")}
    return m


def _langsmith_output_message(outputs: Any) -> dict | None:
    """Find the assistant reply in a LangSmith run's `outputs`, across the
    shapes runs actually take: chat runs ({"messages": [...]} / {"output": ...}),
    LLM runs ({"generations": [[{"message"|"text": ...}]]}), plain strings."""
    if isinstance(outputs, str):
        return {"role": "assistant", "content": outputs} if outputs.strip() else None
    if not isinstance(outputs, dict):
        return None
    if isinstance(outputs.get("messages"), list) and outputs["messages"]:
        return _unwrap_langchain_message(outputs["messages"][-1])
    out = outputs.get("output")
    if isinstance(out, str):
        return {"role": "assistant", "content": out}
    if isinstance(out, dict):
        return _unwrap_langchain_message(out)
    gens = outputs.get("generations")
    if isinstance(gens, list) and gens:
        first = gens[0][0] if isinstance(gens[0], list) and gens[0] else gens[0]
        if isinstance(first, dict):
            if isinstance(first.get("message"), dict):
                return _unwrap_langchain_message(first["message"])
            if isinstance(first.get("text"), str):
                return {"role": "assistant", "content": first["text"]}
    content = outputs.get("content")
    if isinstance(content, (str, list)):
        text = flatten_content(content)
        if text.strip():
            return {"role": "assistant", "content": text}
    return None


def langsmith_reward(record: dict, feedback_key: str | None) -> float | None:
    """Extract a scalar reward from LangSmith feedback.

    Sources, in order: `feedback_stats` ({key: {"n": ..., "avg": ...}}) and a
    `feedback` list ([{"key": ..., "score": ...}]). With no --feedback-key,
    a single-key record is unambiguous and used; multi-key records raise so
    the user picks one rather than training on the wrong signal.
    """
    stats = record.get("feedback_stats")
    if isinstance(stats, dict) and stats:
        if feedback_key is not None:
            entry = stats.get(feedback_key)
            avg = entry.get("avg") if isinstance(entry, dict) else None
            return float(avg) if isinstance(avg, (int, float)) else None
        if len(stats) == 1:
            (entry,) = stats.values()
            avg = entry.get("avg") if isinstance(entry, dict) else None
            return float(avg) if isinstance(avg, (int, float)) else None
        raise ImportFormatError(
            f"run has multiple feedback keys {sorted(stats)}; pass --feedback-key"
        )
    feedback = record.get("feedback")
    if isinstance(feedback, list) and feedback:
        scores = [
            float(f["score"])
            for f in feedback
            if isinstance(f, dict) and isinstance(f.get("score"), (int, float))
            and (feedback_key is None or f.get("key") == feedback_key)
        ]
        if scores:
            keys = {f.get("key") for f in feedback if isinstance(f, dict)}
            if feedback_key is None and len(keys) > 1:
                raise ImportFormatError(
                    f"run has multiple feedback keys {sorted(k for k in keys if k)}; "
                    "pass --feedback-key"
                )
            return sum(scores) / len(scores)
    return None


def _parse_langsmith_line(record: dict, lineno: int) -> list[dict]:
    """LangSmith run exports: prompt history in inputs.messages, the reply in
    outputs (several shapes), feedback handled separately in langsmith_reward."""
    inputs = record.get("inputs")
    if not isinstance(inputs, dict) or "messages" not in inputs:
        raise ImportFormatError(
            f"line {lineno}: expected a LangSmith run with inputs.messages, "
            f"got keys {sorted(record)}"
        )
    raw = [_unwrap_langchain_message(m) for m in inputs["messages"]
           if isinstance(m, dict)]
    messages = _normalize_message_list(raw, lineno)
    reply = _langsmith_output_message(record.get("outputs"))
    if reply is not None:
        normalized = _normalize_message_list([reply], lineno)
        # Chat runs sometimes echo the full history in outputs.messages; only
        # append when it's genuinely the reply, not a duplicate of the tail.
        if normalized and (not messages or messages[-1] != normalized[0]):
            messages.extend(normalized)
    return messages


_PARSERS = {
    "messages": _parse_messages_line,
    "openai": _parse_messages_line,  # tool_calls turns drop via empty content
    "anthropic": _parse_anthropic_line,
    "langsmith": _parse_langsmith_line,
}


def iter_conversations(
    lines: Iterator[str],
    fmt: str,
    min_messages: int = 2,
    feedback_key: str | None = None,
) -> Iterator[tuple[dict, int]]:
    """Yield (record, lineno) per usable input line, where record is the
    train-ready output object: {"messages": [...]} plus, for langsmith runs
    with feedback, a "reward" float.

    A conversation is usable when it has at least `min_messages` messages
    including at least one assistant message (something to train on).
    """
    if fmt not in _PARSERS:
        raise ImportFormatError(f"Unknown format {fmt!r}. Known: {', '.join(FORMATS)}")
    parser = _PARSERS[fmt]
    for lineno, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            raise ImportFormatError(f"line {lineno}: invalid JSON: {e}") from e
        if not isinstance(record, dict):
            raise ImportFormatError(f"line {lineno}: expected a JSON object")
        messages = parser(record, lineno)
        if len(messages) < min_messages:
            continue
        if not any(m["role"] == "assistant" for m in messages):
            continue
        out: dict = {"messages": messages}
        if fmt == "langsmith":
            try:
                reward = langsmith_reward(record, feedback_key)
            except ImportFormatError as e:
                raise ImportFormatError(f"line {lineno}: {e}") from e
            if reward is not None:
                out["reward"] = reward
        elif isinstance(record.get("reward"), (int, float)):
            # Already-imported datasets round-trip: keep the reward a prior
            # `rlcli import -f langsmith` attached.
            out["reward"] = float(record["reward"])
        yield out, lineno
