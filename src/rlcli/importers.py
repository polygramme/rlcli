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

FORMATS = ("messages", "openai", "anthropic")

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


_PARSERS = {
    "messages": _parse_messages_line,
    "openai": _parse_messages_line,  # tool_calls turns drop via empty content
    "anthropic": _parse_anthropic_line,
}


def iter_conversations(
    lines: Iterator[str],
    fmt: str,
    min_messages: int = 2,
) -> Iterator[tuple[list[dict], int]]:
    """Yield (messages, lineno) per usable input line.

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
        yield messages, lineno
