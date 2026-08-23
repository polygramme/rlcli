"""Normalize chat-dump JSONL into training-ready messages JSONL.

Output lines are what rlcli's tool-aware dataset builder (and the cookbook's
FromConversationFileBuilder, for text-only data) consume:
{"messages": [{"role": ..., "content": ...}, ...]} — the input to
`rlcli train sl`. Assistant tool calls are preserved in OpenAI shape
({"tool_calls": [{"id", "type", "function": {"name", "arguments"}}]}) and
tool results as {"role": "tool", "content": ..., "tool_call_id": ...} — the
exact message shapes the cookbook renderers (qwen3, kimi, ...) train on.
Imported traces feed SFT/distillation; RL needs an environment and reward
(Harbor), not raw traces.

Fail-loud: unknown formats and malformed lines raise ImportFormatError with
the line number rather than silently degrading the dataset.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

FORMATS = ("messages", "openai", "anthropic", "langsmith", "csv", "vercel")

_ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "ai": "assistant",
    "model": "assistant",
    "assistant": "assistant",
    "system": "system",
}

# Tool plumbing roles: preserved as role "tool" by default, dropped with
# preserve_tools=False.
_TOOL_ROLES = {"tool", "function", "tool_result"}
_TEXT_BLOCK_TYPES = {"text"}


class ImportFormatError(ValueError):
    pass


def normalize_role(role: Any, preserve_tools: bool = False) -> str | None:
    """Map provider role names onto user/assistant/system/tool; None = drop."""
    if not isinstance(role, str):
        return None
    role = role.lower()
    if role in _TOOL_ROLES:
        return "tool" if preserve_tools else None
    return _ROLE_ALIASES.get(role)


def _normalize_tool_calls(raw: Any) -> list[dict]:
    """Coerce provider tool-call lists to the OpenAI/renderer shape. Entries
    without a function name are dropped; dict arguments are JSON-encoded."""
    calls = []
    if not isinstance(raw, list):
        return calls
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        args = fn.get("arguments", fn.get("input", {}))
        if not isinstance(args, str):
            args = json.dumps(args)
        call = {"id": tc.get("id") or f"call_{len(calls)}", "type": "function",
                "function": {"name": name, "arguments": args}}
        calls.append(call)
    return calls


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


def _normalize_message_list(
    raw_messages: list, lineno: int, preserve_tools: bool = True
) -> list[dict]:
    messages = []
    for m in raw_messages:
        if not isinstance(m, dict):
            raise ImportFormatError(f"line {lineno}: message is not an object: {m!r}")
        role = normalize_role(m.get("role"), preserve_tools=preserve_tools)
        if role is None:
            continue  # unmapped roles; tool plumbing when preserve_tools=False
        content = flatten_content(m.get("content"))
        out: dict = {"role": role, "content": content}
        if role == "assistant" and preserve_tools:
            calls = _normalize_tool_calls(m.get("tool_calls"))
            if calls:
                out["tool_calls"] = calls
        if role == "tool":
            tc_id = m.get("tool_call_id")
            if isinstance(tc_id, str) and tc_id:
                out["tool_call_id"] = tc_id
        if not content.strip() and "tool_calls" not in out and role != "tool":
            continue  # empty non-tool turns
        messages.append(out)
    return messages


def _parse_messages_line(
    record: dict, lineno: int, preserve_tools: bool = True
) -> list[dict]:
    if "messages" not in record:
        raise ImportFormatError(
            f"line {lineno}: expected a 'messages' key, got keys {sorted(record)}"
        )
    if not isinstance(record["messages"], list):
        raise ImportFormatError(f"line {lineno}: 'messages' is not a list")
    return _normalize_message_list(record["messages"], lineno,
                                   preserve_tools=preserve_tools)


def _explode_anthropic_message(m: dict, preserve_tools: bool) -> list[dict]:
    """One Anthropic message → renderer-shaped messages. Assistant tool_use
    blocks become OpenAI-style tool_calls; user tool_result blocks become
    role-"tool" messages (in block order, before any user text)."""
    role = normalize_role(m.get("role"))
    content = m.get("content")
    if isinstance(content, str) or not isinstance(content, list):
        return [dict(m)]
    text = flatten_content(content)
    if not preserve_tools:
        return [{"role": m.get("role"), "content": text}]
    out: list[dict] = []
    if role == "assistant":
        calls = [
            {"id": b.get("id") or f"call_{i}", "type": "function",
             "function": {"name": b["name"],
                          "arguments": json.dumps(b.get("input", {}))}}
            for i, b in enumerate(content)
            if isinstance(b, dict) and b.get("type") == "tool_use"
            and isinstance(b.get("name"), str)
        ]
        msg: dict = {"role": "assistant", "content": text}
        if calls:
            msg["tool_calls"] = calls
        out.append(msg)
    else:
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                tool_msg: dict = {"role": "tool",
                                  "content": flatten_content(b.get("content"))}
                if isinstance(b.get("tool_use_id"), str):
                    tool_msg["tool_call_id"] = b["tool_use_id"]
                out.append(tool_msg)
        if text.strip():
            out.append({"role": m.get("role"), "content": text})
    return out


def _parse_anthropic_line(
    record: dict, lineno: int, preserve_tools: bool = True
) -> list[dict]:
    """Anthropic Messages API request/response dumps: top-level `system` plus
    `messages` with content blocks (text / tool_use / tool_result)."""
    messages = []
    system = record.get("system")
    if isinstance(system, (str, list)):
        text = flatten_content(system)
        if text.strip():
            messages.append({"role": "system", "content": text})
    raw = record.get("messages")
    if not isinstance(raw, list):
        raise ImportFormatError(
            f"line {lineno}: expected a 'messages' key, got keys {sorted(record)}"
        )
    exploded: list[dict] = []
    for m in raw:
        if not isinstance(m, dict):
            raise ImportFormatError(f"line {lineno}: message is not an object: {m!r}")
        exploded.extend(_explode_anthropic_message(m, preserve_tools))
    messages.extend(_normalize_message_list(exploded, lineno,
                                            preserve_tools=preserve_tools))
    return messages


def _parse_vercel_line(
    record: dict, lineno: int, preserve_tools: bool = True
) -> list[dict]:
    """Vercel AI SDK message dumps: messages may carry `parts`
    ([{"type": "text", "text": ...}, ...]) instead of `content`."""
    if "messages" not in record or not isinstance(record["messages"], list):
        raise ImportFormatError(
            f"line {lineno}: expected a 'messages' key, got keys {sorted(record)}"
        )
    flat = []
    for m in record["messages"]:
        if not isinstance(m, dict):
            raise ImportFormatError(f"line {lineno}: message is not an object: {m!r}")
        if "parts" in m and isinstance(m["parts"], list):
            text = "\n".join(
                p.get("text", "") for p in m["parts"]
                if isinstance(p, dict) and p.get("type") == "text"
                and isinstance(p.get("text"), str)
            )
            flat.append({"role": m.get("role"), "content": text})
        else:
            flat.append(m)
    return _normalize_message_list(flat, lineno, preserve_tools=preserve_tools)


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


def _parse_langsmith_line(
    record: dict, lineno: int, preserve_tools: bool = True
) -> list[dict]:
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
    messages = _normalize_message_list(raw, lineno, preserve_tools=preserve_tools)
    reply = _langsmith_output_message(record.get("outputs"))
    if reply is not None:
        normalized = _normalize_message_list([reply], lineno,
                                             preserve_tools=preserve_tools)
        # Chat runs sometimes echo the full history in outputs.messages; only
        # append when it's genuinely the reply, not a duplicate of the tail.
        if normalized and (not messages or messages[-1] != normalized[0]):
            messages.extend(normalized)
    return messages


_PARSERS = {
    "messages": _parse_messages_line,
    "openai": _parse_messages_line,
    "anthropic": _parse_anthropic_line,
    "langsmith": _parse_langsmith_line,
    "vercel": _parse_vercel_line,
}

# CSV column names recognized by iter_csv_conversations, in lookup order.
_CSV_ROLE_COLUMNS = (("user", "assistant"), ("prompt", "completion"),
                     ("input", "output"), ("question", "answer"))


def iter_csv_conversations(
    lines: Iterator[str],
) -> Iterator[tuple[dict, int]]:
    """Single-turn conversations from CSV: one row = one exchange.

    Recognized column pairs (first match wins): user/assistant,
    prompt/completion, input/output, question/answer; optional `system` and
    `reward` columns.
    """
    import csv as _csv

    reader = _csv.DictReader(lines)
    if reader.fieldnames is None:
        raise ImportFormatError("CSV input is empty")
    fields = {f.lower(): f for f in reader.fieldnames}
    pair = next((p for p in _CSV_ROLE_COLUMNS
                 if p[0] in fields and p[1] in fields), None)
    if pair is None:
        raise ImportFormatError(
            f"CSV columns {sorted(fields)} — need one of "
            + ", ".join("/".join(p) for p in _CSV_ROLE_COLUMNS)
        )
    for lineno, row in enumerate(reader, start=2):  # 1 = header
        user = (row.get(fields[pair[0]]) or "").strip()
        assistant = (row.get(fields[pair[1]]) or "").strip()
        if not user or not assistant:
            continue
        messages = []
        system = (row.get(fields.get("system", "")) or "").strip() if "system" in fields else ""
        if system:
            messages.append({"role": "system", "content": system})
        messages += [{"role": "user", "content": user},
                     {"role": "assistant", "content": assistant}]
        out: dict = {"messages": messages}
        if "reward" in fields:
            raw = (row.get(fields["reward"]) or "").strip()
            if raw:
                try:
                    out["reward"] = float(raw)
                except ValueError as e:
                    raise ImportFormatError(
                        f"line {lineno}: reward {raw!r} is not a number") from e
        yield out, lineno


def iter_conversations(
    lines: Iterator[str],
    fmt: str,
    min_messages: int = 2,
    feedback_key: str | None = None,
    preserve_tools: bool = True,
) -> Iterator[tuple[dict, int]]:
    """Yield (record, lineno) per usable input line, where record is the
    train-ready output object: {"messages": [...]} plus, when available, a
    "reward" float and a "trace_id" string (for telemetry joins).

    A conversation is usable when it has at least `min_messages` messages
    including at least one assistant message (text or tool calls — something
    to train on).
    """
    if fmt == "csv":
        yield from iter_csv_conversations(lines)
        return
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
        messages = parser(record, lineno, preserve_tools=preserve_tools)
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
        # Caller-owned trace id (Trajectory-style correlation): lets
        # `--telemetry` join product events onto conversations.
        for key in ("trace_id", "id", "run_id"):
            if isinstance(record.get(key), str) and record[key]:
                out["trace_id"] = record[key]
                break
        yield out, lineno
