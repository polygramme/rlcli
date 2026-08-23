"""Pluggable PII redaction, applied before trace data is written anywhere.

Built-in patterns cover the common identifiers; custom patterns extend them
(`name=regex`). Matches are replaced with [REDACTED:name] so redacted data
stays debuggable. Deliberately regex-based and conservative: rlcli's promise
is that nothing leaves the machine, so redaction here is defense-in-depth for
teams whose training hosts are shared, not a compliance product.
"""

from __future__ import annotations

import re

BUILTIN_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone": re.compile(r"(?<![\d.])\+?\d[\d\s().-]{7,}\d(?![\d.])"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "api_key": re.compile(
        r"\b(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}"
        r"|pypi-[A-Za-z0-9_-]{20,}|xox[bap]-[A-Za-z0-9-]{10,}|AKIA[A-Z0-9]{16})\b"
    ),
}


class RedactionError(ValueError):
    pass


def build_redactors(
    names: list[str], custom: list[str]
) -> dict[str, re.Pattern]:
    """Resolve --redact names and name=regex customs to compiled patterns."""
    redactors: dict[str, re.Pattern] = {}
    for name in names:
        if name == "all":
            redactors.update(BUILTIN_PATTERNS)
            continue
        if name not in BUILTIN_PATTERNS:
            raise RedactionError(
                f"unknown redaction {name!r}; built-ins: "
                f"{', '.join(BUILTIN_PATTERNS)}, all"
            )
        redactors[name] = BUILTIN_PATTERNS[name]
    for spec in custom:
        name, sep, pattern = spec.partition("=")
        if not sep or not name or not pattern:
            raise RedactionError(f"custom pattern must be name=regex, got {spec!r}")
        try:
            redactors[name] = re.compile(pattern)
        except re.error as e:
            raise RedactionError(f"bad regex for {name!r}: {e}") from e
    return redactors


def redact_text(text: str, redactors: dict[str, re.Pattern]) -> str:
    for name, pattern in redactors.items():
        text = pattern.sub(f"[REDACTED:{name}]", text)
    return text


def redact_record(record: dict, redactors: dict[str, re.Pattern]) -> dict:
    """Redact every content string in a conversation record (message content
    and tool-call arguments); other fields pass through."""
    if not redactors:
        return record
    out = dict(record)
    messages = []
    for m in record.get("messages", []):
        m = dict(m)
        if isinstance(m.get("content"), str):
            m["content"] = redact_text(m["content"], redactors)
        if isinstance(m.get("tool_calls"), list):
            calls = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function", {}))
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = redact_text(fn["arguments"], redactors)
                tc["function"] = fn
                calls.append(tc)
            m["tool_calls"] = calls
        messages.append(m)
    out["messages"] = messages
    return out
