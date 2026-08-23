"""Join product telemetry events onto imported conversations.

Trajectory-style correlation: your application emits events carrying the same
caller-owned trace_id as the conversation (thumbs-up, conversion,
ticket-reopened, ...), and those events — not just evaluator feedback —
become the training reward.

Event lines are JSON objects: {"trace_id": ..., "score": ...} with optional
"key" (event name) and anything else (ignored). "value" is accepted as an
alias for "score"; booleans coerce to 0/1.
"""

from __future__ import annotations

import json
from typing import Iterator


class TelemetryFormatError(ValueError):
    pass


def load_telemetry(lines: Iterator[str], key: str | None = None) -> dict[str, float]:
    """Map trace_id → mean score across its events (filtered by `key` if
    given). Fail-loud on malformed lines; events without trace_id or score
    raise rather than silently vanishing."""
    sums: dict[str, list[float]] = {}
    seen_keys: set[str] = set()
    for lineno, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as e:
            raise TelemetryFormatError(f"telemetry line {lineno}: invalid JSON: {e}") from e
        if not isinstance(event, dict):
            raise TelemetryFormatError(f"telemetry line {lineno}: expected a JSON object")
        trace_id = event.get("trace_id")
        if not isinstance(trace_id, str) or not trace_id:
            raise TelemetryFormatError(
                f"telemetry line {lineno}: missing 'trace_id' (keys: {sorted(event)})"
            )
        if isinstance(event.get("key"), str):
            seen_keys.add(event["key"])
            if key is not None and event["key"] != key:
                continue
        score = event.get("score", event.get("value"))
        if isinstance(score, bool):
            score = float(score)
        if not isinstance(score, (int, float)):
            raise TelemetryFormatError(
                f"telemetry line {lineno}: missing numeric 'score'/'value' "
                f"(keys: {sorted(event)})"
            )
        sums.setdefault(trace_id, []).append(float(score))
    if key is None and len(seen_keys) > 1:
        raise TelemetryFormatError(
            f"telemetry has multiple event keys {sorted(seen_keys)}; "
            "pass --telemetry-key"
        )
    return {tid: sum(vals) / len(vals) for tid, vals in sums.items()}
