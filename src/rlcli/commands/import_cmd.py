"""rlcli import — normalize chat dumps into train-ready messages JSONL.

Pipes into training: rlcli import -f openai dump.jsonl | rlcli train sl --dataset - ...
Tool calls are preserved by default (renderer-shaped); telemetry events can be
joined on trace_id to supply rewards; PII redaction runs before anything is
written.
"""

from __future__ import annotations

import json
import sys

import click

from rlcli.importers import (
    FORMATS,
    ImportFormatError,
    iter_conversations,
    iter_langsmith_project,
)
from rlcli.redact import RedactionError, build_redactors, redact_record
from rlcli.telemetry import TelemetryFormatError, load_telemetry


@click.command("import")
@click.argument("input_file", required=False,
                type=click.Path(exists=True, dir_okay=False, allow_dash=True))
@click.option("--project", default=None,
              help="Pull root runs live from this LangSmith project instead of "
                   "reading INPUT_FILE (needs `pip install langsmith` + "
                   "LANGSMITH_API_KEY; implies -f langsmith).")
@click.option("--format", "-f", "fmt", type=click.Choice(FORMATS), default="messages",
              show_default=True,
              help="messages: {'messages': [...]} per line. openai: chat-completions "
                   "dumps. anthropic: Messages API dumps (content blocks, tool_use/"
                   "tool_result). langsmith: run exports (feedback → 'reward'). "
                   "csv: one exchange per row (user/assistant, prompt/completion, "
                   "input/output or question/answer columns; optional system, reward). "
                   "vercel: AI SDK messages with 'parts'.")
@click.option("--out", "-o", type=click.Path(dir_okay=False, allow_dash=True), default="-",
              show_default=True, help="Output path; '-' streams to stdout for piping.")
@click.option("--min-messages", type=int, default=2, show_default=True,
              help="Skip conversations with fewer usable messages.")
@click.option("--limit", type=int, default=None, help="Stop after N conversations.")
@click.option("--drop-tools", is_flag=True, default=False,
              help="Strip tool calls/results instead of preserving them "
                   "(pre-0.2 behavior).")
@click.option("--feedback-key", default=None,
              help="langsmith only: which feedback key supplies the reward "
                   "(required when runs carry more than one).")
@click.option("--telemetry", "telemetry_file", type=click.Path(exists=True, dir_okay=False),
              default=None,
              help="Telemetry events JSONL ({'trace_id', 'score', optional 'key'}); "
                   "mean score per trace_id becomes the conversation's reward.")
@click.option("--telemetry-key", default=None,
              help="Only use telemetry events with this key (required when the "
                   "file carries more than one).")
@click.option("--min-score", type=float, default=None,
              help="Keep only conversations whose reward is >= this; "
                   "conversations without a reward are dropped. "
                   "(Successes → SFT.)")
@click.option("--max-score", type=float, default=None,
              help="Keep only conversations whose reward is <= this; "
                   "conversations without a reward are dropped. "
                   "(Failures → `rlcli synth` test environments.)")
@click.option("--redact", "redact_names", multiple=True,
              help="Built-in PII redaction to apply (email, phone, ssn, "
                   "credit_card, ipv4, api_key, or 'all'). Repeatable.")
@click.option("--redact-pattern", "redact_custom", multiple=True,
              help="Custom redaction as name=regex. Repeatable.")
def cli(input_file, project, fmt, out, min_messages, limit, drop_tools,
        feedback_key, telemetry_file, telemetry_key, min_score, max_score,
        redact_names, redact_custom):
    """Convert INPUT_FILE ('-' for stdin) to training-ready messages JSONL."""
    if project is not None:
        if input_file is not None:
            raise click.UsageError("Give either INPUT_FILE or --project, not both.")
        fmt = "langsmith"
    elif input_file is None:
        raise click.UsageError("Missing INPUT_FILE (or use --project).")
    if fmt != "langsmith" and feedback_key is not None:
        raise click.UsageError("--feedback-key only applies to -f langsmith")
    score_filter = min_score is not None or max_score is not None
    if score_filter and fmt not in ("langsmith", "csv", "messages") \
            and telemetry_file is None:
        raise click.UsageError(
            "--min-score/--max-score need a reward source: -f langsmith, "
            "-f csv with a reward column, -f messages with reward fields, "
            "or --telemetry")
    try:
        redactors = build_redactors(list(redact_names), list(redact_custom))
    except RedactionError as e:
        raise click.ClickException(str(e))
    rewards_by_trace: dict[str, float] = {}
    if telemetry_file is not None:
        try:
            with open(telemetry_file, "r", encoding="utf-8") as tf:
                rewards_by_trace = load_telemetry(tf, key=telemetry_key)
        except TelemetryFormatError as e:
            raise click.ClickException(str(e))

    if project is not None:
        src = iter_langsmith_project(project, limit=limit)
    elif input_file == "-":
        src = sys.stdin
    else:
        src = open(input_file, "r", encoding="utf-8")
    dst = sys.stdout if out == "-" else open(out, "w", encoding="utf-8")
    written = skipped_low = joined = 0
    try:
        for record, _lineno in iter_conversations(
            src, fmt, min_messages=min_messages, feedback_key=feedback_key,
            preserve_tools=not drop_tools,
        ):
            trace_id = record.get("trace_id")
            if trace_id in rewards_by_trace:
                record["reward"] = rewards_by_trace[trace_id]
                joined += 1
            if score_filter and (
                record.get("reward") is None
                or (min_score is not None and record["reward"] < min_score)
                or (max_score is not None and record["reward"] > max_score)
            ):
                skipped_low += 1
                continue
            record = redact_record(record, redactors)
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if limit is not None and written >= limit:
                break
    except ImportFormatError as e:
        raise click.ClickException(str(e))
    finally:
        if src is not sys.stdin:
            src.close()
        if dst is not sys.stdout:
            dst.close()
    notes = []
    if telemetry_file is not None:
        notes.append(f"{joined} telemetry-joined")
    if score_filter:
        notes.append(f"{skipped_low} outside score filter")
    suffix = f" ({', '.join(notes)})" if notes else ""
    click.echo(f"Wrote {written} conversations ({fmt} → messages JSONL){suffix}", err=True)
    if written == 0:
        raise click.ClickException(
            "No usable conversations found (need an assistant message and "
            f"--min-messages={min_messages}). Check --format."
        )
