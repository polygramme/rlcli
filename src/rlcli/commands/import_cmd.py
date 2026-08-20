"""rlcli import — normalize chat-dump JSONL into train-ready messages JSONL.

Pipes into training: rlcli import -f openai dump.jsonl | rlcli train sl --dataset - ...
"""

from __future__ import annotations

import json
import sys

import click

from rlcli.importers import FORMATS, ImportFormatError, iter_conversations


@click.command("import")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, allow_dash=True))
@click.option("--format", "-f", "fmt", type=click.Choice(FORMATS), default="messages",
              show_default=True,
              help="messages: {'messages': [...]} per line. openai: chat-completions "
                   "dumps (tool_calls turns dropped). anthropic: Messages API dumps "
                   "(content blocks flattened, top-level system kept).")
@click.option("--out", "-o", type=click.Path(dir_okay=False, allow_dash=True), default="-",
              show_default=True, help="Output path; '-' streams to stdout for piping.")
@click.option("--min-messages", type=int, default=2, show_default=True,
              help="Skip conversations with fewer usable messages.")
@click.option("--limit", type=int, default=None, help="Stop after N conversations.")
def cli(input_file, fmt, out, min_messages, limit):
    """Convert INPUT_FILE ('-' for stdin) to training-ready messages JSONL."""
    src = sys.stdin if input_file == "-" else open(input_file, "r", encoding="utf-8")
    dst = sys.stdout if out == "-" else open(out, "w", encoding="utf-8")
    written = 0
    try:
        for messages, _lineno in iter_conversations(src, fmt, min_messages=min_messages):
            dst.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
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
    click.echo(f"Wrote {written} conversations ({fmt} → messages JSONL)", err=True)
    if written == 0:
        raise click.ClickException(
            "No usable conversations found (need an assistant message and "
            f"--min-messages={min_messages}). Check --format."
        )
