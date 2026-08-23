"""rlcli synth — traces in, Harbor-format training environments out.

Composes with import: `rlcli import dump.jsonl -f langsmith --min-score 0.8 |
rlcli synth - --out ./tasks ...` then `rlcli train harbor --dataset ./tasks`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click

from rlcli.importers import ImportFormatError, iter_conversations
from rlcli.synth import SynthError, openai_compat_generate, synthesize


@click.command("synth")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, allow_dash=True))
@click.option("--out", "-o", "out_dir", type=click.Path(file_okay=False), required=True,
              help="Directory that receives one Harbor-format task per conversation.")
@click.option("--model", required=True, help="Generator model name at the endpoint.")
@click.option("--api-base", default="https://api.openai.com/v1", show_default=True,
              help="OpenAI-compatible endpoint (vLLM, OpenRouter, Fireworks, ...).")
@click.option("--api-key-env", default="OPENAI_API_KEY", show_default=True,
              help="Env var holding the endpoint's API key (unset = no auth header).")
@click.option("--limit", type=int, default=None, help="Synthesize at most N tasks.")
@click.option("--validate/--no-validate", default=True, show_default=True,
              help="Build each task's image and require the test to FAIL on an "
                   "untouched container (needs a local Docker daemon).")
@click.option("--validate-timeout", type=int, default=300, show_default=True)
@click.option("--report", type=click.Path(dir_okay=False), default=None,
              help="Also write the per-task reports as JSONL.")
def cli(input_file, out_dir, model, api_base, api_key_env, limit, validate,
        validate_timeout, report):
    """Synthesize sandboxed training tasks from a messages-JSONL trace file.

    INPUT_FILE is `rlcli import` output ('-' for stdin): one
    {"messages": [...]} object per line, optional "reward" carried into the
    task's lineage metadata.
    """
    src = sys.stdin if input_file == "-" else open(input_file, "r", encoding="utf-8")
    try:
        conversations = list(iter_conversations(src, "messages"))
    except ImportFormatError as e:
        raise click.ClickException(str(e))
    finally:
        if src is not sys.stdin:
            src.close()
    if limit is not None:
        conversations = conversations[:limit]
    if not conversations:
        raise click.ClickException("No usable conversations in input.")

    api_key = os.environ.get(api_key_env)

    def generate(prompt_messages):
        return openai_compat_generate(api_base, api_key, model, prompt_messages)

    try:
        reports = synthesize(
            conversations,
            Path(out_dir),
            generate,
            source_file="-" if input_file == "-" else str(Path(input_file).resolve()),
            validate=validate,
            validate_timeout=validate_timeout,
        )
    except SynthError as e:
        raise click.ClickException(str(e))

    created = [r for r in reports if r["task_dir"]]
    valid = [r for r in reports if r["valid"]]
    for r in reports:
        status = "VALID" if r["valid"] else ("?" if r["valid"] is None else "INVALID")
        where = r["task_dir"] or f"line {r['lineno']}"
        click.echo(f"[{status}] {where}: {r['detail']}", err=True)
    if report:
        with open(report, "w", encoding="utf-8") as f:
            for r in reports:
                f.write(json.dumps(r) + "\n")
    click.echo(
        f"Synthesized {len(created)}/{len(reports)} tasks"
        + (f", {len(valid)} validated" if validate else "")
        + f" → {out_dir}",
        err=True,
    )
    if not created:
        raise click.ClickException("No tasks were synthesized.")
