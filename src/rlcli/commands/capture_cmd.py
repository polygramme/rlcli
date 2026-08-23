"""rlcli capture — live trace capture via a transparent OpenAI-compatible proxy."""

from __future__ import annotations

import os

import click

from rlcli.capture import serve_capture


@click.command("capture")
@click.option("--listen", default="127.0.0.1:8399", show_default=True,
              help="host:port to listen on.")
@click.option("--upstream", required=True,
              help="Upstream OpenAI-compatible base URL, e.g. "
                   "https://api.openai.com/v1 or a local vLLM.")
@click.option("--out", "out_path", required=True,
              type=click.Path(dir_okay=False),
              help="JSONL file to append captured traces to (import-ready).")
@click.option("--api-key-env", default=None,
              help="Env var whose value is added as the upstream Bearer token "
                   "when the caller sends none.")
def cli(listen, upstream, out_path, api_key_env):
    """Run a capture proxy: point your agent's base_url here and every chat
    completion (streaming included, tool calls included) is appended to OUT
    as training-ready messages JSONL."""
    api_key = os.environ.get(api_key_env) if api_key_env else None
    server = serve_capture(listen, upstream, out_path, api_key)
    click.echo(f"capturing {listen} → {upstream}; traces → {out_path} (Ctrl-C to stop)",
               err=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("capture stopped", err=True)
    finally:
        server.server_close()
