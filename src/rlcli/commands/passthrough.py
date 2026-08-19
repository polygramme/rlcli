"""Management commands: thin passthrough to the official tinker CLI,
pointed at the local server by default. True drop-in parity — same
commands, your base URL."""

from __future__ import annotations

import os
import subprocess
import sys

import click

DEFAULT_BASE_URL = "http://localhost:8000"


def _passthrough(name: str, args: tuple[str, ...], base_url: str) -> None:
    env = {
        **os.environ,
        "TINKER_BASE_URL": os.environ.get("TINKER_BASE_URL", base_url),
    }
    env.setdefault("TINKER_API_KEY", "tml-dummy")
    result = subprocess.run([sys.executable, "-m", "tinker.cli", name, *args], env=env)
    sys.exit(result.returncode)


def _make(name: str, help_text: str):
    @click.command(
        name,
        context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
        add_help_option=False,
        help=help_text,
    )
    @click.option("--base-url", default=DEFAULT_BASE_URL, show_default=True)
    @click.argument("args", nargs=-1, type=click.UNPROCESSED)
    def cmd(base_url, args):
        _passthrough(name, args, base_url)

    return cmd


checkpoint = _make("checkpoint", "Checkpoint management (tinker CLI, local base URL).")
run = _make("run", "Training-run listing/info (tinker CLI, local base URL).")
session = _make("session", "Session management (tinker CLI, local base URL).")
