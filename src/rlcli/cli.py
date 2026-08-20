"""rlcli entrypoint — lazy-loaded Click group for fast startup."""

from __future__ import annotations

import importlib
import sys

import click

LAZY_SUBCOMMANDS = {
    "import": "rlcli.commands.import_cmd:cli",
    "serve": "rlcli.commands.serve:cli",
    "train": "rlcli.commands.train:cli",
    "sample": "rlcli.commands.sample:cli",
    "checkpoint": "rlcli.commands.passthrough:checkpoint",
    "run": "rlcli.commands.passthrough:run",
    "session": "rlcli.commands.passthrough:session",
    "version": "rlcli.commands.version:cli",
}


class LazyGroup(click.Group):
    def list_commands(self, ctx):
        return sorted(LAZY_SUBCOMMANDS)

    def get_command(self, ctx, name):
        target = LAZY_SUBCOMMANDS.get(name)
        if target is None:
            return None
        module_name, attr = target.split(":")
        module = importlib.import_module(module_name)
        return getattr(module, attr)


@click.group(cls=LazyGroup, context_settings={"help_option_names": ["-h", "--help"]})
def main_cli():
    """rlcli — one-shot Tinker-API post-training on your own GPUs (SkyRL backend)."""


def main():
    try:
        main_cli()
    except Exception as e:  # surface clean errors, not tracebacks, for known failures
        from rlcli.compat import LossBackendError
        from rlcli.server import ServerError

        if isinstance(e, (LossBackendError, ServerError)):
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        raise
