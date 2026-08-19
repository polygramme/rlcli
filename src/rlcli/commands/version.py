from __future__ import annotations

import click


@click.command("version")
def cli():
    """Show rlcli and pinned dependency versions."""
    import rlcli

    click.echo(f"rlcli {rlcli.__version__}")
    try:
        import tinker

        click.echo(f"tinker {getattr(tinker, '__version__', 'unknown')}")
    except ImportError:
        click.echo("tinker not installed")
    click.echo(f"skyrl pin {rlcli.SKYRL_PIN[:12]}")
    click.echo(f"tinker-cookbook pin {rlcli.COOKBOOK_PIN[:12]}")
