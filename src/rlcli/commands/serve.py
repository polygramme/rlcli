"""rlcli serve — manage the local SkyRL Tinker server."""

from __future__ import annotations

import json

import click

from rlcli import server as srv


@click.group("serve")
def cli():
    """Start, stop, and inspect the local SkyRL Tinker server."""


@cli.command("start")
@click.option("--base-model", required=True, help="HuggingFace model name or path.")
@click.option(
    "--backend",
    type=click.Choice(["jax", "fsdp", "megatron"]),
    default="jax",
    show_default=True,
    help="jax runs anywhere (CPU ok, no fused gspo); fsdp/megatron need Linux+CUDA and serve the full loss set.",
)
@click.option("--port", type=int, default=8000, show_default=True)
@click.option("--gpus", type=int, default=None, help="GPUs per node; also sets one vLLM engine per GPU.")
@click.option("--nodes", type=int, default=None, help="Training nodes.")
@click.option("--tp", type=int, default=None, help="Tensor-parallel size per inference engine.")
@click.option("--checkpoints-base", default=None, help="Checkpoint directory (default: server's).")
@click.option("--backend-config", default=None, help="Raw JSON of SkyRL-Train overrides; merged last.")
@click.option("--wait", "wait_seconds", type=int, default=900, show_default=True, help="Startup timeout (s).")
def start(base_model, backend, port, gpus, nodes, tp, checkpoints_base, backend_config, wait_seconds):
    """Install (first run) and start the server, then wait until healthy."""
    overrides: dict = {}
    if gpus:
        overrides["trainer.placement.policy_num_gpus_per_node"] = gpus
        overrides["generator.inference_engine.num_engines"] = gpus if not tp else max(1, gpus // tp)
    if nodes:
        overrides["trainer.placement.policy_num_nodes"] = nodes
    if tp:
        overrides["generator.inference_engine.tensor_parallel_size"] = tp
    if backend_config:
        overrides.update(json.loads(backend_config))
    srv.start(
        base_model=base_model,
        backend=backend,
        port=port,
        checkpoints_base=checkpoints_base,
        backend_config=overrides or None,
        wait_seconds=wait_seconds,
        log=click.echo,
    )


@cli.command("stop")
def stop():
    """Stop the managed server."""
    if srv.stop(log=click.echo):
        click.echo("Stopped.")
    else:
        click.echo("No managed server running.")


@cli.command("status")
def status():
    """Show the managed server's state and health."""
    state = srv.status()
    if state is None:
        click.echo("No managed server running.")
        return
    healthy = srv.health(state["port"])
    click.echo(json.dumps({**state, "healthy": healthy}, indent=2))


@cli.command("logs")
@click.option("-n", "lines", type=int, default=40, show_default=True)
def logs(lines):
    """Tail the server log."""
    click.echo(srv.tail_log(lines))
