"""rlcli serve — manage the local SkyRL Tinker server."""

from __future__ import annotations

import json

import click

from rlcli import server as srv

# vLLM inherits the model's native context when uncapped; on long-context
# models (e.g. Qwen3-4B-Instruct-2507 advertises 262K) the engine core
# segfaults sizing its buffers. Cap by default; raise deliberately for
# long-context training with the memory to back it.
DEFAULT_MAX_MODEL_LEN = 16384

TORCH_BACKENDS = ("fsdp", "megatron")


def resolve_max_model_len(base_model: str, requested: int) -> int | None:
    """Default context cap: min(model native, DEFAULT). Different models have
    different native lengths — a cap above native is a vLLM startup error,
    while native uncapped can be 262K+ and crash the engine. `0` disables.

    An explicit non-default request is trusted as-is."""
    if requested == 0:
        return None
    if requested != DEFAULT_MAX_MODEL_LEN:
        return requested
    try:
        from transformers import AutoConfig

        native = getattr(
            AutoConfig.from_pretrained(base_model), "max_position_embeddings", None
        )
    except Exception:
        native = None
    if isinstance(native, int) and native > 0:
        return min(native, DEFAULT_MAX_MODEL_LEN)
    return DEFAULT_MAX_MODEL_LEN


def build_backend_overrides(
    backend: str,
    gpus: int | None,
    nodes: int | None,
    tp: int | None,
    max_model_len: int | None,
    user_json: str | None,
) -> dict:
    """Merge rlcli's defaults with user --backend-config; user keys win.

    Torch-backend keys (trainer.placement, inference engine) are only applied
    on fsdp/megatron — the JAX backend rejects unknown fields.
    """
    overrides: dict = {}
    if backend in TORCH_BACKENDS:
        if gpus:
            overrides["trainer.placement.policy_num_gpus_per_node"] = gpus
            overrides["generator.inference_engine.num_engines"] = (
                gpus if not tp else max(1, gpus // tp)
            )
        if nodes:
            overrides["trainer.placement.policy_num_nodes"] = nodes
        if tp:
            overrides["generator.inference_engine.tensor_parallel_size"] = tp
        if max_model_len:
            overrides["generator.inference_engine.engine_init_kwargs"] = {
                "max_model_len": max_model_len
            }
    elif gpus or nodes or tp:
        raise click.UsageError("--gpus/--nodes/--tp apply to fsdp/megatron backends only.")

    if user_json:
        user = json.loads(user_json)
        user_engine_kwargs = user.get("generator.inference_engine.engine_init_kwargs")
        if isinstance(user_engine_kwargs, dict):
            merged = {
                **overrides.get("generator.inference_engine.engine_init_kwargs", {}),
                **user_engine_kwargs,
            }
            user["generator.inference_engine.engine_init_kwargs"] = merged
        overrides.update(user)
    return overrides


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
@click.option("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN, show_default=True,
              help="Cap vLLM context length (torch backends). 0 = model's native context — "
                   "long-context models can crash the engine without enough GPU/host memory.")
@click.option("--backend-config", default=None, help="Raw JSON of SkyRL-Train overrides; merged last.")
@click.option("--wait", "wait_seconds", type=int, default=900, show_default=True, help="Startup timeout (s).")
def start(base_model, backend, port, gpus, nodes, tp, checkpoints_base, max_model_len,
          backend_config, wait_seconds):
    """Install (first run) and start the server, then wait until healthy."""
    resolved_len = resolve_max_model_len(base_model, max_model_len) if backend in TORCH_BACKENDS else None
    if resolved_len:
        click.echo(f"Context cap: max_model_len={resolved_len} (override with --max-model-len, 0 = native)")
    overrides = build_backend_overrides(backend, gpus, nodes, tp, resolved_len, backend_config)
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
