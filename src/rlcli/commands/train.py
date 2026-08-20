"""rlcli train — one-shot training runs driven through tinker-cookbook.

Recipes are chz-configured plain functions (audit E23): we import them and
call main(config) programmatically instead of shelling out.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
from pathlib import Path

import click

from rlcli.compat import ALL_LOSSES, as_loss_fn, ensure_loss_supported
from rlcli.server import backend_for_url

DEFAULT_BASE_URL = "http://localhost:8000"


def _prepare_env(base_url: str) -> None:
    # Local SkyRL servers ignore the key but the SDK requires one to be set.
    os.environ.setdefault("TINKER_API_KEY", "tml-dummy")
    os.environ.setdefault("TINKER_BASE_URL", base_url)


def _run(coro_or_result):
    if inspect.iscoroutine(coro_or_result):
        return asyncio.run(coro_or_result)
    return coro_or_result


def _default_log_path(kind: str) -> str:
    return str(Path.home() / ".rlcli" / "runs" / f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}")


def _warn_known_upstream_issues(loss: str) -> None:
    # Native gspo at the pinned skyrl can return logprobs in packed (not input)
    # order for variable-length microbatches, failing with an IndexError in the
    # KL/metrics path before the first update. Fix pending upstream:
    # https://github.com/NovaSky-AI/SkyRL/pull/2043 — remove this when it merges
    # and the pin is bumped.
    if loss == "gspo":
        click.echo(
            "note: native gspo has a known upstream ordering bug with variable-length "
            "batches at the pinned SkyRL (NovaSky-AI/SkyRL#2043, fix in review). "
            "Verify the first update completes; importance_sampling/ppo are unaffected.",
            err=True,
        )


def _renderer_for(model_name: str, renderer_name: str | None) -> str:
    if renderer_name:
        return renderer_name
    from tinker_cookbook import model_info

    try:
        return model_info.get_recommended_renderer_name(model_name)
    except (KeyError, ValueError):  # cookbook raises ConfigurationError (a ValueError)
        raise click.UsageError(
            f"No recommended chat renderer known for {model_name!r}. "
            "Pass one explicitly with --renderer (e.g. qwen3, qwen3_instruct, llama3)."
        )


@click.group("train")
def cli():
    """One-shot training runs against a Tinker-API server."""


def _materialize_dataset(dataset_path: str) -> str:
    """Support --dataset - : spool stdin (e.g. piped `rlcli import`) to a file."""
    if dataset_path != "-":
        return dataset_path
    import sys
    import tempfile

    data = sys.stdin.read()
    if not data.strip():
        raise click.UsageError("--dataset - was given but stdin is empty.")
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", prefix="rlcli-dataset-", delete=False, encoding="utf-8"
    )
    f.write(data)
    f.close()
    return f.name


@cli.command("sl")
@click.option("--dataset", "dataset_path", required=True,
              help='JSONL file: one {"messages": [{"role", "content"}, ...]} object per '
                   "line. Use '-' to read from stdin (pipe from `rlcli import`).")
@click.option("--model", "model_name", required=True, help="HuggingFace model name.")
@click.option("--base-url", default=DEFAULT_BASE_URL, show_default=True)
@click.option("--renderer", "renderer_name", default=None,
              help="Chat renderer (default: recommended for the model).")
@click.option("--batch-size", type=int, default=8, show_default=True)
@click.option("--lr", "learning_rate", type=float, default=1e-4, show_default=True)
@click.option("--max-length", type=int, default=2048, show_default=True)
@click.option("--epochs", "num_epochs", type=int, default=1, show_default=True)
@click.option("--lora-rank", type=int, default=32, show_default=True)
@click.option("--save-every", type=int, default=20, show_default=True)
@click.option("--test-size", type=int, default=0, show_default=True)
@click.option("--log-path", default=None, help="Run directory (default: ~/.rlcli/runs/...).")
@click.option("--dry-run", is_flag=True, help="Build and print the config, don't train.")
def sl(dataset_path, model_name, base_url, renderer_name, batch_size, learning_rate,
       max_length, num_epochs, lora_rank, save_every, test_size, log_path, dry_run):
    """Supervised fine-tuning on a messages-JSONL dataset."""
    dataset_path = _materialize_dataset(dataset_path)
    _prepare_env(base_url)
    from tinker_cookbook.supervised import train as sl_train
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    renderer_name = _renderer_for(model_name, renderer_name)
    builder = FromConversationFileBuilder(
        file_path=dataset_path,
        test_size=test_size,
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            max_length=max_length,
            batch_size=batch_size,
        ),
    )
    config = sl_train.Config(
        log_path=log_path or _default_log_path("sl"),
        recipe_name="rlcli_train_sl",
        model_name=model_name,
        dataset_builder=builder,
        renderer_name=renderer_name,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        lora_rank=lora_rank,
        save_every=save_every,
        base_url=base_url,
    )
    if dry_run:
        click.echo(f"[dry-run] supervised config OK:\n{config}")
        return
    click.echo(f"Training {model_name} on {dataset_path} via {base_url} (renderer={renderer_name})")
    _run(sl_train.main(config))


@cli.command("rl")
@click.option("--model", "model_name", required=True, help="HuggingFace model name.")
@click.option("--base-url", default=DEFAULT_BASE_URL, show_default=True)
@click.option("--loss", default="importance_sampling", show_default=True,
              type=click.Choice(ALL_LOSSES),
              help="Fused losses (gspo/dppo/ppo_critic) need a fsdp or megatron server.")
@click.option("--loss-config", default=None, help='JSON, e.g. \'{"clip_low_threshold": 0.8}\'.')
@click.option("--backend", "backend_hint", default=None,
              type=click.Choice(["jax", "fsdp", "megatron"]),
              help="Backend of the target server, for the loss guard (auto-detected for managed servers).")
@click.option("--dataset", type=click.Choice(["gsm8k"]), default="gsm8k", show_default=True)
@click.option("--renderer", "renderer_name", default=None)
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option("--group-size", type=int, default=8, show_default=True)
@click.option("--lr", "learning_rate", type=float, default=4e-5, show_default=True)
@click.option("--max-tokens", type=int, default=256, show_default=True)
@click.option("--lora-rank", type=int, default=32, show_default=True)
@click.option("--save-every", type=int, default=20, show_default=True)
@click.option("--eval-every", type=int, default=0, show_default=True)
@click.option("--max-steps", type=int, default=None)
@click.option("--log-path", default=None)
@click.option("--dry-run", is_flag=True, help="Build and print the config, don't train.")
def rl(model_name, base_url, loss, loss_config, backend_hint, dataset, renderer_name,
       batch_size, group_size, learning_rate, max_tokens, lora_rank, save_every,
       eval_every, max_steps, log_path, dry_run):
    """RL on a built-in environment, with the full SkyRL loss set."""
    backend = backend_hint or backend_for_url(base_url)
    ensure_loss_supported(loss, backend)
    _warn_known_upstream_issues(loss)
    _prepare_env(base_url)

    from tinker_cookbook.recipes.math_rl.math_env import Gsm8kDatasetBuilder
    from tinker_cookbook.rl import train as rl_train

    renderer_name = _renderer_for(model_name, renderer_name)
    dataset_builder = Gsm8kDatasetBuilder(
        batch_size=batch_size,
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        group_size=group_size,
    )
    kwargs = dict(
        learning_rate=learning_rate,
        dataset_builder=dataset_builder,
        model_name=model_name,
        recipe_name="rlcli_train_rl",
        max_tokens=max_tokens,
        log_path=log_path or _default_log_path("rl"),
        renderer_name=renderer_name,
        lora_rank=lora_rank,
        save_every=save_every,
        eval_every=eval_every,
        base_url=base_url,
        loss_fn=as_loss_fn(loss),
        loss_fn_config=json.loads(loss_config) if loss_config else None,
    )
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    config = rl_train.Config(**kwargs)
    if dry_run:
        click.echo(f"[dry-run] rl config OK: loss_fn={loss} backend={backend or 'unknown'}")
        return
    click.echo(f"RL: {model_name} on {dataset} via {base_url}, loss_fn={loss}")
    _run(rl_train.main(config))


@cli.command("harbor")
@click.option("--model", "model_name", required=True, help="HuggingFace model name.")
@click.option("--base-url", default=DEFAULT_BASE_URL, show_default=True)
@click.option("--loss", default="importance_sampling", show_default=True,
              type=click.Choice(ALL_LOSSES),
              help="Fused losses (gspo/dppo/ppo_critic) need a fsdp or megatron server.")
@click.option("--loss-config", default=None, help='JSON, e.g. \'{"clip_low_threshold": 0.8}\'.')
@click.option("--backend", "backend_hint", default=None,
              type=click.Choice(["jax", "fsdp", "megatron"]),
              help="Backend of the target server, for the loss guard.")
@click.option("--dataset", default="terminal-bench@2.0", show_default=True,
              help="Harbor dataset under ~/.cache/harbor/tasks/ "
                   "(auto-downloaded via `uvx harbor datasets download` if missing).")
@click.option("--sandbox", type=click.Choice(["docker", "modal"]), default="docker",
              show_default=True, help="docker = local daemon, no cloud account needed.")
@click.option("--task-filter", default=None, help="Only tasks whose name contains this substring.")
@click.option("--limit", "task_limit", type=int, default=None, help="Use at most N tasks.")
@click.option("--renderer", "renderer_name", default=None)
@click.option("--group-size", type=int, default=4, show_default=True)
@click.option("--groups-per-batch", type=int, default=4, show_default=True)
@click.option("--lr", "learning_rate", type=float, default=1e-5, show_default=True)
@click.option("--max-tokens", type=int, default=4096, show_default=True)
@click.option("--max-turns", type=int, default=10, show_default=True)
@click.option("--lora-rank", type=int, default=32, show_default=True)
@click.option("--save-every", type=int, default=5, show_default=True)
@click.option("--eval-every", type=int, default=0, show_default=True)
@click.option("--max-steps", type=int, default=None)
@click.option("--sandbox-timeout", type=int, default=3600, show_default=True)
@click.option("--command-timeout", type=int, default=120, show_default=True)
@click.option("--log-path", default=None)
@click.option("--dry-run", is_flag=True, help="Load tasks and build the config, don't train.")
def harbor(model_name, base_url, loss, loss_config, backend_hint, dataset, sandbox,
           task_filter, task_limit, renderer_name, group_size, groups_per_batch,
           learning_rate, max_tokens, max_turns, lora_rank, save_every, eval_every,
           max_steps, sandbox_timeout, command_timeout, log_path, dry_run):
    """RL on Harbor tasks in local Docker sandboxes, reward from tests/test.sh."""
    backend = backend_hint or backend_for_url(base_url)
    ensure_loss_supported(loss, backend)
    _warn_known_upstream_issues(loss)
    _prepare_env(base_url)

    from tinker_cookbook.recipes.harbor_rl.harbor_env import (
        HARBOR_CACHE_DIR,
        HarborDatasetBuilder,
        load_harbor_tasks,
    )
    from tinker_cookbook.rl import train as rl_train

    dataset_dir = HARBOR_CACHE_DIR / dataset
    if not dataset_dir.is_dir():
        click.echo(f"Downloading harbor dataset {dataset} to {dataset_dir} …")
        import subprocess
        subprocess.run(
            ["uvx", "harbor", "datasets", "download", dataset, "-o", str(dataset_dir)],
            check=True,
        )
    tasks = load_harbor_tasks(dataset)
    if task_filter:
        tasks = [t for t in tasks if task_filter in t.task_name]
    if task_limit:
        tasks = tasks[:task_limit]
    if not tasks:
        raise click.UsageError(f"No tasks matched in {dataset_dir} (filter={task_filter!r}).")

    sandbox_factory = None
    if sandbox == "docker":
        from rlcli.sandbox_docker import local_docker_sandbox_factory

        sandbox_factory = local_docker_sandbox_factory

    renderer_name = _renderer_for(model_name, renderer_name)
    dataset_builder = HarborDatasetBuilder(
        tasks=tasks,
        batch_size=groups_per_batch,
        group_size=group_size,
        model_name=model_name,
        renderer_name=renderer_name,
        max_turns=max_turns,
        sandbox_timeout=sandbox_timeout,
        command_timeout=command_timeout,
        sandbox_factory=sandbox_factory,
    )
    kwargs = dict(
        learning_rate=learning_rate,
        dataset_builder=dataset_builder,
        model_name=model_name,
        recipe_name="rlcli_train_harbor",
        max_tokens=max_tokens,
        log_path=log_path or _default_log_path("harbor"),
        renderer_name=renderer_name,
        lora_rank=lora_rank,
        save_every=save_every,
        eval_every=eval_every,
        base_url=base_url,
        loss_fn=as_loss_fn(loss),
        loss_fn_config=json.loads(loss_config) if loss_config else None,
    )
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    config = rl_train.Config(**kwargs)
    if dry_run:
        click.echo(
            f"[dry-run] harbor config OK: {len(tasks)} tasks, sandbox={sandbox}, "
            f"loss_fn={loss} backend={backend or 'unknown'}"
        )
        return
    click.echo(
        f"Harbor RL: {model_name}, {len(tasks)} tasks from {dataset}, "
        f"sandbox={sandbox}, loss_fn={loss}, via {base_url}"
    )
    _run(rl_train.main(config))
