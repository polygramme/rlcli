import json

import pytest
from click.testing import CliRunner

from rlcli.cli import main_cli

pytest.importorskip("tinker_cookbook")


@pytest.fixture()
def harbor_cache(tmp_path, monkeypatch):
    """Synthetic harbor dataset under a patched HARBOR_CACHE_DIR."""
    from tinker_cookbook.recipes.harbor_rl import harbor_env

    task = tmp_path / "mytasks" / "hello-task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "environment" / "Dockerfile").write_text("FROM busybox:latest\n")
    (task / "instruction.md").write_text("Create /root/hello.txt containing 'hi'.")
    (task / "tests" / "test.sh").write_text("#!/bin/sh\ngrep -q hi /root/hello.txt\n")
    (task / "task.toml").write_text('[metadata]\nname = "hello-task"\n')
    monkeypatch.setattr(harbor_env, "HARBOR_CACHE_DIR", tmp_path)
    return tmp_path


def test_harbor_dry_run_builds_config(harbor_cache):
    result = CliRunner().invoke(main_cli, [
        "train", "harbor", "--model", "Qwen/Qwen3-4B-Instruct-2507",
        "--dataset", "mytasks", "--loss", "gspo", "--backend", "fsdp", "--dry-run",
    ])
    assert result.exit_code == 0, result.output or result.exception
    assert "1 tasks" in result.output
    assert "sandbox=docker" in result.output
    assert "loss_fn=gspo" in result.output


def test_harbor_guard_blocks_gspo_on_jax(harbor_cache):
    result = CliRunner().invoke(main_cli, [
        "train", "harbor", "--model", "Qwen/Qwen3-4B-Instruct-2507",
        "--dataset", "mytasks", "--loss", "gspo", "--backend", "jax", "--dry-run",
    ])
    assert result.exit_code != 0
    assert "fsdp" in str(result.output) + str(result.exception)


def test_harbor_task_filter_no_match_errors(harbor_cache):
    result = CliRunner().invoke(main_cli, [
        "train", "harbor", "--model", "Qwen/Qwen3-4B-Instruct-2507",
        "--dataset", "mytasks", "--task-filter", "nope", "--backend", "fsdp", "--dry-run",
    ])
    assert result.exit_code != 0
    assert "No tasks matched" in result.output


def test_harbor_loss_config_passthrough(harbor_cache):
    result = CliRunner().invoke(main_cli, [
        "train", "harbor", "--model", "Qwen/Qwen3-4B-Instruct-2507",
        "--dataset", "mytasks", "--loss", "gspo", "--backend", "megatron",
        "--loss-config", json.dumps({"clip_low_threshold": 0.8}), "--dry-run",
    ])
    assert result.exit_code == 0, result.output or result.exception
