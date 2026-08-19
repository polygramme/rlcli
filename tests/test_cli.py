from click.testing import CliRunner

from rlcli.cli import main_cli


def _run(*args):
    return CliRunner().invoke(main_cli, args)


def test_help_lists_commands():
    result = _run("--help")
    assert result.exit_code == 0
    for cmd in ("serve", "train", "sample", "checkpoint", "run", "version"):
        assert cmd in result.output


def test_version():
    result = _run("version")
    assert result.exit_code == 0
    assert "rlcli" in result.output
    assert "skyrl pin" in result.output


def test_serve_help():
    result = _run("serve", "--help")
    assert result.exit_code == 0
    assert "start" in result.output and "stop" in result.output


def test_train_rl_guard_blocks_gspo_on_jax():
    result = _run("train", "rl", "--model", "Qwen/Qwen3-4B", "--loss", "gspo",
                  "--backend", "jax", "--dry-run")
    assert result.exit_code != 0
    assert "fsdp" in str(result.output) + str(result.exception)


def test_train_rl_guard_allows_gspo_on_fsdp():
    # Guard passes before the cookbook import. With the train extra installed
    # the dry-run builds a real Config; without it, only the import may fail.
    result = _run("train", "rl", "--model", "Qwen/Qwen3-4B-Instruct-2507", "--loss", "gspo",
                  "--backend", "fsdp", "--dry-run")
    if result.exit_code == 0:
        assert "loss_fn=gspo" in result.output
    else:
        assert "tinker_cookbook" in str(result.exception) or "No module" in str(result.exception)


def test_train_unknown_model_gets_clear_renderer_error():
    result = _run("train", "rl", "--model", "some/unknown-model", "--loss", "ppo",
                  "--backend", "fsdp", "--dry-run")
    assert result.exit_code != 0
    assert "--renderer" in result.output


def test_train_sl_requires_dataset():
    result = _run("train", "sl", "--model", "Qwen/Qwen3-4B")
    assert result.exit_code != 0
    assert "--dataset" in result.output
