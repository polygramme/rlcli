"""LocalDockerSandbox integration tests — run against the real Docker daemon.

Skipped when Docker isn't available so unit CI stays green everywhere; a CI
job with Docker exercises the full lifecycle.
"""

import asyncio
import shutil
import subprocess

import pytest

pytest.importorskip("tinker_cookbook")

from tinker_cookbook.sandbox.sandbox_interface import SandboxInterface  # noqa: E402

from rlcli.sandbox_docker import local_docker_sandbox_factory  # noqa: E402


def _docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


docker = pytest.mark.skipif(not _docker_available(), reason="docker daemon not available")


@pytest.fixture()
def task_env_dir(tmp_path):
    env = tmp_path / "environment"
    env.mkdir()
    (env / "Dockerfile").write_text("FROM busybox:latest\nWORKDIR /root\n")
    return env


@docker
def test_sandbox_full_lifecycle(task_env_dir):
    async def scenario():
        sandbox = await local_docker_sandbox_factory(task_env_dir, timeout=300)
        try:
            assert isinstance(sandbox, SandboxInterface)  # runtime_checkable protocol

            result = await sandbox.run_command("echo hello && echo err >&2")
            assert result.exit_code == 0
            assert result.stdout.strip() == "hello"
            assert result.stderr.strip() == "err"

            result = await sandbox.run_command("pwd", workdir="/tmp")
            assert result.stdout.strip() == "/tmp"

            result = await sandbox.run_command("exit 3")
            assert result.exit_code == 3

            # write_file creates parents, sets exec bit; read_file round-trips —
            # the exact path shape HarborReward uses (/tests/test.sh from /root).
            w = await sandbox.write_file("/tests/test.sh", "#!/bin/sh\necho PASS\n",
                                         executable=True)
            assert w.exit_code == 0
            result = await sandbox.run_command("sh /tests/test.sh", workdir="/root")
            assert result.exit_code == 0 and "PASS" in result.stdout
            r = await sandbox.read_file("/tests/test.sh", max_bytes=9)
            assert r.stdout == "#!/bin/sh"

            result = await sandbox.run_command("yes x | head -c 4096", max_output_bytes=100)
            assert "[truncated:" in result.stdout

            await sandbox.send_heartbeat()
        finally:
            await sandbox.cleanup()

        # After cleanup the container is gone.
        gone = subprocess.run(
            ["docker", "inspect", sandbox.sandbox_id], capture_output=True
        )
        assert gone.returncode != 0

    asyncio.run(scenario())


@docker
def test_command_timeout_returns_124(task_env_dir):
    async def scenario():
        sandbox = await local_docker_sandbox_factory(task_env_dir, timeout=120)
        try:
            result = await sandbox.run_command("sleep 30", timeout=2)
            assert result.exit_code == 124
            assert "timed out" in result.stderr
        finally:
            await sandbox.cleanup()

    asyncio.run(scenario())
