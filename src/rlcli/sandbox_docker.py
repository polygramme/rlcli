"""Local Docker backend for tinker-cookbook's SandboxInterface.

The cookbook ships only Modal and SandboxFusion sandbox backends — both need
cloud accounts. This one runs Harbor task environments on the local Docker
daemon, injected through harbor_rl's ``sandbox_factory`` parameter (which
receives the task's environment/ dir and a lifetime in seconds).

Containers are kept alive with a bounded sleep equal to the sandbox timeout,
so an unclean shutdown can't leak containers forever; ``cleanup`` removes
them immediately. Commands run via ``docker exec`` so the image's own
filesystem, users, and WORKDIR semantics apply.
"""

from __future__ import annotations

import asyncio
import hashlib
import shlex
import uuid
from pathlib import Path

from tinker_cookbook.sandbox.sandbox_interface import (
    SandboxInterface,
    SandboxResult,
    SandboxTerminatedError,
)

DEFAULT_MAX_OUTPUT_BYTES = 128 * 1024
_IMAGE_PREFIX = "rlcli-harbor"
_CONTAINER_PREFIX = "rlcli-sb"
_BUILD_TIMEOUT = 1800


async def _run(
    *argv: str,
    stdin: bytes | None = None,
    timeout: float | None = None,
) -> tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(input=stdin), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, b"", f"timed out after {timeout}s: {' '.join(argv[:6])} ...".encode()
    return proc.returncode or 0, stdout, stderr


def _truncate(data: bytes, max_bytes: int | None) -> str:
    limit = DEFAULT_MAX_OUTPUT_BYTES if max_bytes is None else max_bytes
    text = data[:limit].decode("utf-8", errors="replace")
    if len(data) > limit:
        text += f"\n[truncated: {len(data) - limit} bytes dropped]"
    return text


class LocalDockerSandbox:
    """SandboxInterface implementation backed by ``docker exec``."""

    def __init__(self, container_name: str):
        self._container = container_name
        self._terminated = False

    @property
    def sandbox_id(self) -> str:
        return self._container

    async def send_heartbeat(self, timeout: int = 30) -> None:
        code, _, _ = await _run(
            "docker", "inspect", "-f", "{{.State.Running}}", self._container, timeout=timeout
        )
        if code != 0:
            self._terminated = True
            raise SandboxTerminatedError(f"container {self._container} is gone")

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        if self._terminated:
            raise SandboxTerminatedError(f"container {self._container} was cleaned up")
        argv = ["docker", "exec"]
        if workdir is not None:
            argv += ["-w", workdir]
        argv += [self._container, "sh", "-lc", command]
        code, out, err = await _run(*argv, timeout=timeout)
        if code == 126 or (code == 1 and b"No such container" in err):
            raise SandboxTerminatedError(f"container {self._container} is gone: {err.decode()!r}")
        return SandboxResult(
            stdout=_truncate(out, max_output_bytes),
            stderr=_truncate(err, max_output_bytes),
            exit_code=code,
        )

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        cmd = f"head -c {max_bytes} {shlex.quote(path)}" if max_bytes else f"cat {shlex.quote(path)}"
        return await self.run_command(cmd, workdir="/", timeout=timeout, max_output_bytes=max_bytes)

    async def write_file(
        self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60
    ) -> SandboxResult:
        data = content.encode() if isinstance(content, str) else content
        quoted = shlex.quote(path)
        parent = shlex.quote(str(Path(path).parent))
        mode = " && chmod +x " + quoted if executable else ""
        argv = [
            "docker", "exec", "-i", self._container,
            "sh", "-c", f"mkdir -p {parent} && cat > {quoted}{mode}",
        ]
        code, out, err = await _run(*argv, stdin=data, timeout=timeout)
        return SandboxResult(
            stdout=_truncate(out, None), stderr=_truncate(err, None), exit_code=code
        )

    async def cleanup(self) -> None:
        self._terminated = True
        await _run("docker", "rm", "-f", self._container, timeout=60)


async def _ensure_image(env_dir: Path) -> str:
    """Build (or reuse) the task image. Tag is content-addressed by env_dir
    path so repeated rollouts of the same task hit Docker's cache."""
    dockerfile = env_dir / "Dockerfile"
    if not dockerfile.exists():
        raise FileNotFoundError(f"No Dockerfile at {dockerfile}")
    tag = f"{_IMAGE_PREFIX}:{hashlib.sha1(str(env_dir.resolve()).encode()).hexdigest()[:12]}"
    code, _, _ = await _run("docker", "image", "inspect", tag, timeout=30)
    if code != 0:
        code, _, err = await _run(
            "docker", "build", "-q", "-t", tag, "-f", str(dockerfile), str(env_dir),
            timeout=_BUILD_TIMEOUT,
        )
        if code != 0:
            raise RuntimeError(f"docker build failed for {env_dir}: {err.decode()[-2000:]}")
    return tag


async def local_docker_sandbox_factory(env_dir: Path, timeout: int) -> SandboxInterface:
    """SandboxFactory for harbor_rl: (task environment dir, lifetime seconds)."""
    tag = await _ensure_image(env_dir)
    name = f"{_CONTAINER_PREFIX}-{uuid.uuid4().hex[:10]}"
    # Bounded keep-alive: the container self-destructs after `timeout` seconds
    # even if cleanup never runs.
    code, _, err = await _run(
        "docker", "run", "-d", "--name", name,
        "--entrypoint", "sh", tag, "-c", f"sleep {int(timeout)}",
        timeout=120,
    )
    if code != 0:
        raise RuntimeError(f"docker run failed for {tag}: {err.decode()[-2000:]}")
    return LocalDockerSandbox(name)
