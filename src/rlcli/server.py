"""SkyRL Tinker server lifecycle.

The server lives in its own uv-managed venv (~/.rlcli/server-venv) because
skyrl caps tinker<=0.24.1 while rlcli's client env uses tinker 0.25.0 (audit
E9/E19). The two only meet over HTTP. On macOS the JAX extra supplies a CPU
path with no vLLM/Ray (audit E20); Linux GPU boxes use fsdp or megatron.
"""

from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import time
from pathlib import Path

import httpx

from . import SKYRL_GIT_SOURCE

RLCLI_HOME = Path(os.environ.get("RLCLI_HOME", Path.home() / ".rlcli"))
STATE_FILE = RLCLI_HOME / "server.json"
LOG_FILE = RLCLI_HOME / "server.log"
VENV_DIR = RLCLI_HOME / "server-venv"

BACKEND_EXTRAS = {"jax": "jax", "fsdp": "fsdp", "megatron": "megatron"}


class ServerError(RuntimeError):
    pass


def _venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def skyrl_source() -> str:
    return os.environ.get("RLCLI_SKYRL_SOURCE", SKYRL_GIT_SOURCE)


def ensure_installed(backend: str, log=print) -> Path:
    """Create the server venv and install skyrl with the right extras."""
    if backend not in BACKEND_EXTRAS:
        raise ServerError(f"Unknown backend {backend!r}. Known: {', '.join(BACKEND_EXTRAS)}")
    RLCLI_HOME.mkdir(parents=True, exist_ok=True)
    python = _venv_python()
    spec = f"skyrl[tinker,{BACKEND_EXTRAS[backend]}] @ {skyrl_source()}"
    marker = RLCLI_HOME / f"installed-{backend}.json"
    if python.exists() and marker.exists() and json.loads(marker.read_text()).get("spec") == spec:
        return python
    if not python.exists():
        log(f"Creating server venv at {VENV_DIR} …")
        subprocess.run(["uv", "venv", "-p", "3.12", str(VENV_DIR)], check=True)
    log(f"Installing {spec} (first run can take a few minutes) …")
    subprocess.run(
        ["uv", "pip", "install", spec],
        check=True,
        env={**os.environ, "VIRTUAL_ENV": str(VENV_DIR)},
    )
    marker.write_text(json.dumps({"spec": spec, "time": time.time()}))
    return python


def read_state() -> dict | None:
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text())
    except json.JSONDecodeError:
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def status() -> dict | None:
    """Return state for a live managed server, cleaning up stale files."""
    state = read_state()
    if state and _pid_alive(state.get("pid", -1)):
        return state
    if state:
        STATE_FILE.unlink(missing_ok=True)
    return None


def health(port: int, timeout: float = 2.0) -> bool:
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/docs", timeout=timeout)
        return r.status_code < 500
    except httpx.HTTPError:
        return False


def start(
    base_model: str,
    backend: str,
    port: int = 8000,
    checkpoints_base: str | None = None,
    backend_config: dict | None = None,
    wait_seconds: int = 900,
    log=print,
) -> dict:
    if status() is not None:
        raise ServerError("A managed server is already running — `rlcli serve stop` first.")
    python = ensure_installed(backend, log=log)

    cmd = [
        str(python),
        "-m",
        "skyrl.tinker.api",
        "--base-model",
        base_model,
        "--backend",
        backend,
        "--port",
        str(port),
    ]
    if checkpoints_base:
        cmd += ["--checkpoints-base", checkpoints_base]
    if backend_config:
        cmd += ["--backend-config", json.dumps(backend_config)]

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log(f"Starting: {' '.join(cmd)}")
    log(f"Logs: {LOG_FILE}")
    with open(LOG_FILE, "ab") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=logf, start_new_session=True)

    state = {
        "pid": proc.pid,
        "port": port,
        "base_model": base_model,
        "backend": backend,
        "base_url": f"http://localhost:{port}",
        "started": time.time(),
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = tail_log(40)
            raise ServerError(f"Server exited early (code {proc.returncode}). Log tail:\n{tail}")
        if health(port):
            log(f"Server ready at http://localhost:{port} ({backend}, {base_model})")
            return state
        time.sleep(2)
    raise ServerError(f"Server did not become healthy within {wait_seconds}s. See {LOG_FILE}")


def stop(log=print) -> bool:
    state = status()
    if state is None:
        return False
    pid = state["pid"]
    log(f"Stopping server pid {pid} …")
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        if not _pid_alive(pid):
            break
        time.sleep(1)
    else:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    STATE_FILE.unlink(missing_ok=True)
    return True


def tail_log(lines: int = 40) -> str:
    if not LOG_FILE.exists():
        return "(no log file)"
    content = LOG_FILE.read_text(errors="replace").splitlines()
    return "\n".join(content[-lines:])


def backend_for_url(base_url: str) -> str | None:
    """Best-effort backend detection: known only for the server we manage."""
    state = status()
    if state and base_url.rstrip("/") in (state["base_url"], f"http://127.0.0.1:{state['port']}"):
        return state["backend"]
    return None
