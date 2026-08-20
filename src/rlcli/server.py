"""SkyRL Tinker server lifecycle.

The server runs out of a pinned skyrl source checkout via `uv run` — not a
plain venv — because skyrl's API process relaunches its background engine
with `uv run <parent flags> --extra tinker --extra <backend> -m
skyrl.tinker.engine`, and its startup parser only accepts servers launched as
`uv run ... -m skyrl.tinker.api` from the project (skyrl/tinker/api.py:184).
uv manages the project env from skyrl's own lockfile, which also isolates the
server's tinker<=0.24.1 cap from rlcli's tinker 0.25.0 client env (audit E9).

On macOS the JAX extra gives a CPU path with no vLLM/Ray (audit E20); Linux
GPU boxes use the fsdp or megatron extras.
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

from . import SKYRL_PIN

SKYRL_REPO_URL = "https://github.com/NovaSky-AI/SkyRL"

RLCLI_HOME = Path(os.environ.get("RLCLI_HOME", Path.home() / ".rlcli"))
STATE_FILE = RLCLI_HOME / "server.json"
LOG_FILE = RLCLI_HOME / "server.log"
CHECKOUT_DIR = RLCLI_HOME / "skyrl-src"

BACKEND_EXTRAS = {"jax": "jax", "fsdp": "fsdp", "megatron": "megatron"}


class ServerError(RuntimeError):
    pass


def _env() -> dict:
    # Don't leak rlcli's own venv into uv's project resolution.
    env = {**os.environ}
    env.pop("VIRTUAL_ENV", None)
    return env


def skyrl_checkout(log=print) -> Path:
    """Return a skyrl source tree: RLCLI_SKYRL_SOURCE if set, else a pinned clone."""
    override = os.environ.get("RLCLI_SKYRL_SOURCE")
    if override:
        path = Path(override.removeprefix("file://"))
        if not (path / "pyproject.toml").exists():
            raise ServerError(f"RLCLI_SKYRL_SOURCE={override} is not a skyrl checkout")
        return path
    if not (CHECKOUT_DIR / "pyproject.toml").exists():
        RLCLI_HOME.mkdir(parents=True, exist_ok=True)
        log(f"Cloning skyrl @ {SKYRL_PIN[:12]} into {CHECKOUT_DIR} …")
        subprocess.run(
            ["git", "clone", "--filter=blob:none", SKYRL_REPO_URL, str(CHECKOUT_DIR)],
            check=True,
        )
        subprocess.run(["git", "-C", str(CHECKOUT_DIR), "checkout", "-q", SKYRL_PIN], check=True)
    return CHECKOUT_DIR


def ensure_installed(backend: str, log=print) -> Path:
    """Sync the skyrl project env for the backend's extras; returns checkout dir."""
    if backend not in BACKEND_EXTRAS:
        raise ServerError(f"Unknown backend {backend!r}. Known: {', '.join(BACKEND_EXTRAS)}")
    checkout = skyrl_checkout(log=log)
    log(f"Syncing skyrl env in {checkout} (first run can take a few minutes) …")
    subprocess.run(
        ["uv", "sync", "--extra", "tinker", "--extra", BACKEND_EXTRAS[backend]],
        check=True,
        cwd=checkout,
        env=_env(),
    )
    if backend == "jax" and platform.system() == "Darwin":
        # skyrl's [tool.uv] override-dependencies marks ml-dtypes and
        # transformers as sys_platform=='linux' only, but the JAX engine
        # imports both on every platform — broken on macOS as locked. uv
        # honors the override even for `uv pip install` run inside the
        # project, so install with an explicit --python from outside the
        # checkout, and launch with --no-sync so `uv run` doesn't strip them
        # back out.
        RLCLI_HOME.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "uv", "pip", "install", "-q",
                "--python", str(checkout / ".venv" / "bin" / "python"),
                "ml_dtypes", "transformers>=5.6.1,<=5.8.0",
            ],
            check=True,
            cwd=RLCLI_HOME,
            env=_env(),
        )
    return checkout


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
    checkout = ensure_installed(backend, log=log)

    # Must match skyrl's accepted startup form: uv run <flags> -m skyrl.tinker.api.
    # --no-sync is inherited by the engine relaunch and keeps our ml_dtypes fix.
    cmd = [
        "uv",
        "run",
        "--no-sync",
        "--extra",
        "tinker",
        "--extra",
        BACKEND_EXTRAS[backend],
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
        proc = subprocess.Popen(
            cmd, stdout=logf, stderr=logf, start_new_session=True, cwd=checkout, env=_env()
        )

    state = {
        "pid": proc.pid,
        "port": port,
        "base_model": base_model,
        "backend": backend,
        "base_url": f"http://localhost:{port}",
        "checkout": str(checkout),
        "started": time.time(),
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            STATE_FILE.unlink(missing_ok=True)
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
