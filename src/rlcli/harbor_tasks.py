"""Content identity for Harbor task directories.

A task is the files in its directory (instruction.md, task.toml,
environment/Dockerfile, tests/test.sh, anything else the Dockerfile copies).
Its content hash is a SHA-256 over the sorted relative paths and bytes of
those files, so the same task pushed twice — or from two machines — is one
task, and any edit is a new one. Hidden files and caches are ignored;
mtimes and ownership never enter the hash.

Torch-free; used by the hosted control plane at push time and by the runs
driver to tag captured trajectories with the task they came from.
"""
from __future__ import annotations

import hashlib
import os
import tomllib
from pathlib import Path
from typing import Iterator

IGNORED_DIRS = frozenset({"__pycache__", ".git", ".hg", ".svn", "node_modules"})
IGNORED_FILES = frozenset({".DS_Store"})


def iter_task_files(task_dir: str | os.PathLike) -> Iterator[tuple[str, Path]]:
    """(relative posix path, absolute path) for every file, deterministic order."""
    base = Path(task_dir)
    found: list[tuple[str, Path]] = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]
        for name in files:
            if name in IGNORED_FILES or name.startswith("."):
                continue
            p = Path(root) / name
            if p.is_symlink():
                continue
            found.append((p.relative_to(base).as_posix(), p))
    # Globally sorted by relative path (os.walk yields a directory's files
    # before its subdirectories, which is neither obvious nor stable to read).
    yield from sorted(found)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def task_content_hash(task_dir: str | os.PathLike) -> str:
    h = hashlib.sha256()
    for rel, p in iter_task_files(task_dir):
        data = p.read_bytes()
        for part in (rel.encode(), str(len(data)).encode(), data):
            h.update(part)
            h.update(b"\0")
    return h.hexdigest()


def task_manifest(task_dir: str | os.PathLike, *, task_name: str | None = None) -> dict:
    """One `tasks` row: content_hash, input, env_params, grader_params, tags, description."""
    base = Path(task_dir)
    name = task_name or base.name
    files = []
    for rel, p in iter_task_files(base):
        data = p.read_bytes()
        files.append({"path": rel, "bytes": len(data), "sha256": _sha256(data)})
    instruction = (base / "instruction.md").read_text() if (base / "instruction.md").exists() else ""
    config: dict = {}
    if (base / "task.toml").exists():
        try:
            config = tomllib.loads((base / "task.toml").read_text())
        except Exception:
            config = {"_parse_error": True}
    has_test = (base / "tests" / "test.sh").exists()
    return {
        "content_hash": task_content_hash(base),
        "input": [{"role": "user", "content": instruction}],
        "env_params": {"files": files, "config": config},
        "grader_params": {"test": "tests/test.sh"} if has_test else None,
        "tags": {"task_name": name},
        "description": name,
    }
