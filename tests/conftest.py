"""Shared pytest configuration.

The ``docker`` marker is declared in pyproject.toml as "needs a local Docker
daemon", but nothing acted on it, so those tests ran anyway on a machine without
one and failed with a connection error that looks like a real regression. Skip
them instead, and say why in the skip reason.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        # `info` (unlike `version`) round-trips to the daemon, which is what
        # these tests actually need.
        return subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def pytest_collection_modifyitems(config, items):
    if not any(item.get_closest_marker("docker") for item in items):
        return
    if _docker_available():
        return
    skip = pytest.mark.skip(reason="needs a local Docker daemon (none reachable)")
    for item in items:
        if item.get_closest_marker("docker"):
            item.add_marker(skip)
