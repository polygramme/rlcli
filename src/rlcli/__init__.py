"""rlcli — one-shot CLI for Tinker-API post-training on your own GPUs."""

__version__ = "0.1.0.dev0"

# Server sources. The skyrl on PyPI (0.3.0) predates the Tinker engine, so the
# server installs from a commit pin until PyPI catches up (audit E22).
SKYRL_PIN = "9719b4f74ae9cbb6ec022a8d67c1e8a835b52c7d"
SKYRL_GIT_SOURCE = f"git+https://github.com/NovaSky-AI/SkyRL@{SKYRL_PIN}"
COOKBOOK_PIN = "f46eddde86e5397138917516a6c69d2ecbf538b1"
