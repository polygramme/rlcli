"""Thread-safe tokenizer shim for cookbook multi-env rollouts.

The cookbook's message_env renders in threads (_render_in_thread) while all
envs in a group share one HuggingFace fast tokenizer; Rust fast tokenizers
forbid concurrent use ("RuntimeError: Already borrowed"). We register a
factory through the cookbook's own register_tokenizer hook that returns a
lock-wrapped tokenizer — and a fresh instance per get_tokenizer call.
"""

from __future__ import annotations

import threading
from typing import Any


class LockedTokenizer:
    """Proxy serializing every method call on the wrapped tokenizer."""

    def __init__(self, inner: Any):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_lock", threading.RLock())

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if callable(attr):
            lock = self._lock

            def locked(*args: Any, **kwargs: Any) -> Any:
                with lock:
                    return attr(*args, **kwargs)

            return locked
        return attr

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._inner(*args, **kwargs)

    def __len__(self) -> int:
        return len(self._inner)


def register_locked_tokenizer(model_name: str) -> None:
    """Route cookbook get_tokenizer(model_name) through the locked proxy."""
    from tinker_cookbook import tokenizer_utils

    if tokenizer_utils.is_tokenizer_registered(model_name):
        return

    def factory() -> Any:
        from transformers import AutoTokenizer

        return LockedTokenizer(AutoTokenizer.from_pretrained(model_name))

    tokenizer_utils.register_tokenizer(model_name, factory)
