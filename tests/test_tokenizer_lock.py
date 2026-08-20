import threading
import time

from rlcli.tokenizer_lock import LockedTokenizer


class RacyTokenizer:
    """Fails like a Rust fast tokenizer if two threads enter concurrently."""

    def __init__(self):
        self._busy = False
        self.calls = 0

    def encode(self, text, add_special_tokens=False):
        if self._busy:
            raise RuntimeError("Already borrowed")
        self._busy = True
        time.sleep(0.001)
        self.calls += 1
        self._busy = False
        return [len(text)]

    @property
    def vocab_size(self):
        return 42


def test_locked_tokenizer_serializes_threads():
    tok = LockedTokenizer(RacyTokenizer())
    errors = []

    def worker():
        try:
            for _ in range(25):
                tok.encode("hello world")
        except RuntimeError as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert tok.calls == 200
    assert tok.vocab_size == 42  # non-callable attributes pass through
