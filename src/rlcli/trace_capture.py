"""Token-level trace capture for hosted runs.

Wires the cookbook's ``capture`` instrumentation (tinker_cookbook/capture) into
a run without its local SQLite daemon: every sampled sequence — prompt tokens,
sampled tokens, logprobs, run coordinates — is appended as one JSON line under
the run's Volume directory and summarised as one ``traces`` row in the store
(any object with an ``insert(table, rows, returning=False)`` method; Polygramme
Cloud passes its Postgres client). The Volume line is the payload; the row is what you filter on.
Both survive the container; neither is on the training path (the exporter is
a bounded background queue that drops rather than blocks).

The cookbook loop never enters a ``capture(...)`` scope of its own, so rows
would land with no iteration. ``StampingDataset`` tags each batch's builders
with ``iteration``/``group_idx``/``task`` from ``get_batch(index)`` and
``install_rollout_tags`` rebinds ``rollouts.do_group_rollout`` — the override
point the cookbook documents — to enter that scope around every group, so the
samples made inside inherit it (scope is snapshotted at SDK call time).
"""
from __future__ import annotations

import contextlib
import json
import os
import threading
from typing import Any, Callable, Iterable, Mapping, Sequence

COORDS_ATTR = "_pg_trace_coords"
PAYLOAD_KEYS = ("prompt_tokens", "sampled_tokens", "logprobs")
PAYLOAD_SUBDIR = "traces"
UNTAGGED_FILE = "untagged.jsonl"


# ---- coordinates -----------------------------------------------------------


def stamp_env_group_builders(builders: Iterable[Any], iteration: int) -> None:
    """Attach batch coordinates to each builder for install_rollout_tags."""
    for group_idx, builder in enumerate(builders):
        coords: dict[str, Any] = {"iteration": int(iteration), "group_idx": group_idx}
        name = getattr(getattr(builder, "task", None), "task_name", None)
        if name:
            coords["task"] = str(name)
        try:
            setattr(builder, COORDS_ATTR, coords)
        except AttributeError:
            # A frozen builder just leaves its rows untagged; never fails a run.
            pass


class StampingDataset:
    """RLDataset proxy whose get_batch stamps coordinates onto the builders."""

    def __init__(self, inner: Any):
        self._inner = inner

    def get_batch(self, index: int) -> Sequence[Any]:
        builders = self._inner.get_batch(index)
        stamp_env_group_builders(builders, index)
        return builders

    def __len__(self) -> int:
        return len(self._inner)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def install_rollout_tags() -> Callable[[], None]:
    """Rebind rollouts.do_group_rollout to enter the stamped scope. Returns restore()."""
    from tinker_cookbook.capture import capture
    from tinker_cookbook.rl import rollouts

    original = rollouts.do_group_rollout
    if getattr(original, "_pg_trace_tagged", False):
        return lambda: None

    async def do_group_rollout(env_group_builder, policy, *args, **kwargs):
        coords = getattr(env_group_builder, COORDS_ATTR, None)
        if not coords:
            return await original(env_group_builder, policy, *args, **kwargs)
        with capture(**coords):
            return await original(env_group_builder, policy, *args, **kwargs)

    do_group_rollout._pg_trace_tagged = True  # type: ignore[attr-defined]
    do_group_rollout.__wrapped__ = original  # type: ignore[attr-defined]
    rollouts.do_group_rollout = do_group_rollout

    def restore() -> None:
        if rollouts.do_group_rollout is do_group_rollout:
            rollouts.do_group_rollout = original

    return restore


# ---- sink ------------------------------------------------------------------


def _phase(row: Mapping[str, Any]) -> str:
    split = str(row.get("split") or "")
    if split.startswith("eval") or row.get("purpose") == "eval":
        return "eval"
    return "train"


def _count_lines(path: str) -> int:
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


class TraceSink:
    """CaptureSink: payload lines on disk, one ``traces`` row per sequence.

    ``store`` may be None (payload only). Line numbers continue from what is
    already on disk so a resumed attempt never re-uses a storage_path.
    """

    def __init__(self, store: Any, run_id: str, payload_dir: str):
        self.store = store
        self.run_id = run_id
        self.payload_dir = payload_dir
        os.makedirs(payload_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._next_line: dict[str, int] = {}
        self.rows = 0
        self.skipped = 0
        self.store_failures = 0

    def export(self, records: Sequence[Mapping[str, Any]], timeout: float | None = None) -> None:
        from tinker_cookbook.capture.store.client import wire_rows_from_sample_record

        rows = []
        for rec in records:
            if rec.get("kind") != "sample":
                self.skipped += 1  # forward_backward / optim_step records
                continue
            for wire in wire_rows_from_sample_record(dict(rec)):
                rows.append(self._persist(wire))
        if not rows:
            return
        self.rows += len(rows)
        if self.store is not None:
            try:
                self.store.insert("traces", rows, returning=False)
            except Exception:
                # Payload lines are already on disk; the exporter counts and
                # logs the failed batch, training never sees it.
                self.store_failures += 1
                raise

    def _persist(self, wire: Mapping[str, Any]) -> dict:
        it = wire.get("iteration")
        tagged = isinstance(it, int) and not isinstance(it, bool)
        fname = f"it{it:05d}.jsonl" if tagged else UNTAGGED_FILE
        path = os.path.join(self.payload_dir, fname)
        payload = {k: wire.get(k) for k in PAYLOAD_KEYS}
        payload["metadata"] = dict(wire.get("metadata") or {})
        line = json.dumps(payload, separators=(",", ":"))
        with self._lock:
            n = self._next_line.get(fname)
            if n is None:
                n = _count_lines(path)
            with open(path, "a") as f:
                f.write(line + "\n")
            self._next_line[fname] = n + 1
        scope = payload["metadata"].get("scope") or {}
        meta = {
            "group_idx": wire.get("group_idx"),
            "traj_idx": wire.get("traj_idx"),
            "policy_version": wire.get("policy_version"),
            "seq_id": wire.get("seq_id"),
            "sampling_session_id": wire.get("sampling_session_id"),
            "stop_reason": payload["metadata"].get("stop_reason"),
            "task": scope.get("task"),
            "split": wire.get("split"),
        }
        return {
            "run_id": self.run_id,
            "step": it if tagged else None,
            "phase": _phase(wire),
            "sample_idx": wire.get("sample_idx"),
            "tokens": len(payload["prompt_tokens"] or []) + len(payload["sampled_tokens"] or []),
            "storage_path": f"{PAYLOAD_SUBDIR}/{fname}:{n}",
            "meta": {k: v for k, v in meta.items() if v is not None},
        }


def read_payloads(rdir: str, rows: Iterable[dict]) -> list[dict]:
    """Attach ``payload`` to each row from its storage_path (None if missing)."""
    rows = list(rows)
    wanted: dict[str, dict[int, list[dict]]] = {}
    for r in rows:
        r["payload"] = None
        fname, _, ln = str(r.get("storage_path") or "").rpartition(":")
        base = fname[len(PAYLOAD_SUBDIR) + 1 :]
        if not fname.startswith(PAYLOAD_SUBDIR + "/") or os.path.basename(base) != base or not ln.isdigit():
            continue
        wanted.setdefault(base, {}).setdefault(int(ln), []).append(r)
    for base, by_line in wanted.items():
        try:
            f = open(os.path.join(rdir, PAYLOAD_SUBDIR, base))
        except OSError:
            continue
        with f:
            for i, line in enumerate(f):
                if i in by_line:
                    try:
                        payload = json.loads(line)
                    except ValueError:
                        payload = None
                    for r in by_line[i]:
                        r["payload"] = payload
    return rows


# ---- session ---------------------------------------------------------------


class CaptureRun:
    def __init__(self, sink: TraceSink, exporter: Any):
        self.sink = sink
        self.exporter = exporter

    def stats(self) -> dict:
        return {
            "rows": self.sink.rows,
            "skipped": self.sink.skipped,
            "store_failures": self.sink.store_failures,
            "dropped": self.exporter.dropped,
            "export_failures": self.exporter.export_failures,
        }


@contextlib.contextmanager
def capture_run(
    run_id: str,
    *,
    store: Any,
    payload_dir: str,
    log: Callable[[str], None] | None = print,
    run_attempt: int = 0,
    drain_timeout_sec: float = 5.0,
    flush_interval_sec: float = 1.0,
):
    """Capture every instrumented SDK call in the block. Mirrors the cookbook's
    capture_to_store teardown order: stop routing new records first, drain
    in-flight futures, flush, then shut down."""
    from tinker_cookbook.capture import CaptureExporter, capture, instrument_tinker, propagate, uninstrument_tinker
    from tinker_cookbook.capture.instrument import current_exporter

    sink = TraceSink(store, run_id, payload_dir)
    exporter = CaptureExporter(sink, flush_interval_sec=flush_interval_sec)
    session = CaptureRun(sink, exporter)
    previous = current_exporter()
    instrument_tinker(exporter)
    threads_were_instrumented = propagate.threads_instrumented()
    propagate.instrument_threads()
    restore_tags = install_rollout_tags()
    try:
        with capture(run_id=run_id, run_attempt=run_attempt):
            yield session
    finally:
        restore_tags()
        if previous is not None:
            instrument_tinker(previous)
        else:
            uninstrument_tinker()
        if not threads_were_instrumented:
            propagate.uninstrument_threads()
        exporter.wait_pending(timeout=drain_timeout_sec)
        exporter.force_flush(timeout=drain_timeout_sec)
        exporter.shutdown()
        if log:
            log(f"trace capture: {session.stats()}")
