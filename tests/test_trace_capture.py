"""rlcli.trace_capture: payload lines + store rows, coordinate tagging, teardown."""
import asyncio
import json
import os

import pytest

from rlcli import trace_capture as tc

pytest.importorskip("tinker_cookbook.capture")


class FakeStore:
    def __init__(self, fail=False):
        self.inserts = []
        self.fail = fail

    def insert(self, table, rows, **kw):
        if self.fail:
            raise RuntimeError("db down")
        self.inserts.append((table, list(rows), kw))
        return []


def sample_record(iteration=3, group_idx=1, task="t1", split="train", n=2, extra=None):
    scope = {"run_id": "run_x", "iteration": iteration, "group_idx": group_idx, "split": split}
    if task:
        scope["task"] = task
    scope.update(extra or {})
    return {
        "kind": "sample",
        "scope": scope,
        "seq_id": "seq-1",
        "sampling_session_id": "sess-1",
        "model_path": "tinker://m/sampler_weights/000010",
        "created_at": "2026-09-04T00:00:00Z",
        "prompt_tokens": [1, 2, 3],
        "samples": [
            {"tokens": [4, 5], "logprobs": [-0.1, -0.2], "stop_reason": "stop"}
            for _ in range(n)
        ],
    }


class Builder:
    def __init__(self, task_name=None, task_dir=None):
        if task_name:
            self.task = type("T", (), {"task_name": task_name, "task_dir": task_dir})()


def test_sink_writes_payload_lines_and_rows(tmp_path):
    store = FakeStore()
    sink = tc.TraceSink(store, "run_x", str(tmp_path / "traces"))
    sink.export([sample_record()])
    (table, rows, kw), = store.inserts
    assert table == "traces" and kw == {"returning": False}
    assert [r["storage_path"] for r in rows] == ["traces/it00003.jsonl:0", "traces/it00003.jsonl:1"]
    r = rows[0]
    assert r["run_id"] == "run_x" and r["step"] == 3 and r["phase"] == "train"
    assert r["sample_idx"] == 0 and r["tokens"] == 5
    assert r["meta"]["group_idx"] == 1 and r["meta"]["task"] == "t1"
    assert r["meta"]["stop_reason"] == "stop" and r["meta"]["policy_version"].endswith("000010")
    lines = (tmp_path / "traces" / "it00003.jsonl").read_text().splitlines()
    assert len(lines) == 2
    p = json.loads(lines[1])
    assert p["prompt_tokens"] == [1, 2, 3] and p["sampled_tokens"] == [4, 5] and p["logprobs"] == [-0.1, -0.2]
    assert p["metadata"]["scope"] == {"task": "t1"}  # non-reserved scope keys survive
    assert sink.rows == 2 and sink.skipped == 0


def test_sink_skips_train_ops_and_works_without_store(tmp_path):
    sink = tc.TraceSink(None, "run_x", str(tmp_path))
    sink.export([{"kind": "forward_backward", "scope": {}}, sample_record(n=1)])
    assert sink.skipped == 1 and sink.rows == 1
    assert (tmp_path / "it00003.jsonl").exists()


def test_sink_continues_line_numbers_after_resume(tmp_path):
    path = tmp_path / "it00003.jsonl"
    path.write_text('{"a":1}\n{"a":2}\n')
    sink = tc.TraceSink(FakeStore(), "run_x", str(tmp_path))
    sink.export([sample_record(n=1)])
    assert sink.store.inserts[0][1][0]["storage_path"] == "traces/it00003.jsonl:2"


def test_sink_untagged_and_eval_phase(tmp_path):
    store = FakeStore()
    sink = tc.TraceSink(store, "run_x", str(tmp_path))
    rec = sample_record(iteration=None, split="eval/heldout", n=1)
    sink.export([rec])
    row = store.inserts[0][1][0]
    assert row["step"] is None and row["phase"] == "eval"
    assert row["storage_path"] == "traces/untagged.jsonl:0"


def test_sink_store_failure_raises_after_payload_written(tmp_path):
    sink = tc.TraceSink(FakeStore(fail=True), "run_x", str(tmp_path))
    with pytest.raises(RuntimeError):
        sink.export([sample_record(n=1)])
    assert sink.store_failures == 1 and (tmp_path / "it00003.jsonl").exists()


def test_read_payloads_round_trip_and_rejects_bad_paths(tmp_path):
    rdir = tmp_path / "run"
    store = FakeStore()
    sink = tc.TraceSink(store, "run_x", str(rdir / "traces"))
    sink.export([sample_record(n=2)])
    rows = [dict(r) for r in store.inserts[0][1]]
    rows.append({"storage_path": "traces/../../etc/passwd:0"})
    rows.append({"storage_path": "traces/it09999.jsonl:0"})
    rows.append({"storage_path": None})
    out = tc.read_payloads(str(rdir), rows)
    assert out[0]["payload"]["sampled_tokens"] == [4, 5]
    assert out[1]["payload"]["prompt_tokens"] == [1, 2, 3]
    assert out[2]["payload"] is None and out[3]["payload"] is None and out[4]["payload"] is None


def test_stamping_dataset_tags_builders():
    class Inner:
        def __init__(self):
            self.batches = {2: [Builder("alpha"), Builder()]}

        def get_batch(self, i):
            return self.batches[i]

        def __len__(self):
            return 7

    ds = tc.StampingDataset(Inner())
    a, b = ds.get_batch(2)
    assert getattr(a, tc.COORDS_ATTR) == {"iteration": 2, "group_idx": 0, "task": "alpha"}  # no task_dir -> no hash
    assert getattr(b, tc.COORDS_ATTR) == {"iteration": 2, "group_idx": 1}
    assert len(ds) == 7 and ds.batches  # delegation


def test_install_rollout_tags_enters_scope_and_restores(monkeypatch):
    from tinker_cookbook.capture.scope import current_scope
    from tinker_cookbook.rl import rollouts

    seen = []

    async def fake(builder, policy):
        seen.append(dict(current_scope()))
        return "group"

    monkeypatch.setattr(rollouts, "do_group_rollout", fake)
    restore = tc.install_rollout_tags()
    tagged, plain = Builder("alpha"), Builder()
    tc.stamp_env_group_builders([tagged], 5)
    assert asyncio.run(rollouts.do_group_rollout(tagged, None)) == "group"
    assert asyncio.run(rollouts.do_group_rollout(plain, None)) == "group"
    assert seen[0]["iteration"] == 5 and seen[0]["group_idx"] == 0 and seen[0]["task"] == "alpha"
    assert "iteration" not in seen[1]
    assert tc.install_rollout_tags()() is None  # idempotent: second install is a no-op
    restore()
    assert rollouts.do_group_rollout is fake


def test_capture_run_flushes_and_tears_down(tmp_path):
    from tinker_cookbook.capture.instrument import current_exporter
    from tinker_cookbook.rl import rollouts

    before = rollouts.do_group_rollout
    store = FakeStore()
    logs = []
    with tc.capture_run("run_x", store=store, payload_dir=str(tmp_path / "traces"), log=logs.append) as cap:
        assert current_exporter() is cap.exporter
        assert rollouts.do_group_rollout is not before
        cap.exporter.enqueue(sample_record(n=1))
    assert current_exporter() is None
    assert rollouts.do_group_rollout is before
    assert cap.stats()["rows"] == 1 and cap.stats()["dropped"] == 0
    assert store.inserts and os.path.exists(tmp_path / "traces" / "it00003.jsonl")
    assert logs and logs[-1].startswith("trace capture: ")


def test_install_eval_tags_enters_eval_scope_and_restores(monkeypatch):
    from tinker_cookbook.capture.scope import current_scope
    from tinker_cookbook.rl import train as rl_train

    seen = []

    async def fake(evaluator, config, i_batch, sampling_client, evaluator_label, store=None):
        seen.append(dict(current_scope()))
        return {"test/reward": 1.0}

    monkeypatch.setattr(rl_train, "run_single_evaluation", fake)
    restore = tc.install_eval_tags()
    out = asyncio.run(rl_train.run_single_evaluation(None, None, 7, None, "test", store=None))
    assert out == {"test/reward": 1.0}
    assert seen[0]["split"] == "eval/test" and seen[0]["iteration"] == 7 and seen[0]["purpose"] == "eval"
    assert tc.install_eval_tags()() is None
    restore()
    assert rl_train.run_single_evaluation is fake


def test_eval_rows_land_as_eval_phase_with_step(tmp_path):
    store = FakeStore()
    sink = tc.TraceSink(store, "run_x", str(tmp_path))
    rec = sample_record(iteration=7, split="eval/test", n=1, extra={"purpose": "eval"})
    sink.export([rec])
    row = store.inserts[0][1][0]
    assert row["phase"] == "eval" and row["step"] == 7 and row["storage_path"] == "traces/it00007.jsonl:0"


def test_stamp_includes_task_hash_and_sink_writes_it(tmp_path):
    d = tmp_path / "task"
    d.mkdir()
    (d / "instruction.md").write_text("do it")
    from rlcli.harbor_tasks import task_content_hash

    b = Builder("alpha", task_dir=d)
    tc.stamp_env_group_builders([b], 1)
    coords = getattr(b, tc.COORDS_ATTR)
    assert coords["task_hash"] == task_content_hash(d)
    store = FakeStore()
    sink = tc.TraceSink(store, "run_x", str(tmp_path / "traces"))
    sink.export([sample_record(n=1, extra={"task_hash": coords["task_hash"]})])
    assert store.inserts[0][1][0]["task_hash"] == coords["task_hash"]


def test_sink_retries_without_task_hash_when_the_store_rejects_it(tmp_path):
    class FKStore(FakeStore):
        def insert(self, table, rows, **kw):
            rows = list(rows)
            if any(r.get("task_hash") for r in rows):
                raise RuntimeError('violates foreign key constraint "traces_task_hash_fkey"')
            return super().insert(table, rows, **kw)

    store = FKStore()
    sink = tc.TraceSink(store, "run_x", str(tmp_path))
    sink.export([sample_record(n=2, extra={"task_hash": "deadbeef"})])
    assert sink.task_hash_fallbacks == 1 and sink.store_failures == 0
    assert [r["task_hash"] for r in store.inserts[0][1]] == [None, None]
    assert sink.rows == 2
