import os
import time

from rlcli.harbor_tasks import iter_task_files, task_content_hash, task_manifest


def _task(tmp_path, name="hello", instruction="Fix it.", test="#!/bin/sh\nexit 0\n"):
    d = tmp_path / name
    (d / "environment").mkdir(parents=True)
    (d / "tests").mkdir()
    (d / "instruction.md").write_text(instruction)
    (d / "task.toml").write_text('[metadata]\nname = "hello"\n')
    (d / "environment" / "Dockerfile").write_text("FROM alpine\n")
    (d / "tests" / "test.sh").write_text(test)
    return d


def test_hash_is_content_only_and_order_independent(tmp_path):
    a = _task(tmp_path / "a")
    b = _task(tmp_path / "b")
    assert task_content_hash(a) == task_content_hash(b)
    # mtimes, hidden files and caches don't count
    os.utime(a / "instruction.md", (time.time() - 10_000, time.time() - 10_000))
    (a / ".DS_Store").write_bytes(b"junk")
    (a / "__pycache__").mkdir()
    (a / "__pycache__" / "x.pyc").write_bytes(b"junk")
    assert task_content_hash(a) == task_content_hash(b)
    # any real edit is a new task
    (a / "tests" / "test.sh").write_text("#!/bin/sh\nexit 1\n")
    assert task_content_hash(a) != task_content_hash(b)
    # renaming a file changes it too (paths are part of identity)
    c = _task(tmp_path / "c")
    (c / "environment" / "Dockerfile").rename(c / "environment" / "Containerfile")
    assert task_content_hash(c) != task_content_hash(b)


def test_iter_task_files_is_sorted_and_relative(tmp_path):
    d = _task(tmp_path)
    (d / "solution").mkdir()
    (d / "solution" / "z.py").write_text("")
    (d / "solution" / "a.py").write_text("")
    rels = [rel for rel, _ in iter_task_files(d)]
    assert rels == sorted(rels)
    assert "solution/a.py" in rels and rels.index("solution/a.py") < rels.index("solution/z.py")
    assert all(not r.startswith("/") for r in rels)


def test_manifest_matches_the_tasks_row_shape(tmp_path):
    d = _task(tmp_path, instruction="Make the tests pass.")
    m = task_manifest(d)
    assert set(m) == {"content_hash", "input", "env_params", "grader_params", "tags", "description"}
    assert m["content_hash"] == task_content_hash(d)
    assert m["input"] == [{"role": "user", "content": "Make the tests pass."}]
    assert m["tags"] == {"task_name": "hello"} and m["description"] == "hello"
    assert m["grader_params"] == {"test": "tests/test.sh"}
    assert m["env_params"]["config"] == {"metadata": {"name": "hello"}}
    paths = [f["path"] for f in m["env_params"]["files"]]
    assert paths == ["environment/Dockerfile", "instruction.md", "task.toml", "tests/test.sh"]
    assert all(len(f["sha256"]) == 64 and f["bytes"] > 0 for f in m["env_params"]["files"])
    assert task_manifest(d, task_name="renamed")["tags"] == {"task_name": "renamed"}


def test_manifest_without_grader_or_toml(tmp_path):
    d = tmp_path / "bare"
    d.mkdir()
    (d / "instruction.md").write_text("hi")
    m = task_manifest(d)
    assert m["grader_params"] is None and m["env_params"]["config"] == {}
