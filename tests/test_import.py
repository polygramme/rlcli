import json

from click.testing import CliRunner

from rlcli.cli import main_cli
from rlcli.importers import flatten_content, iter_conversations, normalize_role


def _records(lines, fmt, **kw):
    return [r for r, _ in iter_conversations(iter(lines), fmt, **kw)]


def _convs(lines, fmt, **kw):
    return [r["messages"] for r in _records(lines, fmt, **kw)]


def test_role_normalization():
    assert normalize_role("human") == "user"
    assert normalize_role("ai") == "assistant"
    assert normalize_role("SYSTEM") == "system"
    assert normalize_role("tool") is None
    assert normalize_role("function") is None
    assert normalize_role("narrator") is None


def test_flatten_blocks_keeps_text_drops_tools():
    blocks = [
        {"type": "text", "text": "hello"},
        {"type": "tool_use", "id": "t1", "name": "bash", "input": {}},
        {"type": "thinking", "thinking": "secret"},
        {"type": "text", "text": "world"},
    ]
    assert flatten_content(blocks) == "hello\nworld"
    assert flatten_content("plain") == "plain"
    assert flatten_content(None) == ""


def test_messages_format_normalizes_langchain_roles():
    line = json.dumps({"messages": [
        {"role": "human", "content": "hi"},
        {"role": "ai", "content": "hello"},
    ]})
    convs = _convs([line], "messages")
    assert convs == [[{"role": "user", "content": "hi"},
                      {"role": "assistant", "content": "hello"}]]


def test_openai_format_preserves_tools_by_default():
    line = json.dumps({"messages": [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "what time is it?"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "clock", "arguments": "{}"}}]},
        {"role": "tool", "content": "12:00", "tool_call_id": "c1"},
        {"role": "assistant", "content": "It's noon."},
    ]})
    (conv,) = _convs([line], "openai")
    assert [m["role"] for m in conv] == ["system", "user", "assistant", "tool", "assistant"]
    call_turn = conv[2]
    assert call_turn["tool_calls"] == [{"id": "c1", "type": "function",
                                        "function": {"name": "clock", "arguments": "{}"}}]
    assert conv[3] == {"role": "tool", "content": "12:00", "tool_call_id": "c1"}


def test_openai_format_drop_tools_flag_restores_text_only():
    line = json.dumps({"messages": [
        {"role": "user", "content": "what time is it?"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"type": "function", "function": {"name": "clock"}}]},
        {"role": "tool", "content": "12:00"},
        {"role": "assistant", "content": "It's noon."},
    ]})
    (conv,) = _convs([line], "openai", preserve_tools=False)
    assert [m["role"] for m in conv] == ["user", "assistant"]
    assert conv[-1]["content"] == "It's noon."


def test_anthropic_format_tool_use_and_result_blocks():
    line = json.dumps({
        "system": "you are terse",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "2+2?"}]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "let me check"},
                {"type": "tool_use", "id": "tu1", "name": "calc",
                 "input": {"expr": "2+2"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu1", "content": "4"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "4"}]},
        ],
    })
    (conv,) = _convs([line], "anthropic")
    assert conv[0] == {"role": "system", "content": "you are terse"}
    assert [m["role"] for m in conv] == ["system", "user", "assistant", "tool", "assistant"]
    tc = conv[2]["tool_calls"][0]
    assert tc["id"] == "tu1" and tc["function"]["name"] == "calc"
    assert json.loads(tc["function"]["arguments"]) == {"expr": "2+2"}
    assert conv[3] == {"role": "tool", "content": "4", "tool_call_id": "tu1"}


def test_vercel_format_flattens_parts():
    line = json.dumps({"messages": [
        {"role": "user", "parts": [{"type": "text", "text": "hi"},
                                   {"type": "step-start"}]},
        {"role": "assistant", "parts": [{"type": "text", "text": "hello"}]},
    ]})
    (conv,) = _convs([line], "vercel")
    assert conv == [{"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"}]


def test_csv_format_column_pairs_and_reward(tmp_path):
    src = tmp_path / "data.csv"
    src.write_text('system,user,assistant,reward\n'
                   'be terse,"2+2?","4",1.0\n'
                   ',"skip me","",0.5\n')
    result = CliRunner().invoke(main_cli, ["import", str(src), "-f", "csv"])
    assert result.exit_code == 0, result.output
    row = json.loads(next(l for l in result.output.splitlines() if l.startswith("{")))
    assert row["messages"][0] == {"role": "system", "content": "be terse"}
    assert row["messages"][-1] == {"role": "assistant", "content": "4"}
    assert row["reward"] == 1.0


def test_telemetry_join_and_min_score(tmp_path):
    src = tmp_path / "convs.jsonl"
    src.write_text("\n".join([
        json.dumps({"trace_id": "t1", "messages": [
            {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}),
        json.dumps({"trace_id": "t2", "messages": [
            {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]}),
    ]) + "\n")
    events = tmp_path / "events.jsonl"
    events.write_text("\n".join([
        json.dumps({"trace_id": "t1", "key": "thumbs", "score": 1}),
        json.dumps({"trace_id": "t1", "key": "thumbs", "score": 0}),
        json.dumps({"trace_id": "t2", "key": "thumbs", "score": 1}),
    ]) + "\n")
    result = CliRunner().invoke(main_cli, [
        "import", str(src), "--telemetry", str(events), "--min-score", "0.9"])
    assert result.exit_code == 0, result.output
    rows = [json.loads(l) for l in result.output.strip().splitlines()
            if l.startswith("{")]
    assert len(rows) == 1
    assert rows[0]["trace_id"] == "t2" and rows[0]["reward"] == 1.0


def test_redaction_covers_content_and_tool_arguments(tmp_path):
    src = tmp_path / "convs.jsonl"
    src.write_text(json.dumps({"messages": [
        {"role": "user", "content": "email bob@example.com please"},
        {"role": "assistant", "content": "on it",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "send",
                                      "arguments": '{"to": "bob@example.com"}'}}]},
    ]}) + "\n")
    result = CliRunner().invoke(main_cli, ["import", str(src), "--redact", "email"])
    assert result.exit_code == 0, result.output
    row = json.loads(next(l for l in result.output.splitlines() if l.startswith("{")))
    assert "bob@example.com" not in json.dumps(row)
    assert "[REDACTED:email]" in row["messages"][0]["content"]
    assert "[REDACTED:email]" in row["messages"][1]["tool_calls"][0]["function"]["arguments"]


def test_skips_conversations_without_assistant_or_too_short():
    lines = [
        json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
        json.dumps({"messages": [{"role": "user", "content": "a"},
                                 {"role": "user", "content": "b"}]}),
    ]
    assert _convs(lines, "messages") == []


def test_bad_json_fails_loud_with_lineno():
    import pytest

    from rlcli.importers import ImportFormatError

    with pytest.raises(ImportFormatError, match="line 2"):
        _convs(['{"messages": [{"role":"user","content":"x"},{"role":"ai","content":"y"}]}',
                "{not json"], "messages")


def test_import_command_end_to_end(tmp_path):
    src = tmp_path / "dump.jsonl"
    src.write_text(json.dumps({"messages": [
        {"role": "human", "content": "hi"},
        {"role": "ai", "content": "hello"},
    ]}) + "\n")
    out = tmp_path / "out.jsonl"
    result = CliRunner().invoke(main_cli, ["import", str(src), "-f", "messages",
                                           "-o", str(out)])
    assert result.exit_code == 0, result.output
    row = json.loads(out.read_text().strip())
    assert row["messages"][0]["role"] == "user"


def _ls_run(feedback_stats=None, feedback=None, outputs=None):
    rec = {
        "inputs": {"messages": [
            {"type": "human", "content": "refund order 123"},
        ]},
        "outputs": outputs if outputs is not None
        else {"output": {"role": "assistant", "content": "Refund issued."}},
    }
    if feedback_stats is not None:
        rec["feedback_stats"] = feedback_stats
    if feedback is not None:
        rec["feedback"] = feedback
    return json.dumps(rec)


def test_langsmith_basic_run_with_feedback_stats():
    (rec,) = _records([_ls_run(feedback_stats={"correctness": {"n": 2, "avg": 0.5}})],
                      "langsmith")
    assert rec["messages"] == [
        {"role": "user", "content": "refund order 123"},
        {"role": "assistant", "content": "Refund issued."},
    ]
    assert rec["reward"] == 0.5


def test_langsmith_langchain_serialized_messages_and_generations():
    line = json.dumps({
        "inputs": {"messages": [
            {"id": ["langchain", "schema", "HumanMessage"],
             "kwargs": {"content": "hi", "type": "human"}},
        ]},
        "outputs": {"generations": [[{"message": {
            "id": ["langchain", "schema", "AIMessage"],
            "kwargs": {"content": "hello", "type": "ai"}}}]]},
        "feedback": [{"key": "helpfulness", "score": 1}],
    })
    (rec,) = _records([line], "langsmith")
    assert rec["messages"][-1] == {"role": "assistant", "content": "hello"}
    assert rec["reward"] == 1.0


def test_langsmith_multi_key_feedback_requires_key():
    import pytest

    from rlcli.importers import ImportFormatError

    line = _ls_run(feedback_stats={"a": {"n": 1, "avg": 1.0}, "b": {"n": 1, "avg": 0.0}})
    with pytest.raises(ImportFormatError, match="feedback-key"):
        _records([line], "langsmith")
    (rec,) = _records([line], "langsmith", feedback_key="b")
    assert rec["reward"] == 0.0


def test_langsmith_no_feedback_no_reward_key():
    (rec,) = _records([_ls_run()], "langsmith")
    assert "reward" not in rec


def test_langsmith_output_echo_not_duplicated():
    line = json.dumps({
        "inputs": {"messages": [{"type": "human", "content": "hi"},
                                {"type": "ai", "content": "hello"}]},
        "outputs": {"messages": [{"type": "ai", "content": "hello"}]},
    })
    (rec,) = _records([line], "langsmith")
    assert [m["role"] for m in rec["messages"]] == ["user", "assistant"]


def test_import_command_langsmith_min_score(tmp_path):
    src = tmp_path / "runs.jsonl"
    src.write_text("\n".join([
        _ls_run(feedback_stats={"correctness": {"n": 1, "avg": 1.0}}),
        _ls_run(feedback_stats={"correctness": {"n": 1, "avg": 0.2}}),
        _ls_run(),  # no feedback → dropped under --min-score
    ]) + "\n")
    out = tmp_path / "out.jsonl"
    result = CliRunner().invoke(main_cli, ["import", str(src), "-f", "langsmith",
                                           "--min-score", "0.8", "-o", str(out)])
    assert result.exit_code == 0, result.output
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["reward"] == 1.0
    assert "2 outside score filter" in result.output


def test_import_command_max_score_selects_failures(tmp_path):
    src = tmp_path / "runs.jsonl"
    src.write_text("\n".join([
        _ls_run(feedback_stats={"correctness": {"n": 1, "avg": 1.0}}),
        _ls_run(feedback_stats={"correctness": {"n": 1, "avg": 0.2}}),
        _ls_run(),  # no feedback → dropped under score filter
    ]) + "\n")
    result = CliRunner().invoke(main_cli, ["import", str(src), "-f", "langsmith",
                                           "--max-score", "0.3"])
    assert result.exit_code == 0, result.output
    rows = [json.loads(l) for l in result.output.splitlines() if l.startswith("{")]
    assert len(rows) == 1 and rows[0]["reward"] == 0.2


def test_import_command_min_score_rejected_for_other_formats(tmp_path):
    src = tmp_path / "x.jsonl"
    src.write_text("{}\n")
    result = CliRunner().invoke(main_cli, ["import", str(src), "-f", "openai",
                                           "--min-score", "0.5"])
    assert result.exit_code != 0
    assert "need a reward source" in result.output
    result2 = CliRunner().invoke(main_cli, ["import", str(src), "-f", "openai",
                                            "--feedback-key", "x"])
    assert result2.exit_code != 0
    assert "only applies to -f langsmith" in result2.output


def test_import_command_empty_result_errors(tmp_path):
    src = tmp_path / "dump.jsonl"
    src.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")
    result = CliRunner().invoke(main_cli, ["import", str(src)])
    assert result.exit_code != 0
    assert "No usable conversations" in result.output
