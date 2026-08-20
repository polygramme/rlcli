import json

from click.testing import CliRunner

from rlcli.cli import main_cli
from rlcli.importers import flatten_content, iter_conversations, normalize_role


def _convs(lines, fmt, **kw):
    return [m for m, _ in iter_conversations(iter(lines), fmt, **kw)]


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


def test_openai_format_drops_tool_plumbing():
    line = json.dumps({"messages": [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "what time is it?"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"type": "function", "function": {"name": "clock"}}]},
        {"role": "tool", "content": "12:00"},
        {"role": "assistant", "content": "It's noon."},
    ]})
    (conv,) = _convs([line], "openai")
    assert [m["role"] for m in conv] == ["system", "user", "assistant"]
    assert conv[-1]["content"] == "It's noon."


def test_anthropic_format_flattens_blocks_and_system():
    line = json.dumps({
        "system": "you are terse",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "2+2?"}]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "4"},
                {"type": "tool_use", "name": "calc", "input": {}},
            ]},
        ],
    })
    (conv,) = _convs([line], "anthropic")
    assert conv[0] == {"role": "system", "content": "you are terse"}
    assert conv[-1] == {"role": "assistant", "content": "4"}


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


def test_import_command_empty_result_errors(tmp_path):
    src = tmp_path / "dump.jsonl"
    src.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")
    result = CliRunner().invoke(main_cli, ["import", str(src)])
    assert result.exit_code != 0
    assert "No usable conversations" in result.output
