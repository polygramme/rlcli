import json

import pytest

# rlcli.tool_dataset imports chz + tinker_cookbook at module level — the
# whole test module skips in cookbook-less environments (unit CI job).
pytest.importorskip("tinker_cookbook")


def test_tool_calls_render_through_builder(tmp_path):
    """A preserved tool-call conversation must survive Arrow round-trip,
    dict→ToolCall coercion, and real rendering into a training Datum."""
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    from rlcli.tool_dataset import ToolAwareConversationFileBuilder

    rows = [
        {"messages": [
            {"role": "user", "content": "what time is it?"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "clock", "arguments": "{}"}}]},
            {"role": "tool", "content": "12:00", "tool_call_id": "c1"},
            {"role": "assistant", "content": "It's noon."},
        ]},
        # text-only row in the same file: Arrow null-fills tool_calls; the
        # coercion must strip those nulls rather than crash the renderer.
        {"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ], "reward": 1.0},
    ]
    path = tmp_path / "convs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    builder = ToolAwareConversationFileBuilder(
        file_path=str(path),
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer="Qwen/Qwen3-4B-Instruct-2507",
            renderer_name="qwen3_instruct",
            max_length=512,
            batch_size=2,
        ),
    )
    train_ds, test_ds = builder()
    batch = train_ds.get_batch(0)
    assert len(batch) == 2
    assert all(d.model_input.length > 0 for d in batch)
    assert test_ds is None


def test_coerce_message_strips_arrow_nulls():
    from rlcli.tool_dataset import coerce_message

    m = coerce_message({"role": "assistant", "content": None, "tool_calls": None,
                        "tool_call_id": None})
    assert m == {"role": "assistant", "content": ""}
    m2 = coerce_message({"role": "assistant", "content": "x",
                         "tool_calls": [{"id": "a", "type": "function",
                                         "function": {"name": "f",
                                                      "arguments": "{}"}}]})
    assert m2["tool_calls"][0].function.name == "f"
