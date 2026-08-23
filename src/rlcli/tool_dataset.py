"""Tool-aware supervised dataset builder.

The cookbook renderers train on tool calls, but they require pydantic
ToolCall objects (`tool_call.function.name`), while a JSONL dataset loads as
plain dicts — and the HuggingFace/Arrow round-trip additionally null-fills
optional keys (a text-only message gains tool_calls=None). This builder is
FromConversationFileBuilder with a coercion step at datum time: strip Arrow
nulls, validate dict tool calls into ToolCall, leave text-only rows
untouched. Lives in rlcli per policy — compatibility code stays here, not in
patches to the cookbook.
"""

from __future__ import annotations

import json

import chz
import datasets
import tinker
from tinker_cookbook.renderers import Message, TrainOnWhat
from tinker_cookbook.renderers.base import ToolCall
from tinker_cookbook.exceptions import DataFormatError
from tinker_cookbook.supervised.data import (
    FromConversationFileBuilder,
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)

import blobfile


def coerce_message(m: dict) -> Message:
    """Plain-dict message (possibly Arrow-null-filled) → renderer-ready."""
    out = {k: v for k, v in m.items() if v is not None}
    out.setdefault("content", "")
    calls = out.get("tool_calls")
    if calls:
        out["tool_calls"] = [
            tc if isinstance(tc, ToolCall) else ToolCall.model_validate(tc)
            for tc in calls
        ]
    else:
        out.pop("tool_calls", None)
    return out  # type: ignore[return-value]


@chz.chz
class ToolAwareConversationFileBuilder(FromConversationFileBuilder):
    """FromConversationFileBuilder that renders preserved tool calls."""

    def __call__(self):
        conversations = []
        with blobfile.BlobFile(self.file_path, "r", streaming=False) as f:
            for line in f:
                data = json.loads(line.strip())
                if "messages" not in data:
                    raise DataFormatError(
                        "Each line in the JSONL file must contain a 'messages' "
                        f"field. Got: {data.keys()}"
                    )
                conversations.append(data)

        dataset = datasets.Dataset.from_list(conversations)
        if self.shuffle_seed is not None:
            dataset = dataset.shuffle(seed=self.shuffle_seed)
        if self.test_size > 0 and len(dataset) > self.test_size:
            test_ds = dataset.take(self.test_size)
            train_ds = dataset.skip(self.test_size)
        else:
            train_ds = dataset
            test_ds = None

        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        def map_fn(row: dict) -> tinker.Datum:
            conversation = [coerce_message(m) for m in row["messages"]]
            return conversation_to_datum(
                conversation, self.renderer, self.common_config.max_length,
                train_on_what,
            )

        supervised_dataset = SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        )
        test_dataset = (
            SupervisedDatasetFromHFDataset(test_ds, batch_size=len(test_ds), map_fn=map_fn)
            if test_ds is not None
            else None
        )
        return supervised_dataset, test_dataset
