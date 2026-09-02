"""On-policy self-distillation (OPSD) glue for `rlcli train opsd`.

The cookbook's on-policy distillation loop
(tinker_cookbook.distillation.train_on_policy) samples the student on a
prompt, scores the student's own tokens under a teacher SamplingClient, and
folds the negative reverse KL into the per-token advantages
(incorporate_kl_penalty) before an importance_sampling update. Its prompt
sources are HuggingFace datasets (deepmath, tulu3); this module adds a
prompts-JSONL builder in the same shape, plus the privileged-hint variant.

Privileged-hint OPSD (same weights, teacher sees hint + prompt, student sees
only the prompt) does not fit the cookbook loop as-is: the teacher is scored
on the *student's* sequence, and train_on_policy.main() builds the teacher
clients itself. So the hint is implemented here as
  - HintedTeacher: a SamplingClient stand-in whose compute_logprobs_async
    swaps the student prompt for the hinted prompt, scores, and re-aligns
    the logprobs to the student's positions; and
  - main(): train_on_policy.main() with the teacher construction swapped
    (only used when a hint is given; the plain path calls the cookbook's
    main() unchanged).
Lives in rlcli per policy — compatibility code stays here, not in patches to
the cookbook.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Sequence

import chz
import tinker
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.distillation import train_on_policy
from tinker_cookbook.distillation.datasets import (
    CompositeDataset,
    PromptOnlyEnv,
    TeacherConfig,
)
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.git_rev import recipe_user_metadata

from rlcli.importers import flatten_content

logger = logging.getLogger(__name__)

HINT_SEPARATOR = "\n\n"


class PromptFormatError(ValueError):
    """A JSONL row is not a usable prompt."""


@dataclass(frozen=True)
class PromptRow:
    """One prompt: the final user turn plus any preceding context turns."""

    question: str
    convo_prefix: tuple[dict, ...] = ()

    def messages(self, hint: str | None = None) -> list[renderers.Message]:
        content = f"{hint}{HINT_SEPARATOR}{self.question}" if hint else self.question
        return [*self.convo_prefix, {"role": "user", "content": content}]  # type: ignore[list-item]


def load_prompt_rows(path: str) -> list[PromptRow]:
    """Read {"prompt": str} or {"messages": [...]} rows.

    For messages rows the last user turn is the prompt and everything before
    it (system/user/assistant text turns) is kept as context; trailing
    assistant turns (e.g. `rlcli import` output) are dropped, since the
    student generates the answer on-policy.
    """
    rows: list[PromptRow] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise PromptFormatError(f"line {lineno}: invalid JSON ({e.msg})") from e
            if not isinstance(record, dict):
                raise PromptFormatError(f"line {lineno}: expected an object, got {type(record).__name__}")
            if "prompt" in record:
                prompt = record["prompt"]
                if not isinstance(prompt, str) or not prompt.strip():
                    raise PromptFormatError(f"line {lineno}: 'prompt' must be a non-empty string")
                rows.append(PromptRow(question=prompt))
                continue
            if "messages" not in record:
                raise PromptFormatError(
                    f"line {lineno}: expected a 'prompt' or 'messages' key, got keys {sorted(record)}"
                )
            messages = record["messages"]
            if not isinstance(messages, list):
                raise PromptFormatError(f"line {lineno}: 'messages' is not a list")
            turns = [
                {"role": m.get("role"), "content": flatten_content(m.get("content"))}
                for m in messages
                if isinstance(m, dict) and m.get("role") in ("system", "user", "assistant")
            ]
            last_user = max((i for i, m in enumerate(turns) if m["role"] == "user"), default=-1)
            if last_user < 0 or not turns[last_user]["content"].strip():
                raise PromptFormatError(f"line {lineno}: no user message with text content")
            rows.append(
                PromptRow(question=turns[last_user]["content"], convo_prefix=tuple(turns[:last_user]))
            )
    if not rows:
        raise PromptFormatError(f"{path}: no prompts found")
    return rows


def teacher_config_for(model_name: str, teacher: str | None) -> TeacherConfig:
    """--teacher: unset = same base (self-distillation); tinker://... = a
    checkpoint of the student's base on this server; anything else = a base
    model name."""
    if teacher is None or teacher == model_name:
        return TeacherConfig(base_model=model_name)
    if teacher.startswith("tinker://"):
        return TeacherConfig(base_model=model_name, load_checkpoint_path=teacher)
    return TeacherConfig(base_model=teacher)


def describe_teacher(teacher_config: TeacherConfig, hint: str | None) -> str:
    desc = teacher_config.load_checkpoint_path or teacher_config.base_model
    return f"{desc} + hint" if hint else desc


class HintedTeacher:
    """Teacher SamplingClient that scores the student's completion under a
    hinted prompt.

    incorporate_kl_penalty calls compute_logprobs_async(student prompt +
    completion) and uses positions >= the prompt length (the mask zeroes the
    rest). We look the student prompt up in `hinted_prompts` (filled by
    JsonlPromptDataset.get_batch as batches are drawn), score hinted prompt
    + completion instead, and return a list of the original length with the
    completion logprobs in the student's positions.
    """

    def __init__(self, client: tinker.SamplingClient, hinted_prompts: dict[tuple[int, ...], list[int]]):
        self.client = client
        self.hinted_prompts = hinted_prompts

    def __getattr__(self, name: str):
        return getattr(self.client, name)

    async def compute_logprobs_async(self, prompt: tinker.ModelInput) -> list[float | None]:
        tokens = prompt.to_ints()
        for student_len in sorted({len(k) for k in self.hinted_prompts}, reverse=True):
            hinted = self.hinted_prompts.get(tuple(tokens[:student_len]))
            if hinted is not None:
                break
        else:
            raise RuntimeError(
                "HintedTeacher: sequence does not start with a registered student prompt; "
                "the hint cannot be applied (was the dataset built by JsonlPromptDatasetBuilder?)"
            )
        completion = tokens[student_len:]
        logprobs = await self.client.compute_logprobs_async(tinker.ModelInput.from_ints(hinted + completion))
        if len(logprobs) != len(hinted) + len(completion):
            raise RuntimeError(
                f"HintedTeacher: teacher returned {len(logprobs)} logprobs for "
                f"{len(hinted) + len(completion)} tokens"
            )
        # Position 0 has no logprob (dropped by the caller); prompt positions are masked out.
        return [None, *([0.0] * (student_len - 1)), *logprobs[len(hinted):]]


class JsonlPromptDataset(RLDataset):
    """Prompt-only RL dataset (zero reward; the KL to the teacher is the only signal)."""

    def __init__(
        self,
        rows: list[PromptRow],
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        teacher_hint: str | None = None,
        dataset_name: str = "prompts",
    ):
        self.rows = rows
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.teacher_hint = teacher_hint
        self.dataset_name = dataset_name
        # student prompt tokens -> hinted prompt tokens, consumed by HintedTeacher.
        self.hinted_prompts: dict[tuple[int, ...], list[int]] = {}

    def _register_hint(self, row: PromptRow) -> None:
        student = tuple(self.renderer.build_generation_prompt(row.messages()).to_ints())
        if student not in self.hinted_prompts:
            hinted = self.renderer.build_generation_prompt(row.messages(self.teacher_hint))
            self.hinted_prompts[student] = hinted.to_ints()

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        rows = self.rows[index * self.batch_size : (index + 1) * self.batch_size]
        assert rows, "Incorrect batch size"
        if self.teacher_hint:
            for row in rows:
                self._register_hint(row)
        return [
            ProblemGroupBuilder(
                env_thunk=partial(
                    PromptOnlyEnv,
                    row.question,
                    self.renderer,
                    convo_prefix=list(row.convo_prefix),
                ),
                num_envs=self.group_size,
                dataset_name=self.dataset_name,
            )
            for row in rows
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.rows) / self.batch_size)


@chz.chz
class JsonlPromptDatasetBuilder(RLDatasetBuilder):
    """RLDatasetBuilder over a prompts JSONL, in the shape of the cookbook's
    PromptOnlyDatasetBuilder (recipes/distillation/on_policy_multi_teacher.py)."""

    file_path: str
    groups_per_batch: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    teacher_hint: str | None = None

    async def __call__(self) -> tuple[JsonlPromptDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        dataset = JsonlPromptDataset(
            rows=load_prompt_rows(self.file_path),
            batch_size=self.groups_per_batch,
            group_size=self.group_size,
            renderer=renderer,
            teacher_hint=self.teacher_hint,
            dataset_name=Path(self.file_path).stem,
        )
        return dataset, None


async def main(config: train_on_policy.Config, teacher_hint: str | None = None) -> None:
    """Run on-policy distillation; with a hint, the teacher scores hint + prompt.

    Without a hint this is the cookbook's train_on_policy.main(). With one, the
    setup below mirrors that function at the pinned cookbook commit (keep in
    sync when bumping COOKBOOK_REQUIREMENT), except that each teacher
    SamplingClient is wrapped in HintedTeacher fed by its dataset.
    """
    if not teacher_hint:
        await train_on_policy.main(config)
        return

    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        config=config,
        wandb_name=config.wandb_name,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    start_batch = resume_info.batch if resume_info else 0

    service_client = tinker.ServiceClient(
        base_url=config.base_url,
        user_metadata=recipe_user_metadata(config.recipe_name),
    )
    user_metadata: dict[str, str] = {}
    if wandb_link := ml_logger.get_logger_url():
        user_metadata["wandb_link"] = wandb_link
    checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, config.renderer_name)
    model_info.warn_if_renderer_not_recommended(config.model_name, config.renderer_name)

    if resume_info:
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, resume_info.state_path, config.renderer_name
        )
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info.state_path, user_metadata=user_metadata
            )
        )
        logger.info(f"Resumed training from {resume_info.state_path}")
    elif config.load_checkpoint_path:
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, config.load_checkpoint_path, config.renderer_name
        )
        training_client = await service_client.create_training_client_from_state_async(
            config.load_checkpoint_path, user_metadata=user_metadata
        )
        logger.info(f"Loaded weights from {config.load_checkpoint_path}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            config.model_name, rank=config.lora_rank, user_metadata=user_metadata
        )

    tokenizer = get_tokenizer(config.model_name)

    datasets = []
    teacher_clients = []
    groups_per_batch_list = []
    evaluators = [evaluator() for evaluator in config.evaluator_builders]
    for dataset_config in config.dataset_configs:
        dataset, maybe_test_dataset = await dataset_config.dataset_builder()
        datasets.append(dataset)
        groups_per_batch_list.append(dataset_config.groups_per_batch)
        if maybe_test_dataset is not None:
            evaluators.append(RLTestSetEvaluator(maybe_test_dataset, max_tokens=config.max_tokens))

        teacher_config = dataset_config.teacher_config
        teacher_client = service_client.create_sampling_client(
            base_model=teacher_config.base_model,
            model_path=teacher_config.load_checkpoint_path,
        )
        if not isinstance(dataset, JsonlPromptDataset) or dataset.teacher_hint != teacher_hint:
            raise ValueError("teacher_hint needs every dataset to be a hinted JsonlPromptDataset")
        teacher_clients.append(HintedTeacher(teacher_client, dataset.hinted_prompts))
        logger.info(
            f"Created hinted teacher sampling client for {teacher_config.base_model} "
            f"(checkpoint: {teacher_config.load_checkpoint_path}, hint: {teacher_hint!r})"
        )

    composite_dataset = CompositeDataset(datasets, groups_per_batch_list)
    num_batches = len(composite_dataset)
    if config.max_steps is not None:
        num_batches = min(config.max_steps, num_batches)
    logger.info(f"Will train on {num_batches} batches")

    checkpoint_mgr = checkpoint_utils.CheckpointManager(
        training_client=training_client,
        service_client=service_client,
        log_path=config.log_path,
        save_every=config.save_every,
        store=ml_logger.store,
    )

    await train_on_policy.do_sync_training(
        start_batch=start_batch,
        end_batch=num_batches,
        num_batches=num_batches,
        config=config,
        training_client=training_client,
        checkpoint_mgr=checkpoint_mgr,
        service_client=service_client,
        evaluators=evaluators,
        dataset=composite_dataset,
        teacher_clients=teacher_clients,  # type: ignore[arg-type]  # HintedTeacher quacks for incorporate_kl_penalty
        ml_logger=ml_logger,
        tokenizer=tokenizer,
    )

    if start_batch < num_batches:
        await checkpoint_mgr.save_final_async(loop_state={"batch": num_batches})
    else:
        logger.info("Training was already complete; nothing to do")

    ml_logger.close()
    logger.info("Training completed successfully")
