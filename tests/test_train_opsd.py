import asyncio
import json

import pytest
from click.testing import CliRunner

from rlcli.cli import main_cli

pytest.importorskip("tinker_cookbook")

MODEL = "Qwen/Qwen3-4B-Instruct-2507"


@pytest.fixture()
def prompts(tmp_path):
    path = tmp_path / "prompts.jsonl"
    rows = [
        {"prompt": "What is 17 * 23?"},
        {"messages": [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Name a prime above 100."},
            {"role": "assistant", "content": "101"},
        ]},
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "Explain LoRA."}]}]},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


def test_opsd_dry_run_self_distillation(prompts):
    result = CliRunner().invoke(main_cli, [
        "train", "opsd", "--model", MODEL, "--dataset", prompts, "--dry-run",
    ])
    assert result.exit_code == 0, result.output or result.exception
    assert "3 prompts" in result.output
    assert f"teacher={MODEL}," in result.output
    assert "loss_fn='importance_sampling'" in result.output
    assert "kl_penalty_coef=1.0" in result.output


def test_opsd_dry_run_teacher_checkpoint_and_hint(prompts):
    result = CliRunner().invoke(main_cli, [
        "train", "opsd", "--model", MODEL, "--dataset", prompts,
        "--teacher", "tinker://run/weights/ckpt-5", "--teacher-hint", "The answer is 391.",
        "--kl-coef", "0.5", "--steps", "7", "--dry-run",
    ])
    assert result.exit_code == 0, result.output or result.exception
    assert "teacher=tinker://run/weights/ckpt-5 + hint" in result.output
    assert "load_checkpoint_path='tinker://run/weights/ckpt-5'" in result.output
    assert "teacher_hint='The answer is 391.'" in result.output
    assert "kl_penalty_coef=0.5" in result.output
    assert "max_steps=7" in result.output


def test_opsd_bad_dataset_row_errors(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"question": "no prompt key"}\n')
    result = CliRunner().invoke(main_cli, [
        "train", "opsd", "--model", MODEL, "--dataset", str(path), "--dry-run",
    ])
    assert result.exit_code != 0
    assert "line 1" in result.output


def test_load_prompt_rows_uses_last_user_turn(prompts):
    from rlcli.opsd import load_prompt_rows

    rows = load_prompt_rows(prompts)
    assert [r.question for r in rows] == ["What is 17 * 23?", "Name a prime above 100.", "Explain LoRA."]
    assert rows[1].convo_prefix == ({"role": "system", "content": "Be terse."},)
    assert rows[1].messages("HINT")[-1] == {"role": "user", "content": "HINT\n\nName a prime above 100."}


def test_hinted_teacher_realigns_logprobs():
    import tinker

    from rlcli.opsd import HintedTeacher

    class FakeClient:
        def __init__(self):
            self.calls = []

        async def compute_logprobs_async(self, prompt):
            tokens = prompt.to_ints()
            self.calls.append(tokens)
            return [None] + [float(t) for t in tokens[1:]]

    student, hinted, completion = [1, 2, 3], [9, 9, 9, 1, 2, 3], [40, 50]
    fake = FakeClient()
    teacher = HintedTeacher(fake, {tuple(student): hinted})
    out = asyncio.run(teacher.compute_logprobs_async(tinker.ModelInput.from_ints(student + completion)))
    assert fake.calls == [hinted + completion]
    assert out == [None, 0.0, 0.0, 40.0, 50.0]
    with pytest.raises(RuntimeError, match="registered student prompt"):
        asyncio.run(teacher.compute_logprobs_async(tinker.ModelInput.from_ints([7, 8, 9])))


def test_row_level_hints_override_the_run_wide_hint(tmp_path):
    from rlcli.opsd import JsonlPromptDataset, PromptRow, load_prompt_rows

    p = tmp_path / "p.jsonl"
    p.write_text(
        '{"prompt": "What is 17*23?", "hint": "The answer is 391."}\n'
        '{"messages": [{"role": "user", "content": "Name a prime."}]}\n'
        '{"prompt": "x", "hint": "  "}\n'
    )
    import pytest
    from rlcli.opsd import PromptFormatError

    with pytest.raises(PromptFormatError):
        load_prompt_rows(str(p))
    p.write_text(p.read_text().rsplit("\n", 2)[0] + "\n")
    rows = load_prompt_rows(str(p))
    assert rows[0].hint == "The answer is 391." and rows[1].hint is None

    class R:  # renderer: one token per char of the joined messages
        def build_generation_prompt(self, messages):
            class MI:
                def __init__(self, ids):
                    self._ids = ids

                def to_ints(self):
                    return self._ids

            return MI([ord(c) % 97 for m in messages for c in m["content"]])

    ds = JsonlPromptDataset(rows, batch_size=2, group_size=1, renderer=R(), teacher_hint=None)
    assert ds.hinted and ds.hint_for(rows[0]) == "The answer is 391." and ds.hint_for(rows[1]) is None
    ds.get_batch(0)
    assert len(ds.hinted_prompts) == 1  # only the hinted row registers a teacher prompt
    ds2 = JsonlPromptDataset(rows, batch_size=2, group_size=1, renderer=R(), teacher_hint="Think step by step.")
    assert ds2.hint_for(rows[0]) == "The answer is 391." and ds2.hint_for(rows[1]) == "Think step by step."
    ds2.get_batch(0)
    assert len(ds2.hinted_prompts) == 2
    assert not JsonlPromptDataset([PromptRow("q")], 1, 1, R()).hinted
