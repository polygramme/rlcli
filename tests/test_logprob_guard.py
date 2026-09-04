"""Tests for the trainer-vs-engine logprob replay check."""

from __future__ import annotations

import pytest
import torch

import tinker
from tinker import TensorData

from rlcli import logprob_guard
from rlcli.logprob_guard import (
    ABS_DIFF_KEY,
    BREACH_KEY,
    MAX_DIFF_KEY,
    LogprobMismatch,
    abs_diff_metrics,
    install,
)


def datum(sampling: list[float], mask: list[float]) -> tinker.Datum:
    n = len(sampling)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(list(range(1, n + 1))),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(list(range(1, n + 1)))),
            "logprobs": TensorData.from_torch(torch.tensor(sampling)),
            "advantages": TensorData.from_torch(torch.tensor([1.0] * n)),
            "mask": TensorData.from_torch(torch.tensor(mask)),
        },
    )


def test_identical_logprobs_report_no_difference():
    d = datum([-0.5, -0.2, -0.9], [0, 1, 1])
    out = abs_diff_metrics([d], [torch.tensor([-0.5, -0.2, -0.9])])
    assert out[ABS_DIFF_KEY] == pytest.approx(0.0)
    assert out[MAX_DIFF_KEY] == pytest.approx(0.0)


def test_only_action_tokens_count():
    """Observation tokens carry logprob 0.0 and must not dilute the mean."""
    d = datum([0.0, -0.2, -0.9], [0, 1, 1])
    # A huge discrepancy on the masked-out token is irrelevant.
    out = abs_diff_metrics([d], [torch.tensor([-99.0, -0.3, -1.0])])
    assert out[ABS_DIFF_KEY] == pytest.approx(0.1, abs=1e-6)


def test_symmetric_divergence_is_not_cancelled_out():
    """The reason we don't rely on the cookbook's signed kl_sample_train_v1."""
    d = datum([-1.0, -1.0], [1, 1])
    training = torch.tensor([-1.5, -0.5])  # signed mean is exactly zero
    assert training.mean() == pytest.approx(torch.tensor([-1.0, -1.0]).mean())
    out = abs_diff_metrics([d], [training])
    assert out[ABS_DIFF_KEY] == pytest.approx(0.5)


def test_datums_without_a_mask_are_skipped():
    d = tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={"target_tokens": TensorData.from_torch(torch.tensor([1, 2]))},
    )
    assert abs_diff_metrics([d], [torch.tensor([0.0, 0.0])]) == {}


def test_a_fully_masked_datum_is_skipped():
    d = datum([-0.5, -0.5], [0, 0])
    assert abs_diff_metrics([d], [torch.tensor([-9.0, -9.0])]) == {}


# --------------------------------------------------------------------------
# installation
# --------------------------------------------------------------------------


@pytest.fixture
def rl_train():
    rl_train = pytest.importorskip("tinker_cookbook.rl.train")
    original = rl_train.compute_kl_sample_train
    yield rl_train
    rl_train.compute_kl_sample_train = original


def test_install_preserves_the_cookbooks_own_metrics(rl_train):
    assert install() is True
    d = datum([-0.5, -0.5], [1, 1])
    metrics = rl_train.compute_kl_sample_train([d], [torch.tensor([-0.5, -0.5])])
    assert "optim/kl_sample_train_v1" in metrics
    assert metrics[ABS_DIFF_KEY] == pytest.approx(0.0)
    assert metrics[BREACH_KEY] == 0.0


def test_a_healthy_batch_does_not_breach(rl_train):
    install(threshold=0.03)
    d = datum([-0.5, -0.5], [1, 1])
    metrics = rl_train.compute_kl_sample_train([d], [torch.tensor([-0.504, -0.496])])
    assert metrics[ABS_DIFF_KEY] == pytest.approx(0.004, abs=1e-6)
    assert metrics[BREACH_KEY] == 0.0


def test_a_breach_is_reported_but_does_not_kill_the_run_by_default(rl_train, caplog):
    install(threshold=0.03)
    d = datum([-0.5, -0.5], [1, 1])
    with caplog.at_level("ERROR"):
        metrics = rl_train.compute_kl_sample_train([d], [torch.tensor([-1.0, -1.0])])
    assert metrics[BREACH_KEY] == 1.0
    assert "logprobs disagree" in caplog.text


def test_strict_mode_raises_on_a_breach(rl_train):
    install(threshold=0.03, strict=True)
    d = datum([-0.5, -0.5], [1, 1])
    with pytest.raises(LogprobMismatch, match="token pipeline"):
        rl_train.compute_kl_sample_train([d], [torch.tensor([-1.0, -1.0])])


def test_installing_twice_does_not_double_wrap(rl_train):
    install()
    once = rl_train.compute_kl_sample_train
    install()
    assert rl_train.compute_kl_sample_train is once


def test_install_reports_failure_when_the_hook_is_missing(rl_train, monkeypatch):
    monkeypatch.delattr(rl_train, "compute_kl_sample_train")
    assert install() is False


def test_a_metric_failure_never_breaks_training(rl_train, monkeypatch, caplog):
    """A diagnostic must not be able to fail a run it was only observing."""
    install()

    def boom(*_args, **_kwargs):
        raise ValueError("metric exploded")

    monkeypatch.setattr(logprob_guard, "abs_diff_metrics", boom)
    d = datum([-0.5, -0.5], [1, 1])
    with caplog.at_level("WARNING"):
        metrics = rl_train.compute_kl_sample_train([d], [torch.tensor([-0.5, -0.5])])
    # the cookbook's own metrics still come back; ours are simply absent
    assert "optim/kl_sample_train_v1" in metrics
    assert ABS_DIFF_KEY not in metrics
    assert "abs-diff metric failed" in caplog.text
