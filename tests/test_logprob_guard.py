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
    SERVER_PREFIX,
    STD_DIFF_KEY,
    LogprobMismatch,
    abs_diff_metrics,
    install,
    server_metrics,
)


class FwdBwdResult:
    """Stand-in for tinker.ForwardBackwardOutput."""

    def __init__(self, metrics=None):
        self.metrics = metrics or {}
        self.loss_fn_outputs = []


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
    original_harvest = rl_train._training_logprobs_from_fwd_bwd
    logprob_guard._harvested.clear()
    yield rl_train
    rl_train.compute_kl_sample_train = original
    rl_train._training_logprobs_from_fwd_bwd = original_harvest
    logprob_guard._harvested.clear()


# --------------------------------------------------------------------------
# harvesting the trainer's own numbers
# --------------------------------------------------------------------------


def test_server_metrics_are_read_with_or_without_the_reduction_suffix():
    """_extract_metrics emits ':mean'/':max' suffixes that may not survive the SDK."""
    stem = "policy/rollout_train_logprobs_abs_diff"
    logprob_guard._harvested.append({f"{stem}_mean": 0.011, f"{stem}_max": 0.4})
    plain = server_metrics()
    logprob_guard._harvested.append({f"{stem}_mean:mean": 0.011, f"{stem}_max:max": 0.4})
    suffixed = server_metrics()
    assert plain == suffixed == {ABS_DIFF_KEY: pytest.approx(0.011), MAX_DIFF_KEY: pytest.approx(0.4)}


def test_several_forward_backwards_reduce_into_one_step():
    """Means average, maxima max -- the reduction the backend already applies."""
    stem = "policy/rollout_train_logprobs_abs_diff"
    logprob_guard._harvested.extend(
        [
            {f"{stem}_mean": 0.010, f"{stem}_max": 0.2, f"{stem}_std": 0.004},
            {f"{stem}_mean": 0.020, f"{stem}_max": 0.9, f"{stem}_std": 0.006},
        ]
    )
    out = server_metrics()
    assert out[ABS_DIFF_KEY] == pytest.approx(0.015)
    assert out[MAX_DIFF_KEY] == pytest.approx(0.9)
    assert out[STD_DIFF_KEY] == pytest.approx(0.005)


def test_draining_leaves_nothing_for_the_next_step():
    logprob_guard._harvested.append({"policy/rollout_train_logprobs_abs_diff_mean": 0.01})
    assert server_metrics()
    assert server_metrics() == {}


def test_the_harvest_buffer_is_bounded(rl_train):
    """A path that forward-backwards without ever draining must not grow forever."""
    install()
    for i in range(logprob_guard._MAX_HARVEST + 50):
        rl_train._training_logprobs_from_fwd_bwd(
            FwdBwdResult({"policy/rollout_train_logprobs_abs_diff_mean": float(i)})
        )
    assert len(logprob_guard._harvested) == logprob_guard._MAX_HARVEST
    # and it kept the newest, not the oldest
    assert logprob_guard._harvested[-1] == {
        "policy/rollout_train_logprobs_abs_diff_mean": float(logprob_guard._MAX_HARVEST + 49)
    }


def test_unrelated_metrics_are_ignored():
    logprob_guard._harvested.append({"total_loss:sum": 1.0, "entropy_loss:sum": 0.2})
    assert server_metrics() == {}


def test_the_harvest_hook_captures_metrics_and_passes_the_result_through(rl_train):
    install()
    seen = rl_train._training_logprobs_from_fwd_bwd(
        FwdBwdResult({"policy/rollout_train_logprobs_abs_diff_mean": 0.012})
    )
    assert seen == []  # the wrapped function still runs and returns its value
    assert logprob_guard._harvested == [
        {"policy/rollout_train_logprobs_abs_diff_mean": 0.012}
    ]


def test_the_breach_is_decided_locally_never_by_the_server(rl_train, caplog):
    """The server averages over prompt tokens too (mask is stripped before
    forward_backward), so a healthy run reads as a breach there. A high server
    number with a clean local diff must NOT breach — and vice versa."""
    install(threshold=0.03)
    rl_train._training_logprobs_from_fwd_bwd(
        FwdBwdResult({"policy/rollout_train_logprobs_abs_diff_mean": 0.4})  # unmasked, alarming
    )
    d = datum([-0.5, -0.5], [1, 1])
    with caplog.at_level("ERROR"):
        metrics = rl_train.compute_kl_sample_train([d], [torch.tensor([-0.5, -0.5])])
    assert metrics[ABS_DIFF_KEY] == pytest.approx(0.0)
    assert metrics[BREACH_KEY] == 0.0, "server's unmasked mean must not decide the breach"
    assert metrics[SERVER_PREFIX + "abs_diff_mean"] == pytest.approx(0.4)
    assert "disagree" not in caplog.text


def test_a_real_local_divergence_breaches_even_when_the_server_looks_fine(rl_train):
    install(threshold=0.03)
    rl_train._training_logprobs_from_fwd_bwd(
        FwdBwdResult({"policy/rollout_train_logprobs_abs_diff_mean": 0.001})
    )
    d = datum([-1.0, -1.0], [1, 1])
    metrics = rl_train.compute_kl_sample_train([d], [torch.tensor([-1.5, -0.5])])
    assert metrics[ABS_DIFF_KEY] == pytest.approx(0.5)
    assert metrics[BREACH_KEY] == 1.0


def test_reinstall_takes_the_new_threshold_and_drops_stale_batches(rl_train):
    """A warm container serving a second run must not keep the first run's settings."""
    install(threshold=0.03)
    rl_train._training_logprobs_from_fwd_bwd(FwdBwdResult({"policy/rollout_train_logprobs_abs_diff_mean": 9.0}))
    install(threshold=0.9)  # new run, looser threshold
    assert logprob_guard._harvested == [], "previous run's batches leaked"
    d = datum([-1.0, -1.0], [1, 1])
    metrics = rl_train.compute_kl_sample_train([d], [torch.tensor([-1.5, -0.5])])  # local diff 0.5
    assert metrics[BREACH_KEY] == 0.0, "0.5 is under the re-installed 0.9"


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
