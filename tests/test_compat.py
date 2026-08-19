"""Compat guarantees. test_gspo_survives_the_wire is the load-bearing one:
it executes the tinker SDK's real request→protobuf conversion and must pass
on every pinned SDK bump (no-upstream-PR policy: we absorb breakage here)."""

import pytest

from rlcli.compat import (
    ALL_LOSSES,
    SERVER_LOSSES,
    LossBackendError,
    as_loss_fn,
    ensure_loss_supported,
)


def test_gspo_survives_the_wire():
    from tinker import types
    from tinker.proto.request_conv import forward_backward_request_to_proto

    request = types.ForwardBackwardRequest(
        forward_backward_input=types.ForwardBackwardInput(
            data=[],
            loss_fn=as_loss_fn("gspo"),
            loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.2},
        ),
        model_id="model-0",
        seq_id=1,
    )
    proto = forward_backward_request_to_proto(request)
    assert proto.loss_fn == "gspo"
    assert proto.loss_fn_config["clip_low_threshold"] == pytest.approx(0.8)
    assert proto.loss_fn_config["clip_high_threshold"] == pytest.approx(1.2)


@pytest.mark.parametrize("loss", ["dppo", "ppo_critic", "cispo"])
def test_extended_losses_survive_the_wire(loss):
    from tinker import types
    from tinker.proto.request_conv import forward_backward_request_to_proto

    request = types.ForwardBackwardRequest(
        forward_backward_input=types.ForwardBackwardInput(data=[], loss_fn=as_loss_fn(loss)),
        model_id="m",
        seq_id=1,
    )
    assert forward_backward_request_to_proto(request).loss_fn == loss


def test_gspo_requires_torch_backend():
    with pytest.raises(LossBackendError, match="fsdp"):
        ensure_loss_supported("gspo", "jax")
    ensure_loss_supported("gspo", "fsdp")
    ensure_loss_supported("gspo", "megatron")


def test_jax_supports_its_documented_set():
    for loss in ("cross_entropy", "importance_sampling", "ppo", "cispo"):
        ensure_loss_supported(loss, "jax")


def test_unknown_backend_and_loss():
    with pytest.raises(LossBackendError):
        ensure_loss_supported("importance_sampling", "tpu")
    with pytest.raises(LossBackendError):
        ensure_loss_supported("not_a_loss", None)


def test_unknown_backend_none_is_permissive():
    for loss in ALL_LOSSES:
        ensure_loss_supported(loss, None)


def test_matrix_shape():
    assert SERVER_LOSSES["fsdp"] == SERVER_LOSSES["megatron"]
    assert "gspo" not in SERVER_LOSSES["jax"]
