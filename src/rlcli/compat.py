"""Loss/backend support matrix and typing bridge to the tinker SDK.

The tinker SDK types ``loss_fn`` as a Literal of the five losses hosted Tinker
serves. SkyRL's server accepts a wider set, and the SDK transports the value as
a plain protobuf string without runtime validation (verified: the request path
uses the frozen dataclass, and ``request_conv.py`` copies ``loss_fn`` verbatim
into a ``str`` proto field). ``as_loss_fn`` documents that bridge in one place;
``tests/test_compat.py`` re-verifies the wire behavior on every SDK bump.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from tinker.types import LossFnType

# Which losses each server backend computes natively in one forward+backward.
SERVER_LOSSES: dict[str, frozenset[str]] = {
    "fsdp": frozenset(
        {"cross_entropy", "importance_sampling", "ppo", "gspo", "cispo", "ppo_critic", "dppo"}
    ),
    "megatron": frozenset(
        {"cross_entropy", "importance_sampling", "ppo", "gspo", "cispo", "ppo_critic", "dppo"}
    ),
    "jax": frozenset({"cross_entropy", "importance_sampling", "ppo", "cispo"}),
}

# Losses hosted Tinker serves natively (SDK LossFnType Literal).
HOSTED_LOSSES = frozenset({"cross_entropy", "importance_sampling", "ppo", "cispo", "dro"})

ALL_LOSSES = sorted(SERVER_LOSSES["fsdp"] | HOSTED_LOSSES)

TORCH_ONLY_LOSSES = frozenset({"gspo", "dppo", "ppo_critic"})


class LossBackendError(ValueError):
    """Raised when a loss_fn is not supported by the selected backend."""


def ensure_loss_supported(loss: str, backend: str | None) -> None:
    """Fail fast if ``loss`` can't run on ``backend``.

    ``backend=None`` means unknown (e.g. remote base_url we don't manage): only
    reject losses no backend supports, and let the server be the authority.
    """
    if backend is None:
        if loss not in set(ALL_LOSSES):
            raise LossBackendError(f"Unknown loss_fn {loss!r}. Known: {', '.join(ALL_LOSSES)}")
        return
    supported = SERVER_LOSSES.get(backend)
    if supported is None:
        raise LossBackendError(
            f"Unknown backend {backend!r}. Known: {', '.join(sorted(SERVER_LOSSES))}"
        )
    if loss not in supported:
        hint = ""
        if loss in TORCH_ONLY_LOSSES:
            hint = " Fused losses like 'gspo' need a torch backend: rerun serve with --backend fsdp or --backend megatron."
        raise LossBackendError(
            f"loss_fn {loss!r} is not supported by the {backend!r} backend "
            f"(supports: {', '.join(sorted(supported))}).{hint}"
        )


def as_loss_fn(loss: str) -> "LossFnType":
    """Pass an extended loss name through the SDK's Literal type.

    Runtime-safe: the SDK sends loss_fn as a plain string on the wire.
    """
    return cast("LossFnType", loss)
