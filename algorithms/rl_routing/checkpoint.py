"""Versioned ARC-Q checkpoint I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from .policy import ARCQPolicy


# The shared state encoder changes the policy state-dict layout. Refuse old
# checkpoints instead of mixing the previous two-encoder model with this one.
CHECKPOINT_SCHEMA_VERSION = 8
METHOD_NAME = "ARC-Q"


def save_arcq_checkpoint(
    path: str | Path,
    policy: ARCQPolicy,
    *,
    hidden_dim: int,
    message_passing_layers: int,
    training_state: Mapping[str, object],
    optimizer_state_dict: Mapping[str, object] | None = None,
    rng_state: Mapping[str, object] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save({
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "method": METHOD_NAME,
        "model": {
            "hidden_dim": int(hidden_dim),
            "message_passing_layers": int(message_passing_layers),
        },
        "policy_state_dict": policy.state_dict(),
        "training_state": dict(training_state),
        "optimizer_state_dict": (
            None
            if optimizer_state_dict is None
            else dict(optimizer_state_dict)
        ),
        "rng_state": None if rng_state is None else dict(rng_state),
    }, temporary)
    temporary.replace(target)
    return target


def load_arcq_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[ARCQPolicy, dict[str, object]]:
    payload = torch.load(
        Path(path),
        map_location=device,
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported ARC-Q checkpoint schema")
    if payload.get("method") != METHOD_NAME:
        raise ValueError("checkpoint belongs to another method")
    model_config = payload.get("model")
    state_dict = payload.get("policy_state_dict")
    if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
        raise ValueError("checkpoint is missing model state")
    policy = ARCQPolicy(
        hidden_dim=int(model_config["hidden_dim"]),
        message_passing_layers=int(
            model_config["message_passing_layers"]
        ),
    ).to(device)
    policy.load_state_dict(state_dict)
    metadata = {
        "schema_version": payload["schema_version"],
        "method": payload["method"],
        "model": dict(model_config),
        "training_state": dict(payload.get("training_state", {})),
        "optimizer_state_dict": payload.get("optimizer_state_dict"),
        "rng_state": payload.get("rng_state"),
    }
    return policy, metadata


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "METHOD_NAME",
    "load_arcq_checkpoint",
    "save_arcq_checkpoint",
]
