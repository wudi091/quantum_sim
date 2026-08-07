"""Versioned checkpoints for reproducible CAAPPO training."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
import os
from pathlib import Path
import platform
import tempfile
from typing import Mapping, Sequence

import torch
from torch import Tensor

from .torch_policy import TorchCAAPPOPolicy
from .torch_trainer import TorchCAAPPORolloutTrainer


CHECKPOINT_FORMAT = "qnet.caappo.training"
CHECKPOINT_SCHEMA_VERSION = 2
IMPLEMENTATION_CONTRACT_VERSION = 1
_RUNTIME_PACKAGES = ("sequence", "numpy", "networkx", "torch")


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint cannot reproduce the requested run."""


@dataclass(frozen=True)
class LoadedCAAPPOCheckpoint:
    policy: TorchCAAPPOPolicy
    trainer: TorchCAAPPORolloutTrainer
    training_metadata: dict[str, object]
    completed_episodes: int
    history: tuple[dict[str, object], ...]
    best_validation: dict[str, object] | None
    best_policy_state_dict: dict[str, Tensor] | None
    best_optimizer_state_dict: dict[str, object] | None
    best_lambda_risk: float | None
    runtime: dict[str, object]


def runtime_manifest() -> dict[str, object]:
    packages: dict[str, str] = {}
    for name in _RUNTIME_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": bool(torch.cuda.is_available()),
        "packages": packages,
    }


def checkpoint_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_config(policy: TorchCAAPPOPolicy) -> dict[str, object]:
    return {
        "hidden_dim": policy.hidden_dim,
        "learning_rate": policy.learning_rate,
        "use_dag_state": policy.use_dag_state,
        "use_capacity_context": policy.use_capacity_context,
    }


def _trainer_config(trainer: TorchCAAPPORolloutTrainer) -> dict[str, object]:
    return {
        "risk_limit": trainer.risk_limit,
        "gamma_per_slot": trainer.gamma_per_slot,
        "gae_lambda": trainer.gae_lambda,
        "alpha": trainer.alpha,
        "beta": trainer.beta,
        "chi": trainer.chi,
        "potential_shaping": trainer.potential_shaping,
        "use_route_overlap_context": trainer.use_route_overlap_context,
        "shaping_coef": trainer.shaping_coef,
        "dynamic_repair_paths": trainer.dynamic_repair_paths,
        "dynamic_repair_construction_kinds": (
            trainer.dynamic_repair_construction_kinds
        ),
    }


def _mismatches(
    expected: object,
    actual: object,
    path: str,
) -> list[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        result: list[str] = []
        for key, expected_value in expected.items():
            child = f"{path}.{key}" if path else str(key)
            if key not in actual:
                result.append(f"{child}: missing")
            else:
                result.extend(_mismatches(expected_value, actual[key], child))
        return result
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)):
            return [f"{path}: expected a sequence"]
        if len(expected) != len(actual):
            return [f"{path}: expected length {len(expected)}, got {len(actual)}"]
        result = []
        for index, (expected_value, actual_value) in enumerate(zip(expected, actual)):
            result.extend(_mismatches(
                expected_value, actual_value, f"{path}[{index}]"
            ))
        return result
    if expected != actual:
        return [f"{path}: expected {expected!r}, got {actual!r}"]
    return []


def _validate_runtime(saved: Mapping[str, object]) -> None:
    current = runtime_manifest()
    mismatches = _mismatches(
        {
            "python": current["python"],
            "platform": current["platform"],
            "cuda_available": current["cuda_available"],
            "packages": current["packages"],
        },
        saved,
        "runtime",
    )
    if mismatches:
        raise CheckpointCompatibilityError(
            "checkpoint runtime mismatch: " + "; ".join(mismatches)
        )


def save_caappo_checkpoint(
    path: Path,
    *,
    policy: TorchCAAPPOPolicy,
    trainer: TorchCAAPPORolloutTrainer,
    training_metadata: Mapping[str, object],
    completed_episodes: int,
    history: Sequence[Mapping[str, object]],
    best_validation: Mapping[str, object] | None = None,
    best_policy_state_dict: Mapping[str, Tensor] | None = None,
    best_optimizer_state_dict: Mapping[str, object] | None = None,
    best_lambda_risk: float | None = None,
) -> Path:
    if completed_episodes < 0:
        raise ValueError("completed_episodes cannot be negative")
    payload = {
        "format": CHECKPOINT_FORMAT,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "implementation_contract_version": IMPLEMENTATION_CONTRACT_VERSION,
        "runtime": runtime_manifest(),
        "policy_config": _policy_config(policy),
        "trainer_config": _trainer_config(trainer),
        "policy_state_dict": deepcopy(policy.state_dict()),
        "optimizer_state_dict": deepcopy(policy.optimizer.state_dict()),
        "lambda_risk": float(policy.lambda_risk),
        "training_metadata": dict(training_metadata),
        "completed_episodes": int(completed_episodes),
        "history": tuple(dict(row) for row in history),
        "best_validation": (
            None if best_validation is None else dict(best_validation)
        ),
        "best_policy_state_dict": (
            None
            if best_policy_state_dict is None
            else deepcopy(dict(best_policy_state_dict))
        ),
        "best_optimizer_state_dict": (
            None
            if best_optimizer_state_dict is None
            else deepcopy(dict(best_optimizer_state_dict))
        ),
        "best_lambda_risk": (
            None if best_lambda_risk is None else float(best_lambda_risk)
        ),
        "rng_state": {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                tuple(torch.cuda.get_rng_state_all())
                if torch.cuda.is_available()
                else None
            ),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        with temporary_path.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


def load_caappo_checkpoint(
    path: Path,
    *,
    device: str | torch.device = "cpu",
    expected_training_metadata: Mapping[str, object] | None = None,
    strict_runtime: bool = True,
    restore_rng: bool = False,
    use_best: bool = False,
    expected_sha256: str | None = None,
) -> LoadedCAAPPOCheckpoint:
    if expected_sha256 is not None:
        actual_sha256 = checkpoint_sha256(path)
        if actual_sha256.lower() != expected_sha256.lower():
            raise CheckpointCompatibilityError(
                "checkpoint SHA-256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise CheckpointCompatibilityError("checkpoint payload must be a mapping")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointCompatibilityError("unrecognized checkpoint format")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointCompatibilityError(
            "unsupported checkpoint schema version: "
            f"{payload.get('schema_version')!r}"
        )
    if (
        payload.get("implementation_contract_version")
        != IMPLEMENTATION_CONTRACT_VERSION
    ):
        raise CheckpointCompatibilityError(
            "checkpoint implementation contract is incompatible"
        )
    saved_runtime = payload.get("runtime")
    if not isinstance(saved_runtime, Mapping):
        raise CheckpointCompatibilityError("checkpoint runtime manifest is missing")
    if strict_runtime:
        _validate_runtime(saved_runtime)

    training_metadata = payload.get("training_metadata")
    if not isinstance(training_metadata, dict):
        raise CheckpointCompatibilityError("checkpoint training metadata is missing")
    if expected_training_metadata is not None:
        mismatches = _mismatches(
            expected_training_metadata,
            training_metadata,
            "training",
        )
        if mismatches:
            raise CheckpointCompatibilityError(
                "checkpoint training configuration mismatch: "
                + "; ".join(mismatches)
            )

    policy_config = payload.get("policy_config")
    trainer_config = payload.get("trainer_config")
    if not isinstance(policy_config, dict) or not isinstance(trainer_config, dict):
        raise CheckpointCompatibilityError("checkpoint model configuration is missing")
    caller_rng = torch.get_rng_state()
    caller_cuda_rng = (
        tuple(torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else None
    )
    policy = TorchCAAPPOPolicy(
        **policy_config,
        seed=int(training_metadata.get("training_seed", 0)),
        device=device,
    )
    final_state = payload.get("policy_state_dict")
    best_state = payload.get("best_policy_state_dict")
    selected_state = best_state if use_best and best_state is not None else final_state
    if not isinstance(selected_state, dict):
        raise CheckpointCompatibilityError("checkpoint policy state is missing")
    policy.load_state_dict(selected_state, strict=True)
    final_optimizer_state = payload.get("optimizer_state_dict")
    best_optimizer_state = payload.get("best_optimizer_state_dict")
    optimizer_state = (
        best_optimizer_state
        if use_best and best_state is not None and best_optimizer_state is not None
        else final_optimizer_state
    )
    if not isinstance(optimizer_state, dict):
        raise CheckpointCompatibilityError("checkpoint optimizer state is missing")
    policy.optimizer.load_state_dict(optimizer_state)
    selected_lambda = (
        payload.get("best_lambda_risk")
        if use_best and best_state is not None
        else payload.get("lambda_risk", 0.0)
    )
    policy.lambda_risk = float(
        0.0 if selected_lambda is None else selected_lambda
    )
    trainer = TorchCAAPPORolloutTrainer(policy, **trainer_config)

    rng_state = payload.get("rng_state")
    if restore_rng:
        if not isinstance(rng_state, dict) or not isinstance(
            rng_state.get("torch_cpu"), Tensor
        ):
            raise CheckpointCompatibilityError("checkpoint RNG state is missing")
        torch.set_rng_state(rng_state["torch_cpu"].cpu())
        cuda_state = rng_state.get("torch_cuda")
        if cuda_state is not None:
            if not torch.cuda.is_available():
                raise CheckpointCompatibilityError(
                    "checkpoint contains CUDA RNG state but CUDA is unavailable"
                )
            torch.cuda.set_rng_state_all(tuple(cuda_state))
    else:
        torch.set_rng_state(caller_rng)
        if caller_cuda_rng is not None:
            torch.cuda.set_rng_state_all(caller_cuda_rng)

    history = payload.get("history", ())
    if not isinstance(history, (tuple, list)) or not all(
        isinstance(row, dict) for row in history
    ):
        raise CheckpointCompatibilityError("checkpoint history is malformed")
    best_validation = payload.get("best_validation")
    if best_validation is not None and not isinstance(best_validation, dict):
        raise CheckpointCompatibilityError("checkpoint best-validation record is malformed")
    best_optimizer_state = payload.get("best_optimizer_state_dict")
    if best_optimizer_state is not None and not isinstance(best_optimizer_state, dict):
        raise CheckpointCompatibilityError("checkpoint best optimizer state is malformed")
    best_lambda_risk = payload.get("best_lambda_risk")
    if best_lambda_risk is not None:
        best_lambda_risk = float(best_lambda_risk)
    best_snapshot_present = any(value is not None for value in (
        best_validation,
        best_state,
        best_optimizer_state,
        best_lambda_risk,
    ))
    best_snapshot_complete = all(value is not None for value in (
        best_validation,
        best_state,
        best_optimizer_state,
        best_lambda_risk,
    ))
    if best_snapshot_present and not best_snapshot_complete:
        raise CheckpointCompatibilityError(
            "best validation, policy, optimizer, and CMDP dual snapshots "
            "must be stored together"
        )
    completed_episodes = int(payload.get("completed_episodes", -1))
    if completed_episodes < 0:
        raise CheckpointCompatibilityError("checkpoint episode counter is malformed")
    training_event_count = sum(
        row.get("event") == "training_episode" for row in history
    )
    if training_event_count != completed_episodes:
        raise CheckpointCompatibilityError(
            "checkpoint history does not match completed episode counter"
        )
    return LoadedCAAPPOCheckpoint(
        policy,
        trainer,
        dict(training_metadata),
        completed_episodes,
        tuple(dict(row) for row in history),
        None if best_validation is None else dict(best_validation),
        None if best_state is None else dict(best_state),
        None if best_optimizer_state is None else dict(best_optimizer_state),
        best_lambda_risk,
        dict(saved_runtime),
    )


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointCompatibilityError",
    "LoadedCAAPPOCheckpoint",
    "checkpoint_sha256",
    "load_caappo_checkpoint",
    "runtime_manifest",
    "save_caappo_checkpoint",
]
