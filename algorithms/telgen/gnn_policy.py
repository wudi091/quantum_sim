"""Online inference policy for the candidate--constraint GNN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np

from .milp_imitation import (
    AUTOREGRESSIVE_ARCHITECTURE,
    AUTOREGRESSIVE_CHECKPOINT_SCHEMA_VERSION,
    CONSTRAINT_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    VARIABLE_FEATURE_NAMES,
    AutoregressiveSelection,
    CandidateConstraintGNN,
    CandidateConstraintGraph,
    autoregressive_rollout,
    torch,
)


@dataclass(frozen=True)
class OnlineGNNDecision:
    """One feasibility-masked discrete action sequence emitted by the GNN."""

    probabilities: np.ndarray
    stop_probability: float
    selection: AutoregressiveSelection
    action_indices: tuple[int, ...]
    inference_seconds: float
    invalid_action_index: int | None = None
    invalid_action_reason: str | None = None


def _resolve_device(name: str):
    if torch is None:  # pragma: no cover - optional dependency environment
        raise ModuleNotFoundError(
            "PyTorch is required for the online GNN policy"
        )
    if name not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unknown GNN device: {name}")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


class OnlineGNNPolicy:
    """Load a checkpoint and emit a feasible discrete action sequence."""

    def __init__(
        self,
        model: CandidateConstraintGNN,
        *,
        device="cpu",
        checkpoint_path: str | Path | None = None,
    ):
        if torch is None:  # pragma: no cover - optional dependency environment
            raise ModuleNotFoundError(
                "PyTorch is required for the online GNN policy"
            )
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.checkpoint_path = (
            None if checkpoint_path is None else Path(checkpoint_path)
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "auto",
    ) -> "OnlineGNNPolicy":
        if torch is None:  # pragma: no cover - optional dependency environment
            raise ModuleNotFoundError(
                "PyTorch is required for the online GNN policy"
            )
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        resolved_device = _resolve_device(device)
        checkpoint = torch.load(
            source,
            map_location=resolved_device,
            weights_only=True,
        )
        if not isinstance(checkpoint, Mapping):
            raise ValueError("GNN checkpoint must contain a mapping")
        if checkpoint.get("model_class") != "CandidateConstraintGNN":
            raise ValueError("checkpoint contains an unsupported model class")
        if checkpoint.get("schema_version") != (
            AUTOREGRESSIVE_CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError(
                "checkpoint predates current masked autoregressive inference and "
                "must be retrained"
            )
        architecture = checkpoint.get("architecture")
        if architecture != AUTOREGRESSIVE_ARCHITECTURE:
            raise ValueError(
                "checkpoint uses an unsupported autoregressive architecture and "
                "must be retrained"
            )
        expected_schema = {
            "variable": list(VARIABLE_FEATURE_NAMES),
            "constraint": list(CONSTRAINT_FEATURE_NAMES),
            "global": list(GLOBAL_FEATURE_NAMES),
        }
        if checkpoint.get("feature_schema") != expected_schema:
            raise ValueError("checkpoint feature schema does not match runtime")
        raw_model_config = checkpoint.get("model_config")
        if not isinstance(raw_model_config, Mapping):
            raise ValueError("checkpoint is missing model configuration")
        try:
            hidden_dim = int(raw_model_config["hidden_dim"])
            layers = int(raw_model_config["layers"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("checkpoint model configuration is invalid") from error
        model = CandidateConstraintGNN(
            hidden_dim=hidden_dim,
            layers=layers,
        )
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("checkpoint is missing model weights")
        if not any(
            str(key).startswith("candidate_action_head.")
            for key in state_dict
        ):
            raise ValueError(
                "checkpoint predates the autoregressive action head and must "
                "be retrained"
            )
        model.load_state_dict(state_dict, strict=True)
        return cls(
            model,
            device=resolved_device,
            checkpoint_path=source,
        )

    def decide(self, graph: CandidateConstraintGraph) -> OnlineGNNDecision:
        started = perf_counter()
        rollout = autoregressive_rollout(
            self.model,
            graph,
            device=self.device,
        )
        return OnlineGNNDecision(
            probabilities=rollout.initial_candidate_probabilities,
            stop_probability=rollout.initial_stop_probability,
            selection=rollout.selection,
            action_indices=rollout.action_indices,
            inference_seconds=perf_counter() - started,
            invalid_action_index=rollout.invalid_action_index,
            invalid_action_reason=rollout.invalid_action_reason,
        )


__all__ = ["OnlineGNNDecision", "OnlineGNNPolicy"]
