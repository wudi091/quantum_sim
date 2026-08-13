"""Online inference policy for the candidate--constraint GNN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np

from .milp_imitation import (
    CONSTRAINT_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    VARIABLE_FEATURE_NAMES,
    CandidateConstraintGNN,
    CandidateConstraintGraph,
    GreedyDecodeResult,
    batch_graph_samples,
    greedy_decode_scores,
    torch,
)


@dataclass(frozen=True)
class OnlineGNNDecision:
    """One GNN score vector and its feasible projected decision."""

    probabilities: np.ndarray
    decoded: GreedyDecodeResult
    decode_threshold: float
    support_variable_count: int
    inference_seconds: float


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
    """Load a trained checkpoint and score unlabelled planning graphs."""

    def __init__(
        self,
        model: CandidateConstraintGNN,
        *,
        device="cpu",
        decode_threshold: float = 0.0,
        checkpoint_path: str | Path | None = None,
    ):
        if torch is None:  # pragma: no cover - optional dependency environment
            raise ModuleNotFoundError(
                "PyTorch is required for the online GNN policy"
            )
        if not 0.0 <= float(decode_threshold) <= 1.0:
            raise ValueError("decode threshold must lie in [0, 1]")
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.decode_threshold = float(decode_threshold)
        self.checkpoint_path = (
            None if checkpoint_path is None else Path(checkpoint_path)
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "auto",
        decode_threshold: float | None = None,
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
        model.load_state_dict(state_dict, strict=True)
        resolved_threshold = (
            checkpoint.get("decode_threshold")
            if decode_threshold is None
            else decode_threshold
        )
        if resolved_threshold is None:
            raise ValueError("checkpoint is missing a decode threshold")
        return cls(
            model,
            device=resolved_device,
            decode_threshold=float(resolved_threshold),
            checkpoint_path=source,
        )

    def decide(self, graph: CandidateConstraintGraph) -> OnlineGNNDecision:
        started = perf_counter()
        batch = batch_graph_samples((graph,), device=self.device)
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model(batch)).cpu().numpy()
        decoded = greedy_decode_scores(
            graph,
            probabilities,
            threshold=self.decode_threshold,
        )
        if not decoded.feasible:
            raise RuntimeError("GNN hard projection returned an infeasible plan")
        return OnlineGNNDecision(
            probabilities=probabilities,
            decoded=decoded,
            decode_threshold=self.decode_threshold,
            support_variable_count=int(np.sum(
                probabilities >= self.decode_threshold
            )),
            inference_seconds=perf_counter() - started,
        )


__all__ = ["OnlineGNNDecision", "OnlineGNNPolicy"]
