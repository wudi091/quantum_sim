

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping

import torch

from .dataset import PlanningBatchProblem
from .ipm_trajectory_pilot import TELGENPaperGNN, build_ipm_graph
from .optimization_model import build_stage_one_model
from .packing import PackingSolution, decode_continuous_primal


@dataclass(frozen=True)
class IPMGNNDecision:
    """One continuous-primal inference followed by minimal decoding."""

    selection: PackingSolution
    inference_seconds: float


def _resolve_device(name: str) -> torch.device:
    if name not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unknown IPM GNN device: {name}")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


class IPMGNNDecoderPolicy:
    """Load a TELGENPaperGNN checkpoint and emit a decoded discrete plan."""

    def __init__(
        self,
        model: TELGENPaperGNN,
        *,
        steps: int = 16,
        device: str = "cpu",
    ):
        if steps < 1:
            raise ValueError("IPM steps must be positive")
        self.device = torch.device(device)
        self.steps = int(steps)
        self.model = model.to(self.device).eval()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        steps: int = 16,
        device: str = "auto",
    ) -> "IPMGNNDecoderPolicy":
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
            raise ValueError("IPM GNN checkpoint must contain a mapping")
        if checkpoint.get("model_class") != "TELGENPaperGNN":
            raise ValueError("checkpoint contains an unsupported model class")
        model_config = checkpoint.get("model_config")
        if not isinstance(model_config, Mapping):
            raise ValueError("checkpoint is missing model configuration")
        model = TELGENPaperGNN(
            hidden_dim=int(model_config["hidden_dim"]),
            inner_layers=int(model_config["inner_layers"]),
            message_mlp_layers=int(model_config["message_mlp_layers"]),
            prediction_layers=int(model_config["prediction_layers"]),
            normalization=model_config.get("normalization", "layer"),
            dropout=float(model_config.get("dropout", 0.0)),
        )
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("checkpoint is missing model weights")
        model.load_state_dict(state_dict, strict=True)
        return cls(model, steps=steps, device=resolved_device)

    def decide(self, problem: PlanningBatchProblem) -> IPMGNNDecision | None:
        """Infer one continuous primal and decode it into a discrete plan."""

        if problem is None or not problem.expansion.variables:
            return None
        started = perf_counter()
        variables = tuple(sorted(
            problem.expansion.variables,
            key=lambda item: item.variable_id,
        ))
        stage = build_stage_one_model(
            variables,
            problem.capacities,
            reserved_usage=problem.reserved_usage_map,
        )
        graph = build_ipm_graph(
            stage,
            variables,
            problem.episode.horizon,
        )
        with torch.no_grad():
            trace = self.model(graph, steps=self.steps).cpu().numpy()
        primal = trace[-1]
        selection = decode_continuous_primal(
            variables,
            primal,
            problem.capacities,
            reserved_usage=problem.reserved_usage_map,
        )
        return IPMGNNDecision(
            selection=selection,
            inference_seconds=perf_counter() - started,
        )


__all__ = ["IPMGNNDecision", "IPMGNNDecoderPolicy"]
