"""Online policy for the learned IPM-trajectory GNN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
import torch

from .ipm_trajectory_pilot import (
    RoundedPlan,
    TELGENPaperGNN,
    build_ipm_graph,
    round_candidate_scores,
)
from .optimization_model import build_delay_model
from .time_expansion import TimeExpandedCandidate


@dataclass(frozen=True)
class OnlineIPMDecision:
    continuous_primal: np.ndarray
    selection: RoundedPlan
    inference_seconds: float
    invalid_action_index: int | None = None
    invalid_action_reason: str | None = None


def _resolve_device(name: str) -> torch.device:
    if name not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unknown GNN device: {name}")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def _candidate_loads(
    variable: TimeExpandedCandidate,
) -> dict[tuple[str, int], int]:
    loads: dict[tuple[str, int], int] = {}
    for entry in variable.resource_usage:
        key = (entry.resource_id, int(entry.slot))
        loads[key] = loads.get(key, 0) + int(entry.amount)
    return loads


def _individually_feasible_variables(
    variables: Sequence[TimeExpandedCandidate],
    capacities: Mapping[str, int],
    reserved_usage: Mapping[tuple[str, int], int],
) -> tuple[TimeExpandedCandidate, ...]:
    feasible: list[TimeExpandedCandidate] = []
    for variable in variables:
        loads = _candidate_loads(variable)
        if all(
            reserved_usage.get(key, 0) + amount <= capacities[key[0]]
            for key, amount in loads.items()
        ):
            feasible.append(variable)
    return tuple(sorted(feasible, key=lambda item: item.variable_id))


class OnlineIPMGNNPolicy:
    """Predict a request-normalized LP primal and round it to a plan."""

    def __init__(
        self,
        model: TELGENPaperGNN,
        *,
        inference_steps: int,
        device: torch.device,
        checkpoint_path: Path | None = None,
    ):
        if inference_steps < 1:
            raise ValueError("inference_steps must be positive")
        self.device = device
        self.model = model.to(device).eval()
        self.inference_steps = int(inference_steps)
        self.checkpoint_path = checkpoint_path

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "auto",
    ) -> "OnlineIPMGNNPolicy":
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
        if checkpoint.get("schema_version") != 4:
            raise ValueError(
                "checkpoint does not use the single-stage delay objective and "
                "must be retrained"
            )
        if checkpoint.get("model_class") != "TELGENPaperGNN":
            raise ValueError("checkpoint contains an unsupported model class")
        if checkpoint.get("objective") != "expected_censored_completion_latency":
            raise ValueError(
                "checkpoint is missing the single-stage delay objective"
            )
        config = checkpoint.get("model_config")
        if not isinstance(config, Mapping):
            raise ValueError("checkpoint is missing model configuration")
        model = TELGENPaperGNN(
            hidden_dim=int(config["hidden_dim"]),
            inner_layers=int(config["inner_layers"]),
            message_mlp_layers=int(config["message_mlp_layers"]),
            prediction_layers=int(config["prediction_layers"]),
            normalization=config.get("normalization"),
            dropout=float(config.get("dropout", 0.0)),
        )
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("checkpoint is missing model weights")
        model.load_state_dict(state_dict, strict=True)
        return cls(
            model,
            inference_steps=int(checkpoint["inference_steps"]),
            device=resolved_device,
            checkpoint_path=source,
        )

    def decide(
        self,
        variables: Sequence[TimeExpandedCandidate],
        resource_capacities: Mapping[str, int],
        *,
        horizon: int,
        reserved_usage: Mapping[tuple[str, int], int] | None = None,
        request_censoring_latencies: Mapping[str, float] | None = None,
    ) -> OnlineIPMDecision:
        started = perf_counter()
        capacities = {
            str(resource_id): int(capacity)
            for resource_id, capacity in resource_capacities.items()
        }
        reservations = {
            (str(resource_id), int(slot)): int(amount)
            for (resource_id, slot), amount in (reserved_usage or {}).items()
            if int(amount) != 0
        }
        feasible_variables = _individually_feasible_variables(
            variables, capacities, reservations
        )
        if not feasible_variables:
            selection = round_candidate_scores(
                np.zeros(0, dtype=np.float32),
                (),
                capacities,
                reserved_usage=reservations,
                request_censoring_latencies=request_censoring_latencies,
            )
            return OnlineIPMDecision(
                continuous_primal=np.zeros(0, dtype=np.float32),
                selection=selection,
                inference_seconds=perf_counter() - started,
            )
        model = build_delay_model(
            feasible_variables,
            capacities,
            reservations,
            request_censoring_latencies=request_censoring_latencies,
        )
        graph = build_ipm_graph(model, feasible_variables, horizon)
        with torch.no_grad():
            trace = self.model(graph, steps=self.inference_steps)
        primal = trace[-1].detach().cpu().numpy()
        rounded = round_candidate_scores(
            primal,
            feasible_variables,
            capacities,
            reserved_usage=reservations,
            request_censoring_latencies=request_censoring_latencies,
        )
        return OnlineIPMDecision(
            continuous_primal=np.asarray(primal, dtype=np.float32),
            selection=rounded,
            inference_seconds=perf_counter() - started,
        )


__all__ = ["OnlineIPMDecision", "OnlineIPMGNNPolicy"]
