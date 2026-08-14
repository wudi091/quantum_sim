"""Q-CAST path ranking adapted to the shared resource--time plan contract.

This module contains planning math only.  It ranks fixed-construction path
candidates by Q-CAST expected throughput (EXT), then greedily packs complete
time-expanded plans without exceeding any opaque resource--slot capacity.
It never imports or accesses SeQUeNCe objects.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from algorithms.telgen.fidelity import candidate_fidelity_estimate_map
from algorithms.telgen.packing import (
    PackingSolution,
    validate_packing_selection,
)
from algorithms.telgen.time_expansion import (
    TimeExpandedCandidate,
    TimeExpansionResult,
    expand_construction_candidates,
    normalize_reserved_usage,
)
from qnet_core.construction_catalog import (
    RouteConstructionCandidate,
    build_route_construction_catalogue,
)
from qnet_core.resource_catalog import build_resource_capacities
from qnet_core.spec import EpisodeSpec, PhysicalConfig

from .expected_throughput import expected_throughput


@dataclass(frozen=True)
class QCASTPlanningRecord:
    """One online-window Q-CAST planning result."""

    candidates: tuple[RouteConstructionCandidate, ...]
    expansion: TimeExpansionResult
    candidate_scores: tuple[tuple[str, float], ...]
    solution: PackingSolution

    @property
    def score_by_candidate(self) -> dict[str, float]:
        return dict(self.candidate_scores)

    @property
    def selected_expected_throughput(self) -> float:
        scores = self.score_by_candidate
        return float(sum(
            scores[variable.candidate_id]
            for variable in self.solution.selected_variables
        ))


def effective_link_generation_probability(physical: PhysicalConfig) -> float:
    """Return the simulator-neutral success probability used by Q-CAST EXT.

    SeQUeNCe creates two half-distance quantum channels for each elementary
    link.  Their combined transmission is therefore
    ``10 ** (-attenuation * full_distance / 10)``.
    """

    transmission = 10.0 ** (
        -physical.quantum_attenuation_db_per_m
        * physical.quantum_distance_m
        / 10.0
    )
    probability = (
        physical.generation_probability
        * transmission
        * physical.detector_efficiency ** 2
        * physical.bsm_success_probability
    )
    return min(max(float(probability), 0.0), 1.0)


def qcast_path_score(
    episode: EpisodeSpec,
    route_nodes: tuple[int, ...],
) -> float:
    """Compute width-one Q-CAST EXT for one candidate route."""

    hop_count = len(route_nodes) - 1
    if hop_count < 1:
        return 0.0
    link_probability = effective_link_generation_probability(episode.physical)
    return expected_throughput(
        (link_probability,) * hop_count,
        width=1,
        swap_probability=episode.physical.swap_probability,
    )


def _memory_cost(variable: TimeExpandedCandidate) -> int:
    candidate = variable.base_candidate
    return 2 * candidate.hop_count * candidate.demand_pairs


def _greedy_qcast_solution(
    expansion: TimeExpansionResult,
    candidate_scores: Mapping[str, float],
    capacities: Mapping[str, int],
    request_ids: tuple[str, ...],
    reserved_usage: Mapping[tuple[str, int], int] | None,
) -> PackingSolution:
    variables = tuple(sorted(
        expansion.variables,
        key=lambda variable: variable.variable_id,
    ))
    maximum_score = max(candidate_scores.values(), default=0.0)
    normalized_scores = {
        variable.variable_id: (
            0.0
            if maximum_score <= 0.0
            else candidate_scores[variable.candidate_id] / maximum_score
        )
        for variable in variables
    }
    usage = normalize_reserved_usage(reserved_usage, capacities)
    selected: list[TimeExpandedCandidate] = []
    selected_requests: set[str] = set()
    ranked = sorted(variables, key=lambda variable: (
        -candidate_scores[variable.candidate_id],
        _memory_cost(variable),
        variable.base_candidate.hop_count,
        variable.completion_latency,
        variable.completion_slot,
        variable.variable_id,
    ))
    for variable in ranked:
        if candidate_scores[variable.candidate_id] <= 0.0:
            continue
        if variable.request_id in selected_requests:
            continue
        delta: dict[tuple[str, int], int] = {}
        for item in variable.resource_usage:
            key = (item.resource_id, item.slot)
            delta[key] = delta.get(key, 0) + item.amount
        if any(
            usage.get(key, 0) + amount > capacities[key[0]]
            for key, amount in delta.items()
        ):
            continue
        selected.append(variable)
        selected_requests.add(variable.request_id)
        for key, amount in delta.items():
            usage[key] = usage.get(key, 0) + amount

    selected_variables = tuple(sorted(
        selected,
        key=lambda variable: variable.variable_id,
    ))
    feasibility = validate_packing_selection(
        selected_variables,
        capacities,
        reserved_usage,
    )
    if not feasibility.feasible:
        raise RuntimeError(
            "Q-CAST greedy packing produced an infeasible selection: "
            + feasibility.violations[0]
        )
    return PackingSolution(
        variables=variables,
        scores=tuple(normalized_scores[variable.variable_id] for variable in variables),
        request_ids=request_ids,
        selected_variables=selected_variables,
        feasibility=feasibility,
        strategy="qcast_ext_greedy",
    )


def plan_qcast_window(
    episode: EpisodeSpec,
    *,
    window_start_slot: int,
    window_end_slot: int,
    completion_end_slot: int | None = None,
    request_ids: tuple[str, ...] | None = None,
    resource_capacities: Mapping[str, int] | None = None,
    reserved_usage: Mapping[tuple[str, int], int] | None = None,
    path_candidate_count: int = 3,
    construction_kind: str = "left_deep",
    purification_kind: str = "none",
) -> QCASTPlanningRecord:
    """Select fixed-construction paths with Q-CAST EXT and greedy packing."""

    if path_candidate_count < 1:
        raise ValueError("path_candidate_count must be positive")
    declared_request_ids = (
        tuple(request.id for request in episode.requests)
        if request_ids is None
        else tuple(request_ids)
    )
    if len(set(declared_request_ids)) != len(declared_request_ids):
        raise ValueError("request_ids must be unique")
    episode_request_ids = {request.id for request in episode.requests}
    unknown = set(declared_request_ids) - episode_request_ids
    if unknown:
        raise ValueError(f"unknown request: {sorted(unknown)[0]}")
    declared = set(declared_request_ids)
    planning_episode = replace(
        episode,
        requests=tuple(
            request
            for request in episode.requests
            if request.id in declared
        ),
    )
    candidates = build_route_construction_catalogue(
        planning_episode.planning,
        candidate_count=path_candidate_count,
        construction_kinds=(construction_kind,),
        purification_kinds=(purification_kind,),
    )
    fidelity_estimates = candidate_fidelity_estimate_map(
        planning_episode,
        candidates,
    )
    capacities = (
        build_resource_capacities(episode)
        if resource_capacities is None
        else {str(key): int(value) for key, value in resource_capacities.items()}
    )
    expansion = expand_construction_candidates(
        planning_episode.planning,
        candidates,
        capacities,
        fidelity_estimates=fidelity_estimates,
        reserved_usage=reserved_usage,
        window_start_slot=window_start_slot,
        window_end_slot=window_end_slot,
        completion_end_slot=completion_end_slot,
    )
    candidate_scores = tuple(sorted(
        (
            candidate.candidate_id,
            qcast_path_score(planning_episode, candidate.route_nodes),
        )
        for candidate in candidates
    ))
    solution = _greedy_qcast_solution(
        expansion,
        dict(candidate_scores),
        capacities,
        declared_request_ids,
        reserved_usage,
    )
    return QCASTPlanningRecord(
        candidates=candidates,
        expansion=expansion,
        candidate_scores=candidate_scores,
        solution=solution,
    )


__all__ = [
    "QCASTPlanningRecord",
    "effective_link_generation_probability",
    "plan_qcast_window",
    "qcast_path_score",
]
