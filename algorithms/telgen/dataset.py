"""Build simulator-neutral candidate expansions for MILP and GNN planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from qnet_core.construction_catalog import (
    RouteConstructionCandidate,
    build_route_construction_catalogue,
)
from qnet_core.resource_catalog import build_resource_capacities
from qnet_core.spec import EpisodeSpec

from .fidelity import FIDELITY_MODEL_NAME, candidate_fidelity_estimate_map
from .success_probability import (
    SUCCESS_PROBABILITY_MODEL_NAME,
    candidate_success_probability_map,
)
from .time_expansion import (
    TimeExpansionResult,
    build_nominal_schedule,
    expand_construction_candidates,
)


@dataclass(frozen=True)
class PlanningBatchProblem:
    """One candidate--constraint problem before choosing MILP or GNN."""

    episode: EpisodeSpec
    path_candidate_count: int
    construction_kinds: tuple[str, ...]
    swap_tree_count: int | None
    purification_kinds: tuple[str, ...]
    resource_capacities: tuple[tuple[str, int], ...]
    candidates: tuple[RouteConstructionCandidate, ...]
    expansion: TimeExpansionResult
    fidelity_model: str
    success_probability_model: str
    planning_window: tuple[int, int] | None = None
    completion_end_slot: int | None = None
    request_censoring_latencies: tuple[tuple[str, float], ...] = ()
    reserved_usage: tuple[tuple[str, int, int], ...] = ()
    equivalent_candidate_aliases: tuple[tuple[str, str], ...] = ()

    @property
    def capacities(self) -> dict[str, int]:
        return dict(self.resource_capacities)

    @property
    def reserved_usage_map(self) -> dict[tuple[str, int], int]:
        return {
            (resource_id, slot): amount
            for resource_id, slot, amount in self.reserved_usage
        }

    @property
    def request_censoring_latency_map(self) -> dict[str, float]:
        """Return the delay charged when a request is not completed.

        The values are measured from each request's arrival slot and are
        capped by the episode horizon and, when present, its TTL deadline.
        Keeping this catalogue beside the candidate expansion lets the LP,
        exact MILP, and IPM-GNN use exactly the same single objective.
        """

        return {
            request_id: float(latency)
            for request_id, latency in self.request_censoring_latencies
        }


def _candidate_dag_semantics(
    candidate: RouteConstructionCandidate,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Canonicalize DAG dataflow while ignoring IDs and redundant edges."""

    operations = candidate.dag.operations
    ordinal_counts: dict[int, int] = {}
    for operation in operations:
        ordinal_counts[operation.ordinal] = (
            ordinal_counts.get(operation.ordinal, 0) + 1
        )
    operation_tokens = {
        operation.op_id: (
            "ordinal",
            operation.ordinal,
        ) if ordinal_counts[operation.ordinal] == 1 else (
            "ambiguous_ordinal",
            operation.ordinal,
            operation.op_id,
        )
        for operation in operations
    }
    producer_by_segment = {
        operation.output_segment_id: operation.op_id
        for operation in operations
        if operation.output_segment_id is not None
    }

    direct_dependencies: dict[str, set[str]] = {}
    for operation in operations:
        dependencies = set(operation.predecessors)
        dependencies.update(
            producer_by_segment[segment_id]
            for segment_id in operation.input_segment_ids
            if segment_id in producer_by_segment
        )
        direct_dependencies[operation.op_id] = dependencies

    ancestor_cache: dict[str, set[str]] = {}

    def ancestors(operation_id: str) -> set[str]:
        cached = ancestor_cache.get(operation_id)
        if cached is not None:
            return cached
        result: set[str] = set()
        for predecessor in direct_dependencies[operation_id]:
            result.add(predecessor)
            result.update(ancestors(predecessor))
        ancestor_cache[operation_id] = result
        return result

    def reduced_dependencies(operation_id: str) -> tuple[object, ...]:
        dependencies = direct_dependencies[operation_id]
        reduced = {
            predecessor
            for predecessor in dependencies
            if not any(
                predecessor in ancestors(other)
                for other in dependencies
                if other != predecessor
            )
        }
        return tuple(sorted(operation_tokens[item] for item in reduced))

    def segment_token(segment_id: str) -> tuple[object, ...]:
        producer = producer_by_segment.get(segment_id)
        if producer is None:
            return ("external_segment", segment_id)
        return ("produced_segment", operation_tokens[producer])

    operation_semantics = tuple(sorted(
        (
            operation_tokens[operation.op_id],
            operation.kind,
            reduced_dependencies(operation.op_id),
            tuple(
                segment_token(segment_id)
                for segment_id in operation.input_segment_ids
            ),
            operation.output_segment_id is not None,
            operation.output_endpoints,
            operation.resource_demand.entries,
            operation.output_resource_hold.entries,
            operation.duration_ps,
            float(operation.success_probability).hex(),
            float(operation.required_fidelity).hex(),
            operation.retry_limit,
            (
                None
                if operation.retry_root_id is None
                else operation_tokens.get(
                    operation.retry_root_id,
                    ("external_operation", operation.retry_root_id),
                )
            ),
            operation.retry_attempt,
            operation.dag_version,
        )
        for operation in operations
    ))
    terminal_semantics = tuple(
        segment_token(segment_id)
        for segment_id in candidate.all_terminal_segment_ids
    )
    return operation_semantics, terminal_semantics


def _canonicalize_planning_equivalent_candidates(
    candidates: tuple[RouteConstructionCandidate, ...],
    fidelity_estimates: Mapping[str, float],
    success_probability_estimates: Mapping[str, float],
) -> tuple[
    tuple[RouteConstructionCandidate, ...],
    tuple[tuple[str, str], ...],
]:
    """Remove candidates that induce the same neutral planning action."""

    representatives: dict[tuple[object, ...], RouteConstructionCandidate] = {}
    aliases: list[tuple[str, str]] = []
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        schedule = build_nominal_schedule(candidate)
        operation_semantics, terminal_semantics = _candidate_dag_semantics(
            candidate
        )
        key = (
            candidate.request_id,
            candidate.route_nodes,
            candidate.purification_kind,
            candidate.demand_pairs,
            candidate.dag.version,
            operation_semantics,
            terminal_semantics,
            schedule.duration_slots,
            schedule.resource_usage,
            float(fidelity_estimates[candidate.candidate_id]).hex(),
            float(success_probability_estimates[candidate.candidate_id]).hex(),
        )
        representative = representatives.get(key)
        if representative is None:
            representatives[key] = candidate
        else:
            aliases.append((candidate.candidate_id, representative.candidate_id))
    return (
        tuple(sorted(
            representatives.values(),
            key=lambda item: item.candidate_id,
        )),
        tuple(sorted(aliases)),
    )


def build_planning_batch_problem(
    episode: EpisodeSpec,
    *,
    path_candidate_count: int = 3,
    construction_kinds: tuple[str, ...] = ("left_deep", "balanced"),
    swap_tree_count: int | None = None,
    purification_kinds: tuple[str, ...] = ("none", "elementary_once"),
    fidelity_estimates: Mapping[str, float] | None = None,
    resource_capacities: Mapping[str, int] | None = None,
    reserved_usage: Mapping[tuple[str, int], int] | None = None,
    window_start_slot: int | None = None,
    window_end_slot: int | None = None,
    completion_end_slot: int | None = None,
) -> PlanningBatchProblem:
    """Build the candidate expansion shared by exact MILP and online GNN."""

    if path_candidate_count < 1:
        raise ValueError("path_candidate_count must be positive")
    if not construction_kinds and swap_tree_count is None:
        raise ValueError("at least one construction policy is required")
    if swap_tree_count is not None and swap_tree_count < 1:
        raise ValueError("swap_tree_count must be positive")
    if not purification_kinds:
        raise ValueError("at least one purification kind is required")

    online_window = window_start_slot is not None or window_end_slot is not None
    if online_window:
        window_start = 0 if window_start_slot is None else int(window_start_slot)
        window_end = (
            episode.horizon
            if window_end_slot is None
            else int(window_end_slot)
        )
        if not 0 <= window_start < window_end <= episode.horizon:
            raise ValueError("planning window must lie inside the episode horizon")
        completion_end = (
            window_end
            if completion_end_slot is None
            else int(completion_end_slot)
        )
        if not window_end <= completion_end <= episode.horizon:
            raise ValueError(
                "completion boundary must follow the planning window"
            )
        future = [
            request.id
            for request in episode.requests
            if request.arrival > window_start
        ]
        if future:
            raise ValueError(
                f"online planning window contains a future request: {future[0]}"
            )
        planning_window = (window_start, window_end)
    else:
        if completion_end_slot is not None:
            raise ValueError(
                "completion_end_slot requires a planning window"
            )
        if any(request.arrival != 0 for request in episode.requests):
            raise ValueError("static planning episodes require arrival slot zero")
        planning_window = None
        completion_end = episode.horizon

    # The single LP objective charges every request that is not completed by
    # its censoring boundary.  Use the same boundary that candidate expansion
    # uses, so static and rolling-horizon calls share identical semantics.
    request_censoring_latencies: dict[str, float] = {}
    for request in episode.requests:
        boundary = int(completion_end)
        if request.deadline is not None:
            boundary = min(boundary, int(request.deadline))
        latency = boundary - int(request.arrival)
        if latency < 0:
            raise ValueError(
                f"request {request.id} is already past its censoring boundary"
            )
        request_censoring_latencies[request.id] = float(latency)

    capacities = (
        build_resource_capacities(episode)
        if resource_capacities is None
        else {
            str(resource_id): int(capacity)
            for resource_id, capacity in resource_capacities.items()
        }
    )
    raw_candidates = build_route_construction_catalogue(
        episode.planning,
        candidate_count=path_candidate_count,
        construction_kinds=construction_kinds,
        purification_kinds=purification_kinds,
        swap_tree_count=swap_tree_count,
    )
    resolved_fidelity_estimates = fidelity_estimates
    fidelity_model = "provided"
    if resolved_fidelity_estimates is None:
        resolved_fidelity_estimates = candidate_fidelity_estimate_map(
            episode,
            raw_candidates,
        )
        fidelity_model = FIDELITY_MODEL_NAME
    else:
        resolved_fidelity_estimates = {
            str(candidate_id): float(value)
            for candidate_id, value in resolved_fidelity_estimates.items()
        }
    missing_fidelity = {
        candidate.candidate_id for candidate in raw_candidates
    } - set(resolved_fidelity_estimates)
    if missing_fidelity:
        raise ValueError(
            f"missing fidelity estimate: {sorted(missing_fidelity)[0]}"
        )

    success_probability_estimates = candidate_success_probability_map(
        episode,
        raw_candidates,
    )
    candidates, equivalent_aliases = (
        _canonicalize_planning_equivalent_candidates(
            raw_candidates,
            resolved_fidelity_estimates,
            success_probability_estimates,
        )
    )
    expansion = expand_construction_candidates(
        episode.planning,
        candidates,
        capacities,
        fidelity_estimates=resolved_fidelity_estimates,
        success_probability_estimates=success_probability_estimates,
        reserved_usage=reserved_usage,
        window_start_slot=(
            None if planning_window is None else planning_window[0]
        ),
        window_end_slot=(
            None if planning_window is None else planning_window[1]
        ),
        completion_end_slot=(
            None if planning_window is None else completion_end
        ),
    )
    return PlanningBatchProblem(
        episode=episode,
        path_candidate_count=path_candidate_count,
        construction_kinds=tuple(construction_kinds),
        swap_tree_count=swap_tree_count,
        purification_kinds=tuple(purification_kinds),
        resource_capacities=tuple(sorted(capacities.items())),
        candidates=candidates,
        expansion=expansion,
        fidelity_model=fidelity_model,
        success_probability_model=SUCCESS_PROBABILITY_MODEL_NAME,
        planning_window=planning_window,
        completion_end_slot=(
            None if planning_window is None else completion_end
        ),
        request_censoring_latencies=tuple(sorted(
            request_censoring_latencies.items()
        )),
        reserved_usage=tuple(sorted(
            (str(resource_id), int(slot), int(amount))
            for (resource_id, slot), amount in (reserved_usage or {}).items()
            if int(amount) != 0
        )),
        equivalent_candidate_aliases=equivalent_aliases,
    )


__all__ = ["PlanningBatchProblem", "build_planning_batch_problem"]
