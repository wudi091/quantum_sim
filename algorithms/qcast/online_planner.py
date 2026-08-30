"""Residual-resource Q-CAST planning on the neutral construction contract.

The implementation follows the official Q-CAST phase-2 structure within the
project's frozen width-one setting: recompute one best EXT path per request,
globally reserve the strongest path, then reserve subpath recovery routes from
the remaining resources.  SeQUeNCe objects never cross this module boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import networkx as nx

from algorithms.telgen.fidelity import candidate_fidelity_estimate_map
from algorithms.telgen.packing import PackingSolution, validate_packing_selection
from algorithms.telgen.time_expansion import (
    CandidateRejection,
    NominalConstructionSchedule,
    TimeExpandedCandidate,
    TimeExpansionResult,
    expand_construction_candidates,
    normalize_reserved_usage,
)
from qnet_core.construction_api import (
    ConstructionDAG,
    OperationKind,
)
from qnet_core.construction_catalog import RouteConstructionCandidate
from qnet_core.construction_plans import left_deep_path_dag
from qnet_core.resource_catalog import build_resource_capacities
from qnet_core.spec import EpisodeSpec, PhysicalConfig

from .expected_throughput import expected_throughput


@dataclass(frozen=True)
class QCASTRecoveryPathPlan:
    """One pre-reserved recovery path for a contiguous major-path interval."""

    recovery_id: str
    major_start_index: int
    major_end_index: int
    route_nodes: tuple[int, ...]
    generation_operation_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.recovery_id:
            raise ValueError("recovery_id must be non-empty")
        if not 0 <= self.major_start_index < self.major_end_index:
            raise ValueError("invalid recovery interval")
        if len(self.route_nodes) < 2:
            raise ValueError("recovery route must contain an edge")
        if len(self.segment_ids) != len(self.route_nodes) - 1:
            raise ValueError("recovery route needs one segment per edge")
        if len(self.generation_operation_ids) != len(self.segment_ids):
            raise ValueError("recovery generation metadata is inconsistent")

    @property
    def covered_major_edges(self) -> frozenset[int]:
        return frozenset(range(self.major_start_index, self.major_end_index))


@dataclass(frozen=True)
class QCASTAllocation:
    """One selected major path and all recovery resources reserved for it."""

    candidate: RouteConstructionCandidate
    expected_throughput: float
    width: int
    major_generation_operation_ids: tuple[str, ...]
    major_segment_ids: tuple[str, ...]
    recovery_paths: tuple[QCASTRecoveryPathPlan, ...] = ()

    def __post_init__(self) -> None:
        if self.width != 1:
            raise ValueError("the shared experiment contract freezes Q-CAST width to one")
        if self.expected_throughput < 0.0:
            raise ValueError("expected_throughput cannot be negative")
        if len(self.major_segment_ids) != self.candidate.hop_count:
            raise ValueError("major path needs one elementary segment per edge")
        if len(self.major_generation_operation_ids) != len(self.major_segment_ids):
            raise ValueError("major generation metadata is inconsistent")
        major_route = self.candidate.route_nodes
        for recovery in self.recovery_paths:
            if recovery.major_end_index >= len(major_route):
                raise ValueError("recovery interval lies outside the major path")
            if (
                recovery.route_nodes[0]
                != major_route[recovery.major_start_index]
                or recovery.route_nodes[-1]
                != major_route[recovery.major_end_index]
            ):
                raise ValueError("recovery endpoints do not match the major interval")
        if len({item.recovery_id for item in self.recovery_paths}) != len(
            self.recovery_paths
        ):
            raise ValueError("recovery path IDs must be unique")
        if len(set(self.all_generation_operation_ids)) != len(
            self.all_generation_operation_ids
        ):
            raise ValueError("Q-CAST generation operation IDs must be unique")
        if len(set(self.all_elementary_segment_ids)) != len(
            self.all_elementary_segment_ids
        ):
            raise ValueError("Q-CAST elementary segment IDs must be unique")

    @property
    def all_generation_operation_ids(self) -> tuple[str, ...]:
        return (
            self.major_generation_operation_ids
            + tuple(
                operation_id
                for recovery in self.recovery_paths
                for operation_id in recovery.generation_operation_ids
            )
        )

    @property
    def all_elementary_segment_ids(self) -> tuple[str, ...]:
        return (
            self.major_segment_ids
            + tuple(
                segment_id
                for recovery in self.recovery_paths
                for segment_id in recovery.segment_ids
            )
        )


@dataclass(frozen=True)
class QCASTPlanningRecord:
    """One online-window Q-CAST planning result."""

    candidates: tuple[RouteConstructionCandidate, ...]
    expansion: TimeExpansionResult
    candidate_scores: tuple[tuple[str, float], ...]
    allocations: tuple[QCASTAllocation, ...]
    solution: PackingSolution

    @property
    def score_by_candidate(self) -> dict[str, float]:
        return dict(self.candidate_scores)

    @property
    def allocation_by_candidate(self) -> dict[str, QCASTAllocation]:
        return {
            allocation.candidate.candidate_id: allocation
            for allocation in self.allocations
        }

    @property
    def selected_expected_throughput(self) -> float:
        scores = self.score_by_candidate
        return float(sum(
            scores[variable.candidate_id]
            for variable in self.solution.selected_variables
        ))


@dataclass(frozen=True)
class _PickedMajor:
    request_id: str
    route_nodes: tuple[int, ...]
    score: float


@dataclass(frozen=True)
class _PickedRecovery:
    major_start_index: int
    major_end_index: int
    route_nodes: tuple[int, ...]


class _ResidualResources:
    """Width-one link/channel/qubit state used by official phase-2 search."""

    def __init__(
        self,
        capacities: Mapping[str, int],
        reserved_usage: Mapping[tuple[str, int], int],
        slot: int,
    ) -> None:
        self.remaining = {
            resource_id: int(capacity)
            - int(reserved_usage.get((resource_id, slot), 0))
            for resource_id, capacity in capacities.items()
        }

    @staticmethod
    def _path_demand(route_nodes: Sequence[int]) -> dict[str, int]:
        demand: dict[str, int] = {}
        for left, right in zip(route_nodes, route_nodes[1:]):
            edge = f"{min(left, right)}-{max(left, right)}"
            for resource_id in (f"link:{edge}", f"genlane:{edge}"):
                demand[resource_id] = demand.get(resource_id, 0) + 1
            demand[f"memory:{left}"] = demand.get(f"memory:{left}", 0) + 1
            demand[f"memory:{right}"] = demand.get(f"memory:{right}", 0) + 1
        return demand

    def can_reserve(self, route_nodes: Sequence[int]) -> bool:
        return all(
            self.remaining.get(resource_id, 0) >= amount
            for resource_id, amount in self._path_demand(route_nodes).items()
        )

    def reserve(self, route_nodes: Sequence[int]) -> None:
        demand = self._path_demand(route_nodes)
        if not all(
            self.remaining.get(resource_id, 0) >= amount
            for resource_id, amount in demand.items()
        ):
            raise ValueError("Q-CAST attempted to reserve an infeasible path")
        for resource_id, amount in demand.items():
            self.remaining[resource_id] -= amount


def effective_link_generation_probability(physical: PhysicalConfig) -> float:
    """Return the end-to-end elementary-link success probability."""

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
    *,
    width: int = 1,
) -> float:
    """Compute Q-CAST EXT using the official ``q ** (h - 1)`` factor."""

    hop_count = len(route_nodes) - 1
    if hop_count < 1:
        return 0.0
    link_probability = effective_link_generation_probability(episode.physical)
    return expected_throughput(
        (link_probability,) * hop_count,
        width=width,
        swap_probability=episode.physical.swap_probability,
    )


def _residual_graph(
    episode: EpisodeSpec,
    resources: _ResidualResources,
    source: int,
    destination: int,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(sorted(episode.nodes))
    for left, right in sorted(
        (tuple(sorted(edge)) for edge in episode.edges)
    ):
        edge = f"{min(left, right)}-{max(left, right)}"
        if (
            resources.remaining.get(f"link:{edge}", 0) >= 1
            and resources.remaining.get(f"genlane:{edge}", 0) >= 1
        ):
            graph.add_edge(left, right)
    for node in tuple(graph.nodes):
        required = 1 if node in {source, destination} else 2
        if resources.remaining.get(f"memory:{node}", 0) < required:
            graph.remove_node(node)
    return graph


def _best_residual_path(
    episode: EpisodeSpec,
    resources: _ResidualResources,
    source: int,
    destination: int,
    *,
    max_search_hops: int,
) -> tuple[tuple[int, ...], float] | None:
    graph = _residual_graph(
        episode,
        resources,
        source,
        destination,
    )
    if source not in graph or destination not in graph:
        return None
    try:
        route = tuple(
            int(node) for node in nx.shortest_path(graph, source, destination)
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    if len(route) - 1 > max_search_hops or not resources.can_reserve(route):
        return None
    return route, qcast_path_score(episode, route)


def _pick_major_paths(
    episode: EpisodeSpec,
    request_ids: tuple[str, ...],
    resources: _ResidualResources,
    *,
    max_search_hops: int,
) -> tuple[_PickedMajor, ...]:
    requests = {request.id: request for request in episode.requests}
    remaining = set(request_ids)
    picked: list[_PickedMajor] = []
    while remaining:
        candidates = []
        for request_id in sorted(remaining):
            request = requests[request_id]
            result = _best_residual_path(
                episode,
                resources,
                request.source,
                request.destination,
                max_search_hops=max_search_hops,
            )
            if result is None:
                continue
            route, score = result
            if score > 0.0:
                candidates.append(_PickedMajor(request_id, route, score))
        if not candidates:
            break
        chosen = min(
            candidates,
            key=lambda item: (
                -item.score,
                len(item.route_nodes),
                item.request_id,
                item.route_nodes,
            ),
        )
        resources.reserve(chosen.route_nodes)
        picked.append(chosen)
        remaining.remove(chosen.request_id)
    return tuple(picked)


def _pick_recovery_paths(
    episode: EpisodeSpec,
    major: _PickedMajor,
    resources: _ResidualResources,
    *,
    recovery_span_limit: int,
    max_search_hops: int,
) -> tuple[_PickedRecovery, ...]:
    route = major.route_nodes
    picked: list[_PickedRecovery] = []
    maximum_span = min(recovery_span_limit, len(route) - 1)
    for span in range(1, maximum_span + 1):
        for start in range(0, len(route) - span):
            end = start + span
            result = _best_residual_path(
                episode,
                resources,
                route[start],
                route[end],
                max_search_hops=max_search_hops,
            )
            if result is None or result[1] <= 0.0:
                continue
            recovery_route, _ = result
            resources.reserve(recovery_route)
            picked.append(_PickedRecovery(start, end, recovery_route))
    return tuple(picked)


def _terminal_segment_id(
    dag: ConstructionDAG,
    source: int,
    destination: int,
) -> str:
    consumers = {
        segment_id
        for operation in dag.operations
        for segment_id in operation.input_segment_ids
    }
    candidates = [
        operation
        for operation in dag.operations
        if (
            operation.output_segment_id is not None
            and operation.output_segment_id not in consumers
            and operation.output_endpoints is not None
            and frozenset(operation.output_endpoints)
            == frozenset((source, destination))
        )
    ]
    if len(candidates) != 1:
        raise ValueError("Q-CAST major DAG must have one terminal segment")
    assert candidates[0].output_segment_id is not None
    return candidates[0].output_segment_id


def _compile_allocation(
    episode: EpisodeSpec,
    major: _PickedMajor,
    recoveries: tuple[_PickedRecovery, ...],
    *,
    start_slot: int,
) -> QCASTAllocation:
    request = next(
        item for item in episode.requests if item.id == major.request_id
    )
    base = left_deep_path_dag(
        request.id,
        major.route_nodes,
        required_fidelity=request.required_fidelity,
    )
    operations = list(base.operations)
    major_generations = tuple(
        operation
        for operation in base.operations
        if operation.kind == OperationKind.GEN
    )
    recovery_plans: list[QCASTRecoveryPathPlan] = []
    for recovery_index, recovery in enumerate(recoveries):
        template = left_deep_path_dag(request.id, recovery.route_nodes)
        generation_ids: list[str] = []
        segment_ids: list[str] = []
        for edge_index, generation in enumerate(
            operation
            for operation in template.operations
            if operation.kind == OperationKind.GEN
        ):
            operation_id = (
                f"{request.id}:qcast:round:{start_slot}:recovery:"
                f"{recovery_index}:gen:{edge_index}"
            )
            segment_id = (
                f"{request.id}:qcast:round:{start_slot}:recovery:"
                f"{recovery_index}:segment:{edge_index}"
            )
            operations.append(replace(
                generation,
                op_id=operation_id,
                request_id=request.id,
                predecessors=(),
                output_segment_id=segment_id,
                required_fidelity=0.0,
                retry_limit=0,
                ordinal=len(operations),
            ))
            generation_ids.append(operation_id)
            segment_ids.append(segment_id)
        recovery_plans.append(QCASTRecoveryPathPlan(
            recovery_id=(
                f"{request.id}:qcast:round:{start_slot}:recovery:{recovery_index}"
            ),
            major_start_index=recovery.major_start_index,
            major_end_index=recovery.major_end_index,
            route_nodes=recovery.route_nodes,
            generation_operation_ids=tuple(generation_ids),
            segment_ids=tuple(segment_ids),
        ))
    dag = ConstructionDAG(request.id, tuple(operations))
    terminal = _terminal_segment_id(
        base,
        request.source,
        request.destination,
    )
    route_token = "-".join(str(node) for node in major.route_nodes)
    candidate = RouteConstructionCandidate(
        candidate_id=(
            f"{request.id}:qcast:round:{start_slot}:major:{route_token}"
        ),
        request_id=request.id,
        route_nodes=major.route_nodes,
        construction_kind="left_deep",
        dag=dag,
        terminal_segment_id=terminal,
        terminal_segment_ids=(terminal,),
        purification_kind="none",
    )
    return QCASTAllocation(
        candidate=candidate,
        expected_throughput=major.score,
        width=1,
        major_generation_operation_ids=tuple(
            operation.op_id for operation in major_generations
        ),
        major_segment_ids=tuple(
            operation.output_segment_id or ""
            for operation in major_generations
        ),
        recovery_paths=tuple(recovery_plans),
    )


def _normalised_scores(
    variables: Sequence[TimeExpandedCandidate],
    score_by_candidate: Mapping[str, float],
) -> tuple[float, ...]:
    maximum = max(score_by_candidate.values(), default=0.0)
    if maximum <= 0.0:
        return (0.0,) * len(variables)
    return tuple(
        score_by_candidate[variable.candidate_id] / maximum
        for variable in variables
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
    path_candidate_count: int = 4,
    construction_kind: str = "left_deep",
    purification_kind: str = "none",
    recovery_span_limit: int = 3,
    max_search_hops: int = 15,
) -> QCASTPlanningRecord:
    """Run Q-CAST major/recovery allocation over one online start window."""

    if path_candidate_count < 1:
        raise ValueError("path_candidate_count must be positive")
    # Kept for the shared experiment CLI.  Unlike Yen-based methods, the
    # official width-one Q-CAST search recomputes one exact best path on the
    # residual graph and therefore does not truncate to this many candidates.
    if recovery_span_limit < 0:
        raise ValueError("recovery_span_limit cannot be negative")
    if max_search_hops < 1:
        raise ValueError("max_search_hops must be positive")
    if construction_kind != "left_deep":
        raise ValueError("official Q-CAST adaptation uses left_deep swapping")
    if purification_kind != "none":
        raise ValueError("Q-CAST does not make purification decisions")
    if episode.physical.max_width != 1:
        raise ValueError("this experiment freezes Q-CAST maximum width to one")
    if not 0 <= window_start_slot < window_end_slot <= episode.horizon:
        raise ValueError("Q-CAST start window lies outside the episode")
    completion_limit = (
        episode.horizon
        if completion_end_slot is None
        else int(completion_end_slot)
    )
    if not window_end_slot <= completion_limit <= episode.horizon:
        raise ValueError("invalid Q-CAST completion boundary")

    declared = (
        tuple(request.id for request in episode.requests)
        if request_ids is None
        else tuple(str(request_id) for request_id in request_ids)
    )
    if len(set(declared)) != len(declared):
        raise ValueError("request_ids must be unique")
    request_by_id = {request.id: request for request in episode.requests}
    unknown = set(declared) - set(request_by_id)
    if unknown:
        raise ValueError(f"unknown request: {sorted(unknown)[0]}")
    if any(request_by_id[request_id].demand_pairs != 1 for request_id in declared):
        raise ValueError("Q-CAST request-completion experiments require one pair per request")

    capacities = (
        build_resource_capacities(episode)
        if resource_capacities is None
        else {str(key): int(value) for key, value in resource_capacities.items()}
    )
    initial_reserved = normalize_reserved_usage(reserved_usage, capacities)
    committed_usage = dict(initial_reserved)
    remaining_requests = set(declared)
    candidates: list[RouteConstructionCandidate] = []
    allocations: list[QCASTAllocation] = []
    variables: list[TimeExpandedCandidate] = []
    schedules: list[NominalConstructionSchedule] = []
    rejections: list[CandidateRejection] = []
    score_by_candidate: dict[str, float] = {}

    for start_slot in range(window_start_slot, window_end_slot):
        eligible = tuple(sorted(
            request_id
            for request_id in remaining_requests
            if request_by_id[request_id].arrival <= start_slot
            and (
                request_by_id[request_id].deadline is None
                or request_by_id[request_id].deadline > start_slot
            )
        ))
        if not eligible:
            continue
        residual = _ResidualResources(capacities, committed_usage, start_slot)
        major_paths = _pick_major_paths(
            episode,
            eligible,
            residual,
            max_search_hops=max_search_hops,
        )
        for major in major_paths:
            recoveries = (
                ()
                if recovery_span_limit == 0
                else _pick_recovery_paths(
                    episode,
                    major,
                    residual,
                    recovery_span_limit=recovery_span_limit,
                    max_search_hops=max_search_hops,
                )
            )
            allocation = _compile_allocation(
                episode,
                major,
                recoveries,
                start_slot=start_slot,
            )
            candidate = allocation.candidate
            candidates.append(candidate)
            score_by_candidate[candidate.candidate_id] = allocation.expected_throughput
            fidelity = candidate_fidelity_estimate_map(episode, (candidate,))
            expanded = expand_construction_candidates(
                episode.planning,
                (candidate,),
                capacities,
                fidelity_estimates=fidelity,
                reserved_usage=committed_usage,
                window_start_slot=start_slot,
                window_end_slot=start_slot + 1,
                completion_end_slot=completion_limit,
            )
            schedules.extend(expanded.schedules)
            rejections.extend(expanded.rejections)
            if not expanded.variables:
                continue
            if len(expanded.variables) != 1:
                raise RuntimeError("fixed Q-CAST round produced multiple start variables")
            variable = replace(
                expanded.variables[0],
                expected_success_probability=allocation.expected_throughput,
            )
            variables.append(variable)
            allocations.append(allocation)
            remaining_requests.remove(major.request_id)
            for usage in variable.resource_usage:
                key = (usage.resource_id, usage.slot)
                committed_usage[key] = committed_usage.get(key, 0) + usage.amount

    selected = tuple(sorted(variables, key=lambda item: item.variable_id))
    feasibility = validate_packing_selection(
        selected,
        capacities,
        initial_reserved,
    )
    if not feasibility.feasible:
        raise RuntimeError(
            "Q-CAST residual allocation produced an infeasible schedule: "
            + feasibility.violations[0]
        )
    solution = PackingSolution(
        variables=selected,
        scores=_normalised_scores(selected, score_by_candidate),
        request_ids=declared,
        selected_variables=selected,
        feasibility=feasibility,
        strategy="qcast_residual_ext_with_recovery",
    )
    return QCASTPlanningRecord(
        candidates=tuple(candidates),
        expansion=TimeExpansionResult(
            variables=selected,
            schedules=tuple(sorted(schedules, key=lambda item: item.candidate_id)),
            rejections=tuple(sorted(rejections)),
        ),
        candidate_scores=tuple(sorted(score_by_candidate.items())),
        allocations=tuple(sorted(
            allocations,
            key=lambda item: item.candidate.candidate_id,
        )),
        solution=solution,
    )


__all__ = [
    "QCASTAllocation",
    "QCASTPlanningRecord",
    "QCASTRecoveryPathPlan",
    "effective_link_generation_probability",
    "plan_qcast_window",
    "qcast_path_score",
]
