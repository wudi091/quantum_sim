"""Route-skeleton and construction-plan catalogue for joint decisions."""

from __future__ import annotations

from dataclasses import dataclass
import itertools

import networkx as nx

from .construction_api import ConstructionDAG
from .construction_plans import balanced_path_dag, left_deep_path_dag
from .planning_spec import PlanningSpec, RequestSpec


@dataclass(frozen=True)
class RouteConstructionCandidate:
    candidate_id: str
    request_id: str
    route_nodes: tuple[int, ...]
    construction_kind: str
    dag: ConstructionDAG
    terminal_segment_id: str

    @property
    def hop_count(self) -> int:
        return len(self.route_nodes) - 1


def _dag_for_kind(
    request: RequestSpec,
    route: tuple[int, ...],
    kind: str,
) -> ConstructionDAG:
    if kind == "left_deep":
        return left_deep_path_dag(
            request.id,
            route,
            required_fidelity=request.required_fidelity,
        )
    if kind == "balanced":
        return balanced_path_dag(
            request.id,
            route,
            required_fidelity=request.required_fidelity,
        )
    raise ValueError(f"unknown construction kind: {kind}")


def build_route_construction_catalogue(
    spec: PlanningSpec,
    candidate_count: int = 3,
    construction_kinds: tuple[str, ...] = ("left_deep", "balanced"),
) -> tuple[RouteConstructionCandidate, ...]:
    """Enumerate a bounded joint ``(path, construction)`` action space."""

    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    if not construction_kinds:
        raise ValueError("at least one construction kind is required")
    graph = nx.Graph()
    graph.add_nodes_from(spec.nodes)
    graph.add_edges_from(spec.edges)
    candidates: list[RouteConstructionCandidate] = []
    for request in sorted(spec.requests, key=lambda item: item.id):
        try:
            paths = itertools.islice(
                nx.shortest_simple_paths(graph, request.source, request.destination),
                candidate_count,
            )
            routes = [tuple(int(node) for node in path) for path in paths]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            routes = []
        for route_index, route in enumerate(routes):
            for kind in construction_kinds:
                dag = _dag_for_kind(request, route, kind)
                terminal = next(
                    operation.output_segment_id
                    for operation in reversed(dag.operations)
                    if operation.output_endpoints == (request.source, request.destination)
                )
                candidates.append(RouteConstructionCandidate(
                    candidate_id=f"{request.id}:path:{route_index}:{kind}",
                    request_id=request.id,
                    route_nodes=route,
                    construction_kind=kind,
                    dag=dag,
                    terminal_segment_id=terminal,
                ))
    return tuple(candidates)


def candidates_by_request(
    candidates: tuple[RouteConstructionCandidate, ...],
) -> dict[str, tuple[RouteConstructionCandidate, ...]]:
    grouped: dict[str, list[RouteConstructionCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.request_id, []).append(candidate)
    return {
        request_id: tuple(sorted(values, key=lambda candidate: candidate.candidate_id))
        for request_id, values in grouped.items()
    }
