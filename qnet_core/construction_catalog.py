"""Route-skeleton and construction-plan catalogue for joint decisions."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from dataclasses import replace

import networkx as nx

from .construction_api import ConstructionDAG, ConstructionOperation, OperationKind
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
    terminal_segment_ids: tuple[str, ...] = ()

    @property
    def hop_count(self) -> int:
        return len(self.route_nodes) - 1

    @property
    def demand_pairs(self) -> int:
        return len(self.terminal_segment_ids or (self.terminal_segment_id,))

    @property
    def all_terminal_segment_ids(self) -> tuple[str, ...]:
        return self.terminal_segment_ids or (self.terminal_segment_id,)


def _repeat_demand_dag(
    base_dag: ConstructionDAG,
    request_id: str,
    demand_pairs: int,
    source: int,
    destination: int,
) -> tuple[ConstructionDAG, tuple[str, ...]]:
    """Compile sequential independent deliveries from one route construction DAG.

    Each copy has fresh operation/segment IDs.  A release operation consumes
    the completed terminal segment before the next copy becomes ready, so the
    physical backend never has to hold multiple terminal pairs for one
    request.  The final copy remains resident until request settlement.
    """

    if demand_pairs < 1:
        raise ValueError("demand_pairs must be positive")
    base_operations = base_dag.operations
    terminal_candidates = [
        operation for operation in reversed(base_operations)
        if (
            operation.output_endpoints is not None
            and set(operation.output_endpoints) == {source, destination}
        )
    ]
    if not terminal_candidates:
        raise ValueError("construction DAG has no terminal operation")
    terminal_base = terminal_candidates[0]
    if terminal_base.output_segment_id is None:
        raise ValueError("terminal operation must produce a segment")
    if demand_pairs == 1:
        return base_dag, (terminal_base.output_segment_id,)

    operations: list[ConstructionOperation] = []
    terminal_ids: list[str] = []
    previous_release_id: str | None = None
    base_count = len(base_operations)
    for pair_index in range(demand_pairs):
        op_id_map = {
            operation.op_id: f"{operation.op_id}:pair:{pair_index}"
            for operation in base_operations
        }
        segment_id_map = {
            operation.output_segment_id: f"{operation.output_segment_id}:pair:{pair_index}"
            for operation in base_operations
            if operation.output_segment_id is not None
        }
        copied: list[ConstructionOperation] = []
        for operation in base_operations:
            predecessors = tuple(op_id_map[pred] for pred in operation.predecessors)
            if previous_release_id is not None and not predecessors:
                predecessors = (previous_release_id,)
            copied.append(replace(
                operation,
                op_id=op_id_map[operation.op_id],
                predecessors=predecessors,
                input_segment_ids=tuple(segment_id_map.get(item, item) for item in operation.input_segment_ids),
                output_segment_id=(
                    None if operation.output_segment_id is None
                    else segment_id_map[operation.output_segment_id]
                ),
                ordinal=pair_index * (base_count + 1) + operation.ordinal,
            ))
        operations.extend(copied)
        terminal_id = segment_id_map[terminal_base.output_segment_id]
        terminal_ids.append(terminal_id)
        if pair_index < demand_pairs - 1:
            release_id = f"{request_id}:pair:{pair_index}:release"
            operations.append(ConstructionOperation(
                op_id=release_id,
                request_id=request_id,
                kind=OperationKind.RELEASE,
                predecessors=(op_id_map[terminal_base.op_id],),
                input_segment_ids=(terminal_id,),
                duration_ps=1,
                ordinal=(pair_index + 1) * (base_count + 1) - 1,
            ))
            previous_release_id = release_id
    return ConstructionDAG(request_id, tuple(operations)), tuple(terminal_ids)


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
    candidate_count: int | None = 3,
    construction_kinds: tuple[str, ...] = ("left_deep", "balanced"),
) -> tuple[RouteConstructionCandidate, ...]:
    """Enumerate a joint ``(path, construction)`` action space.

    ``candidate_count`` bounds the shortest-path catalogue.  Passing ``None``
    enumerates all simple paths and is reserved for small-instance coverage
    and nominal-oracle checks.
    """

    if candidate_count is not None and candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    if not construction_kinds:
        raise ValueError("at least one construction kind is required")
    graph = nx.Graph()
    graph.add_nodes_from(spec.nodes)
    graph.add_edges_from(spec.edges)
    candidates: list[RouteConstructionCandidate] = []
    for request in sorted(spec.requests, key=lambda item: item.id):
        try:
            if candidate_count is None:
                paths = nx.all_simple_paths(
                    graph, request.source, request.destination
                )
            else:
                paths = itertools.islice(
                    nx.shortest_simple_paths(
                        graph, request.source, request.destination
                    ),
                    candidate_count,
                )
            routes = [tuple(int(node) for node in path) for path in paths]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            routes = []
        for route_index, route in enumerate(routes):
            for kind in construction_kinds:
                base_dag = _dag_for_kind(request, route, kind)
                dag, terminal_ids = _repeat_demand_dag(
                    base_dag,
                    request.id,
                    request.demand_pairs,
                    request.source,
                    request.destination,
                )
                candidates.append(RouteConstructionCandidate(
                    candidate_id=f"{request.id}:path:{route_index}:{kind}",
                    request_id=request.id,
                    route_nodes=route,
                    construction_kind=kind,
                    dag=dag,
                    terminal_segment_id=terminal_ids[-1],
                    terminal_segment_ids=terminal_ids,
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
