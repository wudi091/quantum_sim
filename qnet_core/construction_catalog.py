"""Route-skeleton and construction-plan catalogue for joint decisions."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from dataclasses import replace
from typing import Sequence

import networkx as nx

from .construction_api import ConstructionDAG, ConstructionOperation, OperationKind
from .construction_plans import (
    balanced_path_dag,
    elementary_purification_dag,
    left_deep_path_dag,
    swap_tree_kinds,
    swap_tree_path_dag,
)
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
    purification_kind: str = "none"

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
    purification_kind: str = "none",
) -> ConstructionDAG:
    if kind == "left_deep":
        dag = left_deep_path_dag(
            request.id,
            route,
            required_fidelity=request.required_fidelity,
        )
    elif kind == "balanced":
        dag = balanced_path_dag(
            request.id,
            route,
            required_fidelity=request.required_fidelity,
        )
    elif kind.startswith("swap_tree_"):
        index_token = kind.removeprefix("swap_tree_")
        if not index_token.isdigit():
            raise ValueError(f"unknown construction kind: {kind}")
        dag = swap_tree_path_dag(
            request.id,
            route,
            int(index_token),
            required_fidelity=request.required_fidelity,
        )
    else:
        raise ValueError(f"unknown construction kind: {kind}")
    if purification_kind == "none":
        return dag
    if purification_kind == "elementary_once":
        return elementary_purification_dag(dag)
    raise ValueError(f"unknown purification kind: {purification_kind}")


def build_route_construction_catalogue(
    spec: PlanningSpec,
    candidate_count: int | None = 3,
    construction_kinds: tuple[str, ...] = ("left_deep", "balanced"),
    purification_kinds: tuple[str, ...] = ("none",),
    swap_tree_count: int | None = None,
) -> tuple[RouteConstructionCandidate, ...]:
    """Enumerate a joint ``(path, construction)`` action space.

    ``candidate_count`` bounds the shortest-path catalogue.  Passing ``None``
    enumerates all simple paths and is reserved for small-instance coverage
    and nominal-oracle checks.
    """

    if candidate_count is not None and candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    if not construction_kinds and swap_tree_count is None:
        raise ValueError("at least one construction policy is required")
    if swap_tree_count is not None and swap_tree_count < 1:
        raise ValueError("swap_tree_count must be positive")
    if not purification_kinds:
        raise ValueError("at least one purification kind is required")
    if len(set(purification_kinds)) != len(purification_kinds):
        raise ValueError("purification kinds must be unique")
    unknown_purification = set(purification_kinds) - {"none", "elementary_once"}
    if unknown_purification:
        raise ValueError(
            f"unknown purification kind: {sorted(unknown_purification)[0]}"
        )
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
            route_kinds = tuple(dict.fromkeys((
                *construction_kinds,
                *(
                    ()
                    if swap_tree_count is None
                    else swap_tree_kinds(len(route) - 1, swap_tree_count)
                ),
            )))
            for kind in route_kinds:
                for purification_kind in purification_kinds:
                    base_dag = _dag_for_kind(
                        request, route, kind, purification_kind
                    )
                    dag, terminal_ids = _repeat_demand_dag(
                        base_dag,
                        request.id,
                        request.demand_pairs,
                        request.source,
                        request.destination,
                    )
                    suffix = (
                        "" if purification_kind == "none"
                        else f":purify:{purification_kind}"
                    )
                    candidates.append(RouteConstructionCandidate(
                        candidate_id=(
                            f"{request.id}:path:{route_index}:{kind}{suffix}"
                        ),
                        request_id=request.id,
                        route_nodes=route,
                        construction_kind=kind,
                        dag=dag,
                        terminal_segment_id=terminal_ids[-1],
                        terminal_segment_ids=terminal_ids,
                        purification_kind=purification_kind,
                    ))
    return tuple(candidates)


def build_dynamic_repair_catalogue(
    spec: PlanningSpec,
    request_id: str,
    *,
    excluded_routes: Sequence[tuple[int, ...]] = (),
    max_paths: int = 4,
    construction_kinds: tuple[str, ...] = ("left_deep", "balanced"),
    purification_kinds: tuple[str, ...] = ("none",),
    swap_tree_count: int | None = None,
) -> tuple[RouteConstructionCandidate, ...]:
    """Generate previously unseen route/construction candidates at repair time.

    Admission remains bounded by its fixed catalogue.  This repair-only gate
    enumerates at most ``max_paths`` shortest simple paths absent from the
    supplied route set and compiles each into the requested construction DAGs.
    The result consists solely of neutral DTOs and is validated by the same
    environment-side scheduler as an admitted candidate.
    """
    if max_paths < 1:
        raise ValueError("max_paths must be positive")
    if not construction_kinds and swap_tree_count is None:
        raise ValueError("at least one construction policy is required")
    if swap_tree_count is not None and swap_tree_count < 1:
        raise ValueError("swap_tree_count must be positive")
    if not purification_kinds:
        raise ValueError("at least one purification kind is required")
    if set(purification_kinds) - {"none", "elementary_once"}:
        raise ValueError("unknown purification kind")
    requests = {request.id: request for request in spec.requests}
    if request_id not in requests:
        raise KeyError(request_id)
    request = requests[request_id]
    excluded = {tuple(route) for route in excluded_routes}
    graph = nx.Graph()
    graph.add_nodes_from(spec.nodes)
    graph.add_edges_from(spec.edges)
    candidates: list[RouteConstructionCandidate] = []
    new_route_count = 0
    try:
        paths = nx.shortest_simple_paths(
            graph, request.source, request.destination
        )
        for raw_path in paths:
            route = tuple(int(node) for node in raw_path)
            if route in excluded:
                continue
            route_token = "-".join(str(node) for node in route)
            route_kinds = tuple(dict.fromkeys((
                *construction_kinds,
                *(
                    ()
                    if swap_tree_count is None
                    else swap_tree_kinds(len(route) - 1, swap_tree_count)
                ),
            )))
            for kind in route_kinds:
                for purification_kind in purification_kinds:
                    base_dag = _dag_for_kind(
                        request, route, kind, purification_kind
                    )
                    dag, terminal_ids = _repeat_demand_dag(
                        base_dag,
                        request.id,
                        request.demand_pairs,
                        request.source,
                        request.destination,
                    )
                    suffix = (
                        "" if purification_kind == "none"
                        else f":purify:{purification_kind}"
                    )
                    candidates.append(RouteConstructionCandidate(
                        candidate_id=(
                            f"{request.id}:dynamic:path:{route_token}:{kind}{suffix}"
                        ),
                        request_id=request.id,
                        route_nodes=route,
                        construction_kind=kind,
                        dag=dag,
                        terminal_segment_id=terminal_ids[-1],
                        terminal_segment_ids=terminal_ids,
                        purification_kind=purification_kind,
                    ))
            new_route_count += 1
            if new_route_count >= max_paths:
                break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return ()
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
