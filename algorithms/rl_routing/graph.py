"""Sparse heterogeneous graph view of an ARC-Q decision state."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor

from qnet_core.construction_api import OperationKind

from .environment import FeasiblePlanBuilder, RoutingObservation


NODE_FEATURE_DIM = 24
EDGE_FEATURE_DIM = 4
GLOBAL_FEATURE_DIM = 10
NODE_TYPE_COUNT = 4
RELATION_COUNT = 14

PHYSICAL_NODE = 0
REQUEST_NODE = 1
CANDIDATE_NODE = 2
RESOURCE_SLOT_NODE = 3


@dataclass(frozen=True)
class RoutingGraph:
    """Tensor graph plus stable correspondence to candidate actions."""

    node_features: Tensor
    node_types: Tensor
    edge_index: Tensor
    edge_types: Tensor
    edge_features: Tensor
    global_features: Tensor
    candidate_node_indices: Tensor
    candidate_variable_ids: tuple[str, ...]
    candidate_legal_mask: Tensor

    def to(self, device: torch.device | str) -> "RoutingGraph":
        return RoutingGraph(
            node_features=self.node_features.to(device),
            node_types=self.node_types.to(device),
            edge_index=self.edge_index.to(device),
            edge_types=self.edge_types.to(device),
            edge_features=self.edge_features.to(device),
            global_features=self.global_features.to(device),
            candidate_node_indices=self.candidate_node_indices.to(device),
            candidate_variable_ids=self.candidate_variable_ids,
            candidate_legal_mask=self.candidate_legal_mask.to(device),
        )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def _resource_type(resource_id: str) -> str:
    return resource_id.split(":", 1)[0]


def _resource_nodes(resource_id: str) -> tuple[int, ...]:
    kind, _, suffix = resource_id.partition(":")
    try:
        if kind in {"memory", "bsm", "swapnode"}:
            return (int(suffix),)
        if kind in {"link", "genlane", "purify"}:
            left, right = suffix.split("-", 1)
            return int(left), int(right)
    except (TypeError, ValueError):
        return ()
    return ()


def _operation_fractions(variable) -> tuple[float, float, float]:
    operations = variable.base_candidate.dag.operations
    count = max(1, len(operations))
    generation = sum(item.kind == OperationKind.GEN for item in operations)
    swap = sum(item.kind == OperationKind.SWAP for item in operations)
    purification = sum(item.kind == OperationKind.PURIFY for item in operations)
    return generation / count, swap / count, purification / count


def _add_bidirectional_edge(
    sources: list[int],
    destinations: list[int],
    relation_types: list[int],
    attributes: list[tuple[float, float, float, float]],
    left: int,
    right: int,
    forward_relation: int,
    attribute: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> None:
    sources.extend((left, right))
    destinations.extend((right, left))
    relation_types.extend((forward_relation, forward_relation + 1))
    attributes.extend((attribute, attribute))


def build_routing_graph(
    observation: RoutingObservation,
    builder: FeasiblePlanBuilder | None = None,
    *,
    device: torch.device | str | None = None,
) -> RoutingGraph:
    """Build a permutation-compatible graph for one autoregressive state.

    Resource--slot nodes are sparse: only slots touched by a candidate or an
    existing reservation are materialized.  This preserves the exact conflict
    structure without expanding every resource over the whole horizon.
    """

    active_builder = builder or FeasiblePlanBuilder(observation)
    variables = observation.variables
    variable_ids = tuple(variable.variable_id for variable in variables)
    if set(active_builder.selected_variable_ids) - set(variable_ids):
        raise ValueError("plan builder belongs to another observation")

    physical_nodes = tuple(sorted(observation.nodes))
    edges = observation.edges
    request_specs = {
        request.id: request for request in observation.visible_requests
    }
    horizon = observation.horizon_slots
    request_ids = tuple(sorted(observation.visible_request_ids))
    capacities = observation.capacities
    max_capacity = max(capacities.values(), default=1)
    current_usage = dict(active_builder.current_usage)
    current_reservations = dict(observation.physical_snapshot.reservations)

    resource_slots = {
        (item.resource_id, item.slot)
        for variable in variables
        for item in variable.resource_usage
    }
    resource_slots.update(current_usage)
    resource_slots_ordered = tuple(sorted(resource_slots))

    node_keys: list[tuple[str, object]] = []
    node_keys.extend(("physical", node) for node in physical_nodes)
    node_keys.extend(("request", request_id) for request_id in request_ids)
    node_keys.extend(("candidate", item) for item in variable_ids)
    node_keys.extend(("resource", item) for item in resource_slots_ordered)
    node_index = {key: index for index, key in enumerate(node_keys)}

    features = torch.zeros((len(node_keys), NODE_FEATURE_DIM), dtype=torch.float32)
    node_types = torch.empty(len(node_keys), dtype=torch.long)
    current_fraction = _safe_ratio(observation.slot, horizon)
    remaining_fraction = _safe_ratio(horizon - observation.slot, horizon)
    degree = {node: 0 for node in physical_nodes}
    for left, right in edges:
        degree[left] = degree.get(left, 0) + 1
        degree[right] = degree.get(right, 0) + 1
    max_degree = max(degree.values(), default=1)

    segments_by_node: dict[int, list[object]] = {}
    segments_by_request: dict[str, list[object]] = {}
    for segment in observation.physical_snapshot.segments:
        segments_by_node.setdefault(segment.left, []).append(segment)
        segments_by_node.setdefault(segment.right, []).append(segment)
        segments_by_request.setdefault(segment.request_id, []).append(segment)
    inflight_by_request: dict[str, int] = {}
    for operation in observation.physical_snapshot.in_flight:
        inflight_by_request[operation.request_id] = (
            inflight_by_request.get(operation.request_id, 0) + 1
        )
    dag_states = {
        state.request_id: state
        for state in observation.physical_snapshot.dag_states
    }

    for physical_node in physical_nodes:
        index = node_index[("physical", physical_node)]
        node_types[index] = PHYSICAL_NODE
        features[index, PHYSICAL_NODE] = 1.0
        features[index, 4] = current_fraction
        features[index, 5] = remaining_fraction
        features[index, 6] = _safe_ratio(degree.get(physical_node, 0), max_degree)
        memory_id = f"memory:{physical_node}"
        memory_capacity = capacities.get(memory_id, 1)
        features[index, 7] = _safe_ratio(memory_capacity, max_capacity)
        features[index, 8] = _safe_ratio(
            current_reservations.get(memory_id, 0), memory_capacity
        )
        node_segments = segments_by_node.get(physical_node, [])
        features[index, 9] = _safe_ratio(len(node_segments), memory_capacity)
        if node_segments:
            features[index, 10] = sum(
                float(segment.fidelity) for segment in node_segments
            ) / len(node_segments)
            features[index, 11] = max(
                _safe_ratio(
                    observation.physical_snapshot.physical_time_ps
                    - int(segment.born_time_ps),
                    max(observation.physical_snapshot.horizon_ps, 1),
                )
                for segment in node_segments
            )
        features[index, 12] = float(
            current_reservations.get(f"bsm:{physical_node}", 0) > 0
        )
        features[index, 13] = float(
            current_reservations.get(f"swapnode:{physical_node}", 0) > 0
        )

    max_demand = max(
        (request.demand_pairs for request in request_specs.values()),
        default=1,
    )
    for request_id in request_ids:
        index = node_index[("request", request_id)]
        node_types[index] = REQUEST_NODE
        features[index, REQUEST_NODE] = 1.0
        features[index, 4] = current_fraction
        features[index, 5] = remaining_fraction
        request = request_specs.get(request_id)
        if request is not None:
            features[index, 6] = _safe_ratio(
                observation.slot - request.arrival, horizon
            )
            deadline = request.deadline or horizon
            features[index, 7] = _safe_ratio(
                max(0, deadline - observation.slot), horizon
            )
            features[index, 8] = _safe_ratio(request.demand_pairs, max_demand)
            features[index, 9] = float(request.required_fidelity)
            features[index, 10] = _safe_ratio(
                request.max_storage_slots, horizon
            )
        features[index, 11] = float(request_id in observation.running_request_ids)
        features[index, 12] = float(request_id in observation.eligible_request_ids)
        request_segments = segments_by_request.get(request_id, [])
        features[index, 13] = _safe_ratio(len(request_segments), max_capacity)
        if request_segments:
            features[index, 14] = sum(
                float(segment.fidelity) for segment in request_segments
            ) / len(request_segments)
        features[index, 15] = _safe_ratio(
            inflight_by_request.get(request_id, 0), max(1, len(physical_nodes))
        )
        dag_state = dag_states.get(request_id)
        if dag_state is not None:
            operation_count = max(1, len(dag_state.operation_ids))
            features[index, 16] = len(dag_state.completed) / operation_count
            features[index, 17] = len(dag_state.dead) / operation_count

    selected_ids = set(active_builder.selected_variable_ids)
    legal_ids = set(active_builder.legal_action_ids())
    candidate_node_indices: list[int] = []
    candidate_legal: list[bool] = []
    max_route_nodes = max(
        (len(variable.route_nodes) for variable in variables),
        default=1,
    )
    for variable in variables:
        index = node_index[("candidate", variable.variable_id)]
        candidate_node_indices.append(index)
        legal = variable.variable_id in legal_ids
        candidate_legal.append(legal)
        node_types[index] = CANDIDATE_NODE
        features[index, CANDIDATE_NODE] = 1.0
        features[index, 4] = current_fraction
        features[index, 5] = remaining_fraction
        features[index, 6] = _safe_ratio(
            variable.start_slot - observation.slot, horizon
        )
        features[index, 7] = _safe_ratio(variable.duration_slots, horizon)
        features[index, 8] = _safe_ratio(variable.completion_latency, horizon)
        features[index, 9] = _safe_ratio(
            len(variable.route_nodes) - 1, max_route_nodes - 1
        )
        features[index, 10] = float(variable.expected_success_probability)
        features[index, 11] = float(variable.expected_fidelity or 0.0)
        features[index, 12] = _safe_ratio(
            variable.base_candidate.demand_pairs, max_demand
        )
        total_usage = sum(item.amount for item in variable.resource_usage)
        memory_usage = sum(
            item.amount
            for item in variable.resource_usage
            if item.resource_id.startswith("memory:")
        )
        features[index, 13] = _safe_ratio(
            total_usage, max(1, len(variable.resource_usage) * max_capacity)
        )
        features[index, 14] = _safe_ratio(
            memory_usage, max(1, total_usage)
        )
        generation_fraction, swap_fraction, purification_fraction = (
            _operation_fractions(variable)
        )
        features[index, 15] = generation_fraction
        features[index, 16] = swap_fraction
        features[index, 17] = purification_fraction
        features[index, 18] = float(variable.variable_id in selected_ids)
        features[index, 19] = float(legal)

    resource_kinds = {
        "link": 10,
        "genlane": 11,
        "purify": 12,
        "bsm": 13,
        "swapnode": 14,
        "memory": 15,
    }
    for resource_id, resource_slot in resource_slots_ordered:
        index = node_index[("resource", (resource_id, resource_slot))]
        node_types[index] = RESOURCE_SLOT_NODE
        features[index, RESOURCE_SLOT_NODE] = 1.0
        features[index, 4] = current_fraction
        features[index, 5] = remaining_fraction
        capacity = capacities[resource_id]
        used = current_usage.get((resource_id, resource_slot), 0)
        features[index, 6] = _safe_ratio(capacity, max_capacity)
        features[index, 7] = _safe_ratio(used, capacity)
        features[index, 8] = _safe_ratio(max(0, capacity - used), capacity)
        features[index, 9] = _safe_ratio(
            resource_slot - observation.slot, horizon
        )
        type_index = resource_kinds.get(_resource_type(resource_id))
        if type_index is not None:
            features[index, type_index] = 1.0
        features[index, 16] = _safe_ratio(
            current_reservations.get(resource_id, 0), capacity
        )

    sources: list[int] = []
    destinations: list[int] = []
    relations: list[int] = []
    edge_attributes: list[tuple[float, float, float, float]] = []
    for left, right in edges:
        if ("physical", left) in node_index and ("physical", right) in node_index:
            _add_bidirectional_edge(
                sources,
                destinations,
                relations,
                edge_attributes,
                node_index[("physical", left)],
                node_index[("physical", right)],
                0,
            )

    for request_id in request_ids:
        request = request_specs.get(request_id)
        if request is None:
            continue
        request_index = node_index[("request", request_id)]
        _add_bidirectional_edge(
            sources,
            destinations,
            relations,
            edge_attributes,
            request_index,
            node_index[("physical", request.source)],
            2,
        )
        _add_bidirectional_edge(
            sources,
            destinations,
            relations,
            edge_attributes,
            request_index,
            node_index[("physical", request.destination)],
            4,
        )

    for variable in variables:
        candidate_index = node_index[("candidate", variable.variable_id)]
        request_key = ("request", variable.request_id)
        if request_key in node_index:
            _add_bidirectional_edge(
                sources,
                destinations,
                relations,
                edge_attributes,
                candidate_index,
                node_index[request_key],
                6,
            )
        route_denominator = max(1, len(variable.route_nodes) - 1)
        for position, physical_node in enumerate(variable.route_nodes):
            _add_bidirectional_edge(
                sources,
                destinations,
                relations,
                edge_attributes,
                candidate_index,
                node_index[("physical", physical_node)],
                8,
                (0.0, position / route_denominator, 0.0, 0.0),
            )
        for item in variable.resource_usage:
            capacity = capacities[item.resource_id]
            _add_bidirectional_edge(
                sources,
                destinations,
                relations,
                edge_attributes,
                candidate_index,
                node_index[("resource", (item.resource_id, item.slot))],
                10,
                (
                    _safe_ratio(item.amount, capacity),
                    0.0,
                    _safe_ratio(item.slot - observation.slot, horizon),
                    0.0,
                ),
            )

    for resource_id, resource_slot in resource_slots_ordered:
        resource_index = node_index[("resource", (resource_id, resource_slot))]
        resource_endpoints = _resource_nodes(resource_id)
        endpoint_denominator = max(1, len(resource_endpoints) - 1)
        for position, physical_node in enumerate(resource_endpoints):
            physical_key = ("physical", physical_node)
            if physical_key not in node_index:
                continue
            _add_bidirectional_edge(
                sources,
                destinations,
                relations,
                edge_attributes,
                resource_index,
                node_index[physical_key],
                12,
                (0.0, 0.0, 0.0, position / endpoint_denominator),
            )

    if sources:
        edge_index = torch.tensor(
            (sources, destinations), dtype=torch.long
        )
        edge_types = torch.tensor(relations, dtype=torch.long)
        edge_features = torch.tensor(edge_attributes, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_types = torch.empty((0,), dtype=torch.long)
        edge_features = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32)

    total_requests = max(
        1,
        len(observation.visible_request_ids)
        + len(observation.completed_request_ids)
        + len(observation.expired_request_ids),
    )
    legal_count = sum(candidate_legal)
    global_features = torch.tensor((
        current_fraction,
        _safe_ratio(
            observation.window_end_slot - observation.slot, horizon
        ),
        _safe_ratio(len(observation.visible_request_ids), total_requests),
        _safe_ratio(len(observation.eligible_request_ids), total_requests),
        _safe_ratio(len(observation.running_request_ids), total_requests),
        _safe_ratio(len(observation.completed_request_ids), total_requests),
        _safe_ratio(len(observation.expired_request_ids), total_requests),
        _safe_ratio(len(selected_ids), total_requests),
        _safe_ratio(legal_count, max(1, len(variables))),
        _safe_ratio(len(observation.physical_snapshot.segments), max_capacity),
    ), dtype=torch.float32)

    graph = RoutingGraph(
        node_features=features,
        node_types=node_types,
        edge_index=edge_index,
        edge_types=edge_types,
        edge_features=edge_features,
        global_features=global_features,
        candidate_node_indices=torch.tensor(
            candidate_node_indices, dtype=torch.long
        ),
        candidate_variable_ids=variable_ids,
        candidate_legal_mask=torch.tensor(candidate_legal, dtype=torch.bool),
    )
    return graph if device is None else graph.to(device)


__all__ = [
    "CANDIDATE_NODE",
    "EDGE_FEATURE_DIM",
    "GLOBAL_FEATURE_DIM",
    "NODE_FEATURE_DIM",
    "NODE_TYPE_COUNT",
    "PHYSICAL_NODE",
    "RELATION_COUNT",
    "REQUEST_NODE",
    "RESOURCE_SLOT_NODE",
    "RoutingGraph",
    "build_routing_graph",
]
