"""Directly imitate exact construction-aware MILP decisions with a small GNN.

The graph is the sparse candidate--constraint incidence graph of the stage-one
packing model.  Labels are the final binary variables of the exact two-stage
MILP.  No LP trajectory or LP primal is used as supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping, Sequence

import numpy as np

from qnet_core.construction_api import OperationKind
from qnet_core.scenario import ScenarioConfig
from qnet_core.spec import EpisodeSpec

from .hard_decoder import validate_decoded_selection
from .milp_oracle import ConstructionAwareMILPOracle, DiscreteOracleSolution
from .teacher import LinearProgramStage, build_stage_one_lp
from .time_expansion import TimeExpandedCandidate
from .validate_construction_milp import build_construction_problem


VARIABLE_FEATURE_NAMES = (
    "start_slot",
    "completion_slot",
    "start_delay",
    "completion_delay",
    "completion_latency",
    "request_age",
    "request_remaining_ttl",
    "request_attempt_count",
    "is_retry",
    "hop_count",
    "path_index",
    "swap_tree_index",
    "construction_left_deep",
    "construction_balanced",
    "construction_swap_tree",
    "construction_other",
    "generation_fraction",
    "swap_fraction",
    "purification_fraction",
    "release_fraction",
    "resource_entry_count",
    "resource_pressure",
    "expected_fidelity",
    "expected_success_probability",
    "constraint_degree",
    "request_alternative_count",
    "source_degree",
    "destination_degree",
    "mean_internal_degree",
)

CONSTRAINT_FEATURE_NAMES = (
    "is_request",
    "is_resource_time",
    "rhs",
    "slot",
    "relative_slot",
    "degree",
    "total_capacity",
    "reserved_amount",
    "residual_fraction",
    "reserved_fraction",
    "request_age",
    "request_remaining_ttl",
    "request_attempt_count",
    "request_is_retry",
    "resource_link",
    "resource_genlane",
    "resource_purify",
    "resource_bsm",
    "resource_swapnode",
    "resource_memory",
)

GLOBAL_FEATURE_NAMES = (
    "variable_count",
    "constraint_count",
    "nonzero_count",
    "request_count",
    "horizon",
    "mean_constraint_rhs",
    "decision_slot",
    "decision_window",
    "remaining_horizon",
    "running_request_count",
    "reserved_resource_slot_count",
    "mean_reserved_fraction",
)


@dataclass(frozen=True)
class CandidateConstraintGraph:
    """Unlabelled candidate--constraint graph used by online inference."""

    seed: int
    variable_features: np.ndarray
    constraint_features: np.ndarray
    global_features: np.ndarray
    edge_variable_indices: np.ndarray
    edge_constraint_indices: np.ndarray
    edge_features: np.ndarray
    constraint_rhs: np.ndarray
    variables: tuple[TimeExpandedCandidate, ...]
    resource_capacities: Mapping[str, int]
    reserved_usage: Mapping[tuple[str, int], int]
    request_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        variable_count = len(self.variables)
        constraint_count = len(self.constraint_rhs)
        edge_count = len(self.edge_variable_indices)
        if self.variable_features.shape != (
            variable_count, len(VARIABLE_FEATURE_NAMES)
        ):
            raise ValueError("variable feature matrix has the wrong shape")
        if self.constraint_features.shape != (
            constraint_count, len(CONSTRAINT_FEATURE_NAMES)
        ):
            raise ValueError("constraint feature matrix has the wrong shape")
        if self.global_features.shape != (len(GLOBAL_FEATURE_NAMES),):
            raise ValueError("global feature vector has the wrong shape")
        if self.edge_variable_indices.shape != (edge_count,):
            raise ValueError("variable edge indices have the wrong shape")
        if self.edge_constraint_indices.shape != (edge_count,):
            raise ValueError("constraint edge indices have the wrong shape")
        if self.edge_features.shape != (edge_count, 2):
            raise ValueError("edge feature matrix has the wrong shape")
        arrays = (
            self.variable_features,
            self.constraint_features,
            self.global_features,
            self.edge_features,
            self.constraint_rhs,
        )
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("candidate graph arrays must be finite")
        if edge_count and (
            int(np.min(self.edge_variable_indices)) < 0
            or int(np.max(self.edge_variable_indices)) >= variable_count
        ):
            raise ValueError("variable edge index lies outside the graph")
        if edge_count and (
            int(np.min(self.edge_constraint_indices)) < 0
            or int(np.max(self.edge_constraint_indices)) >= constraint_count
        ):
            raise ValueError("constraint edge index lies outside the graph")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("request IDs must be unique")
        if set(variable.request_id for variable in self.variables) - set(
            self.request_ids
        ):
            raise ValueError("candidate belongs to an unknown request")
        if any(int(value) < 1 for value in self.resource_capacities.values()):
            raise ValueError("resource capacities must be positive")
        for (resource_id, slot), amount in self.reserved_usage.items():
            if resource_id not in self.resource_capacities:
                raise ValueError(
                    f"missing capacity for reserved resource: {resource_id}"
                )
            if int(slot) < 0 or not 0 < int(amount) <= int(
                self.resource_capacities[resource_id]
            ):
                raise ValueError("reserved usage lies outside capacity")


@dataclass(frozen=True)
class MILPGraphSample:
    seed: int
    variable_features: np.ndarray
    constraint_features: np.ndarray
    global_features: np.ndarray
    edge_variable_indices: np.ndarray
    edge_constraint_indices: np.ndarray
    edge_features: np.ndarray
    constraint_rhs: np.ndarray
    labels: np.ndarray
    variables: tuple[TimeExpandedCandidate, ...]
    resource_capacities: Mapping[str, int]
    reserved_usage: Mapping[tuple[str, int], int]
    request_ids: tuple[str, ...]
    optimal_completed_request_count: int
    optimal_expected_completed_request_mass: float
    optimal_total_completion_latency: float
    stage_one_mip_gap: float | None
    stage_two_mip_gap: float | None

    def __post_init__(self) -> None:
        variable_count = len(self.variables)
        constraint_count = len(self.constraint_rhs)
        edge_count = len(self.edge_variable_indices)
        if self.variable_features.shape != (
            variable_count, len(VARIABLE_FEATURE_NAMES)
        ):
            raise ValueError("variable feature matrix has the wrong shape")
        if self.constraint_features.shape != (
            constraint_count, len(CONSTRAINT_FEATURE_NAMES)
        ):
            raise ValueError("constraint feature matrix has the wrong shape")
        if self.global_features.shape != (len(GLOBAL_FEATURE_NAMES),):
            raise ValueError("global feature vector has the wrong shape")
        if self.labels.shape != (variable_count,):
            raise ValueError("MILP label vector has the wrong shape")
        if self.edge_variable_indices.shape != (edge_count,):
            raise ValueError("variable edge indices have the wrong shape")
        if self.edge_constraint_indices.shape != (edge_count,):
            raise ValueError("constraint edge indices have the wrong shape")
        if self.edge_features.shape != (edge_count, 2):
            raise ValueError("edge feature matrix has the wrong shape")
        arrays = (
            self.variable_features,
            self.constraint_features,
            self.global_features,
            self.edge_features,
            self.constraint_rhs,
            self.labels,
        )
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("MILP graph arrays must be finite")
        if edge_count and (
            int(np.min(self.edge_variable_indices)) < 0
            or int(np.max(self.edge_variable_indices)) >= variable_count
        ):
            raise ValueError("variable edge index lies outside the graph")
        if edge_count and (
            int(np.min(self.edge_constraint_indices)) < 0
            or int(np.max(self.edge_constraint_indices)) >= constraint_count
        ):
            raise ValueError("constraint edge index lies outside the graph")
        if not np.allclose(self.labels, np.rint(self.labels), atol=1e-7):
            raise ValueError("MILP labels must be binary")
        if np.any(self.labels < -1e-7) or np.any(self.labels > 1.0 + 1e-7):
            raise ValueError("MILP labels must lie in [0, 1]")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("request IDs must be unique")
        if set(variable.request_id for variable in self.variables) - set(
            self.request_ids
        ):
            raise ValueError("MILP variable belongs to an unknown request")
        if any(int(value) < 1 for value in self.resource_capacities.values()):
            raise ValueError("resource capacities must be positive")
        if int(np.sum(self.labels)) != self.optimal_completed_request_count:
            raise ValueError("MILP labels do not match the optimal request count")
        selected = tuple(
            variable
            for variable, label in zip(self.variables, self.labels)
            if label > 0.5
        )
        feasibility = validate_decoded_selection(
            selected,
            self.resource_capacities,
            self.reserved_usage,
        )
        if not feasibility.feasible:
            raise ValueError("MILP labels encode an infeasible selection")
        selected_expected_mass = float(sum(
            variable.expected_success_probability for variable in selected
        ))
        if not math.isclose(
            selected_expected_mass,
            self.optimal_expected_completed_request_mass,
            rel_tol=1e-9,
            abs_tol=1e-7,
        ):
            raise ValueError("MILP labels do not match expected completed mass")
        selected_latency = float(sum(
            variable.expected_success_probability * variable.completion_latency
            for variable in selected
        ))
        if not math.isclose(
            selected_latency,
            self.optimal_total_completion_latency,
            rel_tol=1e-9,
            abs_tol=1e-7,
        ):
            raise ValueError("MILP labels do not match the optimal latency")


@dataclass(frozen=True)
class GreedyDecodeResult:
    selected_variables: tuple[TimeExpandedCandidate, ...]
    feasible: bool
    completed_request_count: int
    total_completion_latency: float
    selected_variable_ids: tuple[str, ...]

    @property
    def expected_completed_request_mass(self) -> float:
        return float(sum(
            variable.expected_success_probability
            for variable in self.selected_variables
        ))


def _construction_family(kind: str) -> tuple[float, float, float, float]:
    return (
        float(kind == "left_deep"),
        float(kind == "balanced"),
        float(kind.startswith("swap_tree_")),
        float(
            kind not in {"left_deep", "balanced"}
            and not kind.startswith("swap_tree_")
        ),
    )


def _index_from_text(text: str, pattern: str) -> int:
    match = re.search(pattern, text)
    return 0 if match is None else int(match.group(1))


def _resource_type(resource_id: str | None) -> str:
    if resource_id is None:
        return ""
    return resource_id.split(":", 1)[0]


def _variable_feature_matrix(
    variables: Sequence[TimeExpandedCandidate],
    episode: EpisodeSpec,
    matrix,
    capacities: Mapping[str, int],
    *,
    decision_slot: int = 0,
    attempt_counts: Mapping[str, int] | None = None,
) -> np.ndarray:
    horizon = max(int(episode.horizon), 1)
    attempts = {
        str(request_id): int(count)
        for request_id, count in (attempt_counts or {}).items()
    }
    requests = {request.id: request for request in episode.requests}
    request_counts: dict[str, int] = {}
    for variable in variables:
        request_counts[variable.request_id] = (
            request_counts.get(variable.request_id, 0) + 1
        )
    maximum_request_count = max(request_counts.values(), default=1)
    path_indices = [
        _index_from_text(variable.candidate_id, r":path:(\d+):")
        for variable in variables
    ]
    tree_indices = [
        _index_from_text(variable.construction_kind, r"swap_tree_(\d+)")
        for variable in variables
    ]
    maximum_path_index = max(path_indices, default=0)
    maximum_tree_index = max(tree_indices, default=0)
    column_degree = np.diff(matrix.tocsc().indptr)

    topology_degree: dict[int, int] = {node: 0 for node in episode.nodes}
    topology_edges = {
        tuple(sorted((int(left), int(right))))
        for left, right in episode.edges
    }
    for left, right in topology_edges:
        topology_degree[left] += 1
        topology_degree[right] += 1
    maximum_topology_degree = max(topology_degree.values(), default=1)

    rows = []
    for index, variable in enumerate(variables):
        request = requests[variable.request_id]
        remaining_ttl = (
            episode.horizon - decision_slot
            if request.deadline is None
            else max(0, request.deadline - decision_slot)
        )
        attempt_count = attempts.get(variable.request_id, 0)
        operations = variable.base_candidate.dag.operations
        operation_count = max(len(operations), 1)
        kind_counts = {
            OperationKind.GEN: 0,
            OperationKind.SWAP: 0,
            OperationKind.PURIFY: 0,
            OperationKind.RELEASE: 0,
        }
        for operation in operations:
            kind_counts[operation.kind] = kind_counts.get(operation.kind, 0) + 1
        pressure = sum(
            usage.amount / capacities[usage.resource_id]
            for usage in variable.resource_usage
        )
        route = variable.route_nodes
        if any(node not in topology_degree for node in route):
            raise ValueError("candidate route contains an undeclared node")
        if any(
            tuple(sorted(edge)) not in topology_edges
            for edge in zip(route, route[1:])
        ):
            raise ValueError("candidate route contains a non-topology edge")
        internal = route[1:-1]
        rows.append((
            variable.start_slot / horizon,
            variable.completion_slot / horizon,
            max(0, variable.start_slot - decision_slot) / horizon,
            max(0, variable.completion_slot - decision_slot) / horizon,
            variable.completion_latency / horizon,
            max(0, decision_slot - request.arrival) / horizon,
            remaining_ttl / horizon,
            math.log1p(attempt_count) / math.log(9.0),
            float(attempt_count > 0),
            variable.base_candidate.hop_count / 8.0,
            path_indices[index] / max(maximum_path_index, 1),
            tree_indices[index] / max(maximum_tree_index, 1),
            *_construction_family(variable.construction_kind),
            kind_counts[OperationKind.GEN] / operation_count,
            kind_counts[OperationKind.SWAP] / operation_count,
            kind_counts[OperationKind.PURIFY] / operation_count,
            kind_counts[OperationKind.RELEASE] / operation_count,
            math.log1p(len(variable.resource_usage)) / math.log(129.0),
            math.log1p(pressure) / math.log(129.0),
            0.0 if variable.expected_fidelity is None
            else variable.expected_fidelity,
            variable.expected_success_probability,
            math.log1p(int(column_degree[index]))
            / math.log1p(max(int(column_degree.max()), 1)),
            request_counts[variable.request_id] / maximum_request_count,
            topology_degree.get(route[0], 0) / maximum_topology_degree,
            topology_degree.get(route[-1], 0) / maximum_topology_degree,
            (
                sum(topology_degree.get(node, 0) for node in internal)
                / max(len(internal), 1)
                / maximum_topology_degree
            ),
        ))
    return np.asarray(rows, dtype=np.float32).reshape(
        len(rows), len(VARIABLE_FEATURE_NAMES)
    )


def _constraint_feature_matrix(
    lp: LinearProgramStage,
    episode: EpisodeSpec,
    capacities: Mapping[str, int],
    *,
    decision_slot: int = 0,
    reserved_usage: Mapping[tuple[str, int], int] | None = None,
    attempt_counts: Mapping[str, int] | None = None,
    extra_request_ids: Sequence[str] = (),
    extra_resource_rows: Sequence[tuple[str, int, float]] = (),
) -> np.ndarray:
    reservations = dict(reserved_usage or {})
    attempts = {
        str(request_id): int(count)
        for request_id, count in (attempt_counts or {}).items()
    }
    requests = {request.id: request for request in episode.requests}
    row_degree = np.diff(lp.a_ub.tocsr().indptr)
    maximum_degree = max(int(row_degree.max()), 1)
    descriptors = [
        (
            descriptor.kind,
            descriptor.rhs,
            descriptor.resource_id,
            descriptor.slot,
            descriptor.request_id,
            int(row_degree[index]),
        )
        for index, descriptor in enumerate(lp.ub_constraints)
    ]
    descriptors.extend(
        ("request", 1.0, None, None, request_id, 0)
        for request_id in extra_request_ids
    )
    descriptors.extend(
        ("resource_time", rhs, resource_id, slot, None, 0)
        for resource_id, slot, rhs in extra_resource_rows
    )
    rows = []
    for kind, rhs, resource_id, slot, request_id, degree in descriptors:
        resource_type = _resource_type(resource_id)
        if kind == "resource_time":
            total_capacity = float(capacities[resource_id])
            reserved_amount = float(reservations.get(
                (resource_id, int(slot)), 0
            ))
        else:
            total_capacity = 1.0
            reserved_amount = 0.0
        request = (
            None
            if request_id is None
            else requests[request_id]
        )
        remaining_ttl = (
            0
            if request is None
            else (
                episode.horizon - decision_slot
                if request.deadline is None
                else max(0, request.deadline - decision_slot)
            )
        )
        attempt_count = (
            0 if request is None else attempts.get(request.id, 0)
        )
        rows.append((
            float(kind == "request"),
            float(kind == "resource_time"),
            math.log1p(rhs) / math.log(3.0),
            0.0 if slot is None
            else slot / max(episode.horizon, 1),
            0.0 if slot is None
            else max(0, slot - decision_slot)
            / max(episode.horizon, 1),
            math.log1p(degree) / math.log1p(maximum_degree),
            math.log1p(total_capacity) / math.log(9.0),
            math.log1p(reserved_amount) / math.log(9.0),
            rhs / max(total_capacity, 1.0),
            reserved_amount / max(total_capacity, 1.0),
            0.0 if request is None
            else max(0, decision_slot - request.arrival)
            / max(episode.horizon, 1),
            remaining_ttl / max(episode.horizon, 1),
            math.log1p(attempt_count) / math.log(9.0),
            float(attempt_count > 0),
            float(resource_type == "link"),
            float(resource_type == "genlane"),
            float(resource_type == "purify"),
            float(resource_type == "bsm"),
            float(resource_type == "swapnode"),
            float(resource_type == "memory"),
        ))
    return np.asarray(rows, dtype=np.float32).reshape(
        len(rows), len(CONSTRAINT_FEATURE_NAMES)
    )


def build_candidate_constraint_graph(
    seed: int,
    episode: EpisodeSpec,
    variables: Sequence[TimeExpandedCandidate],
    resource_capacities: Mapping[str, int],
    *,
    reserved_usage: Mapping[tuple[str, int], int] | None = None,
    decision_slot: int = 0,
    window_end_slot: int | None = None,
    running_request_ids: Sequence[str] = (),
    attempt_counts: Mapping[str, int] | None = None,
) -> CandidateConstraintGraph:
    """Build the exact graph seen by both the MILP teacher and the GNN."""

    ordered_variables = tuple(sorted(
        variables,
        key=lambda item: item.variable_id,
    ))
    if not ordered_variables:
        raise ValueError("candidate graph requires at least one variable")
    request_ids = tuple(str(request.id) for request in episode.requests)
    if not 0 <= decision_slot < episode.horizon:
        raise ValueError("decision slot must lie inside the episode horizon")
    resolved_window_end = (
        episode.horizon if window_end_slot is None else int(window_end_slot)
    )
    if not decision_slot < resolved_window_end <= episode.horizon:
        raise ValueError("decision window must lie inside the episode horizon")
    reservations = {
        (str(resource_id), int(slot)): int(amount)
        for (resource_id, slot), amount in (reserved_usage or {}).items()
        if int(amount) != 0
    }
    for (resource_id, slot), amount in reservations.items():
        if resource_id not in resource_capacities:
            raise ValueError(
                f"missing capacity for reserved resource: {resource_id}"
            )
        if not decision_slot <= slot < episode.horizon:
            raise ValueError(
                "reserved resource slot lies outside the visible future"
            )
        if not 0 < amount <= resource_capacities[resource_id]:
            raise ValueError("reserved usage must lie inside capacity")

    lp = build_stage_one_lp(
        ordered_variables,
        resource_capacities,
        reservations,
    )
    matrix = lp.a_ub.tocoo()
    represented_resource_slots = {
        (descriptor.resource_id, int(descriptor.slot))
        for descriptor in lp.ub_constraints
        if descriptor.kind == "resource_time"
    }
    represented_request_ids = {
        descriptor.request_id
        for descriptor in lp.ub_constraints
        if descriptor.kind == "request"
    }
    extra_request_ids = tuple(
        request_id
        for request_id in request_ids
        if request_id not in represented_request_ids
    )
    extra_resource_rows = tuple(
        (
            resource_id,
            slot,
            float(resource_capacities[resource_id] - amount),
        )
        for (resource_id, slot), amount in sorted(reservations.items())
        if (resource_id, slot) not in represented_resource_slots
    )
    rhs = np.concatenate((
        np.asarray(lp.b_ub, dtype=np.float32),
        np.ones(len(extra_request_ids), dtype=np.float32),
        np.asarray(
            [item[2] for item in extra_resource_rows], dtype=np.float32
        ),
    ))
    coefficients = np.asarray(matrix.data, dtype=np.float32)
    edge_features = np.column_stack((
        coefficients,
        coefficients / np.maximum(rhs[matrix.row], 1.0),
    )).astype(np.float32, copy=False)
    global_features = np.asarray((
        math.log1p(len(ordered_variables)) / math.log(5001.0),
        math.log1p(len(rhs)) / math.log(10001.0),
        math.log1p(matrix.nnz) / math.log(100001.0),
        len(request_ids) / 100.0,
        episode.horizon / 32.0,
        float(np.mean(rhs)) / 2.0 if len(rhs) else 0.0,
        decision_slot / max(episode.horizon, 1),
        (resolved_window_end - decision_slot) / max(episode.horizon, 1),
        (episode.horizon - decision_slot) / max(episode.horizon, 1),
        len(tuple(running_request_ids)) / 100.0,
        math.log1p(len(reservations)) / math.log(10001.0),
        (
            0.0
            if not reservations
            else float(np.mean([
                amount / resource_capacities[resource_id]
                for (resource_id, _), amount in reservations.items()
            ]))
        ),
    ), dtype=np.float32)
    return CandidateConstraintGraph(
        seed=int(seed),
        variable_features=_variable_feature_matrix(
            ordered_variables,
            episode,
            lp.a_ub,
            resource_capacities,
            decision_slot=decision_slot,
            attempt_counts=attempt_counts,
        ),
        constraint_features=_constraint_feature_matrix(
            lp,
            episode,
            resource_capacities,
            decision_slot=decision_slot,
            reserved_usage=reservations,
            attempt_counts=attempt_counts,
            extra_request_ids=extra_request_ids,
            extra_resource_rows=extra_resource_rows,
        ),
        global_features=global_features,
        edge_variable_indices=np.asarray(matrix.col, dtype=np.int64),
        edge_constraint_indices=np.asarray(matrix.row, dtype=np.int64),
        edge_features=edge_features,
        constraint_rhs=rhs,
        variables=ordered_variables,
        resource_capacities=dict(resource_capacities),
        reserved_usage=reservations,
        request_ids=request_ids,
    )


def graph_sample_from_solution(
    seed: int,
    episode: EpisodeSpec,
    solution: DiscreteOracleSolution,
    resource_capacities: Mapping[str, int],
    *,
    reserved_usage: Mapping[tuple[str, int], int] | None = None,
    decision_slot: int = 0,
    window_end_slot: int | None = None,
    running_request_ids: Sequence[str] = (),
    attempt_counts: Mapping[str, int] | None = None,
) -> MILPGraphSample:
    """Convert one exact MILP solution to a label-safe bipartite graph."""

    graph = build_candidate_constraint_graph(
        seed,
        episode,
        solution.variables,
        resource_capacities,
        reserved_usage=reserved_usage,
        decision_slot=decision_slot,
        window_end_slot=window_end_slot,
        running_request_ids=running_request_ids,
        attempt_counts=attempt_counts,
    )
    solution_variable_ids = tuple(
        variable.variable_id for variable in solution.variables
    )
    graph_variable_ids = tuple(
        variable.variable_id for variable in graph.variables
    )
    if solution_variable_ids != graph_variable_ids:
        raise ValueError("MILP solution variables are not canonically ordered")
    if solution.stage_one_lp.variable_ids != graph_variable_ids:
        raise ValueError("MILP graph and solution use different variables")
    if not np.allclose(
        solution.stage_one_lp.b_ub,
        graph.constraint_rhs[:len(solution.stage_one_lp.b_ub)],
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError("MILP graph and solution use different capacities")
    return MILPGraphSample(
        seed=graph.seed,
        variable_features=graph.variable_features,
        constraint_features=graph.constraint_features,
        global_features=graph.global_features,
        edge_variable_indices=graph.edge_variable_indices,
        edge_constraint_indices=graph.edge_constraint_indices,
        edge_features=graph.edge_features,
        constraint_rhs=graph.constraint_rhs,
        labels=np.asarray(solution.stage_two.primal, dtype=np.float32),
        variables=graph.variables,
        resource_capacities=graph.resource_capacities,
        reserved_usage=graph.reserved_usage,
        request_ids=graph.request_ids,
        optimal_completed_request_count=solution.completed_request_count,
        optimal_expected_completed_request_mass=(
            solution.expected_completed_request_mass
        ),
        optimal_total_completion_latency=solution.total_completion_latency,
        stage_one_mip_gap=solution.stage_one.mip_gap,
        stage_two_mip_gap=solution.stage_two.mip_gap,
    )


def generate_milp_graph_sample(
    seed: int,
    scenario: ScenarioConfig,
    *,
    path_candidate_count: int = 4,
    swap_tree_count: int = 5,
    time_limit_seconds: float = 30.0,
) -> MILPGraphSample:
    """Generate one problem and label it with the exact two-stage MILP."""

    problem = build_construction_problem(
        seed,
        scenario,
        path_candidate_count=path_candidate_count,
        swap_tree_count=swap_tree_count,
    )
    solution = ConstructionAwareMILPOracle(
        time_limit_seconds=time_limit_seconds,
        mip_relative_gap=0.0,
    ).solve(problem.variables, problem.resource_capacities)
    return graph_sample_from_solution(
        seed,
        problem.episode,
        solution,
        problem.resource_capacities,
    )


def greedy_decode_scores(
    sample: CandidateConstraintGraph | MILPGraphSample,
    scores: Sequence[float],
    *,
    threshold: float = 0.5,
) -> GreedyDecodeResult:
    """Project binary candidate scores to a feasible plan without search."""

    score_array = np.asarray(scores, dtype=float)
    if score_array.shape != (len(sample.variables),):
        raise ValueError("score vector has the wrong shape")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must be finite")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    selected: list[TimeExpandedCandidate] = []
    selected_requests: set[str] = set()
    usage: dict[tuple[str, int], int] = {}
    order = sorted(
        range(len(sample.variables)),
        key=lambda index: (
            -score_array[index],
            sample.variables[index].completion_latency,
            sample.variables[index].variable_id,
        ),
    )
    for index in order:
        if score_array[index] < threshold:
            break
        variable = sample.variables[index]
        if variable.request_id in selected_requests:
            continue
        feasible = True
        for item in variable.resource_usage:
            key = (item.resource_id, item.slot)
            if (
                sample.reserved_usage.get(key, 0)
                + usage.get(key, 0)
                + item.amount
                > sample.resource_capacities[item.resource_id]
            ):
                feasible = False
                break
        if not feasible:
            continue
        selected.append(variable)
        selected_requests.add(variable.request_id)
        for item in variable.resource_usage:
            key = (item.resource_id, item.slot)
            usage[key] = usage.get(key, 0) + item.amount
    selected_tuple = tuple(sorted(
        selected, key=lambda variable: variable.variable_id
    ))
    feasibility = validate_decoded_selection(
        selected_tuple,
        sample.resource_capacities,
        sample.reserved_usage,
    )
    return GreedyDecodeResult(
        selected_variables=selected_tuple,
        feasible=feasibility.feasible,
        completed_request_count=len(selected_tuple),
        total_completion_latency=float(sum(
            variable.expected_success_probability * variable.completion_latency
            for variable in selected_tuple
        )),
        selected_variable_ids=tuple(
            variable.variable_id for variable in selected_tuple
        ),
    )


# Torch is intentionally imported below the data-generation code.  Exact MILP
# validation remains usable in lightweight environments without PyTorch.
try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised by optional envs
    torch = None
    nn = None


if torch is not None:

    @dataclass(frozen=True)
    class BatchedMILPGraph:
        variable_features: torch.Tensor
        constraint_features: torch.Tensor
        global_features: torch.Tensor
        edge_variable_indices: torch.Tensor
        edge_constraint_indices: torch.Tensor
        edge_features: torch.Tensor
        constraint_rhs: torch.Tensor
        labels: torch.Tensor
        success_probabilities: torch.Tensor
        variable_graph_indices: torch.Tensor
        constraint_graph_indices: torch.Tensor
        graph_count: int
        variable_slices: tuple[tuple[int, int], ...]


    def batch_graph_samples(
        samples: Sequence[CandidateConstraintGraph | MILPGraphSample],
        *,
        device: str | torch.device = "cpu",
    ) -> BatchedMILPGraph:
        if not samples:
            raise ValueError("at least one graph sample is required")
        variable_features = []
        constraint_features = []
        global_features = []
        edge_variables = []
        edge_constraints = []
        edge_features = []
        constraint_rhs = []
        labels = []
        success_probabilities = []
        variable_graph_indices = []
        constraint_graph_indices = []
        variable_slices = []
        variable_offset = 0
        constraint_offset = 0
        has_labels = [isinstance(sample, MILPGraphSample) for sample in samples]
        if any(has_labels) and not all(has_labels):
            raise ValueError("cannot mix labelled and unlabelled graphs")
        labelled = all(has_labels)
        for graph_index, sample in enumerate(samples):
            variable_count = len(sample.variables)
            constraint_count = len(sample.constraint_rhs)
            variable_features.append(sample.variable_features)
            constraint_features.append(sample.constraint_features)
            global_features.append(sample.global_features)
            edge_variables.append(
                sample.edge_variable_indices + variable_offset
            )
            edge_constraints.append(
                sample.edge_constraint_indices + constraint_offset
            )
            edge_features.append(sample.edge_features)
            constraint_rhs.append(sample.constraint_rhs)
            if labelled:
                labels.append(sample.labels)
            success_probabilities.append(np.asarray(
                [
                    variable.expected_success_probability
                    for variable in sample.variables
                ],
                dtype=np.float32,
            ))
            variable_graph_indices.append(np.full(
                variable_count, graph_index, dtype=np.int64
            ))
            constraint_graph_indices.append(np.full(
                constraint_count, graph_index, dtype=np.int64
            ))
            variable_slices.append((
                variable_offset, variable_offset + variable_count
            ))
            variable_offset += variable_count
            constraint_offset += constraint_count

        tensor = lambda values, dtype: torch.as_tensor(
            np.concatenate(values, axis=0), dtype=dtype, device=device
        )
        return BatchedMILPGraph(
            variable_features=tensor(variable_features, torch.float32),
            constraint_features=tensor(constraint_features, torch.float32),
            global_features=torch.as_tensor(
                np.stack(global_features), dtype=torch.float32, device=device
            ),
            edge_variable_indices=tensor(edge_variables, torch.long),
            edge_constraint_indices=tensor(edge_constraints, torch.long),
            edge_features=tensor(edge_features, torch.float32),
            constraint_rhs=tensor(constraint_rhs, torch.float32),
            labels=(
                tensor(labels, torch.float32)
                if labelled
                else torch.empty(
                    0, dtype=torch.float32, device=device
                )
            ),
            success_probabilities=tensor(
                success_probabilities, torch.float32
            ),
            variable_graph_indices=tensor(
                variable_graph_indices, torch.long
            ),
            constraint_graph_indices=tensor(
                constraint_graph_indices, torch.long
            ),
            graph_count=len(samples),
            variable_slices=tuple(variable_slices),
        )


    def _segment_mean(
        values: torch.Tensor,
        indices: torch.Tensor,
        segment_count: int,
    ) -> torch.Tensor:
        result = values.new_zeros((segment_count, values.shape[-1]))
        result.index_add_(0, indices, values)
        count = values.new_zeros((segment_count, 1))
        count.index_add_(
            0, indices, values.new_ones((len(indices), 1))
        )
        return result / count.clamp_min(1.0)


    class _MLP(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.layers(values)


    class CandidateConstraintGNN(nn.Module):
        """Tripartite candidate--constraint--global message-passing model."""

        def __init__(
            self,
            *,
            hidden_dim: int = 64,
            layers: int = 3,
        ):
            super().__init__()
            if hidden_dim < 4:
                raise ValueError("hidden_dim must be at least four")
            if layers < 1:
                raise ValueError("layers must be positive")
            self.variable_encoder = _MLP(
                len(VARIABLE_FEATURE_NAMES), hidden_dim, hidden_dim
            )
            self.constraint_encoder = _MLP(
                len(CONSTRAINT_FEATURE_NAMES), hidden_dim, hidden_dim
            )
            self.global_encoder = _MLP(
                len(GLOBAL_FEATURE_NAMES), hidden_dim, hidden_dim
            )
            self.variable_to_constraint = nn.ModuleList()
            self.constraint_updates = nn.ModuleList()
            self.constraint_to_variable = nn.ModuleList()
            self.variable_updates = nn.ModuleList()
            self.global_updates = nn.ModuleList()
            self.variable_norms = nn.ModuleList()
            self.constraint_norms = nn.ModuleList()
            self.global_norms = nn.ModuleList()
            for _ in range(layers):
                self.variable_to_constraint.append(_MLP(
                    hidden_dim + 2, hidden_dim, hidden_dim
                ))
                self.constraint_updates.append(_MLP(
                    3 * hidden_dim, hidden_dim, hidden_dim
                ))
                self.constraint_to_variable.append(_MLP(
                    hidden_dim + 2, hidden_dim, hidden_dim
                ))
                self.variable_updates.append(_MLP(
                    3 * hidden_dim, hidden_dim, hidden_dim
                ))
                self.global_updates.append(_MLP(
                    3 * hidden_dim, hidden_dim, hidden_dim
                ))
                self.variable_norms.append(nn.LayerNorm(hidden_dim))
                self.constraint_norms.append(nn.LayerNorm(hidden_dim))
                self.global_norms.append(nn.LayerNorm(hidden_dim))
            self.output = _MLP(2 * hidden_dim, hidden_dim, 1)

        def forward(self, graph: BatchedMILPGraph) -> torch.Tensor:
            variable = self.variable_encoder(graph.variable_features)
            constraint = self.constraint_encoder(
                graph.constraint_features
            )
            global_state = self.global_encoder(graph.global_features)
            variable_edge = graph.edge_variable_indices
            constraint_edge = graph.edge_constraint_indices
            for layer_index in range(len(self.variable_updates)):
                messages = self.variable_to_constraint[layer_index](
                    torch.cat((
                        variable[variable_edge], graph.edge_features
                    ), dim=-1)
                )
                aggregate = constraint.new_zeros(constraint.shape)
                aggregate.index_add_(0, constraint_edge, messages)
                degree = constraint.new_zeros((len(constraint), 1))
                degree.index_add_(
                    0,
                    constraint_edge,
                    constraint.new_ones((len(constraint_edge), 1)),
                )
                aggregate = aggregate / degree.clamp_min(1.0)
                constraint_delta = self.constraint_updates[layer_index](
                    torch.cat((
                        constraint,
                        aggregate,
                        global_state[graph.constraint_graph_indices],
                    ), dim=-1)
                )
                constraint = self.constraint_norms[layer_index](
                    constraint + constraint_delta
                )

                messages = self.constraint_to_variable[layer_index](
                    torch.cat((
                        constraint[constraint_edge], graph.edge_features
                    ), dim=-1)
                )
                aggregate = variable.new_zeros(variable.shape)
                aggregate.index_add_(0, variable_edge, messages)
                degree = variable.new_zeros((len(variable), 1))
                degree.index_add_(
                    0,
                    variable_edge,
                    variable.new_ones((len(variable_edge), 1)),
                )
                aggregate = aggregate / degree.clamp_min(1.0)
                variable_delta = self.variable_updates[layer_index](
                    torch.cat((
                        variable,
                        aggregate,
                        global_state[graph.variable_graph_indices],
                    ), dim=-1)
                )
                variable = self.variable_norms[layer_index](
                    variable + variable_delta
                )

                global_delta = self.global_updates[layer_index](
                    torch.cat((
                        global_state,
                        _segment_mean(
                            variable,
                            graph.variable_graph_indices,
                            graph.graph_count,
                        ),
                        _segment_mean(
                            constraint,
                            graph.constraint_graph_indices,
                            graph.graph_count,
                        ),
                    ), dim=-1)
                )
                global_state = self.global_norms[layer_index](
                    global_state + global_delta
                )
            return self.output(torch.cat((
                variable,
                global_state[graph.variable_graph_indices],
            ), dim=-1)).squeeze(-1)


    def imitation_loss(
        logits: torch.Tensor,
        graph: BatchedMILPGraph,
        *,
        constraint_weight: float = 0.1,
        count_weight: float = 0.1,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if logits.shape != graph.labels.shape:
            raise ValueError("logits and labels have different shapes")
        if not bool(torch.all(torch.isfinite(logits)).item()):
            raise ValueError("logits must be finite")
        if (
            not math.isfinite(float(constraint_weight))
            or float(constraint_weight) < 0.0
        ):
            raise ValueError("constraint_weight must be finite and non-negative")
        if (
            not math.isfinite(float(count_weight))
            or float(count_weight) < 0.0
        ):
            raise ValueError("count_weight must be finite and non-negative")
        graph_losses = []
        graph_bce = []
        graph_constraint_penalties = []
        graph_mass_penalties = []
        probabilities = torch.sigmoid(logits)
        loads = graph.constraint_rhs.new_zeros(
            graph.constraint_rhs.shape
        )
        loads.index_add_(
            0,
            graph.edge_constraint_indices,
            probabilities[graph.edge_variable_indices]
            * graph.edge_features[:, 0],
        )
        predicted_mass = probabilities.new_zeros(graph.graph_count)
        target_mass = probabilities.new_zeros(graph.graph_count)
        predicted_mass.index_add_(
            0,
            graph.variable_graph_indices,
            probabilities * graph.success_probabilities,
        )
        target_mass.index_add_(
            0,
            graph.variable_graph_indices,
            graph.labels * graph.success_probabilities,
        )
        expected_mass_penalties = (
            (predicted_mass - target_mass)
            / target_mass.clamp_min(1.0)
        ) ** 2
        constraint_violations = torch.relu(
            loads / graph.constraint_rhs.clamp_min(1.0) - 1.0
        ) ** 2
        for graph_index, (start, end) in enumerate(graph.variable_slices):
            labels = graph.labels[start:end]
            graph_logits = logits[start:end]
            positives = labels.sum().clamp_min(1.0)
            negatives = (len(labels) - labels.sum()).clamp_min(1.0)
            positive_weight = (negatives / positives).clamp(max=50.0)
            bce = nn.functional.binary_cross_entropy_with_logits(
                graph_logits,
                labels,
                pos_weight=positive_weight,
            )
            constraint_mask = graph.constraint_graph_indices == graph_index
            constraint_penalty = constraint_violations[
                constraint_mask
            ].mean()
            expected_mass_penalty = expected_mass_penalties[graph_index]
            graph_bce.append(bce)
            graph_constraint_penalties.append(constraint_penalty)
            graph_mass_penalties.append(expected_mass_penalty)
            graph_losses.append(
                bce
                + float(constraint_weight) * constraint_penalty
                + float(count_weight) * expected_mass_penalty
            )
        loss = torch.stack(graph_losses).mean()
        bce = torch.stack(graph_bce).mean()
        constraint_penalty = torch.stack(
            graph_constraint_penalties
        ).mean()
        expected_mass_penalty = torch.stack(graph_mass_penalties).mean()
        return loss, {
            "bce": float(bce.detach()),
            "constraint_penalty": float(constraint_penalty.detach()),
            "expected_mass_penalty": float(expected_mass_penalty.detach()),
            # Compatibility alias for existing result parsers.
            "count_penalty": float(expected_mass_penalty.detach()),
        }


else:

    def batch_graph_samples(*args, **kwargs):  # pragma: no cover
        raise ModuleNotFoundError(
            "PyTorch is required for MILP imitation; run in the project "
            "Conda environment"
        )
