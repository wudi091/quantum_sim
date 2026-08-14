"""Imitate exact MILP sets with a feasibility-masked autoregressive GNN.

The graph is the sparse candidate--constraint incidence graph of the stage-one
packing model.  At inference the GNN repeatedly chooses one candidate or STOP.
Candidates that would violate request uniqueness, resource--slot capacity, or
positive-success requirements are removed from the current action space before
the categorical decision.  The GNN still decides among every feasible action;
there is no post-hoc repair or local search.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping, Sequence

import numpy as np

from qnet_core.construction_api import OperationKind
from qnet_core.spec import EpisodeSpec

from .milp_oracle import DiscreteOracleSolution
from .optimization_model import PackingModelStage, build_stage_one_model
from .time_expansion import TimeExpandedCandidate


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

AUTOREGRESSIVE_ARCHITECTURE = "constraint_masked_autoregressive_v3"
AUTOREGRESSIVE_CHECKPOINT_SCHEMA_VERSION = 4
FEASIBILITY_TOLERANCE = 1e-7


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
        if any(
            variable.expected_success_probability <= 0.0
            for variable in selected
        ):
            raise ValueError(
                "MILP labels cannot select a zero-success candidate"
            )
        loads = np.zeros(constraint_count, dtype=float)
        np.add.at(
            loads,
            self.edge_constraint_indices,
            self.labels[self.edge_variable_indices]
            * self.edge_features[:, 0],
        )
        if np.any(loads > self.constraint_rhs + FEASIBILITY_TOLERANCE):
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
class SparsePackingIncidence:
    """Sparse ``A x <= b`` columns used to validate emitted actions."""

    constraint_rhs: np.ndarray
    variable_constraint_indices: tuple[np.ndarray, ...]
    variable_constraint_coefficients: tuple[np.ndarray, ...]
    positive_success_flags: np.ndarray

    def __post_init__(self) -> None:
        variable_count = len(self.variable_constraint_indices)
        if len(self.variable_constraint_coefficients) != variable_count:
            raise ValueError("packing incidence columns have different counts")
        if self.constraint_rhs.ndim != 1:
            raise ValueError("constraint RHS must be one-dimensional")
        if self.positive_success_flags.shape != (variable_count,):
            raise ValueError("success flags have the wrong shape")
        if not np.all(np.isfinite(self.constraint_rhs)):
            raise ValueError("constraint RHS must be finite")
        if np.any(self.constraint_rhs < -FEASIBILITY_TOLERANCE):
            raise ValueError("constraint RHS cannot be negative")
        constraint_count = len(self.constraint_rhs)
        for rows, coefficients in zip(
            self.variable_constraint_indices,
            self.variable_constraint_coefficients,
        ):
            if rows.ndim != 1 or coefficients.ndim != 1:
                raise ValueError("packing incidence columns must be vectors")
            if rows.shape != coefficients.shape:
                raise ValueError("packing incidence row/value shapes differ")
            if len(rows) and (
                int(np.min(rows)) < 0 or int(np.max(rows)) >= constraint_count
            ):
                raise ValueError("packing incidence row lies outside constraints")
            if not np.all(np.isfinite(coefficients)):
                raise ValueError("packing coefficients must be finite")
            if np.any(coefficients < -FEASIBILITY_TOLERANCE):
                raise ValueError(
                    "autoregressive state validation requires non-negative "
                    "packing "
                    "coefficients"
                )

    @property
    def variable_count(self) -> int:
        return len(self.variable_constraint_indices)


@dataclass(frozen=True)
class AutoregressiveState:
    """Remaining packing capacity after the GNN's previous actions."""

    residual_capacity: np.ndarray
    selected_mask: np.ndarray
    stopped: bool = False

    def __post_init__(self) -> None:
        if self.residual_capacity.ndim != 1:
            raise ValueError("residual capacity must be one-dimensional")
        if self.selected_mask.ndim != 1:
            raise ValueError("selected mask must be one-dimensional")
        if self.selected_mask.dtype != np.bool_:
            raise ValueError("selected mask must be boolean")
        if not np.all(np.isfinite(self.residual_capacity)):
            raise ValueError("residual capacity must be finite")
        if np.any(self.residual_capacity < -FEASIBILITY_TOLERANCE):
            raise ValueError("autoregressive state is infeasible")

    @property
    def selected_indices(self) -> tuple[int, ...]:
        return tuple(int(index) for index in np.flatnonzero(self.selected_mask))


@dataclass(frozen=True)
class AutoregressiveSelection:
    """The discrete plan emitted directly by the autoregressive GNN."""

    selected_variables: tuple[TimeExpandedCandidate, ...]
    selected_indices: tuple[int, ...]
    selected_variable_ids: tuple[str, ...]
    feasible: bool
    stopped: bool
    action_count: int
    total_completion_latency: float

    @property
    def completed_request_count(self) -> int:
        return len(self.selected_variables)

    @property
    def expected_completed_request_mass(self) -> float:
        return float(sum(
            variable.expected_success_probability
            for variable in self.selected_variables
        ))


@dataclass(frozen=True)
class AutoregressiveRollout:
    """Inference trace returned by the GNN policy itself."""

    selection: AutoregressiveSelection
    action_indices: tuple[int, ...]
    stopped_by_model: bool
    initial_candidate_probabilities: np.ndarray
    initial_stop_probability: float
    invalid_action_index: int | None = None
    invalid_action_reason: str | None = None

    def __post_init__(self) -> None:
        if self.initial_candidate_probabilities.ndim != 1:
            raise ValueError("initial candidate probabilities must be a vector")
        if not np.all(np.isfinite(self.initial_candidate_probabilities)):
            raise ValueError("initial candidate probabilities must be finite")
        if not math.isfinite(float(self.initial_stop_probability)):
            raise ValueError("initial STOP probability must be finite")
        if (self.invalid_action_index is None) != (
            self.invalid_action_reason is None
        ):
            raise ValueError(
                "invalid action index and reason must be reported together"
            )


def build_sparse_packing_incidence(
    sample: CandidateConstraintGraph | MILPGraphSample,
) -> SparsePackingIncidence:
    """Aggregate the graph edges into sparse columns of the exact MILP matrix."""

    variable_count = len(sample.variables)
    constraint_count = len(sample.constraint_rhs)
    columns: list[dict[int, float]] = [
        {} for _ in range(variable_count)
    ]
    coefficients = np.asarray(sample.edge_features[:, 0], dtype=float)
    if np.any(coefficients < -FEASIBILITY_TOLERANCE):
        raise ValueError(
            "autoregressive state validation supports packing constraints only"
        )
    for variable_index, constraint_index, coefficient in zip(
        sample.edge_variable_indices,
        sample.edge_constraint_indices,
        coefficients,
    ):
        variable = int(variable_index)
        constraint = int(constraint_index)
        if not 0 <= variable < variable_count:
            raise ValueError("variable edge index lies outside the graph")
        if not 0 <= constraint < constraint_count:
            raise ValueError("constraint edge index lies outside the graph")
        columns[variable][constraint] = (
            columns[variable].get(constraint, 0.0) + float(coefficient)
        )
    row_indices: list[np.ndarray] = []
    column_coefficients: list[np.ndarray] = []
    for column in columns:
        ordered = tuple(sorted(column.items()))
        row_indices.append(np.asarray(
            [row for row, _ in ordered], dtype=np.int64
        ))
        column_coefficients.append(np.asarray(
            [value for _, value in ordered], dtype=np.float32
        ))
    return SparsePackingIncidence(
        constraint_rhs=np.asarray(
            sample.constraint_rhs, dtype=np.float32
        ).copy(),
        variable_constraint_indices=tuple(row_indices),
        variable_constraint_coefficients=tuple(column_coefficients),
        positive_success_flags=np.asarray(
            [
                variable.expected_success_probability > 0.0
                for variable in sample.variables
            ],
            dtype=np.bool_,
        ),
    )


def initial_autoregressive_state(
    incidence: SparsePackingIncidence,
) -> AutoregressiveState:
    """Create the initial state ``r=b`` before the first GNN action."""

    return AutoregressiveState(
        residual_capacity=incidence.constraint_rhs.copy(),
        selected_mask=np.zeros(incidence.variable_count, dtype=np.bool_),
        stopped=False,
    )


def candidate_action_violation(
    incidence: SparsePackingIncidence,
    state: AutoregressiveState,
    variable_index: int,
) -> str | None:
    """Return why a candidate is absent from the current feasible action set."""

    if state.selected_mask.shape != (incidence.variable_count,):
        raise ValueError("autoregressive state has the wrong variable count")
    if state.residual_capacity.shape != incidence.constraint_rhs.shape:
        raise ValueError("autoregressive state has the wrong constraint count")
    index = int(variable_index)
    if not 0 <= index < incidence.variable_count:
        return "candidate_index_out_of_range"
    if state.stopped:
        return "candidate_after_stop"
    if state.selected_mask[index]:
        return "duplicate_candidate"
    if not incidence.positive_success_flags[index]:
        return "nonpositive_success_probability"
    rows = incidence.variable_constraint_indices[index]
    coefficients = incidence.variable_constraint_coefficients[index]
    if len(rows) and np.any(
        coefficients > state.residual_capacity[rows] + FEASIBILITY_TOLERANCE
    ):
        return "packing_constraint_violation"
    return None


def apply_candidate_action(
    incidence: SparsePackingIncidence,
    state: AutoregressiveState,
    variable_index: int,
) -> AutoregressiveState:
    """Apply one GNN candidate action as ``r <- r - A[:,j]``."""

    index = int(variable_index)
    if not 0 <= index < incidence.variable_count:
        raise IndexError("candidate action lies outside the graph")
    violation = candidate_action_violation(incidence, state, index)
    if violation is not None:
        raise ValueError(f"invalid candidate action: {violation}")
    residual = state.residual_capacity.copy()
    rows = incidence.variable_constraint_indices[index]
    residual[rows] -= incidence.variable_constraint_coefficients[index]
    residual[np.abs(residual) <= FEASIBILITY_TOLERANCE] = 0.0
    selected = state.selected_mask.copy()
    selected[index] = True
    return AutoregressiveState(
        residual_capacity=residual,
        selected_mask=selected,
        stopped=False,
    )


def apply_stop_action(state: AutoregressiveState) -> AutoregressiveState:
    """Terminate the learned action sequence without changing its selection."""

    return AutoregressiveState(
        residual_capacity=state.residual_capacity.copy(),
        selected_mask=state.selected_mask.copy(),
        stopped=True,
    )


def selection_from_state(
    sample: CandidateConstraintGraph | MILPGraphSample,
    incidence: SparsePackingIncidence,
    state: AutoregressiveState,
) -> AutoregressiveSelection:
    """Materialize the GNN's already-made actions; no repair is performed."""

    if state.selected_mask.shape != (len(sample.variables),):
        raise ValueError("selection state does not match graph variables")
    feasible = bool(np.all(
        state.residual_capacity >= -FEASIBILITY_TOLERANCE
    ))
    selected_indices = state.selected_indices
    selected_variables = tuple(
        sample.variables[index] for index in selected_indices
    )
    return AutoregressiveSelection(
        selected_variables=selected_variables,
        selected_indices=selected_indices,
        selected_variable_ids=tuple(
            variable.variable_id for variable in selected_variables
        ),
        feasible=feasible,
        stopped=state.stopped,
        action_count=len(selected_indices),
        total_completion_latency=float(sum(
            variable.expected_success_probability
            * variable.completion_latency
            for variable in selected_variables
        )),
    )


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
    model: PackingModelStage,
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
    row_degree = np.diff(model.a_ub.tocsr().indptr)
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
        for index, descriptor in enumerate(model.ub_constraints)
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

    model = build_stage_one_model(
        ordered_variables,
        resource_capacities,
        reservations,
    )
    matrix = model.a_ub.tocoo()
    represented_resource_slots = {
        (descriptor.resource_id, int(descriptor.slot))
        for descriptor in model.ub_constraints
        if descriptor.kind == "resource_time"
    }
    represented_request_ids = {
        descriptor.request_id
        for descriptor in model.ub_constraints
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
        np.asarray(model.b_ub, dtype=np.float32),
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
            model.a_ub,
            resource_capacities,
            decision_slot=decision_slot,
            attempt_counts=attempt_counts,
        ),
        constraint_features=_constraint_feature_matrix(
            model,
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
    if solution.stage_one_model.variable_ids != graph_variable_ids:
        raise ValueError("MILP graph and solution use different variables")
    if not np.allclose(
        solution.stage_one_model.b_ub,
        graph.constraint_rhs[:len(solution.stage_one_model.b_ub)],
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


    @dataclass(frozen=True)
    class EncodedCandidateConstraintGraph:
        """Static graph embeddings reused by every autoregressive step."""

        variable_embeddings: torch.Tensor
        constraint_embeddings: torch.Tensor
        global_embeddings: torch.Tensor


    @dataclass(frozen=True)
    class AutoregressiveActionLogits:
        """One learned categorical decision: all candidates plus STOP."""

        candidate_logits: torch.Tensor
        stop_logits: torch.Tensor

        def __post_init__(self) -> None:
            if self.candidate_logits.ndim != 1:
                raise ValueError("candidate logits must be one-dimensional")
            if self.stop_logits.ndim != 1:
                raise ValueError("STOP logits must be one-dimensional")


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
        """Feasibility-masked autoregressive candidate/STOP policy."""

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
            self.dynamic_constraint_to_variable = _MLP(
                hidden_dim + 3, hidden_dim, hidden_dim
            )
            self.candidate_action_head = _MLP(
                3 * hidden_dim + 4, hidden_dim, 1
            )
            self.stop_action_head = _MLP(
                3 * hidden_dim + 3, hidden_dim, 1
            )

        def encode(
            self,
            graph: BatchedMILPGraph,
        ) -> EncodedCandidateConstraintGraph:
            """Encode the static incidence graph exactly once per rollout."""

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
            return EncodedCandidateConstraintGraph(
                variable_embeddings=variable,
                constraint_embeddings=constraint,
                global_embeddings=global_state,
            )

        def action_logits(
            self,
            graph: BatchedMILPGraph,
            *,
            encoded: EncodedCandidateConstraintGraph | None = None,
            residual_capacity: torch.Tensor | None = None,
            selected_mask: torch.Tensor | None = None,
        ) -> AutoregressiveActionLogits:
            """Score candidates and STOP from static embeddings plus ``r``."""

            state = self.encode(graph) if encoded is None else encoded
            variable = state.variable_embeddings
            constraint = state.constraint_embeddings
            global_state = state.global_embeddings
            if variable.shape[0] != graph.variable_features.shape[0]:
                raise ValueError("encoded variables do not match the graph")
            if constraint.shape[0] != graph.constraint_features.shape[0]:
                raise ValueError("encoded constraints do not match the graph")
            if global_state.shape[0] != graph.graph_count:
                raise ValueError("encoded global states do not match the graph")
            residual = (
                graph.constraint_rhs
                if residual_capacity is None
                else residual_capacity
            )
            if residual.shape != graph.constraint_rhs.shape:
                raise ValueError("residual capacity has the wrong shape")
            if not bool(torch.all(torch.isfinite(residual)).item()):
                raise ValueError("residual capacity must be finite")
            chosen = (
                torch.zeros(
                    len(variable), dtype=torch.bool, device=variable.device
                )
                if selected_mask is None
                else selected_mask.to(device=variable.device, dtype=torch.bool)
            )
            if chosen.shape != (len(variable),):
                raise ValueError("selected mask has the wrong shape")

            rhs_scale = graph.constraint_rhs.clamp_min(1.0)
            residual_ratio = (residual / rhs_scale).clamp(min=0.0, max=1.0)
            variable_edge = graph.edge_variable_indices
            constraint_edge = graph.edge_constraint_indices
            dynamic_messages = self.dynamic_constraint_to_variable(
                torch.cat((
                    constraint[constraint_edge],
                    residual_ratio[constraint_edge, None],
                    graph.edge_features,
                ), dim=-1)
            )
            dynamic_aggregate = variable.new_zeros(variable.shape)
            dynamic_aggregate.index_add_(
                0, variable_edge, dynamic_messages
            )
            edge_degree = variable.new_zeros((len(variable), 1))
            edge_degree.index_add_(
                0,
                variable_edge,
                variable.new_ones((len(variable_edge), 1)),
            )
            dynamic_aggregate = dynamic_aggregate / edge_degree.clamp_min(1.0)

            edge_slack = residual_ratio[constraint_edge]
            slack_sum = variable.new_zeros(len(variable))
            slack_sum.index_add_(0, variable_edge, edge_slack)
            mean_slack = slack_sum / edge_degree.squeeze(-1).clamp_min(1.0)
            minimum_slack = variable.new_ones(len(variable))
            if len(variable_edge):
                if hasattr(minimum_slack, "scatter_reduce_"):
                    minimum_slack.scatter_reduce_(
                        0,
                        variable_edge,
                        edge_slack,
                        reduce="amin",
                        include_self=True,
                    )
                else:  # pragma: no cover - old PyTorch fallback
                    for edge_index in range(len(variable_edge)):
                        variable_index = int(variable_edge[edge_index])
                        minimum_slack[variable_index] = torch.minimum(
                            minimum_slack[variable_index],
                            edge_slack[edge_index],
                        )
            coefficient_pressure = variable.new_zeros(len(variable))
            coefficient_pressure.index_add_(
                0, variable_edge, graph.edge_features[:, 1]
            )
            selected_fraction = chosen.to(variable.dtype)
            candidate_logits = self.candidate_action_head(torch.cat((
                variable,
                dynamic_aggregate,
                global_state[graph.variable_graph_indices],
                mean_slack[:, None],
                minimum_slack[:, None],
                coefficient_pressure[:, None],
                selected_fraction[:, None],
            ), dim=-1)).squeeze(-1)

            variable_pool = _segment_mean(
                variable,
                graph.variable_graph_indices,
                graph.graph_count,
            )
            constraint_pool = _segment_mean(
                constraint * residual_ratio[:, None],
                graph.constraint_graph_indices,
                graph.graph_count,
            )
            graph_selected = variable.new_zeros(graph.graph_count)
            graph_selected.index_add_(
                0,
                graph.variable_graph_indices,
                chosen.to(variable.dtype),
            )
            graph_variable_count = variable.new_zeros(graph.graph_count)
            graph_variable_count.index_add_(
                0,
                graph.variable_graph_indices,
                variable.new_ones(len(variable)),
            )
            graph_selected = (
                graph_selected / graph_variable_count.clamp_min(1.0)
            )
            graph_residual = residual.new_zeros(graph.graph_count)
            graph_residual.index_add_(
                0, graph.constraint_graph_indices, residual_ratio
            )
            graph_constraint_count = residual.new_zeros(graph.graph_count)
            graph_constraint_count.index_add_(
                0,
                graph.constraint_graph_indices,
                residual.new_ones(len(residual)),
            )
            graph_residual = (
                graph_residual / graph_constraint_count.clamp_min(1.0)
            )
            remaining_fraction = 1.0 - graph_selected
            stop_logits = self.stop_action_head(torch.cat((
                global_state,
                variable_pool,
                constraint_pool,
                graph_selected[:, None],
                graph_residual[:, None],
                remaining_fraction[:, None],
            ), dim=-1)).squeeze(-1)
            return AutoregressiveActionLogits(
                candidate_logits=candidate_logits,
                stop_logits=stop_logits,
            )

        def forward(
            self,
            graph: BatchedMILPGraph,
            *,
            residual_capacity: torch.Tensor | None = None,
            selected_mask: torch.Tensor | None = None,
        ) -> AutoregressiveActionLogits:
            """Return the learned candidate/STOP categorical action logits."""

            return self.action_logits(
                graph,
                residual_capacity=residual_capacity,
                selected_mask=selected_mask,
            )


    def _categorical_log_probabilities(
        actions: AutoregressiveActionLogits,
        valid_candidate_flags: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Normalize feasible candidates and STOP into one action distribution."""

        candidate_logits = actions.candidate_logits
        if valid_candidate_flags is not None:
            valid = torch.as_tensor(
                valid_candidate_flags,
                dtype=torch.bool,
                device=candidate_logits.device,
            )
            if valid.shape != candidate_logits.shape:
                raise ValueError("valid candidate mask has the wrong shape")
            candidate_logits = candidate_logits.masked_fill(
                ~valid,
                float("-inf"),
            )
        logits = torch.cat((
            candidate_logits,
            actions.stop_logits[:1],
        ))
        if not bool(torch.all(torch.isfinite(
            actions.candidate_logits
        )).item()):
            raise ValueError("candidate logits must be finite before masking")
        if not bool(torch.all(torch.isfinite(actions.stop_logits)).item()):
            raise ValueError("STOP logits must be finite")
        return nn.functional.log_softmax(logits, dim=0)


    def _valid_candidate_flags(
        incidence: SparsePackingIncidence,
        state: AutoregressiveState,
    ) -> np.ndarray:
        """Return the exact feasible candidate action mask for one state."""

        return np.asarray([
            candidate_action_violation(incidence, state, index) is None
            for index in range(incidence.variable_count)
        ], dtype=np.bool_)


    def autoregressive_set_loss(
        model: CandidateConstraintGNN,
        samples: Sequence[MILPGraphSample],
        *,
        device: str | torch.device | None = None,
        target_mode: str = "set",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Teacher-force unordered MILP sets through candidate/STOP actions.

        At each non-terminal state every still-unselected MILP candidate is a
        correct action.  The loss maximizes their *combined* probability mass,
        so no arbitrary MILP variable ordering is used as a label sequence.
        """

        if not samples:
            raise ValueError("autoregressive loss requires at least one sample")
        if target_mode not in {"set", "fixed_order"}:
            raise ValueError(f"unknown autoregressive target mode: {target_mode}")
        resolved_device = (
            next(model.parameters()).device
            if device is None
            else torch.device(device)
        )
        graph = batch_graph_samples(samples, device=resolved_device)
        encoded = model.encode(graph)
        incidences = [
            build_sparse_packing_incidence(sample) for sample in samples
        ]
        states = [
            initial_autoregressive_state(incidence)
            for incidence in incidences
        ]
        remaining_teacher = [
            set(int(index) for index in np.flatnonzero(sample.labels > 0.5))
            for sample in samples
        ]
        finished = [False] * len(samples)
        sample_losses: list[list[torch.Tensor]] = [
            [] for _ in samples
        ]
        sample_action_losses: list[list[torch.Tensor]] = [
            [] for _ in samples
        ]
        sample_target_masses: list[list[torch.Tensor]] = [
            [] for _ in samples
        ]
        sample_correct_actions = [0] * len(samples)
        sample_action_counts = [0] * len(samples)
        stop_losses: list[torch.Tensor] = []
        correct_stops = 0
        valid_candidate_total = 0
        candidate_total = 0
        while not all(finished):
            residual_capacity = torch.as_tensor(
                np.concatenate([
                    state.residual_capacity for state in states
                ]),
                dtype=torch.float32,
                device=resolved_device,
            )
            selected_mask = torch.as_tensor(
                np.concatenate([
                    state.selected_mask for state in states
                ]),
                dtype=torch.bool,
                device=resolved_device,
            )
            batch_actions = model.action_logits(
                graph,
                encoded=encoded,
                residual_capacity=residual_capacity,
                selected_mask=selected_mask,
            )
            for graph_index, (
                sample,
                incidence,
                (start, end),
            ) in enumerate(zip(
                samples, incidences, graph.variable_slices
            )):
                if finished[graph_index]:
                    continue
                state = states[graph_index]
                actions = AutoregressiveActionLogits(
                    candidate_logits=batch_actions.candidate_logits[start:end],
                    stop_logits=batch_actions.stop_logits[
                        graph_index:graph_index + 1
                    ],
                )
                valid_candidates = _valid_candidate_flags(
                    incidence,
                    state,
                )
                valid_candidate_total += int(np.sum(valid_candidates))
                candidate_total += len(valid_candidates)
                log_probabilities = _categorical_log_probabilities(
                    actions,
                    valid_candidates,
                )
                targets = remaining_teacher[graph_index]
                if targets:
                    invalid_targets = sorted(
                        index for index in targets
                        if not valid_candidates[index]
                    )
                    if invalid_targets:
                        raise ValueError(
                            "MILP teacher set contains an invalid action: "
                            f"{invalid_targets}"
                        )
                    supervised_targets = (
                        sorted(targets)
                        if target_mode == "set"
                        else [min(targets)]
                    )
                    target_indices = torch.as_tensor(
                        supervised_targets,
                        dtype=torch.long,
                        device=resolved_device,
                    )
                    target_log_mass = torch.logsumexp(
                        log_probabilities[target_indices], dim=0
                    )
                    step_loss = -target_log_mass
                    sample_losses[graph_index].append(step_loss)
                    sample_action_losses[graph_index].append(step_loss)
                    sample_target_masses[graph_index].append(
                        target_log_mass.exp()
                    )
                    predicted_action = int(
                        torch.argmax(log_probabilities).item()
                    )
                    sample_correct_actions[graph_index] += int(
                        predicted_action in supervised_targets
                    )
                    sample_action_counts[graph_index] += 1

                    # Follow the model's preferred correct action.  This is a
                    # dynamic teacher oracle, not a pre-generated sequence.
                    teacher_logits = actions.candidate_logits[target_indices]
                    preferred_offset = int(torch.argmax(
                        teacher_logits.detach()
                    ).item())
                    chosen_teacher = int(
                        target_indices[preferred_offset].item()
                    )
                    states[graph_index] = apply_candidate_action(
                        incidence, state, chosen_teacher
                    )
                    targets.remove(chosen_teacher)
                    continue

                stop_loss = -log_probabilities[-1]
                sample_losses[graph_index].append(stop_loss)
                stop_losses.append(stop_loss)
                correct_stops += int(
                    int(torch.argmax(log_probabilities).item())
                    == len(sample.variables)
                )
                finished[graph_index] = True

        graph_losses = [
            torch.stack(losses).mean() for losses in sample_losses
        ]
        teacher_loss = torch.stack(graph_losses).mean()
        graph_action_losses = [
            (
                torch.stack(losses).mean()
                if losses else teacher_loss.new_zeros(())
            )
            for losses in sample_action_losses
        ]
        graph_target_masses = [
            (
                torch.stack(values).mean()
                if values else teacher_loss.new_ones(())
            )
            for values in sample_target_masses
        ]
        mean_action_loss = torch.stack(graph_action_losses).mean()
        mean_stop_loss = torch.stack(stop_losses).mean()
        mean_target_mass = torch.stack(graph_target_masses).mean()
        return teacher_loss, {
            "autoregressive_nll": float(teacher_loss.detach()),
            "candidate_set_nll": float(mean_action_loss.detach()),
            "stop_nll": float(mean_stop_loss.detach()),
            "valid_candidate_fraction": (
                valid_candidate_total / max(candidate_total, 1)
            ),
            "masked_candidate_fraction": (
                1.0 - valid_candidate_total / max(candidate_total, 1)
            ),
            "mean_target_probability_mass": float(
                mean_target_mass.detach()
            ),
            "candidate_action_accuracy": (
                float(np.mean([
                    correct / max(count, 1)
                    for correct, count in zip(
                        sample_correct_actions, sample_action_counts
                    )
                ]))
            ),
            "stop_accuracy": correct_stops / len(samples),
            "mean_teacher_action_count": float(np.mean([
                int(np.sum(sample.labels > 0.5)) for sample in samples
            ])),
        }


    def autoregressive_rollout(
        model: CandidateConstraintGNN,
        sample: CandidateConstraintGraph | MILPGraphSample,
        *,
        device: str | torch.device | None = None,
    ) -> AutoregressiveRollout:
        """Emit a discrete plan from the dynamically feasible action space."""

        resolved_device = (
            next(model.parameters()).device
            if device is None
            else torch.device(device)
        )
        graph = batch_graph_samples((sample,), device=resolved_device)
        incidence = build_sparse_packing_incidence(sample)
        state = initial_autoregressive_state(incidence)
        action_indices: list[int] = []
        initial_candidate_probabilities: np.ndarray | None = None
        initial_stop_probability: float | None = None
        stopped_by_model = False
        with torch.no_grad():
            encoded = model.encode(graph)
            for _ in range(len(sample.variables) + 1):
                actions = model.action_logits(
                    graph,
                    encoded=encoded,
                    residual_capacity=torch.as_tensor(
                        state.residual_capacity,
                        dtype=torch.float32,
                        device=resolved_device,
                    ),
                    selected_mask=torch.as_tensor(
                        state.selected_mask,
                        dtype=torch.bool,
                        device=resolved_device,
                    ),
                )
                valid_candidates = _valid_candidate_flags(incidence, state)
                log_probabilities = _categorical_log_probabilities(
                    actions,
                    valid_candidates,
                )
                probabilities = log_probabilities.exp()
                if initial_candidate_probabilities is None:
                    initial_candidate_probabilities = (
                        probabilities[:-1].cpu().numpy()
                    )
                    initial_stop_probability = float(probabilities[-1].item())
                action_index = int(torch.argmax(log_probabilities).item())
                if action_index == len(sample.variables):
                    state = apply_stop_action(state)
                    stopped_by_model = True
                    break
                action_indices.append(action_index)
                violation = candidate_action_violation(
                    incidence, state, action_index
                )
                if violation is not None:
                    raise RuntimeError(
                        "feasibility mask admitted an invalid action: "
                        f"{violation}"
                    )
                state = apply_candidate_action(
                    incidence, state, action_index
                )
            else:
                raise RuntimeError(
                    "masked autoregressive policy failed to emit STOP"
                )
        selection = selection_from_state(sample, incidence, state)
        if not selection.feasible:
            raise RuntimeError(
                "masked autoregressive policy produced an infeasible state"
            )
        if initial_candidate_probabilities is None:
            raise RuntimeError("autoregressive policy emitted no action logits")
        return AutoregressiveRollout(
            selection=selection,
            action_indices=tuple(action_indices),
            stopped_by_model=stopped_by_model,
            initial_candidate_probabilities=initial_candidate_probabilities,
            initial_stop_probability=float(initial_stop_probability),
            invalid_action_index=None,
            invalid_action_reason=None,
        )


else:

    class CandidateConstraintGNN:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "PyTorch is required for the autoregressive GNN"
            )

    def batch_graph_samples(*args, **kwargs):  # pragma: no cover
        raise ModuleNotFoundError(
            "PyTorch is required for MILP imitation; run in the project "
            "Conda environment"
        )

    def autoregressive_set_loss(*args, **kwargs):  # pragma: no cover
        raise ModuleNotFoundError(
            "PyTorch is required for the autoregressive GNN"
        )

    def autoregressive_rollout(*args, **kwargs):  # pragma: no cover
        raise ModuleNotFoundError(
            "PyTorch is required for the autoregressive GNN"
        )
