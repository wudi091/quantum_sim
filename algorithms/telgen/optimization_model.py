"""Sparse two-stage packing model shared by exact MILP and GNN graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, spmatrix

from .time_expansion import TimeExpandedCandidate, normalize_reserved_usage


@dataclass(frozen=True)
class ConstraintDescriptor:
    """Semantic identity of one request or resource--time constraint."""

    constraint_id: str
    kind: str
    rhs: float
    sense: str = "<="
    request_id: str | None = None
    resource_id: str | None = None
    slot: int | None = None


@dataclass(frozen=True)
class PackingModelStage:
    """Sparse objective and constraints for one lexicographic MILP stage."""

    name: str
    variable_ids: tuple[str, ...]
    objective: np.ndarray
    a_ub: spmatrix
    b_ub: np.ndarray
    ub_constraints: tuple[ConstraintDescriptor, ...]
    a_eq: spmatrix
    b_eq: np.ndarray
    eq_constraints: tuple[ConstraintDescriptor, ...]

    def __post_init__(self) -> None:
        variable_count = len(self.variable_ids)
        if self.objective.shape != (variable_count,):
            raise ValueError("objective has the wrong shape")
        if self.a_ub.shape != (len(self.b_ub), variable_count):
            raise ValueError("A_ub has the wrong shape")
        if self.a_eq.shape != (len(self.b_eq), variable_count):
            raise ValueError("A_eq has the wrong shape")
        if len(self.ub_constraints) != len(self.b_ub):
            raise ValueError("inequality metadata does not match A_ub")
        if len(self.eq_constraints) != len(self.b_eq):
            raise ValueError("equality metadata does not match A_eq")


def _empty_matrix(rows: int, columns: int) -> csr_matrix:
    return csr_matrix((rows, columns), dtype=float)


def _base_constraints(
    variables: Sequence[TimeExpandedCandidate],
    resource_capacities: Mapping[str, int],
    reserved_usage: Mapping[tuple[str, int], int] | None = None,
) -> tuple[csr_matrix, np.ndarray, tuple[ConstraintDescriptor, ...]]:
    variable_count = len(variables)
    request_rows: dict[str, list[int]] = {}
    resource_rows: dict[tuple[str, int], dict[int, int]] = {}
    for index, variable in enumerate(variables):
        request_rows.setdefault(variable.request_id, []).append(index)
        for usage in variable.resource_usage:
            resource_rows.setdefault(
                (usage.resource_id, usage.slot), {}
            )[index] = usage.amount

    row_indices: list[int] = []
    column_indices: list[int] = []
    data: list[float] = []
    rhs: list[float] = []
    descriptors: list[ConstraintDescriptor] = []
    for request_id in sorted(request_rows):
        row_index = len(rhs)
        indices = request_rows[request_id]
        row_indices.extend([row_index] * len(indices))
        column_indices.extend(indices)
        data.extend([1.0] * len(indices))
        rhs.append(1.0)
        descriptors.append(ConstraintDescriptor(
            constraint_id=f"request:{request_id}",
            kind="request",
            rhs=1.0,
            request_id=request_id,
        ))

    capacities = {
        str(resource_id): int(capacity)
        for resource_id, capacity in resource_capacities.items()
    }
    reservations = normalize_reserved_usage(reserved_usage, capacities)
    for (resource_id, slot), coefficients in sorted(resource_rows.items()):
        if resource_id not in capacities:
            raise ValueError(f"missing capacity for resource: {resource_id}")
        capacity = capacities[resource_id]
        if capacity < 1:
            raise ValueError("resource capacities must be positive")
        row_index = len(rhs)
        for index, amount in coefficients.items():
            row_indices.append(row_index)
            column_indices.append(index)
            data.append(float(amount))
        residual_capacity = capacity - reservations.get((resource_id, slot), 0)
        rhs.append(float(residual_capacity))
        descriptors.append(ConstraintDescriptor(
            constraint_id=f"resource:{resource_id}:slot:{slot}",
            kind="resource_time",
            rhs=float(residual_capacity),
            resource_id=resource_id,
            slot=slot,
        ))

    matrix = coo_matrix(
        (data, (row_indices, column_indices)),
        shape=(len(rhs), variable_count),
        dtype=float,
    ).tocsr()
    return matrix, np.asarray(rhs, dtype=float), tuple(descriptors)


def build_stage_one_model(
    variables: Sequence[TimeExpandedCandidate],
    resource_capacities: Mapping[str, int],
    reserved_usage: Mapping[tuple[str, int], int] | None = None,
) -> PackingModelStage:
    """Build expected-completion maximization as a minimization model."""

    ordered = tuple(sorted(variables, key=lambda item: item.variable_id))
    a_ub, b_ub, descriptors = _base_constraints(
        ordered,
        resource_capacities,
        reserved_usage,
    )
    variable_count = len(ordered)
    return PackingModelStage(
        name="maximize_expected_completed_requests",
        variable_ids=tuple(item.variable_id for item in ordered),
        objective=-np.asarray(
            [item.expected_success_probability for item in ordered],
            dtype=float,
        ),
        a_ub=a_ub,
        b_ub=b_ub,
        ub_constraints=descriptors,
        a_eq=_empty_matrix(0, variable_count),
        b_eq=np.zeros(0, dtype=float),
        eq_constraints=(),
    )


def build_stage_two_model(
    variables: Sequence[TimeExpandedCandidate],
    resource_capacities: Mapping[str, int],
    completed_mass: float,
    reserved_usage: Mapping[tuple[str, int], int] | None = None,
) -> PackingModelStage:
    """Fix optimal expected throughput and minimize expected latency."""

    ordered = tuple(sorted(variables, key=lambda item: item.variable_id))
    a_ub, b_ub, descriptors = _base_constraints(
        ordered,
        resource_capacities,
        reserved_usage,
    )
    success_probabilities = np.asarray(
        [item.expected_success_probability for item in ordered],
        dtype=float,
    )
    return PackingModelStage(
        name="minimize_expected_completion_latency",
        variable_ids=tuple(item.variable_id for item in ordered),
        objective=np.asarray(
            [
                item.expected_success_probability * item.completion_latency
                for item in ordered
            ],
            dtype=float,
        ),
        a_ub=a_ub,
        b_ub=b_ub,
        ub_constraints=descriptors,
        a_eq=csr_matrix(success_probabilities.reshape(1, -1)),
        b_eq=np.asarray([float(completed_mass)], dtype=float),
        eq_constraints=(ConstraintDescriptor(
            constraint_id="throughput:stage_one_optimum",
            kind="throughput_equality",
            rhs=float(completed_mass),
            sense="=",
        ),),
    )


__all__ = [
    "ConstraintDescriptor",
    "PackingModelStage",
    "build_stage_one_model",
    "build_stage_two_model",
]
