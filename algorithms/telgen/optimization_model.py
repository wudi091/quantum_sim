"""Single-stage expected-delay packing LP shared by MILP and GNN.

The planning layer makes one binary/continuous variable for every feasible
``(request, route, construction, start-slot)`` candidate.  The model has one
objective only: minimize the expected censored completion delay of the
requests represented by the planning window.

For request ``r`` let ``D_r`` be its censoring delay (the delay at the episode
horizon or request deadline), and let candidate ``j`` have completion delay
``L_j`` and success probability ``p_j``.  Selecting that candidate changes
the request's expected delay from ``D_r`` to
``p_j L_j + (1-p_j)D_r``.  After removing the constant ``sum_r D_r``, the LP
minimizes ``sum_j p_j (L_j-D_r) x_j``.  Keeping the constant on the model is
important for reporting the actual expected delay, while the sparse reduced
objective is what is passed to the LP/MILP solver and to the GNN graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
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
class PackingModel:
    """Sparse single-stage LP/MILP data.

    ``objective`` is the reduced minimization vector.  ``objective_constant``
    is the request-censoring constant that must be added to ``objective @ x``
    to obtain the actual expected total delay.
    """

    name: str
    variable_ids: tuple[str, ...]
    objective: np.ndarray
    a_ub: spmatrix
    b_ub: np.ndarray
    ub_constraints: tuple[ConstraintDescriptor, ...]
    a_eq: spmatrix
    b_eq: np.ndarray
    eq_constraints: tuple[ConstraintDescriptor, ...]
    objective_constant: float = 0.0
    request_censoring_latencies: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        variable_count = len(self.variable_ids)
        if self.objective.shape != (variable_count,):
            raise ValueError("objective has the wrong shape")
        if not np.isfinite(self.objective).all():
            raise ValueError("objective coefficients must be finite")
        if self.a_ub.shape != (len(self.b_ub), variable_count):
            raise ValueError("A_ub has the wrong shape")
        if self.a_eq.shape != (len(self.b_eq), variable_count):
            raise ValueError("A_eq has the wrong shape")
        if len(self.ub_constraints) != len(self.b_ub):
            raise ValueError("inequality metadata does not match A_ub")
        if len(self.eq_constraints) != len(self.b_eq):
            raise ValueError("equality metadata does not match A_eq")
        if not math.isfinite(float(self.objective_constant)):
            raise ValueError("objective_constant must be finite")
        if float(self.objective_constant) < 0.0:
            raise ValueError("objective_constant cannot be negative")
        previous: str | None = None
        total = 0.0
        for request_id, latency in self.request_censoring_latencies:
            if not request_id:
                raise ValueError("request censoring IDs must be non-empty")
            if previous is not None and request_id <= previous:
                raise ValueError(
                    "request censoring latencies must be sorted and unique"
                )
            value = float(latency)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "request censoring latencies must be finite and non-negative"
                )
            previous = request_id
            total += value
        if not math.isclose(
            total,
            float(self.objective_constant),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "objective_constant must equal the sum of request censoring "
                "latencies"
            )

    @property
    def objective_offset(self) -> float:
        """Readability alias for the reduced-objective offset."""

        return float(self.objective_constant)

    @property
    def request_censoring_latency_map(self) -> dict[str, float]:
        return {
            request_id: float(latency)
            for request_id, latency in self.request_censoring_latencies
        }


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


def _resolve_request_censoring_latencies(
    variables: Sequence[TimeExpandedCandidate],
    supplied: Mapping[str, float] | None,
) -> tuple[tuple[str, float], ...]:
    """Resolve one censoring delay for every represented/requested request.

    A caller with an ``EpisodeSpec`` should provide the explicit map so that
    requests with no feasible candidate still receive their horizon/deadline
    penalty.  Direct low-level callers may omit it; in that case the largest
    candidate latency for each request is the conservative censoring boundary.
    """

    inferred: dict[str, float] = {}
    for variable in variables:
        inferred[variable.request_id] = max(
            inferred.get(variable.request_id, 0.0),
            float(variable.completion_latency),
        )
    resolved: dict[str, float] = {}
    if supplied is not None:
        for raw_id, raw_latency in supplied.items():
            request_id = str(raw_id)
            latency = float(raw_latency)
            if not request_id:
                raise ValueError("request censoring IDs must be non-empty")
            if not math.isfinite(latency) or latency < 0.0:
                raise ValueError(
                    "request censoring latencies must be finite and non-negative"
                )
            resolved[request_id] = latency
    for request_id, inferred_latency in inferred.items():
        if request_id not in resolved:
            resolved[request_id] = inferred_latency
        elif resolved[request_id] + 1e-9 < inferred_latency:
            raise ValueError(
                f"censoring latency for {request_id} is earlier than a "
                "feasible candidate completion"
            )
    return tuple(sorted(resolved.items()))


def build_delay_model(
    variables: Sequence[TimeExpandedCandidate],
    resource_capacities: Mapping[str, int],
    reserved_usage: Mapping[tuple[str, int], int] | None = None,
    *,
    request_censoring_latencies: Mapping[str, float] | None = None,
) -> PackingModel:
    """Build the single-stage expected censored completion-delay model.

    The LP uses ``0 <= x_j <= 1``; the exact oracle changes only the variable
    domain to binary.  Request uniqueness, resource--time capacity, and all
    physical construction legality are represented by the shared candidate
    expansion and its sparse rows.
    """

    ordered = tuple(sorted(variables, key=lambda item: item.variable_id))
    a_ub, b_ub, descriptors = _base_constraints(
        ordered,
        resource_capacities,
        reserved_usage,
    )
    censoring = _resolve_request_censoring_latencies(
        ordered,
        request_censoring_latencies,
    )
    censoring_map = dict(censoring)
    objective = np.asarray(
        [
            float(item.expected_success_probability)
            * (float(item.completion_latency) - censoring_map[item.request_id])
            for item in ordered
        ],
        dtype=float,
    )
    return PackingModel(
        name="minimize_expected_censored_completion_latency",
        variable_ids=tuple(item.variable_id for item in ordered),
        objective=objective,
        a_ub=a_ub,
        b_ub=b_ub,
        ub_constraints=descriptors,
        a_eq=_empty_matrix(0, len(ordered)),
        b_eq=np.zeros(0, dtype=float),
        eq_constraints=(),
        objective_constant=float(sum(latency for _, latency in censoring)),
        request_censoring_latencies=censoring,
    )


def evaluate_expected_censored_delay(
    model: PackingModel,
    values: Sequence[float] | np.ndarray,
) -> float:
    """Evaluate the full expected censored delay for a model point."""

    point = np.asarray(values, dtype=float).reshape(-1)
    if point.shape != (len(model.variable_ids),):
        raise ValueError("model point has the wrong length")
    if not np.isfinite(point).all():
        raise ValueError("model point must be finite")
    return float(model.objective_constant + model.objective @ point)


__all__ = [
    "ConstraintDescriptor",
    "PackingModel",
    "build_delay_model",
    "evaluate_expected_censored_delay",
]
