"""Small resource--time packing utilities used by baselines and audits."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .time_expansion import (
    TimeExpandedCandidate,
    TimeExpansionResult,
    normalize_reserved_usage,
)


@dataclass(frozen=True)
class PackingFeasibility:
    feasible: bool
    violations: tuple[str, ...]
    selected_request_count: int
    used_resource_slot_count: int


@dataclass(frozen=True)
class PackingSolution:
    variables: tuple[TimeExpandedCandidate, ...]
    scores: tuple[float, ...]
    request_ids: tuple[str, ...]
    selected_variables: tuple[TimeExpandedCandidate, ...]
    feasibility: PackingFeasibility
    strategy: str

    @property
    def completed_request_count(self) -> int:
        return len(self.selected_variables)

    @property
    def total_completion_latency(self) -> float:
        return float(sum(
            variable.completion_latency for variable in self.selected_variables
        ))

    @property
    def expected_completed_request_mass(self) -> float:
        return float(sum(
            variable.expected_success_probability
            for variable in self.selected_variables
        ))

    @property
    def expected_total_completion_latency(self) -> float:
        return float(sum(
            variable.expected_success_probability * variable.completion_latency
            for variable in self.selected_variables
        ))

    @property
    def selected_by_request(self) -> dict[str, TimeExpandedCandidate]:
        return {
            variable.request_id: variable
            for variable in self.selected_variables
        }

    @property
    def rejected_request_ids(self) -> tuple[str, ...]:
        selected = {variable.request_id for variable in self.selected_variables}
        return tuple(
            request_id
            for request_id in self.request_ids
            if request_id not in selected
        )


def _selection_usage(
    variables: Sequence[TimeExpandedCandidate],
) -> dict[tuple[str, int], int]:
    usage: dict[tuple[str, int], int] = {}
    for variable in variables:
        for item in variable.resource_usage:
            key = (item.resource_id, item.slot)
            usage[key] = usage.get(key, 0) + item.amount
    return usage


def validate_packing_selection(
    selected: Sequence[TimeExpandedCandidate],
    resource_capacities: Mapping[str, int],
    reserved_usage: Mapping[tuple[str, int], int] | None = None,
) -> PackingFeasibility:
    """Validate request uniqueness and every resource--slot capacity."""

    capacities = {
        str(resource_id): int(capacity)
        for resource_id, capacity in resource_capacities.items()
    }
    violations: list[str] = []
    seen_requests: set[str] = set()
    for variable in selected:
        if variable.request_id in seen_requests:
            violations.append(f"request selected twice: {variable.request_id}")
        seen_requests.add(variable.request_id)
    usage = normalize_reserved_usage(reserved_usage, capacities)
    for key, amount in _selection_usage(selected).items():
        usage[key] = usage.get(key, 0) + amount
    for (resource_id, slot), amount in sorted(usage.items()):
        if resource_id not in capacities:
            violations.append(f"missing capacity: {resource_id}")
        elif amount > capacities[resource_id]:
            violations.append(
                f"capacity exceeded: {resource_id}@{slot}: "
                f"{amount}>{capacities[resource_id]}"
            )
    return PackingFeasibility(
        feasible=not violations,
        violations=tuple(violations),
        selected_request_count=len(seen_requests),
        used_resource_slot_count=len(usage),
    )


class _PackingState:
    def __init__(
        self,
        capacities: Mapping[str, int],
        reserved_usage: Mapping[tuple[str, int], int] | None = None,
    ) -> None:
        self.capacities = {
            str(resource_id): int(capacity)
            for resource_id, capacity in capacities.items()
        }
        if any(capacity < 0 for capacity in self.capacities.values()):
            raise ValueError("resource capacities cannot be negative")
        self.selected: dict[str, TimeExpandedCandidate] = {}
        self.usage = normalize_reserved_usage(reserved_usage, self.capacities)

    def can_add(self, variable: TimeExpandedCandidate) -> bool:
        if variable.request_id in self.selected:
            return False
        delta: dict[tuple[str, int], int] = {}
        for item in variable.resource_usage:
            if item.resource_id not in self.capacities:
                raise ValueError(
                    f"missing capacity for resource: {item.resource_id}"
                )
            key = (item.resource_id, item.slot)
            delta[key] = delta.get(key, 0) + item.amount
        return all(
            self.usage.get(key, 0) + amount <= self.capacities[key[0]]
            for key, amount in delta.items()
        )

    def add(self, variable: TimeExpandedCandidate) -> None:
        if not self.can_add(variable):
            raise ValueError(
                f"cannot add infeasible candidate: {variable.variable_id}"
            )
        self.selected[variable.request_id] = variable
        for item in variable.resource_usage:
            key = (item.resource_id, item.slot)
            self.usage[key] = self.usage.get(key, 0) + item.amount


def greedy_feasible_projection(
    expanded: TimeExpansionResult | Sequence[TimeExpandedCandidate],
    resource_capacities: Mapping[str, int],
    scores: Mapping[str, float] | Sequence[float],
    *,
    request_ids: Sequence[str] | None = None,
    reserved_usage: Mapping[tuple[str, int], int] | None = None,
    support_tolerance: float = 1e-9,
) -> PackingSolution:
    """Pack candidates once in descending score order without local search."""

    if support_tolerance < 0.0:
        raise ValueError("support_tolerance cannot be negative")
    raw_variables = (
        expanded.variables
        if isinstance(expanded, TimeExpansionResult)
        else expanded
    )
    variables = tuple(sorted(raw_variables, key=lambda item: item.variable_id))
    if isinstance(scores, Mapping):
        missing = [
            variable.variable_id
            for variable in variables
            if variable.variable_id not in scores
        ]
        if missing:
            raise ValueError(f"missing projection score: {missing[0]}")
        ordered_scores = tuple(
            float(scores[variable.variable_id]) for variable in variables
        )
    else:
        ordered_scores = tuple(float(value) for value in scores)
        if len(ordered_scores) != len(variables):
            raise ValueError("projection score vector has the wrong length")
    if any(
        not math.isfinite(score) or not 0.0 <= score <= 1.0
        for score in ordered_scores
    ):
        raise ValueError("projection scores must be finite and lie in [0, 1]")
    score_by_id = {
        variable.variable_id: score
        for variable, score in zip(variables, ordered_scores, strict=True)
    }

    variable_request_ids = {variable.request_id for variable in variables}
    if request_ids is None:
        ordered_request_ids = tuple(sorted(variable_request_ids))
    else:
        ordered_request_ids = tuple(str(item) for item in request_ids)
        if len(set(ordered_request_ids)) != len(ordered_request_ids):
            raise ValueError("request_ids must be unique")
        undeclared = variable_request_ids - set(ordered_request_ids)
        if undeclared:
            raise ValueError(
                "candidate belongs to undeclared request: "
                f"{sorted(undeclared)[0]}"
            )

    ranked = sorted(
        (
            variable
            for variable in variables
            if score_by_id[variable.variable_id] > support_tolerance
        ),
        key=lambda variable: (
            -score_by_id[variable.variable_id],
            variable.variable_id,
        ),
    )
    state = _PackingState(resource_capacities, reserved_usage)
    for variable in ranked:
        if state.can_add(variable):
            state.add(variable)
    selected_variables = tuple(sorted(
        state.selected.values(),
        key=lambda variable: variable.variable_id,
    ))
    feasibility = validate_packing_selection(
        selected_variables,
        resource_capacities,
        reserved_usage,
    )
    if not feasibility.feasible:
        raise RuntimeError(
            "greedy packing produced an infeasible plan: "
            f"{feasibility.violations[0]}"
        )
    return PackingSolution(
        variables=variables,
        scores=ordered_scores,
        request_ids=ordered_request_ids,
        selected_variables=selected_variables,
        feasibility=feasibility,
        strategy="score_order_greedy",
    )


def decode_continuous_primal(
    variables: Sequence[TimeExpandedCandidate],
    primal: Sequence[float],
    resource_capacities: Mapping[str, int],
    *,
    reserved_usage: Mapping[tuple[str, int], int] | None = None,
) -> PackingSolution:
    """Round a continuous LP/GNN primal into one feasible discrete schedule.

    This is the entire decoder: rank every time-expanded candidate by its
    fractional value and greedily accept it while its request is unselected
    and every resource--slot capacity is respected.  There is no local search,
    no request-priority rule, and no topology-specific repair.
    """

    ordered = tuple(sorted(variables, key=lambda item: item.variable_id))
    values = np.asarray(primal, dtype=float)
    if values.shape != (len(ordered),):
        raise ValueError("primal vector length does not match variable count")
    if not np.isfinite(values).all():
        raise ValueError("continuous primal must be finite")
    scores = np.clip(values, 0.0, 1.0)
    return greedy_feasible_projection(
        ordered,
        resource_capacities,
        scores,
        reserved_usage=reserved_usage,
    )
