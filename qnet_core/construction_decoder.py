"""Exact capacity-aware canonical decoding for construction operation sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .construction_api import (
    ConstructionDAG,
    ConstructionOperation,
    ConstructionSnapshot,
    ResourceDemand,
)


@dataclass(frozen=True)
class FeasibilityResult:
    feasible: bool
    reason: str = ""


class CapacityFeasibilityOracle:
    """Incremental oracle for the hereditary resource-feasible family."""

    def __init__(
        self,
        capacities: Mapping[str, int],
        resident_holds: Mapping[str, ResourceDemand] | None = None,
    ):
        self.capacities = {str(key): int(value) for key, value in capacities.items()}
        if any(value < 0 for value in self.capacities.values()):
            raise ValueError("capacities must be non-negative")
        self.resident_holds = dict(resident_holds or {})

    @classmethod
    def from_snapshot(
        cls, snapshot: ConstructionSnapshot
    ) -> "CapacityFeasibilityOracle":
        usage = dict(snapshot.reservations)
        return cls(
            {
                resource: max(0, capacity - usage.get(resource, 0))
                for resource, capacity in snapshot.resource_capacities
            },
            {
                segment.segment_id: segment.held_resources
                for segment in snapshot.segments
            },
        )

    def check(self, operations: Iterable[ConstructionOperation]) -> FeasibilityResult:
        usage: dict[str, int] = {}
        seen_inputs: set[str] = set()
        for operation in operations:
            for segment_id in operation.input_segment_ids:
                if segment_id in seen_inputs:
                    return FeasibilityResult(False, "input segment consumed twice")
                seen_inputs.add(segment_id)
            for resource, amount in operation.resource_demand.items():
                usage[resource] = usage.get(resource, 0) + amount
                if usage[resource] > self.capacities.get(resource, 0):
                    return FeasibilityResult(False, f"capacity exceeded: {resource}")
        replacement: dict[str, int] = {}
        for operation in operations:
            for segment_id in operation.input_segment_ids:
                hold = self.resident_holds.get(segment_id)
                if hold is None:
                    continue
                for resource, amount in hold.items():
                    replacement[resource] = replacement.get(resource, 0) - amount
            for resource, amount in operation.output_resource_hold.items():
                replacement[resource] = replacement.get(resource, 0) + amount
        for resource, delta in replacement.items():
            if delta > self.capacities.get(resource, 0):
                return FeasibilityResult(
                    False, f"post-completion capacity exceeded: {resource}"
                )
        return FeasibilityResult(True)

    def can_add(self, prefix: Iterable[ConstructionOperation], operation: ConstructionOperation) -> bool:
        return self.check(tuple(prefix) + (operation,)).feasible


def canonical_decode(
    candidates: Sequence[ConstructionOperation],
    dag: ConstructionDAG,
    available_segment_ids: set[str],
    oracle: CapacityFeasibilityOracle,
    stop_legal: bool,
    selected_indices: Iterable[int],
) -> tuple[ConstructionOperation, ...]:
    """Decode a set action in canonical order without post-hoc projection."""

    ready = {
        operation.op_id: operation
        for operation in candidates
        if operation.op_id in dag.ready_ids(available_segment_ids)
    }
    ordered = sorted(ready.values(), key=lambda operation: operation.canonical_key)
    indices = tuple(int(index) for index in selected_indices)
    if any(index < 0 or index >= len(ordered) for index in indices):
        raise ValueError("selected operation index out of range")
    if tuple(sorted(set(indices))) != indices:
        raise ValueError("selected operation indices must be unique and sorted")
    selected: list[ConstructionOperation] = []
    for index in indices:
        operation = ordered[index]
        if not oracle.can_add(selected, operation):
            raise ValueError(f"infeasible operation set: {operation.op_id}")
        selected.append(operation)
    if not selected and not stop_legal:
        raise ValueError("empty operation set is legal only when stop_legal is true")
    return tuple(selected)


def canonical_decode_ready_set(
    candidates: Sequence[ConstructionOperation],
    oracle: CapacityFeasibilityOracle,
    stop_legal: bool,
    selected_indices: Iterable[int],
) -> tuple[ConstructionOperation, ...]:
    """Decode one canonical concurrent set from an already-ready batch.

    This is the cross-request form used by the policy.  Readiness is owned by
    the environment/executor; this function enforces the injective ordering,
    input exclusivity, capacity feasibility, and STOP condition.
    """

    ordered = tuple(sorted(candidates, key=lambda operation: operation.canonical_key))
    indices = tuple(int(index) for index in selected_indices)
    if any(index < 0 or index >= len(ordered) for index in indices):
        raise ValueError("selected operation index out of range")
    if tuple(sorted(set(indices))) != indices:
        raise ValueError("selected operation indices must be unique and sorted")
    selected: list[ConstructionOperation] = []
    for index in indices:
        operation = ordered[index]
        if not oracle.can_add(selected, operation):
            raise ValueError(f"infeasible operation set: {operation.op_id}")
        selected.append(operation)
    if not selected and not stop_legal:
        raise ValueError("empty operation set is legal only when stop_legal is true")
    return tuple(selected)


def feasible_operation_indices(
    candidates: Sequence[ConstructionOperation],
    dag: ConstructionDAG,
    available_segment_ids: set[str],
    oracle: CapacityFeasibilityOracle,
    selected_indices: Iterable[int] = (),
) -> tuple[bool, ...]:
    """Return the exact add-one mask for a specified canonical prefix."""

    ordered = sorted(
        (operation for operation in candidates if operation.op_id in dag.ready_ids(available_segment_ids)),
        key=lambda operation: operation.canonical_key,
    )
    indices = tuple(int(index) for index in selected_indices)
    if any(index < 0 or index >= len(ordered) for index in indices):
        raise ValueError("selected operation index out of range")
    if tuple(sorted(set(indices))) != indices:
        raise ValueError("selected operation indices must be unique and sorted")
    prefix = [ordered[index] for index in indices]
    if not oracle.check(prefix).feasible:
        raise ValueError("selected prefix is infeasible")
    selected = set(indices)
    return tuple(
        index not in selected and oracle.can_add(prefix, operation)
        for index, operation in enumerate(ordered)
    )
