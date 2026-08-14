"""Incremental feasibility checks for concurrently launched operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .construction_api import (
    ConstructionOperation,
    ConstructionSnapshot,
    ResourceDemand,
)


@dataclass(frozen=True)
class FeasibilityResult:
    feasible: bool
    reason: str = ""


class CapacityFeasibilityOracle:
    """Check input exclusivity and resource capacity incrementally."""

    def __init__(
        self,
        capacities: Mapping[str, int],
        resident_holds: Mapping[str, ResourceDemand] | None = None,
    ):
        self.capacities = {
            str(resource_id): int(capacity)
            for resource_id, capacity in capacities.items()
        }
        if any(capacity < 0 for capacity in self.capacities.values()):
            raise ValueError("capacities must be non-negative")
        self.resident_holds = dict(resident_holds or {})

    @classmethod
    def from_snapshot(
        cls,
        snapshot: ConstructionSnapshot,
    ) -> "CapacityFeasibilityOracle":
        usage = dict(snapshot.reservations)
        declared = dict(snapshot.resource_capacities)
        overcommitted = [
            resource_id
            for resource_id, amount in usage.items()
            if amount > declared.get(resource_id, 0)
        ]
        if overcommitted:
            raise ValueError(
                "snapshot reservations exceed capacity: "
                f"{sorted(overcommitted)[0]}"
            )
        return cls(
            {
                resource_id: capacity - usage.get(resource_id, 0)
                for resource_id, capacity in declared.items()
            },
            {
                segment.segment_id: segment.held_resources
                for segment in snapshot.segments
            },
        )

    def check(
        self,
        operations: Iterable[ConstructionOperation],
    ) -> FeasibilityResult:
        usage: dict[str, int] = {}
        seen_inputs: set[str] = set()
        frozen = tuple(operations)
        for operation in frozen:
            for segment_id in operation.input_segment_ids:
                if segment_id in seen_inputs:
                    return FeasibilityResult(
                        False,
                        "input segment consumed twice",
                    )
                seen_inputs.add(segment_id)
            for resource_id, amount in operation.resource_demand.items():
                usage[resource_id] = usage.get(resource_id, 0) + amount
                if usage[resource_id] > self.capacities.get(resource_id, 0):
                    return FeasibilityResult(
                        False,
                        f"capacity exceeded: {resource_id}",
                    )

        replacement: dict[str, int] = {}
        for operation in frozen:
            for segment_id in operation.input_segment_ids:
                hold = self.resident_holds.get(segment_id)
                if hold is None:
                    continue
                for resource_id, amount in hold.items():
                    replacement[resource_id] = (
                        replacement.get(resource_id, 0) - amount
                    )
            for resource_id, amount in operation.output_resource_hold.items():
                replacement[resource_id] = (
                    replacement.get(resource_id, 0) + amount
                )
        for resource_id, delta in replacement.items():
            if delta > self.capacities.get(resource_id, 0):
                return FeasibilityResult(
                    False,
                    f"post-completion capacity exceeded: {resource_id}",
                )
        return FeasibilityResult(True)

    def can_add(
        self,
        prefix: Iterable[ConstructionOperation],
        operation: ConstructionOperation,
    ) -> bool:
        return self.check((*tuple(prefix), operation)).feasible


__all__ = ["CapacityFeasibilityOracle", "FeasibilityResult"]
