"""Boundary between routing/planning and the physical simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from .command_api import ResourceClaim, SwapAction, SwapLane


@dataclass(frozen=True)
class PhysicalCapabilities:
    """Planner-safe capacity limits exposed by a physical backend."""

    max_width: int
    memory_capacity: int
    node_memory_capacity: int | None


@dataclass(frozen=True)
class PhysicalResource:
    """Immutable routing view of one physical entangled pair."""

    pair_id: str
    left: int
    right: int
    fidelity: float
    born: int
    owner_request: str | None
    reserved_by: str | None
    lane: int | None

    @property
    def endpoints(self) -> tuple[int, int]:
        return self.left, self.right


@dataclass(frozen=True)
class LaneExecutionResult:
    """Physical result for one independently executed swap lane."""

    lane: int
    output_pair_id: str | None
    consumed_pair_ids: tuple[str, ...]
    untouched_pair_ids: tuple[str, ...]
    surviving_pair_ids: tuple[str, ...]
    failed_action_index: int | None
    attempted_swaps: int

    @property
    def success(self) -> bool:
        return self.output_pair_id is not None


class PhysicalBackend(Protocol):
    """Calls available to the routing environment; no simulator types leak."""

    time: int
    capabilities: PhysicalCapabilities

    def synchronize(self) -> None: ...

    def resources(self) -> tuple[PhysicalResource, ...]: ...

    def resource(self, pair_id: str) -> PhysicalResource | None: ...

    def resource_ids(self) -> frozenset[str]: ...

    def resource_count(self) -> int: ...

    def has_resource(self, pair_id: str) -> bool: ...

    def edge_occupancy(self, u: int, v: int) -> int: ...

    def node_occupancy(self, node: int) -> int: ...

    def validate_claim_batch(self, claims: Iterable[ResourceClaim]) -> None: ...

    def can_allocate_claims(self, claims: Iterable[ResourceClaim]) -> bool: ...

    def estimate_route_throughput(
        self, route_nodes: tuple[int, ...], width: int
    ) -> float: ...

    def estimate_swap_throughput(self, swap_counts: Iterable[int]) -> float: ...

    def link_capacities(self) -> tuple[dict[str, object], ...]: ...

    def assign_owner(self, pair_id: str, request_id: str) -> None: ...

    def generate_elementary_pairs(self) -> tuple[str, ...]: ...

    def generate_claimed_pairs(
        self, claims: Iterable[ResourceClaim], allocation_id: str
    ) -> dict[ResourceClaim, str | None]: ...

    def execute_swap(self, action: SwapAction) -> bool: ...

    def execute_lane(
        self, lane: SwapLane, allocation_id: str | None = None
    ) -> LaneExecutionResult: ...

    def execute_lanes(
        self, lanes: Iterable[SwapLane], allocation_id: str | None = None
    ) -> tuple[LaneExecutionResult, ...]: ...

    def discard_pair(self, pair_id: str) -> PhysicalResource | None: ...

    def advance_slot(self) -> None: ...
