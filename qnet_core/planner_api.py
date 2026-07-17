"""The only interface exposed to routing algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

COMMIT = -1


@dataclass(frozen=True)
class SwapAction:
    request_id: str
    middle: int
    left_pair_id: str
    right_pair_id: str


@dataclass(frozen=True)
class ResourceClaim:
    """One elementary-link memory claim for a parallel allocation lane."""

    left: int
    right: int
    lane: int = 0

    def __post_init__(self) -> None:
        if self.left == self.right:
            raise ValueError("a resource claim must connect distinct nodes")
        if self.lane < 0:
            raise ValueError("resource claim lane must be non-negative")
        if self.left > self.right:
            left, right = self.right, self.left
            object.__setattr__(self, "left", left)
            object.__setattr__(self, "right", right)

    @property
    def endpoints(self) -> tuple[int, int]:
        return self.left, self.right

    @property
    def edge(self) -> tuple[int, int]:
        return self.endpoints

    @property
    def lane_index(self) -> int:
        return self.lane


@dataclass(frozen=True)
class SwapLane:
    """An independently executable swap chain within a wider plan."""

    lane: int
    elementary_pair_ids: tuple[str, ...]
    swap_actions: tuple[SwapAction, ...]

    def __post_init__(self) -> None:
        if self.lane < 0:
            raise ValueError("swap lane must be non-negative")
        if len(set(self.elementary_pair_ids)) != len(self.elementary_pair_ids):
            raise ValueError("a swap lane cannot reference a pair more than once")

    @property
    def lane_index(self) -> int:
        return self.lane

    @property
    def pair_ids(self) -> tuple[str, ...]:
        return self.elementary_pair_ids


@dataclass(frozen=True)
class PlanDescriptor:
    plan_id: str
    request_id: str
    route_nodes: tuple[int, ...]
    reached_node: int
    elementary_pair_ids: tuple[str, ...]
    swap_actions: tuple[SwapAction, ...]
    duration: int
    remaining_hops: int
    completes_request: bool
    kind: str = "primary"
    width: int = 1
    claims: tuple[ResourceClaim, ...] = ()
    lanes: tuple[SwapLane, ...] = ()
    allocation_id: str | None = None
    expected_throughput: float = 0.0
    memory_cost: int = 0

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("plan kind must be non-empty")
        if self.width < 1:
            raise ValueError("plan width must be positive")
        if self.duration < 0:
            raise ValueError("plan duration must be non-negative")
        if self.remaining_hops < 0:
            raise ValueError("plan remaining_hops must be non-negative")
        if self.expected_throughput < 0:
            raise ValueError("plan expected_throughput must be non-negative")
        if self.memory_cost < 0:
            raise ValueError("plan memory_cost must be non-negative")
        if any(claim.lane >= self.width for claim in self.claims):
            raise ValueError("resource claim lane lies outside plan width")
        if any(lane.lane >= self.width for lane in self.lanes):
            raise ValueError("swap lane lies outside plan width")


@dataclass(frozen=True)
class PlanningSnapshot:
    time: int
    requests: tuple[dict[str, object], ...]
    resources: tuple[dict[str, object], ...]
    candidates: tuple[PlanDescriptor, ...]
    action_mask: tuple[bool, ...]
    metrics: dict[str, float]
    phase: str = "primary"
    link_capacities: tuple[dict[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not self.phase:
            raise ValueError("planning phase must be non-empty")


class Planner(Protocol):
    """A planner may select plans, but never owns the physical simulation."""

    def reset(self, episode_seed: int) -> None: ...

    def select(self, snapshot: PlanningSnapshot) -> Sequence[str] | int: ...
