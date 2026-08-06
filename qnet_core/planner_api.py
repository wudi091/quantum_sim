"""The only interface exposed to routing algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .command_api import ResourceClaim, SwapAction, SwapLane

COMMIT = -1


@dataclass(frozen=True)
class PlanFeedback:
    """Simulator-neutral result of the preceding allocation/execution round."""

    feedback_id: int
    time: int
    phase: str
    plan_id: str
    request_id: str
    reached_node: int
    accepted: bool
    succeeded: bool
    reason: str = ""


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
    feedback: tuple[PlanFeedback, ...] = ()

    def __post_init__(self) -> None:
        if not self.phase:
            raise ValueError("planning phase must be non-empty")


class Planner(Protocol):
    """A planner may select plans, but never owns the physical simulation."""

    def reset(self, episode_seed: int) -> None: ...

    def select(self, snapshot: PlanningSnapshot) -> Sequence[str] | int: ...
