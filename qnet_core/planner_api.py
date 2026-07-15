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


@dataclass(frozen=True)
class PlanningSnapshot:
    time: int
    requests: tuple[dict[str, object], ...]
    resources: tuple[dict[str, object], ...]
    candidates: tuple[PlanDescriptor, ...]
    action_mask: tuple[bool, ...]
    metrics: dict[str, float]


class Planner(Protocol):
    """A planner may select plans, but never owns the physical simulation."""

    def reset(self, episode_seed: int) -> None: ...

    def select(self, snapshot: PlanningSnapshot) -> Sequence[str] | int: ...
