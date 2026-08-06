"""Simulator-neutral commands shared by routing and physical adapters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SwapAction:
    request_id: str
    middle: int
    left_pair_id: str
    right_pair_id: str


@dataclass(frozen=True)
class ResourceClaim:
    """One elementary-link memory claim in a parallel allocation lane."""

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
    """One independently executable swap chain within a wider plan."""

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
