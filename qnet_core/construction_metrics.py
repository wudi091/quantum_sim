"""Objective accounting for the finite-horizon construction environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RequestSettlement:
    request_id: str
    arrival_time: int
    settlement_time: int
    success: bool

    def __post_init__(self) -> None:
        if self.arrival_time < 0 or self.settlement_time < self.arrival_time:
            raise ValueError("invalid request settlement time")


def censored_completion_time(settlement: RequestSettlement, horizon: int) -> int:
    """Use the true completion time on success and horizon on failure."""

    if horizon < settlement.arrival_time:
        raise ValueError("horizon precedes request arrival")
    return settlement.settlement_time if settlement.success else horizon


def censored_flow_time(settlements: Iterable[RequestSettlement], horizon: int) -> int:
    return sum(
        max(0, censored_completion_time(settlement, horizon) - settlement.arrival_time)
        for settlement in settlements
    )


def event_accounted_flow_time(
    intervals: Iterable[tuple[int, int, int]],
    failed_settlements: Iterable[RequestSettlement],
    horizon: int,
) -> int:
    """Compute the event reward identity's right-hand accounting.

    ``intervals`` are half-open ``(start, end, pending_count)`` intervals.
    ``failed_settlements`` contributes the remaining-horizon lump penalty.
    """

    holding = 0
    for start, end, pending_count in intervals:
        if start < 0 or end < start or pending_count < 0:
            raise ValueError("invalid event interval")
        holding += pending_count * (min(end, horizon) - min(start, horizon))
    holding += sum(
        max(0, horizon - settlement.settlement_time)
        for settlement in failed_settlements
    )
    return holding
