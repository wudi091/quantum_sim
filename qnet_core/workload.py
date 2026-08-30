"""Shared workload resolution for periodic online request streams."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PeriodicArrivalWorkload:
    """Resolved finite representation of one periodic request stream.

    ``fixed_arrival_rounds`` mirrors the workload organization used by the
    official Q-CAST experiments: run a fixed number of traffic rounds and
    inject the same number of new requests in every round.  The simulator
    still needs a finite ``request_count`` because an ``EpisodeSpec`` lists
    every request explicitly, so that count is derived rather than selected
    independently.
    """

    mode: str
    request_count: int
    arrival_rounds: int
    requests_per_round: int
    arrival_interval_slots: int
    arrival_phase_slots: int
    last_arrival_slot: int
    final_round_request_count: int
    offered_load_requests_per_slot: float
    drain_slots: int
    horizon_slots: int


def resolve_periodic_arrival_workload(
    *,
    request_count: int | None,
    arrival_rounds: int | None,
    requests_per_round: int,
    arrival_interval_slots: int,
    ttl_slots: int,
    horizon_slots: int | None,
    default_request_count: int,
) -> PeriodicArrivalWorkload:
    """Resolve legacy fixed-count or Q-CAST-style fixed-round workloads."""

    if request_count is not None and arrival_rounds is not None:
        raise ValueError("choose either request_count or arrival_rounds, not both")
    if requests_per_round < 1:
        raise ValueError("requests_per_round must be positive")
    if arrival_interval_slots < 1:
        raise ValueError("arrival_interval_slots must be positive")
    if ttl_slots < 1:
        raise ValueError("ttl_slots must be positive")
    if default_request_count < 1:
        raise ValueError("default_request_count must be positive")

    resolved_request_count = request_count
    resolved_arrival_rounds = arrival_rounds
    if resolved_request_count is None and resolved_arrival_rounds is None:
        resolved_request_count = default_request_count

    if resolved_arrival_rounds is not None:
        if resolved_arrival_rounds < 1:
            raise ValueError("arrival_rounds must be positive")
        mode = "fixed_arrival_rounds"
        resolved_request_count = resolved_arrival_rounds * requests_per_round
        final_round_request_count = requests_per_round
    else:
        assert resolved_request_count is not None
        if resolved_request_count < 1:
            raise ValueError("request_count must be positive")
        mode = "fixed_request_count"
        resolved_arrival_rounds = (
            resolved_request_count + requests_per_round - 1
        ) // requests_per_round
        final_round_request_count = (
            resolved_request_count
            - (resolved_arrival_rounds - 1) * requests_per_round
        )

    arrival_phase_slots = resolved_arrival_rounds * arrival_interval_slots
    last_arrival_slot = (
        resolved_arrival_rounds - 1
    ) * arrival_interval_slots
    minimum_horizon = last_arrival_slot + ttl_slots
    resolved_horizon = (
        minimum_horizon if horizon_slots is None else horizon_slots
    )
    if resolved_horizon < minimum_horizon:
        raise ValueError("horizon must cover the final arrival's TTL")

    return PeriodicArrivalWorkload(
        mode=mode,
        request_count=resolved_request_count,
        arrival_rounds=resolved_arrival_rounds,
        requests_per_round=requests_per_round,
        arrival_interval_slots=arrival_interval_slots,
        arrival_phase_slots=arrival_phase_slots,
        last_arrival_slot=last_arrival_slot,
        final_round_request_count=final_round_request_count,
        offered_load_requests_per_slot=(
            resolved_request_count / arrival_phase_slots
        ),
        drain_slots=resolved_horizon - last_arrival_slot,
        horizon_slots=resolved_horizon,
    )


__all__ = [
    "PeriodicArrivalWorkload",
    "resolve_periodic_arrival_workload",
]
