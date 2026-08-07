"""Objective accounting for the finite-horizon construction environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .construction_api import ConstructionSnapshot, ExecutionEvent


STOCHASTIC_PHYSICAL_FAILURE_CAUSES = frozenset({
    "physical_failure",
    "stochastic_failure",
})
POST_COMPLETION_VALIDATION_FAILURE_CAUSES = frozenset({
    "physical_output_missing",
    "post_completion_capacity",
})
PHYSICAL_BACKEND_REJECTION_CAUSES = frozenset({
    "physical_backend_rejection",
})


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


def observed_reserved_memory_units(snapshot: ConstructionSnapshot) -> int:
    """Return memory units held or reserved in the executor ledger."""

    return sum(
        amount
        for resource, amount in snapshot.reservations
        if resource.startswith("memory:")
    )


def observed_physical_memory_usage(snapshot: ConstructionSnapshot) -> int:
    """Return non-RAW SeQUeNCe memory units from neutral backend state."""

    backend_state = dict(snapshot.backend_state)
    exact_usage = backend_state.get("physical_memory_usage")
    if isinstance(exact_usage, int):
        return exact_usage
    node_memory = backend_state.get("node_memory")
    if not isinstance(node_memory, tuple):
        return observed_reserved_memory_units(snapshot)
    occupied = 0
    for _node, counts in node_memory:
        for state, amount in counts:
            if str(state).upper() != "RAW":
                occupied += int(amount)
    return occupied


@dataclass
class MemoryTelemetry:
    """Peak and exposure accounting over neutral physical snapshots."""

    peak_physical_memory_usage: int = 0
    peak_reserved_memory_units: int = 0
    physical_memory_time_unit_ps: int = 0
    _last_time_ps: int | None = field(default=None, repr=False)
    _last_physical_usage: int = field(default=0, repr=False)
    _uses_exact_backend_exposure: bool = field(default=False, repr=False)

    def observe(self, snapshot: ConstructionSnapshot) -> None:
        time_ps = int(snapshot.physical_time_ps)
        physical_usage = observed_physical_memory_usage(snapshot)
        reserved_usage = observed_reserved_memory_units(snapshot)
        backend_state = dict(snapshot.backend_state)
        exact_peak = backend_state.get("peak_physical_memory_usage")
        exact_exposure = backend_state.get("physical_memory_time_unit_ps")
        has_exact_exposure = isinstance(exact_exposure, int)
        if has_exact_exposure:
            if (
                self._uses_exact_backend_exposure
                and exact_exposure < self.physical_memory_time_unit_ps
            ):
                raise ValueError("backend memory exposure cannot decrease")
            self._uses_exact_backend_exposure = True
            self.physical_memory_time_unit_ps = int(exact_exposure)
        elif self._last_time_ps is not None:
            if time_ps < self._last_time_ps:
                raise ValueError("memory telemetry time cannot move backwards")
            self.physical_memory_time_unit_ps += (
                self._last_physical_usage * (time_ps - self._last_time_ps)
            )
        self.peak_physical_memory_usage = max(
            self.peak_physical_memory_usage,
            int(exact_peak) if isinstance(exact_peak, int) else physical_usage,
        )
        self.peak_reserved_memory_units = max(
            self.peak_reserved_memory_units, reserved_usage
        )
        self._last_time_ps = time_ps
        self._last_physical_usage = physical_usage

    def metrics(self, slot_duration_ps: int) -> dict[str, float]:
        if slot_duration_ps < 1:
            raise ValueError("slot_duration_ps must be positive")
        return {
            "peak_memory_usage": float(self.peak_physical_memory_usage),
            "peak_physical_memory_usage": float(
                self.peak_physical_memory_usage
            ),
            "peak_reserved_memory_units": float(
                self.peak_reserved_memory_units
            ),
            "physical_memory_time_unit_ps": float(
                self.physical_memory_time_unit_ps
            ),
            "physical_memory_time_unit_slots": float(
                self.physical_memory_time_unit_ps / slot_duration_ps
            ),
        }


def execution_event_metrics(
    events: Iterable[ExecutionEvent],
) -> dict[str, float]:
    """Classify physical event outcomes without exposing simulator objects."""

    expiration_count = 0
    fidelity_violation_count = 0
    physical_failure_count = 0
    physical_backend_rejection_count = 0
    post_completion_validation_failure_count = 0
    generation_event_count = 0
    generation_protocol_attempt_count = 0
    swap_protocol_attempt_count = 0
    fidelity_check_count = 0
    generation_physical_failure_count = 0
    swap_physical_failure_count = 0
    for event in events:
        cause = event.failure_cause
        is_generation = event.event_kind == "gen"
        is_swap = event.event_kind == "swap"
        backend_rejection = cause in PHYSICAL_BACKEND_REJECTION_CAUSES
        generation_event_count += int(is_generation)
        generation_protocol_attempt_count += int(
            is_generation and not backend_rejection
        )
        swap_protocol_attempt_count += int(is_swap)
        fidelity_check_count += int(
            (is_generation or is_swap)
            and (event.success or cause == "fidelity_reject")
        )
        expiration_count += int(cause == "expiration")
        fidelity_violation_count += int(cause == "fidelity_reject")
        stochastic_failure = cause in STOCHASTIC_PHYSICAL_FAILURE_CAUSES
        physical_failure_count += int(stochastic_failure)
        generation_physical_failure_count += int(
            stochastic_failure and event.event_kind == "gen"
        )
        swap_physical_failure_count += int(
            stochastic_failure and event.event_kind == "swap"
        )
        physical_backend_rejection_count += int(
            cause in PHYSICAL_BACKEND_REJECTION_CAUSES
        )
        post_completion_validation_failure_count += int(
            cause in POST_COMPLETION_VALIDATION_FAILURE_CAUSES
        )
    physical_protocol_attempt_count = (
        generation_protocol_attempt_count + swap_protocol_attempt_count
    )
    return {
        "expiration_count": float(expiration_count),
        "fidelity_violation_count": float(fidelity_violation_count),
        "physical_failure_count": float(physical_failure_count),
        "physical_backend_rejection_count": float(
            physical_backend_rejection_count
        ),
        "post_completion_validation_failure_count": float(
            post_completion_validation_failure_count
        ),
        "generation_event_count": float(generation_event_count),
        "generation_protocol_attempt_count": float(
            generation_protocol_attempt_count
        ),
        "swap_protocol_attempt_count": float(swap_protocol_attempt_count),
        "physical_protocol_attempt_count": float(
            physical_protocol_attempt_count
        ),
        "fidelity_check_count": float(fidelity_check_count),
        "generation_physical_failure_count": float(
            generation_physical_failure_count
        ),
        "swap_physical_failure_count": float(
            swap_physical_failure_count
        ),
    }
