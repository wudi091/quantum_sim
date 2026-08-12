"""Neutral slotted construction plans executed by the SeQUeNCe adapter.

The planning layer supplies only immutable request/DAG metadata and integer
planning slots.  This module translates those coarse slots into calls on the
neutral :class:`ConstructionExecutor` contract.  SeQUeNCe remains solely
responsible for protocol timing, stochastic outcomes, fidelity, memory state,
and expiration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .construction_api import (
    ConstructionDAG,
    ConstructionLaunchRejected,
    ConstructionOperation,
    ExecutionEvent,
)
from .construction_metrics import (
    MemoryTelemetry,
    RequestSettlement,
    censored_flow_time,
    execution_event_metrics,
)
from .runtime import make_sequence_construction_executor
from .spec import EpisodeSpec


@dataclass(frozen=True)
class ScheduledRequestPlan:
    """One admitted request with an absolute coarse-slot construction plan."""

    request_id: str
    candidate_id: str
    route_nodes: tuple[int, ...]
    construction_kind: str
    dag: ConstructionDAG
    terminal_segment_ids: tuple[str, ...]
    start_slot: int
    completion_slot: int
    operation_slots: tuple[tuple[str, int], ...]
    purification_kind: str = "none"

    def __post_init__(self) -> None:
        if not self.request_id or not self.candidate_id:
            raise ValueError("request_id and candidate_id must be non-empty")
        if len(self.route_nodes) < 2:
            raise ValueError("scheduled route must contain at least one edge")
        if not self.construction_kind:
            raise ValueError("construction_kind must be non-empty")
        if not self.purification_kind:
            raise ValueError("purification_kind must be non-empty")
        if self.dag.request_id != self.request_id:
            raise ValueError("scheduled DAG belongs to another request")
        if self.start_slot < 0 or self.completion_slot <= self.start_slot:
            raise ValueError("invalid scheduled request time window")
        if not self.terminal_segment_ids:
            raise ValueError("scheduled request needs terminal segments")
        if len(set(self.terminal_segment_ids)) != len(self.terminal_segment_ids):
            raise ValueError("terminal segment IDs must be unique")
        if tuple(sorted(self.operation_slots)) != self.operation_slots:
            raise ValueError("operation_slots must be operation-id sorted")

        slot_by_operation = dict(self.operation_slots)
        if len(slot_by_operation) != len(self.operation_slots):
            raise ValueError("scheduled operation IDs must be unique")
        dag_operation_ids = {operation.op_id for operation in self.dag.operations}
        if set(slot_by_operation) != dag_operation_ids:
            raise ValueError("operation_slots must cover the scheduled DAG exactly")
        if min(slot_by_operation.values()) != self.start_slot:
            raise ValueError("start_slot must equal the first operation slot")
        if any(
            slot < self.start_slot or slot >= self.completion_slot
            for slot in slot_by_operation.values()
        ):
            raise ValueError("scheduled operation lies outside its request window")

        producer_by_segment = {
            operation.output_segment_id: operation.op_id
            for operation in self.dag.operations
            if operation.output_segment_id is not None
        }
        for operation in self.dag.operations:
            operation_slot = slot_by_operation[operation.op_id]
            dependencies = set(operation.predecessors)
            dependencies.update(
                producer_by_segment[segment_id]
                for segment_id in operation.input_segment_ids
                if segment_id in producer_by_segment
            )
            if any(
                slot_by_operation[predecessor] >= operation_slot
                for predecessor in dependencies
            ):
                raise ValueError(
                    f"construction dependency is not in an earlier slot: "
                    f"{operation.op_id}"
                )

        terminal_producers = []
        for segment_id in self.terminal_segment_ids:
            producer = producer_by_segment.get(segment_id)
            if producer is None:
                raise ValueError(
                    f"terminal segment has no scheduled producer: {segment_id}"
                )
            terminal_producers.append(producer)
            operation = self.dag.operation(producer)
            if (
                operation.output_endpoints is None
                or frozenset(operation.output_endpoints)
                != frozenset((self.route_nodes[0], self.route_nodes[-1]))
            ):
                raise ValueError(
                    f"terminal segment endpoints do not match the route: "
                    f"{segment_id}"
                )
        expected_completion = 1 + max(
            slot_by_operation[operation_id]
            for operation_id in terminal_producers
        )
        if self.completion_slot != expected_completion:
            raise ValueError(
                "completion_slot must follow the final terminal operation"
            )

    @property
    def operation_slot_map(self) -> dict[str, int]:
        return dict(self.operation_slots)


@dataclass(frozen=True)
class ConstructionBatchSchedule:
    """A complete admission and slotted construction decision for one batch."""

    horizon_slots: int
    requests: tuple[ScheduledRequestPlan, ...]
    rejected_request_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.horizon_slots < 1:
            raise ValueError("schedule horizon must be positive")
        if tuple(sorted(self.requests, key=lambda item: item.request_id)) != self.requests:
            raise ValueError("scheduled requests must be request-id sorted")
        request_ids = [item.request_id for item in self.requests]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("a request can appear in the schedule at most once")
        if tuple(sorted(set(self.rejected_request_ids))) != self.rejected_request_ids:
            raise ValueError("rejected request IDs must be unique and sorted")
        if set(request_ids).intersection(self.rejected_request_ids):
            raise ValueError("a request cannot be both admitted and rejected")
        if any(item.completion_slot > self.horizon_slots for item in self.requests):
            raise ValueError("scheduled request exceeds the batch horizon")
        operation_ids = [
            operation.op_id
            for item in self.requests
            for operation in item.dag.operations
        ]
        if len(set(operation_ids)) != len(operation_ids):
            raise ValueError("scheduled operation IDs must be globally unique")

    @property
    def selected_request_ids(self) -> tuple[str, ...]:
        return tuple(item.request_id for item in self.requests)

    @property
    def operation_count(self) -> int:
        return sum(len(item.operation_slots) for item in self.requests)


@dataclass(frozen=True, order=True)
class ScheduledOperationLaunch:
    operation_id: str
    request_id: str
    planned_slot: int
    actual_time_ps: int
    attempt_id: str


@dataclass(frozen=True, order=True)
class ScheduleViolation:
    code: str
    slot: int
    request_id: str = ""
    operation_id: str = ""
    detail: str = ""


def _in_flight_dependency_blocked_operation_ids(
    due_operations: Mapping[str, ConstructionOperation],
    operation_by_id: Mapping[str, ConstructionOperation],
    in_flight_operation_ids: set[str],
) -> frozenset[str]:
    """Return due operations delayed by an unfinished DAG ancestor.

    A physical operation may cross a coarse planning-slot boundary.  Any
    descendant whose nominal slot has also arrived is then unlaunchable because
    its construction dependency is still executing, not because the scheduler
    omitted ready work.  Classifying both descendants and the running ancestor
    as launch/completion overruns double-counts that one physical delay.

    Dependencies include explicit predecessor edges and implicit producer edges
    induced by input segments.  The transitive walk is deliberate: several
    nominal DAG levels can become due while one early SeQUeNCe operation remains
    in flight.
    """

    if not due_operations or not in_flight_operation_ids:
        return frozenset()
    producer_by_segment = {
        operation.output_segment_id: operation.op_id
        for operation in operation_by_id.values()
        if operation.output_segment_id is not None
    }
    dependencies_by_operation: dict[str, frozenset[str]] = {}
    for operation_id, operation in operation_by_id.items():
        dependencies = set(operation.predecessors)
        dependencies.update(
            producer_by_segment[segment_id]
            for segment_id in operation.input_segment_ids
            if segment_id in producer_by_segment
        )
        dependencies_by_operation[operation_id] = frozenset(dependencies)

    memo: dict[str, bool] = {}

    def has_in_flight_ancestor(operation_id: str, visiting: set[str]) -> bool:
        cached = memo.get(operation_id)
        if cached is not None:
            return cached
        if operation_id in visiting:
            # ConstructionDAG validation rejects cycles.  Treat a defensive
            # cycle as non-exempt so it cannot hide a real launch failure.
            return False
        visiting.add(operation_id)
        dependencies = dependencies_by_operation.get(operation_id, frozenset())
        blocked = bool(dependencies.intersection(in_flight_operation_ids)) or any(
            has_in_flight_ancestor(predecessor, visiting)
            for predecessor in dependencies
            if predecessor in operation_by_id
        )
        visiting.remove(operation_id)
        memo[operation_id] = blocked
        return blocked

    return frozenset(
        operation_id
        for operation_id in due_operations
        if has_in_flight_ancestor(operation_id, set())
    )


@dataclass(frozen=True)
class ScheduledConstructionEvaluation:
    metrics: Mapping[str, float]
    settlements: tuple[RequestSettlement, ...]
    event_trace: tuple[ExecutionEvent, ...]
    launches: tuple[ScheduledOperationLaunch, ...]
    violations: tuple[ScheduleViolation, ...]


@dataclass(frozen=True, order=True)
class ScheduledRequestAttemptOutcome:
    """Terminal result of one submitted online construction plan."""

    request_id: str
    candidate_id: str
    success: bool
    settlement_time_ps: int
    failure_cause: str = ""


@dataclass(frozen=True)
class PersistentScheduleUpdate:
    """Events and attempt outcomes produced while advancing online time."""

    physical_time_ps: int
    outcomes: tuple[ScheduledRequestAttemptOutcome, ...]
    events: tuple[ExecutionEvent, ...]
    launches: tuple[ScheduledOperationLaunch, ...]
    violations: tuple[ScheduleViolation, ...]


class PersistentConstructionScheduler:
    """Execute complete plans submitted at recurring decision boundaries.

    The scheduler owns one persistent SeQUeNCe-backed executor.  Planning code
    can submit additional neutral :class:`ScheduledRequestPlan` objects, but
    cannot access or mutate simulator internals.  A submitted plan is never
    rearranged; stochastic failure ends that attempt and makes the request
    eligible for a later retry after physical cleanup.
    """

    def __init__(self, spec: EpisodeSpec):
        self.spec = spec
        self.slot_duration_ps = spec.physical.slot_duration_ps
        self.horizon_ps = spec.horizon * self.slot_duration_ps
        self.requests = {request.id: request for request in spec.requests}
        self.executor = make_sequence_construction_executor(spec, ())
        self._active_plans: dict[str, ScheduledRequestPlan] = {}
        self._cleanup_requests: set[str] = set()
        self._completed_times: dict[str, int] = {}
        self._terminal_segments: dict[str, frozenset[str]] = {}
        self._delivered_segments: dict[str, set[str]] = {}
        self._operations_by_slot: dict[int, list[ConstructionOperation]] = {}
        self._planned_slot_by_operation: dict[str, int] = {}
        self._due: dict[str, ConstructionOperation] = {}
        self._event_trace: list[ExecutionEvent] = []
        self._launches: list[ScheduledOperationLaunch] = []
        self._violations: list[ScheduleViolation] = []
        self._violation_keys: set[tuple[str, int, str]] = set()
        self._outcomes: list[ScheduledRequestAttemptOutcome] = []
        self._memory_telemetry = MemoryTelemetry()
        self._event_counter = 0
        self._memory_telemetry.observe(self.executor.snapshot())

    @property
    def physical_time_ps(self) -> int:
        return int(self.executor.physical_time_ps)

    @property
    def current_slot(self) -> int:
        return self.physical_time_ps // self.slot_duration_ps

    @property
    def active_request_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._active_plans))

    @property
    def cleanup_request_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._cleanup_requests))

    @property
    def completed_request_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._completed_times))

    @property
    def completed_times(self) -> Mapping[str, int]:
        return dict(self._completed_times)

    @property
    def event_trace(self) -> tuple[ExecutionEvent, ...]:
        return tuple(self._event_trace)

    @property
    def launches(self) -> tuple[ScheduledOperationLaunch, ...]:
        return tuple(sorted(self._launches))

    @property
    def violations(self) -> tuple[ScheduleViolation, ...]:
        return tuple(sorted(self._violations))

    def can_submit(self, request_id: str) -> bool:
        return (
            request_id in self.requests
            and request_id not in self._active_plans
            and request_id not in self._cleanup_requests
            and request_id not in self._completed_times
            and self.physical_time_ps < self.horizon_ps
        )

    def submit(self, plans: tuple[ScheduledRequestPlan, ...]) -> None:
        """Atomically register plans selected at the current boundary."""

        if not plans:
            return
        if self.physical_time_ps % self.slot_duration_ps:
            raise RuntimeError("plans can only be submitted at a slot boundary")
        if tuple(sorted(plans, key=lambda item: item.request_id)) != plans:
            raise ValueError("submitted plans must be request-id sorted")
        request_ids = [plan.request_id for plan in plans]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("a request can be submitted at most once per boundary")
        current_slot = self.current_slot
        for plan in plans:
            _validate_request_plan_against_episode(self.spec, plan)
            if not self.can_submit(plan.request_id):
                raise ValueError(f"request cannot accept a new plan: {plan.request_id}")
            request = self.requests[plan.request_id]
            if request.arrival > current_slot:
                raise ValueError(f"request has not arrived: {plan.request_id}")
            if plan.start_slot < current_slot:
                raise ValueError(f"online plan starts in the past: {plan.request_id}")
            if plan.completion_slot > self.spec.horizon:
                raise ValueError(f"online plan exceeds the episode horizon: {plan.request_id}")
            if request.deadline is not None and plan.completion_slot > request.deadline:
                raise ValueError(f"online plan exceeds the request deadline: {plan.request_id}")

        registered: list[str] = []
        registered_operation_ids: set[str] = set()
        try:
            for plan in plans:
                self.executor.register_dag(plan.dag)
                registered.append(plan.request_id)
                self._active_plans[plan.request_id] = plan
                self._terminal_segments[plan.request_id] = frozenset(
                    plan.terminal_segment_ids
                )
                self._delivered_segments[plan.request_id] = set()
                slot_by_operation = plan.operation_slot_map
                for operation in plan.dag.operations:
                    registered_operation_ids.add(operation.op_id)
                    slot = slot_by_operation[operation.op_id]
                    self._operations_by_slot.setdefault(slot, []).append(operation)
                    self._planned_slot_by_operation[operation.op_id] = slot
            for operations in self._operations_by_slot.values():
                operations.sort(key=lambda operation: operation.canonical_key)
        except Exception:
            for request_id in reversed(registered):
                self._active_plans.pop(request_id, None)
                self._terminal_segments.pop(request_id, None)
                self._delivered_segments.pop(request_id, None)
                self.executor.unregister_dag(request_id)
            registered_set = set(registered)
            for slot, operations in tuple(self._operations_by_slot.items()):
                retained = [
                    operation for operation in operations
                    if operation.request_id not in registered_set
                ]
                if retained:
                    self._operations_by_slot[slot] = retained
                else:
                    self._operations_by_slot.pop(slot, None)
            for operation_id in tuple(self._planned_slot_by_operation):
                if operation_id in registered_operation_ids:
                    self._planned_slot_by_operation.pop(operation_id, None)
            raise

    def physical_reservations(
        self,
        window_end_slot: int,
    ) -> dict[tuple[str, int], int]:
        """Aggregate request-scoped physical reservations."""

        aggregate: dict[tuple[str, int], int] = {}
        for usage in self.physical_reservations_by_request(
            window_end_slot
        ).values():
            for key, amount in usage.items():
                aggregate[key] = aggregate.get(key, 0) + amount
        return aggregate

    def physical_reservations_by_request(
        self,
        window_end_slot: int,
    ) -> dict[str, dict[tuple[str, int], int]]:
        """Expose current and in-flight holds without simulator objects.

        Input segments remain physically resident while SWAP/PURIFY protocols
        are in flight.  Their held link/memory resources are therefore carried
        through the pending operation's predicted completion slot, alongside
        the operation's own reservation.  If physical delay has pushed an
        operation beyond its nominal slot, a conservative demand/input/output
        envelope is retained for the full planning window so a new plan cannot
        seize resources already promised to the fixed running plan.
        """

        end_slot = int(window_end_slot)
        if not self.current_slot < end_slot <= self.spec.horizon:
            raise ValueError("reservation window must follow the current slot")
        snapshot = self.executor.snapshot()
        reservations: dict[str, dict[tuple[str, int], int]] = {}
        capacities = dict(snapshot.resource_capacities)

        def add(
            request_id: str,
            resource_id: str,
            slot: int,
            amount: int,
        ) -> None:
            if amount <= 0:
                return
            request_usage = reservations.setdefault(request_id, {})
            key = (resource_id, slot)
            request_usage[key] = request_usage.get(key, 0) + int(amount)

        def reserve_at_least(
            request_id: str,
            resource_id: str,
            slot: int,
            amount: int,
        ) -> None:
            if amount <= 0:
                return
            request_usage = reservations.setdefault(request_id, {})
            key = (resource_id, slot)
            request_usage[key] = max(
                request_usage.get(key, 0),
                min(capacities.get(resource_id, amount), int(amount)),
            )

        segments = {segment.segment_id: segment for segment in snapshot.segments}
        for segment in snapshot.segments:
            for resource_id, amount in segment.held_resources.items():
                add(
                    segment.request_id,
                    resource_id,
                    self.current_slot,
                    amount,
                )
        for pending in snapshot.in_flight:
            for resource_id, amount in pending.reserved_resources:
                add(
                    pending.request_id,
                    resource_id,
                    self.current_slot,
                    amount,
                )

        for pending in snapshot.in_flight:
            final_slot = min(
                end_slot - 1,
                max(
                    self.current_slot,
                    (pending.completion_time_ps - 1) // self.slot_duration_ps,
                ),
            )
            input_holds: dict[str, int] = {}
            for segment_id in pending.input_segment_ids:
                segment = segments.get(segment_id)
                if segment is None:
                    continue
                for resource_id, amount in segment.held_resources.items():
                    input_holds[resource_id] = (
                        input_holds.get(resource_id, 0) + amount
                    )
            for slot in range(self.current_slot + 1, final_slot + 1):
                for resource_id, amount in pending.reserved_resources:
                    add(pending.request_id, resource_id, slot, amount)
                for resource_id, amount in input_holds.items():
                    add(pending.request_id, resource_id, slot, amount)

        operation_by_id = {
            operation.op_id: operation
            for plan in self._active_plans.values()
            for operation in plan.dag.operations
        }
        producer_by_segment = {
            operation.output_segment_id: operation
            for operation in operation_by_id.values()
            if operation.output_segment_id is not None
        }
        lagging_operations = dict(self._due)
        for pending in snapshot.in_flight:
            planned_slot = self._planned_slot_by_operation.get(
                pending.operation_id
            )
            operation = operation_by_id.get(pending.operation_id)
            if (
                operation is not None
                and planned_slot is not None
                and planned_slot < self.current_slot
            ):
                lagging_operations[pending.operation_id] = operation

        # A physical overrun invalidates the nominal mutual-exclusion timing
        # between lagging DAG stages.  Summing their envelopes is deliberately
        # conservative: it may defer new admission, but it cannot steal a
        # resource from an already committed request.  Capacity clipping keeps
        # the quarantine bounded, and it disappears as soon as the attempt
        # settles and leaves _active_plans.
        lagging_by_request: dict[str, dict[str, int]] = {}
        for operation in lagging_operations.values():
            if operation.request_id not in self._active_plans:
                continue
            envelope = dict(operation.resource_demand.items())
            for segment_id in operation.input_segment_ids:
                segment = segments.get(segment_id)
                producer = producer_by_segment.get(segment_id)
                if segment is not None:
                    held_resources = segment.held_resources
                elif producer is not None:
                    held_resources = producer.output_resource_hold
                else:
                    held_resources = None
                if held_resources is None:
                    continue
                for resource_id, amount in held_resources.items():
                    envelope[resource_id] = envelope.get(resource_id, 0) + amount
            for resource_id, amount in operation.output_resource_hold.items():
                envelope[resource_id] = max(
                    envelope.get(resource_id, 0),
                    amount,
                )
            request_envelope = lagging_by_request.setdefault(
                operation.request_id,
                {},
            )
            for resource_id, amount in envelope.items():
                total = request_envelope.get(resource_id, 0) + amount
                capacity = capacities.get(resource_id)
                request_envelope[resource_id] = (
                    total if capacity is None else min(capacity, total)
                )

        for request_id, envelope in lagging_by_request.items():
            for slot in range(self.current_slot, end_slot):
                for resource_id, amount in envelope.items():
                    reserve_at_least(request_id, resource_id, slot, amount)
        return reservations

    def _ready_due_operations(self) -> tuple[ConstructionOperation, ...]:
        """Return ready work without letting later plans overtake older slots."""

        slots = sorted({
            self._planned_slot_by_operation[operation_id]
            for operation_id in self._due
        })
        for planned_slot in slots:
            allowed = tuple(
                operation_id
                for operation_id in self._due
                if self._planned_slot_by_operation[operation_id] == planned_slot
            )
            ready = tuple(self.executor.ready_operations(allowed))
            if ready:
                return ready
        return ()

    def _arrival_ps(self, request_id: str) -> int:
        return self.requests[request_id].arrival * self.slot_duration_ps

    def _deadline_ps(self, request_id: str) -> int | None:
        deadline = self.requests[request_id].deadline
        return None if deadline is None else deadline * self.slot_duration_ps

    def _add_violation(
        self,
        code: str,
        slot: int,
        operation: ConstructionOperation,
        detail: str,
    ) -> None:
        key = (code, slot, operation.op_id)
        if key in self._violation_keys:
            return
        self._violation_keys.add(key)
        self._violations.append(ScheduleViolation(
            code=code,
            slot=slot,
            request_id=operation.request_id,
            operation_id=operation.op_id,
            detail=detail,
        ))

    def _finish_attempt(
        self,
        request_id: str,
        *,
        success: bool,
        time_ps: int,
        failure_cause: str = "",
    ) -> None:
        plan = self._active_plans.pop(request_id, None)
        if plan is None:
            return
        if success:
            self._completed_times[request_id] = int(time_ps)
        self._cleanup_requests.add(request_id)
        self._due = {
            operation_id: operation
            for operation_id, operation in self._due.items()
            if operation.request_id != request_id
        }
        self._outcomes.append(ScheduledRequestAttemptOutcome(
            request_id=request_id,
            candidate_id=plan.candidate_id,
            success=bool(success),
            settlement_time_ps=int(time_ps),
            failure_cause="" if success else str(failure_cause or "physical_failure"),
        ))
        self.executor.release_request(request_id)

    def _process_events(self, events: tuple[ExecutionEvent, ...]) -> None:
        self._event_trace.extend(events)
        for event in events:
            request_id = event.request_id
            if request_id not in self._active_plans:
                if request_id in self._cleanup_requests:
                    self.executor.release_request(request_id)
                continue
            if not event.success:
                self._finish_attempt(
                    request_id,
                    success=False,
                    time_ps=event.physical_time_ps,
                    failure_cause=event.failure_cause,
                )
                continue
            if event.output_segment_id not in self._terminal_segments.get(
                request_id,
                frozenset(),
            ):
                continue
            request = self.requests[request_id]
            if (
                event.output_fidelity is None
                or float(event.output_fidelity) + 1e-12 < request.required_fidelity
            ):
                self._finish_attempt(
                    request_id,
                    success=False,
                    time_ps=event.physical_time_ps,
                    failure_cause="fidelity_reject",
                )
                continue
            deadline = self._deadline_ps(request_id)
            if deadline is not None and event.physical_time_ps > deadline:
                self._finish_attempt(
                    request_id,
                    success=False,
                    time_ps=deadline,
                    failure_cause="deadline",
                )
                continue
            self._delivered_segments[request_id].add(event.output_segment_id)
            if len(self._delivered_segments[request_id]) >= request.demand_pairs:
                self._finish_attempt(
                    request_id,
                    success=True,
                    time_ps=event.physical_time_ps,
                )
        self._memory_telemetry.observe(self.executor.snapshot())

    def _cleanup_finished_requests(self) -> None:
        snapshot = self.executor.snapshot()
        in_flight_requests = {item.request_id for item in snapshot.in_flight}
        for request_id in tuple(sorted(self._cleanup_requests)):
            self.executor.release_request(request_id)
            if request_id in in_flight_requests:
                continue
            refreshed = self.executor.snapshot()
            if any(
                segment.request_id == request_id
                for segment in refreshed.segments
            ):
                continue
            self.executor.unregister_dag(request_id)
            self._cleanup_requests.remove(request_id)
            self._terminal_segments.pop(request_id, None)
            self._delivered_segments.pop(request_id, None)
            removed_operation_ids: set[str] = set()
            for slot, operations in tuple(self._operations_by_slot.items()):
                retained = []
                for operation in operations:
                    if operation.request_id == request_id:
                        removed_operation_ids.add(operation.op_id)
                    else:
                        retained.append(operation)
                if retained:
                    self._operations_by_slot[slot] = retained
                else:
                    self._operations_by_slot.pop(slot, None)
            for operation_id in removed_operation_ids:
                self._planned_slot_by_operation.pop(operation_id, None)
        self._memory_telemetry.observe(self.executor.snapshot())

    def _advance_once(self, boundary_ps: int) -> bool:
        before = self.physical_time_ps
        if self.executor.has_in_flight:
            batch = self.executor.advance_to_next_event(boundary_ps=boundary_ps)
        else:
            batch = self.executor.wait_until(boundary_ps)
        self._process_events(batch.events)
        self._cleanup_finished_requests()
        return self.physical_time_ps > before or bool(batch.events)

    def _launch_rejection(
        self,
        request_id: str,
        operation: ConstructionOperation,
        detail: str,
    ) -> None:
        self._event_counter += 1
        event = ExecutionEvent(
            event_id=f"online-launch-rejection-{self._event_counter:08d}",
            operation_id=operation.op_id,
            request_id=request_id,
            attempt_id=(
                f"{operation.op_id}:online-launch-rejection:"
                f"{self._event_counter}"
            ),
            event_kind="launch_rejection",
            physical_time_ps=self.physical_time_ps,
            success=False,
            failure_cause="scheduled_launch_rejection",
            in_flight_operation_ids=tuple(
                item.operation_id for item in self.executor.snapshot().in_flight
            ),
        )
        self._process_events((event,))
        self._add_violation(
            "launch_rejected",
            self._planned_slot_by_operation[operation.op_id],
            operation,
            detail,
        )

    def _expire_deadlines(self, boundary_ps: int) -> None:
        for request_id in tuple(sorted(self._active_plans)):
            deadline = self._deadline_ps(request_id)
            if deadline is None or deadline > boundary_ps:
                continue
            self._event_counter += 1
            event = ExecutionEvent(
                event_id=f"online-deadline-{self._event_counter:08d}",
                operation_id=f"{request_id}:deadline",
                request_id=request_id,
                attempt_id=f"{request_id}:deadline:{deadline}",
                event_kind="deadline",
                physical_time_ps=deadline,
                success=False,
                failure_cause="deadline",
                in_flight_operation_ids=tuple(
                    item.operation_id for item in self.executor.snapshot().in_flight
                ),
            )
            self._process_events((event,))

    def _run_slot(self, slot: int) -> None:
        slot_start_ps = slot * self.slot_duration_ps
        slot_end_ps = (slot + 1) * self.slot_duration_ps
        while self.physical_time_ps < slot_start_ps:
            if not self._advance_once(slot_start_ps):
                break
        for operation in self._operations_by_slot.get(slot, ()):
            if operation.request_id in self._active_plans:
                self._due[operation.op_id] = operation

        while self.physical_time_ps < slot_end_ps:
            self._due = {
                operation_id: operation
                for operation_id, operation in self._due.items()
                if operation.request_id in self._active_plans
            }
            ready = self._ready_due_operations()
            if ready:
                try:
                    attempt_ids = self.executor.launch(ready)
                except ConstructionLaunchRejected as exc:
                    rejection_detail = str(exc)
                else:
                    for operation, attempt_id in zip(ready, attempt_ids):
                        self._launches.append(ScheduledOperationLaunch(
                            operation_id=operation.op_id,
                            request_id=operation.request_id,
                            planned_slot=self._planned_slot_by_operation[operation.op_id],
                            actual_time_ps=self.physical_time_ps,
                            attempt_id=attempt_id,
                        ))
                        self._due.pop(operation.op_id, None)
                    self._memory_telemetry.observe(self.executor.snapshot())
                    continue
            else:
                rejection_detail = "operation is not physically ready"

            if self.executor.has_in_flight:
                if not self._advance_once(slot_end_ps):
                    break
                continue
            if self._due:
                failed: dict[str, ConstructionOperation] = {}
                for operation in self._due.values():
                    failed.setdefault(operation.request_id, operation)
                for request_id, operation in sorted(failed.items()):
                    self._launch_rejection(
                        request_id,
                        operation,
                        rejection_detail,
                    )
                continue
            if not self._advance_once(slot_end_ps):
                break

        while self.physical_time_ps < slot_end_ps:
            if not self._advance_once(slot_end_ps):
                break
        snapshot = self.executor.snapshot()
        operation_by_id = {
            operation.op_id: operation
            for plan in self._active_plans.values()
            for operation in plan.dag.operations
        }
        dependency_blocked = _in_flight_dependency_blocked_operation_ids(
            self._due,
            operation_by_id,
            {item.operation_id for item in snapshot.in_flight},
        )
        for operation in self._due.values():
            planned_slot = self._planned_slot_by_operation[operation.op_id]
            if planned_slot <= slot and operation.op_id not in dependency_blocked:
                self._add_violation(
                    "slot_launch_overrun",
                    planned_slot,
                    operation,
                    "operation was not launched before its slot boundary",
                )
        for in_flight in snapshot.in_flight:
            planned_slot = self._planned_slot_by_operation.get(in_flight.operation_id)
            if planned_slot is None or planned_slot > slot:
                continue
            operation = operation_by_id.get(in_flight.operation_id)
            if operation is not None:
                self._add_violation(
                    "slot_completion_overrun",
                    planned_slot,
                    operation,
                    "physical operation crossed its planning-slot boundary",
                )
        self._expire_deadlines(slot_end_ps)
        self._cleanup_finished_requests()

    def advance_to_slot(self, target_slot: int) -> PersistentScheduleUpdate:
        """Execute all submitted operations up to an absolute slot boundary."""

        target = int(target_slot)
        if self.physical_time_ps % self.slot_duration_ps:
            raise RuntimeError("online advance must begin at a slot boundary")
        if not self.current_slot <= target <= self.spec.horizon:
            raise ValueError("target slot must lie in [current_slot, horizon]")
        event_start = len(self._event_trace)
        launch_start = len(self._launches)
        violation_start = len(self._violations)
        outcome_start = len(self._outcomes)
        for slot in range(self.current_slot, target):
            self._run_slot(slot)
        return PersistentScheduleUpdate(
            physical_time_ps=self.physical_time_ps,
            outcomes=tuple(self._outcomes[outcome_start:]),
            events=tuple(self._event_trace[event_start:]),
            launches=tuple(self._launches[launch_start:]),
            violations=tuple(self._violations[violation_start:]),
        )

    def memory_metrics(self) -> dict[str, float]:
        self._memory_telemetry.observe(self.executor.snapshot())
        return self._memory_telemetry.metrics(self.slot_duration_ps)


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round((percentile / 100.0) * (len(ordered) - 1)))),
    )
    return float(ordered[index])


def _validate_request_plan_against_episode(
    spec: EpisodeSpec,
    item: ScheduledRequestPlan,
) -> None:
    requests = {request.id: request for request in spec.requests}
    if item.request_id not in requests:
        raise ValueError(f"schedule contains unknown request: {item.request_id}")
    request = requests[item.request_id]
    if (
        item.route_nodes[0] != request.source
        or item.route_nodes[-1] != request.destination
    ):
        raise ValueError(f"scheduled route endpoints mismatch: {item.request_id}")
    declared_edges = {
        (min(u, v), max(u, v)) for u, v in spec.edges
    }
    for left, right in zip(item.route_nodes, item.route_nodes[1:]):
        if (min(left, right), max(left, right)) not in declared_edges:
            raise ValueError(f"scheduled route uses a missing edge: {item.request_id}")
    if item.start_slot < request.arrival:
        raise ValueError(f"scheduled request starts before arrival: {item.request_id}")
    if request.deadline is not None and item.completion_slot > request.deadline:
        raise ValueError(f"scheduled request exceeds its deadline: {item.request_id}")
    if len(item.terminal_segment_ids) != request.demand_pairs:
        raise ValueError(f"scheduled terminal demand mismatch: {item.request_id}")


def _validate_schedule_against_episode(
    spec: EpisodeSpec,
    schedule: ConstructionBatchSchedule,
) -> None:
    if schedule.horizon_slots != spec.horizon:
        raise ValueError("schedule and episode horizons differ")
    requests = {request.id: request for request in spec.requests}
    declared = set(schedule.selected_request_ids) | set(schedule.rejected_request_ids)
    if declared != set(requests):
        missing = sorted(set(requests) - declared)
        extra = sorted(declared - set(requests))
        if missing:
            raise ValueError(f"schedule omits request: {missing[0]}")
        raise ValueError(f"schedule contains unknown request: {extra[0]}")
    for item in schedule.requests:
        _validate_request_plan_against_episode(spec, item)


def run_scheduled_construction_plan(
    spec: EpisodeSpec,
    schedule: ConstructionBatchSchedule,
) -> ScheduledConstructionEvaluation:
    """Execute a fixed coarse-slot plan through SeQUeNCe.

    Operations assigned to the same planning slot are allowed to launch at
    different physical instants inside that slot.  This is necessary because
    SeQUeNCe may conservatively serialize protocol families even when the
    planning model places independent operations in one coarse round.  Such
    serialization is considered schedule-preserving only when every operation
    still finishes before the slot boundary; otherwise a violation is
    recorded and execution continues so the physical delay remains visible.
    """

    _validate_schedule_against_episode(spec, schedule)
    slot_duration_ps = spec.physical.slot_duration_ps
    horizon_ps = spec.horizon * slot_duration_ps
    requests = {request.id: request for request in spec.requests}
    plans = {item.request_id: item for item in schedule.requests}
    terminal_segments = {
        item.request_id: frozenset(item.terminal_segment_ids)
        for item in schedule.requests
    }
    delivered_segments = {item.request_id: set() for item in schedule.requests}
    operation_by_id: dict[str, ConstructionOperation] = {}
    planned_slot_by_operation: dict[str, int] = {}
    operations_by_slot: dict[int, list[ConstructionOperation]] = {}
    for item in schedule.requests:
        slots = item.operation_slot_map
        for operation in item.dag.operations:
            operation_by_id[operation.op_id] = operation
            planned_slot = slots[operation.op_id]
            planned_slot_by_operation[operation.op_id] = planned_slot
            operations_by_slot.setdefault(planned_slot, []).append(operation)
    for operations in operations_by_slot.values():
        operations.sort(key=lambda operation: operation.canonical_key)

    settlements: dict[str, RequestSettlement] = {}
    event_trace: list[ExecutionEvent] = []
    launches: list[ScheduledOperationLaunch] = []
    violations: list[ScheduleViolation] = []
    violation_keys: set[tuple[str, int, str]] = set()
    memory_telemetry = MemoryTelemetry()
    launch_rejection_counter = 0

    def arrival_ps(request_id: str) -> int:
        return requests[request_id].arrival * slot_duration_ps

    def deadline_ps(request_id: str) -> int | None:
        deadline = requests[request_id].deadline
        return None if deadline is None else deadline * slot_duration_ps

    for request_id in schedule.rejected_request_ids:
        time_ps = arrival_ps(request_id)
        settlements[request_id] = RequestSettlement(
            request_id, time_ps, time_ps, False
        )
        event_trace.append(ExecutionEvent(
            event_id=f"admission-rejection-{request_id}",
            operation_id=f"{request_id}:admission",
            request_id=request_id,
            attempt_id=f"{request_id}:admission:rejected",
            event_kind="admission_rejection",
            physical_time_ps=time_ps,
            success=False,
            failure_cause="not_admitted",
        ))

    if not schedule.requests:
        ordered_settlements = tuple(
            settlements[request.id] for request in spec.requests
        )
        flow_time = censored_flow_time(ordered_settlements, horizon_ps)
        metrics = {
            "planned_selected_requests": 0.0,
            "planned_rejected_requests": float(len(spec.requests)),
            "scheduled_operation_count": 0.0,
            "launched_operation_count": 0.0,
            "completed_requests": 0.0,
            "completion_rate": 0.0,
            "censored_flow_time_ps": float(flow_time),
            "mean_censored_latency_ps": flow_time / max(len(spec.requests), 1),
            "p95_completion_latency_ps": 0.0,
            "schedule_violation_count": 0.0,
            "schedule_adherence": 1.0,
            "max_launch_delay_ps": 0.0,
            "makespan_ps": 0.0,
            **MemoryTelemetry().metrics(slot_duration_ps),
            **execution_event_metrics(event_trace),
        }
        return ScheduledConstructionEvaluation(
            metrics,
            ordered_settlements,
            tuple(event_trace),
            (),
            (),
        )

    executor = make_sequence_construction_executor(
        spec,
        tuple(item.dag for item in schedule.requests),
    )
    memory_telemetry.observe(executor.snapshot())

    def settle_failure(request_id: str, time_ps: int) -> None:
        if request_id in settlements:
            return
        deadline = deadline_ps(request_id)
        settlement_time = time_ps if deadline is None else min(time_ps, deadline)
        settlement_time = max(arrival_ps(request_id), settlement_time)
        settlements[request_id] = RequestSettlement(
            request_id,
            arrival_ps(request_id),
            settlement_time,
            False,
        )
        executor.release_request(request_id)

    def add_violation(
        code: str,
        slot: int,
        operation: ConstructionOperation,
        detail: str,
    ) -> None:
        key = (code, slot, operation.op_id)
        if key in violation_keys:
            return
        violation_keys.add(key)
        violations.append(ScheduleViolation(
            code=code,
            slot=slot,
            request_id=operation.request_id,
            operation_id=operation.op_id,
            detail=detail,
        ))

    def process_events(events: tuple[ExecutionEvent, ...]) -> None:
        event_trace.extend(events)
        for event in events:
            if event.request_id in settlements:
                continue
            if not event.success:
                settle_failure(event.request_id, event.physical_time_ps)
                continue
            if event.output_segment_id not in terminal_segments.get(
                event.request_id, frozenset()
            ):
                continue
            request = requests[event.request_id]
            if (
                event.output_fidelity is None
                or float(event.output_fidelity) + 1e-12 < request.required_fidelity
            ):
                settle_failure(event.request_id, event.physical_time_ps)
                continue
            deadline = deadline_ps(event.request_id)
            if deadline is not None and event.physical_time_ps > deadline:
                settle_failure(event.request_id, event.physical_time_ps)
                continue
            delivered_segments[event.request_id].add(event.output_segment_id)
            if len(delivered_segments[event.request_id]) >= request.demand_pairs:
                settlements[event.request_id] = RequestSettlement(
                    event.request_id,
                    arrival_ps(event.request_id),
                    event.physical_time_ps,
                    True,
                )
                executor.release_request(event.request_id)
        for request_id in settlements:
            if request_id in plans:
                executor.release_request(request_id)
        memory_telemetry.observe(executor.snapshot())

    def advance_once(boundary_ps: int) -> bool:
        before = executor.physical_time_ps
        if executor.has_in_flight:
            batch = executor.advance_to_next_event(boundary_ps=boundary_ps)
        else:
            batch = executor.wait_until(boundary_ps)
        process_events(batch.events)
        return executor.physical_time_ps > before or bool(batch.events)

    due: dict[str, ConstructionOperation] = {}
    overdue_launch_recorded: set[str] = set()
    completion_overrun_recorded: set[str] = set()

    for slot in range(spec.horizon):
        slot_start_ps = slot * slot_duration_ps
        slot_end_ps = (slot + 1) * slot_duration_ps
        while executor.physical_time_ps < slot_start_ps and not executor.terminated:
            if not advance_once(slot_start_ps):
                break

        for operation in operations_by_slot.get(slot, ()):
            if operation.request_id not in settlements:
                due[operation.op_id] = operation

        while executor.physical_time_ps < slot_end_ps and not executor.terminated:
            due = {
                op_id: operation
                for op_id, operation in due.items()
                if operation.request_id not in settlements
            }
            launch_progress = False
            rejection_details: dict[str, str] = {}
            ready_batch = tuple(executor.ready_operations(due))
            if ready_batch:
                try:
                    attempt_ids = executor.launch(ready_batch)
                except ConstructionLaunchRejected as exc:
                    for operation in ready_batch:
                        rejection_details[operation.op_id] = str(exc)
                else:
                    for operation, attempt_id in zip(ready_batch, attempt_ids):
                        launches.append(ScheduledOperationLaunch(
                            operation_id=operation.op_id,
                            request_id=operation.request_id,
                            planned_slot=planned_slot_by_operation[operation.op_id],
                            actual_time_ps=executor.physical_time_ps,
                            attempt_id=attempt_id,
                        ))
                        due.pop(operation.op_id, None)
                    launch_progress = True
                    memory_telemetry.observe(executor.snapshot())

            if launch_progress:
                continue
            if executor.has_in_flight:
                if not advance_once(slot_end_ps):
                    break
                continue
            if due:
                failed_requests: set[str] = set()
                for operation in tuple(due.values()):
                    detail = rejection_details.get(
                        operation.op_id, "operation is not physically ready"
                    )
                    add_violation("launch_rejected", slot, operation, detail)
                    failed_requests.add(operation.request_id)
                for request_id in failed_requests:
                    launch_rejection_counter += 1
                    event_trace.append(ExecutionEvent(
                        event_id=(
                            f"scheduled-launch-rejection-"
                            f"{launch_rejection_counter:08d}"
                        ),
                        operation_id=next(
                            operation.op_id
                            for operation in due.values()
                            if operation.request_id == request_id
                        ),
                        request_id=request_id,
                        attempt_id=(
                            f"{request_id}:scheduled-launch-rejection:"
                            f"{launch_rejection_counter}"
                        ),
                        event_kind="launch_rejection",
                        physical_time_ps=executor.physical_time_ps,
                        success=False,
                        failure_cause="scheduled_launch_rejection",
                    ))
                    settle_failure(request_id, executor.physical_time_ps)
                due = {
                    op_id: operation
                    for op_id, operation in due.items()
                    if operation.request_id not in failed_requests
                }
                continue
            if not advance_once(slot_end_ps):
                break

        if executor.physical_time_ps < slot_end_ps and not executor.terminated:
            while executor.physical_time_ps < slot_end_ps:
                if not advance_once(slot_end_ps):
                    break

        snapshot = executor.snapshot() if not executor.terminated else None
        dependency_blocked = _in_flight_dependency_blocked_operation_ids(
            due,
            operation_by_id,
            (
                set()
                if snapshot is None
                else {item.operation_id for item in snapshot.in_flight}
            ),
        )
        for operation in due.values():
            planned_slot = planned_slot_by_operation[operation.op_id]
            if (
                planned_slot <= slot
                and operation.op_id not in overdue_launch_recorded
                and operation.op_id not in dependency_blocked
            ):
                overdue_launch_recorded.add(operation.op_id)
                add_violation(
                    "slot_launch_overrun",
                    planned_slot,
                    operation,
                    "operation was not launched before its slot boundary",
                )
        if snapshot is not None:
            for in_flight in snapshot.in_flight:
                planned_slot = planned_slot_by_operation.get(in_flight.operation_id)
                if (
                    planned_slot is not None
                    and planned_slot <= slot
                    and in_flight.operation_id not in completion_overrun_recorded
                ):
                    completion_overrun_recorded.add(in_flight.operation_id)
                    add_violation(
                        "slot_completion_overrun",
                        planned_slot,
                        operation_by_id[in_flight.operation_id],
                        "physical operation crossed its planning-slot boundary",
                    )

    for operation in due.values():
        add_violation(
            "horizon_unlaunched",
            planned_slot_by_operation[operation.op_id],
            operation,
            "operation remained unlaunched at the physical horizon",
        )
        settle_failure(operation.request_id, horizon_ps)

    for request in spec.requests:
        if request.id not in settlements:
            settlements[request.id] = RequestSettlement(
                request.id,
                arrival_ps(request.id),
                horizon_ps,
                False,
            )

    ordered_settlements = tuple(
        settlements[request.id] for request in spec.requests
    )
    successful_latencies = [
        settlement.settlement_time - settlement.arrival_time
        for settlement in ordered_settlements
        if settlement.success
    ]
    flow_time = censored_flow_time(ordered_settlements, horizon_ps)
    completed = sum(settlement.success for settlement in ordered_settlements)
    launch_delays = [
        launch.actual_time_ps - launch.planned_slot * slot_duration_ps
        for launch in launches
    ]
    physical_events = tuple(event_trace)
    metrics = {
        "planned_selected_requests": float(len(schedule.requests)),
        "planned_rejected_requests": float(len(schedule.rejected_request_ids)),
        "scheduled_operation_count": float(schedule.operation_count),
        "launched_operation_count": float(len(launches)),
        "completed_requests": float(completed),
        "delivered_pairs": float(sum(len(items) for items in delivered_segments.values())),
        "completion_rate": completed / max(len(spec.requests), 1),
        "censored_flow_time_ps": float(flow_time),
        "mean_censored_latency_ps": flow_time / max(len(spec.requests), 1),
        "p95_completion_latency_ps": _percentile(successful_latencies, 95.0),
        "schedule_violation_count": float(len(violations)),
        "schedule_adherence": float(not violations),
        "max_launch_delay_ps": float(max(launch_delays, default=0)),
        "makespan_ps": float(max(
            (event.physical_time_ps for event in physical_events),
            default=0,
        )),
        **memory_telemetry.metrics(slot_duration_ps),
        **execution_event_metrics(physical_events),
    }
    return ScheduledConstructionEvaluation(
        metrics=metrics,
        settlements=ordered_settlements,
        event_trace=physical_events,
        launches=tuple(sorted(launches)),
        violations=tuple(sorted(violations)),
    )
