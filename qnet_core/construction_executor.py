"""Deterministic event-driven construction executor used as a contract oracle.

This backend deliberately models only neutral logical segments and abstract
resources.  It is a test/reference backend, not a replacement for SeQUeNCe.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
import random
from typing import Iterable, Mapping

from .construction_api import (
    ConstructionDAG,
    ConstructionOperation,
    ConstructionSnapshot,
    ExecutionEvent,
    ExecutionEventBatch,
    InFlightOperation,
    LogicalSegment,
    OperationKind,
    ResourceDemand,
)
from .construction_repair import generate_repair_options
from .construction_decoder import CapacityFeasibilityOracle


@dataclass(frozen=True)
class _Pending:
    completion_time_ps: int
    event_id: str
    operation: ConstructionOperation
    attempt_id: str
    input_segment_ids: tuple[str, ...]
    reserved: ResourceDemand


class ConstructionDAGExecutor:
    """Event-driven executor with atomic resource reservation and repair."""

    def __init__(
        self,
        dags: Iterable[ConstructionDAG],
        capacities: Mapping[str, int],
        initial_segments: Iterable[LogicalSegment] = (),
        seed: int = 0,
        horizon_ps: int = 10**18,
    ):
        dag_list = tuple(dags)
        self.dags = {dag.request_id: dag for dag in dag_list}
        if len(self.dags) != len(dag_list) or len(self.dags) == 0:
            raise ValueError("at least one construction DAG is required")
        self.capacities = {str(key): int(value) for key, value in capacities.items()}
        self.oracle = CapacityFeasibilityOracle(self.capacities)
        segment_list = tuple(initial_segments)
        self.segments: dict[str, LogicalSegment] = {segment.segment_id: segment for segment in segment_list}
        if len(self.segments) != len(segment_list):
            raise ValueError("duplicate initial segment id")
        self._initial_segment_ids = frozenset(self.segments)
        self._output_owners: dict[str, str] = {}
        self._operation_owners: dict[str, str] = {}
        self._refresh_global_registry()
        self.physical_time_ps = 0
        self.horizon_ps = int(horizon_ps)
        if self.horizon_ps < 0:
            raise ValueError("horizon_ps must be non-negative")
        self._rng = random.Random(seed)
        self._counter = 0
        self._heap: list[tuple[int, int, str, _Pending]] = []
        self._in_flight: dict[str, _Pending] = {}
        self._reservations: dict[str, ResourceDemand] = {}
        self._attempts: dict[str, int] = {}
        self._terminated = False
        self.event_log: list[ExecutionEvent] = []

    @property
    def time(self) -> int:
        return self.physical_time_ps

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def has_in_flight(self) -> bool:
        return bool(self._in_flight)

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter:08d}"

    def _refresh_global_registry(self) -> None:
        """Rebuild cross-DAG identity indexes before accepting mutable DAG state."""

        operation_owners: dict[str, str] = {}
        output_owners: dict[str, str] = {}
        for dag in self.dags.values():
            for operation in dag.operations:
                if operation.request_id != dag.request_id:
                    raise ValueError(
                        f"operation belongs to the wrong request DAG: {operation.op_id}"
                    )
                owner = operation_owners.get(operation.op_id)
                if owner is not None:
                    raise ValueError(
                        f"operation id is shared by multiple DAGs: {operation.op_id}"
                    )
                operation_owners[operation.op_id] = dag.request_id
                output_id = operation.output_segment_id
                if output_id is None:
                    continue
                output_owner = output_owners.get(output_id)
                if output_owner is not None:
                    raise ValueError(
                        f"output segment id is shared by multiple operations: {output_id}"
                    )
                output_owners[output_id] = operation.op_id
        conflicts = self._initial_segment_ids.intersection(output_owners)
        if conflicts:
            raise ValueError(
                f"initial segment id conflicts with a DAG output: {sorted(conflicts)[0]}"
            )
        self._operation_owners = operation_owners
        self._output_owners = output_owners

    def _usage(self, extra: Iterable[ResourceDemand] = ()) -> dict[str, int]:
        usage: dict[str, int] = {}
        held = tuple(
            segment.held_resources for segment in self.segments.values()
        )
        for demand in held + tuple(self._reservations.values()) + tuple(extra):
            for resource, amount in demand.items():
                usage[resource] = usage.get(resource, 0) + amount
        return usage

    def _post_completion_usage(
        self, operations: Iterable[ConstructionOperation]
    ) -> dict[str, int]:
        consumed = {
            segment_id
            for operation in operations
            for segment_id in operation.input_segment_ids
        }
        usage: dict[str, int] = {}
        for segment_id, segment in self.segments.items():
            if segment_id in consumed:
                continue
            for resource, amount in segment.held_resources.items():
                usage[resource] = usage.get(resource, 0) + amount
        for demand in self._reservations.values():
            for resource, amount in demand.items():
                usage[resource] = usage.get(resource, 0) + amount
        for operation in operations:
            for resource, amount in operation.output_resource_hold.items():
                usage[resource] = usage.get(resource, 0) + amount
        return usage

    def _available_segments(self) -> set[str]:
        consumed = {
            segment_id
            for pending in self._in_flight.values()
            for segment_id in pending.input_segment_ids
        }
        return set(self.segments) - consumed

    def available_segments(self) -> tuple[LogicalSegment, ...]:
        return tuple(
            sorted((self.segments[segment_id] for segment_id in self._available_segments()),
                   key=lambda segment: segment.segment_id)
        )

    def ready_operations(self) -> tuple[ConstructionOperation, ...]:
        self._refresh_global_registry()
        available = self._available_segments()
        operations = [
            operation
            for dag in self.dags.values()
            for operation in dag.operations
            if operation.op_id in dag.ready_ids(available)
        ]
        return tuple(sorted(operations, key=lambda operation: operation.canonical_key))

    def snapshot(self) -> ConstructionSnapshot:
        """Return a pure read-only view; no event or physical state changes."""

        self._refresh_global_registry()

        in_flight = tuple(
            InFlightOperation(
                operation_id=pending.operation.op_id,
                request_id=pending.operation.request_id,
                attempt_id=pending.attempt_id,
                start_time_ps=pending.completion_time_ps - pending.operation.duration_ps,
                completion_time_ps=pending.completion_time_ps,
                reserved_resources=pending.reserved.entries,
                input_segment_ids=pending.input_segment_ids,
            )
            for pending in sorted(self._in_flight.values(), key=lambda item: item.operation.op_id)
        )
        pending_events = tuple(
            (pending.event_id, pending.completion_time_ps, pending.operation.kind)
            for pending in sorted(self._in_flight.values(), key=lambda item: item.event_id)
        )
        reservations = tuple(
            (resource, amount)
            for resource, amount in sorted(self._usage().items())
        )
        return ConstructionSnapshot(
            physical_time_ps=self.physical_time_ps,
            horizon_ps=self.horizon_ps,
            dag_states=tuple(dag.state() for dag in sorted(self.dags.values(), key=lambda dag: dag.request_id)),
            operations=tuple(sorted(
                (
                    operation
                    for dag in self.dags.values()
                    for operation in dag.operations
                ),
                key=lambda operation: operation.canonical_key,
            )),
            segments=tuple(sorted(self.segments.values(), key=lambda segment: segment.segment_id)),
            reservations=reservations,
            in_flight=in_flight,
            pending_events=pending_events,
            resource_capacities=tuple(sorted(self.capacities.items())),
        )

    def _validate_launch(self, operations: tuple[ConstructionOperation, ...]) -> None:
        self._refresh_global_registry()
        if self._terminated:
            raise RuntimeError("executor is terminated")
        if not operations:
            raise ValueError("launch requires at least one operation")
        if len({operation.op_id for operation in operations}) != len(operations):
            raise ValueError("duplicate operation in launch set")
        available = self._available_segments()
        for operation in operations:
            dag = self.dags.get(operation.request_id)
            if dag is None:
                raise ValueError(f"unknown request DAG: {operation.request_id}")
            canonical = dag.operation(operation.op_id)
            if canonical != operation:
                raise ValueError(
                    f"operation object does not match DAG canonical operation: {operation.op_id}"
                )
            if operation.op_id not in dag.ready_ids(available):
                raise ValueError(f"operation is not ready: {operation.op_id}")
            self._validate_output_hold(operation)
            for segment_id in operation.input_segment_ids:
                segment = self.segments.get(segment_id)
                if segment is not None and segment.request_id != operation.request_id:
                    raise ValueError(
                        "operation cannot consume another request's segment"
                    )
            if operation.kind == OperationKind.SWAP:
                # The neutral oracle keeps a permissive malformed-SWAP branch
                # for failure/repair contract tests.  Whenever two logical
                # inputs are present, enforce the same outer-endpoint
                # invariant as the SeQUeNCe adapter.
                if len(operation.input_segment_ids) == 2:
                    left = self.segments.get(operation.input_segment_ids[0])
                    right = self.segments.get(operation.input_segment_ids[1])
                    if left is None or right is None:
                        raise ValueError("SWAP input segment is not available")
                    intersection = set(left.endpoints) & set(right.endpoints)
                    if len(intersection) != 1:
                        raise ValueError("SWAP inputs must share exactly one middle node")
                    middle = next(iter(intersection))
                    expected_outer = {
                        next(node for node in left.endpoints if node != middle),
                        next(node for node in right.endpoints if node != middle),
                    }
                    if set(operation.output_endpoints or ()) != expected_outer:
                        raise ValueError(
                            f"SWAP output endpoints do not match logical outer endpoints: {operation.op_id}"
                        )
        result = self.oracle.check(operations)
        if not result.feasible:
            raise ValueError(result.reason)
        usage = self._usage((operation.resource_demand for operation in operations))
        for resource, amount in usage.items():
            if amount > self.capacities.get(resource, 0):
                raise ValueError(f"capacity exceeded: {resource}")
        for resource, amount in self._post_completion_usage(operations).items():
            if amount > self.capacities.get(resource, 0):
                raise ValueError(f"post-completion capacity exceeded: {resource}")
        inputs = [segment_id for operation in operations for segment_id in operation.input_segment_ids]
        if len(set(inputs)) != len(inputs):
            raise ValueError("input segment consumed twice in launch set")
        output_ids = [
            operation.output_segment_id
            for operation in operations
            if operation.output_segment_id is not None
        ]
        if len(output_ids) != len(set(output_ids)):
            raise ValueError("output segment produced twice in launch set")
        for output_id in output_ids:
            if output_id in self.segments:
                raise ValueError(f"output segment id is already live: {output_id}")
            if any(
                pending.operation.output_segment_id == output_id
                for pending in self._in_flight.values()
            ):
                raise ValueError(f"output segment id is already in flight: {output_id}")

    def _validate_output_hold(self, operation: ConstructionOperation) -> None:
        hold = operation.output_resource_hold
        for resource, amount in hold.items():
            if amount > self.capacities.get(resource, 0):
                raise ValueError(f"output resource hold exceeds capacity: {resource}")
        if operation.kind == OperationKind.RELEASE:
            if hold:
                raise ValueError("RELEASE cannot declare an output resource hold")

    def launch(self, feasible_set: Iterable[ConstructionOperation]) -> tuple[str, ...]:
        operations = tuple(feasible_set)
        self._validate_launch(operations)
        reservation_ids: list[str] = []
        for operation in sorted(operations, key=lambda item: item.canonical_key):
            dag = self.dags[operation.request_id]
            dag.mark_started(operation.op_id, self._available_segments())
            attempt_no = self._attempts.get(operation.op_id, 0) + 1
            self._attempts[operation.op_id] = attempt_no
            attempt_id = f"{operation.op_id}:attempt:{attempt_no}"
            event_id = self._next_id("event")
            pending = _Pending(
                completion_time_ps=self.physical_time_ps + operation.duration_ps,
                event_id=event_id,
                operation=operation,
                attempt_id=attempt_id,
                input_segment_ids=operation.input_segment_ids,
                reserved=operation.resource_demand,
            )
            self._in_flight[operation.op_id] = pending
            self._reservations[operation.op_id] = operation.resource_demand
            heapq.heappush(self._heap, (pending.completion_time_ps, 0, event_id, pending))
            reservation_ids.append(event_id)
        return tuple(reservation_ids)

    def _complete(self, pending: _Pending) -> ExecutionEvent:
        operation = pending.operation
        self._in_flight.pop(operation.op_id, None)
        self._reservations.pop(operation.op_id, None)
        dag = self.dags[operation.request_id]
        success = self._rng.random() <= operation.success_probability
        consumed: tuple[str, ...] = ()
        surviving: tuple[str, ...] = ()
        output_id: str | None = None
        output_fidelity: float | None = None
        cause = ""
        if success:
            consumed = tuple(operation.input_segment_ids)
            input_segments = [self.segments.pop(segment_id) for segment_id in consumed]
            if operation.kind in {OperationKind.GEN, OperationKind.SWAP}:
                if operation.output_segment_id is None or operation.output_endpoints is None:
                    raise RuntimeError(f"{operation.kind} operation lacks output metadata")
                output_id = operation.output_segment_id
                fidelity = min((segment.fidelity for segment in input_segments), default=1.0)
                output_fidelity = fidelity
                if output_fidelity < operation.required_fidelity:
                    cause = "fidelity_reject"
                    dag.mark_dead(operation.op_id)
                    event = ExecutionEvent(
                        event_id=pending.event_id,
                        operation_id=operation.op_id,
                        request_id=operation.request_id,
                        attempt_id=pending.attempt_id,
                        event_kind=operation.kind.lower(),
                        physical_time_ps=self.physical_time_ps,
                        success=False,
                        failure_cause=cause,
                        consumed_segment_ids=consumed,
                        surviving_segment_ids=tuple(sorted(self._available_segments())),
                        output_segment_id=None,
                        output_fidelity=output_fidelity,
                        released_resources=pending.reserved.entries,
                        in_flight_operation_ids=tuple(sorted(self._in_flight)),
                    )
                    self.event_log.append(event)
                    return event
                if any(
                    self._usage().get(resource_id, 0) + amount
                    > self.capacities.get(resource_id, 0)
                    for resource_id, amount in operation.output_resource_hold.items()
                ):
                    cause = "post_completion_capacity"
                    dag.mark_dead(operation.op_id)
                    event = ExecutionEvent(
                        event_id=pending.event_id,
                        operation_id=operation.op_id,
                        request_id=operation.request_id,
                        attempt_id=pending.attempt_id,
                        event_kind=operation.kind.lower(),
                        physical_time_ps=self.physical_time_ps,
                        success=False,
                        failure_cause=cause,
                        consumed_segment_ids=consumed,
                        surviving_segment_ids=tuple(sorted(self._available_segments())),
                        output_segment_id=None,
                        output_fidelity=output_fidelity,
                        released_resources=pending.reserved.entries,
                        in_flight_operation_ids=tuple(sorted(self._in_flight)),
                    )
                    self.event_log.append(event)
                    return event
                self.segments[output_id] = LogicalSegment(
                    output_id,
                    operation.request_id,
                    operation.output_endpoints[0],
                    operation.output_endpoints[1],
                    self.physical_time_ps,
                    fidelity,
                    operation.request_id,
                    operation.op_id,
                    operation.output_resource_hold,
                )
            dag.mark_completed(operation.op_id)
        else:
            cause = "stochastic_failure"
            surviving = tuple(sorted(self._available_segments()))
            dag.mark_dead(operation.op_id)
        event = ExecutionEvent(
            event_id=pending.event_id,
            operation_id=operation.op_id,
            request_id=operation.request_id,
            attempt_id=pending.attempt_id,
            event_kind=operation.kind.lower(),
            physical_time_ps=self.physical_time_ps,
            success=success,
            failure_cause=cause,
            consumed_segment_ids=consumed,
            surviving_segment_ids=surviving,
            output_segment_id=output_id,
            output_fidelity=output_fidelity,
            released_resources=pending.reserved.entries,
            in_flight_operation_ids=tuple(sorted(self._in_flight)),
        )
        self.event_log.append(event)
        return event

    def advance_to_next_event(self, boundary_ps: int | None = None) -> ExecutionEventBatch:
        if self._terminated:
            return ExecutionEventBatch(self.physical_time_ps, (), 0, terminal=True)
        if not self._heap:
            self.terminate()
            return ExecutionEventBatch(self.physical_time_ps, (), 0, terminal=True)
        next_time = min(self._heap[0][0], self.horizon_ps)
        if boundary_ps is not None:
            boundary_ps = int(boundary_ps)
            if boundary_ps < self.physical_time_ps or boundary_ps > self.horizon_ps:
                raise ValueError("boundary time must lie in [current_time, horizon]")
            if self.physical_time_ps < boundary_ps < next_time:
                duration = boundary_ps - self.physical_time_ps
                self.physical_time_ps = boundary_ps
                return ExecutionEventBatch(
                    self.physical_time_ps, (), duration, terminal=False
                )
        duration = max(0, next_time - self.physical_time_ps)
        self.physical_time_ps = next_time
        pending: list[_Pending] = []
        while self._heap and self._heap[0][0] == next_time:
            pending.append(heapq.heappop(self._heap)[3])
        if not pending and next_time >= self.horizon_ps:
            pending = list(self._in_flight.values())
            self._heap.clear()
            self._in_flight.clear()
            self._reservations.clear()
            events = []
            for item in sorted(pending, key=lambda value: value.event_id):
                dag = self.dags[item.operation.request_id]
                dag.mark_dead(item.operation.op_id)
                event = ExecutionEvent(
                    event_id=item.event_id,
                    operation_id=item.operation.op_id,
                    request_id=item.operation.request_id,
                    attempt_id=item.attempt_id,
                    event_kind=item.operation.kind.lower(),
                    physical_time_ps=self.physical_time_ps,
                    success=False,
                    failure_cause="horizon_timeout",
                    surviving_segment_ids=tuple(sorted(self._available_segments())),
                    released_resources=item.reserved.entries,
                    in_flight_operation_ids=(),
                )
                events.append(event)
                self.event_log.append(event)
            self._terminated = True
            return ExecutionEventBatch(
                self.physical_time_ps,
                tuple(sorted(events, key=lambda event: event.event_id)),
                duration,
                terminal=True,
            )
        events = tuple(sorted((self._complete(item) for item in pending), key=lambda event: event.event_id))
        if self.physical_time_ps >= self.horizon_ps:
            self._terminated = True
        return ExecutionEventBatch(self.physical_time_ps, events, duration, terminal=self._terminated and not self._heap)

    def repair(self, request_id: str, operations: tuple[ConstructionOperation, ...]) -> None:
        self._refresh_global_registry()
        if request_id not in self.dags:
            raise KeyError(request_id)
        if any(
            pending.operation.request_id == request_id
            for pending in self._in_flight.values()
        ):
            raise RuntimeError("cannot repair a request with in-flight operations")
        dag = self.dags[request_id]
        new_ids = tuple(operation.op_id for operation in operations)
        if len(set(new_ids)) != len(new_ids):
            raise ValueError("repair operation IDs must be unique")
        for operation in operations:
            owner = self._operation_owners.get(operation.op_id)
            if owner is not None:
                raise ValueError(f"repair operation id already exists: {operation.op_id}")
            self._validate_output_hold(operation)
        available = self._available_segments()
        outputs = {
            operation.output_segment_id: operation
            for operation in operations
            if operation.output_segment_id is not None
        }
        other_dag_outputs = {
            operation.output_segment_id
            for current_dag in self.dags.values()
            if current_dag.request_id != request_id
            for operation in current_dag.operations
            if operation.output_segment_id is not None
        }
        for output_id in outputs:
            if output_id in self.segments:
                raise ValueError(f"repair output segment id is already live: {output_id}")
            if any(
                pending.operation.output_segment_id == output_id
                for pending in self._in_flight.values()
            ):
                raise ValueError(f"repair output segment id is already in flight: {output_id}")
            if output_id in other_dag_outputs:
                raise ValueError(f"repair output segment id is owned by another DAG: {output_id}")
        for operation in operations:
            if any(predecessor in dag.dead for predecessor in operation.predecessors):
                raise ValueError("repair operation cannot depend on a dead predecessor")
            for segment_id in operation.input_segment_ids:
                if segment_id in available:
                    if self.segments[segment_id].request_id != request_id:
                        raise ValueError("repair cannot consume another request's segment")
                    continue
                producer = outputs.get(segment_id)
                if producer is None or producer.op_id not in operation.predecessors:
                    raise ValueError(
                        f"repair input segment is not surviving or newly produced: {segment_id}"
                    )
        self.dags[request_id].repair(operations)
        self._operation_owners.update({operation.op_id: request_id for operation in operations})

    def repair_options(
        self, request_id: str
    ) -> tuple[tuple[ConstructionOperation, ...], ...]:
        if request_id not in self.dags:
            raise KeyError(request_id)
        dag = self.dags[request_id]
        return generate_repair_options(
            dag,
            self._available_segments(),
            next_version=dag.version + 1,
            ordinal_start=max(
                (operation.ordinal for operation in dag.operations), default=0
            ) + 1,
        )

    def release_segment(self, segment_id: str) -> LogicalSegment | None:
        segment = self.segments.get(segment_id)
        if segment is None:
            return None
        if any(
            segment_id in pending.input_segment_ids
            for pending in self._in_flight.values()
        ):
            raise RuntimeError(f"cannot release segment used by an in-flight operation: {segment_id}")
        return self.segments.pop(segment_id)

    def release_request(self, request_id: str) -> tuple[str, ...]:
        if request_id not in self.dags:
            raise KeyError(request_id)
        released: list[str] = []
        for segment_id, segment in tuple(self.segments.items()):
            if segment.request_id != request_id:
                continue
            if any(
                segment_id in pending.input_segment_ids
                for pending in self._in_flight.values()
            ):
                continue
            self.segments.pop(segment_id, None)
            released.append(segment_id)
        return tuple(sorted(released))

    def terminate(self) -> None:
        if self._in_flight:
            raise RuntimeError("cannot terminate while operations are in flight")
        self._terminated = True

    def wait_until(self, target_time_ps: int) -> ExecutionEventBatch:
        if self._in_flight:
            raise RuntimeError("wait_until cannot skip in-flight operations")
        if target_time_ps < self.physical_time_ps or target_time_ps > self.horizon_ps:
            raise ValueError("target time must lie in [current_time, horizon]")
        duration = target_time_ps - self.physical_time_ps
        self.physical_time_ps = target_time_ps
        if target_time_ps >= self.horizon_ps:
            self._terminated = True
        return ExecutionEventBatch(target_time_ps, (), duration, terminal=self._terminated)
