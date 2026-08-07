"""Small end-to-end evaluator for joint route/construction baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .construction_api import ConstructionLaunchRejected, ExecutionEvent
from .construction_catalog import RouteConstructionCandidate
from .construction_metrics import (
    MemoryTelemetry,
    RequestSettlement,
    censored_flow_time,
    execution_event_metrics,
)
from .runtime import make_sequence_construction_executor
from .spec import EpisodeSpec


@dataclass(frozen=True)
class ConstructionEvaluation:
    metrics: Mapping[str, float]
    settlements: tuple[RequestSettlement, ...]
    event_trace: tuple[object, ...]


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((percentile / 100.0) * (len(ordered) - 1)))))
    return float(ordered[index])


def run_joint_plan_baseline(
    spec: EpisodeSpec,
    selected: Mapping[str, RouteConstructionCandidate],
) -> ConstructionEvaluation:
    """Execute one fixed joint catalogue choice through the SeQUeNCe adapter."""

    missing = [request.id for request in spec.requests if request.id not in selected]
    if missing:
        raise ValueError(f"missing selected construction plans: {missing}")
    candidates = tuple(selected[request.id] for request in spec.requests)
    requests = {request.id: request for request in spec.requests}
    executor = make_sequence_construction_executor(
        spec,
        tuple(candidate.dag for candidate in candidates),
    )
    horizon_ps = spec.horizon * spec.physical.slot_duration_ps
    terminal_segments = {
        candidate.request_id: frozenset(candidate.all_terminal_segment_ids)
        for candidate in candidates
    }
    delivered_segments = {request.id: set() for request in spec.requests}
    settled: dict[str, RequestSettlement] = {}
    event_trace: list[object] = []
    memory_telemetry = MemoryTelemetry()
    executor_launch_batch_attempt_count = 0
    executor_rejection_count = 0
    launch_rejection_counter = 0

    def observe_memory_usage() -> None:
        memory_telemetry.observe(executor.snapshot())

    observe_memory_usage()

    def arrival_ps(request_id: str) -> int:
        return requests[request_id].arrival * spec.physical.slot_duration_ps

    def deadline_ps(request_id: str) -> int | None:
        deadline = requests[request_id].deadline
        return None if deadline is None else deadline * spec.physical.slot_duration_ps

    def settle_failure(request_id: str, time_ps: int) -> None:
        if request_id in settled:
            return
        deadline = deadline_ps(request_id)
        settlement_time = time_ps if deadline is None else min(time_ps, deadline)
        settled[request_id] = RequestSettlement(
            request_id, arrival_ps(request_id), settlement_time, False
        )
        executor.release_request(request_id)

    def append_due_deadlines(time_ps: int) -> None:
        for request_id in sorted(requests):
            if request_id in settled or deadline_ps(request_id) != time_ps:
                continue
            event = ExecutionEvent(
                event_id=f"deadline-{request_id}-{time_ps}",
                operation_id=f"{request_id}:deadline",
                request_id=request_id,
                attempt_id=f"{request_id}:deadline:{time_ps}",
                event_kind="deadline",
                physical_time_ps=time_ps,
                success=False,
                failure_cause="deadline",
                in_flight_operation_ids=tuple(
                    item.operation_id for item in executor.snapshot().in_flight
                ),
            )
            event_trace.append(event)
            settle_failure(request_id, time_ps)

    def append_launch_rejections(operations) -> None:
        nonlocal launch_rejection_counter
        snapshot = executor.snapshot()
        by_request = {}
        for operation in operations:
            by_request.setdefault(operation.request_id, operation)
        for request_id, operation in sorted(by_request.items()):
            launch_rejection_counter += 1
            event = ExecutionEvent(
                event_id=f"launch-rejection-{launch_rejection_counter:08d}",
                operation_id=operation.op_id,
                request_id=request_id,
                attempt_id=(
                    f"{operation.op_id}:launch-rejection:"
                    f"{launch_rejection_counter}"
                ),
                event_kind="launch_rejection",
                physical_time_ps=executor.physical_time_ps,
                success=False,
                failure_cause="executor_launch_rejection",
                in_flight_operation_ids=tuple(
                    item.operation_id for item in snapshot.in_flight
                ),
            )
            event_trace.append(event)
            settle_failure(request_id, executor.physical_time_ps)

    def process_events(events) -> None:
        for event in events:
            if event.request_id in settled:
                continue
            request = requests[event.request_id]
            terminal = event.output_segment_id in terminal_segments.get(
                event.request_id, frozenset()
            )
            if not event.success:
                settle_failure(event.request_id, event.physical_time_ps)
                continue
            if not terminal:
                continue
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
                settled[event.request_id] = RequestSettlement(
                    event.request_id,
                    arrival_ps(event.request_id),
                    event.physical_time_ps,
                    True,
                )
                executor.release_request(event.request_id)
        # A request can settle while one of its operations is still in
        # flight.  release_request() cannot remove the eventual output until
        # that physical event completes, so repeat the release after every
        # event batch to prevent late outputs from retaining capacity.
        for request_id in settled:
            executor.release_request(request_id)
        observe_memory_usage()

    def pack_ready(operations):
        snapshot = executor.snapshot()
        usage = dict(snapshot.reservations)
        inputs: set[str] = set()
        selected = []
        capacities = dict(snapshot.resource_capacities)
        for operation in operations:
            if any(item.kind == "GEN" for item in operations) and operation.kind == "SWAP":
                continue
            if any(segment_id in inputs for segment_id in operation.input_segment_ids):
                continue
            feasible = True
            for resource, amount in operation.resource_demand.items():
                if usage.get(resource, 0) + amount > capacities.get(resource, 0):
                    feasible = False
                    break
            if not feasible:
                continue
            selected.append(operation)
            inputs.update(operation.input_segment_ids)
            for resource, amount in operation.resource_demand.items():
                usage[resource] = usage.get(resource, 0) + amount
        return tuple(selected)

    while executor.physical_time_ps < horizon_ps and not executor.terminated:
        now = executor.physical_time_ps
        active_ids = {
            request.id for request in spec.requests
            if request.id not in settled and arrival_ps(request.id) <= now
        }
        ready = tuple(
            operation for operation in executor.ready_operations()
            if operation.request_id in active_ids
        )
        if ready:
            packed = pack_ready(ready)
            if packed:
                executor_launch_batch_attempt_count += 1
                try:
                    executor.launch(packed)
                except ConstructionLaunchRejected:
                    executor_rejection_count += 1
                    append_launch_rejections(packed)
                    continue
                observe_memory_usage()
                continue
            if not executor.has_in_flight:
                # A physically unavailable operation is a terminal failed
                # branch for this fixed baseline; leave it for the evaluator's
                # horizon-censored settlement rather than retrying silently.
                break
        if executor.has_in_flight:
            future_deadlines = [
                deadline_ps(request_id)
                for request_id in requests
                if request_id not in settled
                and deadline_ps(request_id) is not None
                and deadline_ps(request_id) > now
            ]
            boundary = min((horizon_ps, *future_deadlines))
            batch = executor.advance_to_next_event(boundary_ps=boundary)
            event_trace.extend(batch.events)
            observe_memory_usage()
            process_events(batch.events)
            append_due_deadlines(executor.physical_time_ps)
            continue
        future_boundaries = [
            arrival_ps(request_id)
            for request_id in requests
            if request_id not in settled and arrival_ps(request_id) > now
        ] + [
            deadline_ps(request_id)
            for request_id in requests
            if request_id not in settled
            and deadline_ps(request_id) is not None
            and deadline_ps(request_id) > now
        ]
        expiration_time = executor.next_expiration_time_ps()
        if expiration_time is not None and expiration_time > now:
            future_boundaries.append(expiration_time)
        if future_boundaries:
            target = min(horizon_ps, *future_boundaries)
            batch = executor.wait_until(target)
            event_trace.extend(batch.events)
            observe_memory_usage()
            process_events(batch.events)
            append_due_deadlines(executor.physical_time_ps)
            continue
        break

    for request in spec.requests:
        if request.id not in settled:
            settled[request.id] = RequestSettlement(
                request.id,
                arrival_ps(request.id),
                horizon_ps,
                False,
            )
    settlements = tuple(settled[request.id] for request in spec.requests)
    successful_latencies = [
        settlement.settlement_time - settlement.arrival_time
        for settlement in settlements if settlement.success
    ]
    flow_time = censored_flow_time(settlements, horizon_ps)
    completed = sum(settlement.success for settlement in settlements)
    event_metrics = execution_event_metrics(event_trace)
    memory_metrics = memory_telemetry.metrics(
        spec.physical.slot_duration_ps
    )
    metrics = {
        "completed_requests": float(completed),
        "delivered_pairs": float(sum(len(values) for values in delivered_segments.values())),
        "completion_rate": completed / max(len(settlements), 1),
        "censored_flow_time_ps": float(flow_time),
        "mean_censored_latency_ps": flow_time / max(len(settlements), 1),
        "p95_completion_latency_ps": _percentile(successful_latencies, 95.0),
        "risk_count": float(len(settlements) - completed),
        "makespan_ps": float(max((event.physical_time_ps for event in event_trace), default=0)),
        "event_count": float(len(event_trace)),
        "executor_launch_batch_attempt_count": float(
            executor_launch_batch_attempt_count
        ),
        "executor_rejection_count": float(executor_rejection_count),
        **memory_metrics,
        **event_metrics,
    }
    return ConstructionEvaluation(metrics, settlements, tuple(event_trace))
