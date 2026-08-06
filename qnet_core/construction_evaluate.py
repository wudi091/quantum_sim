"""Small end-to-end evaluator for joint route/construction baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .construction_catalog import RouteConstructionCandidate
from .construction_metrics import RequestSettlement, censored_flow_time
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

    if any(request.demand_pairs != 1 for request in spec.requests):
        raise ValueError("the first construction evaluator supports demand_pairs=1 only")
    missing = [request.id for request in spec.requests if request.id not in selected]
    if missing:
        raise ValueError(f"missing selected construction plans: {missing}")
    candidates = tuple(selected[request.id] for request in spec.requests)
    executor = make_sequence_construction_executor(
        spec,
        tuple(candidate.dag for candidate in candidates),
    )
    horizon_ps = spec.horizon * spec.physical.slot_duration_ps
    terminal_segments = {
        candidate.request_id: candidate.terminal_segment_id for candidate in candidates
    }
    settled: dict[str, RequestSettlement] = {}
    event_trace: list[object] = []

    def arrival_ps(request_id: str) -> int:
        request = next(item for item in spec.requests if item.id == request_id)
        return request.arrival * spec.physical.slot_duration_ps

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
                executor.launch(packed)
                continue
            if not executor.has_in_flight:
                # A physically unavailable operation is a terminal failed
                # branch for this fixed baseline; leave it for the evaluator's
                # horizon-censored settlement rather than retrying silently.
                break
        if executor.has_in_flight:
            batch = executor.advance_to_next_event()
            event_trace.extend(batch.events)
            for event in batch.events:
                if (
                    event.success
                    and event.output_segment_id == terminal_segments.get(event.request_id)
                ):
                    settled[event.request_id] = RequestSettlement(
                        event.request_id,
                        arrival_ps(event.request_id),
                        event.physical_time_ps,
                        True,
                    )
            continue
        future_arrivals = [arrival_ps(request.id) for request in spec.requests
                           if request.id not in settled and arrival_ps(request.id) > now]
        if future_arrivals:
            target = min(future_arrivals)
            executor.wait_until(target)
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
    metrics = {
        "completed_requests": float(completed),
        "completion_rate": completed / max(len(settlements), 1),
        "censored_flow_time_ps": float(flow_time),
        "mean_censored_latency_ps": flow_time / max(len(settlements), 1),
        "p95_completion_latency_ps": _percentile(successful_latencies, 95.0),
        "risk_count": float(len(settlements) - completed),
        "makespan_ps": float(max((event.physical_time_ps for event in event_trace), default=0)),
    }
    return ConstructionEvaluation(metrics, settlements, tuple(event_trace))
