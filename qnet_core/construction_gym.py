"""Policy-facing event environment for fixed joint route/construction choices."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from .construction_api import (
    ConstructionLaunchRejected,
    ConstructionOperation,
    ConstructionSnapshot,
    ExecutionEvent,
)
from .construction_catalog import RouteConstructionCandidate
from .construction_executor import ConstructionDAGExecutor
from .construction_metrics import (
    MemoryTelemetry,
    RequestSettlement,
    censored_flow_time,
    execution_event_metrics,
)
from .runtime import make_sequence_construction_executor
from .spec import EpisodeSpec


@dataclass(frozen=True)
class ConstructionStep:
    observation: ConstructionSnapshot
    ready_operations: tuple[ConstructionOperation, ...]
    reward: float
    terminated: bool
    info: Mapping[str, object]


class ConstructionBatchEnv:
    """Event-driven interaction wrapper with explicit STOP legality."""

    def __init__(
        self,
        spec: EpisodeSpec,
        selected_candidates: Mapping[str, RouteConstructionCandidate],
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        chi: float = 1.0,
        auto_settle_failures: bool = True,
    ):
        self.spec = spec
        self.selected_candidates = dict(selected_candidates)
        if set(self.selected_candidates) != {request.id for request in spec.requests}:
            raise ValueError("selected_candidates must contain exactly one plan per request")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.chi = float(chi)
        self.auto_settle_failures = bool(auto_settle_failures)
        self._executor: ConstructionDAGExecutor | object | None = None
        self._terminal_segments: dict[str, tuple[str, ...]] = {}
        self._delivered_terminal_segments: dict[str, set[str]] = {}
        self._settled: dict[str, RequestSettlement] = {}
        self._event_log: list[ExecutionEvent] = []
        self._boundary_event_counter = 0
        self._last_time_ps = 0
        self._flow_cost_ps = 0
        self._memory_telemetry = MemoryTelemetry()
        self._executor_launch_batch_attempt_count = 0
        self._executor_rejection_count = 0

    @property
    def executor(self):
        if self._executor is None:
            raise RuntimeError("environment must be reset first")
        return self._executor

    @property
    def time_ps(self) -> int:
        return int(self.executor.physical_time_ps)

    @property
    def horizon_ps(self) -> int:
        return self.spec.horizon * self.spec.physical.slot_duration_ps

    @property
    def event_trace(self) -> tuple[ExecutionEvent, ...]:
        return tuple(self._event_log)

    def _observe_memory_usage(self) -> None:
        self._memory_telemetry.observe(self.executor.snapshot())

    def _arrival_ps(self, request_id: str) -> int:
        request = next(item for item in self.spec.requests if item.id == request_id)
        return request.arrival * self.spec.physical.slot_duration_ps

    def _deadline_ps(self, request_id: str) -> int | None:
        request = next(item for item in self.spec.requests if item.id == request_id)
        return None if request.deadline is None else request.deadline * self.spec.physical.slot_duration_ps

    def reset(self) -> ConstructionStep:
        self._executor = make_sequence_construction_executor(
            self.spec,
            tuple(self.selected_candidates[request.id].dag for request in self.spec.requests),
        )
        self._terminal_segments = {
            request.id: self.selected_candidates[request.id].all_terminal_segment_ids
            for request in self.spec.requests
        }
        self._delivered_terminal_segments = {
            request.id: set() for request in self.spec.requests
        }
        self._settled = {}
        self._event_log = []
        self._boundary_event_counter = 0
        self._last_time_ps = self.time_ps
        self._flow_cost_ps = 0
        self._memory_telemetry = MemoryTelemetry()
        self._executor_launch_batch_attempt_count = 0
        self._executor_rejection_count = 0
        self._observe_memory_usage()
        return self._step_result(0.0, False, {"event_kind": "reset"})

    def _active_request_ids(self) -> set[str]:
        return {
            request.id for request in self.spec.requests
            if request.id not in self._settled and self._arrival_ps(request.id) <= self.time_ps
        }

    def ready_operations(self) -> tuple[ConstructionOperation, ...]:
        active = self._active_request_ids()
        return tuple(
            operation for operation in self.executor.ready_operations()
            if operation.request_id in active
        )

    def stop_legal(self) -> bool:
        if self.executor.has_in_flight:
            return True
        if any(
            request.id not in self._settled and self._arrival_ps(request.id) > self.time_ps
            for request in self.spec.requests
        ):
            return True
        return not self.ready_operations()

    def _settle_events(
        self,
        events: Sequence[ExecutionEvent],
        interval_start_ps: int,
    ) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
        completed = 0
        failed: list[str] = []
        observed_failures: list[str] = []
        event_time = self.time_ps

        def settle_deadline(request_id: str, deadline: int) -> None:
            self._boundary_event_counter += 1
            snapshot = self.executor.snapshot()
            event = ExecutionEvent(
                event_id=f"deadline-{self._boundary_event_counter:08d}",
                operation_id=f"{request_id}:deadline",
                request_id=request_id,
                attempt_id=f"{request_id}:deadline:{deadline}",
                event_kind="deadline",
                physical_time_ps=deadline,
                success=False,
                failure_cause="deadline",
                in_flight_operation_ids=tuple(
                    item.operation_id for item in snapshot.in_flight
                ),
            )
            self._event_log.append(event)
            self._event_log.sort(
                key=lambda item: (item.physical_time_ps, item.event_id)
            )
            self._settled[request_id] = RequestSettlement(
                request_id,
                self._arrival_ps(request_id),
                deadline,
                False,
            )
            failed.append(request_id)

        for request in self.spec.requests:
            if request.id in self._settled:
                continue
            deadline = self._deadline_ps(request.id)
            if (
                deadline is not None
                and interval_start_ps < deadline < event_time
            ):
                settle_deadline(request.id, deadline)
        for event in events:
            if event.request_id in self._settled:
                continue
            terminal_ids = self._terminal_segments.get(event.request_id, ())
            terminal = event.output_segment_id in terminal_ids
            if event.success and terminal:
                request = next(
                    request for request in self.spec.requests
                    if request.id == event.request_id
                )
                if (
                    event.output_fidelity is None
                    or float(event.output_fidelity) + 1e-12 < request.required_fidelity
                ):
                    observed_failures.append(event.request_id)
                    if self.auto_settle_failures:
                        self._settled[event.request_id] = RequestSettlement(
                            event.request_id,
                            self._arrival_ps(event.request_id),
                            event.physical_time_ps,
                            False,
                        )
                        failed.append(event.request_id)
                    continue
                deadline = self._deadline_ps(event.request_id)
                if deadline is not None and event.physical_time_ps > deadline:
                    self._settled[event.request_id] = RequestSettlement(
                        event.request_id,
                        self._arrival_ps(event.request_id),
                        deadline,
                        False,
                    )
                    failed.append(event.request_id)
                    continue
                self._delivered_terminal_segments[event.request_id].add(
                    event.output_segment_id or ""
                )
                delivered_pairs = len(self._delivered_terminal_segments[event.request_id])
                demand_pairs = next(
                    request.demand_pairs
                    for request in self.spec.requests
                    if request.id == event.request_id
                )
                if delivered_pairs >= demand_pairs:
                    self._settled[event.request_id] = RequestSettlement(
                        event.request_id,
                        self._arrival_ps(event.request_id),
                        event.physical_time_ps,
                        True,
                    )
                    completed += 1
                    continue
                continue
            if not event.success:
                observed_failures.append(event.request_id)
                if self.auto_settle_failures:
                    self._settled[event.request_id] = RequestSettlement(
                        event.request_id,
                        self._arrival_ps(event.request_id),
                        event.physical_time_ps,
                        False,
                    )
                    failed.append(event.request_id)

        for request in self.spec.requests:
            if request.id in self._settled:
                continue
            deadline = self._deadline_ps(request.id)
            if deadline is not None and deadline == event_time:
                settle_deadline(request.id, deadline)
        return completed, tuple(failed), tuple(dict.fromkeys(observed_failures))

    def _release_settled_resources(self) -> None:
        """Release resident physical pairs belonging to settled requests.

        In-flight operations retain their input pairs until their own event is
        consumed; calling this helper after every event batch makes release
        idempotent and also cleans outputs created by late operations from a
        request that was already settled.
        """

        for request_id in tuple(sorted(self._settled)):
            self.executor.release_request(request_id)

    def _launch_rejection_events(
        self,
        operations: Sequence[ConstructionOperation],
    ) -> tuple[ExecutionEvent, ...]:
        snapshot = self.executor.snapshot()
        events = []
        by_request: dict[str, ConstructionOperation] = {}
        for operation in operations:
            by_request.setdefault(operation.request_id, operation)
        for request_id, operation in sorted(by_request.items()):
            self._boundary_event_counter += 1
            event = ExecutionEvent(
                event_id=(
                    f"launch-rejection-{self._boundary_event_counter:08d}"
                ),
                operation_id=operation.op_id,
                request_id=request_id,
                attempt_id=(
                    f"{operation.op_id}:launch-rejection:"
                    f"{self._boundary_event_counter}"
                ),
                event_kind="launch_rejection",
                physical_time_ps=self.time_ps,
                success=False,
                failure_cause="executor_launch_rejection",
                in_flight_operation_ids=tuple(
                    item.operation_id for item in snapshot.in_flight
                ),
            )
            events.append(event)
            self._event_log.append(event)
        self._event_log.sort(
            key=lambda item: (item.physical_time_ps, item.event_id)
        )
        return tuple(events)

    def _advance(self) -> tuple[tuple[ExecutionEvent, ...], int]:
        previous = self.time_ps
        if self.executor.has_in_flight:
            future_boundaries = [
                self._arrival_ps(request.id)
                for request in self.spec.requests
                if request.id not in self._settled
                and self._arrival_ps(request.id) > previous
            ] + [
                deadline
                for request in self.spec.requests
                if request.id not in self._settled
                for deadline in (self._deadline_ps(request.id),)
                if deadline is not None and deadline > previous
            ]
            boundary_ps = (
                min(self.horizon_ps, min(future_boundaries))
                if future_boundaries
                else None
            )
            batch = self.executor.advance_to_next_event(boundary_ps=boundary_ps)
        else:
            future_arrivals = [
                self._arrival_ps(request.id) for request in self.spec.requests
                if request.id not in self._settled and self._arrival_ps(request.id) > previous
            ]
            future_deadlines = [
                deadline
                for request in self.spec.requests
                if request.id not in self._settled
                for deadline in (self._deadline_ps(request.id),)
                if deadline is not None and deadline > previous
            ]
            boundaries = future_arrivals + future_deadlines
            if boundaries:
                batch = self.executor.wait_until(
                    min(min(boundaries), self.horizon_ps)
                )
            else:
                batch = self.executor.wait_until(self.horizon_ps)
        self._event_log.extend(batch.events)
        self._observe_memory_usage()
        return batch.events, max(0, batch.physical_time_ps - previous)

    def _finalize_horizon(self) -> tuple[str, ...]:
        failed = []
        for request in self.spec.requests:
            if request.id not in self._settled:
                self._settled[request.id] = RequestSettlement(
                    request.id,
                    self._arrival_ps(request.id),
                    self.horizon_ps,
                    False,
                )
                failed.append(request.id)
        return tuple(failed)

    def _interval_holding_cost(
        self,
        interval_start_ps: int,
        interval_end_ps: int,
        unsettled_at_start: set[str],
    ) -> int:
        cost = 0
        for request in self.spec.requests:
            if request.id not in unsettled_at_start:
                continue
            settlement = self._settled.get(request.id)
            request_end = (
                interval_end_ps
                if settlement is None
                else min(interval_end_ps, settlement.settlement_time)
            )
            request_start = max(interval_start_ps, self._arrival_ps(request.id))
            cost += max(0, request_end - request_start)
        return cost

    def _step_result(self, reward: float, terminated: bool, info: Mapping[str, object]) -> ConstructionStep:
        observation = replace(
            self.executor.snapshot(),
            settled_request_ids=tuple(sorted(self._settled)),
        )
        return ConstructionStep(
            observation,
            self.ready_operations(),
            float(reward),
            bool(terminated),
            dict(info),
        )

    def step(self, operations: Sequence[ConstructionOperation] = ()) -> ConstructionStep:
        if self._executor is None:
            raise RuntimeError("environment must be reset first")
        interval_start_ps = self.time_ps
        delivered_pairs_before = sum(
            len(values) for values in self._delivered_terminal_segments.values()
        )
        unsettled_at_start = {
            request.id for request in self.spec.requests
            if request.id not in self._settled
        }
        selected = tuple(operations)
        launch_rejected = False
        if selected:
            ready_ids = {operation.op_id for operation in self.ready_operations()}
            if any(operation.op_id not in ready_ids for operation in selected):
                raise ValueError("operation set contains a non-ready operation")
            self._executor_launch_batch_attempt_count += 1
            try:
                self.executor.launch(selected)
            except ConstructionLaunchRejected:
                self._executor_rejection_count += 1
                events = self._launch_rejection_events(selected)
                duration = 0
                launch_rejected = True
            else:
                self._observe_memory_usage()
        elif not self.stop_legal():
            raise ValueError("STOP is not legal in the current construction state")
        if not launch_rejected:
            events, duration = self._advance()
        completed_now, failed_ids, observed_failures = self._settle_events(
            events, interval_start_ps
        )
        delivered_pairs_after = sum(
            len(values) for values in self._delivered_terminal_segments.values()
        )
        self._release_settled_resources()
        self._observe_memory_usage()
        terminated = False
        if self.time_ps >= self.horizon_ps or (
            len(self._settled) == len(self.spec.requests)
            and not self.executor.has_in_flight
        ):
            failed_ids += self._finalize_horizon()
            self._release_settled_resources()
            terminated = True
            if not self.executor.terminated:
                self.executor.terminate()
        holding_cost = self._interval_holding_cost(
            interval_start_ps,
            self.time_ps,
            unsettled_at_start,
        )
        failure_lump = sum(
            max(0, self.horizon_ps - self._settled[request_id].settlement_time)
            for request_id in failed_ids
        )
        flow_cost = holding_cost + failure_lump
        self._flow_cost_ps += flow_cost
        batch_size = max(len(self.spec.requests), 1)
        flow_scale = max(batch_size * self.horizon_ps, 1)
        failed_now = len(failed_ids)
        reward = (
            self.alpha * completed_now / batch_size
            - self.beta * flow_cost / flow_scale
            - self.chi * failed_now / batch_size
        )
        self._last_time_ps = self.time_ps
        if terminated:
            expected_flow = censored_flow_time(
                tuple(self._settled[request.id] for request in self.spec.requests),
                self.horizon_ps,
            )
            if self._flow_cost_ps != expected_flow:
                raise RuntimeError(
                    "event flow-time accounting diverged from censored flow-time"
                )
        return self._step_result(
            reward,
            terminated,
            {
                "duration_ps": duration,
                "completed_now": completed_now,
                "delivered_pairs_now": delivered_pairs_after - delivered_pairs_before,
                "delivered_pairs_total": delivered_pairs_after,
                "failed_events_now": failed_now,
                "observed_failures_now": len(observed_failures),
                "observed_failure_request_ids": tuple(observed_failures),
                "observed_expiration_request_ids": tuple(
                    sorted({
                        event.request_id
                        for event in events
                        if event.failure_cause == "expiration"
                    })
                ),
                "settled_request_ids": tuple(sorted(self._settled)),
                "settled": len(self._settled),
                "risk_count": sum(not settlement.success for settlement in self._settled.values()),
                "flow_holding_cost_ps": holding_cost,
                "flow_failure_lump_ps": failure_lump,
                "flow_cost_ps": flow_cost,
                "cumulative_flow_cost_ps": self._flow_cost_ps,
            },
        )

    def repair(
        self,
        request_id: str,
        operations: tuple[ConstructionOperation, ...],
        *,
        terminal_segment_ids: tuple[str, ...] | None = None,
        supersede_uncommitted: bool = False,
    ) -> ConstructionStep:
        if self._executor is None:
            raise RuntimeError("environment must be reset first")
        if self.auto_settle_failures:
            raise RuntimeError("repair requires auto_settle_failures=False")
        if request_id in self._settled:
            raise ValueError("request is already settled")
        if terminal_segment_ids is not None:
            if not terminal_segment_ids or len(set(terminal_segment_ids)) != len(
                terminal_segment_ids
            ):
                raise ValueError(
                    "replacement terminal segment IDs must be unique and non-empty"
                )
        self.executor.repair(
            request_id,
            operations,
            supersede_uncommitted=supersede_uncommitted,
        )
        self._observe_memory_usage()
        if terminal_segment_ids is not None:
            self._terminal_segments[request_id] = tuple(terminal_segment_ids)
        return self._step_result(
            0.0,
            False,
            {
                "event_kind": "repair",
                "duration_ps": 0,
                "request_id": request_id,
                "repair_operation_ids": tuple(operation.op_id for operation in operations),
                "terminal_segment_ids": self._terminal_segments[request_id],
                "superseded_uncommitted": bool(supersede_uncommitted),
                "settled_request_ids": tuple(sorted(self._settled)),
            },
        )

    def terminal_segment_ids(self, request_id: str) -> tuple[str, ...]:
        if self._executor is None:
            raise RuntimeError("environment must be reset first")
        if request_id not in self._terminal_segments:
            raise KeyError(request_id)
        return self._terminal_segments[request_id]

    def repair_options(
        self, request_id: str
    ) -> tuple[tuple[ConstructionOperation, ...], ...]:
        if self._executor is None:
            raise RuntimeError("environment must be reset first")
        if self.auto_settle_failures:
            raise RuntimeError("repair requires auto_settle_failures=False")
        if request_id in self._settled:
            return ()
        return tuple(self.executor.repair_options(request_id))

    def drop(self, request_id: str) -> ConstructionStep:
        if self._executor is None:
            raise RuntimeError("environment must be reset first")
        if self.auto_settle_failures:
            raise RuntimeError("drop requires auto_settle_failures=False")
        if request_id in self._settled:
            raise ValueError("request is already settled")
        if any(
            pending.request_id == request_id
            for pending in self.executor.snapshot().in_flight
        ):
            raise RuntimeError("cannot drop a request with in-flight operations")
        settlement = RequestSettlement(
            request_id,
            self._arrival_ps(request_id),
            self.time_ps,
            False,
        )
        self._settled[request_id] = settlement
        self._release_settled_resources()
        self._observe_memory_usage()
        lump = max(0, self.horizon_ps - self.time_ps)
        self._flow_cost_ps += lump
        batch_size = max(len(self.spec.requests), 1)
        flow_scale = max(batch_size * self.horizon_ps, 1)
        reward = -self.beta * lump / flow_scale - self.chi / batch_size
        terminated = (
            len(self._settled) == len(self.spec.requests)
            and not self.executor.has_in_flight
        )
        if terminated:
            self._finalize_horizon()
            self._release_settled_resources()
            if not self.executor.terminated:
                self.executor.terminate()
        return self._step_result(
            reward,
            terminated,
            {
                "event_kind": "drop",
                "duration_ps": 0,
                "request_id": request_id,
                "flow_failure_lump_ps": lump,
                "settled_request_ids": tuple(sorted(self._settled)),
                "risk_count": sum(
                    not value.success for value in self._settled.values()
                ),
            },
        )

    def metrics(self) -> dict[str, float]:
        self._observe_memory_usage()
        settlements = tuple(
            self._settled.get(
                request.id,
                RequestSettlement(
                    request.id,
                    self._arrival_ps(request.id),
                    self.horizon_ps,
                    False,
                ),
            )
            for request in self.spec.requests
        )
        flow = censored_flow_time(settlements, self.horizon_ps)
        completed = sum(settlement.success for settlement in settlements)
        delivered_pairs = sum(
            len(values) for values in self._delivered_terminal_segments.values()
        )
        successful_latencies = [
            settlement.settlement_time - settlement.arrival_time
            for settlement in settlements
            if settlement.success
        ]
        ordered_latencies = sorted(successful_latencies)
        if ordered_latencies:
            p95_index = min(
                len(ordered_latencies) - 1,
                max(0, int(round(0.95 * (len(ordered_latencies) - 1)))),
            )
            p95_latency = float(ordered_latencies[p95_index])
        else:
            p95_latency = 0.0
        event_metrics = execution_event_metrics(self._event_log)
        memory_metrics = self._memory_telemetry.metrics(
            self.spec.physical.slot_duration_ps
        )
        return {
            "completed_requests": float(completed),
            "delivered_pairs": float(delivered_pairs),
            "completion_rate": completed / max(len(settlements), 1),
            "censored_flow_time_ps": float(flow),
            "mean_censored_latency_ps": flow / max(len(settlements), 1),
            "p95_completion_latency_ps": p95_latency,
            "risk_count": float(len(settlements) - completed),
            "event_count": float(len(self._event_log)),
            "makespan_ps": float(
                max((event.physical_time_ps for event in self._event_log), default=0)
            ),
            "executor_launch_batch_attempt_count": float(
                self._executor_launch_batch_attempt_count
            ),
            "executor_rejection_count": float(
                self._executor_rejection_count
            ),
            **memory_metrics,
            **event_metrics,
        }
