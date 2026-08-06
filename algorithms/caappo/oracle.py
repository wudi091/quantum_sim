"""Exact small-instance oracle for nominal construction plans.

The oracle intentionally uses the deterministic neutral executor. It is a
bounded validation instrument for catalogue and scheduling gaps, not a
replacement for stochastic SeQUeNCe evaluation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

from qnet_core.construction_api import ConstructionOperation
from qnet_core.construction_catalog import (
    RouteConstructionCandidate,
    candidates_by_request,
)
from qnet_core.construction_decoder import CapacityFeasibilityOracle
from qnet_core.construction_executor import ConstructionDAGExecutor
from qnet_core.construction_metrics import RequestSettlement, censored_flow_time
from qnet_core.spec import EpisodeSpec


class OracleLimitError(RuntimeError):
    """Raised when an instance exceeds the declared exact-search budget."""


@dataclass(frozen=True)
class DeterministicOracleResult:
    score: float
    completed_requests: int
    censored_flow_time_ps: int
    risk_count: int
    makespan_ps: int
    selected_candidate_ids: tuple[tuple[str, str], ...]
    action_trace: tuple[tuple[str, ...], ...]
    explored_states: int
    explored_joint_plans: int

    def optimality_gap(self, score: float) -> float:
        """Return the non-negative additive gap to the exact nominal score."""

        return max(0.0, self.score - float(score))


@dataclass
class _OracleState:
    executor: ConstructionDAGExecutor
    settled: dict[str, RequestSettlement]
    delivered: dict[str, set[str]]
    action_trace: list[tuple[str, ...]]


class DeterministicJointPlanOracle:
    """Exhaustively search catalogue choices and feasible concurrent sets."""

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        chi: float = 1.0,
        max_joint_plans: int = 4096,
        max_operations: int = 16,
        max_states: int = 100_000,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.chi = float(chi)
        self.max_joint_plans = int(max_joint_plans)
        self.max_operations = int(max_operations)
        self.max_states = int(max_states)
        if min(self.max_joint_plans, self.max_operations, self.max_states) < 1:
            raise ValueError("oracle limits must be positive")

    @staticmethod
    def _capacities(spec: EpisodeSpec) -> dict[str, int]:
        capacities: dict[str, int] = {}
        for raw_u, raw_v in spec.edges:
            u, v = sorted((raw_u, raw_v))
            capacities[f"link:{u}-{v}"] = spec.physical.memory_capacity
            capacities[f"genlane:{u}-{v}"] = spec.physical.max_width
        for node in spec.nodes:
            degree = sum(node in edge for edge in spec.edges)
            capacities[f"bsm:{node}"] = 1
            capacities[f"memory:{node}"] = (
                spec.physical.node_memory_capacity
                if spec.physical.node_memory_capacity is not None
                else max(1, degree * spec.physical.memory_capacity)
            )
        return capacities

    @staticmethod
    def _arrival_ps(spec: EpisodeSpec, request_id: str) -> int:
        request = next(request for request in spec.requests if request.id == request_id)
        return request.arrival * spec.physical.slot_duration_ps

    @staticmethod
    def _deadline_ps(spec: EpisodeSpec, request_id: str) -> int | None:
        request = next(request for request in spec.requests if request.id == request_id)
        return (
            None
            if request.deadline is None
            else request.deadline * spec.physical.slot_duration_ps
        )

    @staticmethod
    def _feasible_sets(
        ready: Sequence[ConstructionOperation],
        oracle: CapacityFeasibilityOracle,
        stop_legal: bool,
    ) -> tuple[tuple[ConstructionOperation, ...], ...]:
        ordered = tuple(sorted(ready, key=lambda operation: operation.canonical_key))
        results: list[tuple[ConstructionOperation, ...]] = []

        def visit(index: int, prefix: tuple[ConstructionOperation, ...]) -> None:
            if index == len(ordered):
                if prefix or stop_legal:
                    results.append(prefix)
                return
            visit(index + 1, prefix)
            operation = ordered[index]
            if oracle.can_add(prefix, operation):
                visit(index + 1, prefix + (operation,))

        visit(0, ())
        return tuple(sorted(
            set(results),
            key=lambda values: (len(values), tuple(item.canonical_key for item in values)),
        ))

    def _settle_events(
        self,
        spec: EpisodeSpec,
        terminal_segments: Mapping[str, frozenset[str]],
        state: _OracleState,
        events,
    ) -> None:
        requests = {request.id: request for request in spec.requests}
        for event in events:
            if event.request_id in state.settled:
                continue
            request = requests[event.request_id]
            if not event.success:
                state.settled[event.request_id] = RequestSettlement(
                    event.request_id,
                    self._arrival_ps(spec, event.request_id),
                    event.physical_time_ps,
                    False,
                )
                state.executor.release_request(event.request_id)
                continue
            if event.output_segment_id not in terminal_segments[event.request_id]:
                continue
            if (
                event.output_fidelity is None
                or event.output_fidelity + 1e-12 < request.required_fidelity
            ):
                state.settled[event.request_id] = RequestSettlement(
                    event.request_id,
                    self._arrival_ps(spec, event.request_id),
                    event.physical_time_ps,
                    False,
                )
                state.executor.release_request(event.request_id)
                continue
            state.delivered[event.request_id].add(event.output_segment_id)
            if len(state.delivered[event.request_id]) >= request.demand_pairs:
                state.settled[event.request_id] = RequestSettlement(
                    event.request_id,
                    self._arrival_ps(spec, event.request_id),
                    event.physical_time_ps,
                    True,
                )
                state.executor.release_request(event.request_id)
        # Settlement can precede completion of an already-launched operation.
        # Releasing again after the batch removes any late output hold.
        for request_id in state.settled:
            state.executor.release_request(request_id)

    def _settle_deadlines(self, spec: EpisodeSpec, state: _OracleState) -> None:
        now = state.executor.physical_time_ps
        for request in spec.requests:
            deadline = self._deadline_ps(spec, request.id)
            if request.id in state.settled or deadline != now:
                continue
            state.settled[request.id] = RequestSettlement(
                request.id,
                self._arrival_ps(spec, request.id),
                now,
                False,
            )
            state.executor.release_request(request.id)

    def _advance(
        self,
        spec: EpisodeSpec,
        terminal_segments: Mapping[str, frozenset[str]],
        state: _OracleState,
    ) -> None:
        now = state.executor.physical_time_ps
        horizon_ps = spec.horizon * spec.physical.slot_duration_ps
        boundaries = [horizon_ps]
        boundaries.extend(
            self._arrival_ps(spec, request.id)
            for request in spec.requests
            if request.id not in state.settled
            and self._arrival_ps(spec, request.id) > now
        )
        boundaries.extend(
            deadline
            for request in spec.requests
            if request.id not in state.settled
            for deadline in (self._deadline_ps(spec, request.id),)
            if deadline is not None and deadline > now
        )
        boundary = min(boundaries)
        if state.executor.has_in_flight:
            batch = state.executor.advance_to_next_event(boundary_ps=boundary)
        else:
            batch = state.executor.wait_until(boundary)
        self._settle_events(spec, terminal_segments, state, batch.events)
        self._settle_deadlines(spec, state)

    @staticmethod
    def _terminal(spec: EpisodeSpec, state: _OracleState) -> bool:
        horizon_ps = spec.horizon * spec.physical.slot_duration_ps
        return (
            state.executor.physical_time_ps >= horizon_ps
            or (
                len(state.settled) == len(spec.requests)
                and not state.executor.has_in_flight
            )
        )

    def _finalize(self, spec: EpisodeSpec, state: _OracleState) -> None:
        horizon_ps = spec.horizon * spec.physical.slot_duration_ps
        for request in spec.requests:
            if request.id not in state.settled:
                state.settled[request.id] = RequestSettlement(
                    request.id,
                    self._arrival_ps(spec, request.id),
                    horizon_ps,
                    False,
                )

    def _score(
        self,
        spec: EpisodeSpec,
        state: _OracleState,
        selected_ids: tuple[tuple[str, str], ...],
        explored_states: int,
        explored_joint_plans: int,
    ) -> DeterministicOracleResult:
        self._finalize(spec, state)
        horizon_ps = spec.horizon * spec.physical.slot_duration_ps
        settlements = tuple(state.settled[request.id] for request in spec.requests)
        completed = sum(settlement.success for settlement in settlements)
        risk = len(settlements) - completed
        flow = censored_flow_time(settlements, horizon_ps)
        batch_size = max(len(settlements), 1)
        score = (
            self.alpha * completed / batch_size
            - self.beta * flow / max(batch_size * horizon_ps, 1)
            - self.chi * risk / batch_size
        )
        return DeterministicOracleResult(
            score=float(score),
            completed_requests=completed,
            censored_flow_time_ps=flow,
            risk_count=risk,
            makespan_ps=max(
                (event.physical_time_ps for event in state.executor.event_log),
                default=0,
            ),
            selected_candidate_ids=selected_ids,
            action_trace=tuple(state.action_trace),
            explored_states=explored_states,
            explored_joint_plans=explored_joint_plans,
        )

    @staticmethod
    def _better(
        candidate: DeterministicOracleResult,
        incumbent: DeterministicOracleResult | None,
    ) -> bool:
        if incumbent is None:
            return True
        return (
            candidate.score,
            candidate.completed_requests,
            -candidate.censored_flow_time_ps,
            -candidate.risk_count,
            tuple(candidate.selected_candidate_ids),
            candidate.action_trace,
        ) > (
            incumbent.score,
            incumbent.completed_requests,
            -incumbent.censored_flow_time_ps,
            -incumbent.risk_count,
            tuple(incumbent.selected_candidate_ids),
            incumbent.action_trace,
        )

    def solve(
        self,
        spec: EpisodeSpec,
        candidates: tuple[RouteConstructionCandidate, ...],
    ) -> DeterministicOracleResult:
        grouped = candidates_by_request(candidates)
        request_order = tuple(sorted(grouped))
        if request_order != tuple(sorted(request.id for request in spec.requests)):
            raise ValueError("candidate catalogue must cover every request")
        joint_plan_count = 1
        for request_id in request_order:
            joint_plan_count *= len(grouped[request_id])
        if joint_plan_count > self.max_joint_plans:
            raise OracleLimitError(
                f"joint plan count {joint_plan_count} exceeds {self.max_joint_plans}"
            )
        if any(
            operation.success_probability != 1.0
            for candidate in candidates
            for operation in candidate.dag.operations
        ):
            raise ValueError("exact nominal oracle requires success_probability=1")

        explored_states = 0
        explored_joint_plans = 0
        best: DeterministicOracleResult | None = None
        capacities = self._capacities(spec)
        horizon_ps = spec.horizon * spec.physical.slot_duration_ps
        if (
            spec.physical.memory_lifetime * spec.physical.slot_duration_ps
            < horizon_ps
        ):
            raise ValueError(
                "nominal oracle does not model memory expiration; require "
                "memory_lifetime to cover the episode horizon"
            )

        for choice in product(*(grouped[request_id] for request_id in request_order)):
            operation_count = sum(len(candidate.dag.operations) for candidate in choice)
            if operation_count > self.max_operations:
                raise OracleLimitError(
                    f"joint plan has {operation_count} operations; limit is {self.max_operations}"
                )
            explored_joint_plans += 1
            selected_ids = tuple(
                (candidate.request_id, candidate.candidate_id) for candidate in choice
            )
            terminal_segments = {
                candidate.request_id: frozenset(candidate.all_terminal_segment_ids)
                for candidate in choice
            }
            initial = _OracleState(
                ConstructionDAGExecutor(
                    tuple(copy.deepcopy(candidate.dag) for candidate in choice),
                    capacities,
                    seed=spec.seed,
                    horizon_ps=horizon_ps,
                ),
                {},
                {request.id: set() for request in spec.requests},
                [],
            )

            def search(state: _OracleState) -> None:
                nonlocal explored_states, best
                explored_states += 1
                if explored_states > self.max_states:
                    raise OracleLimitError(
                        f"state count exceeds exact-search limit {self.max_states}"
                    )
                if self._terminal(spec, state):
                    result = self._score(
                        spec,
                        state,
                        selected_ids,
                        explored_states,
                        explored_joint_plans,
                    )
                    if self._better(result, best):
                        best = result
                    return

                now = state.executor.physical_time_ps
                active = {
                    request.id
                    for request in spec.requests
                    if request.id not in state.settled
                    and self._arrival_ps(spec, request.id) <= now
                }
                ready = tuple(
                    operation for operation in state.executor.ready_operations()
                    if operation.request_id in active
                )
                future_arrivals = any(
                    request.id not in state.settled
                    and self._arrival_ps(spec, request.id) > now
                    for request in spec.requests
                )
                stop_legal = state.executor.has_in_flight or future_arrivals or not ready
                actions = self._feasible_sets(
                    ready,
                    CapacityFeasibilityOracle.from_snapshot(state.executor.snapshot()),
                    stop_legal,
                )
                if not actions:
                    failed = copy.deepcopy(state)
                    failed.action_trace.append(())
                    self._advance(spec, terminal_segments, failed)
                    search(failed)
                    return
                for action in actions:
                    branch = copy.deepcopy(state)
                    branch.action_trace.append(tuple(
                        operation.op_id for operation in action
                    ))
                    if action:
                        branch.executor.launch(action)
                    self._advance(spec, terminal_segments, branch)
                    search(branch)

            search(initial)

        if best is None:
            raise RuntimeError("oracle search produced no terminal state")
        return DeterministicOracleResult(
            best.score,
            best.completed_requests,
            best.censored_flow_time_ps,
            best.risk_count,
            best.makespan_ps,
            best.selected_candidate_ids,
            best.action_trace,
            explored_states,
            explored_joint_plans,
        )


__all__ = [
    "DeterministicJointPlanOracle",
    "DeterministicOracleResult",
    "OracleLimitError",
]
