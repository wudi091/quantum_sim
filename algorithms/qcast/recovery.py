"""Q-CAST phase-4 recovery decisions over neutral physical events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import networkx as nx

from qnet_core.construction_api import (
    ConstructionSnapshot,
    ExecutionEvent,
    OperationKind,
)
from qnet_core.construction_plans import left_deep_swap_suffix
from qnet_core.scheduled_execution import (
    ScheduledEventDisposition,
    ScheduledEventResponse,
    ScheduledPlanRevision,
)
from qnet_core.spec import EpisodeSpec

from .online_planner import QCASTAllocation, QCASTRecoveryPathPlan


@dataclass(frozen=True)
class QCASTRecoveryDecision:
    request_id: str
    broken_major_edges: tuple[int, ...]
    available_recovery_path_ids: tuple[str, ...]
    selected_recovery_path_ids: tuple[str, ...]
    repaired_route_nodes: tuple[int, ...]
    repaired: bool
    failure_cause: str = ""


@dataclass
class _RecoveryState:
    allocation: QCASTAllocation
    resolved: bool = False


def _select_recovery_paths(
    broken_edges: tuple[int, ...],
    available: Sequence[QCASTRecoveryPathPlan],
) -> tuple[QCASTRecoveryPathPlan, ...] | None:
    """Apply the official interval-ordered greedy broken-edge cover."""

    selected: list[QCASTRecoveryPathPlan] = []
    covered: set[int] = set()
    broken = set(broken_edges)
    broken_coverage = {
        recovery: recovery.covered_major_edges.intersection(broken)
        for recovery in available
    }
    for broken_edge in broken_edges:
        if broken_edge in covered:
            continue
        next_major_index = 0
        repaired = False
        for recovery in sorted(
            (
                recovery
                for recovery in available
                if broken_edge in recovery.covered_major_edges
                and recovery not in selected
            ),
            key=lambda recovery: (
                recovery.major_start_index,
                recovery.major_end_index,
                len(recovery.route_nodes),
                recovery.recovery_id,
            ),
        ):
            if recovery.major_start_index < next_major_index:
                continue
            next_major_index = recovery.major_end_index
            other_broken_edges = (
                broken_coverage[recovery] - {broken_edge}
            )
            if any(edge in covered for edge in other_broken_edges):
                continue
            selected.append(recovery)
            covered.add(broken_edge)
            covered.update(other_broken_edges)
            repaired = True
            break
        if not repaired:
            return None
    if not broken.issubset(covered):
        return None
    return tuple(sorted(
        selected,
        key=lambda recovery: (
            recovery.major_start_index,
            recovery.major_end_index,
            recovery.recovery_id,
        ),
    ))


def _repaired_route_and_segments(
    allocation: QCASTAllocation,
    selected_recoveries: Sequence[QCASTRecoveryPathPlan],
    live_segment_ids: set[str],
) -> tuple[tuple[int, ...], tuple[str, ...]] | None:
    """Reproduce Q-CAST's shortest path over major/recovery edge union."""

    graph = nx.Graph()
    segment_choices: dict[tuple[int, int], list[tuple[int, str]]] = {}

    def add_edge(left: int, right: int, segment_id: str, priority: int) -> None:
        if segment_id not in live_segment_ids:
            return
        key = tuple(sorted((left, right)))
        graph.add_edge(left, right)
        segment_choices.setdefault(key, []).append((priority, segment_id))

    major_route = allocation.candidate.route_nodes
    for index, segment_id in enumerate(allocation.major_segment_ids):
        add_edge(
            major_route[index],
            major_route[index + 1],
            segment_id,
            0,
        )
    for recovery_index, recovery in enumerate(selected_recoveries, start=1):
        for edge_index, segment_id in enumerate(recovery.segment_ids):
            add_edge(
                recovery.route_nodes[edge_index],
                recovery.route_nodes[edge_index + 1],
                segment_id,
                recovery_index,
            )
    source = major_route[0]
    destination = major_route[-1]
    try:
        route = tuple(
            int(node) for node in nx.shortest_path(graph, source, destination)
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    segments = []
    for left, right in zip(route, route[1:]):
        choices = segment_choices.get(tuple(sorted((left, right))), ())
        if not choices:
            return None
        segments.append(min(choices)[1])
    return route, tuple(segments)


class QCASTRecoveryPolicy:
    """Algorithm-owned event policy used by the generic persistent scheduler."""

    def __init__(self, episode: EpisodeSpec):
        self.episode = episode
        self.requests = {request.id: request for request in episode.requests}
        self._states: dict[str, _RecoveryState] = {}
        self._decisions: list[QCASTRecoveryDecision] = []

    @property
    def decisions(self) -> tuple[QCASTRecoveryDecision, ...]:
        return tuple(self._decisions)

    def register(self, allocation: QCASTAllocation) -> None:
        request_id = allocation.candidate.request_id
        if request_id in self._states and not self._states[request_id].resolved:
            raise ValueError(f"Q-CAST recovery state is already active: {request_id}")
        self._states[request_id] = _RecoveryState(allocation)

    def forget(self, request_id: str) -> None:
        self._states.pop(request_id, None)

    @staticmethod
    def _dag_state(snapshot: ConstructionSnapshot, request_id: str):
        return next(
            (
                state
                for state in snapshot.dag_states
                if state.request_id == request_id
            ),
            None,
        )

    def _resolve_request(
        self,
        request_id: str,
        events: tuple[ExecutionEvent, ...],
        snapshot: ConstructionSnapshot,
    ) -> ScheduledEventResponse | None:
        state = self._states.get(request_id)
        if state is None or state.resolved:
            return None
        allocation = state.allocation
        dag_state = self._dag_state(snapshot, request_id)
        if dag_state is None:
            return None
        generation_ids = set(allocation.all_generation_operation_ids)
        terminal_generation_ids = set(dag_state.completed) | set(dag_state.dead)
        if not generation_ids.issubset(terminal_generation_ids):
            if any(
                not event.success
                and event.event_kind == OperationKind.GEN.lower()
                and event.operation_id in generation_ids
                for event in events
            ):
                return ScheduledEventResponse(
                    request_id=request_id,
                    disposition=ScheduledEventDisposition.CONTINUE,
                )
            return None

        live_segments = {
            segment.segment_id
            for segment in snapshot.segments
            if segment.request_id == request_id
        }
        broken_edges = tuple(
            index
            for index, segment_id in enumerate(allocation.major_segment_ids)
            if segment_id not in live_segments
        )
        available_recoveries = tuple(
            recovery
            for recovery in allocation.recovery_paths
            if set(recovery.segment_ids).issubset(live_segments)
        )
        available_ids = tuple(
            recovery.recovery_id for recovery in available_recoveries
        )
        if not broken_edges:
            used = set(allocation.major_segment_ids)
            released = tuple(sorted(
                live_segments.intersection(allocation.all_elementary_segment_ids)
                - used
            ))
            state.resolved = True
            self._decisions.append(QCASTRecoveryDecision(
                request_id=request_id,
                broken_major_edges=(),
                available_recovery_path_ids=available_ids,
                selected_recovery_path_ids=(),
                repaired_route_nodes=allocation.candidate.route_nodes,
                repaired=False,
            ))
            return ScheduledEventResponse(
                request_id=request_id,
                disposition=ScheduledEventDisposition.CONTINUE,
                release_segment_ids=released,
            )

        selected = _select_recovery_paths(broken_edges, available_recoveries)
        if selected is None:
            state.resolved = True
            self._decisions.append(QCASTRecoveryDecision(
                request_id=request_id,
                broken_major_edges=broken_edges,
                available_recovery_path_ids=available_ids,
                selected_recovery_path_ids=(),
                repaired_route_nodes=(),
                repaired=False,
                failure_cause="recovery_cover_unavailable",
            ))
            return ScheduledEventResponse(
                request_id=request_id,
                disposition=ScheduledEventDisposition.FAIL,
                failure_cause="recovery_cover_unavailable",
            )

        repaired = _repaired_route_and_segments(
            allocation,
            selected,
            live_segments,
        )
        if repaired is None:
            state.resolved = True
            self._decisions.append(QCASTRecoveryDecision(
                request_id=request_id,
                broken_major_edges=broken_edges,
                available_recovery_path_ids=available_ids,
                selected_recovery_path_ids=tuple(
                    recovery.recovery_id for recovery in selected
                ),
                repaired_route_nodes=(),
                repaired=False,
                failure_cause="recovery_route_unusable",
            ))
            return ScheduledEventResponse(
                request_id=request_id,
                disposition=ScheduledEventDisposition.FAIL,
                failure_cause="recovery_route_unusable",
            )

        route_nodes, segment_ids = repaired
        if len(route_nodes) == 2:
            state.resolved = True
            self._decisions.append(QCASTRecoveryDecision(
                request_id=request_id,
                broken_major_edges=broken_edges,
                available_recovery_path_ids=available_ids,
                selected_recovery_path_ids=tuple(
                    recovery.recovery_id for recovery in selected
                ),
                repaired_route_nodes=route_nodes,
                repaired=True,
            ))
            return ScheduledEventResponse(
                request_id=request_id,
                disposition=ScheduledEventDisposition.COMPLETE,
                completion_segment_id=segment_ids[0],
            )

        next_version = dag_state.version + 1
        ordinal_start = 1 + max(
            (operation.ordinal for operation in snapshot.operations
             if operation.request_id == request_id),
            default=0,
        )
        operations, terminal_segment_id = left_deep_swap_suffix(
            request_id,
            route_nodes,
            segment_ids,
            next_version=next_version,
            ordinal_start=ordinal_start,
            operation_prefix=f"{request_id}:qcast:repair:v{next_version}",
            required_fidelity=self.requests[request_id].required_fidelity,
        )
        used = set(segment_ids)
        released = tuple(sorted(
            live_segments.intersection(allocation.all_elementary_segment_ids)
            - used
        ))
        earliest_start_slot = (
            snapshot.physical_time_ps // self.episode.physical.slot_duration_ps
            + 1
        )
        revision = ScheduledPlanRevision(
            request_id=request_id,
            operations=operations,
            relative_operation_slots=tuple(sorted(
                (operation.op_id, index)
                for index, operation in enumerate(operations)
            )),
            terminal_segment_ids=(terminal_segment_id,),
            earliest_start_slot=earliest_start_slot,
            release_segment_ids=released,
            supersede_uncommitted=True,
        )
        state.resolved = True
        self._decisions.append(QCASTRecoveryDecision(
            request_id=request_id,
            broken_major_edges=broken_edges,
            available_recovery_path_ids=available_ids,
            selected_recovery_path_ids=tuple(
                recovery.recovery_id for recovery in selected
            ),
            repaired_route_nodes=route_nodes,
            repaired=True,
        ))
        return ScheduledEventResponse(
            request_id=request_id,
            disposition=ScheduledEventDisposition.REVISE,
            revision=revision,
        )

    def on_event_batch(
        self,
        events: tuple[ExecutionEvent, ...],
        snapshot: ConstructionSnapshot,
        active_request_ids: tuple[str, ...],
    ) -> Sequence[ScheduledEventResponse]:
        by_request: dict[str, list[ExecutionEvent]] = {}
        active = set(active_request_ids)
        for event in events:
            if event.request_id in active:
                by_request.setdefault(event.request_id, []).append(event)
        responses = []
        for request_id in sorted(by_request):
            response = self._resolve_request(
                request_id,
                tuple(by_request[request_id]),
                snapshot,
            )
            if response is not None:
                responses.append(response)
        return tuple(responses)


__all__ = [
    "QCASTRecoveryDecision",
    "QCASTRecoveryPolicy",
]
