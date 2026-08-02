"""Greedy batch-conflict-aware schedule-portfolio baseline.

The generator is deterministic and exact for the currently configured path
lengths.  It first enumerates every legal complete schedule for each path,
then keeps at most four schedules with complementary functions:

* a minimum-depth balanced/swap-asap baseline;
* an early-release schedule for the strongest releasable hotspot;
* an early-release schedule for the second hotspot;
* marginal request-level conflict coverage and resource-profile diversity.

Pairwise request coverage is only a generation heuristic.  It never replaces
the full batch feasibility check performed by the MILP selector or executor.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import cached_property
from typing import Mapping, Sequence

from qnet_core.contracts.complete_schedule import (
    CompleteSchedule,
    Node,
    enumerate_complete_schedules,
)


MAX_SCHEDULES_PER_PATH = 4


def _stable_key(value: object) -> tuple[str, str]:
    return type(value).__name__, repr(value)


@dataclass(frozen=True)
class MemoryBoundary:
    """Nominal per-path memory occupancy after ``completed_groups`` groups."""

    completed_groups: int
    occupancy: tuple[tuple[Node, int], ...]

    @cached_property
    def by_node(self) -> dict[Node, int]:
        return dict(self.occupancy)

    def at(self, node: Node) -> int:
        return self.by_node.get(node, 0)


@dataclass(frozen=True)
class ConflictSignature:
    """Request-level conflicts that one schedule can nominally unlock."""

    contested_nodes: tuple[Node, ...]
    coverage_request_ids: tuple[str, ...]
    unlock_rounds: tuple[tuple[str, int], ...]

    @cached_property
    def unlock_round_by_request(self) -> dict[str, int]:
        return dict(self.unlock_rounds)


@dataclass(frozen=True)
class ScheduleEstimate:
    """Slot-snapshot annotation kept separate from structural schedule data."""

    memory_profile: tuple[MemoryBoundary, ...]
    release_rounds: tuple[tuple[Node, int], ...]
    conflict_signature: ConflictSignature
    node_conflict_counts: tuple[tuple[Node, int], ...]
    hotspot_scores: tuple[tuple[Node, float], ...]
    memory_time: int
    weighted_memory_time: int
    selection_reason: str = "unselected"
    target_node: Node | None = None

    @cached_property
    def release_round_by_node(self) -> dict[Node, int]:
        return dict(self.release_rounds)


@dataclass(frozen=True)
class GeneratedSchedule:
    """One complete schedule plus its current-batch resource annotation."""

    schedule: CompleteSchedule
    estimate: ScheduleEstimate

    @property
    def groups(self):
        return self.schedule.groups

    @property
    def path(self):
        return self.schedule.path


def _path_footprint(path: tuple[Node, ...]) -> dict[Node, int]:
    footprint: dict[Node, int] = defaultdict(int)
    for left, right in zip(path, path[1:]):
        footprint[left] += 1
        footprint[right] += 1
    return dict(footprint)


def _memory_profile(schedule: CompleteSchedule) -> tuple[MemoryBoundary, ...]:
    path = schedule.path
    release = schedule.release_round_by_node
    boundaries: list[MemoryBoundary] = []
    for completed_groups in range(schedule.round_count + 1):
        occupancy: list[tuple[Node, int]] = []
        for index, node in enumerate(path):
            if index in (0, len(path) - 1):
                value = 1
            else:
                value = 2 if release[node] > completed_groups else 0
            occupancy.append((node, value))
        boundaries.append(MemoryBoundary(
            completed_groups=completed_groups,
            occupancy=tuple(occupancy),
        ))
    return tuple(boundaries)


def _normalize_request_paths(
    request_paths: Mapping[str, Sequence[Sequence[Node]]],
) -> dict[str, tuple[tuple[Node, ...], ...]]:
    normalized: dict[str, tuple[tuple[Node, ...], ...]] = {}
    for request_id, paths in request_paths.items():
        if not request_id:
            raise ValueError("request IDs must be non-empty")
        concrete = tuple(tuple(path) for path in paths)
        if not concrete:
            raise ValueError("every request needs at least one candidate path")
        # Construction performs the structural path validation.
        for path in concrete:
            enumerate_complete_schedules(path)
        normalized[str(request_id)] = concrete
    if not normalized:
        raise ValueError("a batch needs at least one request")
    return normalized


def _request_pressure(
    request_paths: Mapping[str, tuple[tuple[Node, ...], ...]],
) -> dict[Node, int]:
    requests_by_node: dict[Node, set[str]] = defaultdict(set)
    for request_id, paths in request_paths.items():
        seen_for_request: set[Node] = set()
        for path in paths:
            seen_for_request.update(path)
        for node in seen_for_request:
            requests_by_node[node].add(request_id)
    return {
        node: len(request_ids)
        for node, request_ids in requests_by_node.items()
    }


def _relief_masks_by_request(
    owner_request_id: str,
    path: tuple[Node, ...],
    request_paths: Mapping[str, tuple[tuple[Node, ...], ...]],
    available_memory: Mapping[Node, int],
) -> dict[str, tuple[frozenset[Node], ...]]:
    """Precompute conflicts this path can remove before it completes.

    A request that already has an initially compatible alternative path is
    excluded: it does not need this schedule to resolve its conflict.
    """

    own_footprint = _path_footprint(path)
    releasable = set(path[1:-1])
    result: dict[str, tuple[frozenset[Node], ...]] = {}
    for other_request_id in sorted(request_paths, key=_stable_key):
        if other_request_id == owner_request_id:
            continue
        masks: set[frozenset[Node]] = set()
        already_compatible = False
        for other_path in request_paths[other_request_id]:
            other_footprint = _path_footprint(other_path)
            if any(
                demand > available_memory.get(node, 0)
                for node, demand in other_footprint.items()
            ):
                continue
            shared_nodes = set(own_footprint) | set(other_footprint)
            blockers = frozenset(
                node for node in shared_nodes
                if own_footprint.get(node, 0)
                + other_footprint.get(node, 0)
                > available_memory.get(node, 0)
            )
            if not blockers:
                already_compatible = True
                break
            if blockers <= releasable:
                masks.add(blockers)
        if not already_compatible and masks:
            result[other_request_id] = tuple(sorted(
                masks,
                key=lambda mask: (
                    len(mask),
                    tuple(sorted(map(_stable_key, mask))),
                ),
            ))
    return result


def _annotate_catalogue(
    schedules: tuple[CompleteSchedule, ...],
    *,
    relief_masks: Mapping[str, tuple[frozenset[Node], ...]],
    pressure: Mapping[Node, int],
    available_memory: Mapping[Node, int],
) -> tuple[GeneratedSchedule, ...]:
    if not schedules:
        return ()
    path = schedules[0].path
    internal = path[1:-1]
    conflict_count = {
        node: sum(
            any(node in mask for mask in masks)
            for masks in relief_masks.values()
        )
        for node in internal
    }
    contested_nodes = tuple(
        node for node in internal
        if pressure.get(node, 0) > 1 or conflict_count[node] > 0
    )
    hotspot_scores = tuple(
        (
            node,
            float(Fraction(
                pressure.get(node, 0),
                max(available_memory.get(node, 0), 1),
            )),
        )
        for node in internal
    )

    result: list[GeneratedSchedule] = []
    for schedule in schedules:
        release = schedule.release_round_by_node
        unlock: list[tuple[str, int]] = []
        coverage: list[str] = []
        for request_id in sorted(relief_masks, key=_stable_key):
            unlock_round = min(
                max(release[node] for node in mask)
                for mask in relief_masks[request_id]
            )
            unlock.append((request_id, unlock_round))
            if unlock_round < schedule.round_count:
                coverage.append(request_id)

        # Internal nodes occupy two memories until their swap starts; the two
        # path endpoints remain occupied for the K nominal group intervals.
        memory_time = (
            2 * sum(release[node] for node in internal)
            + 2 * schedule.round_count
        )
        weighted_memory_time = (
            2 * sum(
                (1 + conflict_count[node]) * release[node]
                for node in internal
            )
            + 2 * schedule.round_count
        )
        result.append(GeneratedSchedule(
            schedule=schedule,
            estimate=ScheduleEstimate(
                memory_profile=_memory_profile(schedule),
                release_rounds=schedule.release_rounds,
                conflict_signature=ConflictSignature(
                    contested_nodes=contested_nodes,
                    coverage_request_ids=tuple(coverage),
                    unlock_rounds=tuple(unlock),
                ),
                node_conflict_counts=tuple(
                    (node, conflict_count[node]) for node in internal
                ),
                hotspot_scores=hotspot_scores,
                memory_time=memory_time,
                weighted_memory_time=weighted_memory_time,
            ),
        ))
    return tuple(result)


def _hotspot_nodes(
    catalogue: tuple[GeneratedSchedule, ...],
    pressure: Mapping[Node, int],
    available_memory: Mapping[Node, int],
) -> tuple[Node, ...]:
    if not catalogue:
        return ()
    path = catalogue[0].path
    conflict_count = dict(catalogue[0].estimate.node_conflict_counts)
    candidates = tuple(
        node for node in path[1:-1]
        if pressure.get(node, 0) > 1 or conflict_count.get(node, 0) > 0
    )
    position = {node: index for index, node in enumerate(path)}
    return tuple(sorted(candidates, key=lambda node: (
        -conflict_count.get(node, 0),
        -Fraction(
            pressure.get(node, 0),
            max(available_memory.get(node, 0), 1),
        ),
        available_memory.get(node, 0),
        position[node],
    )))


def _release_distance(
    left: GeneratedSchedule,
    right: GeneratedSchedule,
) -> int:
    left_release = left.estimate.release_round_by_node
    right_release = right.estimate.release_round_by_node
    conflict_count = dict(left.estimate.node_conflict_counts)
    return (
        2 * sum(
            (1 + conflict_count[node])
            * abs(left_release[node] - right_release[node])
            for node in left.schedule.path[1:-1]
        )
        + 2 * abs(
            left.schedule.round_count - right.schedule.round_count
        )
    )


def select_schedule_portfolio(
    catalogue: Sequence[GeneratedSchedule],
    *,
    pressure: Mapping[Node, int],
    available_memory: Mapping[Node, int],
    limit: int,
) -> tuple[GeneratedSchedule, ...]:
    """Select a deterministic, functionally complementary finite portfolio."""

    if not 1 <= limit <= MAX_SCHEDULES_PER_PATH:
        raise ValueError(
            f"schedule portfolio limit must lie in [1, {MAX_SCHEDULES_PER_PATH}]"
        )
    candidates = tuple(catalogue)
    if not candidates:
        return ()
    path = candidates[0].path
    if any(candidate.path != path for candidate in candidates):
        raise ValueError("one portfolio can contain schedules for only one path")

    selected: list[GeneratedSchedule] = []
    selected_keys: set[object] = set()

    def add(candidate: GeneratedSchedule, reason: str, target=None) -> None:
        key = candidate.schedule.structural_key
        if key in selected_keys or len(selected) >= limit:
            return
        selected_keys.add(key)
        selected.append(replace(
            candidate,
            estimate=replace(
                candidate.estimate,
                selection_reason=reason,
                target_node=target,
            ),
        ))

    baseline = min(candidates, key=lambda candidate: (
        candidate.schedule.round_count,
        candidate.estimate.memory_time,
        candidate.schedule.structural_key,
    ))
    add(baseline, "balanced")

    for rank, hotspot in enumerate(
        _hotspot_nodes(candidates, pressure, available_memory)[:2],
        start=1,
    ):
        targeted = min(candidates, key=lambda candidate: (
            candidate.estimate.release_round_by_node[hotspot],
            candidate.schedule.round_count,
            candidate.estimate.weighted_memory_time,
            candidate.estimate.memory_time,
            candidate.schedule.structural_key,
        ))
        add(targeted, f"hotspot_{rank}", hotspot)

    while len(selected) < min(limit, len(candidates)):
        covered = {
            request_id
            for candidate in selected
            for request_id in (
                candidate.estimate.conflict_signature.coverage_request_ids
            )
        }
        best_unlock: dict[str, int] = {}
        for candidate in selected:
            signature = candidate.estimate.conflict_signature
            coverage = set(signature.coverage_request_ids)
            for request_id, unlock_round in signature.unlock_rounds:
                if request_id not in coverage:
                    continue
                best_unlock[request_id] = min(
                    best_unlock.get(request_id, unlock_round),
                    unlock_round,
                )

        remaining = tuple(
            candidate for candidate in candidates
            if candidate.schedule.structural_key not in selected_keys
        )

        def marginal_key(candidate: GeneratedSchedule):
            signature = candidate.estimate.conflict_signature
            candidate_coverage = set(signature.coverage_request_ids)
            delta_coverage = len(candidate_coverage - covered)
            unlock = dict(signature.unlock_rounds)
            advance = sum(
                max(0, best_unlock[request_id] - unlock[request_id])
                for request_id in candidate_coverage & set(best_unlock)
            )
            distance = min(
                _release_distance(candidate, existing)
                for existing in selected
            )
            return (
                -delta_coverage,
                -advance,
                -distance,
                candidate.schedule.round_count,
                candidate.estimate.weighted_memory_time,
                candidate.estimate.memory_time,
                candidate.schedule.structural_key,
            )

        add(min(remaining, key=marginal_key), "marginal_coverage")

    return tuple(selected)


def generate_batch_schedule_portfolios(
    request_paths: Mapping[str, Sequence[Sequence[Node]]],
    node_capacities: Mapping[Node, int],
    *,
    available_memory: Mapping[Node, int] | None = None,
    limit_per_path: int | None = MAX_SCHEDULES_PER_PATH,
) -> dict[tuple[str, int], tuple[GeneratedSchedule, ...]]:
    """Generate complete schedule portfolios for every request/path in a batch.

    ``limit_per_path=None`` returns the full legal catalogue for an exhaustive
    small-scale ablation.  The formal online action catalogue uses a limit in
    ``[1, 4]``.
    """

    paths_by_request = _normalize_request_paths(request_paths)
    capacity = {node: int(value) for node, value in node_capacities.items()}
    if any(value < 1 for value in capacity.values()):
        raise ValueError("node capacities must be positive")
    available = (
        dict(capacity)
        if available_memory is None
        else {node: int(value) for node, value in available_memory.items()}
    )
    used_nodes = {
        node
        for paths in paths_by_request.values()
        for path in paths
        for node in path
    }
    if not used_nodes <= set(capacity) or not used_nodes <= set(available):
        raise ValueError("capacity and available-memory maps must cover all paths")
    if any(available[node] < 0 or available[node] > capacity[node]
           for node in used_nodes):
        raise ValueError("available memory must lie between zero and capacity")
    if limit_per_path is not None and not (
        1 <= limit_per_path <= MAX_SCHEDULES_PER_PATH
    ):
        raise ValueError(
            f"limit_per_path must be None or lie in [1, "
            f"{MAX_SCHEDULES_PER_PATH}]"
        )

    pressure = _request_pressure(paths_by_request)
    result: dict[tuple[str, int], tuple[GeneratedSchedule, ...]] = {}
    for request_id in sorted(paths_by_request, key=_stable_key):
        for path_index, path in enumerate(paths_by_request[request_id]):
            relief_masks = _relief_masks_by_request(
                request_id,
                path,
                paths_by_request,
                available,
            )
            catalogue = _annotate_catalogue(
                enumerate_complete_schedules(path),
                relief_masks=relief_masks,
                pressure=pressure,
                available_memory=available,
            )
            if limit_per_path is None:
                portfolio = tuple(
                    replace(
                        candidate,
                        estimate=replace(
                            candidate.estimate,
                            selection_reason="exhaustive",
                        ),
                    )
                    for candidate in catalogue
                )
            else:
                portfolio = select_schedule_portfolio(
                    catalogue,
                    pressure=pressure,
                    available_memory=available,
                    limit=limit_per_path,
                )
            result[(request_id, path_index)] = portfolio
    return result


def generate_static_schedule_portfolio(
    path: Sequence[Node],
    *,
    limit: int | None = MAX_SCHEDULES_PER_PATH,
) -> tuple[CompleteSchedule, ...]:
    """Topology/request-independent fallback library for one fixed path.

    This is a deterministic fixed-template heuristic, not the proposed
    offline scenario MILP.  It is useful before a fitted library is available
    and as an ablation baseline for that MILP.
    """

    path = tuple(path)
    capacities = {node: 2 for node in path}
    generated = generate_batch_schedule_portfolios(
        {"offline-template": (path,)},
        capacities,
        limit_per_path=limit,
    )[("offline-template", 0)]
    return tuple(candidate.schedule for candidate in generated)
