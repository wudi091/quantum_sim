"""P4 recovery-loop operations for Q-CAST major paths."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .model import ChannelRef, MajorReservation, RecoveryReservation

Edge = tuple[int, int]


def edge(u: int, v: int) -> Edge:
    return (u, v) if u < v else (v, u)


def path_edges(path: Sequence[int]) -> frozenset[Edge]:
    return frozenset(edge(u, v) for u, v in zip(path, path[1:]))


def xor_edges(*edge_sets: Iterable[Edge]) -> frozenset[Edge]:
    """Symmetric difference used by Fig. 10's recovery-loop construction."""

    result: set[Edge] = set()
    for values in edge_sets:
        for value in values:
            item = edge(*value)
            if item in result:
                result.remove(item)
            else:
                result.add(item)
    return frozenset(result)


def recovery_loop_edges(major_path: Sequence[int], recovery: RecoveryReservation) -> frozenset[Edge]:
    """Return major segment XOR recovery-path edges."""

    segment = major_path[recovery.start_index:recovery.end_index + 1]
    return xor_edges(path_edges(segment), path_edges(recovery.path))


def connected(edges: Iterable[Edge], source: int, destination: int) -> bool:
    adjacency: dict[int, set[int]] = {}
    for u, v in edges:
        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)
    todo = [source]
    seen = {source}
    while todo:
        node = todo.pop()
        if node == destination:
            return True
        for neighbour in adjacency.get(node, ()):
            if neighbour not in seen:
                seen.add(neighbour)
                todo.append(neighbour)
    return destination in seen


def shortest_path_from_edges(edges: Iterable[Edge], source: int, destination: int) -> tuple[int, ...] | None:
    adjacency: dict[int, list[int]] = {}
    for u, v in edges:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)
    queue = [source]
    previous: dict[int, int | None] = {source: None}
    for node in queue:
        if node == destination:
            break
        for neighbour in sorted(adjacency.get(node, ())):
            if neighbour not in previous:
                previous[neighbour] = node
                queue.append(neighbour)
    if destination not in previous:
        return None
    path: list[int] = []
    node: int | None = destination
    while node is not None:
        path.append(node)
        node = previous[node]
    return tuple(reversed(path))


@dataclass(frozen=True)
class LaneOutcome:
    success: bool
    path: tuple[int, ...] | None
    broken_edges: frozenset[Edge]
    picked_recoveries: tuple[RecoveryReservation, ...]


def select_recovery_loops(
    major: MajorReservation,
    recoveries: Sequence[RecoveryReservation],
    broken_edges: Iterable[Edge],
    *,
    compatibility: str = "author_code",
) -> tuple[tuple[RecoveryReservation, ...], frozenset[Edge]]:
    """Greedily choose non-overlapping loops in the source-code order.

    A recovery path is considered only once and may cover several consecutive
    broken major edges.  ``author_code`` follows OnlineAlgorithm.P4's
    ``next``/switch-node ordering; ``corrected`` allows a loop only when its
    symmetric difference covers the currently broken edge directly.
    """

    broken = frozenset(edge(*item) for item in broken_edges)
    ordered = sorted(
        recoveries,
        key=lambda item: (item.start_index, item.end_index, len(item.path), item.path),
    )
    picked: list[RecoveryReservation] = []
    repaired: set[Edge] = set()
    next_index = 0
    for candidate in ordered:
        if candidate.width <= 0 or candidate in picked:
            continue
        if compatibility == "author_code" and candidate.start_index < next_index:
            continue
        covered = {
            edge(*pair)
            for pair in zip(
                major.path[candidate.start_index:candidate.end_index],
                major.path[candidate.start_index + 1:candidate.end_index + 1],
            )
        }
        overlap = covered & broken
        if not overlap or overlap <= repaired:
            continue
        loop = recovery_loop_edges(major.path, candidate)
        # The loop must contribute a connected route after XOR.  This avoids
        # treating a recovery path that does not actually repair a break as
        # useful, while preserving the paper's edge-set semantics.
        if compatibility == "corrected" and not connected(
            xor_edges(path_edges(major.path), loop), major.path[0], major.path[-1]
        ):
            continue
        picked.append(candidate)
        repaired.update(overlap)
        next_index = candidate.end_index
        if repaired >= broken:
            break
    return tuple(picked), frozenset(repaired)


def recover_lane(
    major: MajorReservation,
    recoveries: Sequence[RecoveryReservation],
    successful_channels: Mapping[ChannelRef, bool],
    lane: int,
    *,
    rng: random.Random | None = None,
    swap_probability: float = 0.9,
    compatibility: str = "author_code",
    used_channels: set[ChannelRef] | None = None,
) -> LaneOutcome:
    """Run one width-1 P4 lane using major/recovery channel outcomes."""

    if rng is None:
        rng = random.Random(0)
    if used_channels is None:
        used_channels = set()
    major_edges = list(zip(major.path, major.path[1:]))
    lane_channels = major.channels[lane::major.width]
    broken: set[Edge] = set()
    for index, (u, v) in enumerate(major_edges):
        lane_refs = major.channels[index * major.width:(index + 1) * major.width]
        if compatibility == "author_code":
            failed = any(not successful_channels.get(ref, False) for ref in lane_refs)
        else:
            failed = not successful_channels.get(lane_channels[index], False)
        if failed:
            broken.add(edge(u, v))
    if not broken:
        selected: tuple[RecoveryReservation, ...] = ()
        route = major.path
    else:
        selected, repaired = select_recovery_loops(
            major, recoveries, broken, compatibility=compatibility,
        )
        # OnlineAlgorithm.P4 does not abort here.  If a candidate cannot cover
        # every broken edge it folds the candidates found so far into the
        # original major path and lets the subsequent link/swap phase decide.
        route_edges = path_edges(major.path)
        for recovery in selected:
            route_edges = xor_edges(route_edges, recovery_loop_edges(major.path, recovery))
        route = shortest_path_from_edges(route_edges, major.path[0], major.path[-1])
        if route is None:
            return LaneOutcome(False, None, frozenset(broken), selected)
    # Source P4 chooses the lowest-ID successful unswapped link at each node;
    # it is not lane-locked.  Consume one such channel per route edge.
    refs_by_edge: dict[Edge, list[ChannelRef]] = {}
    for index, (u, v) in enumerate(major_edges):
        refs = major.channels[index * major.width:(index + 1) * major.width]
        refs_by_edge.setdefault(edge(u, v), []).extend(refs)
    for recovery in selected:
        for index, (u, v) in enumerate(zip(recovery.path, recovery.path[1:])):
            refs = recovery.channels[index * recovery.width:(index + 1) * recovery.width]
            refs_by_edge.setdefault(edge(u, v), []).extend(refs)
    chosen: list[ChannelRef] = []
    for u, v in zip(route, route[1:]):
        options = sorted(refs_by_edge.get(edge(u, v), ()), key=lambda ref: ref.id)
        channel = next((ref for ref in options
                        if ref not in used_channels and successful_channels.get(ref, False)), None)
        if channel is None:
            return LaneOutcome(False, route, frozenset(broken), selected)
        chosen.append(channel)
        used_channels.add(channel)
    if len(chosen) != len(route) - 1:
        return LaneOutcome(False, route, frozenset(broken), selected)
    for _ in range(max(0, len(route) - 2)):
        if rng.random() > swap_probability:
            return LaneOutcome(False, route, frozenset(broken), selected)
    return LaneOutcome(True, route, frozenset(broken), selected)
