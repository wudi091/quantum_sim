"""Fixed-width EDA and greedy G-EDA allocation for Q-CAST P2."""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable, Sequence

from .ext import expected_throughput
from .model import (
    MajorReservation, PathCandidate, QCastTopology, RecoveryReservation,
    ResidualResources, SDPair,
)


def _as_pair(value: SDPair | Sequence[int]) -> SDPair:
    if isinstance(value, SDPair):
        return value
    values = tuple(value)
    if len(values) != 2:
        raise ValueError("S-D pair must contain exactly two node IDs")
    return SDPair(int(values[0]), int(values[1]))


def _edge_probabilities(residual: ResidualResources, path: Sequence[int], width: int) -> tuple[float, ...]:
    """Use the first ``width`` globally ordered channels on each edge.

    The author simulator gives all channels on an edge the same distance-
    derived probability.  For intentionally heterogeneous per-channel input,
    taking the arithmetic mean preserves the expected one-channel rate while
    keeping the fixed-width recurrence well-defined.
    """

    values: list[float] = []
    for u, v in zip(path, path[1:]):
        refs = residual.channels_on_edge((u, v))
        if len(refs) < width:
            raise ValueError("path does not have enough residual channels")
        values.append(sum(residual.channels[ref].probability for ref in refs[:width]) / width)
    return tuple(values)


def _eligible_nodes(residual: ResidualResources, source: int, destination: int, width: int) -> set[int]:
    if source not in residual.node_remaining or destination not in residual.node_remaining:
        return set()
    if residual.node_remaining[source] < width or residual.node_remaining[destination] < width:
        return set()
    return {
        node for node, remaining in residual.node_remaining.items()
        if node in (source, destination) or remaining >= 2 * width
    }


def eda_fixed_width(
    residual: ResidualResources,
    source: int,
    destination: int,
    width: int,
    *,
    swap_probability: float | None = None,
    max_hops: int | None = None,
) -> PathCandidate | None:
    """Run the author's Extended Dijkstra at one fixed width.

    EDA maintains one best complete path per node and pops the highest EXT
    first.  Width is *not* changed during this search; callers that implement
    Q-CAST's width policy should use :func:`width_first_path`.
    """

    source, destination, width = int(source), int(destination), int(width)
    if source == destination or width < 1:
        return None
    eligible = _eligible_nodes(residual, source, destination, width)
    if source not in eligible or destination not in eligible:
        return None
    q_swap = residual.swap_probability if swap_probability is None else float(swap_probability)
    best: dict[int, tuple[float, tuple[int, ...]]] = {source: (math.inf, (source,))}
    heap: list[tuple[float, int, tuple[int, ...]]] = [(-math.inf, source, (source,))]
    while heap:
        neg_score, node, path = heapq.heappop(heap)
        score = -neg_score
        known = best.get(node)
        if known is None or score < known[0] - 1e-15 or path != known[1]:
            continue
        if node == destination:
            # A zero-EXT path is not considered by calCandidates/G-EDA.
            return None if score <= 0.0 else PathCandidate(path, width, score)
        if max_hops is not None and len(path) - 1 >= max_hops:
            continue
        neighbours = residual.neighbours(node, width, eligible)
        for neighbour in neighbours:
            if neighbour in path:
                continue
            candidate_path = path + (neighbour,)
            probabilities = _edge_probabilities(residual, candidate_path, width)
            candidate_score = expected_throughput(probabilities, width, q_swap)
            old = best.get(neighbour)
            # Java's PriorityQueue only replaces a node on strictly greater E.
            # The path tuple is a deterministic tie-break for reproducibility.
            if old is None or candidate_score > old[0] + 1e-15:
                best[neighbour] = (candidate_score, candidate_path)
                heapq.heappush(heap, (-candidate_score, neighbour, candidate_path))
    return None


def max_width(residual: ResidualResources, source: int, destination: int) -> int:
    """Author-code upper bound before descending fixed-width EDA."""

    if source not in residual.node_remaining or destination not in residual.node_remaining:
        return 0
    return min(residual.node_remaining[source], residual.node_remaining[destination])


def width_first_path(
    residual: ResidualResources,
    source: int,
    destination: int,
    *,
    swap_probability: float | None = None,
    max_hops: int | None = None,
) -> PathCandidate | None:
    """Choose the first reachable width while scanning ``maxW ... 1``.

    This is intentionally not ``argmax EXT`` across widths: the Kotlin source
    breaks immediately after the first width with a reachable EDA path.
    """

    for width in range(max_width(residual, source, destination), 0, -1):
        candidate = eda_fixed_width(
            residual, source, destination, width,
            swap_probability=swap_probability, max_hops=max_hops,
        )
        if candidate is not None:
            return candidate
    return None


def geda_allocate(
    topology_or_residual: QCastTopology | ResidualResources,
    sd_pairs: Iterable[SDPair | Sequence[int]],
    *,
    path_cap: int = 200,
    swap_probability: float | None = None,
    max_hops: int | None = None,
    mutate: bool | None = None,
) -> tuple[MajorReservation, ...]:
    """Greedy G-EDA over all concurrent S-D pairs and residual resources."""

    if path_cap < 1:
        return ()
    if isinstance(topology_or_residual, QCastTopology):
        residual = topology_or_residual.residual()
    else:
        # A ResidualResources object is the live G-EDA state in the author
        # simulator, so consume it by default.  ``mutate=False`` is available
        # to callers that need a what-if allocation.
        residual = topology_or_residual if mutate is not False else topology_or_residual.copy()
    pairs = tuple(_as_pair(pair) for pair in sd_pairs)
    result: list[MajorReservation] = []
    while len(result) < int(path_cap):
        candidates: list[tuple[float, int, PathCandidate, SDPair]] = []
        for index, pair in enumerate(pairs):
            candidate = width_first_path(
                residual, pair.source, pair.destination,
                swap_probability=swap_probability, max_hops=max_hops,
            )
            if candidate is not None:
                candidates.append((candidate.expected_throughput, index, candidate, pair))
        if not candidates:
            break
        # max EXT, then input pair order and lexicographic path for stable ties.
        _, _, candidate, pair = max(
            candidates,
            key=lambda item: (item[0], -item[1], tuple(-node for node in item[2].path)),
        )
        channels = residual.reserve(candidate.path, candidate.width)
        result.append(MajorReservation(
            pair, candidate.path, candidate.width,
            candidate.expected_throughput, channels,
        ))
    return tuple(result)


def allocate_recovery_paths(
    residual: ResidualResources,
    major: MajorReservation,
    link_state_range: int,
    *,
    swap_probability: float | None = None,
    max_per_segment: int = 1,
) -> tuple[RecoveryReservation, ...]:
    """Reserve P2Extra recovery paths in the source-code loop order.

    The official implementation's small recovery constant is effectively one
    path per switch-node segment: after each successful EDA call it updates
    residual resources before trying the next segment.
    """

    if link_state_range < 1 or max_per_segment < 1:
        return ()
    path = major.path
    result: list[RecoveryReservation] = []
    max_range = min(int(link_state_range), len(path) - 1)
    # Source order is l=1..k, then each i, matching OnlineAlgorithm.P2Extra.
    for distance in range(1, max_range + 1):
        for start in range(0, len(path) - distance - 0):
            end = start + distance
            # The source code asks for one candidate and only keeps it when E>0.
            for _ in range(max_per_segment):
                candidate = width_first_path(
                    residual, path[start], path[end],
                    swap_probability=swap_probability,
                )
                if candidate is None:
                    break
                channels = residual.reserve(candidate.path, candidate.width)
                result.append(RecoveryReservation(
                    major, candidate.path, candidate.width,
                    candidate.expected_throughput, channels,
                    start, end,
                ))
                # With max_per_segment > 1 this intentionally searches for a
                # second disjoint path; default 1 is author-compatible.
    return tuple(result)
