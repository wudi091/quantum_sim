"""Author-compatible random topology generator from ``Topo.generate``.

The Kotlin source uses a 100 x 100 square, a minimum point spacing, a local
Waxman-like radius (``2 * 100/sqrt(n)``), and a dynamic search for beta that
hits the requested average degree.  This module mirrors those steps while
keeping Python's RNG injectable for reproducible experiments.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping

from .model import EdgeSpec, QCastTopology


@dataclass(frozen=True)
class AuthorTopologyConfig:
    node_count: int = 100
    average_degree: int = 6
    target_link_probability: float = 0.6
    swap_probability: float = 0.9
    link_state_range: int = 3
    seed: int = 19900111
    side_length: float = 100.0
    alpha_precision: float = 0.001


@dataclass(frozen=True)
class AuthorTopologyResult:
    topology: QCastTopology
    positions: Mapping[int, tuple[float, float]]
    beta: float
    alpha: float


def calibrate_alpha(lengths, target: float, *, precision: float = 0.001) -> float:
    """Match mean ``exp(-alpha * distance)`` to target by author dynSearch."""

    distances = tuple(float(value) for value in lengths)
    if not distances:
        raise ValueError("at least one link length is required")
    if not 0.0 < target <= 1.0:
        raise ValueError("target link probability must be in (0, 1]")
    lo, hi = 1e-10, 1.0
    x = (lo + hi) / 2.0
    step = x
    for _ in range(100):
        step /= 2.0
        value = sum(math.exp(-x * distance) for distance in distances) / len(distances)
        if abs(value - target) < abs(precision):
            break
        # f(alpha) decreases, exactly as utils.dynSearch(..., false).
        if value > target:
            x += step
        else:
            x -= step
    return x


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _generate_raw_edges(positions, beta: float, radius: float, rng: random.Random):
    edges: list[tuple[int, int, float]] = []
    n = len(positions)
    for left in range(n):
        for right in range(left + 1, n):
            distance = _distance(positions[left], positions[right])
            if distance >= radius:
                continue
            threshold = math.exp(-beta * distance)
            # Kotlin uses min(50 random values), not one random value.
            draw = min(rng.random() for _ in range(50))
            if draw < threshold:
                edges.append((left, right, distance))
    return edges


def _components(n: int, edges) -> list[list[int]]:
    parent = list(range(n))
    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    for left, right, _ in edges:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a
    groups: dict[int, list[int]] = {}
    for node in range(n):
        groups.setdefault(find(node), []).append(node)
    return sorted(groups.values(), key=lambda group: -len(group))


def generate_author_topology_with_metadata(
    config: AuthorTopologyConfig = AuthorTopologyConfig(),
    rng: random.Random | None = None,
) -> AuthorTopologyResult:
    if config.node_count < 2 or config.average_degree < 1:
        raise ValueError("node_count and average_degree must be positive")
    rng = random.Random(config.seed) if rng is None else rng
    n = config.node_count
    controlling_distance = config.side_length / math.sqrt(n)
    positions: list[tuple[float, float]] = []
    while len(positions) < n:
        candidate = (rng.random() * config.side_length, rng.random() * config.side_length)
        if all(_distance(existing, candidate) > controlling_distance / 1.2 for existing in positions):
            positions.append(candidate)
    # Kotlin's peculiar y-bucket ordering is observable in endpoint choices.
    positions.sort(key=lambda point: point[0] + int(point[1] * 10 / config.side_length) * 1_000_000)
    radius = 2.0 * controlling_distance
    lo, hi = 0.0, 20.0
    beta = (lo + hi) / 2.0
    step = beta
    edges = []
    for _ in range(100):
        step /= 2.0
        edges = _generate_raw_edges(positions, beta, radius, rng)
        degree = 2.0 * len(edges) / n
        if abs(degree - config.average_degree) < 0.2:
            break
        if degree > config.average_degree:
            beta += step
        else:
            beta -= step
    # Connect components exactly as Topo.generate does.
    components = _components(n, edges)
    biggest = components[0]
    for component in components[1:]:
        for to_connect in rng.sample(component, min(3, len(component))):
            nearest = min(biggest, key=lambda node: _distance(positions[node], positions[to_connect]))
            left, right = sorted((nearest, to_connect))
            edges.append((left, right, _distance(positions[left], positions[right])))
    degree_counts = [0] * n
    for left, right, _ in edges:
        degree_counts[left] += 1
        degree_counts[right] += 1
    # Kotlin iterates a snapshot grouped by node and may append parallel rows.
    for node, degree in enumerate(degree_counts):
        # Kotlin's ``occ.size / 2`` is integer division.  ``occ`` contains
        # one endpoint occurrence per incident link, so the historical code
        # actually tests floor(degree / 2), not degree itself.
        half_degree = degree // 2
        if half_degree < 5:
            count = 6 - half_degree
            nearest = sorted(
                (other for other in range(n) if other != node),
                key=lambda other: _distance(positions[other], positions[node]),
            )[:count][1:]
            for other in nearest:
                left, right = sorted((node, other))
                edges.append((left, right, _distance(positions[left], positions[right])))
    node_qubits = {node: int(rng.random() * 5 + 10) for node in range(n)}
    # Topo.toConfig draws node qubits first and edge widths second.  Alpha is
    # calibrated over physical links, hence each edge length is repeated by
    # its sampled width in the mean success-rate target.
    widths = [int(rng.random() * 5 + 3) for _ in edges]
    weighted_lengths = [distance for (_, _, distance), width in zip(edges, widths) for _ in range(width)]
    alpha = calibrate_alpha(weighted_lengths, config.target_link_probability,
                            precision=config.alpha_precision)
    edge_specs = [
        EdgeSpec(left, right, width, math.exp(-alpha * distance))
        for (left, right, distance), width in zip(edges, widths)
    ]
    topology = QCastTopology(
        node_qubits, edge_specs,
        swap_probability=config.swap_probability,
        link_state_range=config.link_state_range,
    )
    return AuthorTopologyResult(
        topology, {node: point for node, point in enumerate(positions)}, beta, alpha,
    )


def generate_author_topology(
    config: AuthorTopologyConfig = AuthorTopologyConfig(),
    rng: random.Random | None = None,
) -> QCastTopology:
    return generate_author_topology_with_metadata(config, rng).topology
