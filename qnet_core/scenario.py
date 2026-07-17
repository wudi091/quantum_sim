"""Deterministic Waxman workloads shared by training and baselines."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from .spec import EpisodeSpec, PhysicalConfig, RequestSpec


@dataclass(frozen=True)
class ScenarioConfig:
    request_count: int = 4
    min_hops: int = 2
    max_hops: int = 5
    ttl: int = 20
    horizon: int = 100
    arrival_rate: float = 1.0
    physical: PhysicalConfig = PhysicalConfig()
    topology_nodes: int | None = None
    waxman_alpha: float = 0.05
    waxman_beta: float = 0.02
    topology_attempts: int = 128
    demand_pairs: int = 1


def _add_euclidean_mst(graph: nx.Graph) -> None:
    """Add a deterministic Euclidean MST over Waxman's seeded positions."""
    nodes = list(graph.nodes)
    if len(nodes) < 2:
        return
    positions = nx.get_node_attributes(graph, "pos")
    points = np.asarray([positions[node] for node in nodes], dtype=np.float64)
    in_tree = np.zeros(len(nodes), dtype=bool)
    best = np.full(len(nodes), np.inf, dtype=np.float64)
    parent = np.full(len(nodes), -1, dtype=np.int64)
    best[0] = 0.0
    for _ in nodes:
        available = np.where(in_tree, np.inf, best)
        current = int(np.argmin(available))
        in_tree[current] = True
        if parent[current] >= 0:
            graph.add_edge(nodes[int(parent[current])], nodes[current])
        delta = points - points[current]
        distances = np.einsum("ij,ij->i", delta, delta)
        improve = (~in_tree) & (distances < best)
        best[improve] = distances[improve]
        parent[improve] = current


def _make_waxman_graph(config: ScenarioConfig, seed: int) -> tuple[nx.Graph, dict[int, dict[int, int]]]:
    node_count = config.topology_nodes or max(4 * config.max_hops, 16)
    if node_count <= config.max_hops:
        raise ValueError("Waxman topology needs more nodes than max_hops")
    topology_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5741584D]))
    for _ in range(config.topology_attempts):
        graph_seed = int(topology_rng.integers(0, np.iinfo(np.int32).max))
        graph = nx.waxman_graph(
            node_count,
            beta=config.waxman_beta,
            alpha=config.waxman_alpha,
            seed=graph_seed,
        )
        _add_euclidean_mst(graph)
        distances = {
            int(source): {int(target): int(distance) for target, distance in targets.items()}
            for source, targets in nx.all_pairs_shortest_path_length(
                graph, cutoff=config.max_hops
            )
        }
        diameter_reached = any(
            distance >= config.max_hops
            for targets in distances.values()
            for distance in targets.values()
        )
        if diameter_reached:
            return graph, distances
    raise RuntimeError(
        "could not generate a connected Waxman topology with diameter "
        f">= {config.max_hops} after {config.topology_attempts} attempts"
    )


def _hop_targets(config: ScenarioConfig) -> list[int]:
    if config.request_count == 1:
        return [config.max_hops]
    span = config.max_hops - config.min_hops
    return [
        config.min_hops + round(span * index / (config.request_count - 1))
        for index in range(config.request_count)
    ]


def make_episode(config: ScenarioConfig, seed: int) -> EpisodeSpec:
    if config.request_count < 1 or config.min_hops < 1 or config.max_hops < config.min_hops:
        raise ValueError("invalid request or hop configuration")
    if config.arrival_rate <= 0:
        raise ValueError("arrival_rate must be positive")
    if not 0 < config.waxman_alpha or not 0 < config.waxman_beta <= 1:
        raise ValueError("invalid Waxman alpha or beta")
    if config.topology_attempts < 1:
        raise ValueError("topology_attempts must be positive")
    if config.demand_pairs < 1:
        raise ValueError("demand_pairs must be positive")

    graph, distances = _make_waxman_graph(config, seed)
    nodes = tuple(sorted(int(node) for node in graph.nodes))
    edges = tuple(sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges))

    pair_buckets: dict[int, list[tuple[int, int]]] = {
        hop: [] for hop in range(config.min_hops, config.max_hops + 1)
    }
    for source, targets in distances.items():
        for destination, distance in targets.items():
            if source < destination and distance in pair_buckets:
                pair_buckets[distance].append((source, destination))
    missing = [hop for hop, pairs in pair_buckets.items() if not pairs]
    if missing:
        raise RuntimeError(f"Waxman topology is missing shortest-path distances: {missing}")

    request_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x52455153]))
    hops = _hop_targets(config)
    request_rng.shuffle(hops)
    for pairs in pair_buckets.values():
        request_rng.shuffle(pairs)
    bucket_offsets = {hop: 0 for hop in pair_buckets}
    endpoints: list[tuple[int, int]] = []
    for hop in hops:
        pairs = pair_buckets[int(hop)]
        offset = bucket_offsets[int(hop)]
        left, right = pairs[offset % len(pairs)]
        bucket_offsets[int(hop)] = offset + 1
        endpoints.append((left, right) if request_rng.random() < 0.5 else (right, left))

    # A homogeneous Poisson arrival process: exponential inter-arrival times,
    # discretized into physical slots. Multiple requests may arrive together.
    inter_arrivals = request_rng.exponential(1.0 / config.arrival_rate, config.request_count)
    arrivals = np.floor(np.cumsum(inter_arrivals)).astype(int)
    requests = tuple(
        RequestSpec(
            f"r{index}", endpoints[index][0], endpoints[index][1],
            arrival=int(arrivals[index]), ttl=config.ttl,
            demand_pairs=config.demand_pairs,
        )
        for index in range(config.request_count)
    )
    horizon = max(config.horizon, int(arrivals[-1]) + config.ttl)
    return EpisodeSpec(seed, nodes, edges, requests, horizon, config.physical)
