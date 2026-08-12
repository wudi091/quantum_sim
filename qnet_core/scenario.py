"""Deterministic Waxman workloads shared by training and baselines."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from .spec import EpisodeSpec, PhysicalConfig, RequestSpec


@dataclass(frozen=True)
class ScenarioConfig:
    request_count: int = 4
    min_hops: int | None = 2
    max_hops: int | None = 5
    ttl: int = 20
    horizon: int = 100
    physical: PhysicalConfig = PhysicalConfig()
    topology_nodes: int | None = None
    waxman_alpha: float = 0.05
    waxman_beta: float = 0.02
    topology_attempts: int = 128
    waxman_add_mst: bool = True
    endpoint_mode: str = "distance_stratified"
    demand_pairs: int = 1
    topology_mode: str = "waxman"
    parallel_corridors: int = 2
    arrival_batch_size: int | None = None
    arrival_interval: int = 1


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


def _make_waxman_graph(
    config: ScenarioConfig,
    seed: int,
) -> tuple[nx.Graph, dict[int, dict[int, int]]]:
    configured_max_hops = config.max_hops or 1
    node_count = config.topology_nodes or max(4 * configured_max_hops, 16)
    if node_count < 2:
        raise ValueError("Waxman topology needs at least two nodes")
    if (
        config.endpoint_mode == "distance_stratified"
        and node_count <= configured_max_hops
    ):
        raise ValueError("Waxman topology needs more nodes than max_hops")
    topology_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5741584D]))
    for _ in range(config.topology_attempts):
        graph_seed = int(
            topology_rng.integers(0, np.iinfo(np.int32).max)
        )
        graph = nx.waxman_graph(
            node_count,
            beta=config.waxman_beta,
            alpha=config.waxman_alpha,
            seed=graph_seed,
        )
        if config.waxman_add_mst:
            _add_euclidean_mst(graph)
        if not nx.is_connected(graph):
            continue
        if config.endpoint_mode == "uniform_random":
            return graph, {}
        distances = {
            int(source): {
                int(target): int(distance)
                for target, distance in targets.items()
            }
            for source, targets in nx.all_pairs_shortest_path_length(
                graph, cutoff=configured_max_hops
            )
        }
        pair_buckets = _candidate_pair_buckets(distances, config)
        if all(pair_buckets[hop] for hop in set(_hop_targets(config))):
            return graph, distances
    raise RuntimeError(
        "could not generate a connected Waxman topology satisfying the "
        f"endpoint contract with {node_count} nodes after "
        f"{config.topology_attempts} attempts"
    )


def _make_parallel_corridor_graph(
    config: ScenarioConfig,
) -> tuple[nx.Graph, dict[int, dict[int, int]]]:
    """Build equal-length disjoint corridors between source 0 and sink 1."""

    if (
        config.min_hops is None
        or config.max_hops is None
        or config.min_hops != config.max_hops
        or config.max_hops < 2
    ):
        raise ValueError(
            "parallel_corridors requires min_hops == max_hops >= 2"
        )
    if config.parallel_corridors < 2:
        raise ValueError("parallel_corridors must be at least 2")
    graph = nx.Graph()
    graph.add_nodes_from((0, 1))
    next_node = 2
    for _ in range(config.parallel_corridors):
        route = [0]
        route.extend(range(next_node, next_node + config.max_hops - 1))
        next_node += config.max_hops - 1
        route.append(1)
        graph.add_edges_from(zip(route, route[1:]))
    distances = {
        int(source): {
            int(target): int(distance)
            for target, distance in targets.items()
        }
        for source, targets in nx.all_pairs_shortest_path_length(graph)
    }
    return graph, distances


def _hop_targets(config: ScenarioConfig) -> list[int]:
    if config.min_hops is None or config.max_hops is None:
        raise ValueError("distance-stratified endpoints require hop bounds")
    if config.request_count == 1:
        return [config.max_hops]
    span = config.max_hops - config.min_hops
    return [
        config.min_hops + round(span * index / (config.request_count - 1))
        for index in range(config.request_count)
    ]


def _candidate_pair_buckets(
    distances: dict[int, dict[int, int]],
    config: ScenarioConfig,
) -> dict[int, list[tuple[int, int]]]:
    buckets: dict[int, list[tuple[int, int]]] = {
        hop: [] for hop in set(_hop_targets(config))
    }
    for source, targets in distances.items():
        for destination, distance in targets.items():
            if source >= destination or distance not in buckets:
                continue
            buckets[distance].append((source, destination))
    return buckets


def make_episode(config: ScenarioConfig, seed: int) -> EpisodeSpec:
    if config.request_count < 1:
        raise ValueError("request_count must be positive")
    if config.endpoint_mode == "distance_stratified" and (
        config.min_hops is None
        or config.max_hops is None
        or config.min_hops < 1
        or config.max_hops < config.min_hops
    ):
        raise ValueError("invalid hop configuration")
    if not 0 < config.waxman_alpha <= 1 or not 0 < config.waxman_beta <= 1:
        raise ValueError("invalid Waxman alpha or beta")
    if config.topology_attempts < 1:
        raise ValueError("topology_attempts must be positive")
    if config.demand_pairs < 1:
        raise ValueError("demand_pairs must be positive")
    if config.arrival_batch_size is not None and config.arrival_batch_size < 1:
        raise ValueError("arrival_batch_size must be positive when set")
    if config.arrival_interval < 1:
        raise ValueError("arrival_interval must be positive")
    if config.topology_mode not in {"waxman", "parallel_corridors"}:
        raise ValueError(f"unknown topology_mode: {config.topology_mode}")
    if config.endpoint_mode not in {"distance_stratified", "uniform_random"}:
        raise ValueError(f"unknown endpoint_mode: {config.endpoint_mode}")

    if config.topology_mode == "parallel_corridors":
        graph, distances = _make_parallel_corridor_graph(config)
    else:
        graph, distances = _make_waxman_graph(config, seed)
    nodes = tuple(sorted(int(node) for node in graph.nodes))
    edges = tuple(sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges))

    request_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x52455153]))
    endpoints: list[tuple[int, int]] = []
    if config.topology_mode == "parallel_corridors":
        endpoints = [(0, 1) for _ in range(config.request_count)]
    elif config.endpoint_mode == "uniform_random":
        endpoints = [
            tuple(int(node) for node in request_rng.choice(
                nodes, size=2, replace=False
            ))
            for _ in range(config.request_count)
        ]
    else:
        pair_buckets = _candidate_pair_buckets(distances, config)
        missing = [
            hop for hop in set(_hop_targets(config))
            if not pair_buckets[hop]
        ]
        if missing:
            raise RuntimeError(
                f"Waxman topology is missing shortest-path distances: {missing}"
            )
        hops = _hop_targets(config)
        request_rng.shuffle(hops)
        for pairs in pair_buckets.values():
            request_rng.shuffle(pairs)
        bucket_offsets = {hop: 0 for hop in pair_buckets}
        for hop in hops:
            pairs = pair_buckets[int(hop)]
            offset = bucket_offsets[int(hop)]
            left, right = pairs[offset % len(pairs)]
            bucket_offsets[int(hop)] = offset + 1
            endpoints.append(
                (left, right) if request_rng.random() < 0.5 else (right, left)
            )

    arrivals = (
        [0] * config.request_count
        if config.arrival_batch_size is None
        else [
            (index // config.arrival_batch_size) * config.arrival_interval
            for index in range(config.request_count)
        ]
    )
    if arrivals[-1] > config.horizon:
        raise ValueError(
            "episode horizon cannot precede the final request arrival"
        )
    if arrivals[-1] + config.ttl > config.horizon:
        raise ValueError(
            "episode horizon must cover the final request arrival's TTL"
        )
    requests = tuple(
        RequestSpec(
            f"r{index}", endpoints[index][0], endpoints[index][1],
            arrival=arrivals[index], ttl=config.ttl,
            demand_pairs=config.demand_pairs,
        )
        for index in range(config.request_count)
    )
    return EpisodeSpec(
        seed,
        nodes,
        edges,
        requests,
        config.horizon,
        config.physical,
    )
