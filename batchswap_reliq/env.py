"""Centralized BatchSwap environment using RELiQ's physical link model.

The upstream RELiQ and Q-DDCA trees are intentionally left untouched.  This
module composes RELiQ's ``QuantumNetwork``, concrete ``Edge.links`` EPR tokens,
fidelity decay/generation, and ``QuantumLink.swap`` kernel with the sequential
``(request, plan)`` / ``STOP`` action interface used by masked PPO.
"""

from __future__ import annotations

from collections import Counter
import contextlib
import copy
from dataclasses import dataclass, replace
import heapq
import importlib
import io
import itertools
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


EdgeKey = tuple[int, int]


def edge_key(u: int, v: int) -> EdgeKey:
    return (u, v) if u < v else (v, u)


@dataclass(frozen=True)
class RequestSpec:
    id: str
    path: tuple[int, ...]
    arrival: int = 0

    @property
    def hops(self) -> int:
        return len(self.path) - 1

    @property
    def source(self) -> int:
        return self.path[0]

    @property
    def destination(self) -> int:
        return self.path[-1]


@dataclass
class ReliqInstance:
    network: Any
    requests: tuple[RequestSpec, ...]


@dataclass(frozen=True)
class ResourcePath:
    """A path that exists in the current elementary-EPR resource graph."""

    nodes: tuple[int, ...]
    base_links: tuple[Any, ...]

    @property
    def hops(self) -> int:
        return len(self.base_links)


@dataclass(frozen=True)
class CandidatePlan:
    request_id: str
    request_index: int
    plan_slot: int
    kind: str
    start_index: int
    reach_index: int
    base_links: tuple[Any, ...]
    input_links: tuple[Any, ...]
    swap_nodes_by_layer: tuple[tuple[int, ...], ...]
    completed: bool

    @property
    def progress(self) -> int:
        return len(self.base_links)

    @property
    def swap_depth(self) -> int:
        return len(self.swap_nodes_by_layer)

    @property
    def swap_count(self) -> int:
        return max(0, len(self.input_links) - 1)


@dataclass
class EnvConfig:
    max_requests: int = 30
    max_candidates_per_request: int = 3
    max_hops: int = 50
    max_subslots: int = 1000
    # Fixed lifetime in physical subslots from each request's arrival.
    # It is deliberately independent of source-destination hop count.
    request_ttl: int | None = None
    request_count: int = 4
    min_hops: int = 2
    curriculum_max_hops: int = 5
    node_capacity: int = 2
    network_nodes: int = 12
    node_degree: int = 2
    n_quantum_links: int = 2
    attenuation_coefficient: float = 0.2
    refresh_rate: float = 1.0
    timestep_decay: float = 0.995
    swap_probability: float = 1.0
    initial_fidelity: float = 0.99
    auto_distillation_threshold: float = 0.0
    candidate_route_count: int = 8
    balanced_hop_buckets: bool = False
    topology_mode: str = "random"
    seed: int = 0


@dataclass
class RewardConfig:
    flow_time_weight: float = 1.0
    completion_bonus: float = 5.0
    progress_weight: float = 1.0
    makespan_weight: float = 0.05
    elementary_epr_weight: float = 0.01
    swap_weight: float = 0.005
    failure_weight: float = 1.0
    timeout_weight: float = 7.0


CURRICULUM = (
    dict(max_requests=4, request_count=4, min_hops=2, curriculum_max_hops=5,
         network_nodes=12, node_degree=3, n_quantum_links=4, max_subslots=100,
         topology_mode="random", balanced_hop_buckets=False),
    dict(max_requests=10, request_count=8, min_hops=5, curriculum_max_hops=15,
         network_nodes=25, node_degree=2, n_quantum_links=6, max_subslots=500,
         topology_mode="line", balanced_hop_buckets=False),
    dict(max_requests=100, request_count=100, min_hops=20, curriculum_max_hops=50,
         network_nodes=110, node_degree=3, n_quantum_links=4, max_subslots=500,
         topology_mode="ladder", balanced_hop_buckets=True),
)


_RELIQ_MODULE: Any | None = None


def _load_reliq_module() -> Any:
    """Import RELiQ without changing its source or leaking a cwd change."""
    global _RELIQ_MODULE
    if _RELIQ_MODULE is not None:
        return _RELIQ_MODULE
    root = Path(__file__).resolve().parents[1] / "RELiQ"
    source = root / "src"
    old_cwd = Path.cwd()
    inserted = False
    try:
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
            inserted = True
        os.chdir(root)
        _RELIQ_MODULE = importlib.import_module("env.quantum_network")
    finally:
        os.chdir(old_cwd)
        if inserted:
            try:
                sys.path.remove(str(source))
            except ValueError:
                pass
    return _RELIQ_MODULE


def _make_network(config: EnvConfig, seed: int) -> Any:
    module = _load_reliq_module()
    network = module.QuantumNetwork(
        n_nodes=max(config.network_nodes, config.curriculum_max_hops + 1),
        n_nodes_connected=max(config.network_nodes, config.curriculum_max_hops + 1),
        random_topology=False,
        topology_init_seed=seed,
        provided_seeds=[seed],
        n_quantum_links=config.n_quantum_links,
        attenuation_coefficient=config.attenuation_coefficient,
        timestep_decay=config.timestep_decay,
        swap_probability=config.swap_probability,
        swap_probability_std=0,
        neighbor_count=config.node_degree,
        node_degree=config.node_degree,
        refresh_rate=config.refresh_rate,
        fixed_path_length=-1,
        initial_fidelity=config.initial_fidelity,
        auto_distillation_threshold=config.auto_distillation_threshold,
        episode_steps=config.max_subslots,
        eval_episode_steps=config.max_subslots,
    )
    return network


def _build_line_network(network: Any, config: EnvConfig, seed: int) -> None:
    """Build a deterministic long-hop topology from RELiQ's native classes."""
    module = _load_reliq_module()
    import networkx as nx

    count = max(config.network_nodes, config.curriculum_max_hops + 1)
    network.n_nodes = count
    network.n_nodes_connected = count
    network.set_seeds(seed)
    network.nodes = []
    network.edges = []
    network.G = nx.Graph()
    network.node_distances = {}
    for node_id in range(count):
        node = module.QuantumRepeater(
            node_id,
            node_id * 10.0,
            0.0,
            swap_prob=config.swap_probability,
            decay=network.get_random_decay(),
            n_decoupling_pulses=network.get_n_decoupling_pulses(),
        )
        network.nodes.append(node)
        network.G.add_node(node_id, pos=(node.x, node.y))
    for edge_id in range(count - 1):
        u, v = edge_id, edge_id + 1
        links = [
            module.QuantumLink(u, v, config.initial_fidelity, creation=0)
            for _ in range(config.n_quantum_links)
        ]
        edge = module.Edge(u, v, links)
        network.edges.append(edge)
        network.nodes[u].neighbors.append(v)
        network.nodes[v].neighbors.append(u)
        network.nodes[u].edges.append(edge_id)
        network.nodes[v].edges.append(edge_id)
        network.G.add_edge(u, v, weight=1)
    network.edge_node_association = {
        (edge.start, edge.end): edge for edge in network.edges
    }
    network.total_active_quantum_links = sum(len(edge.links) for edge in network.edges)
    network.min_active_quantum_links = config.n_quantum_links
    network.max_active_quantum_links = network.total_active_quantum_links
    network.average_active_quantum_links = float(network.total_active_quantum_links)
    network.env_steps = 0
    network.last_refresh = 0
    network.done = False
    network.reservation_cleanup = {}
    network.probability_storage = {}
    network.current_topology_seed = seed
    network.diameter = count - 1
    network._update_shortest_paths()
    network._update_nodes_adjacency()


def _build_ladder_network(network: Any, config: EnvConfig, seed: int) -> None:
    """Two long rails with sparse rungs: high diameter plus route diversity."""
    module = _load_reliq_module()
    import networkx as nx

    count = max(config.network_nodes, 2 * (config.curriculum_max_hops + 5))
    if count % 2:
        count += 1
    rail = count // 2
    network.n_nodes = count
    network.n_nodes_connected = count
    network.set_seeds(seed)
    network.nodes = []
    network.edges = []
    network.G = nx.Graph()
    network.node_distances = {}
    for node_id in range(count):
        position = node_id % rail
        row = node_id // rail
        node = module.QuantumRepeater(
            node_id,
            position * 10.0,
            row * 10.0,
            swap_prob=config.swap_probability,
            decay=network.get_random_decay(),
            n_decoupling_pulses=network.get_n_decoupling_pulses(),
        )
        network.nodes.append(node)
        network.G.add_node(node_id, pos=(node.x, node.y))

    edge_pairs: list[tuple[int, int]] = []
    for row in range(2):
        offset = row * rail
        edge_pairs.extend((offset + i, offset + i + 1) for i in range(rail - 1))
    rung_positions = set(range(0, rail, 10)) | {rail - 1}
    edge_pairs.extend((position, rail + position) for position in sorted(rung_positions))

    for edge_id, (u, v) in enumerate(edge_pairs):
        links = [
            module.QuantumLink(min(u, v), max(u, v), config.initial_fidelity, creation=0)
            for _ in range(config.n_quantum_links)
        ]
        edge = module.Edge(min(u, v), max(u, v), links)
        network.edges.append(edge)
        network.nodes[u].neighbors.append(v)
        network.nodes[v].neighbors.append(u)
        network.nodes[u].edges.append(edge_id)
        network.nodes[v].edges.append(edge_id)
        network.G.add_edge(u, v, weight=1)
    network.edge_node_association = {
        (edge.start, edge.end): edge for edge in network.edges
    }
    network.total_active_quantum_links = sum(len(edge.links) for edge in network.edges)
    network.min_active_quantum_links = config.n_quantum_links
    network.max_active_quantum_links = network.total_active_quantum_links
    network.average_active_quantum_links = float(network.total_active_quantum_links)
    network.env_steps = 0
    network.last_refresh = 0
    network.done = False
    network.reservation_cleanup = {}
    network.probability_storage = {}
    network.current_topology_seed = seed
    network.diameter = nx.diameter(network.G)
    network._update_shortest_paths()
    network._update_nodes_adjacency()


def _sample_requests(network: Any, config: EnvConfig, seed: int) -> tuple[RequestSpec, ...]:
    import networkx as nx

    rng = np.random.default_rng(seed + 17_003)
    nodes = list(network.G.nodes)
    candidates: list[tuple[int, int, tuple[int, ...]]] = []
    for source in nodes:
        for destination in nodes:
            if source == destination:
                continue
            try:
                path = tuple(nx.shortest_path(network.G, source, destination))
            except nx.NetworkXNoPath:
                continue
            hops = len(path) - 1
            if config.min_hops <= hops <= config.curriculum_max_hops:
                candidates.append((source, destination, path))
    if not candidates:
        raise RuntimeError(
            "RELiQ topology has no request path inside the curriculum hop range"
        )
    requests = []
    if config.balanced_hop_buckets:
        span = config.curriculum_max_hops - config.min_hops + 1
        width = max(1, span // 3)
        buckets = (
            (config.min_hops, config.min_hops + width - 1),
            (config.min_hops + width, config.min_hops + 2 * width - 1),
            (config.min_hops + 2 * width, config.curriculum_max_hops),
        )
        quotient, remainder = divmod(config.request_count, len(buckets))
        index = 0
        for bucket_index, (lower, upper) in enumerate(buckets):
            bucket = [item for item in candidates if lower <= len(item[2]) - 1 <= upper]
            if not bucket:
                raise RuntimeError(f"no request candidates in hop bucket {lower}--{upper}")
            rng.shuffle(bucket)
            count = quotient + int(bucket_index < remainder)
            for offset in range(count):
                _, _, path = bucket[offset % len(bucket)]
                requests.append(RequestSpec(f"r{index}", path))
                index += 1
        rng.shuffle(requests)
        requests = [replace(request, id=f"r{index}") for index, request in enumerate(requests)]
    else:
        rng.shuffle(candidates)
        for index in range(config.request_count):
            _, _, path = candidates[index % len(candidates)]
            requests.append(RequestSpec(f"r{index}", path))
    return tuple(requests)


class BatchSwapReliqEnv:
    global_feature_dim = 12
    request_feature_dim = 11
    candidate_feature_dim = 20

    def __init__(
        self,
        config: EnvConfig | None = None,
        reward_config: RewardConfig | None = None,
        instance: ReliqInstance | None = None,
    ) -> None:
        self.config = config or EnvConfig()
        self.reward_config = reward_config or RewardConfig()
        self._fixed_instance = instance
        self._network_template: Any | None = None
        self._next_episode_seed = self.config.seed
        self.stop_action = self.config.max_requests * self.config.max_candidates_per_request
        self.action_size = self.stop_action + 1
        self._token_counter = 0
        # These depend only on the physical topology, not the changing EPR
        # inventory, so reuse them across planning epochs and episode resets.
        self._distance_cache: dict[int, dict[int, int]] = {}
        self._topology_path_cache: dict[
            tuple[int, int, tuple[int, ...]], tuple[tuple[int, ...], ...]
        ] = {}

    def set_curriculum(self, stage: int | object) -> None:
        if isinstance(stage, (int, np.integer)):
            index = int(stage)
        else:
            index = {"short": 0, "medium": 1, "long": 2}.get(
                str(getattr(stage, "name", "short")).lower(), 0
            )
        if not 0 <= index < len(CURRICULUM):
            raise ValueError("curriculum stage must be 0, 1, or 2")
        values = CURRICULUM[index].copy()
        if not isinstance(stage, (int, np.integer)):
            capacity = min(
                int(values.get("max_requests", self.config.max_requests)),
                int(getattr(stage, "max_requests")),
            )
            values.update(
                max_requests=capacity,
                request_count=min(int(values["request_count"]), capacity),
                min_hops=int(getattr(stage, "min_hops")),
                curriculum_max_hops=int(getattr(stage, "max_hops")),
            )
        self.config = replace(self.config, **values)
        self.stop_action = self.config.max_requests * self.config.max_candidates_per_request
        self.action_size = self.stop_action + 1
        self._fixed_instance = None
        self._network_template = None
        self._distance_cache.clear()
        self._topology_path_cache.clear()
        self._next_episode_seed = self.config.seed

    def reset(
        self, seed: int | None = None, options: Mapping[str, object] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        del options
        episode_seed = self._next_episode_seed if seed is None else int(seed)
        self._next_episode_seed = episode_seed + 1
        if self._fixed_instance is None:
            if self._network_template is None:
                network = _make_network(self.config, self.config.seed)
                if self.config.topology_mode == "line":
                    _build_line_network(network, self.config, self.config.seed)
                elif self.config.topology_mode == "ladder":
                    _build_ladder_network(network, self.config, self.config.seed)
                else:
                    # RELiQ prints topology diagnostics during every rejected
                    # topology attempt. Keep upstream untouched but avoid
                    # flooding training logs; cache the accepted topology.
                    with contextlib.redirect_stdout(io.StringIO()):
                        network.reset()
                self._network_template = copy.deepcopy(network)
            else:
                network = copy.deepcopy(self._network_template)
            if hasattr(network, "set_seeds"):
                network.set_seeds(episode_seed)
            requests = _sample_requests(network, self.config, episode_seed)
            self.instance = ReliqInstance(network, requests)
        else:
            self.instance = self._fixed_instance
            self.instance.network.reset()
        self.network = self.instance.network
        self.time = int(getattr(self.network, "env_steps", 0))
        # The frontier is a physical repeater node, not a fixed-path index.
        self.frontier = {request.id: request.source for request in self.instance.requests}
        self.carried_links: dict[str, Any | None] = {
            request.id: None for request in self.instance.requests
        }
        self.completed_at: dict[str, int] = {}
        self.expired_at: dict[str, int] = {}
        self.last_service = {request.id: request.arrival for request in self.instance.requests}
        self.elementary_eprs = 0
        self.swaps = 0
        self.failed_plans = 0
        self.planning_slots = 0
        self._normalize_base_tokens()
        self._begin_selection()
        return self.observe(), self._info(duration=0, completed_now=0)

    def _new_token_id(self, prefix: str) -> str:
        value = f"{prefix}:{self._token_counter}"
        self._token_counter += 1
        return value

    def _normalize_link(self, link: Any, *, owner: str | None = None) -> Any:
        if not hasattr(link, "token_id") or link.token_id is None:
            link.token_id = self._new_token_id("epr")
        link.owner_request_id = owner
        if not hasattr(link, "available_time"):
            creation = getattr(link, "creation", self.time)
            link.available_time = self.time if creation is None else int(max(0, creation))
        return link

    def _normalize_base_tokens(self) -> None:
        for edge in self.network.edges:
            for link in edge.links:
                self._normalize_link(link, owner=None)

    def _edge(self, u: int, v: int) -> Any | None:
        return getattr(self.network, "edge_node_association", {}).get(edge_key(u, v))

    def _best_base_link(self, u: int, v: int) -> Any | None:
        edge = self._edge(u, v)
        if edge is None or getattr(edge, "dead", False):
            return None
        available = [
            link for link in edge.links
            if getattr(link, "owner_request_id", None) is None
            and getattr(link, "available_time", 0) <= self.time
            and getattr(link, "fidelity", 0.0) > 0.0
        ]
        return max(available, key=lambda item: (item.fidelity, str(item.token_id)), default=None)

    @staticmethod
    def _shared_node(left: Any, right: Any) -> int:
        shared = {left.start, left.end} & {right.start, right.end}
        if len(shared) != 1:
            raise RuntimeError("swap inputs must share exactly one repeater")
        return next(iter(shared))

    @classmethod
    def _swap_schedule(cls, inputs: Sequence[Any]) -> tuple[tuple[int, ...], ...]:
        segments = list(inputs)
        layers: list[tuple[int, ...]] = []
        while len(segments) > 1:
            nodes: list[int] = []
            next_segments: list[Any] = []
            for index in range(0, len(segments), 2):
                if index + 1 >= len(segments):
                    next_segments.append(segments[index])
                    continue
                left, right = segments[index], segments[index + 1]
                nodes.append(cls._shared_node(left, right))
                # Endpoint-only placeholder for planning the following layer.
                shared = nodes[-1]
                outer = [
                    endpoint for endpoint in (left.start, left.end, right.start, right.end)
                    if endpoint != shared
                ]
                placeholder = type("Segment", (), {})()
                placeholder.start, placeholder.end = outer[0], outer[1]
                next_segments.append(placeholder)
            layers.append(tuple(nodes))
            segments = next_segments
        return tuple(layers)

    def _begin_selection(self) -> None:
        self._normalize_base_tokens()
        self.selected_plans: list[CandidatePlan] = []
        self.selected_requests: set[str] = set()
        self.reserved_token_ids: set[str] = set()
        self.node_load_by_layer: list[Counter[int]] = []
        self.node_load: Counter[int] = Counter()
        self.current_plans: list[CandidatePlan | None] = [None] * self.stop_action
        for request_index, request in enumerate(self.instance.requests):
            if (request.id in self.completed_at or request.id in self.expired_at
                    or request.arrival > self.time):
                continue
            for plan in self._plans_for_request(request, request_index):
                action = request_index * self.config.max_candidates_per_request + plan.plan_slot
                self.current_plans[action] = plan

    def _physical_distances(self, destination: int) -> dict[int, int]:
        cached = self._distance_cache.get(destination)
        if cached is not None:
            return cached
        adjacency: dict[int, set[int]] = {}
        for edge in self.network.edges:
            adjacency.setdefault(edge.start, set()).add(edge.end)
            adjacency.setdefault(edge.end, set()).add(edge.start)
        distances = {destination: 0}
        queue = [destination]
        for node in queue:
            for neighbor in adjacency.get(node, ()):
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        self._distance_cache[destination] = distances
        return distances

    def _remaining_distance(self, request: RequestSpec) -> int:
        return self._physical_distances(request.destination).get(
            self.frontier[request.id], self.config.max_hops
        )

    def _request_ttl(self, request: RequestSpec) -> int | None:
        """Return this request's deadline budget in physical subslots."""
        del request
        if self.config.request_ttl is not None:
            return max(1, int(self.config.request_ttl))
        return None

    def _deadline(self, request: RequestSpec) -> int | None:
        ttl = self._request_ttl(request)
        return None if ttl is None else request.arrival + ttl

    def _expire_unresolved_requests(self) -> int:
        """Expire unresolved requests after completion settlement at this time.

        Finishing exactly at the deadline counts as success.  Timeout is an
        exogenous deadline failure, never an action that a policy can choose.
        """
        expired_now = 0
        for request in self.instance.requests:
            if request.id in self.completed_at or request.id in self.expired_at:
                continue
            deadline = self._deadline(request)
            if deadline is None or self.time < deadline:
                continue
            self.expired_at[request.id] = self.time
            self.carried_links[request.id] = None
            expired_now += 1
        return expired_now

    def _candidate_topology_paths(
        self, start: int, destination: int, forbidden: set[int] | None = None
    ) -> list[tuple[int, ...]]:
        """Return N unweighted shortest simple paths in the physical topology."""
        forbidden = set() if forbidden is None else set(forbidden)
        forbidden.discard(start)
        forbidden.discard(destination)
        cache_key = (start, destination, tuple(sorted(forbidden)))
        cached = self._topology_path_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        limit = max(self.config.candidate_route_count,
                    self.config.max_candidates_per_request)
        try:
            import networkx as nx

            graph = self.network.G.copy()
            graph.remove_nodes_from([node for node in forbidden if node in graph])
            results = [
                tuple(path)
                for path in itertools.islice(
                    nx.shortest_simple_paths(graph, start, destination), limit
                )
            ]
        except (AttributeError, TypeError, ImportError):
            # Deterministic fallback used by the small fake-network tests.
            adjacency: dict[int, set[int]] = {}
            for edge in self.network.edges:
                adjacency.setdefault(edge.start, set()).add(edge.end)
                adjacency.setdefault(edge.end, set()).add(edge.start)
            heap: list[tuple[int, tuple[int, ...]]] = [(0, (start,))]
            results: list[tuple[int, ...]] = []
            while heap and len(results) < limit:
                _, path = heapq.heappop(heap)
                node = path[-1]
                if node == destination:
                    results.append(path)
                    continue
                for neighbor in sorted(adjacency.get(node, ())):
                    if neighbor in path or neighbor in forbidden:
                        continue
                    next_path = path + (neighbor,)
                    heapq.heappush(heap, (len(next_path) - 1, next_path))
        self._topology_path_cache[cache_key] = tuple(results)
        return results

    def _validate_route_prefix(self, route: Sequence[int]) -> ResourcePath | None:
        """Bind concrete EPRs until the first unavailable edge on a route."""
        nodes = [int(route[0])]
        links: list[Any] = []
        for u, v in zip(route, route[1:]):
            link = self._best_base_link(int(u), int(v))
            if link is None:
                break
            links.append(link)
            nodes.append(int(v))
        if not links:
            return None
        return ResourcePath(tuple(nodes), tuple(links))

    def _compile_swap_plan(
        self,
        request: RequestSpec,
        request_index: int,
        plan_slot: int,
        kind: str,
        resource_path: ResourcePath,
    ) -> CandidatePlan:
        """Compile one resource path into a concrete token-level swap plan."""
        carried = self.carried_links[request.id]
        inputs = ((carried,) if carried is not None else ()) + resource_path.base_links
        return CandidatePlan(
            request.id,
            request_index,
            plan_slot,
            kind,
            resource_path.nodes[0],
            resource_path.nodes[-1],
            resource_path.base_links,
            inputs,
            self._swap_schedule(inputs),
            resource_path.nodes[-1] == request.destination,
        )

    def _plans_for_request(
        self, request: RequestSpec, request_index: int
    ) -> tuple[CandidatePlan, ...]:
        start = self.frontier[request.id]
        if start != request.source and self.carried_links[request.id] is None:
            return ()
        topology_paths = self._candidate_topology_paths(
            start,
            request.destination,
            forbidden={request.source} if start != request.source else set(),
        )
        proposals: list[tuple[str, ResourcePath]] = []
        labels = ("max", "half", "short")  # stable feature slots: route ranks 0,1,2
        for route in topology_paths:
            resource_path = self._validate_route_prefix(route)
            if resource_path is None:
                continue
            proposals.append((labels[min(len(proposals), len(labels) - 1)], resource_path))
            if len(proposals) >= self.config.max_candidates_per_request:
                break
        if not proposals:
            return ()
        seen: set[tuple[str, ...]] = set()
        plans: list[CandidatePlan] = []
        for kind, resource_path in proposals:
            signature = tuple(str(link.token_id) for link in resource_path.base_links)
            if signature in seen:
                continue
            seen.add(signature)
            plans.append(self._compile_swap_plan(
                request, request_index, len(plans), kind, resource_path
            ))
            if len(plans) >= self.config.max_candidates_per_request:
                break
        return tuple(plans)

    def decode_action(self, action: int) -> CandidatePlan | None:
        if action == self.stop_action:
            return None
        if not 0 <= action < self.stop_action:
            raise ValueError(f"action {action} outside action space")
        return self.current_plans[action]

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_size, dtype=bool)
        for action, plan in enumerate(self.current_plans):
            if plan is None or plan.request_id in self.selected_requests:
                continue
            ids = {str(link.token_id) for link in plan.input_links}
            if ids & self.reserved_token_ids:
                continue
            legal = True
            for layer, nodes in enumerate(plan.swap_nodes_by_layer):
                counts = Counter(nodes)
                current = self.node_load_by_layer[layer] if layer < len(self.node_load_by_layer) else Counter()
                if any(current[node] + count > self.config.node_capacity
                       for node, count in counts.items()):
                    legal = False
                    break
            if legal:
                mask[action] = True
        # A controller may end a batch after selecting at least one plan. If no
        # plan is executable, STOP remains the wait/time-advance action. This
        # prevents a policy from starving the final pending request by choosing
        # an empty batch forever while service is possible.
        mask[self.stop_action] = bool(self.selected_plans) or not bool(mask[:-1].any())
        return mask

    def step(self, action: int):
        action = int(action)
        mask = self.action_mask()
        if not 0 <= action < self.action_size or not mask[action]:
            raise ValueError(f"invalid or masked action {action}")
        if action != self.stop_action:
            plan = self.current_plans[action]
            assert plan is not None
            self.selected_plans.append(plan)
            self.selected_requests.add(plan.request_id)
            self.reserved_token_ids.update(str(link.token_id) for link in plan.input_links)
            while len(self.node_load_by_layer) < plan.swap_depth:
                self.node_load_by_layer.append(Counter())
            for layer, nodes in enumerate(plan.swap_nodes_by_layer):
                self.node_load_by_layer[layer].update(nodes)
                self.node_load.update(nodes)
            info = self._info(duration=0, completed_now=0)
            info.update(phase="select", selected_action=action)
            return self.observe(), 0.0, False, False, info
        return self._execute_batch()

    def _validate_batch(self) -> None:
        seen: set[str] = set()
        for plan in self.selected_plans:
            for link in plan.input_links:
                token_id = str(link.token_id)
                if token_id in seen:
                    raise RuntimeError("batch attempts to consume an EPR token twice")
                seen.add(token_id)
            carried = self.carried_links[plan.request_id]
            request = self.instance.requests[plan.request_index]
            expected = plan.input_links[0] if plan.start_index != request.source else None
            if carried is not expected:
                raise RuntimeError("request-owned frontier EPR changed before commit")
            for link in plan.base_links:
                edge = self._edge(link.start, link.end)
                if edge is None or link not in edge.links:
                    raise RuntimeError("reserved elementary EPR disappeared before commit")

    def _consume_base_links(self) -> None:
        for plan in self.selected_plans:
            for link in plan.base_links:
                edge = self._edge(link.start, link.end)
                edge.links.remove(link)
                if hasattr(self.network, "total_active_quantum_links"):
                    self.network.total_active_quantum_links -= 1

    def _swap_pair(self, left: Any, right: Any) -> Any | None:
        node = self._shared_node(left, right)
        probability = float(getattr(self.network.nodes[node], "swap_prob", 1.0))
        generator = getattr(self.network, "swap_generator", None)
        if generator is None:
            generator = getattr(self.network, "quantum_generator", np.random.default_rng(0))
        output = type(left).swap(left, right, probability, node, generator)
        self.swaps += 1
        threshold = float(getattr(type(left), "FIDELITY_THRESHOLD", 0.5))
        if output is None or float(getattr(output, "fidelity", 0.0)) <= threshold:
            return None
        return output

    def _advance_subslot(self) -> None:
        self.network.pre_step()
        self.network.step()
        self.time = int(self.network.env_steps)
        self._normalize_base_tokens()

    @staticmethod
    def _fidelity_threshold(link: Any) -> float:
        return float(getattr(type(link), "FIDELITY_THRESHOLD", 0.5))

    def _decay_private_links(self, states: Mapping[str, Sequence[Any | None]]) -> None:
        """Decay request-owned and in-flight links omitted from Edge.links."""
        links: dict[int, Any] = {}
        selected = set(states)
        for request_id, link in self.carried_links.items():
            if request_id not in selected and link is not None:
                links[id(link)] = link
        for segments in states.values():
            for link in segments:
                if link is not None:
                    links[id(link)] = link
        for link in links.values():
            if hasattr(self.network, "calculate_decay"):
                factor = float(self.network.calculate_decay(link.start, link.end))
                if hasattr(link, "decay"):
                    link.decay(factor)
                else:
                    link.fidelity *= factor
        for request_id, link in list(self.carried_links.items()):
            if (link is not None
                    and float(getattr(link, "fidelity", 0.0)) <= self._fidelity_threshold(link)):
                self.carried_links[request_id] = None
                request = next(item for item in self.instance.requests if item.id == request_id)
                self.frontier[request_id] = request.source

    def _execute_batch(self):
        self._validate_batch()
        batch_start_time = self.time
        active_before = len(self._active_requests())
        old_remaining = sum(
            self._remaining_distance(request)
            for request in self._active_requests()
        )
        self._consume_base_links()
        for plan in self.selected_plans:
            self.carried_links[plan.request_id] = None

        duration = max(1, max((plan.swap_depth for plan in self.selected_plans), default=0))
        states: dict[str, list[Any | None]] = {
            plan.request_id: list(plan.input_links) for plan in self.selected_plans
        }
        failed: set[str] = set()
        swaps_before = self.swaps
        for layer in range(duration):
            for plan in self.selected_plans:
                if layer >= plan.swap_depth or plan.request_id in failed:
                    continue
                segments = states[plan.request_id]
                next_segments: list[Any | None] = []
                for index in range(0, len(segments), 2):
                    if index + 1 >= len(segments):
                        next_segments.append(segments[index])
                        continue
                    left, right = segments[index], segments[index + 1]
                    if left is None or right is None:
                        output = None
                    else:
                        output = self._swap_pair(left, right)
                    next_segments.append(output)
                states[plan.request_id] = next_segments
                if any(segment is None for segment in next_segments):
                    failed.add(plan.request_id)
            self._decay_private_links(states)
            self._advance_subslot()

        completed_now = 0
        for plan in self.selected_plans:
            request = self.instance.requests[plan.request_index]
            plan_finish_time = batch_start_time + max(1, plan.swap_depth)
            output = states[plan.request_id][0] if states[plan.request_id] else None
            if plan.request_id in failed or output is None:
                self.failed_plans += 1
                self.frontier[plan.request_id] = request.source
                self.carried_links[plan.request_id] = None
                continue
            self.last_service[plan.request_id] = plan_finish_time
            deadline = self._deadline(request)
            if plan.completed and (deadline is None or plan_finish_time <= deadline):
                self.frontier[plan.request_id] = plan.reach_index
                self.completed_at[plan.request_id] = plan_finish_time
                completed_now += 1
            else:
                output.start = min(request.source, plan.reach_index)
                output.end = max(request.source, plan.reach_index)
                self._normalize_link(output, owner=plan.request_id)
                output.token_id = self._new_token_id(f"owned:{plan.request_id}")
                output.available_time = self.time
                self.frontier[plan.request_id] = plan.reach_index
                self.carried_links[plan.request_id] = output

        # Measure progress before timeouts are removed from the active set;
        # otherwise expiration would look like successful forward progress.
        new_remaining = sum(
            self._remaining_distance(request)
            for request in self._active_requests()
        )
        expired_now = self._expire_unresolved_requests()
        elementary_now = sum(len(plan.base_links) for plan in self.selected_plans)
        swaps_now = self.swaps - swaps_before
        self.elementary_eprs += elementary_now
        self.planning_slots += 1
        failures_now = len(failed)
        reward = -self.reward_config.flow_time_weight * active_before * duration / max(self.config.max_requests, 1)
        reward -= self.reward_config.makespan_weight * duration
        reward -= self.reward_config.elementary_epr_weight * elementary_now / max(self.config.max_hops, 1)
        reward -= self.reward_config.swap_weight * swaps_now / max(self.config.max_hops, 1)
        reward -= self.reward_config.failure_weight * failures_now
        reward -= self.reward_config.timeout_weight * expired_now
        reward += self.reward_config.completion_bonus * completed_now
        reward += self.reward_config.progress_weight * (old_remaining - new_remaining) / max(self.config.max_hops, 1)

        terminated = (
            len(self.completed_at) + len(self.expired_at) == len(self.instance.requests)
        )
        truncated = not terminated and self.time >= self.config.max_subslots
        if not terminated and not truncated:
            self._begin_selection()
        else:
            self.selected_plans = []
            self.selected_requests = set()
            self.reserved_token_ids = set()
            self.node_load_by_layer = []
            self.node_load = Counter()
            self.current_plans = [None] * self.stop_action
        info = self._info(duration=duration, completed_now=completed_now)
        info.update(
            phase="execute", elementary_now=elementary_now, swaps_now=swaps_now,
            failed_now=failures_now, expired_now=expired_now,
        )
        return self.observe(), float(reward), terminated, truncated, info

    def _active_requests(self) -> list[RequestSpec]:
        return [
            request for request in self.instance.requests
            if (request.arrival <= self.time
                and request.id not in self.completed_at
                and request.id not in self.expired_at)
        ]

    def observe(self) -> dict[str, np.ndarray]:
        self._normalize_base_tokens()
        request_features = np.zeros(
            (self.config.max_requests, self.request_feature_dim), dtype=np.float32
        )
        request_mask = np.zeros(self.config.max_requests, dtype=bool)
        for index, request in enumerate(self.instance.requests):
            if (request.id in self.completed_at or request.id in self.expired_at
                    or request.arrival > self.time):
                continue
            request_mask[index] = True
            frontier = self.frontier[request.id]
            carried = self.carried_links[request.id]
            remaining = self._remaining_distance(request)
            request_plans = [
                plan for plan in self.current_plans
                if plan is not None and plan.request_id == request.id
            ]
            contiguous = max((plan.progress for plan in request_plans), default=0)
            fidelities = [
                float(link.fidelity) for plan in request_plans for link in plan.base_links
            ]
            ttl = self._request_ttl(request)
            age_scale = ttl if ttl is not None else self.config.max_subslots
            request_features[index] = np.asarray([
                1.0,
                request.hops / max(self.config.max_hops, 1),
                max(0, request.hops - remaining) / max(request.hops, 1),
                remaining / max(self.config.max_hops, 1),
                contiguous / max(self.config.max_hops, 1),
                min((self.time - request.arrival) / max(age_scale, 1), 1.0),
                (self.time - self.last_service[request.id]) / max(self.config.max_subslots, 1),
                float(carried is not None),
                float(getattr(carried, "fidelity", 0.0)) if carried is not None else 0.0,
                float(np.mean(fidelities)) if fidelities else 0.0,
                float(request.id in self.selected_requests),
            ], dtype=np.float32)

        candidate_features = np.zeros(
            (self.stop_action, self.candidate_feature_dim), dtype=np.float32
        )
        legal = self.action_mask()
        for action, plan in enumerate(self.current_plans):
            if plan is None:
                continue
            request = self.instance.requests[plan.request_index]
            fidelities = [float(link.fidelity) for link in plan.input_links]
            distances = self._physical_distances(request.destination)
            before = distances.get(plan.start_index, self.config.max_hops)
            after = distances.get(plan.reach_index, self.config.max_hops)
            ttl = self._request_ttl(request)
            age_scale = ttl if ttl is not None else self.config.max_subslots
            ttl_remaining = (
                1.0 if ttl is None else
                max(0.0, (request.arrival + ttl - self.time) / max(ttl, 1))
            )
            candidate_features[action] = np.asarray([
                1.0,
                plan.request_index / max(self.config.max_requests - 1, 1),
                plan.progress / max(self.config.max_hops, 1),
                before / max(self.config.max_hops, 1),
                after / max(self.config.max_hops, 1),
                float(plan.completed),
                len(plan.base_links) / max(self.config.max_hops, 1),
                plan.swap_count / max(self.config.max_hops, 1),
                plan.swap_depth / max(math.ceil(math.log2(self.config.max_hops + 1)), 1),
                float(np.min(fidelities)) if fidelities else 0.0,
                float(np.mean(fidelities)) if fidelities else 0.0,
                float(np.prod(fidelities)) if fidelities else 0.0,
                float(plan.start_index != request.source),
                min((self.time - request.arrival) / max(age_scale, 1), 1.0),
                float(legal[action]),
                float(plan.kind == "max"),
                float(plan.kind == "half"),
                float(plan.kind == "short"),
                len({node for layer in plan.swap_nodes_by_layer for node in layer}) / max(len(self.network.nodes), 1),
                ttl_remaining,
            ], dtype=np.float32)

        active = self._active_requests()
        base_links = [link for edge in self.network.edges for link in edge.links]
        carried = [link for link in self.carried_links.values() if link is not None]
        fidelities = [float(link.fidelity) for link in base_links]
        remaining = [self._remaining_distance(request) for request in active]
        global_features = np.asarray([
            self.time / max(self.config.max_subslots, 1),
            self.planning_slots / max(self.config.max_subslots, 1),
            len(active) / max(self.config.max_requests, 1),
            len(self.completed_at) / max(len(self.instance.requests), 1),
            len(base_links) / max(len(self.network.edges) * self.config.n_quantum_links, 1),
            float(np.mean(fidelities)) if fidelities else 0.0,
            float(np.min(fidelities)) if fidelities else 0.0,
            len(carried) / max(self.config.max_requests, 1),
            len(self.selected_plans) / max(self.config.max_requests, 1),
            (sum(remaining) / max(len(remaining), 1)) / max(self.config.max_hops, 1),
            self.failed_plans / max(self.planning_slots, 1),
            len(self.expired_at) / max(len(self.instance.requests), 1),
        ], dtype=np.float32)
        return {
            "global_features": global_features,
            "request_features": request_features,
            "request_mask": request_mask,
            "candidate_features": candidate_features,
            "action_mask": legal,
        }

    def _info(self, *, duration: int, completed_now: int) -> dict[str, object]:
        delays = [
            self.completed_at[request.id] - request.arrival
            for request in self.instance.requests if request.id in self.completed_at
        ]
        return {
            "time": self.time,
            "duration": duration,
            "planning_slots": self.planning_slots,
            "completed": len(self.completed_at),
            "completed_now": completed_now,
            "expired": len(self.expired_at),
            "timeout_rate": len(self.expired_at) / max(len(self.instance.requests), 1),
            "active": len(self._active_requests()),
            "mean_delay": float(np.mean(delays)) if delays else 0.0,
            "sum_delay": float(sum(delays)),
            "elementary_eprs": self.elementary_eprs,
            "swaps": self.swaps,
            "failed_plans": self.failed_plans,
            "base_eprs": sum(len(edge.links) for edge in self.network.edges),
            "stop_action": self.stop_action,
        }


def make_env(
    stage: int = 0,
    seed: int = 0,
    reward_config: RewardConfig | None = None,
    **kwargs: Any,
) -> BatchSwapReliqEnv:
    config = EnvConfig(seed=seed, **kwargs)
    env = BatchSwapReliqEnv(config, reward_config)
    env.set_curriculum(stage)
    return env
