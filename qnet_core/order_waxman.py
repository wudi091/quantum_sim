"""Author-style Waxman topology and Poisson request workload for swap order.

The workload is deliberately separate from the three-request mechanism
motif.  It uses the Q-CAST author's Waxman-like topology generator, samples
uniform random source/destination pairs, and supports either exponential
inter-arrival times or a fixed-count homogeneous Poisson trace conditioned on
all requests arriving inside a fixed episode horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
import itertools
import random
from typing import Iterable

import networkx as nx
import numpy as np

from .order_core import (
    Edge,
    OrderBatchProblem,
    OrderCoreConfig,
    OrderLinkSpec,
    OrderPlan,
    OrderStoredPair,
)
from .contracts.complete_schedule import (
    CompleteSchedule,
    complete_schedule_count,
    enumerate_complete_schedules,
)
from .qcast_paper.topology import (
    AuthorTopologyConfig,
    generate_author_topology_with_metadata,
)


@dataclass(frozen=True, init=False)
class WaxmanOrderConfig:
    """Reproducible large-workload configuration.

    By default every arrived, unexpired pending request is exposed to the
    controller.  ``candidate_request_cap`` is an optional EDF-prefix pruning
    approximation, not a statement about the request-arrival process or the
    number of requests that the controller may complete.

    The formal workload exposes four preconfigured candidate paths and up to
    four complete swap-group schedules per path.  Supplying
    ``order_variants_per_path=None`` explicitly requests the exhaustive
    permutation catalogue for a separate ablation.

    """

    node_count: int = 100
    average_degree: int = 6
    target_link_probability: float = 0.6
    request_count: int = 100
    arrival_rate: float = 4.0
    episode_steps: int | None = None
    request_ttl_slots: int = 10
    min_hops: int = 2
    max_hops: int = 6
    candidate_paths: int = 4
    order_variants_per_path: int | None = 4
    candidate_request_cap: int | None = None
    node_memory_cap: int | None = 4
    slot_duration_ps: int = 6_000
    generation_interval_ps: int = 1_000
    swap_service_ps: int = 1_000
    memory_reset_ps: int = 100
    swap_probability: float = 0.9
    bsm_capacity_per_node: int = 1
    epr_ttl_slots: int = 3

    def __init__(
        self,
        node_count: int = 100,
        average_degree: int = 6,
        target_link_probability: float = 0.6,
        request_count: int = 100,
        arrival_rate: float = 4.0,
        episode_steps: int | None = None,
        request_ttl_slots: int = 10,
        min_hops: int = 2,
        max_hops: int = 6,
        candidate_paths: int = 4,
        order_variants_per_path: int | None = 4,
        candidate_request_cap: int | None = None,
        node_memory_cap: int | None = 4,
        slot_duration_ps: int = 6_000,
        generation_interval_ps: int = 1_000,
        swap_service_ps: int = 1_000,
        memory_reset_ps: int = 100,
        swap_probability: float = 0.9,
        bsm_capacity_per_node: int = 1,
        epr_ttl_slots: int = 3,
    ) -> None:
        values = {
            "node_count": node_count,
            "average_degree": average_degree,
            "target_link_probability": target_link_probability,
            "request_count": request_count,
            "arrival_rate": arrival_rate,
            "episode_steps": episode_steps,
            "request_ttl_slots": request_ttl_slots,
            "min_hops": min_hops,
            "max_hops": max_hops,
            "candidate_paths": candidate_paths,
            "order_variants_per_path": order_variants_per_path,
            "candidate_request_cap": candidate_request_cap,
            "node_memory_cap": node_memory_cap,
            "slot_duration_ps": slot_duration_ps,
            "generation_interval_ps": generation_interval_ps,
            "swap_service_ps": swap_service_ps,
            "memory_reset_ps": memory_reset_ps,
            "swap_probability": swap_probability,
            "bsm_capacity_per_node": bsm_capacity_per_node,
            "epr_ttl_slots": epr_ttl_slots,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        for name in (
            "node_count", "average_degree", "request_count",
            "request_ttl_slots", "min_hops", "max_hops",
            "candidate_paths",
            "slot_duration_ps", "generation_interval_ps",
            "swap_service_ps", "bsm_capacity_per_node", "epr_ttl_slots",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if (
            self.order_variants_per_path is not None
            and self.order_variants_per_path < 1
        ):
            raise ValueError(
                "order_variants_per_path must be positive when supplied"
            )
        if self.node_count < 2:
            raise ValueError("Waxman topology needs at least two nodes")
        if self.max_hops < self.min_hops:
            raise ValueError("max_hops cannot be smaller than min_hops")
        if self.arrival_rate <= 0.0:
            raise ValueError("arrival_rate must be positive")
        if self.episode_steps is not None and self.episode_steps < 1:
            raise ValueError("episode_steps must be positive when supplied")
        if not 0.0 < self.target_link_probability <= 1.0:
            raise ValueError("target_link_probability must lie in (0, 1]")
        if not 0.0 <= self.swap_probability <= 1.0:
            raise ValueError("swap_probability must lie in [0, 1]")
        if self.node_memory_cap is not None and self.node_memory_cap < 1:
            raise ValueError("node_memory_cap must be positive when supplied")
        if (
            self.candidate_request_cap is not None
            and self.candidate_request_cap < 1
        ):
            raise ValueError(
                "candidate_request_cap must be positive when supplied"
            )

    @property
    def max_swap_orders_per_path(self) -> int:
        """Tensor-safe upper bound for one configured path's schedule library."""

        exhaustive = complete_schedule_count(max(int(self.max_hops) - 1, 0))
        cap = self.order_variants_per_path
        return exhaustive if cap is None else min(int(cap), exhaustive)


@dataclass(frozen=True)
class WaxmanOrderRequest:
    request_id: str
    source: int
    destination: int
    arrival_slot: int
    deadline_slot: int
    shortest_hops: int


@dataclass(frozen=True)
class WaxmanOrderEpisode:
    seed: int
    config: WaxmanOrderConfig
    nodes: tuple[int, ...]
    links: tuple[OrderLinkSpec, ...]
    node_capacities: tuple[tuple[int, int], ...]
    positions: tuple[tuple[int, tuple[float, float]], ...]
    requests: tuple[WaxmanOrderRequest, ...]
    request_paths: tuple[tuple[str, tuple[tuple[int, ...], ...]], ...]
    topology_beta: float
    link_alpha: float
    horizon_slots: int
    schedule_library: tuple[
        tuple[tuple[int, ...], tuple[CompleteSchedule, ...]], ...
    ] = ()
    schedule_library_source: str = "static-template"
    schedule_library_digest: str = ""

    @property
    def capacity(self) -> dict[int, int]:
        return dict(self.node_capacities)

    @property
    def paths(self) -> dict[str, tuple[tuple[int, ...], ...]]:
        return dict(self.request_paths)

    @property
    def request_by_id(self) -> dict[str, WaxmanOrderRequest]:
        return {request.request_id: request for request in self.requests}

    @cached_property
    def schedules_by_path(self) -> dict[tuple[int, ...], tuple[CompleteSchedule, ...]]:
        if self.schedule_library:
            return dict(self.schedule_library)
        unique_paths = tuple(dict.fromkeys(
            path
            for _, paths in self.request_paths
            for path in paths
        ))
        return {
            path: _canonical_schedule_catalogue(
                path, self.config.order_variants_per_path
            )
            for path in unique_paths
        }

    def with_schedule_library(
        self,
        schedules_by_path: dict[
            tuple[int, ...], tuple[CompleteSchedule, ...]
        ],
        *,
        source: str,
        structural_digest: str,
    ) -> "WaxmanOrderEpisode":
        """Return an episode reading a fitted immutable offline library."""

        if not source or not structural_digest:
            raise ValueError("offline library source and digest must be non-empty")
        required_paths = {
            path for _, paths in self.request_paths for path in paths
        }
        if set(schedules_by_path) != required_paths:
            raise ValueError(
                "offline library must cover exactly every configured episode path"
            )
        normalized = []
        for path in sorted(required_paths):
            schedules = tuple(schedules_by_path[path])
            expected = min(
                self.config.max_swap_orders_per_path,
                complete_schedule_count(len(path) - 2),
            )
            if len(schedules) != expected:
                raise ValueError(
                    f"path {path} needs its effective budget of {expected} "
                    "unique schedules"
                )
            if any(schedule.path != path for schedule in schedules):
                raise ValueError("offline schedule is attached to the wrong path")
            keys = {schedule.structural_key for schedule in schedules}
            if len(keys) != len(schedules):
                raise ValueError("offline library cannot pad with duplicate schedules")
            normalized.append((path, schedules))
        return replace(
            self,
            schedule_library=tuple(normalized),
            schedule_library_source=source,
            schedule_library_digest=structural_digest,
        )

    def eligible_request_ids(
        self,
        pending_request_ids: Iterable[str],
        slot: int,
    ) -> tuple[str, ...]:
        """Return every arrived, unexpired pending request in EDF order."""

        lookup = self.request_by_id
        eligible = [
            request_id for request_id in pending_request_ids
            if lookup[request_id].arrival_slot <= slot
            < lookup[request_id].deadline_slot
        ]
        eligible.sort(key=lambda request_id: (
            lookup[request_id].deadline_slot,
            lookup[request_id].arrival_slot,
            request_id,
        ))
        return tuple(eligible)

    def considered_request_ids(
        self,
        pending_request_ids: Iterable[str],
        slot: int,
    ) -> tuple[str, ...]:
        """Apply the optional EDF candidate-pruning cap to eligible requests."""

        eligible = self.eligible_request_ids(pending_request_ids, slot)
        cap = self.config.candidate_request_cap
        return eligible if cap is None else eligible[:cap]

    def active_request_ids(
        self,
        pending_request_ids: Iterable[str],
        slot: int,
    ) -> tuple[str, ...]:
        """Compatibility alias for the full, unpruned eligible request set."""

        return self.eligible_request_ids(pending_request_ids, slot)

    def problem_for_slot(
        self,
        request_ids: Iterable[str],
        slot: int,
        *,
        physics_seed: int,
        initial_inventory: Iterable[OrderStoredPair] = (),
    ) -> OrderBatchProblem:
        request_ids = tuple(request_ids)
        if not request_ids:
            raise ValueError("a decision slot needs at least one request")
        lookup = self.request_by_id
        paths_by_request = self.paths

        schedules_by_path = self.schedules_by_path
        candidates: list[OrderPlan] = []
        for priority, request_id in enumerate(request_ids):
            request = lookup[request_id]
            if not request.arrival_slot <= slot < request.deadline_slot:
                raise ValueError("request is not active in this control slot")
            for path_index, path in enumerate(paths_by_request[request_id]):
                schedules = schedules_by_path[path]
                for schedule_index, schedule in enumerate(schedules):
                    candidates.append(OrderPlan(
                        plan_id=(
                            f"t{slot}:{request_id}:p{path_index}:o{schedule_index}"
                        ),
                        request_id=request_id,
                        path=path,
                        swap_order=schedule.swap_order,
                        priority=priority,
                        arrival_slot=request.arrival_slot,
                        deadline_slot=request.deadline_slot,
                        decision_slot=slot,
                        swap_groups=schedule.groups,
                        fixed_path_baseline=(schedule_index == 0),
                    ))

        mean_link_probability = sum(
            link.generation_probability for link in self.links
        ) / len(self.links)
        return OrderBatchProblem.create(
            candidates=candidates,
            node_capacity=self.capacity,
            links=self.links,
            config=OrderCoreConfig(
                slot_duration_ps=self.config.slot_duration_ps,
                generation_interval_ps=self.config.generation_interval_ps,
                swap_service_ps=self.config.swap_service_ps,
                memory_reset_ps=self.config.memory_reset_ps,
                generation_probability=mean_link_probability,
                swap_probability=self.config.swap_probability,
                edge_capacity=1,
                bsm_capacity_per_node=self.config.bsm_capacity_per_node,
                epr_ttl_slots=self.config.epr_ttl_slots,
                seed=int(physics_seed),
                slot_id=int(slot),
            ),
            initial_inventory=initial_inventory,
            # Keep the public snapshot free of a structure-seed label that
            # could be combined with source code to infer hidden test RNG.
            name=f"waxman-slot{slot}",
        )


def _canonical_schedule_catalogue(
    path: tuple[int, ...],
    limit: int | None,
) -> tuple[CompleteSchedule, ...]:
    """Algorithm-neutral deterministic catalogue used when no library is injected."""

    schedules = enumerate_complete_schedules(path)
    return schedules if limit is None else schedules[:limit]


def _aggregate_links(topology) -> tuple[OrderLinkSpec, ...]:
    capacity: dict[Edge, int] = {}
    weighted_probability: dict[Edge, float] = {}
    for item in topology.edges:
        elementary_edge = item.edge
        probabilities = item.probabilities
        capacity[elementary_edge] = (
            capacity.get(elementary_edge, 0) + len(probabilities)
        )
        weighted_probability[elementary_edge] = (
            weighted_probability.get(elementary_edge, 0.0)
            + sum(probabilities)
        )
    return tuple(
        OrderLinkSpec(
            *elementary_edge,
            capacity=capacity[elementary_edge],
            generation_probability=(
                weighted_probability[elementary_edge]
                / capacity[elementary_edge]
            ),
        )
        for elementary_edge in sorted(capacity)
    )


def _candidate_paths(
    graph: nx.Graph,
    source: int,
    destination: int,
    *,
    count: int,
    max_hops: int,
) -> tuple[tuple[int, ...], ...]:
    result: list[tuple[int, ...]] = []
    for path in itertools.islice(
        nx.shortest_simple_paths(graph, source, destination),
        max(8, count * 4),
    ):
        value = tuple(map(int, path))
        if len(value) - 1 > max_hops:
            continue
        result.append(value)
        if len(result) >= count:
            break
    if not result:
        raise RuntimeError("request has no path inside the configured hop cap")
    return tuple(result)


def make_waxman_order_episode(
    config: WaxmanOrderConfig = WaxmanOrderConfig(),
    seed: int = 0,
) -> WaxmanOrderEpisode:
    """Generate one topology, 100-request Poisson trace, and path catalogue."""

    topology_result = generate_author_topology_with_metadata(
        AuthorTopologyConfig(
            node_count=config.node_count,
            average_degree=config.average_degree,
            target_link_probability=config.target_link_probability,
            swap_probability=config.swap_probability,
            seed=int(seed),
        ),
        random.Random(seed),
    )
    topology = topology_result.topology
    graph = nx.Graph()
    graph.add_nodes_from(topology.nodes)
    graph.add_edges_from(topology.edge_pairs)
    if not nx.is_connected(graph):
        raise RuntimeError("Waxman topology generator returned a disconnected graph")

    distances = dict(nx.all_pairs_shortest_path_length(graph))
    eligible_pairs = tuple(
        (int(source), int(destination), int(distance))
        for source, targets in distances.items()
        for destination, distance in targets.items()
        if source < destination
        and config.min_hops <= distance <= config.max_hops
    )
    if not eligible_pairs:
        raise RuntimeError("Waxman topology has no endpoint pairs in hop range")

    if config.episode_steps is None:
        # Preserve the original unbounded-horizon trace exactly for legacy
        # mechanism studies.
        endpoint_rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), 0x4F52444552])
        )
        inter_arrivals = endpoint_rng.exponential(
            1.0 / config.arrival_rate,
            config.request_count,
        )
        arrivals = np.floor(np.cumsum(inter_arrivals)).astype(int)
        horizon = int(arrivals[-1]) + config.request_ttl_slots
    else:
        # A homogeneous Poisson process conditioned on exactly N arrivals in
        # [0, H) has multinomial slot counts with equal probabilities.  This
        # gives a fixed 100-request, 30-step workload without falsely claiming
        # independent Poisson counts after conditioning on their total.
        arrival_rng = np.random.default_rng(np.random.SeedSequence([
            int(seed), 0x4F52444552, 0x4152524956414C,
        ]))
        endpoint_rng = np.random.default_rng(np.random.SeedSequence([
            int(seed), 0x4F52444552, 0x454E44504F494E54,
        ]))
        slot_probabilities = np.full(
            config.episode_steps,
            1.0 / config.episode_steps,
            dtype=float,
        )
        slot_counts = arrival_rng.multinomial(
            config.request_count,
            slot_probabilities,
        )
        arrivals = np.repeat(
            np.arange(config.episode_steps, dtype=int),
            slot_counts,
        )
        horizon = config.episode_steps
    requests: list[WaxmanOrderRequest] = []
    request_paths: list[tuple[str, tuple[tuple[int, ...], ...]]] = []
    for index in range(config.request_count):
        pair_index = int(endpoint_rng.integers(0, len(eligible_pairs)))
        left, right, hops = eligible_pairs[pair_index]
        if endpoint_rng.random() < 0.5:
            source, destination = left, right
        else:
            source, destination = right, left
        request_id = f"r{index}"
        arrival = int(arrivals[index])
        requests.append(WaxmanOrderRequest(
            request_id=request_id,
            source=source,
            destination=destination,
            arrival_slot=arrival,
            deadline_slot=arrival + config.request_ttl_slots,
            shortest_hops=hops,
        ))
        request_paths.append((
            request_id,
            _candidate_paths(
                graph,
                source,
                destination,
                count=config.candidate_paths,
                max_hops=config.max_hops,
            ),
        ))

    raw_capacity = topology.node_qubits
    node_capacities = tuple(
        (node, (
            raw_capacity[node]
            if config.node_memory_cap is None
            else min(raw_capacity[node], config.node_memory_cap)
        ))
        for node in topology.nodes
    )
    links = _aggregate_links(topology)
    unique_paths = tuple(dict.fromkeys(
        path
        for _, paths in request_paths
        for path in paths
    ))
    schedule_library = tuple(
        (
            path,
            _canonical_schedule_catalogue(
                path, config.order_variants_per_path
            ),
        )
        for path in unique_paths
    )
    return WaxmanOrderEpisode(
        seed=int(seed),
        config=config,
        nodes=topology.nodes,
        links=links,
        node_capacities=node_capacities,
        positions=tuple(sorted(topology_result.positions.items())),
        requests=tuple(requests),
        request_paths=tuple(request_paths),
        topology_beta=float(topology_result.beta),
        link_alpha=float(topology_result.alpha),
        horizon_slots=horizon,
        schedule_library=schedule_library,
    )
