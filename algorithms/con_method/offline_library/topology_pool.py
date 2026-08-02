"""Topology-wide path and complete-schedule pool construction for CON."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import hashlib
import itertools
from typing import Iterable, Mapping, TypeAlias

import networkx as nx

from qnet_core.contracts.complete_schedule import enumerate_complete_schedules

from .models import LibraryPathCandidate, LibraryScheduleTemplate


Node: TypeAlias = int | str
UNDIRECTED_DIRECTIONALITY = "undirected-symmetric-v1"
PATH_GENERATOR_VERSION = "networkx-shortest-simple-paths-v1"


def node_sort_key(node: Node) -> tuple[str, str]:
    return type(node).__name__, repr(node)


def canonical_pair(source: Node, destination: Node) -> tuple[Node, Node]:
    if source == destination:
        raise ValueError("source and destination must be different nodes")
    return tuple(sorted((source, destination), key=node_sort_key))  # type: ignore[return-value]


def canonical_physical_topology_fingerprint(
    nodes: Iterable[Node],
    links: Iterable[tuple[Node, Node, int, float]],
    node_capacities: Mapping[Node, int] | Iterable[tuple[Node, int]],
) -> str:
    """Hash all path/execution-relevant Waxman topology fields canonically."""

    node_values = tuple(sorted(set(nodes), key=node_sort_key))
    capacities = dict(node_capacities)
    if set(capacities) != set(node_values):
        raise ValueError("node capacities must cover exactly the topology nodes")
    normalized_links = []
    for left, right, capacity, probability in links:
        edge = canonical_pair(left, right)
        if edge[0] not in capacities or edge[1] not in capacities:
            raise ValueError("topology link references an unknown node")
        if int(capacity) < 1:
            raise ValueError("link capacity must be positive")
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError("link probability must lie in [0, 1]")
        normalized_links.append((
            edge[0], edge[1], int(capacity), round(float(probability), 15)
        ))
    normalized_links.sort(key=lambda value: (
        node_sort_key(value[0]), node_sort_key(value[1]), value[2], value[3]
    ))
    payload = (
        tuple((type(node).__name__, repr(node)) for node in node_values),
        tuple(
            (
                type(left).__name__, repr(left),
                type(right).__name__, repr(right),
                capacity, probability,
            )
            for left, right, capacity, probability in normalized_links
        ),
        tuple(
            (type(node).__name__, repr(node), int(capacities[node]))
            for node in node_values
        ),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def pair_id_for(
    topology_fingerprint: str,
    endpoints: tuple[Node, Node],
) -> str:
    digest = hashlib.sha256(
        repr((topology_fingerprint, endpoints)).encode("utf-8")
    ).hexdigest()[:20]
    return f"pair:{digest}"


def path_id_for(
    topology_fingerprint: str,
    path: tuple[Node, ...],
) -> str:
    digest = hashlib.sha256(
        repr((topology_fingerprint, path)).encode("utf-8")
    ).hexdigest()[:20]
    return f"path:{digest}"


def template_id_for(
    topology_fingerprint: str,
    path: tuple[Node, ...],
    structural_key: object,
) -> str:
    digest = hashlib.sha256(
        repr((topology_fingerprint, path, structural_key)).encode("utf-8")
    ).hexdigest()[:24]
    return f"schedule:{digest}"


@dataclass(frozen=True)
class TopologyTemplatePool:
    """All unordered pairs and their topology-bound structural candidates."""

    topology_fingerprint: str
    pair_entries: tuple[tuple[str, tuple[Node, Node]], ...]
    paths: tuple[LibraryPathCandidate, ...]
    templates: tuple[LibraryScheduleTemplate, ...]
    path_pool_per_pair: int
    schedules_per_path_pool: int | None
    max_hops: int | None
    structural_digest: str
    directionality: str = UNDIRECTED_DIRECTIONALITY
    path_generator_version: str = PATH_GENERATOR_VERSION

    def __post_init__(self) -> None:
        if not self.topology_fingerprint or not self.structural_digest:
            raise ValueError("topology and pool digests must be non-empty")
        if self.path_pool_per_pair < 1:
            raise ValueError("path_pool_per_pair must be positive")
        if (
            self.schedules_per_path_pool is not None
            and self.schedules_per_path_pool < 1
        ):
            raise ValueError("schedule pool limit must be positive or None")
        if self.max_hops is not None and self.max_hops < 1:
            raise ValueError("max_hops must be positive or None")
        pair_ids = tuple(pair_id for pair_id, _ in self.pair_entries)
        endpoints = tuple(endpoints for _, endpoints in self.pair_entries)
        if len(set(pair_ids)) != len(pair_ids):
            raise ValueError("pair IDs must be unique")
        if len(set(endpoints)) != len(endpoints):
            raise ValueError("canonical endpoint pairs must be unique")
        if any(canonical_pair(*pair) != pair for pair in endpoints):
            raise ValueError("pool endpoint pairs must use canonical direction")
        pair_by_id = dict(self.pair_entries)
        for path in self.paths:
            if path.pair_id not in pair_by_id:
                raise ValueError("path references an unknown endpoint pair")
            if (path.path[0], path.path[-1]) != pair_by_id[path.pair_id]:
                raise ValueError("path endpoints do not match its pair")

    @cached_property
    def pair_by_id(self) -> dict[str, tuple[Node, Node]]:
        return dict(self.pair_entries)

    @cached_property
    def pair_id_by_endpoints(self) -> dict[tuple[Node, Node], str]:
        return {
            endpoints: pair_id for pair_id, endpoints in self.pair_entries
        }

    @cached_property
    def path_by_id(self) -> dict[str, LibraryPathCandidate]:
        return {path.path_id: path for path in self.paths}

    @cached_property
    def template_by_id(self) -> dict[str, LibraryScheduleTemplate]:
        return {
            template.template_id: template for template in self.templates
        }

    @cached_property
    def paths_by_pair(self) -> dict[str, tuple[LibraryPathCandidate, ...]]:
        result: dict[str, list[LibraryPathCandidate]] = {
            pair_id: [] for pair_id, _ in self.pair_entries
        }
        for path in self.paths:
            result[path.pair_id].append(path)
        return {
            pair_id: tuple(sorted(values, key=lambda path: path.pool_rank))
            for pair_id, values in result.items()
        }

    @cached_property
    def templates_by_path(self) -> dict[str, tuple[LibraryScheduleTemplate, ...]]:
        result: dict[str, list[LibraryScheduleTemplate]] = {
            path.path_id: [] for path in self.paths
        }
        for template in self.templates:
            result[template.path_id].append(template)
        return {
            path_id: tuple(sorted(
                values,
                key=lambda template: (
                    template.structural_key, template.template_id
                ),
            ))
            for path_id, values in result.items()
        }


def _pool_digest(
    *,
    topology_fingerprint: str,
    pair_entries: tuple[tuple[str, tuple[Node, Node]], ...],
    paths: tuple[LibraryPathCandidate, ...],
    templates: tuple[LibraryScheduleTemplate, ...],
    path_pool_per_pair: int,
    schedules_per_path_pool: int | None,
    max_hops: int | None,
) -> str:
    payload = (
        "con-topology-pool-v1",
        topology_fingerprint,
        UNDIRECTED_DIRECTIONALITY,
        PATH_GENERATOR_VERSION,
        path_pool_per_pair,
        schedules_per_path_pool,
        max_hops,
        tuple(
            (pair_id, tuple(map(repr, endpoints)))
            for pair_id, endpoints in pair_entries
        ),
        tuple(
            (
                path.pair_id,
                path.path_id,
                path.pool_rank,
                tuple(map(repr, path.path)),
            )
            for path in paths
        ),
        tuple(
            (
                template.pair_id,
                template.path_id,
                template.template_id,
                template.structural_key,
            )
            for template in templates
        ),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def build_topology_template_pool(
    *,
    nodes: Iterable[Node],
    edges: Iterable[tuple[Node, Node]],
    topology_fingerprint: str,
    path_pool_per_pair: int = 8,
    schedules_per_path_pool: int | None = None,
    max_hops: int | None = None,
) -> TopologyTemplatePool:
    """Enumerate paths for every unordered pair, independent of request traces."""

    if not topology_fingerprint:
        raise ValueError("topology_fingerprint must be non-empty")
    if path_pool_per_pair < 1:
        raise ValueError("path_pool_per_pair must be positive")
    if schedules_per_path_pool is not None and schedules_per_path_pool < 1:
        raise ValueError("schedules_per_path_pool must be positive or None")
    if max_hops is not None and max_hops < 1:
        raise ValueError("max_hops must be positive or None")

    node_values = tuple(sorted(set(nodes), key=node_sort_key))
    if len(node_values) < 2:
        raise ValueError("a topology pool needs at least two nodes")
    graph = nx.Graph()
    graph.add_nodes_from(node_values)
    normalized_edges = tuple(sorted(
        {canonical_pair(left, right) for left, right in edges},
        key=lambda edge: (node_sort_key(edge[0]), node_sort_key(edge[1])),
    ))
    graph.add_edges_from(normalized_edges)

    pair_entries: list[tuple[str, tuple[Node, Node]]] = []
    paths: list[LibraryPathCandidate] = []
    templates: list[LibraryScheduleTemplate] = []
    for source, destination in itertools.combinations(node_values, 2):
        endpoints = canonical_pair(source, destination)
        pair_id = pair_id_for(topology_fingerprint, endpoints)
        pair_entries.append((pair_id, endpoints))
        try:
            path_generator = nx.shortest_simple_paths(
                graph, endpoints[0], endpoints[1]
            )
            path_values = []
            for raw_path in path_generator:
                path = tuple(raw_path)
                hops = len(path) - 1
                if max_hops is not None and hops > max_hops:
                    break
                path_values.append(path)
                if len(path_values) >= path_pool_per_pair:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path_values = []

        for pool_rank, path in enumerate(path_values):
            path_id = path_id_for(topology_fingerprint, path)
            path_candidate = LibraryPathCandidate(
                pair_id=pair_id,
                path_id=path_id,
                path=path,
                pool_rank=pool_rank,
            )
            paths.append(path_candidate)
            internal_count = len(path) - 2
            if internal_count > 5 and schedules_per_path_pool is None:
                raise ValueError(
                    "exhaustive complete-schedule enumeration above five "
                    "internal nodes is intentionally disabled; supply max_hops "
                    "or a bounded schedule-pool policy"
                )
            schedules = enumerate_complete_schedules(path)
            if schedules_per_path_pool is not None:
                schedules = schedules[:schedules_per_path_pool]
            templates.extend(
                LibraryScheduleTemplate(
                    template_id=template_id_for(
                        topology_fingerprint,
                        path,
                        schedule.structural_key,
                    ),
                    path_id=path_id,
                    schedule=schedule,
                    pair_id=pair_id,
                )
                for schedule in schedules
            )

    pair_entries_tuple = tuple(pair_entries)
    paths_tuple = tuple(paths)
    templates_tuple = tuple(templates)
    return TopologyTemplatePool(
        topology_fingerprint=topology_fingerprint,
        pair_entries=pair_entries_tuple,
        paths=paths_tuple,
        templates=templates_tuple,
        path_pool_per_pair=path_pool_per_pair,
        schedules_per_path_pool=schedules_per_path_pool,
        max_hops=max_hops,
        structural_digest=_pool_digest(
            topology_fingerprint=topology_fingerprint,
            pair_entries=pair_entries_tuple,
            paths=paths_tuple,
            templates=templates_tuple,
            path_pool_per_pair=path_pool_per_pair,
            schedules_per_path_pool=schedules_per_path_pool,
            max_hops=max_hops,
        ),
    )
