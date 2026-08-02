"""Request-independent path and complete-schedule portfolio algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import hashlib
import itertools
import math
from statistics import fmean
from typing import Iterable, Mapping

import networkx as nx

from qnet_core.contracts.complete_schedule import CompleteSchedule
from qnet_core.order_waxman import WaxmanOrderEpisode

from ..models import LibraryPathCandidate, LibraryScheduleTemplate, library_digest
from ..topology_pool import (
    Node,
    TopologyTemplatePool,
    canonical_pair,
    canonical_physical_topology_fingerprint,
    node_sort_key,
)


@dataclass(frozen=True)
class GeneratorPreset:
    name: str
    path_strategy: str
    schedule_strategy: str

    def __post_init__(self) -> None:
        if not self.name or not self.path_strategy or not self.schedule_strategy:
            raise ValueError("generator preset labels must be non-empty")


GENERATOR_PRESETS: dict[str, GeneratorPreset] = {
    "canonical": GeneratorPreset(
        "canonical", "shortest", "canonical"
    ),
    "quality": GeneratorPreset(
        "quality", "quality", "memory"
    ),
    "diverse": GeneratorPreset(
        "diverse", "diverse", "release_diverse"
    ),
    "hybrid": GeneratorPreset(
        "hybrid", "quality_diverse", "hybrid"
    ),
    "facility": GeneratorPreset(
        "facility", "quality_diverse", "facility"
    ),
    "pareto": GeneratorPreset(
        "pareto", "exact_portfolio", "pareto_diverse"
    ),
    "banded": GeneratorPreset(
        "banded", "exact_portfolio", "banded_maxmin"
    ),
    "exact_kcenter": GeneratorPreset(
        "exact_kcenter", "exact_portfolio", "exact_kcenter"
    ),
}


@dataclass(frozen=True)
class TopologySelectionContext:
    topology_fingerprint: str
    node_capacities: tuple[tuple[Node, int], ...]
    link_entries: tuple[tuple[Node, Node, int, float], ...]

    def __post_init__(self) -> None:
        if not self.topology_fingerprint:
            raise ValueError("topology_fingerprint must be non-empty")
        capacities = dict(self.node_capacities)
        if len(capacities) != len(self.node_capacities):
            raise ValueError("node capacities cannot repeat nodes")
        if any(value < 1 for value in capacities.values()):
            raise ValueError("node capacities must be positive")
        seen_edges = set()
        for left, right, capacity, probability in self.link_entries:
            edge = canonical_pair(left, right)
            if edge in seen_edges:
                raise ValueError("selection context links cannot repeat")
            seen_edges.add(edge)
            if capacity < 1 or not 0.0 <= probability <= 1.0:
                raise ValueError("invalid link capacity or probability")

    @cached_property
    def capacity(self) -> dict[Node, int]:
        return dict(self.node_capacities)

    @cached_property
    def link_by_edge(self) -> dict[tuple[Node, Node], tuple[int, float]]:
        return {
            canonical_pair(left, right): (capacity, probability)
            for left, right, capacity, probability in self.link_entries
        }

    @cached_property
    def graph(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(self.capacity)
        graph.add_edges_from(self.link_by_edge)
        return graph

    @cached_property
    def node_pressure(self) -> dict[Node, float]:
        """Topology-only hotspot proxy: betweenness divided by memory."""

        centrality = nx.betweenness_centrality(self.graph, normalized=True)
        return {
            node: (1.0 + centrality.get(node, 0.0)) / self.capacity[node]
            for node in self.capacity
        }


def build_waxman_selection_context(
    episode: WaxmanOrderEpisode,
    *,
    topology_fingerprint: str | None = None,
) -> TopologySelectionContext:
    computed_fingerprint = canonical_physical_topology_fingerprint(
        episode.nodes,
        (
            (
                link.left,
                link.right,
                link.capacity,
                link.generation_probability,
            )
            for link in episode.links
        ),
        episode.node_capacities,
    )
    if (
        topology_fingerprint is not None
        and topology_fingerprint != computed_fingerprint
    ):
        raise ValueError("supplied topology fingerprint does not match episode")
    return TopologySelectionContext(
        topology_fingerprint=computed_fingerprint,
        node_capacities=tuple(sorted(
            episode.node_capacities, key=lambda item: node_sort_key(item[0])
        )),
        link_entries=tuple(sorted(
            (
                link.left,
                link.right,
                int(link.capacity),
                float(link.generation_probability),
            )
            for link in episode.links
        )),
    )


@dataclass(frozen=True)
class _PathProfile:
    candidate: LibraryPathCandidate
    hops: int
    reliability_cost: float
    edge_cost: float
    node_cost: float
    bottleneck_cost: float
    edge_scarcity: tuple[tuple[tuple[Node, Node], float], ...]
    node_scarcity: tuple[tuple[Node, float], ...]

    @property
    def path(self):
        return self.candidate.path


@dataclass(frozen=True)
class _ScheduleProfile:
    template: LibraryScheduleTemplate
    round_count: int
    memory_time: float
    weighted_memory_time: float
    release_vector: tuple[float, ...]
    node_weights: tuple[float, ...]
    parallel_pairs: frozenset[tuple[int, int]]

    @property
    def schedule(self) -> CompleteSchedule:
        return self.template.schedule


@dataclass(frozen=True)
class StructuralLibrarySelection:
    generator_name: str
    path_strategy: str
    schedule_strategy: str
    selected_path_ids: tuple[str, ...]
    selected_template_ids: tuple[str, ...]
    selected_by_pair: tuple[tuple[str, tuple[str, ...]], ...]
    selected_by_path: tuple[tuple[str, tuple[str, ...]], ...]
    diagnostics: tuple[tuple[str, float], ...]
    structural_digest: str
    generator_fingerprint: str

    @cached_property
    def path_ids_by_pair(self) -> dict[str, tuple[str, ...]]:
        return dict(self.selected_by_pair)

    @cached_property
    def template_ids_by_path(self) -> dict[str, tuple[str, ...]]:
        return dict(self.selected_by_path)

    @cached_property
    def diagnostic_values(self) -> dict[str, float]:
        return dict(self.diagnostics)


def _normalize(values: Mapping[str, float]) -> dict[str, float]:
    if not values:
        return {}
    minimum = min(values.values())
    maximum = max(values.values())
    if math.isclose(minimum, maximum):
        return {key: 0.0 for key in values}
    scale = maximum - minimum
    return {key: (value - minimum) / scale for key, value in values.items()}


def _path_profile(
    path: LibraryPathCandidate,
    context: TopologySelectionContext | None,
) -> _PathProfile:
    if context is None:
        return _PathProfile(
            path, len(path.path) - 1, 0.0, 0.0, 0.0, 0.0, (), ()
        )
    reliability_cost = 0.0
    edge_resource_cost = 0.0
    edge_scarcity = []
    for left, right in zip(path.path, path.path[1:]):
        concrete_edge = canonical_pair(left, right)
        capacity, probability = context.link_by_edge[concrete_edge]
        reliability_cost += -math.log(max(probability, 1e-12))
        scarcity = 1.0 / capacity
        edge_resource_cost += scarcity
        edge_scarcity.append((concrete_edge, scarcity))
    internal_pressure = tuple(
        (node, context.node_pressure[node]) for node in path.path[1:-1]
    )
    node_resource_cost = sum(value for _, value in internal_pressure)
    bottleneck_cost = max(
        max((value for _, value in edge_scarcity), default=0.0),
        max(
            (2.0 / context.capacity[node] for node in path.path[1:-1]),
            default=0.0,
        ),
    )
    return _PathProfile(
        candidate=path,
        hops=len(path.path) - 1,
        reliability_cost=reliability_cost,
        edge_cost=edge_resource_cost,
        node_cost=node_resource_cost,
        bottleneck_cost=bottleneck_cost,
        edge_scarcity=tuple(edge_scarcity),
        node_scarcity=internal_pressure,
    )


def _path_distance(left: _PathProfile, right: _PathProfile) -> float:
    left_edges = {
        canonical_pair(u, v) for u, v in zip(left.path, left.path[1:])
    }
    right_edges = {
        canonical_pair(u, v) for u, v in zip(right.path, right.path[1:])
    }
    edge_union = left_edges | right_edges
    edge_weight = dict(left.edge_scarcity)
    for edge, value in right.edge_scarcity:
        edge_weight[edge] = max(edge_weight.get(edge, 0.0), value)
    edge_total = sum(edge_weight.get(edge, 1.0) for edge in edge_union)
    edge_shared = sum(
        edge_weight.get(edge, 1.0) for edge in left_edges & right_edges
    )
    edge_distance = 1.0 - edge_shared / edge_total if edge_total else 0.0
    left_nodes = set(left.path[1:-1])
    right_nodes = set(right.path[1:-1])
    node_union = left_nodes | right_nodes
    node_weight = dict(left.node_scarcity)
    for node, value in right.node_scarcity:
        node_weight[node] = max(node_weight.get(node, 0.0), value)
    node_total = sum(node_weight.get(node, 1.0) for node in node_union)
    node_shared = sum(
        node_weight.get(node, 1.0) for node in left_nodes & right_nodes
    )
    node_distance = 1.0 - node_shared / node_total if node_total else 0.0
    return 0.7 * edge_distance + 0.3 * node_distance


def _select_paths(
    paths: tuple[LibraryPathCandidate, ...],
    *,
    strategy: str,
    limit: int,
    context: TopologySelectionContext | None,
) -> tuple[LibraryPathCandidate, ...]:
    if not paths:
        return ()
    budget = min(limit, len(paths))
    profiles = tuple(_path_profile(path, context) for path in paths)
    by_id = {profile.candidate.path_id: profile for profile in profiles}
    hop = _normalize({p.candidate.path_id: float(p.hops) for p in profiles})
    risk = _normalize({
        p.candidate.path_id: p.reliability_cost for p in profiles
    })
    edge_cost = _normalize({
        p.candidate.path_id: p.edge_cost for p in profiles
    })
    node_cost = _normalize({
        p.candidate.path_id: p.node_cost for p in profiles
    })
    bottleneck = _normalize({
        p.candidate.path_id: p.bottleneck_cost for p in profiles
    })

    def quality(profile: _PathProfile) -> float:
        path_id = profile.candidate.path_id
        return (
            0.35 * hop[path_id]
            + 0.30 * risk[path_id]
            + 0.15 * edge_cost[path_id]
            + 0.10 * node_cost[path_id]
            + 0.10 * bottleneck[path_id]
        )

    stable = lambda profile: (
        profile.candidate.pool_rank,
        tuple(map(repr, profile.path)),
        profile.candidate.path_id,
    )
    if strategy == "shortest":
        chosen = sorted(profiles, key=lambda profile: (
            profile.hops, stable(profile)
        ))[:budget]
        return tuple(profile.candidate for profile in chosen)
    if strategy == "quality":
        chosen = sorted(profiles, key=lambda profile: (
            quality(profile), profile.hops, stable(profile)
        ))[:budget]
        return tuple(profile.candidate for profile in chosen)
    if strategy == "exact_portfolio":
        def exact_key(chosen):
            qualities = tuple(quality(profile) for profile in chosen)
            distances = tuple(
                _path_distance(left, right)
                for index, left in enumerate(chosen)
                for right in chosen[index + 1:]
            )
            overlaps = tuple(1.0 - value for value in distances)
            mean_overlap = fmean(overlaps) if overlaps else 0.0
            max_overlap = max(overlaps, default=0.0)
            score = (
                0.50 * fmean(qualities)
                + 0.20 * max(qualities)
                + 0.20 * mean_overlap
                + 0.10 * max_overlap
            )
            return (
                round(score, 12),
                round(max(qualities), 12),
                round(fmean(qualities), 12),
                round(max_overlap, 12),
                sum(profile.hops for profile in chosen),
                tuple(stable(profile) for profile in chosen),
            )

        best = min(
            itertools.combinations(profiles, budget),
            key=exact_key,
        )
        return tuple(profile.candidate for profile in best)
    if strategy not in {"diverse", "quality_diverse"}:
        raise ValueError(f"unknown path strategy: {strategy}")

    selected = [min(profiles, key=lambda profile: (
        quality(profile), profile.hops, stable(profile)
    ))]
    while len(selected) < budget:
        remaining = tuple(
            profile for profile in profiles if profile not in selected
        )

        def selection_key(profile: _PathProfile):
            minimum_distance = min(
                _path_distance(profile, chosen) for chosen in selected
            )
            if strategy == "diverse":
                return (-minimum_distance, quality(profile), stable(profile))
            maximum_overlap = max(
                1.0 - _path_distance(profile, chosen) for chosen in selected
            )
            combined = quality(profile) + 0.65 * maximum_overlap
            return (combined, -minimum_distance, stable(profile))

        selected.append(min(remaining, key=selection_key))
    return tuple(profile.candidate for profile in selected)


def _schedule_profile(
    template: LibraryScheduleTemplate,
    context: TopologySelectionContext | None,
) -> _ScheduleProfile:
    schedule = template.schedule
    internal = schedule.path[1:-1]
    releases = schedule.release_round_by_node
    node_weights = tuple(
        context.node_pressure[node] if context is not None else 1.0
        for node in internal
    )
    release_vector = tuple(float(releases[node]) for node in internal)
    parallel_pairs = frozenset(
        (left, right)
        for left in range(len(internal))
        for right in range(left + 1, len(internal))
        if release_vector[left] == release_vector[right]
    )
    memory_time = float(
        2 * sum(releases[node] for node in internal)
        + 2 * schedule.round_count
    )
    weighted_memory_time = float(
        2 * sum(
            weight * releases[node]
            for node, weight in zip(internal, node_weights)
        )
        + 2 * schedule.round_count
    )
    return _ScheduleProfile(
        template=template,
        round_count=schedule.round_count,
        memory_time=memory_time,
        weighted_memory_time=weighted_memory_time,
        release_vector=release_vector,
        node_weights=node_weights,
        parallel_pairs=parallel_pairs,
    )


def _schedule_distance(
    left: _ScheduleProfile,
    right: _ScheduleProfile,
) -> float:
    if len(left.release_vector) != len(right.release_vector):
        raise ValueError("schedule distance requires the same concrete path")
    denominator = max(
        left.round_count, right.round_count, len(left.release_vector), 1
    )
    weights = tuple(
        max(left_weight, right_weight)
        for left_weight, right_weight in zip(
            left.node_weights, right.node_weights
        )
    )
    total_weight = sum(weights) or 1.0
    release_distance = sum(
        weight * abs(left_value - right_value) / denominator
        for left_value, right_value, weight in zip(
            left.release_vector, right.release_vector, weights
        )
    ) / total_weight
    pair_denominator = max(
        len(left.release_vector) * (len(left.release_vector) - 1) // 2,
        1,
    )
    parallel_distance = (
        len(left.parallel_pairs ^ right.parallel_pairs) / pair_denominator
    )
    round_distance = abs(left.round_count - right.round_count) / max(
        len(left.release_vector) - 1, 1
    )
    return (
        0.60 * release_distance
        + 0.25 * parallel_distance
        + 0.15 * round_distance
    )


def _pareto_schedule_profiles(
    profiles: tuple[_ScheduleProfile, ...],
) -> tuple[_ScheduleProfile, ...]:
    """Remove schedules no better in rounds, memory, or any node release."""

    result = []
    for candidate in profiles:
        dominated = False
        for other in profiles:
            if other is candidate:
                continue
            no_worse = (
                other.round_count <= candidate.round_count
                and other.memory_time <= candidate.memory_time
                and len(other.parallel_pairs) >= len(candidate.parallel_pairs)
                and all(
                    left <= right
                    for left, right in zip(
                        other.release_vector, candidate.release_vector
                    )
                )
            )
            strictly_better = (
                other.round_count < candidate.round_count
                or other.memory_time < candidate.memory_time
                or len(other.parallel_pairs) > len(candidate.parallel_pairs)
                or any(
                    left < right
                    for left, right in zip(
                        other.release_vector, candidate.release_vector
                    )
                )
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            result.append(candidate)
    return tuple(result)


def _select_schedules(
    templates: tuple[LibraryScheduleTemplate, ...],
    *,
    strategy: str,
    limit: int,
    context: TopologySelectionContext | None,
) -> tuple[LibraryScheduleTemplate, ...]:
    if not templates:
        return ()
    budget = min(limit, len(templates))
    profiles = tuple(
        _schedule_profile(template, context) for template in templates
    )
    weighted = _normalize({
        profile.template.template_id: profile.weighted_memory_time
        for profile in profiles
    })
    memory = _normalize({
        profile.template.template_id: profile.memory_time
        for profile in profiles
    })
    rounds = _normalize({
        profile.template.template_id: float(profile.round_count)
        for profile in profiles
    })

    def quality(profile: _ScheduleProfile) -> float:
        template_id = profile.template.template_id
        return (
            0.60 * weighted[template_id]
            + 0.25 * rounds[template_id]
            + 0.15 * memory[template_id]
        )

    stable = lambda profile: (
        profile.schedule.structural_key,
        profile.template.template_id,
    )
    if strategy == "canonical":
        return tuple(
            profile.template
            for profile in sorted(profiles, key=lambda profile: (
                profile.round_count,
                profile.memory_time,
                stable(profile),
            ))[:budget]
        )
    if strategy == "memory":
        return tuple(
            profile.template
            for profile in sorted(profiles, key=lambda profile: (
                quality(profile), stable(profile)
            ))[:budget]
        )
    if strategy not in {
        "release_diverse",
        "facility",
        "hybrid",
        "pareto_diverse",
        "banded_maxmin",
        "exact_kcenter",
    }:
        raise ValueError(f"unknown schedule strategy: {strategy}")

    if strategy == "banded_maxmin":
        minimum_rounds = min(profile.round_count for profile in profiles)
        minimum_memory = min(profile.memory_time for profile in profiles)
        internal_count = len(profiles[0].release_vector)
        profiles = tuple(
            profile for profile in profiles
            if profile.round_count <= minimum_rounds + 1
            and profile.memory_time <= minimum_memory + 2 * internal_count
        )
    else:
        profiles = _pareto_schedule_profiles(profiles)
    budget = min(budget, len(profiles))

    selected: list[_ScheduleProfile] = []

    def add(profile: _ScheduleProfile) -> None:
        if profile not in selected and len(selected) < budget:
            selected.append(profile)

    if strategy == "exact_kcenter":
        anchor = min(profiles, key=lambda profile: (
            quality(profile), stable(profile)
        ))
        if budget == 1:
            selected = [anchor]
        elif len(profiles) <= 24:
            combinations = (
                chosen
                for chosen in itertools.combinations(profiles, budget)
                if anchor in chosen
            )

            def kcenter_key(chosen):
                nearest = tuple(
                    min(_schedule_distance(item, center) for center in chosen)
                    for item in profiles
                )
                return (
                    round(max(nearest, default=0.0), 12),
                    round(sum(nearest), 12),
                    sum(item.round_count for item in chosen),
                    sum(item.memory_time for item in chosen),
                    -sum(len(item.parallel_pairs) for item in chosen),
                    tuple(sorted(stable(item) for item in chosen)),
                )

            selected = list(min(combinations, key=kcenter_key))
            selected.sort(key=lambda profile: (
                0 if profile is anchor else 1,
                quality(profile),
                stable(profile),
            ))
        else:
            selected = [anchor]
            while len(selected) < budget:
                remaining = tuple(
                    profile for profile in profiles if profile not in selected
                )
                add(min(remaining, key=lambda profile: (
                    -min(
                        _schedule_distance(profile, chosen)
                        for chosen in selected
                    ),
                    quality(profile),
                    stable(profile),
                )))
    elif strategy == "facility":
        add(min(profiles, key=lambda profile: (
            quality(profile), stable(profile)
        )))
        while len(selected) < budget:
            current_cost = sum(
                min(_schedule_distance(item, chosen) for chosen in selected)
                for item in profiles
            )
            remaining = tuple(
                candidate for candidate in profiles if candidate not in selected
            )

            def facility_key(candidate: _ScheduleProfile):
                new_cost = sum(
                    min(
                        _schedule_distance(item, candidate),
                        min(
                            _schedule_distance(item, chosen)
                            for chosen in selected
                        ),
                    )
                    for item in profiles
                )
                gain = current_cost - new_cost
                return (-gain, quality(candidate), stable(candidate))

            add(min(remaining, key=facility_key))
    elif strategy == "pareto_diverse":
        add(min(profiles, key=lambda profile: (
            quality(profile), stable(profile)
        )))
        while len(selected) < budget:
            best_release = tuple(
                min(profile.release_vector[index] for profile in selected)
                for index in range(len(profiles[0].release_vector))
            )
            remaining = tuple(
                profile for profile in profiles if profile not in selected
            )
            add(min(remaining, key=lambda profile: (
                -sum(
                    weight * max(0.0, best - value)
                    for weight, best, value in zip(
                        profile.node_weights,
                        best_release,
                        profile.release_vector,
                    )
                ),
                -min(
                    _schedule_distance(profile, chosen)
                    for chosen in selected
                ),
                quality(profile),
                stable(profile),
            )))
    else:
        add(min(profiles, key=lambda profile: (
            quality(profile), stable(profile)
        )))
        if strategy == "hybrid":
            add(min(profiles, key=lambda profile: (
                profile.round_count,
                profile.memory_time,
                stable(profile),
            )))
            internal = profiles[0].schedule.path[1:-1]
            if internal:
                hottest_index = max(
                    range(len(internal)),
                    key=lambda index: (
                        profiles[0].node_weights[index], -index
                    ),
                )
                add(min(profiles, key=lambda profile: (
                    profile.release_vector[hottest_index],
                    quality(profile),
                    stable(profile),
                )))
        while len(selected) < budget:
            remaining = tuple(
                profile for profile in profiles if profile not in selected
            )
            add(min(remaining, key=lambda profile: (
                -min(
                    _schedule_distance(profile, chosen)
                    for chosen in selected
                ),
                quality(profile),
                stable(profile),
            )))
    return tuple(profile.template for profile in selected)


def _mean_pairwise_distance(
    values: tuple[object, ...],
    distance,
) -> float:
    pairs = tuple(
        distance(left, right)
        for index, left in enumerate(values)
        for right in values[index + 1:]
    )
    return fmean(pairs) if pairs else 0.0


def select_structural_library(
    pool: TopologyTemplatePool,
    *,
    preset: str | GeneratorPreset = "hybrid",
    context: TopologySelectionContext | None = None,
    paths_per_pair: int = 4,
    schedules_per_path: int = 4,
) -> StructuralLibrarySelection:
    """Select a topology-only fixed portfolio without observing requests."""

    if context is not None and context.topology_fingerprint != pool.topology_fingerprint:
        raise ValueError("selection context and topology pool do not match")
    if paths_per_pair < 1 or schedules_per_path < 1:
        raise ValueError("path and schedule budgets must be positive")
    if isinstance(preset, str):
        try:
            preset_value = GENERATOR_PRESETS[preset]
        except KeyError as exc:
            raise ValueError(f"unknown generator preset: {preset}") from exc
    else:
        preset_value = preset

    selected_paths: list[LibraryPathCandidate] = []
    selected_templates: list[LibraryScheduleTemplate] = []
    by_pair = []
    by_path = []
    path_profiles_for_diagnostics = []
    schedule_profiles_for_diagnostics = []
    for pair_id, _ in pool.pair_entries:
        pair_paths = _select_paths(
            pool.paths_by_pair[pair_id],
            strategy=preset_value.path_strategy,
            limit=paths_per_pair,
            context=context,
        )
        selected_paths.extend(pair_paths)
        by_pair.append((pair_id, tuple(path.path_id for path in pair_paths)))
        path_profiles_for_diagnostics.extend(
            _path_profile(path, context) for path in pair_paths
        )
        for path in pair_paths:
            schedules = _select_schedules(
                pool.templates_by_path[path.path_id],
                strategy=preset_value.schedule_strategy,
                limit=schedules_per_path,
                context=context,
            )
            selected_templates.extend(schedules)
            by_path.append((
                path.path_id,
                tuple(template.template_id for template in schedules),
            ))
            schedule_profiles_for_diagnostics.extend(
                _schedule_profile(template, context) for template in schedules
            )

    selected_path_tuple = tuple(selected_paths)
    selected_template_tuple = tuple(selected_templates)
    path_distances = []
    for _, path_ids in by_pair:
        profiles = tuple(
            _path_profile(pool.path_by_id[path_id], context)
            for path_id in path_ids
        )
        if profiles:
            path_distances.append(_mean_pairwise_distance(
                profiles, _path_distance
            ))
    schedule_distances = []
    for _, template_ids in by_path:
        profiles = tuple(
            _schedule_profile(pool.template_by_id[template_id], context)
            for template_id in template_ids
        )
        if profiles:
            schedule_distances.append(_mean_pairwise_distance(
                profiles, _schedule_distance
            ))
    diagnostics = tuple(sorted({
        "selected_paths": float(len(selected_path_tuple)),
        "selected_candidates": float(len(selected_template_tuple)),
        "mean_path_hops": (
            fmean(profile.hops for profile in path_profiles_for_diagnostics)
            if path_profiles_for_diagnostics else 0.0
        ),
        "mean_path_diversity": (
            fmean(path_distances) if path_distances else 0.0
        ),
        "mean_schedule_rounds": (
            fmean(
                profile.round_count
                for profile in schedule_profiles_for_diagnostics
            ) if schedule_profiles_for_diagnostics else 0.0
        ),
        "mean_schedule_memory_time": (
            fmean(
                profile.memory_time
                for profile in schedule_profiles_for_diagnostics
            ) if schedule_profiles_for_diagnostics else 0.0
        ),
        "mean_schedule_diversity": (
            fmean(schedule_distances) if schedule_distances else 0.0
        ),
    }.items()))
    digest = library_digest(selected_template_tuple)
    generator_payload = (
        "con-structural-generator-v1",
        pool.structural_digest,
        preset_value,
        paths_per_pair,
        schedules_per_path,
        tuple(path.path_id for path in selected_path_tuple),
        tuple(template.template_id for template in selected_template_tuple),
    )
    return StructuralLibrarySelection(
        generator_name=preset_value.name,
        path_strategy=preset_value.path_strategy,
        schedule_strategy=preset_value.schedule_strategy,
        selected_path_ids=tuple(path.path_id for path in selected_path_tuple),
        selected_template_ids=tuple(
            template.template_id for template in selected_template_tuple
        ),
        selected_by_pair=tuple(by_pair),
        selected_by_path=tuple(by_path),
        diagnostics=diagnostics,
        structural_digest=digest,
        generator_fingerprint=hashlib.sha256(
            repr(generator_payload).encode("utf-8")
        ).hexdigest(),
    )
