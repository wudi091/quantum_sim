"""Waxman workload adapter for CON's topology-specific offline library."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
import hashlib
from typing import Iterable

from qnet_core.contracts.complete_schedule import (
    CompleteSchedule,
    enumerate_complete_schedules,
)
from qnet_core.order_core import (
    OrderBatchProblem,
    OrderCoreConfig,
    OrderLinkSpec,
    OrderPlan,
    OrderStoredPair,
)
from qnet_core.order_waxman import WaxmanOrderEpisode

from .models import (
    FixedPathLibraryProblem,
    LibraryScheduleTemplate,
    OfflineLibraryMilpResult,
    OfflineLibraryScenario,
)
from .artifact import ConLibrary, PATH_SLOTS, SCHEDULE_SLOTS
from .topology_pool import (
    TopologyTemplatePool,
    build_topology_template_pool,
    canonical_pair,
    canonical_physical_topology_fingerprint,
)


@dataclass(frozen=True)
class WaxmanTemplatePool:
    topology_fingerprint: str
    templates: tuple[LibraryScheduleTemplate, ...]
    path_entries: tuple[tuple[str, tuple[int, ...]], ...]
    structural_digest: str

    @cached_property
    def path_by_id(self) -> dict[str, tuple[int, ...]]:
        return dict(self.path_entries)

    @cached_property
    def template_by_id(self) -> dict[str, LibraryScheduleTemplate]:
        return {
            template.template_id: template for template in self.templates
        }


@dataclass(frozen=True)
class WaxmanPoolSlotProblem:
    """Deterministic planning snapshot plus plan-to-template identity map."""

    problem: OrderBatchProblem
    template_entries: tuple[tuple[str, str], ...]

    @cached_property
    def template_id_by_plan_id(self) -> dict[str, str]:
        return dict(self.template_entries)


def waxman_topology_fingerprint(episode: WaxmanOrderEpisode) -> str:
    return canonical_physical_topology_fingerprint(
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


def build_waxman_topology_pool(
    episode: WaxmanOrderEpisode,
    *,
    path_pool_per_pair: int = 8,
    schedules_per_path_pool: int | None = None,
    max_hops: int | None = None,
) -> TopologyTemplatePool:
    """Build candidates for every unordered topology pair, not trace paths."""

    return build_topology_template_pool(
        nodes=episode.nodes,
        edges=((link.left, link.right) for link in episode.links),
        topology_fingerprint=waxman_topology_fingerprint(episode),
        path_pool_per_pair=path_pool_per_pair,
        schedules_per_path_pool=schedules_per_path_pool,
        max_hops=max_hops,
    )


def compile_waxman_topology_library(
    episode: WaxmanOrderEpisode,
    scenarios=(),
    *,
    generator: str = "hybrid",
    path_pool_per_pair: int = 8,
    schedules_per_path_pool: int | None = None,
    max_hops: int | None = None,
    paths_per_pair: int = 4,
    schedules_per_path: int = 4,
    output_path=None,
):
    """Fixed Waxman topology -> cached CON library.

    With no ``scenarios`` this is the formal request-independent algorithmic
    generator.  Supplying scenarios retains the older scenario-MILP path only
    as an experimental baseline.
    """

    # Local import avoids coupling the low-level pool adapter to compiler load
    # order while keeping the public call a single operation.
    from .compiler import (
        compile_structural_topology_library,
        compile_topology_library,
    )
    from .generators import (
        build_waxman_selection_context,
        select_structural_library,
    )

    pool = build_waxman_topology_pool(
        episode,
        path_pool_per_pair=path_pool_per_pair,
        schedules_per_path_pool=schedules_per_path_pool,
        max_hops=max_hops,
    )
    supplied_scenarios = tuple(scenarios)
    if supplied_scenarios:
        return compile_topology_library(
            pool,
            supplied_scenarios,
            paths_per_pair=paths_per_pair,
            schedules_per_path=schedules_per_path,
            output_path=output_path,
        )
    context = build_waxman_selection_context(
        episode,
        topology_fingerprint=pool.topology_fingerprint,
    )
    selection = select_structural_library(
        pool,
        preset=generator,
        context=context,
        paths_per_pair=paths_per_pair,
        schedules_per_path=schedules_per_path,
    )
    return compile_structural_topology_library(
        pool,
        selection,
        output_path=output_path,
    )


def make_waxman_pool_problem_for_slot(
    episode: WaxmanOrderEpisode,
    pool: TopologyTemplatePool,
    request_ids: Iterable[str],
    slot: int,
    *,
    planning_seed: int = 0,
    initial_inventory: Iterable[OrderStoredPair] = (),
) -> WaxmanPoolSlotProblem:
    """Expose the full offline path/schedule pool to a training scenario.

    The returned snapshot keeps the episode's capacities and timing but sets
    link generation and swaps to probability one, as required by the nominal
    offline planning model.  Hidden stochastic outcomes are therefore never
    used while fitting the reusable library.
    """

    if waxman_topology_fingerprint(episode) != pool.topology_fingerprint:
        raise ValueError("topology pool belongs to a different Waxman episode")
    request_ids = tuple(request_ids)
    if not request_ids:
        raise ValueError("an offline scenario slot needs at least one request")
    request_by_id = episode.request_by_id
    candidates = []
    template_entries = []
    for priority, request_id in enumerate(request_ids):
        try:
            request = request_by_id[request_id]
        except KeyError as exc:
            raise ValueError("offline scenario references an unknown request") from exc
        if not request.arrival_slot <= slot < request.deadline_slot:
            raise ValueError("offline scenario request is inactive in this slot")
        endpoints = canonical_pair(request.source, request.destination)
        pair_id = pool.pair_id_by_endpoints[endpoints]
        pair_paths = pool.paths_by_pair[pair_id]
        if not pair_paths:
            raise ValueError("topology pool has no path for an active request")
        for path_slot, path_candidate in enumerate(pair_paths):
            for schedule_slot, template in enumerate(
                pool.templates_by_path[path_candidate.path_id]
            ):
                schedule = template.schedule
                if schedule.path[0] != request.source:
                    schedule = CompleteSchedule(
                        path=tuple(reversed(schedule.path)),
                        groups=schedule.groups,
                    )
                plan_id = (
                    f"offline:t{slot}:{request_id}:"
                    f"p{path_slot}:o{schedule_slot}"
                )
                candidates.append(OrderPlan(
                    plan_id=plan_id,
                    request_id=request_id,
                    path=schedule.path,
                    swap_order=schedule.swap_order,
                    priority=priority,
                    arrival_slot=request.arrival_slot,
                    deadline_slot=request.deadline_slot,
                    decision_slot=slot,
                    swap_groups=schedule.groups,
                    fixed_path_baseline=(schedule_slot == 0),
                ))
                template_entries.append((plan_id, template.template_id))

    deterministic_links = tuple(
        OrderLinkSpec(
            link.left,
            link.right,
            capacity=link.capacity,
            generation_probability=1.0,
        )
        for link in episode.links
    )
    problem = OrderBatchProblem.create(
        candidates=tuple(candidates),
        node_capacity=episode.capacity,
        links=deterministic_links,
        config=OrderCoreConfig(
            slot_duration_ps=episode.config.slot_duration_ps,
            generation_interval_ps=episode.config.generation_interval_ps,
            swap_service_ps=episode.config.swap_service_ps,
            memory_reset_ps=episode.config.memory_reset_ps,
            generation_probability=1.0,
            swap_probability=1.0,
            edge_capacity=1,
            bsm_capacity_per_node=episode.config.bsm_capacity_per_node,
            epr_ttl_slots=episode.config.epr_ttl_slots,
            seed=int(planning_seed),
            slot_id=int(slot),
        ),
        initial_inventory=tuple(initial_inventory),
        name=f"con-offline-slot{slot}",
    )
    return WaxmanPoolSlotProblem(
        problem=problem,
        template_entries=tuple(template_entries),
    )


def _path_id(topology_fingerprint: str, path: tuple[int, ...]) -> str:
    digest = hashlib.sha256(
        repr((topology_fingerprint, path)).encode("utf-8")
    ).hexdigest()[:20]
    return f"path:{digest}"


def _template_id(
    topology_fingerprint: str,
    path: tuple[int, ...],
    structural_key,
) -> str:
    digest = hashlib.sha256(
        repr((topology_fingerprint, path, structural_key)).encode("utf-8")
    ).hexdigest()[:24]
    return f"schedule:{digest}"


def build_waxman_template_pool(
    episode: WaxmanOrderEpisode,
    *,
    schedules_per_path_pool: int | None = None,
) -> WaxmanTemplatePool:
    """Enumerate a topology-bound structural schedule pool for fixed paths."""

    if schedules_per_path_pool is not None and schedules_per_path_pool < 1:
        raise ValueError("schedules_per_path_pool must be positive or None")
    topology = waxman_topology_fingerprint(episode)
    unique_paths = tuple(sorted({
        path for _, paths in episode.request_paths for path in paths
    }))
    templates: list[LibraryScheduleTemplate] = []
    path_entries: list[tuple[str, tuple[int, ...]]] = []
    for path in unique_paths:
        path_id = _path_id(topology, path)
        path_entries.append((path_id, path))
        schedules = enumerate_complete_schedules(path)
        if schedules_per_path_pool is not None:
            schedules = schedules[:schedules_per_path_pool]
        templates.extend(
            LibraryScheduleTemplate(
                template_id=_template_id(
                    topology, path, schedule.structural_key
                ),
                path_id=path_id,
                schedule=schedule,
            )
            for schedule in schedules
        )
    templates_tuple = tuple(templates)
    structural_digest = hashlib.sha256(repr(tuple(
        (
            template.template_id,
            template.path_id,
            template.structural_key,
        )
        for template in templates_tuple
    )).encode("utf-8")).hexdigest()
    return WaxmanTemplatePool(
        topology_fingerprint=topology,
        templates=templates_tuple,
        path_entries=tuple(path_entries),
        structural_digest=structural_digest,
    )


def make_waxman_library_problem(
    pool: WaxmanTemplatePool,
    scenarios: tuple[OfflineLibraryScenario, ...],
    *,
    schedules_per_path: int = 4,
) -> FixedPathLibraryProblem:
    if any(
        scenario.topology_fingerprint != pool.topology_fingerprint
        for scenario in scenarios
    ):
        raise ValueError("offline scenarios do not match the Waxman topology")
    return FixedPathLibraryProblem(
        templates=pool.templates,
        scenarios=scenarios,
        schedules_per_path=schedules_per_path,
    )


def apply_fitted_library_to_episode(
    episode: WaxmanOrderEpisode,
    pool: WaxmanTemplatePool,
    result: OfflineLibraryMilpResult,
) -> WaxmanOrderEpisode:
    """Install a fitted immutable CON library into a compatible episode."""

    topology = waxman_topology_fingerprint(episode)
    if topology != pool.topology_fingerprint:
        raise ValueError("template pool belongs to a different topology")
    if result.library.topology_fingerprint != topology:
        raise ValueError("fitted library belongs to a different topology")
    template_by_id = pool.template_by_id
    selected = tuple(
        template_by_id[template_id]
        for template_id in result.library.selected_template_ids
    )
    schedules_by_path: dict[tuple[int, ...], list[object]] = {
        path: [] for path in pool.path_by_id.values()
    }
    for template in selected:
        path = pool.path_by_id[template.path_id]
        schedules_by_path[path].append(template.schedule)
    normalized = {
        path: tuple(sorted(
            schedules,
            key=lambda schedule: schedule.structural_key,
        ))
        for path, schedules in schedules_by_path.items()
    }
    return episode.with_schedule_library(
        normalized,
        source="con-offline-scenario-milp",
        structural_digest=result.library.structural_digest,
    )


def instantiate_con_library_for_episode(
    episode: WaxmanOrderEpisode,
    library: ConLibrary,
) -> WaxmanOrderEpisode:
    """Replace request-local paths with topology-cache lookups for online use."""

    topology = waxman_topology_fingerprint(episode)
    if library.topology_fingerprint != topology:
        raise ValueError("CON artifact belongs to a different Waxman topology")

    request_paths = []
    schedules_by_path: dict[tuple[int, ...], tuple[object, ...]] = {}
    for request in episode.requests:
        grid = library.lookup_grid(request.source, request.destination)
        paths = []
        for path_slot in range(PATH_SLOTS):
            row = tuple(
                grid.candidates[path_slot * SCHEDULE_SLOTS + schedule_slot]
                for schedule_slot in range(SCHEDULE_SLOTS)
                if grid.valid_mask[
                    path_slot * SCHEDULE_SLOTS + schedule_slot
                ]
            )
            if not row:
                continue
            path = row[0].path
            schedules = tuple(candidate.schedule for candidate in row)
            previous = schedules_by_path.setdefault(path, schedules)
            if previous != schedules:
                raise ValueError("one oriented path has inconsistent cached schedules")
            paths.append(path)
        if not paths:
            raise ValueError(
                "CON topology cache has no candidate for an episode request"
            )
        request_paths.append((request.request_id, tuple(paths)))

    return replace(
        episode,
        request_paths=tuple(request_paths),
        schedule_library=tuple(sorted(schedules_by_path.items())),
        schedule_library_source="con-offline-artifact-v1",
        schedule_library_digest=library.layout_digest,
    )


def instantiate_topology_pool_for_episode(
    episode: WaxmanOrderEpisode,
    pool: TopologyTemplatePool,
) -> WaxmanOrderEpisode:
    """Install the raw path/schedule pool as a small-scale oracle upper bound."""

    if waxman_topology_fingerprint(episode) != pool.topology_fingerprint:
        raise ValueError("topology pool belongs to a different Waxman episode")
    request_paths = []
    schedules_by_path: dict[tuple[int, ...], tuple[CompleteSchedule, ...]] = {}
    max_paths = 1
    max_schedules = 1
    for request in episode.requests:
        endpoints = canonical_pair(request.source, request.destination)
        pair_id = pool.pair_id_by_endpoints[endpoints]
        oriented_paths = []
        for path_candidate in pool.paths_by_pair[pair_id]:
            path = path_candidate.path
            templates = pool.templates_by_path[path_candidate.path_id]
            schedules = tuple(template.schedule for template in templates)
            if path[0] != request.source:
                path = tuple(reversed(path))
                schedules = tuple(
                    CompleteSchedule(path=path, groups=schedule.groups)
                    for schedule in schedules
                )
            previous = schedules_by_path.setdefault(path, schedules)
            if previous != schedules:
                raise ValueError("raw pool produced inconsistent oriented schedules")
            oriented_paths.append(path)
            max_schedules = max(max_schedules, len(schedules))
        if not oriented_paths:
            raise ValueError("raw topology pool has no path for a request")
        max_paths = max(max_paths, len(oriented_paths))
        request_paths.append((request.request_id, tuple(oriented_paths)))

    pool_config = replace(
        episode.config,
        candidate_paths=max_paths,
        order_variants_per_path=max_schedules,
    )
    return replace(
        episode,
        config=pool_config,
        request_paths=tuple(request_paths),
        schedule_library=tuple(sorted(schedules_by_path.items())),
        schedule_library_source="con-full-topology-pool-oracle",
        schedule_library_digest=pool.structural_digest,
    )
