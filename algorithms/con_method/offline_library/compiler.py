"""End-to-end compiler from a topology pool to the online 4x4 cache."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

from .artifact import (
    CANDIDATE_SLOTS,
    PATH_SLOTS,
    SCHEDULE_SLOTS,
    CachedScheduleCandidate,
    ConLibrary,
    SolverCertificate,
    StoredPairLibrary,
    compute_layout_digest,
)
from .generators import StructuralLibrarySelection
from .milp import solve_topology_schedule_library
from .models import (
    OfflineLibraryMilpResult,
    OfflineLibraryScenario,
    ScenarioConfiguration,
    TopologyLibraryProblem,
)
from .topology_pool import TopologyTemplatePool


COMPILER_MODEL_VERSION = "con-path-schedule-scenario-milp-v1"


@dataclass(frozen=True)
class ConCompilationResult:
    milp_result: OfflineLibraryMilpResult
    library: ConLibrary


@dataclass(frozen=True)
class StructuralConCompilationResult:
    selection: StructuralLibrarySelection
    library: ConLibrary


def _structural_fallback_scenario(
    pool: TopologyTemplatePool,
) -> OfflineLibraryScenario:
    """Zero-information scenario used for a topology-only deterministic cache."""

    return OfflineLibraryScenario(
        scenario_id="topology-only-selection",
        trace_digest=f"topology-only:{pool.structural_digest}",
        topology_fingerprint=pool.topology_fingerprint,
        request_distribution_fingerprint="requests:none-topology-only-v1",
        physics_fingerprint="physics:none-topology-only-v1",
        configurations=(ScenarioConfiguration("empty"),),
    )


def compiler_fingerprint(
    pool: TopologyTemplatePool,
    scenarios: Iterable[OfflineLibraryScenario],
    *,
    paths_per_pair: int,
    schedules_per_path: int,
) -> str:
    scenarios = tuple(sorted(scenarios, key=lambda value: value.scenario_id))
    payload = (
        COMPILER_MODEL_VERSION,
        pool.structural_digest,
        paths_per_pair,
        schedules_per_path,
        tuple(
            (
                scenario.scenario_id,
                scenario.trace_digest,
                scenario.request_distribution_fingerprint,
                scenario.physics_fingerprint,
                scenario.weight,
            )
            for scenario in scenarios
        ),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def make_topology_library_problem(
    pool: TopologyTemplatePool,
    scenarios: Iterable[OfflineLibraryScenario],
    *,
    paths_per_pair: int = PATH_SLOTS,
    schedules_per_path: int = SCHEDULE_SLOTS,
) -> TopologyLibraryProblem:
    scenarios = tuple(scenarios)
    if not 1 <= paths_per_pair <= PATH_SLOTS:
        raise ValueError("paths_per_pair must lie in [1, 4]")
    if not 1 <= schedules_per_path <= SCHEDULE_SLOTS:
        raise ValueError("schedules_per_path must lie in [1, 4]")
    if any(
        scenario.topology_fingerprint != pool.topology_fingerprint
        for scenario in scenarios
    ):
        raise ValueError("offline scenarios do not match the topology pool")
    return TopologyLibraryProblem(
        paths=pool.paths,
        templates=pool.templates,
        scenarios=scenarios,
        paths_per_pair=paths_per_pair,
        schedules_per_path=schedules_per_path,
    )


def build_con_library_artifact(
    pool: TopologyTemplatePool,
    result: OfflineLibraryMilpResult,
    *,
    compiler_digest: str,
    selection_mode: str,
    paths_per_pair: int = PATH_SLOTS,
    schedules_per_path: int = SCHEDULE_SLOTS,
) -> ConLibrary:
    if result.library.topology_fingerprint != pool.topology_fingerprint:
        raise ValueError("MILP result and topology pool do not match")
    if not compiler_digest or not selection_mode:
        raise ValueError("compiler metadata must be non-empty")

    pair_entries = _build_pair_entries(
        pool,
        selected_path_ids=result.library.selected_path_ids,
        selected_template_ids=result.library.selected_template_ids,
    )
    return ConLibrary(
        topology_fingerprint=pool.topology_fingerprint,
        pool_structural_digest=pool.structural_digest,
        library_structural_digest=result.library.structural_digest,
        layout_digest=compute_layout_digest(pair_entries),
        compiler_fingerprint=compiler_digest,
        request_distribution_fingerprint=(
            result.library.request_distribution_fingerprint
        ),
        physics_fingerprint=result.library.physics_fingerprint,
        training_scenario_ids=result.library.training_scenario_ids,
        training_trace_digests=result.library.training_trace_digests,
        pair_entries=pair_entries,
        solver_certificate=SolverCertificate(
            solver="scipy-highs",
            status="optimal",
            objective=float(result.training_weighted_completed),
            mip_gap=float(result.solver_mip_gap),
        ),
        selection_mode=selection_mode,
        paths_per_pair=PATH_SLOTS,
        schedules_per_path=SCHEDULE_SLOTS,
    )


def _build_pair_entries(
    pool: TopologyTemplatePool,
    *,
    selected_path_ids: Iterable[str],
    selected_template_ids: Iterable[str],
) -> tuple[StoredPairLibrary, ...]:
    selected_path_ids = frozenset(selected_path_ids)
    selected_template_ids = frozenset(selected_template_ids)
    unknown_paths = selected_path_ids - set(pool.path_by_id)
    unknown_templates = selected_template_ids - set(pool.template_by_id)
    if unknown_paths or unknown_templates:
        raise ValueError("selection refers to candidates outside the topology pool")
    for template_id in selected_template_ids:
        if pool.template_by_id[template_id].path_id not in selected_path_ids:
            raise ValueError("a selected schedule belongs to an unselected path")
    entries = []
    for pair_id, endpoints in pool.pair_entries:
        selected_paths = tuple(
            path for path in pool.paths_by_pair[pair_id]
            if path.path_id in selected_path_ids
        )
        if len(selected_paths) > PATH_SLOTS:
            raise ValueError("MILP selected more than four paths for one pair")
        slots: list[CachedScheduleCandidate | None] = []
        for path in selected_paths:
            selected_templates = tuple(
                template for template in pool.templates_by_path[path.path_id]
                if template.template_id in selected_template_ids
            )
            if len(selected_templates) > SCHEDULE_SLOTS:
                raise ValueError("MILP selected more than four schedules for one path")
            slots.extend(
                CachedScheduleCandidate(
                    pair_id=pair_id,
                    path_id=path.path_id,
                    template_id=template.template_id,
                    schedule=template.schedule,
                )
                for template in selected_templates
            )
            slots.extend([None] * (SCHEDULE_SLOTS - len(selected_templates)))
        slots.extend(
            [None] * ((PATH_SLOTS - len(selected_paths)) * SCHEDULE_SLOTS)
        )
        if len(slots) != CANDIDATE_SLOTS:
            raise RuntimeError("compiler failed to construct a 16-slot grid")
        entries.append(StoredPairLibrary(
            pair_id=pair_id,
            endpoints=endpoints,
            candidates=tuple(slots),
        ))

    return tuple(entries)


def build_structural_con_library_artifact(
    pool: TopologyTemplatePool,
    selection: StructuralLibrarySelection,
) -> ConLibrary:
    """Compile a request-independent algorithmic selection into 16-slot grids."""

    pair_entries = _build_pair_entries(
        pool,
        selected_path_ids=selection.selected_path_ids,
        selected_template_ids=selection.selected_template_ids,
    )
    return ConLibrary(
        topology_fingerprint=pool.topology_fingerprint,
        pool_structural_digest=pool.structural_digest,
        library_structural_digest=selection.structural_digest,
        layout_digest=compute_layout_digest(pair_entries),
        compiler_fingerprint=selection.generator_fingerprint,
        request_distribution_fingerprint="requests:not-observed-v1",
        physics_fingerprint="topology-resource-profile-v1",
        training_scenario_ids=(),
        training_trace_digests=(),
        pair_entries=pair_entries,
        solver_certificate=SolverCertificate(
            solver=f"algorithm:{selection.generator_name}",
            status="deterministic-complete",
            objective=0.0,
            mip_gap=0.0,
        ),
        selection_mode=f"topology-only:{selection.generator_name}",
        paths_per_pair=PATH_SLOTS,
        schedules_per_path=SCHEDULE_SLOTS,
    )


def compile_structural_topology_library(
    pool: TopologyTemplatePool,
    selection: StructuralLibrarySelection,
    *,
    output_path: str | Path | None = None,
) -> StructuralConCompilationResult:
    artifact = build_structural_con_library_artifact(pool, selection)
    if output_path is not None:
        artifact.save(output_path)
    return StructuralConCompilationResult(selection=selection, library=artifact)


def compile_topology_library(
    pool: TopologyTemplatePool,
    scenarios: Iterable[OfflineLibraryScenario] = (),
    *,
    paths_per_pair: int = PATH_SLOTS,
    schedules_per_path: int = SCHEDULE_SLOTS,
    output_path: str | Path | None = None,
) -> ConCompilationResult:
    """Fit, validate, optionally save, and return a topology-wide cache.

    Passing real scenarios performs the paper's scenario-MILP portfolio fit.
    Passing no scenarios is useful for smoke tests and produces a deterministic
    shortest/canonical structural cache; its metadata is explicitly labelled so
    it cannot be mistaken for a scenario-trained library.
    """

    supplied_scenarios = tuple(scenarios)
    selection_mode = (
        "scenario-milp" if supplied_scenarios else "topology-only-deterministic"
    )
    fit_scenarios = supplied_scenarios or (_structural_fallback_scenario(pool),)
    problem = make_topology_library_problem(
        pool,
        fit_scenarios,
        paths_per_pair=paths_per_pair,
        schedules_per_path=schedules_per_path,
    )
    result = solve_topology_schedule_library(problem)
    digest = compiler_fingerprint(
        pool,
        fit_scenarios,
        paths_per_pair=paths_per_pair,
        schedules_per_path=schedules_per_path,
    )
    artifact = build_con_library_artifact(
        pool,
        result,
        compiler_digest=digest,
        selection_mode=selection_mode,
        paths_per_pair=paths_per_pair,
        schedules_per_path=schedules_per_path,
    )
    if output_path is not None:
        artifact.save(output_path)
    return ConCompilationResult(milp_result=result, library=artifact)
