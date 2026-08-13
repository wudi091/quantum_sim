"""End-to-end batch generation for TELGEN-style LP teacher records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Iterable, Mapping

from qnet_core.construction_catalog import (
    RouteConstructionCandidate,
    build_route_construction_catalogue,
)
from qnet_core.resource_catalog import build_resource_capacities
from qnet_core.scenario import ScenarioConfig, make_episode
from qnet_core.spec import EpisodeSpec

from .teacher import (
    ConstructionAwareLPTeacher,
    TeacherSolution,
    save_teacher_solution,
)
from .fidelity import (
    FIDELITY_MODEL_NAME,
    candidate_fidelity_estimate_map,
)
from .success_probability import (
    SUCCESS_PROBABILITY_MODEL_NAME,
    candidate_success_probability_map,
)
from .time_expansion import (
    TimeExpansionResult,
    build_nominal_schedule,
    expand_construction_candidates,
)


@dataclass(frozen=True)
class TeacherBatchRecord:
    """One reproducible batch, its LP, and both interior-point trajectories."""

    episode: EpisodeSpec
    path_candidate_count: int
    construction_kinds: tuple[str, ...]
    swap_tree_count: int | None
    purification_kinds: tuple[str, ...]
    resource_capacities: tuple[tuple[str, int], ...]
    candidates: tuple[RouteConstructionCandidate, ...]
    expansion: TimeExpansionResult
    solution: TeacherSolution
    solve_seconds: float
    fidelity_model: str
    success_probability_model: str
    load_profile: str | None = None
    planning_window: tuple[int, int] | None = None
    completion_end_slot: int | None = None
    reserved_usage: tuple[tuple[str, int, int], ...] = ()
    equivalent_candidate_aliases: tuple[tuple[str, str], ...] = ()

    @property
    def capacities(self) -> dict[str, int]:
        return dict(self.resource_capacities)

    def context(self) -> dict[str, object]:
        """Return JSON-compatible provenance stored beside the LP arrays."""

        return {
            "schema_version": 1,
            "episode": {
                "seed": self.episode.seed,
                "nodes": list(self.episode.nodes),
                "edges": [list(edge) for edge in self.episode.edges],
                "requests": [asdict(request) for request in self.episode.requests],
                "horizon": self.episode.horizon,
                "physical": asdict(self.episode.physical),
            },
            "catalogue": {
                "path_candidate_count": self.path_candidate_count,
                "construction_kinds": list(self.construction_kinds),
                "swap_tree_count": self.swap_tree_count,
                "purification_kinds": list(self.purification_kinds),
                "candidate_count": len(self.candidates),
                "fidelity_model": self.fidelity_model,
                "success_probability_model": self.success_probability_model,
            },
            "time_expansion": {
                "variable_count": len(self.expansion.variables),
                "rejections": [asdict(item) for item in self.expansion.rejections],
                "equivalent_candidate_aliases": [
                    {
                        "alias_candidate_id": alias,
                        "representative_candidate_id": representative,
                    }
                    for alias, representative in self.equivalent_candidate_aliases
                ],
            },
            "resource_capacities": dict(self.resource_capacities),
            "reserved_usage": [
                {
                    "resource_id": resource_id,
                    "slot": slot,
                    "amount": amount,
                }
                for resource_id, slot, amount in self.reserved_usage
            ],
            "planning_window": (
                None
                if self.planning_window is None
                else list(self.planning_window)
            ),
            "completion_end_slot": self.completion_end_slot,
            "solve_seconds": self.solve_seconds,
            "load_profile": self.load_profile,
        }


@dataclass(frozen=True)
class PlanningBatchProblem:
    """One simulator-neutral candidate expansion before choosing a solver."""

    episode: EpisodeSpec
    path_candidate_count: int
    construction_kinds: tuple[str, ...]
    swap_tree_count: int | None
    purification_kinds: tuple[str, ...]
    resource_capacities: tuple[tuple[str, int], ...]
    candidates: tuple[RouteConstructionCandidate, ...]
    expansion: TimeExpansionResult
    fidelity_model: str
    success_probability_model: str
    planning_window: tuple[int, int] | None = None
    completion_end_slot: int | None = None
    reserved_usage: tuple[tuple[str, int, int], ...] = ()
    equivalent_candidate_aliases: tuple[tuple[str, str], ...] = ()

    @property
    def capacities(self) -> dict[str, int]:
        return dict(self.resource_capacities)

    @property
    def reserved_usage_map(self) -> dict[tuple[str, int], int]:
        return {
            (resource_id, slot): amount
            for resource_id, slot, amount in self.reserved_usage
        }


@dataclass(frozen=True)
class TeacherDatasetEntry:
    seed: int
    file: str
    request_count: int
    candidate_count: int
    variable_count: int
    rejection_count: int
    completed_request_mass: float
    total_completion_latency: float
    stage_one_iterations: int
    stage_two_iterations: int
    solve_seconds: float


@dataclass(frozen=True)
class TeacherDatasetResult:
    manifest_path: Path
    entries: tuple[TeacherDatasetEntry, ...]


def _candidate_dag_semantics(
    candidate: RouteConstructionCandidate,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Canonicalize DAG dataflow while ignoring IDs and redundant edges.

    Operation and segment identifiers are catalogue-local names, so a pure
    alpha-renaming must remain equivalent.  Explicit predecessor edges that
    are already implied transitively (or by an input segment's producer) are
    also execution-neutral.  All other operation physics, dataflow, terminal
    outputs, and retry lineage remain part of the key.
    """

    operations = candidate.dag.operations
    ordinal_counts: dict[int, int] = {}
    for operation in operations:
        ordinal_counts[operation.ordinal] = (
            ordinal_counts.get(operation.ordinal, 0) + 1
        )
    operation_tokens = {
        operation.op_id: (
            "ordinal",
            operation.ordinal,
        ) if ordinal_counts[operation.ordinal] == 1 else (
            # Ambiguous ordinals are uncommon in catalogue DAGs.  Retaining
            # the ID in that case is conservative: it can only under-merge,
            # never collapse physically distinct actions.
            "ambiguous_ordinal",
            operation.ordinal,
            operation.op_id,
        )
        for operation in operations
    }
    producer_by_segment = {
        operation.output_segment_id: operation.op_id
        for operation in operations
        if operation.output_segment_id is not None
    }

    direct_dependencies: dict[str, set[str]] = {}
    for operation in operations:
        dependencies = set(operation.predecessors)
        dependencies.update(
            producer_by_segment[segment_id]
            for segment_id in operation.input_segment_ids
            if segment_id in producer_by_segment
        )
        direct_dependencies[operation.op_id] = dependencies

    ancestor_cache: dict[str, set[str]] = {}

    def ancestors(operation_id: str) -> set[str]:
        cached = ancestor_cache.get(operation_id)
        if cached is not None:
            return cached
        result: set[str] = set()
        for predecessor in direct_dependencies[operation_id]:
            result.add(predecessor)
            result.update(ancestors(predecessor))
        ancestor_cache[operation_id] = result
        return result

    def reduced_dependencies(operation_id: str) -> tuple[object, ...]:
        dependencies = direct_dependencies[operation_id]
        reduced = {
            predecessor
            for predecessor in dependencies
            if not any(
                predecessor in ancestors(other)
                for other in dependencies
                if other != predecessor
            )
        }
        return tuple(sorted(operation_tokens[item] for item in reduced))

    def segment_token(segment_id: str) -> tuple[object, ...]:
        producer = producer_by_segment.get(segment_id)
        if producer is None:
            return ("external_segment", segment_id)
        return ("produced_segment", operation_tokens[producer])

    operation_semantics = tuple(sorted(
        (
            operation_tokens[operation.op_id],
            operation.kind,
            reduced_dependencies(operation.op_id),
            tuple(
                segment_token(segment_id)
                for segment_id in operation.input_segment_ids
            ),
            operation.output_segment_id is not None,
            operation.output_endpoints,
            operation.resource_demand.entries,
            operation.output_resource_hold.entries,
            operation.duration_ps,
            float(operation.success_probability).hex(),
            float(operation.required_fidelity).hex(),
            operation.retry_limit,
            (
                None
                if operation.retry_root_id is None
                else operation_tokens.get(
                    operation.retry_root_id,
                    ("external_operation", operation.retry_root_id),
                )
            ),
            operation.retry_attempt,
            operation.dag_version,
        )
        for operation in operations
    ))
    terminal_semantics = tuple(
        segment_token(segment_id)
        for segment_id in candidate.all_terminal_segment_ids
    )
    return operation_semantics, terminal_semantics


def _canonicalize_planning_equivalent_candidates(
    candidates: tuple[RouteConstructionCandidate, ...],
    fidelity_estimates: Mapping[str, float],
    success_probability_estimates: Mapping[str, float],
) -> tuple[
    tuple[RouteConstructionCandidate, ...],
    tuple[tuple[str, str], ...],
]:
    """Remove aliases that induce the exact same neutral planning action.

    A MILP cannot distinguish two candidates with the same request, route,
    relative resource--time footprint, completion duration, fidelity, and
    success probability.  Keeping both creates arbitrary one-hot labels for
    otherwise identical GNN nodes.  The lexicographically first candidate ID
    is retained as a deterministic representative.

    Distinct swap trees are preserved whenever their resource--time behavior
    differs, even if they have the same duration and objective coefficients.
    """

    representatives: dict[tuple[object, ...], RouteConstructionCandidate] = {}
    aliases: list[tuple[str, str]] = []
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        schedule = build_nominal_schedule(candidate)
        operation_semantics, terminal_semantics = _candidate_dag_semantics(
            candidate
        )
        key = (
            candidate.request_id,
            candidate.route_nodes,
            candidate.purification_kind,
            candidate.demand_pairs,
            candidate.dag.version,
            operation_semantics,
            terminal_semantics,
            schedule.duration_slots,
            schedule.resource_usage,
            float(fidelity_estimates[candidate.candidate_id]).hex(),
            float(success_probability_estimates[candidate.candidate_id]).hex(),
        )
        representative = representatives.get(key)
        if representative is None:
            representatives[key] = candidate
            continue
        aliases.append((candidate.candidate_id, representative.candidate_id))
    return (
        tuple(sorted(representatives.values(), key=lambda item: item.candidate_id)),
        tuple(sorted(aliases)),
    )


def build_teacher_batch_record(
    scenario: ScenarioConfig,
    seed: int,
    *,
    path_candidate_count: int = 3,
    construction_kinds: tuple[str, ...] = ("left_deep", "balanced"),
    swap_tree_count: int | None = None,
    purification_kinds: tuple[str, ...] = ("none", "elementary_once"),
    fidelity_estimates: Mapping[str, float] | None = None,
    teacher: ConstructionAwareLPTeacher | None = None,
) -> TeacherBatchRecord:
    """Generate and solve one static, simultaneous-arrival request batch."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    if path_candidate_count < 1:
        raise ValueError("path_candidate_count must be positive")
    if not construction_kinds and swap_tree_count is None:
        raise ValueError("at least one construction policy is required")

    episode = make_episode(scenario, seed)
    return solve_teacher_episode(
        episode,
        path_candidate_count=path_candidate_count,
        construction_kinds=construction_kinds,
        swap_tree_count=swap_tree_count,
        purification_kinds=purification_kinds,
        fidelity_estimates=fidelity_estimates,
        teacher=teacher,
    )


def solve_teacher_episode(
    episode: EpisodeSpec,
    *,
    path_candidate_count: int = 3,
    construction_kinds: tuple[str, ...] = ("left_deep", "balanced"),
    swap_tree_count: int | None = None,
    purification_kinds: tuple[str, ...] = ("none", "elementary_once"),
    fidelity_estimates: Mapping[str, float] | None = None,
    teacher: ConstructionAwareLPTeacher | None = None,
    load_profile: str | None = None,
    resource_capacities: Mapping[str, int] | None = None,
    reserved_usage: Mapping[tuple[str, int], int] | None = None,
    window_start_slot: int | None = None,
    window_end_slot: int | None = None,
    completion_end_slot: int | None = None,
) -> TeacherBatchRecord:
    """Solve one episode with the static teacher or an online time window.

    With no window arguments this retains the original simultaneous-arrival
    static semantics.  Supplying a window enables online replanning over the
    same candidate expansion, LP teacher, and resource catalogue.
    """

    problem = build_planning_batch_problem(
        episode,
        path_candidate_count=path_candidate_count,
        construction_kinds=construction_kinds,
        swap_tree_count=swap_tree_count,
        purification_kinds=purification_kinds,
        fidelity_estimates=fidelity_estimates,
        resource_capacities=resource_capacities,
        reserved_usage=reserved_usage,
        window_start_slot=window_start_slot,
        window_end_slot=window_end_slot,
        completion_end_slot=completion_end_slot,
    )
    return solve_planning_batch_problem(
        problem,
        teacher=teacher,
        load_profile=load_profile,
    )


def build_planning_batch_problem(
    episode: EpisodeSpec,
    *,
    path_candidate_count: int = 3,
    construction_kinds: tuple[str, ...] = ("left_deep", "balanced"),
    swap_tree_count: int | None = None,
    purification_kinds: tuple[str, ...] = ("none", "elementary_once"),
    fidelity_estimates: Mapping[str, float] | None = None,
    resource_capacities: Mapping[str, int] | None = None,
    reserved_usage: Mapping[tuple[str, int], int] | None = None,
    window_start_slot: int | None = None,
    window_end_slot: int | None = None,
    completion_end_slot: int | None = None,
) -> PlanningBatchProblem:
    """Build the shared candidate/constraint input used by LP and MILP."""

    if path_candidate_count < 1:
        raise ValueError("path_candidate_count must be positive")
    if not construction_kinds and swap_tree_count is None:
        raise ValueError("at least one construction policy is required")
    if swap_tree_count is not None and swap_tree_count < 1:
        raise ValueError("swap_tree_count must be positive")
    if not purification_kinds:
        raise ValueError("at least one purification kind is required")
    online_window = window_start_slot is not None or window_end_slot is not None
    if online_window:
        window_start = 0 if window_start_slot is None else int(window_start_slot)
        window_end = (
            episode.horizon
            if window_end_slot is None
            else int(window_end_slot)
        )
        if not 0 <= window_start < window_end <= episode.horizon:
            raise ValueError("planning window must lie inside the episode horizon")
        completion_end = (
            window_end
            if completion_end_slot is None
            else int(completion_end_slot)
        )
        if not window_end <= completion_end <= episode.horizon:
            raise ValueError(
                "completion boundary must follow the planning window"
            )
        future = [
            request.id
            for request in episode.requests
            if request.arrival > window_start
        ]
        if future:
            raise ValueError(
                f"online teacher window contains a future request: {future[0]}"
            )
        planning_window = (window_start, window_end)
    else:
        if completion_end_slot is not None:
            raise ValueError(
                "completion_end_slot requires a planning window"
            )
        if any(request.arrival != 0 for request in episode.requests):
            raise ValueError("static teacher episodes require arrival slot zero")
        planning_window = None

    capacities = (
        build_resource_capacities(episode)
        if resource_capacities is None
        else {str(key): int(value) for key, value in resource_capacities.items()}
    )
    raw_candidates = build_route_construction_catalogue(
        episode.planning,
        candidate_count=path_candidate_count,
        construction_kinds=construction_kinds,
        purification_kinds=purification_kinds,
        swap_tree_count=swap_tree_count,
    )
    resolved_fidelity_estimates = fidelity_estimates
    fidelity_model = "provided"
    if resolved_fidelity_estimates is None:
        resolved_fidelity_estimates = candidate_fidelity_estimate_map(
            episode,
            raw_candidates,
        )
        fidelity_model = FIDELITY_MODEL_NAME
    else:
        resolved_fidelity_estimates = {
            str(candidate_id): float(value)
            for candidate_id, value in resolved_fidelity_estimates.items()
        }
    missing_fidelity = {
        candidate.candidate_id for candidate in raw_candidates
    } - set(resolved_fidelity_estimates)
    if missing_fidelity:
        raise ValueError(
            f"missing fidelity estimate: {sorted(missing_fidelity)[0]}"
        )
    success_probability_estimates = candidate_success_probability_map(
        episode,
        raw_candidates,
    )
    candidates, equivalent_aliases = (
        _canonicalize_planning_equivalent_candidates(
            raw_candidates,
            resolved_fidelity_estimates,
            success_probability_estimates,
        )
    )
    expansion = expand_construction_candidates(
        episode.planning,
        candidates,
        capacities,
        fidelity_estimates=resolved_fidelity_estimates,
        success_probability_estimates=success_probability_estimates,
        reserved_usage=reserved_usage,
        window_start_slot=(None if planning_window is None else planning_window[0]),
        window_end_slot=(None if planning_window is None else planning_window[1]),
        completion_end_slot=(
            None if planning_window is None else completion_end
        ),
    )
    return PlanningBatchProblem(
        episode=episode,
        path_candidate_count=path_candidate_count,
        construction_kinds=tuple(construction_kinds),
        swap_tree_count=swap_tree_count,
        purification_kinds=tuple(purification_kinds),
        resource_capacities=tuple(sorted(capacities.items())),
        candidates=candidates,
        expansion=expansion,
        fidelity_model=fidelity_model,
        success_probability_model=SUCCESS_PROBABILITY_MODEL_NAME,
        planning_window=planning_window,
        completion_end_slot=(
            None if planning_window is None else completion_end
        ),
        reserved_usage=tuple(sorted(
            (str(resource_id), int(slot), int(amount))
            for (resource_id, slot), amount in (reserved_usage or {}).items()
            if int(amount) != 0
        )),
        equivalent_candidate_aliases=equivalent_aliases,
    )


def solve_planning_batch_problem(
    problem: PlanningBatchProblem,
    *,
    teacher: ConstructionAwareLPTeacher | None = None,
    load_profile: str | None = None,
) -> TeacherBatchRecord:
    """Solve an already-built planning problem with the continuous teacher."""

    solver = teacher if teacher is not None else ConstructionAwareLPTeacher()
    started = perf_counter()
    solution = solver.solve(
        problem.expansion,
        problem.capacities,
        reserved_usage=problem.reserved_usage_map,
    )
    solve_seconds = perf_counter() - started
    return TeacherBatchRecord(
        episode=problem.episode,
        path_candidate_count=problem.path_candidate_count,
        construction_kinds=problem.construction_kinds,
        swap_tree_count=problem.swap_tree_count,
        purification_kinds=problem.purification_kinds,
        resource_capacities=problem.resource_capacities,
        candidates=problem.candidates,
        expansion=problem.expansion,
        solution=solution,
        solve_seconds=solve_seconds,
        fidelity_model=problem.fidelity_model,
        success_probability_model=problem.success_probability_model,
        load_profile=load_profile,
        planning_window=problem.planning_window,
        completion_end_slot=problem.completion_end_slot,
        reserved_usage=problem.reserved_usage,
        equivalent_candidate_aliases=problem.equivalent_candidate_aliases,
    )


def solve_teacher_window(
    episode: EpisodeSpec,
    *,
    window_start_slot: int,
    window_end_slot: int,
    completion_end_slot: int | None = None,
    reserved_usage: Mapping[tuple[str, int], int] | None = None,
    resource_capacities: Mapping[str, int] | None = None,
    path_candidate_count: int = 3,
    construction_kinds: tuple[str, ...] = ("left_deep", "balanced"),
    swap_tree_count: int | None = None,
    purification_kinds: tuple[str, ...] = ("none", "elementary_once"),
    fidelity_estimates: Mapping[str, float] | None = None,
    teacher: ConstructionAwareLPTeacher | None = None,
) -> TeacherBatchRecord:
    """Online-window entry point backed by the unchanged TELGEN teacher."""

    return solve_teacher_episode(
        episode,
        path_candidate_count=path_candidate_count,
        construction_kinds=construction_kinds,
        swap_tree_count=swap_tree_count,
        purification_kinds=purification_kinds,
        fidelity_estimates=fidelity_estimates,
        teacher=teacher,
        resource_capacities=resource_capacities,
        reserved_usage=reserved_usage,
        window_start_slot=window_start_slot,
        window_end_slot=window_end_slot,
        completion_end_slot=completion_end_slot,
    )


def save_teacher_batch_record(
    record: TeacherBatchRecord,
    path: str | Path,
) -> Path:
    """Save a self-contained teacher sample as one compressed NPZ file."""

    return save_teacher_solution(
        record.solution,
        path,
        context=record.context(),
    )


def _entry_for(record: TeacherBatchRecord, file_name: str) -> TeacherDatasetEntry:
    solution = record.solution
    return TeacherDatasetEntry(
        seed=record.episode.seed,
        file=file_name,
        request_count=len(record.episode.requests),
        candidate_count=len(record.candidates),
        variable_count=len(record.expansion.variables),
        rejection_count=len(record.expansion.rejections),
        completed_request_mass=solution.completed_request_mass,
        total_completion_latency=solution.total_completion_latency,
        stage_one_iterations=solution.stage_one.iterations,
        stage_two_iterations=solution.stage_two.iterations,
        solve_seconds=record.solve_seconds,
    )


def generate_teacher_dataset(
    scenario: ScenarioConfig,
    seeds: Iterable[int],
    output_directory: str | Path,
    *,
    path_candidate_count: int = 3,
    construction_kinds: tuple[str, ...] = ("left_deep", "balanced"),
    swap_tree_count: int | None = None,
    purification_kinds: tuple[str, ...] = ("none", "elementary_once"),
    overwrite: bool = False,
) -> TeacherDatasetResult:
    """Solve several batches and write NPZ samples plus one JSON manifest."""

    ordered_seeds = tuple(int(seed) for seed in seeds)
    if not ordered_seeds:
        raise ValueError("at least one seed is required")
    if any(seed < 0 for seed in ordered_seeds):
        raise ValueError("seeds must be non-negative")
    if len(set(ordered_seeds)) != len(ordered_seeds):
        raise ValueError("dataset seeds must be unique")

    output = Path(output_directory)
    targets = {
        seed: output / f"teacher_seed_{seed:08d}.npz"
        for seed in ordered_seeds
    }
    manifest_path = output / "manifest.json"
    if not overwrite:
        existing = [path for path in (*targets.values(), manifest_path) if path.exists()]
        if existing:
            raise FileExistsError(f"teacher dataset output already exists: {existing[0]}")

    output.mkdir(parents=True, exist_ok=True)
    entries: list[TeacherDatasetEntry] = []
    for seed in ordered_seeds:
        record = build_teacher_batch_record(
            scenario,
            seed,
            path_candidate_count=path_candidate_count,
            construction_kinds=construction_kinds,
            swap_tree_count=swap_tree_count,
            purification_kinds=purification_kinds,
        )
        target = save_teacher_batch_record(record, targets[seed])
        entries.append(_entry_for(record, target.name))

    payload = {
        "schema_version": 1,
        "scenario": asdict(scenario),
        "path_candidate_count": path_candidate_count,
        "construction_kinds": list(construction_kinds),
        "swap_tree_count": swap_tree_count,
        "purification_kinds": list(purification_kinds),
        "fidelity_model": FIDELITY_MODEL_NAME,
        "records": [asdict(entry) for entry in entries],
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return TeacherDatasetResult(manifest_path, tuple(entries))
