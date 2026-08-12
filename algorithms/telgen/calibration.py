"""Static batch load profiles and LP-teacher calibration statistics."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np

from qnet_core.scenario import ScenarioConfig, make_episode
from qnet_core.spec import EpisodeSpec

from .dataset import (
    TeacherBatchRecord,
    save_teacher_batch_record,
    solve_teacher_episode,
)


@dataclass(frozen=True)
class StaticLoadProfile:
    """One offered-load level on a shared topology and request pool."""

    name: str
    request_count: int
    horizon: int

    def __post_init__(self) -> None:
        if re.fullmatch(r"[A-Za-z0-9_-]+", self.name) is None:
            raise ValueError("load profile name must be path-safe")
        if self.request_count < 1:
            raise ValueError("load profile request_count must be positive")
        if self.horizon < 1:
            raise ValueError("load profile horizon must be positive")


DEFAULT_STATIC_LOAD_PROFILES = (
    StaticLoadProfile("light", request_count=8, horizon=12),
    StaticLoadProfile("medium", request_count=24, horizon=6),
    StaticLoadProfile("heavy", request_count=40, horizon=5),
)


@dataclass(frozen=True)
class TeacherBatchStatistics:
    request_count: int
    candidate_count: int
    variable_count: int
    constraint_count: int
    rejected_candidate_count: int
    completed_request_mass: float
    completion_ratio: float
    average_completion_latency: float
    active_variable_count: int
    fractional_variable_count: int
    fractional_request_count: int
    fractional_request_ratio: float
    mean_resource_utilization: float
    peak_resource_utilization: float
    tight_resource_constraint_ratio: float
    max_constraint_violation: float
    stage_one_iterations: int
    stage_two_iterations: int
    solve_seconds: float


@dataclass(frozen=True)
class StaticLoadCalibrationEntry:
    load_profile: str
    seed: int
    file: str
    statistics: TeacherBatchStatistics

    def flat_dict(self) -> dict[str, object]:
        return {
            "load_profile": self.load_profile,
            "seed": self.seed,
            "file": self.file,
            **asdict(self.statistics),
        }


@dataclass(frozen=True)
class StaticLoadAggregate:
    load_profile: str
    sample_count: int
    mean_completion_ratio: float
    min_completion_ratio: float
    max_completion_ratio: float
    mean_average_completion_latency: float
    mean_fractional_request_ratio: float
    mean_peak_resource_utilization: float
    mean_solve_seconds: float


@dataclass(frozen=True)
class StaticLoadCalibrationResult:
    manifest_path: Path
    csv_path: Path
    entries: tuple[StaticLoadCalibrationEntry, ...]
    aggregates: tuple[StaticLoadAggregate, ...]


def summarize_teacher_batch(
    record: TeacherBatchRecord,
    *,
    tolerance: float = 1e-7,
) -> TeacherBatchStatistics:
    """Measure throughput, fractionality, utilization, and solver health."""

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    solution = record.solution
    values = np.asarray(solution.stage_two.primal, dtype=float)
    active_variable_count = int(np.sum(values > tolerance))
    fractional_variable_count = int(np.sum(
        (values > tolerance) & (values < 1.0 - tolerance)
    ))

    values_by_request: dict[str, list[float]] = {
        request.id: [] for request in record.episode.requests
    }
    for variable, value in zip(solution.variables, values):
        values_by_request[variable.request_id].append(float(value))
    fractional_request_count = 0
    for request_values in values_by_request.values():
        active = [value for value in request_values if value > tolerance]
        is_integral_selection = (
            len(active) == 1 and abs(active[0] - 1.0) <= tolerance
        )
        if active and not is_integral_selection:
            fractional_request_count += 1

    resource_utilizations: list[float] = []
    for row_index, (rhs, descriptor) in enumerate(zip(
        solution.stage_two_lp.b_ub,
        solution.stage_two_lp.ub_constraints,
    )):
        if descriptor.kind != "resource_time" or rhs <= 0:
            continue
        activity = solution.stage_two_lp.a_ub.getrow(row_index) @ values
        resource_utilizations.append(float(np.asarray(activity).item() / rhs))
    mean_utilization = (
        float(np.mean(resource_utilizations)) if resource_utilizations else 0.0
    )
    peak_utilization = max(resource_utilizations, default=0.0)
    if 1.0 < peak_utilization <= 1.0 + 10 * tolerance:
        peak_utilization = 1.0
    tight_ratio = (
        sum(value >= 1.0 - tolerance for value in resource_utilizations)
        / len(resource_utilizations)
        if resource_utilizations else 0.0
    )

    request_count = len(record.episode.requests)
    completed_mass = solution.completed_request_mass
    average_latency = (
        solution.total_completion_latency / completed_mass
        if completed_mass > tolerance else 0.0
    )
    completion_ratio = completed_mass / request_count if request_count else 0.0
    if -10 * tolerance <= completion_ratio < 0.0:
        completion_ratio = 0.0
    if 1.0 < completion_ratio <= 1.0 + 10 * tolerance:
        completion_ratio = 1.0
    return TeacherBatchStatistics(
        request_count=request_count,
        candidate_count=len(record.candidates),
        variable_count=len(record.expansion.variables),
        constraint_count=len(solution.stage_two_lp.b_ub),
        rejected_candidate_count=len(record.expansion.rejections),
        completed_request_mass=completed_mass,
        completion_ratio=completion_ratio,
        average_completion_latency=average_latency,
        active_variable_count=active_variable_count,
        fractional_variable_count=fractional_variable_count,
        fractional_request_count=fractional_request_count,
        fractional_request_ratio=(
            fractional_request_count / request_count if request_count else 0.0
        ),
        mean_resource_utilization=mean_utilization,
        peak_resource_utilization=peak_utilization,
        tight_resource_constraint_ratio=tight_ratio,
        max_constraint_violation=float(
            solution.stage_two.max_violation_trajectory[-1]
        ),
        stage_one_iterations=solution.stage_one.iterations,
        stage_two_iterations=solution.stage_two.iterations,
        solve_seconds=record.solve_seconds,
    )


def _episode_for_profile(
    request_pool: EpisodeSpec,
    profile: StaticLoadProfile,
) -> EpisodeSpec:
    if profile.request_count > len(request_pool.requests):
        raise ValueError("load profile exceeds the shared request pool")
    requests = tuple(
        replace(request, arrival=0, ttl=profile.horizon)
        for request in request_pool.requests[:profile.request_count]
    )
    return EpisodeSpec(
        seed=request_pool.seed,
        nodes=request_pool.nodes,
        edges=request_pool.edges,
        requests=requests,
        horizon=profile.horizon,
        physical=request_pool.physical,
    )


def _aggregate_entries(
    profiles: Sequence[StaticLoadProfile],
    entries: Sequence[StaticLoadCalibrationEntry],
) -> tuple[StaticLoadAggregate, ...]:
    aggregates: list[StaticLoadAggregate] = []
    for profile in profiles:
        selected = [
            entry.statistics for entry in entries
            if entry.load_profile == profile.name
        ]
        completion_ratios = [item.completion_ratio for item in selected]
        aggregates.append(StaticLoadAggregate(
            load_profile=profile.name,
            sample_count=len(selected),
            mean_completion_ratio=float(np.mean(completion_ratios)),
            min_completion_ratio=min(completion_ratios),
            max_completion_ratio=max(completion_ratios),
            mean_average_completion_latency=float(np.mean([
                item.average_completion_latency for item in selected
            ])),
            mean_fractional_request_ratio=float(np.mean([
                item.fractional_request_ratio for item in selected
            ])),
            mean_peak_resource_utilization=float(np.mean([
                item.peak_resource_utilization for item in selected
            ])),
            mean_solve_seconds=float(np.mean([
                item.solve_seconds for item in selected
            ])),
        ))
    return tuple(aggregates)


def generate_static_load_calibration(
    scenario_template: ScenarioConfig,
    seeds: Iterable[int],
    output_directory: str | Path,
    *,
    profiles: Sequence[StaticLoadProfile] = DEFAULT_STATIC_LOAD_PROFILES,
    path_candidate_count: int = 1,
    construction_kinds: tuple[str, ...] = ("left_deep", "balanced"),
    overwrite: bool = False,
) -> StaticLoadCalibrationResult:
    """Generate nested static batches and write per-instance load statistics."""

    ordered_profiles = tuple(profiles)
    if not ordered_profiles:
        raise ValueError("at least one load profile is required")
    if len({profile.name for profile in ordered_profiles}) != len(ordered_profiles):
        raise ValueError("load profile names must be unique")
    ordered_seeds = tuple(int(seed) for seed in seeds)
    if not ordered_seeds:
        raise ValueError("at least one seed is required")
    if any(seed < 0 for seed in ordered_seeds):
        raise ValueError("seeds must be non-negative")
    if len(set(ordered_seeds)) != len(ordered_seeds):
        raise ValueError("calibration seeds must be unique")

    output = Path(output_directory)
    manifest_path = output / "calibration.json"
    csv_path = output / "calibration.csv"
    targets = {
        (profile.name, seed): (
            output / profile.name / f"teacher_seed_{seed:08d}.npz"
        )
        for profile in ordered_profiles
        for seed in ordered_seeds
    }
    if not overwrite:
        existing = [
            path for path in (*targets.values(), manifest_path, csv_path)
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                f"static load calibration output already exists: {existing[0]}"
            )

    output.mkdir(parents=True, exist_ok=True)
    max_requests = max(profile.request_count for profile in ordered_profiles)
    max_horizon = max(profile.horizon for profile in ordered_profiles)
    pool_config = replace(
        scenario_template,
        request_count=max_requests,
        ttl=max_horizon,
        horizon=max_horizon,
    )
    entries: list[StaticLoadCalibrationEntry] = []
    for seed in ordered_seeds:
        request_pool = make_episode(pool_config, seed)
        for profile in ordered_profiles:
            episode = _episode_for_profile(request_pool, profile)
            record = solve_teacher_episode(
                episode,
                path_candidate_count=path_candidate_count,
                construction_kinds=construction_kinds,
                load_profile=profile.name,
            )
            target = targets[(profile.name, seed)]
            save_teacher_batch_record(record, target)
            entries.append(StaticLoadCalibrationEntry(
                load_profile=profile.name,
                seed=seed,
                file=str(target.relative_to(output)).replace("\\", "/"),
                statistics=summarize_teacher_batch(record),
            ))

    aggregates = _aggregate_entries(ordered_profiles, entries)
    manifest = {
        "schema_version": 1,
        "scenario_template": asdict(scenario_template),
        "profiles": [asdict(profile) for profile in ordered_profiles],
        "path_candidate_count": path_candidate_count,
        "construction_kinds": list(construction_kinds),
        "records": [entry.flat_dict() for entry in entries],
        "aggregates": [asdict(item) for item in aggregates],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rows = [entry.flat_dict() for entry in entries]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return StaticLoadCalibrationResult(
        manifest_path=manifest_path,
        csv_path=csv_path,
        entries=tuple(entries),
        aggregates=aggregates,
    )
