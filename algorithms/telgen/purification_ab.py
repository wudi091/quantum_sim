"""Static-batch A/B evaluation for optional elementary-link purification."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
from statistics import fmean
from typing import Iterable

from qnet_core.spec import EpisodeSpec

from .dataset import solve_teacher_episode
from .hard_decoder import HardConstraintDecoder
from .physical_validation import validate_decoded_physics


@dataclass(frozen=True)
class PurificationABTrial:
    """One fixed physical seed evaluated under one planning variant."""

    seed: int
    completed_requests: int
    completion_rate: float
    mean_successful_latency_slots: float | None
    mean_censored_latency_slots: float
    schedule_adherent: bool
    peak_memory_usage: float
    memory_time_unit_slots: float
    memory_time_per_completed_request_slots: float | None
    purification_attempts: int
    purification_successes: int
    physical_failures: int
    fidelity_violations: int


@dataclass(frozen=True)
class PurificationABVariantResult:
    """Teacher, decoder, and repeated SeQUeNCe results for one variant."""

    variant: str
    purification_kinds: tuple[str, ...]
    fidelity_model: str
    candidate_count: int
    variable_count: int
    rejection_counts: tuple[tuple[str, int], ...]
    teacher_completed_mass: float
    planned_selected_requests: int
    planned_purified_requests: int
    planned_purified_request_ids: tuple[str, ...]
    planned_completion_latency_slots: float
    selected_variable_ids: tuple[str, ...]
    trials: tuple[PurificationABTrial, ...]
    mean_completed_requests: float
    mean_completion_rate: float
    mean_completion_retention: float | None
    mean_successful_latency_slots: float | None
    mean_censored_latency_slots: float
    schedule_adherence_rate: float
    mean_peak_memory_usage: float
    mean_memory_time_unit_slots: float
    pooled_memory_time_per_completed_request_slots: float | None
    mean_purification_attempts: float
    purification_success_rate: float | None
    mean_physical_failures: float
    mean_fidelity_violations: float


@dataclass(frozen=True)
class PurificationABProvenance:
    """Inputs and metric semantics needed to reproduce and interpret a run."""

    episode: EpisodeSpec
    path_candidate_count: int
    construction_kinds: tuple[str, ...]
    decoder_beam_width: int
    decoder_random_restarts: int
    pairing_semantics: str
    evidence_scope: str
    on_demand_scope: str
    successful_latency_aggregation: str
    purification_success_definition: str
    code_revision: str | None
    working_directory: str | None
    run_command: tuple[str, ...]
    source_tree_sha256: str | None
    source_file_hashes: tuple[tuple[str, str], ...]
    software_versions: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class PurificationABReport:
    """Seed-stratified comparison of two purification planning variants."""

    schema_version: int
    episode_seed: int
    request_count: int
    request_required_fidelities: tuple[tuple[str, float], ...]
    horizon_slots: int
    physical_seeds: tuple[int, ...]
    provenance: PurificationABProvenance
    baseline: PurificationABVariantResult
    on_demand: PurificationABVariantResult
    deltas: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class PurificationABOutputPaths:
    timestamped_json: Path
    latest_json: Path
    timestamped_csv: Path
    latest_csv: Path


def _software_versions() -> tuple[tuple[str, str], ...]:
    resolved = [("python", platform.python_version())]
    for distribution in ("networkx", "numpy", "scipy", "sequence"):
        try:
            installed = version(distribution)
        except PackageNotFoundError:
            installed = "not-installed"
        resolved.append((distribution, installed))
    return tuple(resolved)


def _trial_result(
    seed: int,
    evaluation,
    *,
    request_count: int,
    slot_duration_ps: int,
) -> PurificationABTrial:
    successful_latencies = [
        (settlement.settlement_time - settlement.arrival_time)
        / slot_duration_ps
        for settlement in evaluation.settlements
        if settlement.success
    ]
    purification_events = tuple(
        event for event in evaluation.event_trace
        if event.event_kind == "purify"
    )
    completed = int(evaluation.metrics["completed_requests"])
    memory_time_unit_slots = float(
        evaluation.metrics["physical_memory_time_unit_slots"]
    )
    return PurificationABTrial(
        seed=int(seed),
        completed_requests=completed,
        completion_rate=completed / max(request_count, 1),
        mean_successful_latency_slots=(
            fmean(successful_latencies) if successful_latencies else None
        ),
        mean_censored_latency_slots=(
            evaluation.metrics["mean_censored_latency_ps"]
            / slot_duration_ps
        ),
        schedule_adherent=bool(evaluation.metrics["schedule_adherence"]),
        peak_memory_usage=float(evaluation.metrics["peak_memory_usage"]),
        memory_time_unit_slots=memory_time_unit_slots,
        memory_time_per_completed_request_slots=(
            memory_time_unit_slots / completed
            if completed else None
        ),
        purification_attempts=len(purification_events),
        purification_successes=sum(event.success for event in purification_events),
        physical_failures=int(evaluation.metrics["physical_failure_count"]),
        fidelity_violations=int(evaluation.metrics["fidelity_violation_count"]),
    )


def _evaluate_variant(
    episode: EpisodeSpec,
    physical_seeds: tuple[int, ...],
    *,
    variant: str,
    purification_kinds: tuple[str, ...],
    path_candidate_count: int,
    construction_kinds: tuple[str, ...],
    decoder_beam_width: int,
    decoder_random_restarts: int,
) -> PurificationABVariantResult:
    record = solve_teacher_episode(
        episode,
        path_candidate_count=path_candidate_count,
        construction_kinds=construction_kinds,
        purification_kinds=purification_kinds,
    )
    decoded = HardConstraintDecoder(
        beam_width=decoder_beam_width,
        random_restarts=decoder_random_restarts,
        random_seed=episode.seed,
    ).decode(
        record.expansion,
        record.capacities,
        record.solution.stage_two.primal,
        request_ids=tuple(request.id for request in episode.requests),
    )
    physical = validate_decoded_physics(
        episode,
        decoded,
        physical_seeds,
    )
    trials = tuple(
        _trial_result(
            trial.seed,
            trial.evaluation,
            request_count=len(episode.requests),
            slot_duration_ps=episode.physical.slot_duration_ps,
        )
        for trial in physical.trials
    )
    successful_latencies = [
        (settlement.settlement_time - settlement.arrival_time)
        / episode.physical.slot_duration_ps
        for physical_trial in physical.trials
        for settlement in physical_trial.evaluation.settlements
        if settlement.success
    ]
    purification_attempts = sum(item.purification_attempts for item in trials)
    purification_successes = sum(item.purification_successes for item in trials)
    total_completed = sum(item.completed_requests for item in trials)
    total_memory_time = sum(item.memory_time_unit_slots for item in trials)
    rejection_counts = Counter(
        rejection.reason for rejection in record.expansion.rejections
    )
    return PurificationABVariantResult(
        variant=variant,
        purification_kinds=purification_kinds,
        fidelity_model=record.fidelity_model,
        candidate_count=len(record.candidates),
        variable_count=len(record.expansion.variables),
        rejection_counts=tuple(sorted(rejection_counts.items())),
        teacher_completed_mass=record.solution.completed_request_mass,
        planned_selected_requests=decoded.completed_request_count,
        planned_purified_requests=sum(
            variable.purification_kind != "none"
            for variable in decoded.selected_variables
        ),
        planned_purified_request_ids=tuple(sorted(
            variable.request_id
            for variable in decoded.selected_variables
            if variable.purification_kind != "none"
        )),
        planned_completion_latency_slots=decoded.total_completion_latency,
        selected_variable_ids=tuple(
            variable.variable_id for variable in decoded.selected_variables
        ),
        trials=trials,
        mean_completed_requests=physical.mean_completed_requests,
        mean_completion_rate=fmean(item.completion_rate for item in trials),
        mean_completion_retention=physical.mean_completion_retention,
        mean_successful_latency_slots=(
            fmean(successful_latencies) if successful_latencies else None
        ),
        mean_censored_latency_slots=physical.mean_censored_latency_slots,
        schedule_adherence_rate=physical.schedule_adherence_rate,
        mean_peak_memory_usage=fmean(item.peak_memory_usage for item in trials),
        mean_memory_time_unit_slots=fmean(
            item.memory_time_unit_slots for item in trials
        ),
        pooled_memory_time_per_completed_request_slots=(
            total_memory_time / total_completed
            if total_completed else None
        ),
        mean_purification_attempts=fmean(
            item.purification_attempts for item in trials
        ),
        purification_success_rate=(
            purification_successes / purification_attempts
            if purification_attempts else None
        ),
        mean_physical_failures=fmean(item.physical_failures for item in trials),
        mean_fidelity_violations=fmean(
            item.fidelity_violations for item in trials
        ),
    )


def run_purification_ab(
    episode: EpisodeSpec,
    physical_seeds: Iterable[int],
    *,
    path_candidate_count: int = 3,
    construction_kinds: tuple[str, ...] = ("left_deep", "balanced"),
    decoder_beam_width: int = 512,
    decoder_random_restarts: int = 128,
    code_revision: str | None = None,
    working_directory: str | None = None,
    run_command: tuple[str, ...] = (),
    source_tree_sha256: str | None = None,
    source_file_hashes: tuple[tuple[str, str], ...] = (),
) -> PurificationABReport:
    """Run two static-batch variants under the same episode and seed labels.

    The variants are not strict common-random-number replications because
    purification changes the physical event stream and random-key sequence.
    """

    if not episode.requests:
        raise ValueError("purification A/B evaluation requires requests")
    if any(request.arrival != 0 for request in episode.requests):
        raise ValueError("purification A/B evaluation requires a static batch")
    seeds = tuple(int(seed) for seed in physical_seeds)
    if not seeds:
        raise ValueError("at least one physical seed is required")
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("physical seeds must be unique and non-negative")
    baseline = _evaluate_variant(
        episode,
        seeds,
        variant="no_purification",
        purification_kinds=("none",),
        path_candidate_count=path_candidate_count,
        construction_kinds=construction_kinds,
        decoder_beam_width=decoder_beam_width,
        decoder_random_restarts=decoder_random_restarts,
    )
    on_demand = _evaluate_variant(
        episode,
        seeds,
        variant="on_demand_elementary_once",
        purification_kinds=("none", "elementary_once"),
        path_candidate_count=path_candidate_count,
        construction_kinds=construction_kinds,
        decoder_beam_width=decoder_beam_width,
        decoder_random_restarts=decoder_random_restarts,
    )
    delta_values = {
        "planned_selected_requests": float(
            on_demand.planned_selected_requests
            - baseline.planned_selected_requests
        ),
        "planned_completion_latency_slots": (
            on_demand.planned_completion_latency_slots
            - baseline.planned_completion_latency_slots
        ),
        "mean_completed_requests": (
            on_demand.mean_completed_requests
            - baseline.mean_completed_requests
        ),
        "mean_completion_rate": (
            on_demand.mean_completion_rate - baseline.mean_completion_rate
        ),
        "mean_censored_latency_slots": (
            on_demand.mean_censored_latency_slots
            - baseline.mean_censored_latency_slots
        ),
        "mean_peak_memory_usage": (
            on_demand.mean_peak_memory_usage
            - baseline.mean_peak_memory_usage
        ),
        "mean_memory_time_unit_slots": (
            on_demand.mean_memory_time_unit_slots
            - baseline.mean_memory_time_unit_slots
        ),
        "mean_purification_attempts": (
            on_demand.mean_purification_attempts
            - baseline.mean_purification_attempts
        ),
        "mean_physical_failures": (
            on_demand.mean_physical_failures
            - baseline.mean_physical_failures
        ),
        "mean_fidelity_violations": (
            on_demand.mean_fidelity_violations
            - baseline.mean_fidelity_violations
        ),
        "schedule_adherence_rate": (
            on_demand.schedule_adherence_rate
            - baseline.schedule_adherence_rate
        ),
    }
    if (
        baseline.mean_successful_latency_slots is not None
        and on_demand.mean_successful_latency_slots is not None
    ):
        delta_values["mean_successful_latency_slots"] = (
            on_demand.mean_successful_latency_slots
            - baseline.mean_successful_latency_slots
        )
    if (
        baseline.mean_completion_retention is not None
        and on_demand.mean_completion_retention is not None
    ):
        delta_values["mean_completion_retention"] = (
            on_demand.mean_completion_retention
            - baseline.mean_completion_retention
        )
    if (
        baseline.pooled_memory_time_per_completed_request_slots is not None
        and on_demand.pooled_memory_time_per_completed_request_slots is not None
    ):
        delta_values["pooled_memory_time_per_completed_request_slots"] = (
            on_demand.pooled_memory_time_per_completed_request_slots
            - baseline.pooled_memory_time_per_completed_request_slots
        )
    deltas = tuple(sorted(delta_values.items()))
    return PurificationABReport(
        schema_version=2,
        episode_seed=episode.seed,
        request_count=len(episode.requests),
        request_required_fidelities=tuple(
            (request.id, float(request.required_fidelity))
            for request in episode.requests
        ),
        horizon_slots=episode.horizon,
        physical_seeds=seeds,
        provenance=PurificationABProvenance(
            episode=episode,
            path_candidate_count=path_candidate_count,
            construction_kinds=tuple(construction_kinds),
            decoder_beam_width=decoder_beam_width,
            decoder_random_restarts=decoder_random_restarts,
            pairing_semantics=(
                "same episode and physical seed labels; not strict common "
                "random numbers because variants execute different event streams"
            ),
            evidence_scope=(
                "single-episode sanity evaluation; results do not establish "
                "population-level superiority across topologies or loads"
            ),
            on_demand_scope=(
                "candidate-level: purification is removed only when the matching "
                "request-route-construction unpurified candidate meets fidelity"
            ),
            successful_latency_aggregation=(
                "pooled successful request-trial micro-average"
            ),
            purification_success_definition=(
                "successful PURIFY events divided by PURIFY events; event success "
                "includes physical completion and output-fidelity acceptance"
            ),
            code_revision=code_revision,
            working_directory=working_directory,
            run_command=tuple(run_command),
            source_tree_sha256=source_tree_sha256,
            source_file_hashes=tuple(source_file_hashes),
            software_versions=_software_versions(),
        ),
        baseline=baseline,
        on_demand=on_demand,
        deltas=deltas,
    )


def save_purification_ab_report(
    report: PurificationABReport,
    output_directory: str | Path,
    *,
    stem: str = "purification_ab_results",
) -> PurificationABOutputPaths:
    """Save timestamped and latest JSON/CSV copies for downstream analysis."""

    if not stem:
        raise ValueError("output stem must be non-empty")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = timestamp
    collision_index = 2
    while (
        (output / f"{stem}_{version}.json").exists()
        or (output / f"{stem}_{version}.csv").exists()
    ):
        version = f"{timestamp}_{collision_index}"
        collision_index += 1
    timestamped_json = output / f"{stem}_{version}.json"
    latest_json = output / f"{stem}.json"
    timestamped_csv = output / f"{stem}_{version}.csv"
    latest_csv = output / f"{stem}.csv"

    json_text = json.dumps(
        asdict(report),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    timestamped_json.write_text(json_text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")

    fieldnames = [
        "provenance",
        "row_type",
        "schema_version",
        "episode_seed",
        "request_count",
        "horizon_slots",
        "request_required_fidelities",
        "variant",
        "purification_kinds",
        "fidelity_model",
        "candidate_count",
        "variable_count",
        "teacher_completed_mass",
        "planned_selected_requests",
        "planned_purified_requests",
        "planned_purified_request_ids",
        "planned_completion_latency_slots",
        "seed",
        "completed_requests",
        "completion_rate",
        "mean_successful_latency_slots",
        "mean_censored_latency_slots",
        "schedule_adherent",
        "peak_memory_usage",
        "memory_time_unit_slots",
        "memory_time_per_completed_request_slots",
        "purification_attempts",
        "purification_successes",
        "physical_failures",
        "fidelity_violations",
        "aggregate_mean_completed_requests",
        "aggregate_mean_completion_rate",
        "aggregate_mean_completion_retention",
        "aggregate_mean_successful_latency_slots",
        "aggregate_mean_censored_latency_slots",
        "aggregate_schedule_adherence_rate",
        "aggregate_mean_peak_memory_usage",
        "aggregate_mean_memory_time_unit_slots",
        "aggregate_pooled_memory_time_per_completed_request_slots",
        "aggregate_mean_purification_attempts",
        "aggregate_purification_success_rate",
        "aggregate_mean_physical_failures",
        "aggregate_mean_fidelity_violations",
        "delta_metrics",
    ]
    provenance_json = json.dumps(
        asdict(report.provenance),
        ensure_ascii=False,
    )
    common_row = {
        "schema_version": report.schema_version,
        "episode_seed": report.episode_seed,
        "request_count": report.request_count,
        "horizon_slots": report.horizon_slots,
        "request_required_fidelities": json.dumps(
            report.request_required_fidelities,
            ensure_ascii=False,
        ),
    }

    def variant_row(variant: PurificationABVariantResult) -> dict[str, object]:
        return {
            "variant": variant.variant,
            "purification_kinds": json.dumps(
                variant.purification_kinds,
                ensure_ascii=False,
            ),
            "fidelity_model": variant.fidelity_model,
            "candidate_count": variant.candidate_count,
            "variable_count": variant.variable_count,
            "teacher_completed_mass": variant.teacher_completed_mass,
            "planned_selected_requests": variant.planned_selected_requests,
            "planned_purified_requests": variant.planned_purified_requests,
            "planned_purified_request_ids": json.dumps(
                variant.planned_purified_request_ids,
                ensure_ascii=False,
            ),
            "planned_completion_latency_slots": (
                variant.planned_completion_latency_slots
            ),
        }

    trial_rows = [
        {
            **common_row,
            "row_type": "trial",
            **variant_row(variant),
            **asdict(trial),
        }
        for variant in (report.baseline, report.on_demand)
        for trial in variant.trials
    ]
    aggregate_rows = [
        {
            **common_row,
            "row_type": "aggregate",
            **variant_row(variant),
            "aggregate_mean_completed_requests": (
                variant.mean_completed_requests
            ),
            "aggregate_mean_completion_rate": variant.mean_completion_rate,
            "aggregate_mean_completion_retention": (
                variant.mean_completion_retention
            ),
            "aggregate_mean_successful_latency_slots": (
                variant.mean_successful_latency_slots
            ),
            "aggregate_mean_censored_latency_slots": (
                variant.mean_censored_latency_slots
            ),
            "aggregate_schedule_adherence_rate": (
                variant.schedule_adherence_rate
            ),
            "aggregate_mean_peak_memory_usage": variant.mean_peak_memory_usage,
            "aggregate_mean_memory_time_unit_slots": (
                variant.mean_memory_time_unit_slots
            ),
            "aggregate_pooled_memory_time_per_completed_request_slots": (
                variant.pooled_memory_time_per_completed_request_slots
            ),
            "aggregate_mean_purification_attempts": (
                variant.mean_purification_attempts
            ),
            "aggregate_purification_success_rate": (
                variant.purification_success_rate
            ),
            "aggregate_mean_physical_failures": (
                variant.mean_physical_failures
            ),
            "aggregate_mean_fidelity_violations": (
                variant.mean_fidelity_violations
            ),
        }
        for variant in (report.baseline, report.on_demand)
    ]
    delta_row = {
        **common_row,
        "row_type": "delta",
        "variant": "on_demand_minus_no_purification",
        "delta_metrics": json.dumps(dict(report.deltas), ensure_ascii=False),
    }
    rows = [*trial_rows, *aggregate_rows, delta_row]
    rows[0]["provenance"] = provenance_json
    for target in (timestamped_csv, latest_csv):
        with target.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return PurificationABOutputPaths(
        timestamped_json=timestamped_json,
        latest_json=latest_json,
        timestamped_csv=timestamped_csv,
        latest_csv=latest_csv,
    )


__all__ = [
    "PurificationABOutputPaths",
    "PurificationABProvenance",
    "PurificationABReport",
    "PurificationABTrial",
    "PurificationABVariantResult",
    "run_purification_ab",
    "save_purification_ab_report",
]
