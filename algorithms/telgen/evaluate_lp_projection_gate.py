"""Evaluate whether LP scores survive a minimal feasible projection.

This is a go/no-go gate for LP-trajectory imitation.  It re-solves the
continuous teacher on already certified MILP graph samples, applies exactly
one score-ordered feasibility scan, and compares the result with both the
stored MILP optimum and random scores passed through the same scan.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import time
from typing import Iterable, Mapping

import numpy as np

from .hard_decoder import (
    greedy_feasible_projection,
    validate_decoded_selection,
)
from .milp_imitation import MILPGraphSample
from .online_milp_dataset import load_online_milp_dataset
from .teacher import ConstructionAwareLPTeacher


DEFAULT_DATASET = Path(
    "results/diverse_teacher_pilot/online_milp_dataset.json"
)
DEFAULT_OUTPUT_DIRECTORY = Path("results/lp_projection_gate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gate LP trajectory imitation with a minimal greedy projection."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--solver-backend",
        choices=("highs_ipm", "trajectory_ipm"),
        default="highs_ipm",
    )
    parser.add_argument("--solver-tolerance", type=float, default=1e-7)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--support-tolerance", type=float, default=1e-9)
    parser.add_argument("--retention-threshold", type=float, default=0.95)
    parser.add_argument("--random-trials", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=20260814)
    parser.add_argument(
        "--limit-samples",
        type=int,
        help="Optional deterministic prefix for a quick sanity run.",
    )
    return parser


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _mean_or_none(values: Iterable[float | None]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return None if not finite else float(np.mean(finite))


def _selected_from_labels(sample: MILPGraphSample):
    return tuple(
        variable
        for variable, label in zip(sample.variables, sample.labels)
        if float(label) > 0.5
    )


def _random_scores(
    variable_count: int,
    *,
    base_seed: int,
    sample_index: int,
    trial_index: int,
) -> np.ndarray:
    seed = np.random.SeedSequence(
        [int(base_seed), int(sample_index), int(trial_index)]
    )
    return np.random.default_rng(seed).random(variable_count)


def _evaluate_sample(
    sample: MILPGraphSample,
    *,
    sample_path: Path,
    sample_index: int,
    teacher: ConstructionAwareLPTeacher,
    support_tolerance: float,
    random_trials: int,
    random_seed: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    started = time.perf_counter()
    lp_solution = teacher.solve(
        sample.variables,
        sample.resource_capacities,
        reserved_usage=sample.reserved_usage,
    )
    stage_one_projected = greedy_feasible_projection(
        sample.variables,
        sample.resource_capacities,
        {
            variable.variable_id: float(value)
            for variable, value in zip(
                lp_solution.variables,
                lp_solution.stage_one.primal,
            )
        },
        request_ids=sample.request_ids,
        reserved_usage=sample.reserved_usage,
        support_tolerance=support_tolerance,
    )
    projected = greedy_feasible_projection(
        sample.variables,
        sample.resource_capacities,
        lp_solution.final_values,
        request_ids=sample.request_ids,
        reserved_usage=sample.reserved_usage,
        support_tolerance=support_tolerance,
    )
    milp_selected = _selected_from_labels(sample)
    milp_feasibility = validate_decoded_selection(
        milp_selected,
        sample.resource_capacities,
        sample.reserved_usage,
    )
    if not milp_feasibility.feasible:
        raise ValueError(
            f"stored MILP label is infeasible: {sample_path}: "
            f"{milp_feasibility.violations[0]}"
        )

    lp_values = np.asarray(
        [lp_solution.final_values[item.variable_id] for item in sample.variables],
        dtype=float,
    )
    stage_one_values_by_id = {
        variable.variable_id: float(value)
        for variable, value in zip(
            lp_solution.variables,
            lp_solution.stage_one.primal,
        )
    }
    stage_one_values = np.asarray(
        [stage_one_values_by_id[item.variable_id] for item in sample.variables],
        dtype=float,
    )
    stage_one_fractional = (stage_one_values > support_tolerance) & (
        stage_one_values < 1.0 - support_tolerance
    )
    fractional = (lp_values > support_tolerance) & (
        lp_values < 1.0 - support_tolerance
    )
    projection_mass = projected.expected_completed_request_mass
    projection_latency = projected.expected_total_completion_latency
    milp_mass = float(sample.optimal_expected_completed_request_mass)
    milp_latency = float(sample.optimal_total_completion_latency)

    random_counts = np.zeros(random_trials, dtype=float)
    random_masses = np.zeros(random_trials, dtype=float)
    random_latencies = np.zeros(random_trials, dtype=float)
    for trial_index in range(random_trials):
        random_projected = greedy_feasible_projection(
            sample.variables,
            sample.resource_capacities,
            _random_scores(
                len(sample.variables),
                base_seed=random_seed,
                sample_index=sample_index,
                trial_index=trial_index,
            ),
            request_ids=sample.request_ids,
            reserved_usage=sample.reserved_usage,
            support_tolerance=support_tolerance,
        )
        random_counts[trial_index] = random_projected.completed_request_count
        random_masses[trial_index] = (
            random_projected.expected_completed_request_mass
        )
        random_latencies[trial_index] = (
            random_projected.expected_total_completion_latency
        )

    row: dict[str, object] = {
        "sample_index": sample_index,
        "sample_path": str(sample_path),
        "episode_seed": sample.seed,
        "variable_count": len(sample.variables),
        "request_count": len(sample.request_ids),
        "reserved_resource_slot_count": len(sample.reserved_usage),
        "lp_support_variable_count": projected.support_variable_count,
        "lp_fractional_variable_count": int(np.sum(fractional)),
        "lp_stage_one_support_variable_count": (
            stage_one_projected.support_variable_count
        ),
        "lp_stage_one_fractional_variable_count": int(
            np.sum(stage_one_fractional)
        ),
        "lp_stage_one_max_violation": float(
            lp_solution.stage_one.max_violation_trajectory[-1]
        ),
        "lp_stage_two_max_violation": float(
            lp_solution.stage_two.max_violation_trajectory[-1]
        ),
        "lp_upper_bound_expected_mass": lp_solution.stage_one_completed_mass,
        "milp_completed_request_count": sample.optimal_completed_request_count,
        "milp_expected_completed_mass": milp_mass,
        "milp_expected_total_latency": milp_latency,
        "stage_one_projection_feasible": (
            stage_one_projected.feasibility.feasible
        ),
        "stage_one_projection_completed_request_count": (
            stage_one_projected.completed_request_count
        ),
        "stage_one_projection_expected_completed_mass": (
            stage_one_projected.expected_completed_request_mass
        ),
        "stage_one_projection_expected_total_latency": (
            stage_one_projected.expected_total_completion_latency
        ),
        "stage_one_projection_count_retention": _safe_ratio(
            stage_one_projected.completed_request_count,
            sample.optimal_completed_request_count,
        ),
        "stage_one_projection_mass_retention": _safe_ratio(
            stage_one_projected.expected_completed_request_mass,
            milp_mass,
        ),
        "projection_feasible": projected.feasibility.feasible,
        "projection_completed_request_count": projected.completed_request_count,
        "projection_expected_completed_mass": projection_mass,
        "projection_expected_total_latency": projection_latency,
        "projection_count_retention": _safe_ratio(
            projected.completed_request_count,
            sample.optimal_completed_request_count,
        ),
        "projection_mass_retention": _safe_ratio(projection_mass, milp_mass),
        "projection_average_latency": _safe_ratio(
            projection_latency,
            projection_mass,
        ),
        "milp_average_latency": _safe_ratio(milp_latency, milp_mass),
        "random_mean_completed_request_count": float(np.mean(random_counts)),
        "random_mean_expected_completed_mass": float(np.mean(random_masses)),
        "random_p95_expected_completed_mass": float(
            np.percentile(random_masses, 95)
        ),
        "random_mean_expected_total_latency": float(
            np.mean(random_latencies)
        ),
        "solve_and_projection_seconds": time.perf_counter() - started,
    }
    return row, random_counts, random_masses


def _write_outputs(
    rows: list[dict[str, object]],
    summary: Mapping[str, object],
    output_directory: Path,
) -> dict[str, str]:
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = ""
    collision_index = 1
    while (
        output_directory / f"summary_{timestamp}{suffix}.json"
    ).exists():
        collision_index += 1
        suffix = f"_{collision_index}"
    versioned_json = output_directory / f"summary_{timestamp}{suffix}.json"
    latest_json = output_directory / "summary.json"
    versioned_csv = output_directory / f"samples_{timestamp}{suffix}.csv"
    latest_csv = output_directory / "samples.csv"

    json_text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    versioned_json.write_text(json_text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")
    fieldnames = list(rows[0]) if rows else []
    for path in (versioned_csv, latest_csv):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return {
        "versioned_summary": str(versioned_json),
        "latest_summary": str(latest_json),
        "versioned_samples": str(versioned_csv),
        "latest_samples": str(latest_csv),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.retention_threshold <= 1.0:
        raise ValueError("retention threshold must lie in (0, 1]")
    if args.random_trials < 1:
        raise ValueError("random trials must be positive")
    if args.limit_samples is not None and args.limit_samples < 1:
        raise ValueError("limit samples must be positive")

    dataset = load_online_milp_dataset(args.dataset)
    indexed_samples = list(zip(dataset.sample_paths, dataset.samples))
    if args.limit_samples is not None:
        indexed_samples = indexed_samples[:args.limit_samples]
    teacher = ConstructionAwareLPTeacher(
        tolerance=args.solver_tolerance,
        max_iterations=args.max_iterations,
        solver_backend=args.solver_backend,
    )

    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    random_count_totals = np.zeros(args.random_trials, dtype=float)
    random_mass_totals = np.zeros(args.random_trials, dtype=float)
    for sample_index, (sample_path, sample) in enumerate(indexed_samples):
        row, random_counts, random_masses = _evaluate_sample(
            sample,
            sample_path=sample_path,
            sample_index=sample_index,
            teacher=teacher,
            support_tolerance=args.support_tolerance,
            random_trials=args.random_trials,
            random_seed=args.random_seed,
        )
        rows.append(row)
        random_count_totals += random_counts
        random_mass_totals += random_masses

    milp_total_count = float(sum(
        int(row["milp_completed_request_count"]) for row in rows
    ))
    projected_total_count = float(sum(
        int(row["projection_completed_request_count"]) for row in rows
    ))
    stage_one_projected_total_count = float(sum(
        int(row["stage_one_projection_completed_request_count"])
        for row in rows
    ))
    milp_total_mass = float(sum(
        float(row["milp_expected_completed_mass"]) for row in rows
    ))
    projected_total_mass = float(sum(
        float(row["projection_expected_completed_mass"]) for row in rows
    ))
    stage_one_projected_total_mass = float(sum(
        float(row["stage_one_projection_expected_completed_mass"])
        for row in rows
    ))
    lp_total_upper_bound = float(sum(
        float(row["lp_upper_bound_expected_mass"]) for row in rows
    ))
    milp_total_latency = float(sum(
        float(row["milp_expected_total_latency"]) for row in rows
    ))
    projected_total_latency = float(sum(
        float(row["projection_expected_total_latency"]) for row in rows
    ))
    stage_one_projected_total_latency = float(sum(
        float(row["stage_one_projection_expected_total_latency"])
        for row in rows
    ))
    lp_max_violation = max(
        max(
            float(row["lp_stage_one_max_violation"]),
            float(row["lp_stage_two_max_violation"]),
        )
        for row in rows
    )
    lp_feasibility_tolerance = max(1e-7, 10.0 * args.solver_tolerance)
    count_retention = _safe_ratio(projected_total_count, milp_total_count)
    mass_retention = _safe_ratio(projected_total_mass, milp_total_mass)
    stage_one_count_retention = _safe_ratio(
        stage_one_projected_total_count,
        milp_total_count,
    )
    stage_one_mass_retention = _safe_ratio(
        stage_one_projected_total_mass,
        milp_total_mass,
    )
    random_mean_total_count = float(np.mean(random_count_totals))
    random_mean_total_mass = float(np.mean(random_mass_totals))
    random_p95_total_mass = float(np.percentile(random_mass_totals, 95))
    feasible_rate = float(np.mean([
        bool(row["projection_feasible"]) for row in rows
    ]))
    stage_two_quality_gate_pass = bool(
        feasible_rate == 1.0
        and lp_max_violation <= lp_feasibility_tolerance
        and count_retention is not None
        and count_retention >= args.retention_threshold
        and mass_retention is not None
        and mass_retention >= args.retention_threshold
    )
    stage_one_feasible_rate = float(np.mean([
        bool(row["stage_one_projection_feasible"]) for row in rows
    ]))
    stage_one_quality_gate_pass = bool(
        stage_one_feasible_rate == 1.0
        and lp_max_violation <= lp_feasibility_tolerance
        and stage_one_count_retention is not None
        and stage_one_count_retention >= args.retention_threshold
        and stage_one_mass_retention is not None
        and stage_one_mass_retention >= args.retention_threshold
    )
    quality_gate_pass = stage_one_quality_gate_pass or stage_two_quality_gate_pass
    if stage_one_projected_total_mass >= projected_total_mass:
        best_lp_stage = "stage_one"
        best_projected_total_count = stage_one_projected_total_count
        best_projected_total_mass = stage_one_projected_total_mass
        best_count_retention = stage_one_count_retention
        best_mass_retention = stage_one_mass_retention
    else:
        best_lp_stage = "stage_two"
        best_projected_total_count = projected_total_count
        best_projected_total_mass = projected_total_mass
        best_count_retention = count_retention
        best_mass_retention = mass_retention
    random_control_pass = bool(
        best_projected_total_mass > random_p95_total_mass + 1e-9
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "experiment": "lp_minimal_projection_gate",
        "dataset_manifest": str(dataset.manifest_path),
        "sample_count": len(rows),
        "episode_seeds": sorted({int(row["episode_seed"]) for row in rows}),
        "solver_backend": args.solver_backend,
        "solver_tolerance": args.solver_tolerance,
        "support_tolerance": args.support_tolerance,
        "retention_threshold": args.retention_threshold,
        "random_trials": args.random_trials,
        "random_seed": args.random_seed,
        "runtime_seconds": time.perf_counter() - started,
        "feasible_rate": feasible_rate,
        "stage_one_feasible_rate": stage_one_feasible_rate,
        "lp_max_constraint_violation": lp_max_violation,
        "lp_feasibility_tolerance": lp_feasibility_tolerance,
        "milp_total_completed_request_count": milp_total_count,
        "stage_one_projection_total_completed_request_count": (
            stage_one_projected_total_count
        ),
        "stage_one_pooled_count_retention": stage_one_count_retention,
        "stage_one_mean_sample_count_retention": _mean_or_none(
            row["stage_one_projection_count_retention"] for row in rows
        ),
        "projection_total_completed_request_count": projected_total_count,
        "pooled_count_retention": count_retention,
        "mean_sample_count_retention": _mean_or_none(
            row["projection_count_retention"] for row in rows
        ),
        "milp_total_expected_completed_mass": milp_total_mass,
        "lp_total_upper_bound_expected_mass": lp_total_upper_bound,
        "lp_relaxation_upper_bound_ratio": _safe_ratio(
            lp_total_upper_bound,
            milp_total_mass,
        ),
        "stage_one_projection_total_expected_completed_mass": (
            stage_one_projected_total_mass
        ),
        "stage_one_pooled_expected_mass_retention": stage_one_mass_retention,
        "stage_one_mean_sample_expected_mass_retention": _mean_or_none(
            row["stage_one_projection_mass_retention"] for row in rows
        ),
        "projection_total_expected_completed_mass": projected_total_mass,
        "pooled_expected_mass_retention": mass_retention,
        "mean_sample_expected_mass_retention": _mean_or_none(
            row["projection_mass_retention"] for row in rows
        ),
        "fractional_sample_count": sum(
            int(row["lp_fractional_variable_count"]) > 0 for row in rows
        ),
        "fractional_variable_count": sum(
            int(row["lp_fractional_variable_count"]) for row in rows
        ),
        "stage_one_fractional_sample_count": sum(
            int(row["lp_stage_one_fractional_variable_count"]) > 0
            for row in rows
        ),
        "stage_one_fractional_variable_count": sum(
            int(row["lp_stage_one_fractional_variable_count"])
            for row in rows
        ),
        "milp_pooled_average_latency": _safe_ratio(
            milp_total_latency,
            milp_total_mass,
        ),
        "projection_pooled_average_latency": _safe_ratio(
            projected_total_latency,
            projected_total_mass,
        ),
        "stage_one_projection_pooled_average_latency": _safe_ratio(
            stage_one_projected_total_latency,
            stage_one_projected_total_mass,
        ),
        "random_mean_total_completed_request_count": random_mean_total_count,
        "random_mean_count_retention": _safe_ratio(
            random_mean_total_count,
            milp_total_count,
        ),
        "random_mean_total_expected_completed_mass": random_mean_total_mass,
        "random_mean_expected_mass_retention": _safe_ratio(
            random_mean_total_mass,
            milp_total_mass,
        ),
        "random_p95_total_expected_completed_mass": random_p95_total_mass,
        "random_p95_expected_mass_retention": _safe_ratio(
            random_p95_total_mass,
            milp_total_mass,
        ),
        "best_lp_stage": best_lp_stage,
        "best_projection_total_completed_request_count": (
            best_projected_total_count
        ),
        "best_projection_total_expected_completed_mass": (
            best_projected_total_mass
        ),
        "best_pooled_count_retention": best_count_retention,
        "best_pooled_expected_mass_retention": best_mass_retention,
        "stage_one_quality_gate_pass": stage_one_quality_gate_pass,
        "stage_two_quality_gate_pass": stage_two_quality_gate_pass,
        "quality_gate_pass": quality_gate_pass,
        "random_control_pass": random_control_pass,
        "overall_gate_pass": quality_gate_pass and random_control_pass,
        "gate_definition": {
            "quality": (
                "At least one LP stage has residual within tolerance, a "
                "100% feasible projection, and pooled count/mass retention "
                "meeting the configured threshold"
            ),
            "random_control": (
                "The better LP-stage projection total expected mass exceeds "
                "the 95th percentile of random-score projections using the "
                "same scan"
            ),
        },
    }
    output_paths = _write_outputs(rows, summary, args.output_directory)
    summary["output_paths"] = output_paths
    # Refresh JSON after paths are known while preserving the same versioned name.
    json_text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    Path(output_paths["versioned_summary"]).write_text(
        json_text,
        encoding="utf-8",
    )
    Path(output_paths["latest_summary"]).write_text(
        json_text,
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
