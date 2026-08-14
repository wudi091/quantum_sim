"""Analyze paired adaptive-versus-fixed construction GNN reports."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import shutil
from typing import Mapping, Sequence

from .analyze_online_benchmark import (
    _balanced_mean,
    _bootstrap_ci,
    _randomization_p_value,
)


_METRICS = {
    "completed_requests": True,
    "mean_censored_latency_ps": False,
    "mean_decision_seconds": False,
}

_HARD_GATE_METRICS = (
    "gnn_invalid_decision_count",
    "schedule_violation_count",
    "fidelity_violation_count",
    "physical_backend_rejection_count",
    "post_completion_validation_failure_count",
)


def _validate_report(
    case_name: str,
    payload: Mapping[str, object],
    *,
    adaptive: bool,
) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError(f"{case_name}: unsupported report schema")
    if payload.get("experiment") != "paired_online_gnn_milp_qcast":
        raise ValueError(f"{case_name}: unexpected experiment type")
    contract = payload.get("comparison_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"{case_name}: comparison contract is missing")
    required = {
        "paired_episode_spec": True,
        "independent_persistent_executors": True,
        "future_requests_hidden": True,
        "gnn_calls_milp_online": False,
        "qcast_included": False,
    }
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise ValueError(f"{case_name}: contract mismatch for {key}")
    policy = str(contract.get("gnn_construction_policy", ""))
    if adaptive and policy != "adaptive_swap_tree_selection":
        raise ValueError(f"{case_name}: adaptive report uses {policy!r}")
    if not adaptive and not policy.startswith("fixed_swap_tree_"):
        raise ValueError(f"{case_name}: fixed report uses {policy!r}")
    configuration = payload.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError(f"{case_name}: configuration is missing")
    if configuration.get("skip_milp") is not True:
        raise ValueError(f"{case_name}: online MILP must be disabled")
    if configuration.get("skip_qcast") is not True:
        raise ValueError(f"{case_name}: Q-CAST must be skipped")
    if payload.get("milp_config") is not None:
        raise ValueError(f"{case_name}: report unexpectedly contains MILP config")
    if payload.get("qcast_config") is not None:
        raise ValueError(f"{case_name}: report unexpectedly contains Q-CAST config")
    fixed_index = configuration.get("fixed_swap_tree_index")
    if adaptive and fixed_index is not None:
        raise ValueError(f"{case_name}: adaptive report fixes a swap tree")
    if not adaptive and policy != f"fixed_swap_tree_{fixed_index}":
        raise ValueError(f"{case_name}: fixed policy and tree index disagree")
    gnn_config = payload.get("gnn_config")
    if not isinstance(gnn_config, Mapping):
        raise ValueError(f"{case_name}: GNN configuration is missing")
    if adaptive:
        expected_count = configuration.get("construction_plans")
        if gnn_config.get("construction_kinds") not in ([], ()):
            raise ValueError(f"{case_name}: adaptive construction set is restricted")
        if gnn_config.get("swap_tree_count") != expected_count:
            raise ValueError(f"{case_name}: adaptive swap-tree count is incorrect")
    else:
        construction_kind = policy.removeprefix("fixed_")
        if gnn_config.get("construction_kinds") not in (
            [construction_kind],
            (construction_kind,),
        ):
            raise ValueError(f"{case_name}: fixed construction kind is incorrect")
        if gnn_config.get("swap_tree_count") is not None:
            raise ValueError(f"{case_name}: fixed report also generates swap trees")
    trials = payload.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError(f"{case_name}: report contains no trials")
    seeds = [int(trial["seed"]) for trial in trials]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"{case_name}: duplicate trial seeds")
    for trial in trials:
        episode = trial.get("episode")
        if not isinstance(episode, Mapping):
            raise ValueError(f"{case_name}: trial is missing its EpisodeSpec")
        if int(episode.get("seed", -1)) != int(trial["seed"]):
            raise ValueError(f"{case_name}: trial and EpisodeSpec seeds differ")
        methods = trial.get("methods")
        if not isinstance(methods, Mapping) or set(methods) != {"gnn"}:
            raise ValueError(f"{case_name}: report must contain only GNN")
        method = methods["gnn"]
        metrics = method.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"{case_name}: GNN metrics are missing")
        required_metrics = set(_METRICS) | set(_HARD_GATE_METRICS)
        missing = required_metrics - set(metrics)
        if missing:
            raise ValueError(
                f"{case_name}: GNN metrics are missing {sorted(missing)}"
            )
        if not isinstance(method.get("violations"), list):
            raise ValueError(f"{case_name}: violation records are missing")


def _shared_configuration(payload: Mapping[str, object]) -> dict[str, object]:
    configuration = dict(payload["configuration"])
    configuration.pop("output", None)
    configuration.pop("fixed_swap_tree_index", None)
    return configuration


def _pair_trials(
    case_name: str,
    adaptive: Mapping[str, object],
    fixed: Mapping[str, object],
) -> list[tuple[Mapping[str, object], Mapping[str, object]]]:
    if adaptive.get("checkpoint_sha256") != fixed.get("checkpoint_sha256"):
        raise ValueError(f"{case_name}: checkpoint hashes differ")
    if adaptive.get("scenario") != fixed.get("scenario"):
        raise ValueError(f"{case_name}: scenario configurations differ")
    if _shared_configuration(adaptive) != _shared_configuration(fixed):
        raise ValueError(f"{case_name}: non-construction configurations differ")
    adaptive_by_seed = {
        int(trial["seed"]): trial for trial in adaptive["trials"]
    }
    fixed_by_seed = {int(trial["seed"]): trial for trial in fixed["trials"]}
    if set(adaptive_by_seed) != set(fixed_by_seed):
        raise ValueError(f"{case_name}: paired seed sets differ")
    pairs = []
    for seed in sorted(adaptive_by_seed):
        adaptive_trial = adaptive_by_seed[seed]
        fixed_trial = fixed_by_seed[seed]
        if adaptive_trial.get("episode") != fixed_trial.get("episode"):
            raise ValueError(f"{case_name}: EpisodeSpec differs for seed {seed}")
        pairs.append((adaptive_trial, fixed_trial))
    return pairs


def _metric_values(
    pairs: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
    metric: str,
) -> tuple[list[float], list[float]]:
    adaptive = []
    fixed = []
    for adaptive_trial, fixed_trial in pairs:
        adaptive.append(float(
            adaptive_trial["methods"]["gnn"]["metrics"][metric]
        ))
        fixed.append(float(fixed_trial["methods"]["gnn"]["metrics"][metric]))
    return adaptive, fixed


def _summarize_metric(
    metric: str,
    adaptive_groups: Sequence[Sequence[float]],
    fixed_groups: Sequence[Sequence[float]],
    *,
    bootstrap_samples: int,
    randomization_samples: int,
    random_seed: int,
) -> dict[str, object]:
    higher_is_better = _METRICS[metric]
    advantage_groups = [
        [
            adaptive - fixed if higher_is_better else fixed - adaptive
            for adaptive, fixed in zip(
                adaptive_group,
                fixed_group,
                strict=True,
            )
        ]
        for adaptive_group, fixed_group in zip(
            adaptive_groups,
            fixed_groups,
            strict=True,
        )
    ]
    flattened = [value for group in advantage_groups for value in group]
    ci_low, ci_high = _bootstrap_ci(
        advantage_groups,
        samples=bootstrap_samples,
        rng=random.Random(random_seed),
    )
    p_value = _randomization_p_value(
        advantage_groups,
        samples=randomization_samples,
        rng=random.Random(random_seed + 1),
    )
    adaptive_mean = _balanced_mean(adaptive_groups)
    fixed_mean = _balanced_mean(fixed_groups)
    advantage_mean = _balanced_mean(advantage_groups)
    return {
        "advantage_definition": "positive_means_adaptive_better",
        "adaptive_mean": adaptive_mean,
        "fixed_mean": fixed_mean,
        "adaptive_advantage_mean": advantage_mean,
        "adaptive_relative_improvement_percent": (
            100.0 * advantage_mean / fixed_mean if fixed_mean else None
        ),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "paired_randomization_p": p_value,
        "adaptive_wins": sum(value > 1e-12 for value in flattened),
        "ties": sum(abs(value) <= 1e-12 for value in flattened),
        "fixed_wins": sum(value < -1e-12 for value in flattened),
        "all_paired_advantages_zero": all(value == 0.0 for value in flattened),
    }


def _hard_gate_totals(
    pairs: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
) -> dict[str, float]:
    totals = {
        f"{variant}_{metric}": 0.0
        for variant in ("adaptive", "fixed")
        for metric in _HARD_GATE_METRICS
    }
    totals["adaptive_violation_record_count"] = 0.0
    totals["fixed_violation_record_count"] = 0.0
    for adaptive_trial, fixed_trial in pairs:
        for variant, trial in (
            ("adaptive", adaptive_trial),
            ("fixed", fixed_trial),
        ):
            method = trial["methods"]["gnn"]
            metrics = method["metrics"]
            for metric in _HARD_GATE_METRICS:
                totals[f"{variant}_{metric}"] += float(metrics.get(metric, 0.0))
            totals[f"{variant}_violation_record_count"] += len(
                method.get("violations", ())
            )
    return totals


def _throughput_verdict(
    completed: Mapping[str, object],
    *,
    valid: bool,
) -> str:
    if not valid:
        return "invalid_hard_gate_failure"
    low = float(completed["ci95_low"])
    high = float(completed["ci95_high"])
    p_value = float(completed["paired_randomization_p"])
    if low > 0.0 and p_value < 0.05:
        return "adaptive_better_throughput"
    if high < 0.0 and p_value < 0.05:
        return "fixed_better_throughput"
    return "inconclusive_throughput"


def analyze_pairs(
    reports: Sequence[tuple[str, Path, Path]],
    *,
    bootstrap_samples: int = 20_000,
    randomization_samples: int = 20_000,
    random_seed: int = 20260814,
) -> dict[str, object]:
    if not reports:
        raise ValueError("at least one report pair is required")
    cases: dict[str, object] = {}
    metric_groups = {
        metric: {"adaptive": [], "fixed": []} for metric in _METRICS
    }
    all_pairs = []
    checkpoint_hashes = set()
    inputs = []
    for case_index, (case_name, adaptive_path, fixed_path) in enumerate(reports):
        if case_name in cases:
            raise ValueError(f"duplicate case name: {case_name}")
        adaptive = json.loads(adaptive_path.read_text(encoding="utf-8"))
        fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
        _validate_report(case_name, adaptive, adaptive=True)
        _validate_report(case_name, fixed, adaptive=False)
        pairs = _pair_trials(case_name, adaptive, fixed)
        all_pairs.extend(pairs)
        checkpoint_hashes.add(str(adaptive.get("checkpoint_sha256", "")))
        checkpoint_hashes.add(str(fixed.get("checkpoint_sha256", "")))
        inputs.extend((str(adaptive_path), str(fixed_path)))
        metrics = {}
        for metric_index, metric in enumerate(_METRICS):
            adaptive_values, fixed_values = _metric_values(pairs, metric)
            metric_groups[metric]["adaptive"].append(adaptive_values)
            metric_groups[metric]["fixed"].append(fixed_values)
            metrics[metric] = _summarize_metric(
                metric,
                [adaptive_values],
                [fixed_values],
                bootstrap_samples=bootstrap_samples,
                randomization_samples=randomization_samples,
                random_seed=random_seed + case_index * 100 + metric_index * 2,
            )
        hard_gates = _hard_gate_totals(pairs)
        valid = all(value == 0.0 for value in hard_gates.values())
        cases[case_name] = {
            "trial_count": len(pairs),
            "fixed_policy": fixed["comparison_contract"][
                "gnn_construction_policy"
            ],
            "test_seed_start": min(int(pair[0]["seed"]) for pair in pairs),
            "test_seed_end": max(int(pair[0]["seed"]) for pair in pairs),
            "hard_gates": hard_gates,
            "valid": valid,
            "throughput_verdict": _throughput_verdict(
                metrics["completed_requests"],
                valid=valid,
            ),
            "metrics": metrics,
        }
    if "" in checkpoint_hashes or len(checkpoint_hashes) != 1:
        raise ValueError("all report pairs must use one checkpoint")
    overall_metrics = {}
    for metric_index, metric in enumerate(_METRICS):
        overall_metrics[metric] = _summarize_metric(
            metric,
            metric_groups[metric]["adaptive"],
            metric_groups[metric]["fixed"],
            bootstrap_samples=bootstrap_samples,
            randomization_samples=randomization_samples,
            random_seed=random_seed + 10_000 + metric_index * 2,
        )
    overall_hard_gates = _hard_gate_totals(all_pairs)
    overall_valid = all(value == 0.0 for value in overall_hard_gates.values())
    return {
        "schema_version": 1,
        "experiment": "online_construction_awareness_ablation",
        "analysis_contract": {
            "paired_episode_specs": True,
            "balanced_case_weighting": True,
            "primary_metric": "completed_requests",
            "secondary_metric": "mean_censored_latency_ps",
            "runtime_metric": "mean_decision_seconds",
            "positive_advantage_means": "adaptive_better",
            "confidence_level": 0.95,
            "bootstrap_samples": bootstrap_samples,
            "randomization_samples": randomization_samples,
            "random_seed": random_seed,
        },
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "case_count": len(cases),
        "paired_trial_count": sum(case["trial_count"] for case in cases.values()),
        "inputs": inputs,
        "cases": cases,
        "overall": {
            "hard_gates": overall_hard_gates,
            "valid": overall_valid,
            "throughput_verdict": _throughput_verdict(
                overall_metrics["completed_requests"],
                valid=overall_valid,
            ),
            "metrics": overall_metrics,
        },
    }


def save_analysis(
    analysis: Mapping[str, object],
    output_directory: Path,
) -> dict[str, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = ""
    index = 1
    while (
        output_directory / f"construction_ablation_{timestamp}{suffix}.json"
    ).exists():
        index += 1
        suffix = f"_{index}"
    stem = f"construction_ablation_{timestamp}{suffix}"
    paths = {
        "json": output_directory / f"{stem}.json",
        "csv": output_directory / f"{stem}.csv",
        "latest_json": output_directory / "construction_ablation.json",
        "latest_csv": output_directory / "construction_ablation.csv",
    }
    paths["json"].write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with paths["csv"].open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = (
            "case",
            "metric",
            "adaptive_mean",
            "fixed_mean",
            "adaptive_advantage_mean",
            "adaptive_relative_improvement_percent",
            "ci95_low",
            "ci95_high",
            "paired_randomization_p",
            "adaptive_wins",
            "ties",
            "fixed_wins",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        rows = (*analysis["cases"].items(), ("overall", analysis["overall"]))
        for case_name, case in rows:
            for metric, values in case["metrics"].items():
                writer.writerow({
                    "case": case_name,
                    "metric": metric,
                    **{key: values[key] for key in fieldnames[2:]},
                })
    shutil.copyfile(paths["json"], paths["latest_json"])
    shutil.copyfile(paths["csv"], paths["latest_csv"])
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze adaptive-versus-fixed construction GNN reports."
    )
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        required=True,
        metavar=("CASE", "ADAPTIVE_REPORT", "FIXED_REPORT"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--randomization-samples", type=int, default=20_000)
    parser.add_argument("--random-seed", type=int, default=20260814)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reports = [
        (case_name, Path(adaptive), Path(fixed))
        for case_name, adaptive, fixed in args.pair
    ]
    analysis = analyze_pairs(
        reports,
        bootstrap_samples=args.bootstrap_samples,
        randomization_samples=args.randomization_samples,
        random_seed=args.random_seed,
    )
    paths = save_analysis(analysis, args.output)
    completed = analysis["overall"]["metrics"]["completed_requests"]
    print(
        f"completed_advantage={completed['adaptive_advantage_mean']:.4f} "
        f"ci95=[{completed['ci95_low']:.4f}, {completed['ci95_high']:.4f}] "
        f"p={completed['paired_randomization_p']:.6f}"
    )
    print(f"hard_gates_valid={analysis['overall']['valid']}")
    print(f"json: {paths['json']}")
    print(f"csv: {paths['csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
