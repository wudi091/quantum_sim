"""Paired statistics for the frozen formal online experiment suite."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import shutil
from statistics import fmean
from typing import Mapping, Sequence

from .comparison_methods import ordered_present_methods, validate_profile_methods


_SUPPORTED_EXPERIMENTS = {
    "paired_online_gnn_milp_qcast",
    "paired_online_gnn_milp_routing_baselines",
}

_METRICS = {
    "completed_requests": ("完成请求数", True),
    "mean_censored_latency_ps": ("平均截尾延迟（ps）", False),
    "mean_decision_seconds": ("平均决策时间（秒）", False),
    "p95_decision_seconds": ("P95 决策时间（秒）", False),
    "mean_planner_seconds": ("平均规划核心时间（秒）", False),
}

_HARD_GATE_METRICS = (
    "gnn_invalid_decision_count",
    "schedule_violation_count",
    "fidelity_violation_count",
    "physical_backend_rejection_count",
    "post_completion_validation_failure_count",
)


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else float(fmean(values))


def _balanced_mean(groups: Sequence[Sequence[float]]) -> float:
    return _mean([_mean(group) for group in groups])


def _bootstrap_ci(
    groups: Sequence[Sequence[float]],
    *,
    samples: int,
    rng: random.Random,
) -> tuple[float, float]:
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    draws = []
    for _ in range(samples):
        resampled = [
            [group[rng.randrange(len(group))] for _ in group]
            for group in groups
        ]
        draws.append(_balanced_mean(resampled))
    draws.sort()
    return (
        float(draws[int(0.025 * (samples - 1))]),
        float(draws[int(0.975 * (samples - 1))]),
    )


def _randomization_p_value(
    groups: Sequence[Sequence[float]],
    *,
    samples: int,
    rng: random.Random,
) -> float:
    if samples < 100:
        raise ValueError("randomization samples must be at least 100")
    observed = abs(_balanced_mean(groups))
    if observed == 0.0 and all(
        value == 0.0 for group in groups for value in group
    ):
        return 1.0
    extreme = 0
    for _ in range(samples):
        randomized = [
            [value if rng.random() < 0.5 else -value for value in group]
            for group in groups
        ]
        if abs(_balanced_mean(randomized)) >= observed - 1e-12:
            extreme += 1
    return float((extreme + 1) / (samples + 1))


def _validate_source_report(
    name: str,
    payload: Mapping[str, object],
) -> tuple[str, ...]:
    if payload.get("schema_version") != 1:
        raise ValueError(f"{name}: unsupported schema")
    if payload.get("experiment") not in _SUPPORTED_EXPERIMENTS:
        raise ValueError(f"{name}: unsupported experiment type")
    contract = payload.get("comparison_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"{name}: comparison contract is missing")
    for key, expected in {
        "paired_episode_spec": True,
        "independent_persistent_executors": True,
        "future_requests_hidden": True,
        "gnn_calls_milp_online": False,
    }.items():
        if contract.get(key) != expected:
            raise ValueError(f"{name}: contract mismatch for {key}")
    trials = payload.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError(f"{name}: no trials")
    seeds = [int(trial["seed"]) for trial in trials]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"{name}: duplicate seeds")
    methods: tuple[str, ...] | None = None
    for trial in trials:
        episode = trial.get("episode")
        if not isinstance(episode, Mapping):
            raise ValueError(f"{name}: trial is missing EpisodeSpec")
        if int(episode.get("seed", -1)) != int(trial["seed"]):
            raise ValueError(f"{name}: trial and EpisodeSpec seeds differ")
        trial_methods = trial.get("methods")
        if not isinstance(trial_methods, Mapping) or not trial_methods:
            raise ValueError(f"{name}: trial methods are missing")
        current = tuple(sorted(str(method) for method in trial_methods))
        if methods is None:
            methods = current
        elif methods != current:
            raise ValueError(f"{name}: method set changes across trials")
        for method_name, method in trial_methods.items():
            metrics = method.get("metrics")
            if not isinstance(metrics, Mapping):
                raise ValueError(f"{name}/{method_name}: metrics are missing")
            missing = set(_METRICS) - set(metrics)
            if missing:
                raise ValueError(
                    f"{name}/{method_name}: missing metrics {sorted(missing)}"
                )
            if not isinstance(method.get("violations"), list):
                raise ValueError(
                    f"{name}/{method_name}: violation records are missing"
                )
    assert methods is not None
    profile = contract.get("comparison_profile")
    if profile is not None:
        expected_order = validate_profile_methods(str(profile), set(methods))
        active_order = contract.get("active_method_order")
        if active_order is not None and tuple(active_order) != expected_order:
            raise ValueError(f"{name}: active method order does not match profile")
        return expected_order
    try:
        return ordered_present_methods(set(methods))
    except ValueError:
        return methods


def _metric_values(
    payload: Mapping[str, object],
    method: str,
    metric: str,
) -> list[float]:
    return [
        float(trial["methods"][method]["metrics"][metric])
        for trial in payload["trials"]
    ]


def _hard_gate_totals(
    payload: Mapping[str, object],
    method: str,
) -> dict[str, float]:
    totals = {metric: 0.0 for metric in _HARD_GATE_METRICS}
    totals["violation_record_count"] = 0.0
    for trial in payload["trials"]:
        result = trial["methods"][method]
        metrics = result["metrics"]
        for metric in _HARD_GATE_METRICS:
            totals[metric] += float(metrics.get(metric, 0.0))
        totals["violation_record_count"] += float(
            len(result.get("violations", ()))
        )
    return totals


def _summarize_pair(
    metric: str,
    reference_groups: Sequence[Sequence[float]],
    baseline_groups: Sequence[Sequence[float]],
    *,
    bootstrap_samples: int,
    randomization_samples: int,
    random_seed: int,
) -> dict[str, object]:
    higher_is_better = _METRICS[metric][1]
    advantage_groups = [
        [
            reference - baseline if higher_is_better else baseline - reference
            for reference, baseline in zip(
                reference_group,
                baseline_group,
                strict=True,
            )
        ]
        for reference_group, baseline_group in zip(
            reference_groups,
            baseline_groups,
            strict=True,
        )
    ]
    flattened = [value for group in advantage_groups for value in group]
    reference_mean = _balanced_mean(reference_groups)
    baseline_mean = _balanced_mean(baseline_groups)
    advantage_mean = _balanced_mean(advantage_groups)
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
    return {
        "label": _METRICS[metric][0],
        "higher_is_better": higher_is_better,
        "advantage_definition": "positive_means_reference_better",
        "reference_mean": reference_mean,
        "baseline_mean": baseline_mean,
        "reference_advantage_mean": advantage_mean,
        "reference_to_baseline_ratio": (
            reference_mean / baseline_mean if baseline_mean else None
        ),
        "baseline_to_reference_ratio": (
            baseline_mean / reference_mean if reference_mean else None
        ),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "paired_randomization_p": p_value,
        "reference_wins": sum(value > 1e-12 for value in flattened),
        "ties": sum(abs(value) <= 1e-12 for value in flattened),
        "baseline_wins": sum(value < -1e-12 for value in flattened),
    }


def analyze_cases(
    payloads: Mapping[str, Mapping[str, object]],
    *,
    reference: str,
    bootstrap_samples: int = 20_000,
    randomization_samples: int = 20_000,
    random_seed: int = 20260816,
) -> dict[str, object]:
    if not payloads:
        raise ValueError("at least one case is required")
    method_sets = {}
    method_orders = {}
    checkpoint_hashes = set()
    for case_name, payload in payloads.items():
        methods = _validate_source_report(case_name, payload)
        if reference not in methods:
            raise ValueError(f"{case_name}: reference method {reference!r} missing")
        method_sets[case_name] = set(methods)
        method_orders[case_name] = methods
        checkpoint_hashes.add(str(payload.get("checkpoint_sha256", "")))
    if "" in checkpoint_hashes or len(checkpoint_hashes) != 1:
        raise ValueError("all cases must use one recorded checkpoint")

    cases = {}
    all_hard_gates: dict[str, dict[str, float]] = {}
    for case_index, (case_name, payload) in enumerate(payloads.items()):
        methods = method_orders[case_name]
        means = {
            method: {
                metric: _mean(_metric_values(payload, method, metric))
                for metric in _METRICS
            }
            for method in methods
        }
        hard_gates = {
            method: _hard_gate_totals(payload, method) for method in methods
        }
        all_hard_gates[case_name] = hard_gates
        comparisons = {}
        for baseline_index, baseline in enumerate(
            method for method in methods if method != reference
        ):
            metrics = {}
            for metric_index, metric in enumerate(_METRICS):
                metrics[metric] = _summarize_pair(
                    metric,
                    [_metric_values(payload, reference, metric)],
                    [_metric_values(payload, baseline, metric)],
                    bootstrap_samples=bootstrap_samples,
                    randomization_samples=randomization_samples,
                    random_seed=(
                        random_seed
                        + case_index * 1000
                        + baseline_index * 100
                        + metric_index * 2
                    ),
                )
            comparisons[baseline] = {"metrics": metrics}
        valid = all(
            value == 0.0
            for method_totals in hard_gates.values()
            for value in method_totals.values()
        )
        cases[case_name] = {
            "trial_count": len(payload["trials"]),
            "scenario": payload.get("scenario"),
            "method_means": means,
            "comparisons": comparisons,
            "hard_gates": hard_gates,
            "valid": valid,
        }

    common_methods = set.intersection(*method_sets.values())
    try:
        common_method_order = ordered_present_methods(common_methods)
    except ValueError:
        common_method_order = tuple(sorted(common_methods))
    common_baselines = tuple(
        method for method in common_method_order if method != reference
    )
    overall_means = {
        method: {
            metric: _balanced_mean([
                _metric_values(payload, method, metric)
                for payload in payloads.values()
            ])
            for metric in _METRICS
        }
        for method in common_method_order
    }
    overall_comparisons = {}
    for baseline_index, baseline in enumerate(common_baselines):
        metrics = {}
        for metric_index, metric in enumerate(_METRICS):
            metrics[metric] = _summarize_pair(
                metric,
                [
                    _metric_values(payload, reference, metric)
                    for payload in payloads.values()
                ],
                [
                    _metric_values(payload, baseline, metric)
                    for payload in payloads.values()
                ],
                bootstrap_samples=bootstrap_samples,
                randomization_samples=randomization_samples,
                random_seed=(
                    random_seed
                    + 100_000
                    + baseline_index * 100
                    + metric_index * 2
                ),
            )
        overall_comparisons[baseline] = {"metrics": metrics}
    overall_valid = all(case["valid"] for case in cases.values())
    return {
        "schema_version": 1,
        "experiment": "formal_paired_online_comparison",
        "analysis_contract": {
            "paired_episode_specs": True,
            "balanced_case_weighting": True,
            "reference_method": reference,
            "positive_advantage_means": "reference_better",
            "confidence_level": 0.95,
            "bootstrap_samples": bootstrap_samples,
            "randomization_samples": randomization_samples,
            "random_seed": random_seed,
        },
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "case_count": len(cases),
        "paired_trial_count": sum(case["trial_count"] for case in cases.values()),
        "cases": cases,
        "overall": {
            "common_methods": list(common_method_order),
            "method_means": overall_means,
            "comparisons": overall_comparisons,
            "valid": overall_valid,
        },
    }


def load_case_reports(
    reports: Sequence[tuple[str, Path]],
) -> tuple[dict[str, Mapping[str, object]], list[str]]:
    payloads = {}
    inputs = []
    for name, path in reports:
        if name in payloads:
            raise ValueError(f"duplicate case name: {name}")
        payloads[name] = json.loads(path.read_text(encoding="utf-8"))
        inputs.append(str(path))
    return payloads, inputs


def load_variant_reports(
    reports: Sequence[tuple[str, Path]],
    *,
    reference: str,
    case_name: str,
) -> tuple[dict[str, Mapping[str, object]], list[str]]:
    if len(reports) < 2:
        raise ValueError("variant mode requires at least two reports")
    raw = {}
    inputs = []
    for label, path in reports:
        if label in raw:
            raise ValueError(f"duplicate variant label: {label}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        methods = _validate_source_report(label, payload)
        if methods != ("gnn",):
            raise ValueError(f"{label}: variant report must contain only GNN")
        raw[label] = payload
        inputs.append(str(path))
    if reference not in raw:
        raise ValueError(f"reference variant {reference!r} is missing")
    reference_payload = raw[reference]
    checkpoint = reference_payload.get("checkpoint_sha256")
    scenario = reference_payload.get("scenario")
    reference_trials = {
        int(trial["seed"]): trial for trial in reference_payload["trials"]
    }
    combined_trials = []
    for seed in sorted(reference_trials):
        reference_trial = reference_trials[seed]
        combined_trials.append({
            "seed": seed,
            "episode": reference_trial["episode"],
            "methods": {},
        })
    by_seed = {trial["seed"]: trial for trial in combined_trials}
    for label, payload in raw.items():
        if payload.get("checkpoint_sha256") != checkpoint:
            raise ValueError(f"{label}: checkpoint hash differs")
        if payload.get("scenario") != scenario:
            raise ValueError(f"{label}: scenario differs")
        trials = {int(trial["seed"]): trial for trial in payload["trials"]}
        if set(trials) != set(reference_trials):
            raise ValueError(f"{label}: paired seed set differs")
        for seed, trial in trials.items():
            if trial["episode"] != reference_trials[seed]["episode"]:
                raise ValueError(f"{label}: EpisodeSpec differs for seed {seed}")
            by_seed[seed]["methods"][label] = trial["methods"]["gnn"]
    synthetic = {
        "schema_version": 1,
        "experiment": "paired_online_gnn_milp_routing_baselines",
        "comparison_contract": {
            "paired_episode_spec": True,
            "independent_persistent_executors": True,
            "future_requests_hidden": True,
            "gnn_calls_milp_online": False,
        },
        "checkpoint_sha256": checkpoint,
        "scenario": scenario,
        "trials": combined_trials,
    }
    return {case_name: synthetic}, inputs


def _markdown(analysis: Mapping[str, object]) -> str:
    reference = analysis["analysis_contract"]["reference_method"]
    lines = [
        "# 正式在线实验配对统计",
        "",
        f"- 参考方法：`{reference}`",
        f"- 场景数：{analysis['case_count']}",
        f"- 配对 episode 数：{analysis['paired_trial_count']}",
        f"- 所有硬门槛：{'通过' if analysis['overall']['valid'] else '失败'}",
        "",
    ]
    for case_name, case in analysis["cases"].items():
        lines.extend([
            f"## {case_name}",
            "",
            "| 方法 | 完成请求数 | 平均截尾延迟（ps） | 平均决策时间（秒） |",
            "|---|---:|---:|---:|",
        ])
        for method, means in case["method_means"].items():
            lines.append(
                f"| {method} | {means['completed_requests']:.3f} | "
                f"{means['mean_censored_latency_ps']:.3f} | "
                f"{means['mean_decision_seconds']:.6f} |"
            )
        lines.extend([
            "",
            f"| 对比（{reference} vs） | 完成量优势 | 95% CI | p 值 | 胜/平/负 |",
            "|---|---:|---:|---:|---:|",
        ])
        for baseline, comparison in case["comparisons"].items():
            item = comparison["metrics"]["completed_requests"]
            lines.append(
                f"| {baseline} | {item['reference_advantage_mean']:.3f} | "
                f"[{item['ci95_low']:.3f}, {item['ci95_high']:.3f}] | "
                f"{item['paired_randomization_p']:.6f} | "
                f"{item['reference_wins']}/{item['ties']}/{item['baseline_wins']} |"
            )
        lines.append("")
    if analysis["case_count"] > 1:
        lines.extend([
            "## 跨场景平衡统计",
            "",
            f"| 对比（{reference} vs） | 完成量优势 | 95% CI | p 值 | 胜/平/负 |",
            "|---|---:|---:|---:|---:|",
        ])
        for baseline, comparison in analysis["overall"]["comparisons"].items():
            item = comparison["metrics"]["completed_requests"]
            lines.append(
                f"| {baseline} | {item['reference_advantage_mean']:.3f} | "
                f"[{item['ci95_low']:.3f}, {item['ci95_high']:.3f}] | "
                f"{item['paired_randomization_p']:.6f} | "
                f"{item['reference_wins']}/{item['ties']}/{item['baseline_wins']} |"
            )
        lines.append("")
    return "\n".join(lines)


def save_analysis(
    analysis: Mapping[str, object],
    output_directory: Path,
) -> dict[str, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"formal_comparison_{timestamp}"
    paths = {
        "json": output_directory / f"{stem}.json",
        "csv": output_directory / f"{stem}.csv",
        "markdown": output_directory / f"{stem}.md",
        "latest_json": output_directory / "formal_comparison.json",
        "latest_csv": output_directory / "formal_comparison.csv",
        "latest_markdown": output_directory / "formal_comparison.md",
    }
    paths["json"].write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with paths["csv"].open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = (
            "case",
            "reference",
            "baseline",
            "metric",
            "reference_mean",
            "baseline_mean",
            "reference_advantage_mean",
            "ci95_low",
            "ci95_high",
            "paired_randomization_p",
            "reference_wins",
            "ties",
            "baseline_wins",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        reference = analysis["analysis_contract"]["reference_method"]
        for case_name, case in analysis["cases"].items():
            for baseline, comparison in case["comparisons"].items():
                for metric, item in comparison["metrics"].items():
                    writer.writerow({
                        "case": case_name,
                        "reference": reference,
                        "baseline": baseline,
                        "metric": metric,
                        **{key: item[key] for key in fieldnames[4:]},
                    })
    paths["markdown"].write_text(_markdown(analysis), encoding="utf-8")
    shutil.copyfile(paths["json"], paths["latest_json"])
    shutil.copyfile(paths["csv"], paths["latest_csv"])
    shutil.copyfile(paths["markdown"], paths["latest_markdown"])
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze modern paired online comparison reports."
    )
    parser.add_argument(
        "--case",
        nargs=2,
        action="append",
        metavar=("NAME", "REPORT"),
    )
    parser.add_argument(
        "--variant",
        nargs=2,
        action="append",
        metavar=("LABEL", "REPORT"),
    )
    parser.add_argument("--variant-case-name", default="variants")
    parser.add_argument("--reference", default="gnn")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--randomization-samples", type=int, default=20_000)
    parser.add_argument("--random-seed", type=int, default=20260816)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.case) == bool(args.variant):
        raise ValueError("choose exactly one of --case or --variant mode")
    if args.case:
        payloads, inputs = load_case_reports([
            (name, Path(path)) for name, path in args.case
        ])
    else:
        payloads, inputs = load_variant_reports(
            [(label, Path(path)) for label, path in args.variant],
            reference=args.reference,
            case_name=args.variant_case_name,
        )
    analysis = analyze_cases(
        payloads,
        reference=args.reference,
        bootstrap_samples=args.bootstrap_samples,
        randomization_samples=args.randomization_samples,
        random_seed=args.random_seed,
    )
    analysis["inputs"] = inputs
    paths = save_analysis(analysis, args.output)
    print(f"hard_gates_valid={analysis['overall']['valid']}")
    print(f"json: {paths['json']}")
    print(f"markdown: {paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
