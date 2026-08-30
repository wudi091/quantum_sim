"""Statistical analysis for paired TELGEN/Q-CAST online comparisons."""

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


_CONTRACT_FLAGS = {
    "paired_episode_spec": True,
    "independent_persistent_executors": True,
    "common_runtime_metric": "mean_decision_seconds",
    "qcast_baseline": "width_one_residual_ext_with_recovery",
    "qcast_uses_telgen_lp_or_search_decoder": False,
}

_METRICS = {
    "completed_requests": {
        "label": "完成请求数",
        "higher_is_better": True,
        "role": "primary",
    },
    "mean_censored_latency_ps": {
        "label": "平均删失完成延迟（ps）",
        "higher_is_better": False,
        "role": "secondary",
    },
    "mean_decision_seconds": {
        "label": "平均决策时间（秒）",
        "higher_is_better": False,
        "role": "runtime",
    },
}

_NON_FATAL_SCHEDULE_CODES = frozenset({"slot_completion_overrun"})

_HARD_GATE_METRICS = (
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
    draws: list[float] = []
    for _ in range(samples):
        resampled_groups = [
            [group[rng.randrange(len(group))] for _ in group]
            for group in groups
        ]
        draws.append(_balanced_mean(resampled_groups))
    draws.sort()
    lower = draws[int(0.025 * (samples - 1))]
    upper = draws[int(0.975 * (samples - 1))]
    return float(lower), float(upper)


def _randomization_p_value(
    groups: Sequence[Sequence[float]],
    *,
    samples: int,
    rng: random.Random,
) -> float:
    if samples < 100:
        raise ValueError("randomization samples must be at least 100")
    observed = abs(_balanced_mean(groups))
    if observed == 0.0 and all(value == 0.0 for group in groups for value in group):
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


def _metric_advantage(metric: str, telgen: float, qcast: float) -> float:
    if _METRICS[metric]["higher_is_better"]:
        return float(telgen - qcast)
    return float(qcast - telgen)


def _summarize_metric(
    metric: str,
    telgen_groups: Sequence[Sequence[float]],
    qcast_groups: Sequence[Sequence[float]],
    *,
    bootstrap_samples: int,
    randomization_samples: int,
    random_seed: int,
) -> dict[str, object]:
    advantage_groups = [
        [
            _metric_advantage(metric, telgen, qcast)
            for telgen, qcast in zip(telgen_group, qcast_group, strict=True)
        ]
        for telgen_group, qcast_group in zip(
            telgen_groups,
            qcast_groups,
            strict=True,
        )
    ]
    flattened = [value for group in advantage_groups for value in group]
    wins = sum(value > 1e-12 for value in flattened)
    losses = sum(value < -1e-12 for value in flattened)
    ties = len(flattened) - wins - losses
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
        "label": _METRICS[metric]["label"],
        "role": _METRICS[metric]["role"],
        "advantage_definition": "positive_means_telgen_better",
        "telgen_mean": _balanced_mean(telgen_groups),
        "qcast_mean": _balanced_mean(qcast_groups),
        "telgen_advantage_mean": _balanced_mean(advantage_groups),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "paired_randomization_p": p_value,
        "telgen_wins": wins,
        "ties": ties,
        "qcast_wins": losses,
        "all_paired_advantages_zero": all(value == 0.0 for value in flattened),
    }


def _quality_verdict(
    metrics: Mapping[str, Mapping[str, object]],
    *,
    valid: bool,
) -> str:
    if not valid:
        return "invalid_hard_gate_failure"
    completion = metrics["completed_requests"]
    completion_low = float(completion["ci95_low"])
    completion_high = float(completion["ci95_high"])
    completion_p = float(completion["paired_randomization_p"])
    if completion_low > 0.0 and completion_p < 0.05:
        return "telgen_better_throughput"
    if completion_high < 0.0 and completion_p < 0.05:
        return "qcast_better_throughput"
    if not bool(completion["all_paired_advantages_zero"]):
        return "inconclusive_throughput"
    latency = metrics["mean_censored_latency_ps"]
    latency_low = float(latency["ci95_low"])
    latency_high = float(latency["ci95_high"])
    latency_p = float(latency["paired_randomization_p"])
    if latency_low > 0.0 and latency_p < 0.05:
        return "telgen_better_latency_at_equal_throughput"
    if latency_high < 0.0 and latency_p < 0.05:
        return "qcast_better_latency_at_equal_throughput"
    return "statistically_inconclusive"


def _runtime_verdict(metrics: Mapping[str, Mapping[str, object]]) -> str:
    runtime = metrics["mean_decision_seconds"]
    low = float(runtime["ci95_low"])
    high = float(runtime["ci95_high"])
    p_value = float(runtime["paired_randomization_p"])
    if low > 0.0 and p_value < 0.05:
        return "telgen_faster"
    if high < 0.0 and p_value < 0.05:
        return "qcast_faster"
    return "runtime_inconclusive"


def _validate_payload(case_name: str, payload: Mapping[str, object]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError(f"{case_name}: unsupported comparison schema")
    contract = payload.get("comparison_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"{case_name}: comparison contract is missing")
    for key, expected in _CONTRACT_FLAGS.items():
        if contract.get(key) != expected:
            raise ValueError(f"{case_name}: comparison contract mismatch for {key}")
    trials = payload.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError(f"{case_name}: no paired trials")
    seeds = [trial["seed"] for trial in trials]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"{case_name}: duplicate seeds")


def _case_groups(
    payload: Mapping[str, object],
    metric: str,
) -> tuple[list[float], list[float]]:
    telgen: list[float] = []
    qcast: list[float] = []
    for trial in payload["trials"]:
        telgen.append(float(trial["telgen"]["metrics"][metric]))
        qcast.append(float(trial["qcast"]["metrics"][metric]))
    return telgen, qcast


def _hard_gate_totals(payloads: Mapping[str, Mapping[str, object]]) -> dict[str, float]:
    totals = {
        f"{method}_{metric}": 0.0
        for method in ("telgen", "qcast")
        for metric in _HARD_GATE_METRICS
    }
    for method in ("telgen", "qcast"):
        totals[f"{method}_unsafe_schedule_violation_count"] = 0.0
        totals[f"{method}_nominal_completion_overrun_count"] = 0.0
    for payload in payloads.values():
        for trial in payload["trials"]:
            for method in ("telgen", "qcast"):
                metrics = trial[method]["metrics"]
                for metric in _HARD_GATE_METRICS:
                    totals[f"{method}_{metric}"] += float(metrics.get(metric, 0.0))
                for violation in trial[method].get("violations", ()):
                    code = str(violation.get("code", ""))
                    if code in _NON_FATAL_SCHEDULE_CODES:
                        totals[f"{method}_nominal_completion_overrun_count"] += 1.0
                    else:
                        totals[f"{method}_unsafe_schedule_violation_count"] += 1.0
    return totals


def _hard_gates_valid(totals: Mapping[str, float]) -> bool:
    return all(
        value == 0.0
        for name, value in totals.items()
        if not name.endswith("_nominal_completion_overrun_count")
    )


def analyze_online_payloads(
    payloads: Mapping[str, Mapping[str, object]],
    *,
    bootstrap_samples: int = 20_000,
    randomization_samples: int = 20_000,
    random_seed: int = 20260811,
) -> dict[str, object]:
    if not payloads:
        raise ValueError("at least one comparison payload is required")
    for case_name, payload in payloads.items():
        _validate_payload(case_name, payload)

    case_results: dict[str, object] = {}
    for case_index, (case_name, payload) in enumerate(payloads.items()):
        metrics: dict[str, object] = {}
        for metric_index, metric in enumerate(_METRICS):
            telgen, qcast = _case_groups(payload, metric)
            metrics[metric] = _summarize_metric(
                metric,
                [telgen],
                [qcast],
                bootstrap_samples=bootstrap_samples,
                randomization_samples=randomization_samples,
                random_seed=random_seed + case_index * 100 + metric_index * 2,
            )
        gate_payload = {case_name: payload}
        hard_gates = _hard_gate_totals(gate_payload)
        valid = _hard_gates_valid(hard_gates)
        case_results[case_name] = {
            "trial_count": len(payload["trials"]),
            "scenario": payload["scenario"],
            "hard_gates": hard_gates,
            "valid": valid,
            "quality_verdict": _quality_verdict(metrics, valid=valid),
            "runtime_verdict": _runtime_verdict(metrics),
            "metrics": metrics,
        }

    overall_metrics: dict[str, object] = {}
    for metric_index, metric in enumerate(_METRICS):
        telgen_groups = []
        qcast_groups = []
        for payload in payloads.values():
            telgen, qcast = _case_groups(payload, metric)
            telgen_groups.append(telgen)
            qcast_groups.append(qcast)
        overall_metrics[metric] = _summarize_metric(
            metric,
            telgen_groups,
            qcast_groups,
            bootstrap_samples=bootstrap_samples,
            randomization_samples=randomization_samples,
            random_seed=random_seed + 10_000 + metric_index * 2,
        )
    hard_gates = _hard_gate_totals(payloads)
    valid = _hard_gates_valid(hard_gates)
    return {
        "schema_version": 1,
        "analysis_contract": {
            "paired_trials": True,
            "balanced_case_weighting": True,
            "primary_metric": "completed_requests",
            "secondary_metric": "mean_censored_latency_ps",
            "runtime_metric": "mean_decision_seconds",
            "positive_advantage_means": "telgen_better",
            "quality_rule": (
                "compare throughput first; use latency only when every paired "
                "throughput difference is zero"
            ),
            "non_fatal_schedule_codes": sorted(_NON_FATAL_SCHEDULE_CODES),
            "confidence_level": 0.95,
            "bootstrap_samples": bootstrap_samples,
            "randomization_samples": randomization_samples,
            "random_seed": random_seed,
        },
        "case_count": len(payloads),
        "paired_trial_count": sum(len(payload["trials"]) for payload in payloads.values()),
        "cases": case_results,
        "overall": {
            "hard_gates": hard_gates,
            "valid": valid,
            "quality_verdict": _quality_verdict(overall_metrics, valid=valid),
            "runtime_verdict": _runtime_verdict(overall_metrics),
            "metrics": overall_metrics,
        },
    }


def analyze_online_reports(
    report_paths: Sequence[str | Path],
    **kwargs: object,
) -> dict[str, object]:
    payloads: dict[str, Mapping[str, object]] = {}
    inputs: list[str] = []
    for raw_path in report_paths:
        path = Path(raw_path)
        case_name = path.parent.name
        if case_name in payloads:
            raise ValueError(f"duplicate case directory name: {case_name}")
        payloads[case_name] = json.loads(path.read_text(encoding="utf-8"))
        inputs.append(str(path))
    analysis = analyze_online_payloads(payloads, **kwargs)
    analysis["inputs"] = inputs
    return analysis


def _verdict_text(verdict: str) -> str:
    return {
        "telgen_better_throughput": "TELGEN 吞吐显著更优",
        "qcast_better_throughput": "Q-CAST 吞吐显著更优",
        "telgen_better_latency_at_equal_throughput": "吞吐完全相同，TELGEN 延迟显著更低",
        "qcast_better_latency_at_equal_throughput": "吞吐完全相同，Q-CAST 延迟显著更低",
        "inconclusive_throughput": "吞吐差异尚不确定，不能降级到延迟判优",
        "statistically_inconclusive": "统计上无法判定业务质量优劣",
        "invalid_hard_gate_failure": "存在硬约束违例，实验无效",
        "telgen_faster": "TELGEN 决策显著更快",
        "qcast_faster": "Q-CAST 决策显著更快",
        "runtime_inconclusive": "决策时间差异不显著",
    }[verdict]


def _markdown(analysis: Mapping[str, object]) -> str:
    overall = analysis["overall"]
    lines = [
        "# TELGEN 与 Q-CAST 在线配对基准",
        "",
        f"- 场景数：{analysis['case_count']}",
        f"- 配对 episode 数：{analysis['paired_trial_count']}",
        f"- 业务质量结论：**{_verdict_text(overall['quality_verdict'])}**",
        f"- 规划耗时结论：**{_verdict_text(overall['runtime_verdict'])}**",
        f"- 可执行性硬门槛：{'通过' if overall['valid'] else '失败'}",
        "",
        "判定顺序是先比较完成请求数；只有每个配对 episode 的完成数都相同时，才使用删失完成延迟判优。规划耗时单独报告，不与业务质量混合成一个分数。",
        "",
        "## 分场景结果",
        "",
        "| 场景 | 样本数 | TELGEN 完成数 | Q-CAST 完成数 | 完成数优势及 95% CI | TELGEN 延迟 | Q-CAST 延迟 | 业务质量结论 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for case_name, case in analysis["cases"].items():
        completed = case["metrics"]["completed_requests"]
        latency = case["metrics"]["mean_censored_latency_ps"]
        lines.append(
            f"| {case_name} | {case['trial_count']} | "
            f"{completed['telgen_mean']:.3f} | {completed['qcast_mean']:.3f} | "
            f"{completed['telgen_advantage_mean']:.3f} "
            f"[{completed['ci95_low']:.3f}, {completed['ci95_high']:.3f}] | "
            f"{latency['telgen_mean']:.3f} | {latency['qcast_mean']:.3f} | "
            f"{_verdict_text(case['quality_verdict'])} |"
        )
    lines.extend([
        "",
        "## 总体统计",
        "",
        "| 指标 | TELGEN 均值 | Q-CAST 均值 | TELGEN 优势 | 95% CI | 配对随机化 p 值 | TELGEN/平/Q-CAST |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for metric in _METRICS:
        item = overall["metrics"][metric]
        lines.append(
            f"| {item['label']} | {item['telgen_mean']:.6f} | "
            f"{item['qcast_mean']:.6f} | {item['telgen_advantage_mean']:.6f} | "
            f"[{item['ci95_low']:.6f}, {item['ci95_high']:.6f}] | "
            f"{item['paired_randomization_p']:.6f} | "
            f"{item['telgen_wins']}/{item['ties']}/{item['qcast_wins']} |"
        )
    lines.extend([
        "",
        "所有“优势”均统一为正值表示 TELGEN 更好：完成数使用 TELGEN−Q-CAST，延迟和决策时间使用 Q-CAST−TELGEN。",
        "",
        "## 可执行性门槛与名义超时",
        "",
        "`slot_completion_overrun` 表示 SeQUeNCe 物理操作跨过粗粒度名义时隙；在线调度器会继续保留资源 envelope，因此该项单独报告但不视为资源不可行。其他调度违例、物理后端拒绝和完成后验证失败仍是硬失败。",
        "",
    ])
    for name, value in overall["hard_gates"].items():
        lines.append(f"- `{name}`：{value:.0f}")
    lines.append("")
    return "\n".join(lines)


def save_analysis(
    analysis: Mapping[str, object],
    output_directory: str | Path,
) -> dict[str, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = ""
    index = 1
    while (output / f"online_benchmark_{timestamp}{suffix}.json").exists():
        index += 1
        suffix = f"_{index}"
    stem = f"online_benchmark_{timestamp}{suffix}"
    paths = {
        "json": output / f"{stem}.json",
        "csv": output / f"{stem}.csv",
        "markdown": output / f"{stem}.md",
        "latest_json": output / "online_benchmark.json",
        "latest_csv": output / "online_benchmark.csv",
        "latest_markdown": output / "online_benchmark.md",
    }
    paths["json"].write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with paths["csv"].open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case",
                "metric",
                "telgen_mean",
                "qcast_mean",
                "telgen_advantage_mean",
                "ci95_low",
                "ci95_high",
                "paired_randomization_p",
                "telgen_wins",
                "ties",
                "qcast_wins",
            ),
        )
        writer.writeheader()
        for case_name, case in (*analysis["cases"].items(), ("overall", analysis["overall"])):
            for metric, item in case["metrics"].items():
                writer.writerow({"case": case_name, "metric": metric, **{
                    key: item[key] for key in writer.fieldnames if key not in {"case", "metric"}
                }})
    paths["markdown"].write_text(_markdown(analysis), encoding="utf-8")
    shutil.copyfile(paths["json"], paths["latest_json"])
    shutil.copyfile(paths["csv"], paths["latest_csv"])
    shutil.copyfile(paths["markdown"], paths["latest_markdown"])
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze paired TELGEN/Q-CAST online comparison reports."
    )
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", default="results/telgen_qcast_benchmark")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--randomization-samples", type=int, default=20_000)
    parser.add_argument("--random-seed", type=int, default=20260811)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analysis = analyze_online_reports(
        args.reports,
        bootstrap_samples=args.bootstrap_samples,
        randomization_samples=args.randomization_samples,
        random_seed=args.random_seed,
    )
    paths = save_analysis(analysis, args.output)
    print(_verdict_text(analysis["overall"]["quality_verdict"]))
    print(_verdict_text(analysis["overall"]["runtime_verdict"]))
    print(f"json: {paths['json']}")
    print(f"csv: {paths['csv']}")
    print(f"markdown: {paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
