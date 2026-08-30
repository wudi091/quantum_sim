"""Analyze and plot the P0 evaluation suite for construction-aware routing.

The suite compares the frozen online GNN against the two routing baselines
(Q-PASS and Greedy) across two sweeps:

* a load sweep (50, 100, 150, 200, 300 requests on a 64-node Waxman network);
* a physical-noise sweep (generation/swap success probabilities).

Every report is a paired-trial JSON produced by compare_online_gnn, so the same
episode is executed independently by each method.  This module computes
balanced means, percentile bootstrap confidence intervals, and paired
randomization p-values, and renders publication-oriented figures plus a
machine-readable summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from matplotlib import pyplot as plt

from .comparison_methods import SCALABLE_METHOD_ORDER
from .plot_utils import (
    DOUBLE_COLUMN,
    DOUBLE_COLUMN_TALL,
    bootstrap_mean_ci,
    configure_paper_style,
    method_style,
    panel_label,
    save_figure,
    style_axis,
)


LOAD_CASES = (
    ("L050", 50),
    ("L100", 100),
    ("L150", 150),
    ("L200", 200),
    ("L300", 300),
)

NOISE_CASES = (
    ("N_low", "Low\n0.9 / 0.9", 0.9, 0.9),
    ("N_mid", "Mid\n0.7 / 0.8", 0.7, 0.8),
    ("N_high", "High\n0.5 / 0.7", 0.5, 0.7),
)

PRIMARY_METRIC = "completed_requests"
SECONDARY_METRIC = "mean_censored_latency_ps"
RUNTIME_METRIC = "mean_decision_seconds"

_METRIC_OFFSETS = {
    PRIMARY_METRIC: 0,
    SECONDARY_METRIC: 3,
    RUNTIME_METRIC: 7,
}


def _load_report(suite_root: Path, case: str) -> dict[str, object]:
    path = suite_root / case / "online_gnn_comparison.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing P0 report: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"report root must be an object: {path}")
    trials = payload.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError(f"report has no trials: {path}")
    return payload


def _metric_values(
    payload: Mapping[str, object],
    method: str,
    metric: str,
) -> np.ndarray:
    trials = payload["trials"]
    values: list[float] = []
    for trial in trials:
        methods = trial.get("methods")
        if not isinstance(methods, Mapping) or method not in methods:
            raise ValueError(f"method {method!r} missing from trial")
        metrics = methods[method].get("metrics")
        if not isinstance(metrics, Mapping) or metric not in metrics:
            raise ValueError(f"metric {metric!r} missing for {method!r}")
        values.append(float(metrics[metric]))
    return np.asarray(values, dtype=float)


def _request_count(payload: Mapping[str, object]) -> int:
    scenario = payload.get("scenario")
    if isinstance(scenario, Mapping) and "request_count" in scenario:
        return int(scenario["request_count"])
    raise ValueError("request_count is missing from the report")


def _slot_duration_ps(payload: Mapping[str, object]) -> float:
    scenario = payload.get("scenario")
    if isinstance(scenario, Mapping):
        physical = scenario.get("physical")
        if isinstance(physical, Mapping) and "slot_duration_ps" in physical:
            return float(physical["slot_duration_ps"])
    raise ValueError("slot_duration_ps is missing from the report")


def _paired_ci_and_p(
    gnn: np.ndarray,
    baseline: np.ndarray,
    *,
    higher_is_better: bool,
    samples: int,
    seed: int,
) -> dict[str, float]:
    raw_delta = gnn - baseline
    delta = raw_delta if higher_is_better else -raw_delta
    mean = float(np.mean(delta))
    _, low, high = bootstrap_mean_ci(
        delta,
        samples=samples,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 1)
    observed = abs(mean)
    extreme = 1
    for _ in range(samples):
        signs = np.where(rng.random(delta.size) < 0.5, -1.0, 1.0)
        if abs(float(np.mean(delta * signs))) >= observed - 1e-12:
            extreme += 1
    p_value = extreme / (samples + 1)
    wins = int(np.sum(delta > 1e-12))
    losses = int(np.sum(delta < -1e-12))
    ties = int(delta.size) - wins - losses
    return {
        "gnn_mean": float(np.mean(gnn)),
        "baseline_mean": float(np.mean(baseline)),
        "advantage_mean": mean,
        "ci95_low": low,
        "ci95_high": high,
        "paired_p": p_value,
        "gnn_wins": wins,
        "ties": ties,
        "baseline_wins": losses,
    }


def _case_stats(
    payload: Mapping[str, object],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    methods = tuple(
        method for method in SCALABLE_METHOD_ORDER
        if method in payload["trials"][0]["methods"]
    )
    result: dict[str, object] = {
        "trial_count": len(payload["trials"]),
        "request_count": _request_count(payload),
        "slot_duration_ps": _slot_duration_ps(payload),
        "methods": list(methods),
        "metrics": {},
        "paired_vs_baselines": {},
    }
    for metric, higher_is_better in (
        (PRIMARY_METRIC, True),
        (SECONDARY_METRIC, False),
        (RUNTIME_METRIC, False),
    ):
        per_method: dict[str, object] = {}
        base_offset = _METRIC_OFFSETS[metric] * 1000
        for method_index, method in enumerate(methods):
            values = _metric_values(payload, method, metric)
            mean, low, high = bootstrap_mean_ci(
                values,
                samples=samples,
                seed=seed + base_offset + method_index,
            )
            per_method[method] = {
                "mean": mean,
                "ci95_low": low,
                "ci95_high": high,
            }
        result["metrics"][metric] = per_method
        if metric != RUNTIME_METRIC and "gnn" in methods:
            gnn = _metric_values(payload, "gnn", metric)
            result["paired_vs_baselines"][metric] = {}
            for baseline_index, baseline in enumerate(("qpass", "greedy")):
                if baseline in methods:
                    result["paired_vs_baselines"][metric][baseline] = _paired_ci_and_p(
                        gnn,
                        _metric_values(payload, baseline, metric),
                        higher_is_better=higher_is_better,
                        samples=samples,
                        seed=seed + base_offset + 10 + baseline_index,
                    )
    return result


def _completion_rate(payload: Mapping[str, object], values: np.ndarray) -> np.ndarray:
    return values / _request_count(payload)


def _to_slots(payload: Mapping[str, object], values: np.ndarray) -> np.ndarray:
    return values / _slot_duration_ps(payload)


def _line_plot(
    axes,
    payloads: Sequence[Mapping[str, object]],
    x_values: Sequence[float],
    *,
    metric: str,
    ylabel: str,
    transform=None,
    samples: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in SCALABLE_METHOD_ORDER:
        style = method_style(method)
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for index, payload in enumerate(payloads):
            values = _metric_values(payload, method, metric)
            if transform is not None:
                values = transform(payload, values)
            mean, low, high = bootstrap_mean_ci(
                values,
                samples=samples,
                seed=seed + 1000 * index + 100 * _METRIC_OFFSETS.get(metric, 0),
            )
            means.append(mean)
            lows.append(low)
            highs.append(high)
            for trial, value in zip(payload["trials"], values, strict=True):
                rows.append(
                    {
                        "x": float(x_values[index]),
                        "seed": int(trial["seed"]),
                        "method": style.label,
                        "metric": metric,
                        "value": float(value),
                    }
                )
        axes.plot(
            np.asarray(x_values, dtype=float),
            np.asarray(means),
            color=style.color,
            marker=style.marker,
            linestyle=style.linestyle,
            linewidth=style.linewidth,
            label=style.label,
            zorder=3,
        )
        axes.fill_between(
            np.asarray(x_values, dtype=float),
            np.asarray(lows),
            np.asarray(highs),
            color=style.color,
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )
    axes.set_ylabel(ylabel)
    style_axis(axes, grid_axis="both")
    return rows


def _grouped_bars(
    axes,
    payloads: Sequence[Mapping[str, object]],
    labels: Sequence[str],
    *,
    metric: str,
    ylabel: str,
    transform=None,
    samples: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    group_x = np.arange(len(payloads), dtype=float)
    width = 0.78 / len(SCALABLE_METHOD_ORDER)
    for method_index, method in enumerate(SCALABLE_METHOD_ORDER):
        style = method_style(method)
        means: list[float] = []
        low_errors: list[float] = []
        high_errors: list[float] = []
        for report_index, payload in enumerate(payloads):
            values = _metric_values(payload, method, metric)
            if transform is not None:
                values = transform(payload, values)
            mean, low, high = bootstrap_mean_ci(
                values,
                samples=samples,
                seed=seed + 2000 * report_index + 100 * method_index,
            )
            means.append(mean)
            low_errors.append(mean - low)
            high_errors.append(high - mean)
            for trial, value in zip(payload["trials"], values, strict=True):
                rows.append(
                    {
                        "case": labels[report_index],
                        "seed": int(trial["seed"]),
                        "method": style.label,
                        "metric": metric,
                        "value": float(value),
                    }
                )
        positions = group_x - 0.39 + width / 2 + method_index * width
        bars = axes.bar(
            positions,
            means,
            width=width,
            color=style.color,
            edgecolor="#333333",
            linewidth=0.45,
            label=style.label,
            yerr=np.asarray([low_errors, high_errors]),
            capsize=2.0,
            error_kw={"elinewidth": 0.7, "capthick": 0.7},
            zorder=2,
        )
        for patch in bars.patches:
            patch.set_hatch(style.hatch)
    axes.set_xticks(group_x, labels)
    axes.set_ylabel(ylabel)
    style_axis(axes, zero_floor=True)
    return rows


def build_analysis(
    suite_root: Path,
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    load_payloads = [_load_report(suite_root, case) for case, _ in LOAD_CASES]
    noise_payloads = [_load_report(suite_root, case) for case, _, _, _ in NOISE_CASES]

    return {
        "schema_version": 1,
        "analysis_contract": {
            "paired_trials": True,
            "primary_metric": PRIMARY_METRIC,
            "secondary_metric": SECONDARY_METRIC,
            "runtime_metric": RUNTIME_METRIC,
            "positive_advantage_means": "gnn_better",
            "confidence_level": 0.95,
            "bootstrap_samples": samples,
            "randomization_samples": samples,
            "random_seed": seed,
        },
        "load_sweep": {
            case: _case_stats(payload, samples=samples, seed=seed + index)
            for index, (case, _) in enumerate(LOAD_CASES)
            for payload in (load_payloads[index],)
        },
        "noise_sweep": {
            case: _case_stats(payload, samples=samples, seed=seed + 100 + index)
            for index, (case, _, _, _) in enumerate(NOISE_CASES)
            for payload in (noise_payloads[index],)
        },
    }


def generate_figures(
    suite_root: Path,
    output: Path,
    *,
    samples: int,
    seed: int,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    configure_paper_style()
    output.mkdir(parents=True, exist_ok=True)
    load_payloads = [_load_report(suite_root, case) for case, _ in LOAD_CASES]
    noise_payloads = [_load_report(suite_root, case) for case, _, _, _ in NOISE_CASES]
    load_x = [float(count) for _, count in LOAD_CASES]

    figure_paths: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COLUMN_TALL)
    _line_plot(
        axes[0],
        load_payloads,
        load_x,
        metric=PRIMARY_METRIC,
        ylabel="Completion rate",
        transform=_completion_rate,
        samples=samples,
        seed=seed,
    )
    axes[0].set_xlabel("Offered requests")
    axes[0].set_xticks(load_x)
    axes[0].set_ylim(0, 1.0)
    _line_plot(
        axes[1],
        load_payloads,
        load_x,
        metric="throughput_per_slot",
        ylabel="Throughput (requests / slot)",
        samples=samples,
        seed=seed,
    )
    axes[1].set_xlabel("Offered requests")
    axes[1].set_xticks(load_x)
    panel_label(axes[0], "(a) Completion Rate")
    panel_label(axes[1], "(b) Network Throughput")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=len(SCALABLE_METHOD_ORDER),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.subplots_adjust(top=0.84, wspace=0.31)
    figure_paths.extend(
        save_figure(
            fig,
            output_directory=output,
            stem="fig_p0_load_throughput",
            formats=formats,
            dpi=dpi,
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COLUMN_TALL)
    _line_plot(
        axes[0],
        load_payloads,
        load_x,
        metric=SECONDARY_METRIC,
        ylabel="Mean censored latency (slots)",
        transform=_to_slots,
        samples=samples,
        seed=seed,
    )
    axes[0].set_xlabel("Offered requests")
    axes[0].set_xticks(load_x)
    _line_plot(
        axes[1],
        load_payloads,
        load_x,
        metric="p95_completion_latency_ps",
        ylabel="P95 completion latency (slots)",
        transform=_to_slots,
        samples=samples,
        seed=seed,
    )
    axes[1].set_xlabel("Offered requests")
    axes[1].set_xticks(load_x)
    panel_label(axes[0], "(a) Mean Censored Latency")
    panel_label(axes[1], "(b) P95 Completion Latency")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=len(SCALABLE_METHOD_ORDER),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.subplots_adjust(top=0.84, wspace=0.31)
    figure_paths.extend(
        save_figure(
            fig,
            output_directory=output,
            stem="fig_p0_load_latency",
            formats=formats,
            dpi=dpi,
        )
    )

    noise_labels = [label for _, label, _, _ in NOISE_CASES]
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COLUMN)
    _grouped_bars(
        axes[0],
        noise_payloads,
        noise_labels,
        metric=PRIMARY_METRIC,
        ylabel="Completed requests",
        samples=samples,
        seed=seed,
    )
    axes[0].set_xlabel("Physical noise (generation / swap success)")
    _grouped_bars(
        axes[1],
        noise_payloads,
        noise_labels,
        metric=SECONDARY_METRIC,
        ylabel="Mean censored latency (slots)",
        transform=_to_slots,
        samples=samples,
        seed=seed,
    )
    axes[1].set_xlabel("Physical noise (generation / swap success)")
    panel_label(axes[0], "(a) Completed Requests")
    panel_label(axes[1], "(b) Mean Censored Latency")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=len(SCALABLE_METHOD_ORDER),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.subplots_adjust(top=0.80, wspace=0.32)
    figure_paths.extend(
        save_figure(
            fig,
            output_directory=output,
            stem="fig_p0_noise_robustness",
            formats=formats,
            dpi=dpi,
        )
    )

    fig, ax = plt.subplots(figsize=(4.35, 2.55))
    _line_plot(
        ax,
        load_payloads,
        load_x,
        metric=RUNTIME_METRIC,
        ylabel="Mean decision time (s)",
        samples=samples,
        seed=seed,
    )
    ax.set_xlabel("Offered requests")
    ax.set_xticks(load_x)
    ax.legend(ncol=1, loc="upper left")
    figure_paths.extend(
        save_figure(
            fig,
            output_directory=output,
            stem="fig_p0_decision_time",
            formats=formats,
            dpi=dpi,
        )
    )
    return figure_paths


def _markdown(analysis: Mapping[str, object]) -> str:
    lines = [
        "# P0 evaluation suite: construction-aware routing vs. baselines",
        "",
        "Paired-trial comparison of the frozen online GNN against Q-PASS and Greedy.",
        "Positive advantage always means the GNN is better; for latency and decision",
        "time the signed difference is negated before testing.",
        "",
        "## Load sweep (64-node Waxman, no physical noise)",
        "",
        "| Case | Trials | GNN completed | Q-PASS completed | Greedy completed | "
        "GNN vs Q-PASS (95% CI, p) | GNN vs Greedy (95% CI, p) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case, _ in LOAD_CASES:
        stats = analysis["load_sweep"][case]
        completed = stats["metrics"][PRIMARY_METRIC]
        gnn = completed["gnn"]["mean"]
        qpass = completed["qpass"]["mean"]
        greedy = completed["greedy"]["mean"]
        vs_q = stats["paired_vs_baselines"][PRIMARY_METRIC]["qpass"]
        vs_g = stats["paired_vs_baselines"][PRIMARY_METRIC]["greedy"]
        lines.append(
            f"| {case} | {stats['trial_count']} | {gnn:.2f} | {qpass:.2f} | "
            f"{greedy:.2f} | {vs_q['advantage_mean']:+.2f} "
            f"[{vs_q['ci95_low']:+.2f}, {vs_q['ci95_high']:+.2f}], "
            f"p={vs_q['paired_p']:.4f} | {vs_g['advantage_mean']:+.2f} "
            f"[{vs_g['ci95_low']:+.2f}, {vs_g['ci95_high']:+.2f}], "
            f"p={vs_g['paired_p']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Noise sweep (150 requests, 64-node Waxman)",
            "",
            "| Case | Generation / swap | Trials | GNN completed | Q-PASS completed | "
            "Greedy completed | GNN vs Q-PASS (95% CI, p) | GNN vs Greedy (95% CI, p) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case, label, gen, sw in NOISE_CASES:
        stats = analysis["noise_sweep"][case]
        completed = stats["metrics"][PRIMARY_METRIC]
        gnn = completed["gnn"]["mean"]
        qpass = completed["qpass"]["mean"]
        greedy = completed["greedy"]["mean"]
        vs_q = stats["paired_vs_baselines"][PRIMARY_METRIC]["qpass"]
        vs_g = stats["paired_vs_baselines"][PRIMARY_METRIC]["greedy"]
        lines.append(
            f"| {case} | {gen} / {sw} | {stats['trial_count']} | {gnn:.2f} | "
            f"{qpass:.2f} | {greedy:.2f} | {vs_q['advantage_mean']:+.2f} "
            f"[{vs_q['ci95_low']:+.2f}, {vs_q['ci95_high']:+.2f}], "
            f"p={vs_q['paired_p']:.4f} | {vs_g['advantage_mean']:+.2f} "
            f"[{vs_g['ci95_low']:+.2f}, {vs_g['ci95_high']:+.2f}], "
            f"p={vs_g['paired_p']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze and plot the P0 evaluation suite."
    )
    parser.add_argument(
        "--suite-root",
        type=Path,
        default=Path("results/p0_suite"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/p0_suite"),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("pdf", "svg", "png"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260829)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    analysis = build_analysis(
        args.suite_root,
        samples=args.bootstrap_samples,
        seed=args.random_seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    analysis_path = args.output / "p0_analysis.json"
    analysis_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_path = args.output / "P0_SUMMARY.md"
    summary_path.write_text(_markdown(analysis), encoding="utf-8")

    figure_paths = generate_figures(
        args.suite_root,
        args.output / "figures",
        samples=args.bootstrap_samples,
        seed=args.random_seed,
        formats=args.formats,
        dpi=args.dpi,
    )

    print(f"analysis: {analysis_path}")
    print(f"summary: {summary_path}")
    for path in figure_paths:
        print(f"figure: {path}")


if __name__ == "__main__":
    raise SystemExit(main())

