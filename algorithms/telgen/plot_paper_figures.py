"""Generate the ten reusable TELGEN experimental figures.

The command consumes raw paired-trial JSON reports rather than hand-entered
summary numbers.  Re-running the experiments and invoking this module is
therefore sufficient to refresh every plot, its long-form CSV data, and the
figure manifest.

Example::

    python -m algorithms.telgen.plot_paper_figures \
        --results-root results/formal_experiments \
        --output results/paper_figures_preview \
        --formats pdf svg png
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

from matplotlib import image as mpimg
from matplotlib import pyplot as plt
import numpy as np

from .comparison_methods import FORMAL_METHOD_ORDER, SCALABLE_METHOD_ORDER
from .plot_utils import (
    ADAPTIVE_COLOR,
    DOUBLE_COLUMN,
    DOUBLE_COLUMN_TALL,
    FIXED_TREE_COLORS,
    METHOD_STYLES,
    bootstrap_mean_ci,
    configure_paper_style,
    ecdf,
    method_style,
    panel_label,
    save_figure,
    style_axis,
)


METHOD_ORDER = FORMAL_METHOD_ORDER
ROUTING_METHOD_ORDER = SCALABLE_METHOD_ORDER


@dataclass(frozen=True)
class FigureContext:
    results_root: Path
    bootstrap_samples: int
    random_seed: int


@dataclass
class BuiltFigure:
    figure: plt.Figure
    rows: list[dict[str, object]]
    sources: list[Path]


Builder = Callable[[FigureContext], BuiltFigure]


@dataclass(frozen=True)
class FigureSpec:
    number: int
    stem: str
    caption: str
    builder: Builder


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"required experiment report is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"report root must be an object: {path}")
    return payload


def _online_report(context: FigureContext, case: str) -> tuple[Path, dict[str, object]]:
    path = context.results_root / case / "online_gnn_comparison.json"
    payload = _load_json(path)
    trials = payload.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError(f"online report has no trials: {path}")
    return path, payload


def _trials(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw_trials = payload.get("trials")
    if not isinstance(raw_trials, list) or not raw_trials:
        raise ValueError("report has no trials")
    return raw_trials


def _available_methods(payload: Mapping[str, object]) -> tuple[str, ...]:
    first = _trials(payload)[0]
    methods = first.get("methods")
    if not isinstance(methods, Mapping) or not methods:
        raise ValueError("trial methods are missing")
    return tuple(str(method) for method in methods)


def _metric_values(
    payload: Mapping[str, object],
    method: str,
    metric: str,
) -> np.ndarray:
    values = []
    for trial in _trials(payload):
        methods = trial.get("methods")
        if not isinstance(methods, Mapping) or method not in methods:
            raise ValueError(f"method {method!r} is missing from a trial")
        method_payload = methods[method]
        metrics = method_payload.get("metrics")
        if not isinstance(metrics, Mapping) or metric not in metrics:
            raise ValueError(f"metric {metric!r} is missing for {method!r}")
        values.append(float(metrics[metric]))
    return np.asarray(values, dtype=float)


def _slot_duration_ps(payload: Mapping[str, object]) -> float:
    scenario = payload.get("scenario")
    if isinstance(scenario, Mapping):
        physical = scenario.get("physical")
        if isinstance(physical, Mapping) and "slot_duration_ps" in physical:
            return float(physical["slot_duration_ps"])
    episode = _trials(payload)[0].get("episode")
    if isinstance(episode, Mapping):
        physical = episode.get("physical")
        if isinstance(physical, Mapping) and "slot_duration_ps" in physical:
            return float(physical["slot_duration_ps"])
    raise ValueError("slot_duration_ps is missing from the report")


def _request_count(payload: Mapping[str, object]) -> int:
    scenario = payload.get("scenario")
    if isinstance(scenario, Mapping) and "request_count" in scenario:
        return int(scenario["request_count"])
    episode = _trials(payload)[0].get("episode")
    if isinstance(episode, Mapping):
        requests = episode.get("requests")
        if isinstance(requests, list):
            return len(requests)
    raise ValueError("request count is missing from the report")


def _arrival_batch_size(payload: Mapping[str, object]) -> int:
    scenario = payload.get("scenario")
    if isinstance(scenario, Mapping) and "arrival_batch_size" in scenario:
        return int(scenario["arrival_batch_size"])
    raise ValueError("arrival_batch_size is missing from the report")


def _seed(trial: Mapping[str, object]) -> int:
    if "seed" in trial:
        return int(trial["seed"])
    if "planning_seed" in trial:
        return int(trial["planning_seed"])
    raise ValueError("trial seed is missing")


def _mean_ci(
    context: FigureContext,
    values: Sequence[float] | np.ndarray,
    *,
    offset: int,
) -> tuple[float, float, float]:
    return bootstrap_mean_ci(
        values,
        samples=context.bootstrap_samples,
        seed=context.random_seed + offset,
    )


def _simple_bar(
    ax: plt.Axes,
    context: FigureContext,
    payload: Mapping[str, object],
    *,
    methods: Sequence[str],
    metric: str,
    ylabel: str,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    zero_floor: bool = True,
    log_scale: bool = False,
    raw_points: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    means: list[float] = []
    errors_low: list[float] = []
    errors_high: list[float] = []
    colors: list[str] = []
    labels: list[str] = []
    value_sets: list[np.ndarray] = []
    trials = _trials(payload)

    for method_index, method in enumerate(methods):
        values = _metric_values(payload, method, metric)
        if transform is not None:
            values = transform(values)
        value_sets.append(values)
        mean, low, high = _mean_ci(
            context,
            values,
            offset=100 * method_index + len(metric),
        )
        means.append(mean)
        errors_low.append(mean - low)
        errors_high.append(high - mean)
        style = method_style(method)
        colors.append(style.color)
        labels.append(style.label)
        for trial, value in zip(trials, values, strict=True):
            rows.append(
                {
                    "seed": _seed(trial),
                    "method": style.label,
                    "metric": metric,
                    "value": float(value),
                }
            )

    x = np.arange(len(methods), dtype=float)
    bars = ax.bar(
        x,
        means,
        width=0.68,
        color=colors,
        edgecolor="#333333",
        linewidth=0.55,
        yerr=np.asarray([errors_low, errors_high]),
        capsize=2.5,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
        zorder=2,
    )
    for patch, method in zip(bars.patches, methods, strict=True):
        patch.set_hatch(method_style(method).hatch)
    if raw_points:
        for index, values in enumerate(value_sets):
            jitter = np.linspace(-0.16, 0.16, values.size)
            ax.scatter(
                np.full(values.size, x[index]) + jitter,
                values,
                s=6,
                color="#111111",
                alpha=0.22,
                linewidths=0,
                zorder=3,
            )
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    style_axis(ax, zero_floor=zero_floor and not log_scale)
    return rows


def _grouped_bars(
    ax: plt.Axes,
    context: FigureContext,
    reports: Sequence[tuple[str, Mapping[str, object]]],
    *,
    methods: Sequence[str],
    metric: str,
    ylabel: str,
    transform: Callable[[Mapping[str, object], np.ndarray], np.ndarray] | None = None,
    zero_floor: bool = True,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    group_x = np.arange(len(reports), dtype=float)
    width = 0.78 / len(methods)
    for method_index, method in enumerate(methods):
        style = method_style(method)
        means: list[float] = []
        low_errors: list[float] = []
        high_errors: list[float] = []
        for report_index, (case_label, payload) in enumerate(reports):
            values = _metric_values(payload, method, metric)
            if transform is not None:
                values = transform(payload, values)
            mean, low, high = _mean_ci(
                context,
                values,
                offset=1000 * report_index + 100 * method_index + len(metric),
            )
            means.append(mean)
            low_errors.append(mean - low)
            high_errors.append(high - mean)
            for trial, value in zip(_trials(payload), values, strict=True):
                rows.append(
                    {
                        "case": case_label,
                        "seed": _seed(trial),
                        "method": style.label,
                        "metric": metric,
                        "value": float(value),
                    }
                )
        positions = group_x - 0.39 + width / 2 + method_index * width
        bars = ax.bar(
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
    ax.set_xticks(group_x, [label for label, _ in reports])
    ax.set_ylabel(ylabel)
    style_axis(ax, zero_floor=zero_floor)
    return rows


def _line_metric(
    ax: plt.Axes,
    context: FigureContext,
    reports: Sequence[tuple[int, Mapping[str, object]]],
    *,
    methods: Sequence[str],
    metric: str,
    ylabel: str,
    transform: Callable[[Mapping[str, object], np.ndarray], np.ndarray] | None = None,
    zero_floor: bool = True,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    x = np.asarray([point for point, _ in reports], dtype=float)
    for method_index, method in enumerate(methods):
        style = method_style(method)
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for report_index, (load, payload) in enumerate(reports):
            values = _metric_values(payload, method, metric)
            if transform is not None:
                values = transform(payload, values)
            mean, low, high = _mean_ci(
                context,
                values,
                offset=2000 * report_index + 100 * method_index + len(metric),
            )
            means.append(mean)
            lows.append(low)
            highs.append(high)
            for trial, value in zip(_trials(payload), values, strict=True):
                rows.append(
                    {
                        "arrival_batch_size": load,
                        "seed": _seed(trial),
                        "method": style.label,
                        "metric": metric,
                        "value": float(value),
                    }
                )
        means_array = np.asarray(means)
        ax.plot(
            x,
            means_array,
            color=style.color,
            marker=style.marker,
            linestyle=style.linestyle,
            linewidth=style.linewidth,
            label=style.label,
            zorder=3,
        )
        ax.fill_between(
            x,
            np.asarray(lows),
            np.asarray(highs),
            color=style.color,
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )
    ax.set_xlabel("Requests per arrival batch")
    ax.set_xticks(x)
    ax.set_ylabel(ylabel)
    style_axis(ax, grid_axis="both", zero_floor=zero_floor)
    return rows


def _fig01_construction_value(context: FigureContext) -> BuiltFigure:
    path = (
        context.results_root
        / "B1_construction_physical_100"
        / "construction_physical_validation.json"
    )
    payload = _load_json(path)
    trials = _trials(payload)
    planned_adaptive = np.asarray(
        [float(trial["construction_aware"]["planned_selected_requests"]) for trial in trials]
    )
    planned_fixed = np.asarray(
        [float(trial["best_fixed"]["planned_selected_requests"]) for trial in trials]
    )
    completed_adaptive = np.asarray(
        [float(trial["construction_aware"]["completed_requests"]) for trial in trials]
    )
    completed_fixed = np.asarray(
        [float(trial["best_fixed"]["completed_requests"]) for trial in trials]
    )

    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COLUMN)
    pairs = (
        (
            planned_adaptive,
            planned_fixed,
            "Planned requests",
            "(a) Planned Requests",
        ),
        (
            completed_adaptive,
            completed_fixed,
            "Physically completed requests",
            "(b) Physical Completion",
        ),
    )
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(context.random_seed)
    for axis_index, (adaptive, fixed, ylabel, label) in enumerate(pairs):
        box = axes[axis_index].boxplot(
            [adaptive, fixed],
            positions=[0, 1],
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.2},
            whiskerprops={"color": "#555555", "linewidth": 0.8},
            capprops={"color": "#555555", "linewidth": 0.8},
            boxprops={"edgecolor": "#333333", "linewidth": 0.7},
        )
        box["boxes"][0].set_facecolor(ADAPTIVE_COLOR)
        box["boxes"][0].set_alpha(0.78)
        box["boxes"][1].set_facecolor("#A6A6A6")
        for position, values in enumerate((adaptive, fixed)):
            jitter = rng.uniform(-0.13, 0.13, size=values.size)
            axes[axis_index].scatter(
                np.full(values.size, position) + jitter,
                values,
                s=6,
                color="#111111",
                alpha=0.18,
                linewidths=0,
                zorder=3,
            )
        delta = adaptive - fixed
        mean, low, high = _mean_ci(context, delta, offset=axis_index + 10)
        axes[axis_index].text(
            0.98,
            0.96,
            f"paired Δ = {mean:+.2f}\n95% CI [{low:+.2f}, {high:+.2f}]",
            transform=axes[axis_index].transAxes,
            ha="right",
            va="top",
            fontsize=7.1,
        )
        axes[axis_index].set_xticks([0, 1], ["Construction-aware", "Best fixed"])
        axes[axis_index].set_ylabel(ylabel)
        style_axis(axes[axis_index], zero_floor=True)
        panel_label(axes[axis_index], label)
        metric_name = "planned_requests" if axis_index == 0 else "completed_requests"
        for trial, adaptive_value, fixed_value in zip(
            trials, adaptive, fixed, strict=True
        ):
            rows.extend(
                [
                    {
                        "seed": _seed(trial),
                        "strategy": "Construction-aware",
                        "metric": metric_name,
                        "value": float(adaptive_value),
                    },
                    {
                        "seed": _seed(trial),
                        "strategy": "Best fixed",
                        "metric": metric_name,
                        "value": float(fixed_value),
                    },
                ]
            )
    fig.subplots_adjust(wspace=0.30)
    return BuiltFigure(fig, rows, [path])


def _fig02_small_scale_throughput(context: FigureContext) -> BuiltFigure:
    path, payload = _online_report(context, "B2_small_oracle")
    fig, ax = plt.subplots(figsize=(4.35, 2.55))
    methods = tuple(method for method in METHOD_ORDER if method in _available_methods(payload))
    rows = _simple_bar(
        ax,
        context,
        payload,
        methods=methods,
        metric="completed_requests",
        ylabel="Completed requests per episode",
    )
    ax.axhline(_request_count(payload), color="#555555", linewidth=0.7, linestyle=":")
    ax.set_ylim(0, _request_count(payload) * 1.05)
    return BuiltFigure(fig, rows, [path])


def _fig03_small_scale_latency(context: FigureContext) -> BuiltFigure:
    path, payload = _online_report(context, "B2_small_oracle")
    methods = tuple(method for method in METHOD_ORDER if method in _available_methods(payload))
    slot_duration = _slot_duration_ps(payload)
    transform = lambda values: values / slot_duration
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COLUMN)
    rows = _simple_bar(
        axes[0],
        context,
        payload,
        methods=methods,
        metric="mean_censored_latency_ps",
        ylabel="Mean censored latency (slots)",
        transform=transform,
    )
    rows.extend(
        _simple_bar(
            axes[1],
            context,
            payload,
            methods=methods,
            metric="p95_completion_latency_ps",
            ylabel="P95 completion latency (slots)",
            transform=transform,
        )
    )
    panel_label(axes[0], "(a) Mean Censored Latency")
    panel_label(axes[1], "(b) P95 Completion Latency")
    fig.subplots_adjust(wspace=0.31)
    return BuiltFigure(fig, rows, [path])


def _fig04_gnn_milp_quality(context: FigureContext) -> BuiltFigure:
    path, payload = _online_report(context, "B2_small_oracle")
    gnn = _metric_values(payload, "gnn", "completed_requests")
    milp = _metric_values(payload, "milp", "completed_requests")
    retention = np.divide(gnn, milp, out=np.ones_like(gnn), where=milp > 0)
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COLUMN)

    axes[0].scatter(
        milp,
        gnn,
        color=METHOD_STYLES["gnn"].color,
        edgecolor="#111111",
        linewidth=0.45,
        s=24,
        alpha=0.82,
    )
    upper = max(float(np.max(milp)), float(np.max(gnn)), 1.0) + 0.5
    axes[0].plot([0, upper], [0, upper], color="#555555", linestyle=":", linewidth=0.9)
    axes[0].set_xlim(0, upper)
    axes[0].set_ylim(0, upper)
    axes[0].set_xlabel("MILP completed requests")
    axes[0].set_ylabel("TELGEN completed requests")
    axes[0].set_aspect("equal", adjustable="box")
    style_axis(axes[0], grid_axis="both")
    panel_label(axes[0], "(a) Per-instance Completion")

    x, y = ecdf(retention)
    axes[1].step(x, y, where="post", color=ADAPTIVE_COLOR, linewidth=1.35)
    axes[1].axvline(0.9, color="#555555", linestyle=":", linewidth=0.9)
    mean, low, high = _mean_ci(context, retention, offset=404)
    axes[1].text(
        0.04,
        0.94,
        f"mean retention = {mean:.3f}\n95% CI [{low:.3f}, {high:.3f}]",
        transform=axes[1].transAxes,
        va="top",
        fontsize=7.1,
    )
    axes[1].set_xlim(0.55, 1.05)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("TELGEN / MILP completion ratio")
    axes[1].set_ylabel("Empirical CDF")
    style_axis(axes[1])
    panel_label(axes[1], "(b) MILP Retention")

    rows = [
        {
            "seed": _seed(trial),
            "milp_completed": float(milp_value),
            "gnn_completed": float(gnn_value),
            "completion_retention": float(retention_value),
        }
        for trial, milp_value, gnn_value, retention_value in zip(
            _trials(payload), milp, gnn, retention, strict=True
        )
    ]
    fig.subplots_adjust(wspace=0.31)
    return BuiltFigure(fig, rows, [path])


def _fig05_decision_time(context: FigureContext) -> BuiltFigure:
    path, payload = _online_report(context, "B2_small_oracle")
    methods = tuple(method for method in METHOD_ORDER if method in _available_methods(payload))
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COLUMN)
    rows = _simple_bar(
        axes[0],
        context,
        payload,
        methods=methods,
        metric="mean_decision_seconds",
        ylabel="Mean decision time (s, log scale)",
        zero_floor=False,
        log_scale=True,
    )
    rows.extend(
        _simple_bar(
            axes[1],
            context,
            payload,
            methods=methods,
            metric="p95_decision_seconds",
            ylabel="P95 decision time (s, log scale)",
            zero_floor=False,
            log_scale=True,
        )
    )
    panel_label(axes[0], "(a) Mean Decision Time")
    panel_label(axes[1], "(b) P95 Decision Time")
    fig.subplots_adjust(wspace=0.32)
    return BuiltFigure(fig, rows, [path])


def _b3_reports(
    context: FigureContext,
) -> tuple[list[Path], list[tuple[str, Mapping[str, object]]]]:
    cases = (
        ("Waxman\n(192 nodes)", "B3_waxman192"),
        ("Barabási–Albert\n(128 nodes)", "B3_barabasi128"),
    )
    sources: list[Path] = []
    reports: list[tuple[str, Mapping[str, object]]] = []
    for label, case in cases:
        path, payload = _online_report(context, case)
        sources.append(path)
        reports.append((label, payload))
    return sources, reports


def _fig06_topology_generalization_throughput(context: FigureContext) -> BuiltFigure:
    sources, reports = _b3_reports(context)
    fig, ax = plt.subplots(figsize=(5.35, 2.65))
    rows = _grouped_bars(
        ax,
        context,
        reports,
        methods=ROUTING_METHOD_ORDER,
        metric="completed_requests",
        ylabel="Completion rate",
        transform=lambda payload, values: values / _request_count(payload),
    )
    ax.set_ylim(0, 1.0)
    ax.legend(
        ncol=len(ROUTING_METHOD_ORDER),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
    )
    return BuiltFigure(fig, rows, sources)


def _fig07_topology_generalization_latency(context: FigureContext) -> BuiltFigure:
    sources, reports = _b3_reports(context)
    transform = lambda payload, values: values / _slot_duration_ps(payload)
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COLUMN_TALL)
    rows = _grouped_bars(
        axes[0],
        context,
        reports,
        methods=ROUTING_METHOD_ORDER,
        metric="mean_censored_latency_ps",
        ylabel="Mean censored latency (slots)",
        transform=transform,
    )
    rows.extend(
        _grouped_bars(
            axes[1],
            context,
            reports,
            methods=ROUTING_METHOD_ORDER,
            metric="p95_completion_latency_ps",
            ylabel="P95 completion latency (slots)",
            transform=transform,
        )
    )
    panel_label(axes[0], "(a) Mean Censored Latency")
    panel_label(axes[1], "(b) P95 Completion Latency")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=len(ROUTING_METHOD_ORDER),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.subplots_adjust(top=0.83, wspace=0.31)
    return BuiltFigure(fig, rows, sources)


def _b4_reports(
    context: FigureContext,
) -> tuple[list[Path], list[tuple[int, Mapping[str, object]]]]:
    cases = (
        "B4_load_low_50",
        "B4_load_medium_100",
        "B4_load_high_150",
    )
    sources: list[Path] = []
    reports: list[tuple[int, Mapping[str, object]]] = []
    for case in cases:
        path, payload = _online_report(context, case)
        sources.append(path)
        reports.append((_arrival_batch_size(payload), payload))
    reports.sort(key=lambda item: item[0])
    return sources, reports


def _fig08_load_throughput(context: FigureContext) -> BuiltFigure:
    sources, reports = _b4_reports(context)
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COLUMN_TALL)
    rows = _line_metric(
        axes[0],
        context,
        reports,
        methods=ROUTING_METHOD_ORDER,
        metric="completed_requests",
        ylabel="Completion rate",
        transform=lambda payload, values: values / _request_count(payload),
    )
    axes[0].set_ylim(0, 1.0)
    rows.extend(
        _line_metric(
            axes[1],
            context,
            reports,
            methods=ROUTING_METHOD_ORDER,
            metric="throughput_per_slot",
            ylabel="Throughput (requests / slot)",
        )
    )
    panel_label(axes[0], "(a) Completion Rate")
    panel_label(axes[1], "(b) Network Throughput")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=len(ROUTING_METHOD_ORDER),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.subplots_adjust(top=0.83, wspace=0.31)
    return BuiltFigure(fig, rows, sources)


def _fig09_load_latency(context: FigureContext) -> BuiltFigure:
    sources, reports = _b4_reports(context)
    transform = lambda payload, values: values / _slot_duration_ps(payload)
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COLUMN_TALL)
    rows = _line_metric(
        axes[0],
        context,
        reports,
        methods=ROUTING_METHOD_ORDER,
        metric="mean_censored_latency_ps",
        ylabel="Mean censored latency (slots)",
        transform=transform,
    )
    rows.extend(
        _line_metric(
            axes[1],
            context,
            reports,
            methods=ROUTING_METHOD_ORDER,
            metric="p95_completion_latency_ps",
            ylabel="P95 completion latency (slots)",
            transform=transform,
        )
    )
    panel_label(axes[0], "(a) Mean Censored Latency")
    panel_label(axes[1], "(b) P95 Completion Latency")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=len(ROUTING_METHOD_ORDER),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.subplots_adjust(top=0.83, wspace=0.31)
    return BuiltFigure(fig, rows, sources)


def _variant_values(
    context: FigureContext,
    case: str,
    metric: str,
) -> tuple[Path, dict[str, object], np.ndarray]:
    path, payload = _online_report(context, case)
    return path, payload, _metric_values(payload, "gnn", metric)


def _fig10_construction_ablation(context: FigureContext) -> BuiltFigure:
    adaptive_case = "B5_adaptive_5"
    fixed_cases = [f"B5_fixed_tree_{index}" for index in range(5)]
    candidate_cases = {
        1: "B5_candidates_1",
        3: "B5_candidates_3",
        5: adaptive_case,
    }
    sources: list[Path] = []
    rows: list[dict[str, object]] = []

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.75))
    variant_cases = [("Adaptive", adaptive_case)] + [
        (f"Tree {index}", case) for index, case in enumerate(fixed_cases)
    ]
    variant_means: list[float] = []
    variant_low: list[float] = []
    variant_high: list[float] = []
    for variant_index, (label, case) in enumerate(variant_cases):
        path, payload, values = _variant_values(context, case, "completed_requests")
        sources.append(path)
        mean, low, high = _mean_ci(context, values, offset=5100 + variant_index)
        variant_means.append(mean)
        variant_low.append(low)
        variant_high.append(high)
        for trial, value in zip(_trials(payload), values, strict=True):
            rows.append(
                {
                    "seed": _seed(trial),
                    "variant": label,
                    "candidate_count": 5,
                    "metric": "completed_requests",
                    "value": float(value),
                }
            )
    x = np.arange(len(variant_cases))
    colors = [ADAPTIVE_COLOR, *FIXED_TREE_COLORS]
    axes[0].errorbar(
        x,
        variant_means,
        yerr=np.asarray(
            [
                np.asarray(variant_means) - np.asarray(variant_low),
                np.asarray(variant_high) - np.asarray(variant_means),
            ]
        ),
        fmt="none",
        ecolor="#333333",
        elinewidth=0.9,
        capsize=2.5,
        zorder=2,
    )
    axes[0].scatter(x, variant_means, c=colors, edgecolor="#222222", s=34, zorder=3)
    axes[0].set_xticks(x, [label for label, _ in variant_cases], rotation=32, ha="right")
    axes[0].set_ylabel("Completed requests")
    axes[0].set_ylim(40, 55)
    style_axis(axes[0])
    panel_label(axes[0], "(a) Fixed-tree Comparison", y=-0.43)

    candidate_counts = sorted(candidate_cases)
    completion_means: list[float] = []
    completion_lows: list[float] = []
    completion_highs: list[float] = []
    decision_means: list[float] = []
    decision_lows: list[float] = []
    decision_highs: list[float] = []
    for candidate_index, count in enumerate(candidate_counts):
        case = candidate_cases[count]
        path, payload, completed = _variant_values(context, case, "completed_requests")
        if path not in sources:
            sources.append(path)
        decision = _metric_values(payload, "gnn", "mean_decision_seconds")
        completed_stats = _mean_ci(context, completed, offset=5200 + candidate_index)
        decision_stats = _mean_ci(context, decision, offset=5300 + candidate_index)
        completion_means.append(completed_stats[0])
        completion_lows.append(completed_stats[1])
        completion_highs.append(completed_stats[2])
        decision_means.append(decision_stats[0])
        decision_lows.append(decision_stats[1])
        decision_highs.append(decision_stats[2])
        for trial, completed_value, decision_value in zip(
            _trials(payload), completed, decision, strict=True
        ):
            rows.extend(
                [
                    {
                        "seed": _seed(trial),
                        "variant": "Adaptive",
                        "candidate_count": count,
                        "metric": "completed_requests",
                        "value": float(completed_value),
                    },
                    {
                        "seed": _seed(trial),
                        "variant": "Adaptive",
                        "candidate_count": count,
                        "metric": "mean_decision_seconds",
                        "value": float(decision_value),
                    },
                ]
            )

    counts = np.asarray(candidate_counts, dtype=float)
    for axis, means, lows, highs, ylabel, label in (
        (
            axes[1],
            completion_means,
            completion_lows,
            completion_highs,
            "Completed requests",
            "(b) Completed Requests",
        ),
        (
            axes[2],
            decision_means,
            decision_lows,
            decision_highs,
            "Mean decision time (s)",
            "(c) Decision Time",
        ),
    ):
        means_array = np.asarray(means)
        axis.plot(
            counts,
            means_array,
            color=ADAPTIVE_COLOR,
            marker="o",
            linewidth=1.35,
        )
        axis.fill_between(
            counts,
            np.asarray(lows),
            np.asarray(highs),
            color=ADAPTIVE_COLOR,
            alpha=0.12,
            linewidth=0,
        )
        axis.set_xticks(counts)
        axis.set_xlabel("Construction candidates")
        axis.set_ylabel(ylabel)
        style_axis(axis, zero_floor=(axis is axes[2]))
        panel_label(axis, label)
    axes[1].set_ylim(40, 55)
    fig.subplots_adjust(wspace=0.42)
    return BuiltFigure(fig, rows, sources)


FIGURE_SPECS: tuple[FigureSpec, ...] = (
    FigureSpec(
        1,
        "fig01_construction_value",
        (
            "Construction-aware selection improves both nominal planning "
            "and physical completion over the strongest per-instance "
            "fixed-tree oracle."
        ),
        _fig01_construction_value,
    ),
    FigureSpec(
        2,
        "fig02_small_scale_throughput",
        (
            "On exactly solvable online instances, TELGEN closes most of "
            "the MILP completion gap and exceeds the routing baselines."
        ),
        _fig02_small_scale_throughput,
    ),
    FigureSpec(
        3,
        "fig03_small_scale_latency",
        (
            "TELGEN lowers mean censored and tail completion latency "
            "relative to the routing baselines on small online instances."
        ),
        _fig03_small_scale_latency,
    ),
    FigureSpec(
        4,
        "fig04_gnn_milp_quality",
        (
            "TELGEN tracks the per-instance MILP solution while retaining "
            "a high fraction of the exact solver's completed requests."
        ),
        _fig04_gnn_milp_quality,
    ),
    FigureSpec(
        5,
        "fig05_decision_time",
        (
            "TELGEN is substantially faster than exact MILP, although "
            "lightweight routing heuristics remain faster."
        ),
        _fig05_decision_time,
    ),
    FigureSpec(
        6,
        "fig06_topology_generalization_throughput",
        (
            "The frozen TELGEN model preserves its completion advantage on "
            "unseen Waxman and Barabási–Albert topologies."
        ),
        _fig06_topology_generalization_throughput,
    ),
    FigureSpec(
        7,
        "fig07_topology_generalization_latency",
        (
            "The frozen TELGEN model consistently reduces mean censored "
            "latency and remains comparable in P95 completion latency "
            "across unseen topologies."
        ),
        _fig07_topology_generalization_latency,
    ),
    FigureSpec(
        8,
        "fig08_load_throughput",
        (
            "Construction-aware routing maintains higher completion rate "
            "and throughput as offered load increases."
        ),
        _fig08_load_throughput,
    ),
    FigureSpec(
        9,
        "fig09_load_latency",
        (
            "TELGEN reduces mean censored latency as contention increases, "
            "while P95 completion latency remains comparable rather than "
            "uniformly dominant."
        ),
        _fig09_load_latency,
    ),
    FigureSpec(
        10,
        "fig10_construction_ablation",
        (
            "Adaptive construction outperforms every fixed swap tree, while "
            "gains increase with the number of available construction "
            "candidates."
        ),
        _fig10_construction_ablation,
    ),
)


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write an empty figure dataset: {path}")
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(
    output: Path,
    entries: Sequence[Mapping[str, object]],
    *,
    results_root: Path,
) -> None:
    manifest = {
        "schema_version": 1,
        "results_root": str(results_root.resolve()),
        "figure_count": len(entries),
        "figures": list(entries),
    }
    with (output / "figure_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    lines = [
        "# TELGEN experimental figure preview",
        "",
        (
            "All plots are generated from raw paired-trial JSON reports. "
            "PDF and SVG are vector outputs; PNG files are inspection "
            "previews."
        ),
        "",
    ]
    for entry in entries:
        lines.extend(
            [
                f"## Figure {entry['number']}: {entry['stem']}",
                "",
                str(entry["caption"]),
                "",
                f"- Data: `{entry['data_csv']}`",
                f"- Sources: {', '.join(f'`{source}`' for source in entry['sources'])}",
                "",
            ]
        )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _contact_sheet(output: Path, specs: Sequence[FigureSpec]) -> Path | None:
    png_paths = [output / f"{spec.stem}.png" for spec in specs]
    if not all(path.is_file() for path in png_paths):
        return None
    columns = 2
    rows = math.ceil(len(png_paths) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(12.0, 3.8 * rows))
    axes_array = np.asarray(axes, dtype=object).reshape(-1)
    for axis, spec, path in zip(axes_array, specs, png_paths, strict=False):
        axis.imshow(mpimg.imread(path))
        axis.set_title(f"Figure {spec.number}: {spec.stem}", fontsize=10)
        axis.axis("off")
    for axis in axes_array[len(png_paths) :]:
        axis.axis("off")
    fig.tight_layout()
    path = output / "preview_contact_sheet.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_figures(
    *,
    results_root: Path,
    output: Path,
    formats: Sequence[str],
    selected_numbers: set[int] | None = None,
    bootstrap_samples: int = 10_000,
    random_seed: int = 20260820,
    dpi: int = 300,
) -> list[dict[str, object]]:
    """Generate selected paper figures and return manifest entries."""

    configure_paper_style()
    context = FigureContext(
        results_root=results_root,
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed,
    )
    specs = [
        spec
        for spec in FIGURE_SPECS
        if selected_numbers is None or spec.number in selected_numbers
    ]
    if not specs:
        raise ValueError("no figures selected")
    output.mkdir(parents=True, exist_ok=True)
    data_directory = output / "data"
    entries: list[dict[str, object]] = []
    for spec in specs:
        built = spec.builder(context)
        figure_paths = save_figure(
            built.figure,
            output_directory=output,
            stem=spec.stem,
            formats=formats,
            dpi=dpi,
        )
        data_path = data_directory / f"{spec.stem}.csv"
        _write_rows(data_path, built.rows)
        entries.append(
            {
                "number": spec.number,
                "stem": spec.stem,
                "caption": spec.caption,
                "files": [str(path.relative_to(output)) for path in figure_paths],
                "data_csv": str(data_path.relative_to(output)),
                "sources": [
                    str(source.relative_to(results_root.parent))
                    if source.is_relative_to(results_root.parent)
                    else str(source)
                    for source in built.sources
                ],
            }
        )
    _write_manifest(output, entries, results_root=results_root)
    if "png" in {file_format.lower().lstrip(".") for file_format in formats}:
        _contact_sheet(output, specs)
    return entries


def _parse_numbers(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    numbers = {int(part.strip()) for part in raw.split(",") if part.strip()}
    valid = {spec.number for spec in FIGURE_SPECS}
    unknown = numbers - valid
    if unknown:
        raise ValueError(f"unknown figure numbers: {sorted(unknown)}")
    return numbers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ten reusable TELGEN experimental paper figures."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/formal_experiments"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/paper_figures_preview"),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("pdf", "svg", "png"),
        help="output formats: pdf svg png eps",
    )
    parser.add_argument(
        "--figures",
        help="comma-separated figure numbers; omit to generate all ten",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260820)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    entries = generate_figures(
        results_root=args.results_root,
        output=args.output,
        formats=args.formats,
        selected_numbers=_parse_numbers(args.figures),
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
        dpi=args.dpi,
    )
    print(f"generated {len(entries)} figures in {args.output}")


if __name__ == "__main__":
    main()
