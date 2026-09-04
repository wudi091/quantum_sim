"""Plot ARC-Q paper results from an existing raw result file only.

This module deliberately has no dependency on the experiment runner.  It
validates a completed paired design, writes auditable source-data tables, and
then renders one single-column line figure per suite and metric.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from math import isfinite, sqrt
from pathlib import Path
from statistics import fmean, stdev
from typing import Callable, Mapping, Sequence


RESULT_SCHEMA_VERSION = 1
PLOT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MetricSpec:
    key: str
    raw_key: str
    label: str
    direction: str
    transform: Callable[[float, float], float]
    lower_bound: float | None = None
    upper_bound: float | None = None


METRICS = (
    MetricSpec(
        key="mean_censored_latency",
        raw_key="mean_censored_latency_ps",
        label="Mean censored latency (slots)",
        direction="lower",
        transform=lambda value, slot_duration_ps: value / slot_duration_ps,
        lower_bound=0.0,
    ),
    MetricSpec(
        key="completion_rate",
        raw_key="completion_rate",
        label="Completion rate (%)",
        direction="higher",
        transform=lambda value, _slot_duration_ps: 100.0 * value,
        lower_bound=0.0,
        upper_bound=100.0,
    ),
    MetricSpec(
        key="delay_fairness",
        raw_key="completion_delay_gini",
        label="Delay fairness (1 - Gini)",
        direction="higher",
        transform=lambda value, _slot_duration_ps: 1.0 - value,
        lower_bound=0.0,
        upper_bound=1.0,
    ),
    MetricSpec(
        key="planning_time",
        raw_key="mean_planner_seconds",
        label="Planning time per decision (ms)",
        direction="lower",
        transform=lambda value, _slot_duration_ps: 1000.0 * value,
        lower_bound=0.0,
    ),
)


X_LABELS = {
    "requests_per_decision": "Requests per decision",
    "nodes": "Number of nodes",
    "memory_units_per_node": "Memory units per node",
    "elementary_generation_probability": (
        "Elementary-link success probability"
    ),
    "unseen_topology": "Unseen topology",
}


METHOD_STYLES = {
    "ARC-Q": {
        "color": "#D55E00",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 1.8,
        "filled": True,
    },
    "Greedy": {
        "color": "#222222",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 1.1,
        "filled": False,
    },
    "Path-only": {
        "color": "#0072B2",
        "marker": "^",
        "linestyle": "-.",
        "linewidth": 1.1,
        "filled": False,
    },
    "Construction-only": {
        "color": "#009E73",
        "marker": "D",
        "linestyle": ":",
        "linewidth": 1.1,
        "filled": False,
    },
    "Q-LEAP": {
        "color": "#CC79A7",
        "marker": "v",
        "linestyle": (0, (3, 1, 1, 1)),
        "linewidth": 1.1,
        "filled": False,
    },
    "Q-CAST": {
        "color": "#E69F00",
        "marker": "P",
        "linestyle": (0, (5, 2)),
        "linewidth": 1.1,
        "filled": False,
    },
}


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _sequence(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return list(value)


def _finite_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def load_result_payload(path: str | Path) -> dict[str, object]:
    """Load a recorded result artifact without importing experiment code."""

    result_path = Path(path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    root = _mapping(payload, "result payload")
    if root.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported result schema")
    if root.get("method_under_test") != "ARC-Q":
        raise ValueError("result file is not an ARC-Q evaluation")
    _mapping(root.get("protocol"), "result protocol")
    _sequence(root.get("records"), "result records")
    return root


def _protocol_layout(
    payload: Mapping[str, object],
) -> tuple[
    tuple[str, ...],
    dict[str, dict[str, object]],
    tuple[int, ...],
    float,
]:
    protocol = _mapping(payload["protocol"], "result protocol")
    replication = _mapping(protocol.get("replication"), "replication")
    seed_start = int(replication.get("episode_seed_start", 0))
    seed_count = int(replication.get("episodes_per_topology", 0))
    if seed_count < 1:
        raise ValueError("episodes_per_topology must be positive")
    episode_seeds = tuple(range(seed_start, seed_start + seed_count))

    base_scenario = _mapping(protocol.get("base_scenario"), "base_scenario")
    base_physical = _mapping(base_scenario.get("physical"), "physical")
    slot_duration_ps = _finite_float(
        base_physical.get("slot_duration_ps"),
        "physical.slot_duration_ps",
    )
    if slot_duration_ps <= 0.0:
        raise ValueError("slot_duration_ps must be positive")

    baselines = _sequence(protocol.get("baselines"), "baselines")
    methods = ["ARC-Q"]
    for index, raw_baseline in enumerate(baselines):
        baseline = _mapping(raw_baseline, f"baselines[{index}]")
        name = str(baseline.get("name", "")).strip()
        if not name:
            raise ValueError("baseline name must be non-empty")
        methods.append(name)
    if len(methods) != len(set(methods)):
        raise ValueError("method names must be unique")

    suite_layout: dict[str, dict[str, object]] = {}
    for raw_suite in _sequence(protocol.get("suites"), "suites"):
        suite = _mapping(raw_suite, "suite")
        suite_id = str(suite.get("id", "")).strip()
        if not suite_id or suite_id in suite_layout:
            raise ValueError("suite IDs must be non-empty and unique")
        points: list[dict[str, object]] = []
        point_ids: set[str] = set()
        for raw_point in _sequence(suite.get("points"), "suite.points"):
            point = _mapping(raw_point, "point")
            point_id = str(point.get("id", "")).strip()
            if not point_id or point_id in point_ids:
                raise ValueError("point IDs must be non-empty and unique")
            point_ids.add(point_id)
            topology_seeds = tuple(
                int(seed)
                for seed in _sequence(
                    point.get("topology_seeds"),
                    "point.topology_seeds",
                )
            )
            if not topology_seeds or len(topology_seeds) != len(
                set(topology_seeds)
            ):
                raise ValueError("topology seeds must be non-empty and unique")
            scenario = _mapping(point.get("scenario"), "point.scenario")
            physical = _mapping(scenario.get("physical", {}), "point.physical")
            local_slot_duration = _finite_float(
                physical.get("slot_duration_ps", slot_duration_ps),
                "point slot_duration_ps",
            )
            if local_slot_duration <= 0.0:
                raise ValueError("point slot_duration_ps must be positive")
            points.append({
                "id": point_id,
                "value": point.get("value"),
                "topology_seeds": topology_seeds,
                "slot_duration_ps": local_slot_duration,
            })
        suite_layout[suite_id] = {
            "x_label": str(suite.get("x_label", "")).strip(),
            "points": tuple(points),
        }
    if not suite_layout:
        raise ValueError("at least one suite is required")
    return tuple(methods), suite_layout, episode_seeds, slot_duration_ps


def _record_key(record: Mapping[str, object]) -> tuple[str, str, int, int, str]:
    return (
        str(record.get("suite", "")),
        str(record.get("point_id", "")),
        int(record.get("topology_seed", -1)),
        int(record.get("episode_seed", -1)),
        str(record.get("method", "")),
    )


def _mean_ci95(
    values: Sequence[float],
    *,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> tuple[float, float, float]:
    if not values:
        raise ValueError("a confidence interval needs at least one value")
    mean = fmean(values)
    if len(values) == 1:
        low = high = mean
    else:
        from scipy.stats import t

        radius = float(t.ppf(0.975, len(values) - 1)) * (
            stdev(values) / sqrt(len(values))
        )
        low = mean - radius
        high = mean + radius
    if lower_bound is not None:
        low = max(lower_bound, low)
        high = max(lower_bound, high)
    if upper_bound is not None:
        low = min(upper_bound, low)
        high = min(upper_bound, high)
    return float(mean), float(low), float(high)


def summarize_results(
    payload: Mapping[str, object],
    *,
    suite_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Validate the paired design and aggregate publication source data."""

    methods, suite_layout, episode_seeds, _ = _protocol_layout(payload)
    selected = (
        tuple(suite_layout)
        if suite_ids is None
        else tuple(dict.fromkeys(str(item) for item in suite_ids))
    )
    unknown = set(selected) - set(suite_layout)
    if unknown:
        raise ValueError(f"unknown result suite: {sorted(unknown)[0]}")
    if not selected:
        raise ValueError("at least one result suite must be selected")

    records_by_key: dict[tuple[str, str, int, int, str], dict[str, object]] = {}
    maximum_reward_identity_error = 0.0
    for raw_record in _sequence(payload["records"], "result records"):
        record = _mapping(raw_record, "record")
        key = _record_key(record)
        if key in records_by_key:
            raise ValueError(f"duplicate result record: {key}")
        if key[0] not in suite_layout:
            raise ValueError(f"record references unknown suite: {key[0]}")
        point_by_id = {
            str(point["id"]): point
            for point in suite_layout[key[0]]["points"]
        }
        if key[1] not in point_by_id:
            raise ValueError(f"record references unknown point: {key[1]}")
        if key[4] not in methods:
            raise ValueError(f"record references unknown method: {key[4]}")
        metrics = _mapping(record.get("metrics"), "record.metrics")
        for metric in METRICS:
            _finite_float(metrics.get(metric.raw_key), metric.raw_key)
        if _finite_float(
            metrics.get("schedule_violation_count"),
            "schedule_violation_count",
        ) != 0.0:
            raise ValueError(f"schedule violation in result record: {key}")
        if _finite_float(
            metrics.get("physical_backend_rejection_count"),
            "physical_backend_rejection_count",
        ) != 0.0:
            raise ValueError(f"physical backend rejection in result record: {key}")
        if key[4] == "ARC-Q":
            identity_error = abs(_finite_float(
                metrics.get("reward_identity_error"),
                "reward_identity_error",
            ))
            maximum_reward_identity_error = max(
                maximum_reward_identity_error,
                identity_error,
            )
            if identity_error > 1e-8:
                raise ValueError(f"reward identity failure in result record: {key}")
        records_by_key[key] = record

    summary_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    expected_record_count = 0
    expected_keys: set[tuple[str, str, int, int, str]] = set()
    for suite_id in selected:
        suite = suite_layout[suite_id]
        for point_index, point in enumerate(suite["points"]):
            point_id = str(point["id"])
            topology_seeds = tuple(point["topology_seeds"])
            pair_keys = tuple(
                (int(topology_seed), int(episode_seed))
                for topology_seed in topology_seeds
                for episode_seed in episode_seeds
            )
            expected_record_count += len(pair_keys) * len(methods)
            transformed: dict[str, dict[str, list[float]]] = {
                metric.key: {method: [] for method in methods}
                for metric in METRICS
            }
            for topology_seed, episode_seed in pair_keys:
                for method in methods:
                    key = (
                        suite_id,
                        point_id,
                        topology_seed,
                        episode_seed,
                        method,
                    )
                    expected_keys.add(key)
                    if key not in records_by_key:
                        raise ValueError(f"missing paired result record: {key}")
                    record = records_by_key[key]
                    if record.get("point_value") != point["value"]:
                        raise ValueError(f"point value mismatch in record: {key}")
                    metrics = _mapping(record["metrics"], "record.metrics")
                    for metric in METRICS:
                        raw_value = _finite_float(
                            metrics[metric.raw_key],
                            metric.raw_key,
                        )
                        value = metric.transform(
                            raw_value,
                            float(point["slot_duration_ps"]),
                        )
                        if not isfinite(value):
                            raise ValueError(
                                f"non-finite transformed metric in record: {key}"
                            )
                        transformed[metric.key][method].append(value)

            for metric in METRICS:
                for method in methods:
                    values = transformed[metric.key][method]
                    mean, ci_low, ci_high = _mean_ci95(
                        values,
                        lower_bound=metric.lower_bound,
                        upper_bound=metric.upper_bound,
                    )
                    summary_rows.append({
                        "suite": suite_id,
                        "x_label": suite["x_label"],
                        "point_index": point_index,
                        "point_id": point_id,
                        "point_value": point["value"],
                        "method": method,
                        "metric": metric.key,
                        "metric_label": metric.label,
                        "direction": metric.direction,
                        "sample_count": len(values),
                        "mean": mean,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                    })
                arcq_values = transformed[metric.key]["ARC-Q"]
                for baseline in methods[1:]:
                    baseline_values = transformed[metric.key][baseline]
                    if metric.direction == "lower":
                        improvements = [
                            baseline_value - arcq_value
                            for arcq_value, baseline_value in zip(
                                arcq_values,
                                baseline_values,
                                strict=True,
                            )
                        ]
                    else:
                        improvements = [
                            arcq_value - baseline_value
                            for arcq_value, baseline_value in zip(
                                arcq_values,
                                baseline_values,
                                strict=True,
                            )
                        ]
                    mean, ci_low, ci_high = _mean_ci95(improvements)
                    paired_rows.append({
                        "suite": suite_id,
                        "point_index": point_index,
                        "point_id": point_id,
                        "point_value": point["value"],
                        "baseline": baseline,
                        "metric": metric.key,
                        "sample_count": len(improvements),
                        "mean_arcq_improvement": mean,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                    })

    selected_record_keys = {
        key for key in records_by_key if key[0] in selected
    }
    unexpected_keys = selected_record_keys - expected_keys
    if unexpected_keys:
        raise ValueError(
            f"unexpected result record: {sorted(unexpected_keys)[0]}"
        )

    canonical_source = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return {
        "schema_version": PLOT_SCHEMA_VERSION,
        "source_schema_version": payload["schema_version"],
        "source_protocol_fingerprint": payload.get("protocol_fingerprint"),
        "source_checkpoint_sha256": payload.get("checkpoint_sha256"),
        "source_checkpoint_provenance": payload.get(
            "checkpoint_provenance"
        ),
        "source_repository_provenance": payload.get(
            "repository_provenance"
        ),
        "source_runtime_provenance": payload.get("runtime_provenance"),
        "source_result_sha256": hashlib.sha256(canonical_source).hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_suites": list(selected),
        "methods": list(methods),
        "confidence_interval": (
            "two-sided 95% Student-t interval over paired topology/request "
            "instances, clipped to the metric domain"
        ),
        "paired_improvement_sign": "positive values favor ARC-Q",
        "validity": {
            "expected_record_count": expected_record_count,
            "observed_record_count": len(selected_record_keys),
            "schedule_violation_count": 0,
            "physical_backend_rejection_count": 0,
            "maximum_reward_identity_error": maximum_reward_identity_error,
        },
        "summary_rows": summary_rows,
        "paired_rows": paired_rows,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty source-data table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_source_data(
    summary: Mapping[str, object],
    output_directory: str | Path,
) -> tuple[Path, Path, Path]:
    """Write aggregate and paired source data using atomic replacement."""

    output = Path(output_directory)
    source_directory = output / "source_data"
    source_directory.mkdir(parents=True, exist_ok=True)
    json_path = source_directory / "summary.json"
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(json_path)
    summary_csv = source_directory / "summary.csv"
    paired_csv = source_directory / "paired_differences.csv"
    _write_csv(summary_csv, list(summary["summary_rows"]))
    _write_csv(paired_csv, list(summary["paired_rows"]))
    return json_path, summary_csv, paired_csv


def _apply_paper_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 6.7,
        "axes.linewidth": 0.8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "legend.frameon": False,
        "grid.color": "#D9D9D9",
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "savefig.dpi": 300,
    })


def _fallback_style(index: int) -> dict[str, object]:
    colors = ("#56B4E9", "#000000", "#F0E442", "#7E6148")
    markers = ("X", "<", ">", "*")
    return {
        "color": colors[index % len(colors)],
        "marker": markers[index % len(markers)],
        "linestyle": "--",
        "linewidth": 1.1,
        "filled": False,
    }


def _plot_one(
    rows: Sequence[Mapping[str, object]],
    *,
    methods: Sequence[str],
    suite_id: str,
    metric: MetricSpec,
    x_label: str,
    output_directory: Path,
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    metric_rows = [
        row
        for row in rows
        if row["suite"] == suite_id and row["metric"] == metric.key
    ]
    if not metric_rows:
        raise ValueError(f"no source data for {suite_id}/{metric.key}")
    ordered_points = sorted(
        {
            (int(row["point_index"]), str(row["point_id"]))
            for row in metric_rows
        }
    )
    point_values = []
    for point_index, point_id in ordered_points:
        matching = next(
            row
            for row in metric_rows
            if int(row["point_index"]) == point_index
            and str(row["point_id"]) == point_id
        )
        point_values.append(matching["point_value"])
    numeric_x = all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in point_values
    )
    if numeric_x:
        x_positions = np.asarray(point_values, dtype=float)
        x_tick_labels = [f"{value:g}" for value in x_positions]
    else:
        x_positions = np.arange(len(point_values), dtype=float)
        x_tick_labels = [str(value) for value in point_values]

    fig, ax = plt.subplots(figsize=(3.5, 2.45))
    all_lows: list[float] = []
    all_highs: list[float] = []
    for method_index, method in enumerate(methods):
        series = sorted(
            (row for row in metric_rows if row["method"] == method),
            key=lambda row: int(row["point_index"]),
        )
        if len(series) != len(ordered_points):
            raise ValueError(f"incomplete plotted series: {suite_id}/{method}")
        means = np.asarray([float(row["mean"]) for row in series])
        lows = np.asarray([float(row["ci95_low"]) for row in series])
        highs = np.asarray([float(row["ci95_high"]) for row in series])
        all_lows.extend(lows.tolist())
        all_highs.extend(highs.tolist())
        style = dict(METHOD_STYLES.get(method, _fallback_style(method_index)))
        color = str(style.pop("color"))
        filled = bool(style.pop("filled"))
        ax.errorbar(
            x_positions,
            means,
            yerr=np.vstack((means - lows, highs - means)),
            color=color,
            markerfacecolor=color if filled else "white",
            markeredgecolor=color,
            markeredgewidth=0.8,
            markersize=4.0,
            capsize=2.0,
            capthick=0.7,
            elinewidth=0.7,
            label=method,
            zorder=4 if method == "ARC-Q" else 3,
            **style,
        )

    ax.set_xlabel(X_LABELS.get(x_label, x_label.replace("_", " ").title()))
    ax.set_ylabel(metric.label)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_tick_labels)
    if not numeric_x and any(len(label) > 8 for label in x_tick_labels):
        ax.tick_params(axis="x", labelrotation=18)
    ax.grid(True, axis="both")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    low = min(all_lows)
    high = max(all_highs)
    span = high - low
    minimum_span = {
        "mean_censored_latency": 1.0,
        "completion_rate": 5.0,
        "delay_fairness": 0.05,
        "planning_time": 0.1,
    }[metric.key]
    span = max(span, minimum_span)
    lower = low - 0.08 * span
    upper = high + 0.12 * span
    if metric.lower_bound is not None:
        lower = max(metric.lower_bound, lower)
    if metric.upper_bound is not None:
        upper = min(metric.upper_bound, upper)
    if upper <= lower:
        upper = lower + minimum_span
    ax.set_ylim(lower, upper)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(3, len(methods)),
        handlelength=2.0,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.20, top=0.76)

    output_directory.mkdir(parents=True, exist_ok=True)
    stem = output_directory / f"{suite_id}_{metric.key}"
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_summary(
    summary: Mapping[str, object],
    output_directory: str | Path,
) -> tuple[Path, ...]:
    """Render deterministic figures from already-aggregated source data."""

    import matplotlib

    matplotlib.use("Agg")
    _apply_paper_style()
    output = Path(output_directory)
    rows = list(summary["summary_rows"])
    methods = tuple(str(item) for item in summary["methods"])
    saved: list[Path] = []
    for suite_id in summary["selected_suites"]:
        suite_rows = [row for row in rows if row["suite"] == suite_id]
        if not suite_rows:
            raise ValueError(f"no summary rows for suite: {suite_id}")
        x_label = str(suite_rows[0]["x_label"])
        for metric in METRICS:
            saved.extend(_plot_one(
                rows,
                methods=methods,
                suite_id=str(suite_id),
                metric=metric,
                x_label=x_label,
                output_directory=output,
            ))
    for path in saved:
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"empty plot artifact: {path}")
    return tuple(saved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/arcq/formal/raw_results.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=None,
        help="plot one completed suite; repeat to select several",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="validate and write source data without rendering figures",
    )
    arguments = parser.parse_args()
    output_directory = (
        arguments.output_dir
        if arguments.output_dir is not None
        else arguments.input.parent / "figures"
    )
    payload = load_result_payload(arguments.input)
    summary = summarize_results(payload, suite_ids=arguments.suite)
    source_paths = write_source_data(summary, output_directory)
    figure_paths: tuple[Path, ...] = ()
    if not arguments.summary_only:
        figure_paths = plot_summary(summary, output_directory)
    print(json.dumps({
        "input": str(arguments.input),
        "output_directory": str(output_directory),
        "selected_suites": summary["selected_suites"],
        "source_data": [str(path) for path in source_paths],
        "figure_count": len(figure_paths),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "METRICS",
    "load_result_payload",
    "main",
    "plot_summary",
    "summarize_results",
    "write_source_data",
]
