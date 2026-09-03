"""Plot the configurable online-experiment summaries.

The execution protocol and this plotting layer are deliberately separate.  The
script reads the ``experiment_summary.csv`` (or its JSON counterpart) produced
by :mod:`experiments.run_online_experiments` and writes one line-plot figure per
experiment for the recorded delay, fidelity, fairness, planning-time, and
legacy completion-count metrics.  It never
re-runs an experiment and never embeds measurements in the source code.

Examples
--------
Use the newest completed run automatically::

    python -m experiments.plot_online_experiments

Use a specific summary and output directory::

    python -m experiments.plot_online_experiments \
        --input results/online_long_run/run_20260903_120000/experiment_summary.csv \
        --output results/online_long_run/run_20260903_120000/figures
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ROOT / "results" / "online_long_run"

# Okabe-Ito colours are readable in colour and grayscale when paired with the
# marker/line differences below.  The first three are kept aligned with the
# existing online-comparison figures.
METHOD_COLORS = {
    "gnn": "#0072B2",
    "milp": "#D55E00",
    "qcast": "#009E73",
    "qpass": "#CC79A7",
    "greedy": "#E69F00",
}
FALLBACK_COLORS = (
    "#56B4E9",
    "#F0E442",
    "#000000",
)
METHOD_MARKERS = {
    "gnn": "o",
    "milp": "s",
    "qcast": "^",
    "qpass": "D",
    "greedy": "v",
}
FALLBACK_MARKERS = ("P", "X", "<", ">", "d", "p", "h")
LINE_STYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)))
METHOD_LABELS = {
    "gnn": "GNN",
    "milp": "MILP",
    "qcast": "Q-CAST",
    "qpass": "Q-PASS",
    "greedy": "Greedy",
}
METHOD_ORDER = ("gnn", "milp", "qcast", "qpass", "greedy")

EXPERIMENT_TITLES = {
    "standard_stability": "Temporal stability",
    "request_load": "Request-load scaling",
    "network_scale": "Network-scale scaling",
    "topology_generalization": "Topology generalization",
    "physical_conditions_link_generation": "Link-generation probability",
    "physical_conditions_swapping": "Swapping probability",
}
AXIS_LABELS = {
    "time_segment": "Time Segment",
    "requests": "Number of Requests",
    "nodes": "Number of Quantum Nodes",
    "topology": "Topology",
    "generation_probability": "Link Generation Probability",
    "swap_probability": "Swap Success Probability",
}
X_VALUE_LABELS = {
    "unseen_waxman": "Waxman",
    "waxman": "Waxman",
    "barabasi_albert": "BA",
    "ba": "BA",
    "cost266": "Cost266",
    "germany": "Germany",
    "bellcanada": "Bellcanada",
}
REQUIRED_COLUMNS = {
    "experiment",
    "x_axis",
    "x_value",
    "method",
    "completed_requests",
    "planning_time_seconds",
}
OPTIONAL_METRIC_COLUMNS = (
    "mean_completion_delay_slots",
    "max_completion_delay_slots",
    "mean_final_fidelity_loss",
    "completion_delay_gini",
)
PLOT_METRIC_SPECS = (
    (
        "mean_completion_delay_slots",
        "Mean Completion Delay",
        "Delay (slots)",
    ),
    (
        "max_completion_delay_slots",
        "Maximum Completion Delay",
        "Delay (slots)",
    ),
    (
        "mean_final_fidelity_loss",
        "Final Fidelity Loss",
        "Loss (1 - fidelity)",
    ),
    (
        "completion_delay_gini",
        "Completion-delay Fairness",
        "Gini (lower is fairer)",
    ),
    (
        "planning_time_seconds",
        "Planning Time",
        "Planning Time (s)",
    ),
    (
        "completed_requests",
        "Completed Requests",
        "Completed Requests",
    ),
)


def _setup_style() -> None:
    """Use the compact IEEE style used by qnet_sim's sweep plots."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 1.05,
            "grid.linewidth": 0.6,
            "grid.linestyle": ":",
            "grid.alpha": 0.45,
            "legend.frameon": False,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
        }
    )


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _latest_summary(root: Path) -> Path:
    candidates = sorted(
        (
            path
            for path in root.glob("run_*/experiment_summary.csv")
            if path.is_file() and path.stat().st_size > 0
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "no completed experiment_summary.csv found below "
            f"{root}; run the experiment protocol first or pass --input"
        )
    return candidates[0]


def _parse_x_value(value: str) -> int | float | str:
    text = value.strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if not math.isfinite(number):
        return text
    if number.is_integer() and re.fullmatch(r"[+-]?\d+", text):
        return int(number)
    return number


def _finite_float(value: object, field: str, row_number: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"row {row_number}: {field} must be numeric, got {value!r}"
        ) from exc
    if not math.isfinite(result):
        raise ValueError(f"row {row_number}: {field} must be finite")
    return result


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(sorted(missing))}"
            )
        rows: list[dict[str, Any]] = []
        for row_number, raw in enumerate(reader, start=2):
            if not raw.get("experiment"):
                raise ValueError(f"row {row_number}: experiment is empty")
            method = str(raw.get("method", "")).strip().lower()
            if not method:
                raise ValueError(f"row {row_number}: method is empty")
            normalized = {
                    "experiment": str(raw["experiment"]),
                    "x_axis": str(raw["x_axis"]),
                    "x_value": _parse_x_value(str(raw["x_value"])),
                    "method": method,
                    "completed_requests": _finite_float(
                        raw["completed_requests"],
                        "completed_requests",
                        row_number,
                    ),
                    "planning_time_seconds": _finite_float(
                        raw["planning_time_seconds"],
                        "planning_time_seconds",
                        row_number,
                    ),
            }
            for metric in OPTIONAL_METRIC_COLUMNS:
                value = raw.get(metric, "")
                if value not in (None, ""):
                    normalized[metric] = _finite_float(
                        value,
                        metric,
                        row_number,
                    )
            rows.append(normalized)
    if not rows:
        raise ValueError(f"{path} contains no experiment rows")
    return rows


def _read_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"summary JSON must be an object: {path}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"summary JSON contains no rows: {path}")
    normalized: list[dict[str, Any]] = []
    for row_number, raw in enumerate(rows, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"row {row_number}: expected an object")
        required = REQUIRED_COLUMNS - set(raw)
        if required:
            raise ValueError(
                f"row {row_number}: missing columns {', '.join(sorted(required))}"
            )
        method = str(raw["method"]).strip().lower()
        normalized_row = {
                "experiment": str(raw["experiment"]),
                "x_axis": str(raw["x_axis"]),
                "x_value": raw["x_value"],
                "method": method,
                "completed_requests": _finite_float(
                    raw["completed_requests"],
                    "completed_requests",
                    row_number,
                ),
                "planning_time_seconds": _finite_float(
                    raw["planning_time_seconds"],
                    "planning_time_seconds",
                    row_number,
                ),
        }
        for metric in OPTIONAL_METRIC_COLUMNS:
            if metric in raw and raw[metric] not in (None, ""):
                normalized_row[metric] = _finite_float(
                    raw[metric],
                    metric,
                    row_number,
                )
        normalized.append(normalized_row)
    return normalized


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read and validate a CSV or JSON experiment summary."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv(path)
    if suffix == ".json":
        return _read_json(path)
    raise ValueError(f"input must be .csv or .json: {path}")


def _method_order(methods: Iterable[str]) -> list[str]:
    available = set(methods)
    preferred = [method for method in METHOD_ORDER if method in available]
    return preferred + sorted(available - set(preferred))


def _method_color(method: str, index: int) -> str:
    return METHOD_COLORS.get(method, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def _method_marker(method: str, index: int) -> str:
    return METHOD_MARKERS.get(method, FALLBACK_MARKERS[index % len(FALLBACK_MARKERS)])


def _display_method(method: str) -> str:
    return METHOD_LABELS.get(method, method.upper())


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip().lower())
    return value.strip("_") or "experiment"


def _experiment_title(experiment: str) -> str:
    if experiment in EXPERIMENT_TITLES:
        return EXPERIMENT_TITLES[experiment]
    return experiment.replace("_", " ").title()


def _axis_label(axis: str) -> str:
    return AXIS_LABELS.get(axis, axis.replace("_", " ").title())


def _x_value_label(value: object) -> str:
    text = str(value)
    return X_VALUE_LABELS.get(text.lower(), text)


def _group_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group duplicate rows by experiment/x/method using a mean.

    The protocol normally emits one aggregate row per key.  Averaging duplicate
    keys makes the plotting layer tolerant of a user-concatenated summary while
    keeping the aggregation explicit and deterministic.
    """

    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    x_order: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        experiment = str(row["experiment"])
        axis = str(row["x_axis"])
        method = str(row["method"]).lower()
        x_value = row["x_value"]
        x_key = str(x_value)
        grouped.setdefault((experiment, axis, method, x_key), []).append(row)
        order_key = (experiment, axis)
        if x_key not in x_order.setdefault(order_key, []):
            x_order[order_key].append(x_key)

    result: dict[str, list[dict[str, Any]]] = {}
    for (experiment, axis), ordered_x in x_order.items():
        output_rows: list[dict[str, Any]] = []
        methods = _method_order(
            method
            for (exp, current_axis, method, _), values in grouped.items()
            if exp == experiment and current_axis == axis and values
        )
        for x_key in ordered_x:
            for method in methods:
                values = grouped.get((experiment, axis, method, x_key))
                if not values:
                    continue
                output_row: dict[str, Any] = {
                        "experiment": experiment,
                        "x_axis": axis,
                        "x_value": values[0]["x_value"],
                        "method": method,
                        "completed_requests": sum(
                            float(item["completed_requests"])
                            for item in values
                        ) / len(values),
                        "planning_time_seconds": sum(
                            float(item["planning_time_seconds"]) for item in values
                        )
                        / len(values),
                }
                for metric in OPTIONAL_METRIC_COLUMNS:
                    metric_values = [
                        float(item[metric])
                        for item in values
                        if metric in item
                    ]
                    if metric_values:
                        output_row[metric] = sum(metric_values) / len(metric_values)
                output_rows.append(output_row)
        result[experiment] = output_rows
    return result


def _x_coordinates(values: Sequence[object]) -> tuple[list[float], list[str], bool]:
    """Map sweep values to equally spaced positions, matching qnet_sim.

    The reference sweep plot treats a sweep as a categorical sequence even
    when its labels are numeric.  This keeps every sampled operating point
    equally visible instead of placing points according to numeric gaps.
    """

    numeric = all(isinstance(value, (int, float)) for value in values)
    if numeric:
        ordered = sorted(values, key=float)
        labels = [
            str(int(value)) if float(value).is_integer() else f"{float(value):.2f}".rstrip("0").rstrip(".")
            for value in ordered
        ]
        return [float(index) for index in range(len(ordered))], labels, True
    ordered = list(values)
    coordinates = list(range(len(ordered)))
    labels = [_x_value_label(value) for value in ordered]
    return [float(value) for value in coordinates], labels, False


def _style_axis(ax: Any) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", direction="in", length=3.0, width=0.8)
    ax.grid(True, axis="both", linestyle=":", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)


def _set_limits(ax: Any, values: Sequence[float]) -> None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return
    low = min(finite)
    high = max(finite)
    span = high - low
    if span <= 0.0:
        span = max(abs(high), 1.0)
    margin = 0.05 * span
    lower = 0.0 if low >= 0.0 else low - margin
    ax.set_ylim(lower, high + margin)


def _plot_metric(
    ax: Any,
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    ylabel: str,
) -> None:
    x_values: list[object] = []
    for row in rows:
        if row["x_value"] not in x_values:
            x_values.append(row["x_value"])
    coordinates, labels, numeric = _x_coordinates(x_values)
    ordered_values = sorted(x_values, key=float) if numeric else x_values
    coordinate_by_key = {
        str(value): coordinate
        for value, coordinate in zip(ordered_values, coordinates)
    }
    methods = _method_order(str(row["method"]) for row in rows)
    all_values: list[float] = []
    for index, method in enumerate(methods):
        method_rows = [row for row in rows if str(row["method"]) == method]
        points = []
        for row in method_rows:
            if metric not in row:
                continue
            value = float(row[metric])
            if math.isfinite(value):
                points.append((coordinate_by_key[str(row["x_value"])], value))
        points.sort(key=lambda item: item[0])
        if not points:
            continue
        x, y = zip(*points)
        all_values.extend(y)
        ax.plot(
            x,
            y,
            color=_method_color(method, index),
            marker=_method_marker(method, index),
            linestyle=LINE_STYLES[index % len(LINE_STYLES)],
            markersize=2.8,
            linewidth=1.05,
            markerfacecolor="white",
            markeredgecolor=_method_color(method, index),
            markeredgewidth=0.7,
            label=_display_method(method),
        )
    ax.set_ylabel(ylabel)
    ax.set_xticks(coordinates, labels)
    if not numeric and max(len(label) for label in labels) > 10:
        ax.tick_params(axis="x", labelrotation=25)
    _set_limits(ax, all_values)
    _style_axis(ax)


def _save_figure(fig: Any, output: Path, stem: str, formats: Sequence[str], dpi: int) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    figure_width, figure_height = fig.get_size_inches()
    save_bbox = Bbox.from_bounds(
        -0.08,
        -0.14,
        float(figure_width) + 0.16,
        float(figure_height) + 0.16,
    )
    legends = fig.legends
    if legends:
        fig.canvas.draw()
        legend_bbox = legends[0].get_window_extent(
            renderer=fig.canvas.get_renderer()
        ).transformed(fig.dpi_scale_trans.inverted())
        save_bbox = Bbox.union([save_bbox, legend_bbox.padded(0.06)])
    paths: list[str] = []
    for file_format in formats:
        path = output / f"{stem}.{file_format}"
        save_kwargs: dict[str, Any] = {
            "bbox_inches": save_bbox,
            "facecolor": "white",
            "edgecolor": "white",
            "transparent": False,
        }
        if file_format == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(path, **save_kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_experiment(
    experiment: str,
    rows: Sequence[Mapping[str, Any]],
    output: Path,
    formats: Sequence[str],
    dpi: int,
) -> dict[str, Any]:
    axes_name = str(rows[0]["x_axis"])
    has_new_metrics = any(
        metric in row
        for row in rows
        for metric in OPTIONAL_METRIC_COLUMNS
    )
    if has_new_metrics:
        available_specs = [
            spec
            for spec in PLOT_METRIC_SPECS
            if any(spec[0] in row for row in rows)
        ]
    else:
        # Keep legacy summaries readable and backwards-compatible.
        available_specs = [
            next(spec for spec in PLOT_METRIC_SPECS if spec[0] == metric)
            for metric in ("completed_requests", "planning_time_seconds")
            if any(metric in row for row in rows)
        ]
    if not available_specs:
        raise ValueError(f"{experiment} contains no supported metrics")
    column_count = 2
    row_count = (len(available_specs) + column_count - 1) // column_count
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(4.70, 2.35 * row_count),
        constrained_layout=True,
    )
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    title_y = -0.40 if not all(
        isinstance(row["x_value"], (int, float)) for row in rows
    ) else -0.29
    for index, (metric, title, ylabel) in enumerate(available_specs):
        axis = axes_flat[index]
        _plot_metric(axis, rows, metric, ylabel)
        axis.text(
            0.5,
            title_y,
            f"({chr(ord('a') + index)}) {title}",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8.2,
            fontweight="bold",
        )
    # Category names already identify topology points; retain an x-axis label
    # for numeric sweeps where it carries additional meaning.
    x_label = "" if axes_name == "topology" else _axis_label(axes_name)
    if x_label:
        for axis in axes_flat[:len(available_specs)]:
            axis.set_xlabel(x_label)
    for axis in axes_flat[len(available_specs):]:
        axis.set_visible(False)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(labels),
            frameon=True,
            fancybox=True,
            edgecolor="#BFBFBF",
            fontsize=6.2,
            handlelength=1.8,
            columnspacing=0.9,
            borderaxespad=0.2,
        )
    paths = _save_figure(fig, output, "sweep_comparison_ieee", formats, dpi)
    return {
        "experiment": experiment,
        "x_axis": axes_name,
        "rows": len(rows),
        "metrics": [spec[0] for spec in available_specs],
        "outputs": paths,
    }


def generate_figures(
    input_path: Path,
    output: Path,
    formats: Sequence[str] = ("png", "pdf", "svg"),
    dpi: int = 300,
) -> dict[str, Any]:
    if not input_path.is_file():
        raise FileNotFoundError(f"summary input does not exist: {input_path}")
    rows = read_rows(input_path)
    grouped = _group_rows(rows)
    _setup_style()
    figures = [
        plot_experiment(
            experiment,
            experiment_rows,
            output / _slug(experiment) / "plots",
            formats,
            dpi,
        )
        for experiment, experiment_rows in grouped.items()
        if experiment_rows
    ]
    if not figures:
        raise ValueError("summary contains no plottable experiment rows")
    manifest = {
        "schema_version": 1,
        "experiment": "telgen_online_experiments",
        "source": str(input_path),
        "output_root": str(output),
        "metrics": {
            "completed_requests": "completed_requests",
            "mean_completion_delay_slots": "mean_completion_delay_ps / slot_duration_ps",
            "max_completion_delay_slots": "max_completion_delay_ps / slot_duration_ps",
            "mean_final_fidelity_loss": "mean_final_fidelity_loss",
            "completion_delay_gini": "completion_delay_gini",
            "planning_time_seconds": "mean_planner_seconds",
        },
        "uncertainty": "not available in experiment_summary.csv; no error bars are plotted",
        "palette": METHOD_COLORS,
        "figures": figures,
        "no_fabrication": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest["manifest"] = str(manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="experiment_summary.csv or experiment_summary.json; defaults to newest completed CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="figure directory; defaults to <summary parent>/figures",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf,svg",
        help="comma-separated output formats (png, pdf, svg)",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = (
        _resolve_path(args.input)
        if args.input is not None
        else _latest_summary(DEFAULT_RESULTS_ROOT)
    )
    output = (
        _resolve_path(args.output)
        if args.output is not None
        else input_path.parent / "figures"
    )
    formats = tuple(
        item.strip().lower()
        for item in str(args.formats).split(",")
        if item.strip()
    )
    supported = {"png", "pdf", "svg"}
    unknown = set(formats) - supported
    if not formats or unknown:
        raise ValueError(
            "formats must be a non-empty comma-separated subset of png,pdf,svg"
        )
    if args.dpi < 72:
        raise ValueError("dpi must be at least 72")
    manifest = generate_figures(input_path, output, formats, args.dpi)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
