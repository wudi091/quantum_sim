"""Render publication-style figures from the canonical core-value JSON files.

The plotting layer is evidence-bound: all measurements are read from result
JSON files.  The online panels use the recorded per-episode values to show
mean and 95% confidence intervals; no experiment or algorithm code is called.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "core_value"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "figures"

# Okabe-Ito colors used consistently for method identity across all figures.
COLORS = {
    "trained": "#0072B2",
    "untrained": "#8C8C8C",
    "gnn": "#0072B2",
    "milp": "#D55E00",
    "qcast": "#009E73",
}
HATCHES = {"trained": "", "untrained": "///"}
METHOD_LABELS = {"gnn": "GNN", "milp": "MILP", "qcast": "Q-CAST"}
METHOD_ORDER = ("gnn", "milp", "qcast")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"result must be a JSON object: {path}")
    return payload


def _setup_style() -> None:
    """Use the compact publication style already used by qnet_sim."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.8,
            "figure.titlesize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
        }
    )


def _style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", direction="in", length=3.0, width=0.8)
    ax.grid(True, axis=grid_axis, linestyle=":", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)


def _save_figure(fig: plt.Figure, output: Path, name: str) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / f"{name}.png", output / f"{name}.pdf", output / f"{name}.svg"]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    fig.savefig(paths[2], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [str(path) for path in paths]


def _annotate_bars(ax: plt.Axes, bars: Iterable[Any], values: Iterable[float], fmt: str) -> None:
    for bar, value in zip(bars, values):
        ax.annotate(
            fmt.format(float(value)),
            (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )


def plot_quality(results: Path, output: Path) -> dict[str, Any]:
    source = results / "offline" / "lp_quality.json"
    payload = _read_json(source)
    objective_ratio = [
        float(payload["trained"]["mean_objective_ratio"]),
        float(payload["untrained"]["mean_objective_ratio"]),
    ]
    teacher_gap = [100.0 * (1.0 - value) for value in objective_ratio]
    feasible = [
        100.0 * float(payload["trained"]["rounded_feasible_rate"]),
        100.0 * float(payload["untrained"]["rounded_feasible_rate"]),
    ]
    labels = ["Trained\nGNN", "Untrained\nGNN"]
    x = np.arange(2)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), constrained_layout=True)
    bars = axes[0].bar(
        x,
        objective_ratio,
        width=0.56,
        color=[COLORS["trained"], COLORS["untrained"]],
        edgecolor="black",
        linewidth=0.5,
        hatch=[HATCHES["trained"], HATCHES["untrained"]],
    )
    axes[0].set_title("(a) Objective recovery")
    axes[0].set_ylabel("Ratio to LP teacher")
    axes[0].set_ylim(0.0, 1.08)
    axes[0].set_xticks(x, labels)
    _annotate_bars(axes[0], bars, objective_ratio, "{:.3f}")
    _style_axis(axes[0])

    bars = axes[1].bar(
        x,
        teacher_gap,
        width=0.56,
        color=[COLORS["trained"], COLORS["untrained"]],
        edgecolor="black",
        linewidth=0.5,
        hatch=[HATCHES["trained"], HATCHES["untrained"]],
    )
    axes[1].set_title("(b) Gap to teacher")
    axes[1].set_ylabel("Objective gap (%)")
    axes[1].set_ylim(0.0, max(1.0, max(teacher_gap) * 1.18))
    axes[1].set_xticks(x, labels)
    _annotate_bars(axes[1], bars, teacher_gap, "{:.2f}%")
    _style_axis(axes[1])

    bars = axes[2].bar(
        x,
        feasible,
        width=0.56,
        color=[COLORS["trained"], COLORS["untrained"]],
        edgecolor="black",
        linewidth=0.5,
        hatch=[HATCHES["trained"], HATCHES["untrained"]],
    )
    axes[2].set_title("(c) Rounded feasibility")
    axes[2].set_ylabel("Feasible rate (%)")
    axes[2].set_ylim(0.0, 108.0)
    axes[2].set_xticks(x, labels)
    _annotate_bars(axes[2], bars, feasible, "{:.1f}")
    _style_axis(axes[2])

    n_samples = int(payload["case"]["samples"])
    fig.text(0.5, -0.04, f"LP-teacher recovery on held-out samples (n={n_samples})", ha="center", fontsize=8.5)
    paths = _save_figure(fig, output, "core_value_quality")
    return {"name": "quality", "source": str(source), "outputs": paths}


def plot_generalization(results: Path, output: Path) -> dict[str, Any]:
    names = ["waxman_scale", "ba_transfer"]
    sources = [results / "offline" / f"{name}.json" for name in names]
    payloads = [_read_json(path) for path in sources]
    labels = ["Waxman\nscale", "BA\ntransfer"]
    x = np.arange(len(labels))
    trained_ratio = [float(item["trained"]["mean_objective_ratio"]) for item in payloads]
    untrained_ratio = [float(item["untrained"]["mean_objective_ratio"]) for item in payloads]
    trained_gap = [100.0 * (1.0 - value) for value in trained_ratio]
    untrained_gap = [100.0 * (1.0 - value) for value in untrained_ratio]
    trained_feasible = [100.0 * float(item["trained"]["rounded_feasible_rate"]) for item in payloads]
    untrained_feasible = [100.0 * float(item["untrained"]["rounded_feasible_rate"]) for item in payloads]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.35), constrained_layout=True)
    for values, label, color, marker in (
        (trained_gap, "Trained GNN", COLORS["trained"], "o"),
        (untrained_gap, "Untrained GNN", COLORS["untrained"], "s"),
    ):
        axes[0].plot(x, values, color=color, marker=marker, markersize=4.5, linewidth=1.25, label=label)
    axes[0].set_title("(a) Teacher gap on unseen graphs")
    axes[0].set_ylabel("Objective gap (%)")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0, max(untrained_gap) * 1.16)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2, handlelength=1.5, columnspacing=0.8)
    _style_axis(axes[0])

    for values, label, color, marker in (
        (trained_feasible, "Trained GNN", COLORS["trained"], "o"),
        (untrained_feasible, "Untrained GNN", COLORS["untrained"], "s"),
    ):
        axes[1].plot(x, values, color=color, marker=marker, markersize=4.5, linewidth=1.25, label=label)
    axes[1].set_title("(b) Feasibility on unseen graphs")
    axes[1].set_ylabel("Rounded feasible rate (%)")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.0, 108.0)
    _style_axis(axes[1])

    n_samples = int(payloads[0]["case"]["samples"])
    fig.text(0.5, -0.04, f"Held-out topology and scale transfer (n={n_samples} per setting)", ha="center", fontsize=8.5)
    paths = _save_figure(fig, output, "core_value_generalization")
    return {"name": "generalization", "source": [str(path) for path in sources], "outputs": paths}


def _online_payloads(results: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    records = []
    for name in ("waxman_main", "ba_transfer"):
        path = results / "online" / name / "online_gnn_comparison.json"
        if path.is_file():
            records.append((name, path, _read_json(path)))
    if not records:
        raise FileNotFoundError("no completed online comparison JSON files found")
    return records


def _mean_ci(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("cannot summarize an empty series")
    mean = float(np.mean(array))
    if array.size < 2:
        return mean, 0.0
    ci = 1.96 * float(np.std(array, ddof=1)) / np.sqrt(array.size)
    return mean, ci


def _online_series(payload: dict[str, Any], method: str, metric: str) -> tuple[float, float]:
    values = []
    slot_duration = float(payload["scenario"]["physical"]["slot_duration_ps"])
    for trial in payload["trials"]:
        value = float(trial["methods"][method]["metrics"][metric])
        if metric == "mean_censored_latency_ps":
            value /= slot_duration
        values.append(value)
    return _mean_ci(values)


def plot_online(results: Path, output: Path) -> dict[str, Any]:
    records = _online_payloads(results)
    topology_labels = [name.replace("_", " ").title().replace("Ba", "BA") for name, _, _ in records]
    metrics = [
        ("completed_requests", "(a) Completed requests", "Requests", False),
        ("throughput_per_slot", "(b) Throughput", "Requests / slot", False),
        ("mean_censored_latency_ps", "(c) Completion latency", "Slots", False),
        ("mean_decision_seconds", "(d) Decision time", "Seconds (log scale)", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.55), constrained_layout=True)
    x = np.arange(len(records))
    for ax, (metric, title, ylabel, log_scale) in zip(axes.flat, metrics):
        for method in METHOD_ORDER:
            means = []
            cis = []
            for _, _, payload in records:
                mean, ci = _online_series(payload, method, metric)
                means.append(mean)
                cis.append(ci)
            ax.errorbar(
                x,
                means,
                yerr=cis,
                color=COLORS[method],
                marker={"gnn": "o", "milp": "^", "qcast": "s"}[method],
                markersize=4.5,
                linewidth=1.25,
                capsize=2.5,
                capthick=0.7,
                elinewidth=0.7,
                label=METHOD_LABELS[method],
            )
        ax.set_title(title, pad=4.0)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, topology_labels)
        if log_scale:
            ax.set_yscale("log")
        _style_axis(ax)

    axes[0, 0].legend(
        loc="upper center",
        bbox_to_anchor=(1.05, 1.28),
        ncol=3,
        handlelength=1.5,
        columnspacing=0.8,
    )
    episode_counts = [int(payload["configuration"]["seeds"]) for _, _, payload in records]
    fig.text(0.5, -0.03, f"Paired online episodes; markers show means and bars show 95% CI ({episode_counts[0]} episodes per topology)", ha="center", fontsize=8.5)
    paths = _save_figure(fig, output, "core_value_online")
    return {"name": "online", "source": [str(path) for _, path, _ in records], "outputs": paths}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    results = args.results if args.results.is_absolute() else ROOT / args.results
    output = args.output if args.output.is_absolute() else ROOT / args.output
    _setup_style()
    generated = [plot_quality(results, output), plot_generalization(results, output), plot_online(results, output)]
    manifest = {
        "schema_version": 2,
        "experiment": "core_value_figures",
        "source_root": str(results),
        "output_root": str(output),
        "style_reference": "qnet_sim IEEE publication plotting conventions",
        "palette": COLORS,
        "uncertainty": "online figures use mean +/- 1.96*sample_std/sqrt(n) from recorded trials",
        "figures": generated,
        "note": "All values are read from canonical experiment JSON files; no measurements are embedded in the plotting code.",
    }
    (output / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
