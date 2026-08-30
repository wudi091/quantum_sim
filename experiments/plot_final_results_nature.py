"""Nature-style figure for the fixed final experiments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results/final_experiments")
OUT = ROOT / "figures" / "final_experiments_nature.png"
CASES = (
    ("E1_oracle", "E1: Oracle (64 nodes, 20 req)"),
    ("E2_throughput", "E2: Throughput (64 nodes, 250 req)"),
    ("E3_ba128", "E3: BA-128 generalization"),
    ("E3_waxman192", "E3: Waxman-192 generalization"),
)
METHOD_ORDER = ("milp", "gnn", "qpass", "greedy")
METHOD_LABELS = {
    "milp": "MILP",
    "gnn": "GNN",
    "qpass": "Q-PASS",
    "greedy": "Greedy",
}
METHOD_COLORS = {
    "milp": "#4C72B0",
    "gnn": "#55A868",
    "qpass": "#DD8452",
    "greedy": "#C44E52",
}
RNG = np.random.default_rng(20260829)


def latest_json(case: str) -> Path:
    candidates = sorted((ROOT / case).glob("online_gnn_comparison_*.json"))
    if not candidates:
        raise FileNotFoundError(f"missing report for {case}")
    return candidates[-1]


def trial_values(case: str, metric: str) -> dict[str, np.ndarray]:
    payload = json.loads(latest_json(case).read_text(encoding="utf-8"))
    values: dict[str, list[float]] = {}
    for trial in payload["trials"]:
        for method, method_payload in trial["methods"].items():
            values.setdefault(method, []).append(
                float(method_payload["metrics"][metric])
            )
    return {m: np.asarray(v, dtype=float) for m, v in values.items()}


def bootstrap_ci(values: np.ndarray) -> tuple[float, float, float]:
    n = len(values)
    boots = RNG.choice(values, size=(2000, n), replace=True)
    means = boots.mean(axis=1)
    low, high = np.percentile(means, (2.5, 97.5))
    return float(values.mean()), float(low), float(high)


def style_axis(ax: plt.Axes) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#666666")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", length=2.5, width=0.8,
                   color="#666666", labelsize=7)
    ax.yaxis.grid(False)


def plot_panel(ax: plt.Axes, values: dict[str, np.ndarray], title: str,
               ylabel: str, transform=None) -> None:
    methods = [m for m in METHOD_ORDER if m in values]
    means, lows, highs = [], [], []
    for m in methods:
        v = values[m]
        if transform is not None:
            v = transform(v)
        mean, low, high = bootstrap_ci(v)
        means.append(mean)
        lows.append(mean - low)
        highs.append(high - mean)
    x = np.arange(len(methods))
    colors = [METHOD_COLORS[m] for m in methods]
    ax.bar(x, means, color=colors, width=0.72, edgecolor="black",
           linewidth=0.5, zorder=2)
    ax.errorbar(x, means, yerr=[lows, highs], fmt="none", ecolor="black",
                elinewidth=0.8, capsize=2.5, zorder=3)
    for xi, m in zip(x, methods):
        v = values[m]
        if transform is not None:
            v = transform(v)
        jitter = RNG.uniform(-0.18, 0.18, size=len(v))
        ax.scatter(xi + jitter, v, s=6, color="black", alpha=0.45,
                   linewidth=0, zorder=4)
    ax.set_xticks(x, [METHOD_LABELS[m] for m in methods], fontsize=7)
    ax.set_title(title, fontsize=8, pad=5)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.margins(y=0.12)
    style_axis(ax)


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 4, figsize=(9.2, 4.6), constrained_layout=False)
    for col, (case, title) in enumerate(CASES):
        completed = trial_values(case, "completed_requests")
        latency = trial_values(case, "mean_censored_latency_ps")
        plot_panel(axes[0][col], completed, title, "Completed requests")
        plot_panel(axes[1][col], latency, "", "Mean censored latency (ms)",
                   transform=lambda v: v / 1e6)
    fig.suptitle("Construction-aware routing results", fontsize=9, y=0.985)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.93, bottom=0.11,
                        hspace=0.32, wspace=0.38)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"))
    print(OUT)


if __name__ == "__main__":
    main()
