"""Plot the fixed final experiments (E1/E2/E3) from saved JSON reports."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("results/final_experiments")
CASES = (
    ("E1_oracle", "E1: Oracle (64 nodes, 20 req)"),
    ("E2_throughput", "E2: Throughput (64 nodes, 250 req)"),
    ("E3_ba128", "E3: BA-128 generalization"),
    ("E3_waxman192", "E3: Waxman-192 generalization"),
)
METHOD_ORDER = ("milp", "gnn", "qpass", "greedy")
METHOD_COLORS = {
    "milp": "#1f77b4",
    "gnn": "#2ca02c",
    "qpass": "#ff7f0e",
    "greedy": "#d62728",
}


def latest_json(case: str) -> Path:
    folder = ROOT / case
    candidates = sorted(folder.glob("online_gnn_comparison_*.json"))
    if not candidates:
        raise FileNotFoundError(f"no report for {case}")
    return candidates[-1]


def aggregate(case: str) -> dict[str, dict[str, float]]:
    payload = json.loads(latest_json(case).read_text(encoding="utf-8"))
    return payload["aggregate"]


def main() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for col, (case, title) in enumerate(CASES):
        agg = aggregate(case)
        methods = [m for m in METHOD_ORDER if m in agg]
        labels = methods
        completed = [agg[m]["completed_requests"] for m in methods]
        latency_ms = [agg[m]["mean_censored_latency_ps"] / 1e6 for m in methods]

        ax_top = axes[0][col]
        colors = [METHOD_COLORS[m] for m in methods]
        ax_top.bar(labels, completed, color=colors)
        ax_top.set_title(title, fontsize=11)
        ax_top.set_ylabel("Completed requests")
        ax_top.grid(axis="y", alpha=0.25)

        ax_bottom = axes[1][col]
        ax_bottom.bar(labels, latency_ms, color=colors)
        ax_bottom.set_ylabel("Mean censored latency (ms)")
        ax_bottom.grid(axis="y", alpha=0.25)

    fig.suptitle("Fixed final experiments", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_dir = ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "final_experiments_summary.png"
    fig.savefig(out, dpi=180)
    print(out)


if __name__ == "__main__":
    main()
