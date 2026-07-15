"""Plot normalized official and SeQUeNCe Q-DDCA trend curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _norm(values):
    values = np.asarray(values, dtype=float)
    maximum = np.max(values)
    return values if maximum <= 0 else values / maximum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/qddca_sequence_trends_3seed.json"))
    parser.add_argument("--output", type=Path, default=Path("results/qddca_sequence_trends.png"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    official, sequence = data["official"], data["sequence"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    axes[0].plot(official["retry"]["x"], _norm(official["retry"]["throughput"]), "o-", label="Official")
    axes[0].plot(official["retry"]["x"], _norm([row["throughput"] for row in sequence["retry"]]), "s--", label="SeQUeNCe")
    axes[0].set(title="Retry count vs throughput", xlabel="M", ylabel="Normalized throughput")

    axes[1].plot(official["window"]["x"], _norm(official["window"]["throughput"]), "o-")
    axes[1].plot(official["window"]["x"], _norm([row["throughput"] for row in sequence["window"]]), "s--")
    axes[1].set(title="Window congestion curve", xlabel="Window", ylabel="Normalized throughput")

    labels = ["No reroute", "Reroute"]
    x = np.arange(2)
    axes[2].bar(x - 0.18, _norm([official["reroute"]["false"]["throughput"], official["reroute"]["true"]["throughput"]]), 0.36, label="Official")
    axes[2].bar(x + 0.18, _norm([sequence["reroute"]["false"]["throughput"], sequence["reroute"]["true"]["throughput"]]), 0.36, label="SeQUeNCE")
    axes[2].set(title="Rerouting gain", xticks=x, xticklabels=labels, ylabel="Normalized throughput")
    axes[0].legend()
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
