"""Shared, publication-oriented plotting utilities for TELGEN figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np


@dataclass(frozen=True)
class MethodStyle:
    """Visual identity used consistently across all paper figures."""

    label: str
    color: str
    marker: str
    linestyle: str
    linewidth: float = 1.15
    hatch: str = ""


# Okabe-Ito-derived palette plus neutral greys.  Every line also has a
# distinct marker/linestyle so the figures remain readable in greyscale.
METHOD_STYLES: dict[str, MethodStyle] = {
    "gnn": MethodStyle("TELGEN", "#0072B2", "o", "-", 1.35, ""),
    "milp": MethodStyle("MILP", "#CC79A7", "D", ":", 1.15, "xx"),
    "qpass": MethodStyle("Q-PASS", "#D55E00", "s", "--", 1.15, "//"),
    "greedy": MethodStyle("Greedy", "#009E73", "^", "-.", 1.15, ".."),
    # Retained only for reading legacy reports.  Q-CAST is not part of the
    # frozen formal comparison set.
    "qcast": MethodStyle("Q-CAST-W1", "#7A7A7A", "X", ":", 1.15, "\\\\"),
}

ADAPTIVE_COLOR = METHOD_STYLES["gnn"].color
FIXED_TREE_COLORS = (
    "#BDBDBD",
    "#A6A6A6",
    "#8F8F8F",
    "#787878",
    "#616161",
)

SINGLE_COLUMN = (3.4, 2.4)
DOUBLE_COLUMN = (6.9, 2.75)
DOUBLE_COLUMN_TALL = (6.9, 3.1)


def configure_paper_style() -> None:
    """Apply a compact vector-friendly style suitable for two-column papers."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.8,
            "axes.linewidth": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.fontsize": 8.0,
            "legend.frameon": False,
            "legend.handlelength": 2.0,
            "legend.columnspacing": 1.1,
            "legend.handletextpad": 0.45,
            "lines.linewidth": 1.15,
            "lines.markersize": 3.8,
            "grid.color": "#B8B8B8",
            "grid.linewidth": 0.6,
            "grid.linestyle": ":",
            "grid.alpha": 0.45,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "telgen-paper-figures-v1",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 20260820,
) -> tuple[float, float, float]:
    """Return mean and percentile-bootstrap 95% confidence interval."""

    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("bootstrap input must be a non-empty 1-D sequence")
    mean = float(np.mean(data))
    if data.size == 1:
        return mean, mean, mean
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, data.size, size=(samples, data.size))
    draws = np.mean(data[indices], axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return mean, float(low), float(high)


def ecdf(values: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return x/y coordinates for an empirical cumulative distribution."""

    data = np.sort(np.asarray(values, dtype=float))
    if data.ndim != 1 or data.size == 0:
        raise ValueError("ECDF input must be a non-empty 1-D sequence")
    return data, np.arange(1, data.size + 1, dtype=float) / data.size


def style_axis(
    ax: plt.Axes,
    *,
    grid_axis: str = "y",
    zero_floor: bool = False,
) -> None:
    """Apply the compact framed-axis style used by the reference paper."""

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=3.0,
        width=0.8,
    )
    if grid_axis:
        ax.grid(True, axis=grid_axis)
    if zero_floor:
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom=0.0, top=max(top, 1e-12))


def panel_label(ax: plt.Axes, label: str, *, y: float = -0.30) -> None:
    """Place a bold subfigure caption below the axis."""

    ax.text(
        0.5,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.0,
        fontweight="bold",
        va="top",
        ha="center",
    )


def asymmetric_error(
    mean: float,
    low: float,
    high: float,
) -> np.ndarray:
    """Create Matplotlib's 2x1 asymmetric error-bar representation."""

    return np.asarray([[max(0.0, mean - low)], [max(0.0, high - mean)]])


def save_figure(
    fig: plt.Figure,
    *,
    output_directory: Path,
    stem: str,
    formats: Iterable[str],
    dpi: int = 300,
) -> list[Path]:
    """Save one figure in all requested formats and close it."""

    output_directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for raw_format in formats:
        file_format = raw_format.lower().lstrip(".")
        if file_format not in {"pdf", "svg", "png", "eps"}:
            raise ValueError(f"unsupported figure format: {raw_format}")
        path = output_directory / f"{stem}.{file_format}"
        metadata: dict[str, object] = {"Creator": "TELGEN paper figure generator"}
        if file_format == "pdf":
            metadata.update({"CreationDate": None, "ModDate": None})
        elif file_format == "svg":
            metadata["Date"] = None
        fig.savefig(path, format=file_format, dpi=dpi, metadata=metadata)
        paths.append(path)
    plt.close(fig)
    return paths


def method_style(method: str) -> MethodStyle:
    try:
        return METHOD_STYLES[method]
    except KeyError as exc:
        raise KeyError(f"missing paper style for method {method!r}") from exc
