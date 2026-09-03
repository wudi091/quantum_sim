from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.plot_online_experiments import generate_figures


def test_generate_figures_from_summary_csv() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        summary = root / "experiment_summary.csv"
        with summary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "experiment",
                    "x_axis",
                    "x_value",
                    "method",
                    "completed_requests",
                    "planning_time_seconds",
                    "source",
                ],
            )
            writer.writeheader()
            for x_value in (20, 50, 100, 200, 300):
                for method in ("gnn", "qcast", "qpass", "greedy"):
                    writer.writerow(
                        {
                            "experiment": "request_load",
                            "x_axis": "requests",
                            "x_value": x_value,
                            "method": method,
                            "completed_requests": 10.0,
                            "planning_time_seconds": 0.01,
                            "source": "aggregate",
                        }
                    )

        output = root / "figures"
        manifest = generate_figures(summary, output, formats=("svg",))

        assert len(manifest["figures"]) == 1
        figure_path = output / "request_load" / "plots" / "sweep_comparison_ieee.svg"
        assert figure_path.is_file()
        assert figure_path.stat().st_size > 0
        svg = figure_path.read_text(encoding="utf-8")
        assert svg.count("(a) Completed Requests") == 1
        assert svg.count("(b) Planning Time") == 1
        assert (output / "figure_manifest.json").is_file()


def test_new_metric_summary_keeps_completed_requests_as_supplementary_plot() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        summary = root / "experiment_summary.csv"
        with summary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "experiment",
                    "x_axis",
                    "x_value",
                    "method",
                    "completed_requests",
                    "planning_time_seconds",
                    "mean_completion_delay_slots",
                    "max_completion_delay_slots",
                    "mean_final_fidelity_loss",
                    "completion_delay_gini",
                ],
            )
            writer.writeheader()
            for x_value in (20, 50, 100, 200, 300):
                writer.writerow({
                    "experiment": "request_load",
                    "x_axis": "requests",
                    "x_value": x_value,
                    "method": "gnn",
                    "completed_requests": 10.0,
                    "planning_time_seconds": 0.01,
                    "mean_completion_delay_slots": 4.0,
                    "max_completion_delay_slots": 8.0,
                    "mean_final_fidelity_loss": 0.1,
                    "completion_delay_gini": 0.2,
                })

        output = root / "figures"
        generate_figures(summary, output, formats=("svg",))
        svg = (
            output / "request_load" / "plots" / "sweep_comparison_ieee.svg"
        ).read_text(encoding="utf-8")
        assert "(a) Mean Completion Delay" in svg
        assert "Completed Requests" in svg
