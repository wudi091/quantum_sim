import json
import tempfile
import unittest
from pathlib import Path

from experiments.plot_arcq_experiments import (
    load_result_payload,
    plot_summary,
    summarize_results,
    write_source_data,
)


def _payload() -> dict[str, object]:
    points = [
        {
            "id": f"point_{point}",
            "value": point,
            "topology_seeds": [11, 12],
            "scenario": {},
        }
        for point in range(1, 6)
    ]
    payload: dict[str, object] = {
        "schema_version": 1,
        "method_under_test": "ARC-Q",
        "protocol_fingerprint": "fingerprint",
        "checkpoint_sha256": "checkpoint",
        "protocol": {
            "replication": {
                "episode_seed_start": 21,
                "episodes_per_topology": 2,
            },
            "base_scenario": {
                "physical": {"slot_duration_ps": 100.0},
            },
            "baselines": [{"name": "Greedy"}],
            "suites": [{
                "id": "offered_load",
                "x_label": "requests_per_decision",
                "points": points,
            }],
        },
        "records": [],
    }
    records = payload["records"]
    for point in points:
        for topology_seed in point["topology_seeds"]:
            for episode_seed in (21, 22):
                for method in ("ARC-Q", "Greedy"):
                    arcq = method == "ARC-Q"
                    records.append({
                        "suite": "offered_load",
                        "point_id": point["id"],
                        "point_value": point["value"],
                        "topology_seed": topology_seed,
                        "episode_seed": episode_seed,
                        "method": method,
                        "metrics": {
                            "mean_censored_latency_ps": (
                                500.0 if arcq else 700.0
                            ),
                            "completion_rate": 0.9 if arcq else 0.8,
                            "completion_delay_gini": 0.1 if arcq else 0.2,
                            "mean_planner_seconds": (
                                0.002 if arcq else 0.001
                            ),
                            "schedule_violation_count": 0.0,
                            "physical_backend_rejection_count": 0.0,
                            **({"reward_identity_error": 0.0} if arcq else {}),
                        },
                    })
    return payload


class ARCQPlotTests(unittest.TestCase):
    def test_summary_uses_paired_instances_and_metric_units(self):
        summary = summarize_results(_payload())
        self.assertEqual(summary["validity"]["expected_record_count"], 40)
        latency = next(
            row
            for row in summary["summary_rows"]
            if row["point_id"] == "point_1"
            and row["method"] == "ARC-Q"
            and row["metric"] == "mean_censored_latency"
        )
        self.assertEqual(latency["sample_count"], 4)
        self.assertAlmostEqual(latency["mean"], 5.0)
        fairness = next(
            row
            for row in summary["summary_rows"]
            if row["point_id"] == "point_1"
            and row["method"] == "ARC-Q"
            and row["metric"] == "delay_fairness"
        )
        self.assertAlmostEqual(fairness["mean"], 0.9)
        improvement = next(
            row
            for row in summary["paired_rows"]
            if row["point_id"] == "point_1"
            and row["baseline"] == "Greedy"
            and row["metric"] == "mean_censored_latency"
        )
        self.assertAlmostEqual(improvement["mean_arcq_improvement"], 2.0)

    def test_summary_rejects_incomplete_or_invalid_evidence(self):
        incomplete = _payload()
        incomplete["records"].pop()
        with self.assertRaisesRegex(ValueError, "missing paired result"):
            summarize_results(incomplete)

        invalid = _payload()
        invalid["records"][0]["metrics"]["schedule_violation_count"] = 1.0
        with self.assertRaisesRegex(ValueError, "schedule violation"):
            summarize_results(invalid)

    def test_plotter_only_reads_results_and_emits_nonempty_artifacts(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            raw_path = root / "raw_results.json"
            raw_path.write_text(json.dumps(_payload()), encoding="utf-8")
            payload = load_result_payload(raw_path)
            summary = summarize_results(payload)
            source_paths = write_source_data(summary, root / "figures")
            figure_paths = plot_summary(summary, root / "figures")
            self.assertEqual(len(source_paths), 3)
            self.assertEqual(len(figure_paths), 8)
            self.assertTrue(all(
                path.is_file() and path.stat().st_size > 0
                for path in (*source_paths, *figure_paths)
            ))


if __name__ == "__main__":
    unittest.main()
