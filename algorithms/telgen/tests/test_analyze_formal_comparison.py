import json
from pathlib import Path
import tempfile
import unittest

from algorithms.telgen.analyze_formal_comparison import (
    analyze_cases,
    load_variant_reports,
)


def _metrics(completed: float) -> dict[str, float]:
    return {
        "completed_requests": completed,
        "mean_censored_latency_ps": 1000.0 - completed,
        "mean_decision_seconds": 0.2,
        "p95_decision_seconds": 0.3,
        "mean_planner_seconds": 0.1,
        "gnn_invalid_decision_count": 0.0,
        "schedule_violation_count": 0.0,
        "fidelity_violation_count": 0.0,
        "physical_backend_rejection_count": 0.0,
        "post_completion_validation_failure_count": 0.0,
    }


def _report(values: dict[str, tuple[float, float]]) -> dict[str, object]:
    trials = []
    for offset, seed in enumerate((100, 101)):
        methods = {
            method: {
                "metrics": _metrics(completed[offset]),
                "violations": [],
            }
            for method, completed in values.items()
        }
        trials.append({
            "seed": seed,
            "episode": {"seed": seed, "requests": [{"id": f"r{seed}"}]},
            "methods": methods,
        })
    return {
        "schema_version": 1,
        "experiment": "paired_online_gnn_milp_routing_baselines",
        "comparison_contract": {
            "paired_episode_spec": True,
            "independent_persistent_executors": True,
            "future_requests_hidden": True,
            "gnn_calls_milp_online": False,
        },
        "checkpoint_sha256": "abc123",
        "scenario": {"request_count": 2},
        "trials": trials,
    }


class FormalComparisonAnalysisTests(unittest.TestCase):
    def test_rejects_method_drift_when_profile_is_recorded(self):
        report = _report({
            "gnn": (5.0, 7.0),
            "qpass": (4.0, 7.0),
        })
        report["comparison_contract"]["comparison_profile"] = "scalable"
        with self.assertRaisesRegex(ValueError, "requires"):
            analyze_cases(
                {"case": report},
                reference="gnn",
                bootstrap_samples=100,
                randomization_samples=100,
            )

    def test_rejects_serialized_method_order_drift(self):
        report = _report({
            "gnn": (5.0, 7.0),
            "qpass": (4.0, 7.0),
            "greedy": (3.0, 6.0),
        })
        report["comparison_contract"].update({
            "comparison_profile": "scalable",
            "active_method_order": ["gnn", "greedy", "qpass"],
        })
        with self.assertRaisesRegex(ValueError, "method order"):
            analyze_cases(
                {"case": report},
                reference="gnn",
                bootstrap_samples=100,
                randomization_samples=100,
            )

    def test_formal_analysis_preserves_the_paper_method_order(self):
        report = _report({
            "greedy": (3.0, 6.0),
            "qpass": (4.0, 7.0),
            "milp": (6.0, 8.0),
            "gnn": (5.0, 7.0),
        })
        report["comparison_contract"].update({
            "comparison_profile": "formal",
            "active_method_order": ["gnn", "milp", "qpass", "greedy"],
        })
        analysis = analyze_cases(
            {"case": report},
            reference="gnn",
            bootstrap_samples=100,
            randomization_samples=100,
        )
        self.assertEqual(
            list(analysis["cases"]["case"]["method_means"]),
            ["gnn", "milp", "qpass", "greedy"],
        )
        self.assertEqual(
            analysis["overall"]["common_methods"],
            ["gnn", "milp", "qpass", "greedy"],
        )

    def test_analyzes_all_methods_against_one_reference(self):
        analysis = analyze_cases(
            {"case": _report({
                "gnn": (5.0, 7.0),
                "qpass": (4.0, 7.0),
                "greedy": (3.0, 6.0),
            })},
            reference="gnn",
            bootstrap_samples=100,
            randomization_samples=100,
            random_seed=1,
        )

        qpass = analysis["cases"]["case"]["comparisons"]["qpass"][
            "metrics"
        ]["completed_requests"]
        self.assertEqual(qpass["reference_advantage_mean"], 0.5)
        self.assertEqual(qpass["reference_wins"], 1)
        self.assertEqual(qpass["ties"], 1)
        self.assertEqual(qpass["baseline_wins"], 0)
        self.assertTrue(analysis["overall"]["valid"])

    def test_hard_gate_failure_invalidates_case(self):
        report = _report({"gnn": (5.0, 7.0), "qpass": (4.0, 7.0)})
        report["trials"][0]["methods"]["gnn"]["metrics"][
            "schedule_violation_count"
        ] = 1.0

        analysis = analyze_cases(
            {"case": report},
            reference="gnn",
            bootstrap_samples=100,
            randomization_samples=100,
        )

        self.assertFalse(analysis["cases"]["case"]["valid"])
        self.assertFalse(analysis["overall"]["valid"])

    def test_variant_loader_rejects_episode_mismatch(self):
        adaptive = _report({"gnn": (5.0, 7.0)})
        fixed = _report({"gnn": (4.0, 6.0)})
        fixed["trials"][0]["episode"]["requests"] = [{"id": "different"}]
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            adaptive_path = directory / "adaptive.json"
            fixed_path = directory / "fixed.json"
            adaptive_path.write_text(json.dumps(adaptive), encoding="utf-8")
            fixed_path.write_text(json.dumps(fixed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "EpisodeSpec differs"):
                load_variant_reports(
                    [("adaptive", adaptive_path), ("fixed", fixed_path)],
                    reference="adaptive",
                    case_name="B5",
                )


if __name__ == "__main__":
    unittest.main()
