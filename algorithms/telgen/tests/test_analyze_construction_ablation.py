import json
from pathlib import Path
import tempfile
import unittest

from algorithms.telgen.analyze_construction_ablation import analyze_pairs


def _report(*, adaptive: bool, completed: tuple[int, int]) -> dict[str, object]:
    policy = (
        "adaptive_swap_tree_selection"
        if adaptive
        else "fixed_swap_tree_2"
    )
    fixed_index = None if adaptive else 2
    trials = []
    for offset, value in enumerate(completed):
        seed = 100 + offset
        trials.append({
            "seed": seed,
            "episode": {"seed": seed, "requests": [{"id": f"r{seed}"}]},
            "methods": {
                "gnn": {
                    "metrics": {
                        "completed_requests": float(value),
                        "mean_censored_latency_ps": float(1000 - value),
                        "mean_decision_seconds": 0.1 + offset * 0.01,
                        "gnn_invalid_decision_count": 0.0,
                        "schedule_violation_count": 0.0,
                        "fidelity_violation_count": 0.0,
                        "physical_backend_rejection_count": 0.0,
                        "post_completion_validation_failure_count": 0.0,
                    },
                    "violations": [],
                },
            },
        })
    return {
        "schema_version": 1,
        "experiment": "paired_online_gnn_milp_qcast",
        "comparison_contract": {
            "paired_episode_spec": True,
            "independent_persistent_executors": True,
            "future_requests_hidden": True,
            "gnn_calls_milp_online": False,
            "qcast_included": False,
            "gnn_construction_policy": policy,
        },
        "configuration": {
            "checkpoint": "model.pt",
            "output": "adaptive" if adaptive else "fixed",
            "skip_milp": True,
            "skip_qcast": True,
            "fixed_swap_tree_index": fixed_index,
            "construction_plans": 5,
            "requests": 2,
        },
        "checkpoint_sha256": "abc123",
        "scenario": {"request_count": 2},
        "gnn_config": {
            "construction_kinds": [] if adaptive else ["swap_tree_2"],
            "swap_tree_count": 5 if adaptive else None,
        },
        "milp_config": None,
        "qcast_config": None,
        "trials": trials,
    }


class ConstructionAblationAnalysisTests(unittest.TestCase):
    def _write_pair(
        self,
        directory: Path,
        adaptive: dict[str, object],
        fixed: dict[str, object],
    ) -> tuple[Path, Path]:
        adaptive_path = directory / "adaptive.json"
        fixed_path = directory / "fixed.json"
        adaptive_path.write_text(json.dumps(adaptive), encoding="utf-8")
        fixed_path.write_text(json.dumps(fixed), encoding="utf-8")
        return adaptive_path, fixed_path

    def test_analyzes_exactly_paired_reports(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            adaptive_path, fixed_path = self._write_pair(
                directory,
                _report(adaptive=True, completed=(5, 7)),
                _report(adaptive=False, completed=(4, 7)),
            )
            analysis = analyze_pairs(
                [("case", adaptive_path, fixed_path)],
                bootstrap_samples=100,
                randomization_samples=100,
                random_seed=1,
            )

        completed = analysis["overall"]["metrics"]["completed_requests"]
        self.assertEqual(analysis["paired_trial_count"], 2)
        self.assertEqual(completed["adaptive_advantage_mean"], 0.5)
        self.assertEqual(completed["adaptive_wins"], 1)
        self.assertEqual(completed["ties"], 1)
        self.assertEqual(completed["fixed_wins"], 0)
        self.assertTrue(analysis["overall"]["valid"])

    def test_rejects_an_episode_mismatch(self):
        adaptive = _report(adaptive=True, completed=(5, 7))
        fixed = _report(adaptive=False, completed=(4, 7))
        fixed["trials"][0]["episode"]["requests"] = [{"id": "different"}]
        with tempfile.TemporaryDirectory() as raw_directory:
            paths = self._write_pair(Path(raw_directory), adaptive, fixed)
            with self.assertRaisesRegex(ValueError, "EpisodeSpec differs"):
                analyze_pairs(
                    [("case", *paths)],
                    bootstrap_samples=100,
                    randomization_samples=100,
                )

    def test_hard_gate_failure_invalidates_the_result(self):
        adaptive = _report(adaptive=True, completed=(5, 7))
        fixed = _report(adaptive=False, completed=(4, 7))
        adaptive["trials"][0]["methods"]["gnn"]["metrics"][
            "gnn_invalid_decision_count"
        ] = 1.0
        with tempfile.TemporaryDirectory() as raw_directory:
            paths = self._write_pair(Path(raw_directory), adaptive, fixed)
            analysis = analyze_pairs(
                [("case", *paths)],
                bootstrap_samples=100,
                randomization_samples=100,
            )

        self.assertFalse(analysis["overall"]["valid"])
        self.assertEqual(
            analysis["overall"]["throughput_verdict"],
            "invalid_hard_gate_failure",
        )


if __name__ == "__main__":
    unittest.main()
