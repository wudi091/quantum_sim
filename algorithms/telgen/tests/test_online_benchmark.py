import unittest

from algorithms.telgen.analyze_online_benchmark import analyze_online_payloads


def payload(rows):
    trials = []
    for seed, telgen_completed, qcast_completed, telgen_latency, qcast_latency in rows:
        trials.append({
            "seed": seed,
            "telgen": {"violations": [], "metrics": {
                "completed_requests": telgen_completed,
                "mean_completion_delay_ps": telgen_latency,
                "mean_censored_latency_ps": telgen_latency,
                "max_completion_delay_ps": telgen_latency + 20.0,
                "mean_final_fidelity_loss": 0.1,
                "completion_delay_gini": 0.2,
                "mean_planner_seconds": 1.0,
                "mean_decision_seconds": 1.0,
                "schedule_violation_count": 0.0,
                "fidelity_violation_count": 0.0,
                "post_completion_validation_failure_count": 0.0,
            }},
            "qcast": {"violations": [], "metrics": {
                "completed_requests": qcast_completed,
                "mean_completion_delay_ps": qcast_latency,
                "mean_censored_latency_ps": qcast_latency,
                "max_completion_delay_ps": qcast_latency + 20.0,
                "mean_final_fidelity_loss": 0.1,
                "completion_delay_gini": 0.2,
                "mean_planner_seconds": 0.1,
                "mean_decision_seconds": 0.1,
                "schedule_violation_count": 0.0,
                "fidelity_violation_count": 0.0,
                "post_completion_validation_failure_count": 0.0,
            }},
        })
    return {
        "schema_version": 1,
        "comparison_contract": {
            "paired_episode_spec": True,
            "independent_persistent_executors": True,
            "common_runtime_metric": "mean_decision_seconds",
            "qcast_baseline": "width_one_ext_fixed_construction",
            "qcast_uses_telgen_lp_or_search_decoder": False,
        },
        "scenario": {"name": "test"},
        "trials": trials,
    }


class OnlineBenchmarkAnalysisTests(unittest.TestCase):
    def test_average_completion_delay_is_the_quality_metric(self):
        report = analyze_online_payloads(
            {"case": payload([
                (seed, 3, 2, 200.0, 100.0)
                for seed in range(1, 9)
            ])},
            bootstrap_samples=200,
            randomization_samples=2_000,
            random_seed=7,
        )
        self.assertEqual(
            report["overall"]["quality_verdict"],
            "qcast_better_latency",
        )
        self.assertEqual(report["overall"]["runtime_verdict"], "qcast_faster")

    def test_lower_average_completion_delay_is_better(self):
        report = analyze_online_payloads(
            {"case": payload([
                (seed, 2, 2, 80.0, 100.0)
                for seed in range(1, 9)
            ])},
            bootstrap_samples=200,
            randomization_samples=2_000,
            random_seed=11,
        )
        self.assertEqual(
            report["overall"]["quality_verdict"],
            "telgen_better_latency",
        )

    def test_latency_remains_primary_when_throughput_differs(self):
        report = analyze_online_payloads(
            {"case": payload([
                (seed, 3 if seed % 2 else 2, 2 if seed % 2 else 3, 80.0, 100.0)
                for seed in range(1, 9)
            ])},
            bootstrap_samples=200,
            randomization_samples=2_000,
            random_seed=13,
        )
        self.assertEqual(
            report["overall"]["quality_verdict"],
            "telgen_better_latency",
        )

    def test_hard_gate_failure_invalidates_quality_verdict(self):
        case = payload([(1, 3, 2, 80.0, 100.0)])
        case["trials"][0]["telgen"]["violations"].append({
            "code": "launch_rejected",
        })
        report = analyze_online_payloads(
            {"case": case},
            bootstrap_samples=100,
            randomization_samples=100,
            random_seed=17,
        )
        self.assertFalse(report["overall"]["valid"])
        self.assertEqual(
            report["overall"]["quality_verdict"],
            "invalid_hard_gate_failure",
        )

    def test_nominal_completion_overrun_is_reported_but_not_invalid(self):
        case = payload([
            (seed, 3, 2, 80.0, 100.0)
            for seed in range(1, 9)
        ])
        case["trials"][0]["telgen"]["violations"].append({
            "code": "slot_completion_overrun",
        })
        report = analyze_online_payloads(
            {"case": case},
            bootstrap_samples=200,
            randomization_samples=2_000,
            random_seed=19,
        )
        self.assertTrue(report["overall"]["valid"])
        self.assertEqual(
            report["overall"]["hard_gates"][
                "telgen_nominal_completion_overrun_count"
            ],
            1.0,
        )
        self.assertEqual(
            report["overall"]["quality_verdict"],
            "telgen_better_latency",
        )


if __name__ == "__main__":
    unittest.main()
