import unittest
from collections import Counter
from pathlib import Path

from algorithms.caappo.experiment import (
    CAAPPOVariant,
    ConstructionExperimentConfig,
    _aggregate,
    _best_validation,
    _parse_checkpoint_specs,
    _run_baselines,
    _training_episode_seed,
    run_experiment,
)
from qnet_core.scenario import ScenarioConfig
from qnet_core.spec import PhysicalConfig


class ConstructionExperimentTests(unittest.TestCase):
    def test_checkpoint_specs_reject_exact_duplicates(self):
        self.assertEqual(
            _parse_checkpoint_specs(["main=results/main.pt"]),
            (("main", Path("results/main.pt")),),
        )
        self.assertEqual(
            _parse_checkpoint_specs(["main=a.pt", "main=b.pt"]),
            (("main", Path("a.pt")), ("main", Path("b.pt"))),
        )
        with self.assertRaises(ValueError):
            _parse_checkpoint_specs(["main=a.pt", "main=a.pt"])
        with self.assertRaises(ValueError):
            _parse_checkpoint_specs(["malformed"])
    def test_training_episode_seed_protocol_is_injective_and_held_out(self):
        config = ConstructionExperimentConfig(
            scenario=ScenarioConfig(),
            evaluation_seeds=(101,),
            training_seeds=(1, 2),
            validation_seeds=(51,),
            training_episodes=2,
        )
        derived = {
            _training_episode_seed(config, training_seed, episode_index)
            for training_seed in config.training_seeds
            for episode_index in range(config.training_episodes)
        }
        self.assertEqual(len(derived), 4)
        self.assertFalse(derived.intersection({1, 2, 51, 101}))

    def test_validation_selection_respects_risk_limit(self):
        current = {
            "mean_objective": 0.50,
            "mean_risk_count": 0.0,
            "selection_eligible": True,
        }
        infeasible = {
            "mean_objective": 0.90,
            "mean_risk_count": 1.0,
            "selection_eligible": True,
        }
        self.assertFalse(_best_validation(current, infeasible, 0.0))

    def test_duplicate_seed_groups_are_rejected(self):
        with self.assertRaises(ValueError):
            ConstructionExperimentConfig(
                scenario=ScenarioConfig(),
                evaluation_seeds=(41, 41),
                training_seeds=(1,),
                validation_seeds=(31,),
            )

    def test_aggregate_averages_training_replicas_before_ci(self):
        rows = [
            {
                "method": "torch_caappo",
                "variant": "caappo",
                "seed": 101,
                "training_seed": 1,
                "completed_requests": 1.0,
            },
            {
                "method": "torch_caappo",
                "variant": "caappo",
                "seed": 101,
                "training_seed": 2,
                "completed_requests": 3.0,
            },
            {
                "method": "torch_caappo",
                "variant": "caappo",
                "seed": 102,
                "training_seed": 1,
                "completed_requests": 2.0,
            },
            {
                "method": "torch_caappo",
                "variant": "caappo",
                "seed": 102,
                "training_seed": 2,
                "completed_requests": 4.0,
            },
        ]
        aggregate = _aggregate(rows)
        metric = next(row for row in aggregate if row["metric"] == "completed_requests")
        self.assertEqual(metric["n"], 2)
        self.assertEqual(metric["mean"], 2.5)

    def test_seeded_harness_reports_rows_and_confidence_intervals(self):
        config = ConstructionExperimentConfig(
            scenario=ScenarioConfig(
                request_count=1,
                min_hops=1,
                max_hops=1,
                ttl=10,
                horizon=10,
                topology_nodes=4,
                physical=PhysicalConfig(
                    generation_probability=1.0,
                    swap_probability=1.0,
                    memory_capacity=1,
                    node_memory_capacity=2,
                    quantum_distance_m=1.0,
                ),
            ),
            evaluation_seeds=(41,),
            training_seeds=(1,),
            training_episodes=0,
            candidate_count=1,
            variants=(CAAPPOVariant("caappo", candidate_count=1),),
        )
        result = run_experiment(config)
        self.assertEqual(len(result["rows"]), 5)
        self.assertTrue(result["aggregate"])
        self.assertIn("paired_differences", result)
        self.assertEqual(
            {row["reference"] for row in result["paired_differences"]},
            {"balanced", "memory_aware", "shortest_left_deep"},
        )
        self.assertIn("catalogue_coverage", result)
        self.assertTrue(all(
            "ci95_low" in row and "ci95_high" in row
            for row in result["aggregate"]
        ))
        primary = {
            "completed_requests",
            "completion_rate",
            "censored_flow_time_ps",
            "mean_censored_latency_ps",
            "p95_completion_latency_ps",
            "risk_count",
            "makespan_ps",
        }
        self.assertTrue(all(
            primary.issubset(row)
            for row in result["rows"]
            if row["method"] != "nominal_oracle"
        ))
        self.assertEqual(result["manifest"]["physical_backend"], "SeQUeNCe")
        self.assertTrue(result["catalogue_coverage"]["route_coverage_exact"])
        exact_oracle = next(
            row for row in result["rows"]
            if row["method"] == "nominal_oracle"
            and row["variant"] == "exact_nominal"
        )
        self.assertIn("full_path_oracle_gap", exact_oracle)
        self.assertGreaterEqual(exact_oracle["full_path_oracle_gap"], 0.0)

    def test_parallel_corridor_baseline_harness_includes_split_policies(self):
        config = ConstructionExperimentConfig(
            scenario=ScenarioConfig(
                request_count=2,
                min_hops=3,
                max_hops=3,
                ttl=8,
                horizon=8,
                topology_mode="parallel_corridors",
                parallel_corridors=2,
                batch_mode=True,
                physical=PhysicalConfig(
                    generation_probability=1.0,
                    swap_probability=1.0,
                    memory_capacity=1,
                    node_memory_capacity=8,
                    quantum_distance_m=1.0,
                ),
            ),
            evaluation_seeds=(41, 42),
            training_seeds=(1,),
            validation_seeds=(31,),
            training_episodes=0,
            candidate_count=2,
        )

        rows = _run_baselines(config)

        variants = Counter(str(row["variant"]) for row in rows)
        self.assertEqual(
            variants,
            Counter({
                "shortest_left_deep": 2,
                "balanced": 2,
                "memory_aware": 2,
                "split_left_deep": 2,
                "split_balanced": 2,
            }),
        )
        self.assertTrue(all(row["method"] == "fixed_baseline" for row in rows))


if __name__ == "__main__":
    unittest.main()
