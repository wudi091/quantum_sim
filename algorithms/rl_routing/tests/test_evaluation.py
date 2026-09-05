import json
import tempfile
import unittest
from pathlib import Path

import torch

from algorithms.rl_routing.evaluation import (
    BaselineDefinition,
    run_paired_evaluation,
    save_evaluation_records,
)
from algorithms.rl_routing.policy import ARCQPolicy
from algorithms.routing_core.execution import OnlineExecutionConfig
from qnet_core.scenario import ScenarioConfig
from qnet_core.spec import PhysicalConfig


class PairedEvaluationTests(unittest.TestCase):
    def test_methods_share_episode_and_results_are_recorded_without_plotting(self):
        torch.manual_seed(8700)
        policy = ARCQPolicy(hidden_dim=16, message_passing_layers=1)
        scenario = ScenarioConfig(
            request_count=2,
            min_hops=2,
            max_hops=2,
            ttl=5,
            horizon=5,
            topology_mode="parallel_corridors",
            parallel_corridors=2,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                detector_efficiency=1.0,
                bsm_success_probability=1.0,
                quantum_distance_m=1.0,
                memory_capacity=1,
                node_memory_capacity=2,
                max_width=1,
            ),
        )
        environment = OnlineExecutionConfig(
            decision_interval=2,
            path_candidate_count=2,
            construction_kinds=("left_deep", "balanced"),
            purification_kinds=("none",),
        )
        records = run_paired_evaluation(
            policy,
            scenario_name="test",
            scenario_config=scenario,
            environment_config=environment,
            episode_seeds=(8701,),
            topology_seed=1,
            baselines=(
                BaselineDefinition("Greedy", "greedy", 1, "balanced"),
            ),
        )
        self.assertEqual({record.method for record in records}, {"ARC-Q", "Greedy"})
        self.assertEqual({record.episode_seed for record in records}, {8701})
        self.assertTrue(all(
            record.metrics["schedule_violation_count"] == 0.0
            for record in records
        ))

        with tempfile.TemporaryDirectory(dir=".") as directory:
            json_path, csv_path = save_evaluation_records(
                records,
                Path(directory) / "evaluation",
                metadata={"purpose": "unit-test"},
            )
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["records"]), 2)
            self.assertTrue(csv_path.exists())
            self.assertGreater(csv_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
