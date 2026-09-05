import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from algorithms.rl_routing.checkpoint import save_arcq_checkpoint
from algorithms.rl_routing.policy import ARCQPolicy
from experiments.run_arcq_experiments import (
    load_experiment_protocol,
    protocol_summary,
    run_experiments,
    scenario_for_point,
)


class ARCQExperimentProtocolTests(unittest.TestCase):
    def test_repository_protocol_has_five_complete_sweeps(self):
        protocol = load_experiment_protocol(
            "configs/arcq_experiments.yaml"
        )
        summary = protocol_summary(protocol)
        self.assertEqual(summary["suite_count"], 5)
        self.assertEqual(summary["total_paired_instances"], 750)
        self.assertEqual(summary["total_method_episodes"], 4500)
        self.assertTrue(all(
            len(suite.points) >= 5 for suite in protocol.suites
        ))
        memory_point = next(
            point
            for suite in protocol.suites
            if suite.suite_id == "memory_capacity"
            for point in suite.points
            if point.point_id == "memory_2"
        )
        scenario = scenario_for_point(protocol.base_scenario, memory_point)
        self.assertEqual(scenario.physical.node_memory_capacity, 2)
        self.assertEqual(
            scenario.physical.generation_probability,
            protocol.base_scenario.physical.generation_probability,
        )

    def test_runner_records_and_resumes_complete_paired_instances(self):
        torch.manual_seed(8800)
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            output = root / "raw.json"
            config_path = root / "protocol.yaml"
            policy = ARCQPolicy(hidden_dim=16, message_passing_layers=1)
            points = [
                {
                    "id": f"point_{index}",
                    "value": index,
                    "topology_seeds": [1],
                    "scenario": {},
                }
                for index in range(5)
            ]
            config = {
                "schema_version": 1,
                "checkpoint_path": str(checkpoint),
                "output_path": str(output),
                "replication": {
                    "episode_seed_start": 8801,
                    "episodes_per_topology": 1,
                },
                "environment": {
                    "decision_interval": 2,
                    "path_candidate_count": 1,
                    "construction_kinds": ["balanced"],
                    "swap_tree_count": None,
                    "purification_kinds": ["none"],
                },
                "base_scenario": {
                    "request_count": 1,
                    "min_hops": 2,
                    "max_hops": 2,
                    "ttl": 4,
                    "horizon": 4,
                    "topology_mode": "parallel_corridors",
                    "parallel_corridors": 2,
                    "endpoint_mode": "distance_stratified",
                    "demand_pairs": 1,
                    "physical": {
                        "generation_probability": 1.0,
                        "swap_probability": 1.0,
                        "memory_capacity": 1,
                        "node_memory_capacity": 2,
                        "max_width": 1,
                        "quantum_distance_m": 1.0,
                        "detector_efficiency": 1.0,
                        "bsm_success_probability": 1.0,
                    },
                },
                "baselines": [{
                    "name": "Greedy",
                    "algorithm": "greedy",
                    "path_candidate_count": 1,
                    "construction_kind": "balanced",
                }],
                "suites": [{
                    "id": "test_suite",
                    "x_label": "test_value",
                    "points": points,
                }],
            }
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )
            protocol = load_experiment_protocol(config_path)
            save_arcq_checkpoint(
                checkpoint,
                policy,
                hidden_dim=16,
                message_passing_layers=1,
                training_state={
                    "update": 1,
                    "episodes_completed": 2,
                    "fixed_training_topology_seed": 99,
                    "config": {
                        "environment": asdict(protocol.environment),
                        "run": {
                            "episode_count": 2,
                            "random_seed": 100,
                            "validation_seed": 200,
                            "validation_episode_count": 1,
                        },
                    },
                    "best_validation_update": 1,
                    "best_validation_latency_slots": 2.0,
                    "model_selection_metric": "mean_censored_latency_slots",
                    "selection_finalized": True,
                    "training_completed_episodes": 2,
                    "final_update": 1,
                    "final_best_validation_update": 1,
                    "final_best_validation_latency_slots": 2.0,
                },
            )
            json_path, csv_path = run_experiments(
                protocol,
                max_instances=1,
                allow_dirty=True,
            )
            first = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(len(first["records"]), 2)
            self.assertTrue(csv_path.exists())
            run_experiments(
                protocol,
                max_instances=1,
                allow_dirty=True,
            )
            second = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(len(second["records"]), 4)
            self.assertTrue(all(
                record["metrics"]["schedule_violation_count"] == 0.0
                for record in second["records"]
            ))


if __name__ == "__main__":
    unittest.main()
