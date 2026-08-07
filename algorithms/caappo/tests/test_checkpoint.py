import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from algorithms.caappo.checkpoint import (
    CheckpointCompatibilityError,
    load_caappo_checkpoint,
)
from algorithms.caappo.experiment import (
    CAAPPOVariant,
    ConstructionExperimentConfig,
    evaluate_checkpoint,
    train_variant_checkpoint,
)
from qnet_core.scenario import ScenarioConfig
from qnet_core.spec import PhysicalConfig


class CAAPPOCheckpointTests(unittest.TestCase):
    @staticmethod
    def _config(episodes: int) -> ConstructionExperimentConfig:
        variant = CAAPPOVariant(
            "caappo",
            candidate_count=1,
            construction_kinds=("left_deep",),
        )
        return ConstructionExperimentConfig(
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
            validation_seeds=(31,),
            training_episodes=episodes,
            validation_interval=1,
            candidate_count=1,
            include_nominal_oracle=False,
            variants=(variant,),
        )

    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(2)
            config = replace(config, validation_interval=2)
            variant = config.variants[0]
            uninterrupted_path = root / "uninterrupted.pt"
            resumed_path = root / "resumed.pt"

            train_variant_checkpoint(
                config, variant, 1, uninterrupted_path
            )
            train_variant_checkpoint(
                replace(config, training_episodes=1),
                variant,
                1,
                resumed_path,
            )
            train_variant_checkpoint(
                config,
                variant,
                1,
                resumed_path,
                resume=True,
            )

            uninterrupted = load_caappo_checkpoint(uninterrupted_path)
            resumed = load_caappo_checkpoint(resumed_path)
            self.assertEqual(uninterrupted.completed_episodes, 2)
            self.assertEqual(resumed.completed_episodes, 2)
            for name, expected in uninterrupted.policy.state_dict().items():
                self.assertTrue(torch.equal(
                    expected,
                    resumed.policy.state_dict()[name],
                ), name)
            self.assertEqual(
                uninterrupted.policy.lambda_risk,
                resumed.policy.lambda_risk,
            )
            self.assertEqual(
                uninterrupted.best_validation,
                resumed.best_validation,
            )
            self.assertIsNotNone(uninterrupted.best_policy_state_dict)
            self.assertIsNotNone(resumed.best_policy_state_dict)
            for name, expected in uninterrupted.best_policy_state_dict.items():
                self.assertTrue(torch.equal(
                    expected,
                    resumed.best_policy_state_dict[name],
                ), name)
            self.assertEqual(
                [
                    row["episode_seed"]
                    for row in uninterrupted.history
                    if row["event"] == "training_episode"
                ],
                [
                    row["episode_seed"]
                    for row in resumed.history
                    if row["event"] == "training_episode"
                ],
            )

    def test_frozen_evaluation_and_strict_config_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "caappo.pt"
            config = self._config(1)
            train_variant_checkpoint(
                config, config.variants[0], 1, checkpoint
            )
            torch.manual_seed(9876)
            rng_before_load = torch.get_rng_state().clone()
            before = load_caappo_checkpoint(checkpoint)
            self.assertTrue(torch.equal(rng_before_load, torch.get_rng_state()))
            best_loaded = load_caappo_checkpoint(checkpoint, use_best=True)
            self.assertIsNotNone(best_loaded.best_optimizer_state_dict)
            self.assertEqual(
                best_loaded.policy.lambda_risk,
                best_loaded.best_lambda_risk,
            )
            rows, run = evaluate_checkpoint(checkpoint, (41,))
            after = load_caappo_checkpoint(checkpoint)

            self.assertEqual(len(rows), 1)
            self.assertEqual(run["selected_state"], "best")
            self.assertEqual(run["evaluation_seeds"], (41,))
            for name, expected in before.policy.state_dict().items():
                self.assertTrue(torch.equal(
                    expected,
                    after.policy.state_dict()[name],
                ), name)
            with self.assertRaises(CheckpointCompatibilityError):
                load_caappo_checkpoint(
                    checkpoint,
                    expected_training_metadata={"training_seed": 999},
                )


if __name__ == "__main__":
    unittest.main()
