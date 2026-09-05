import tempfile
import unittest
from pathlib import Path

import torch

from algorithms.rl_routing.checkpoint import (
    load_arcq_checkpoint,
    save_arcq_checkpoint,
)
from algorithms.rl_routing.policy import ARCQPolicy
from algorithms.rl_routing.train import TrainingRunConfig, load_training_config


class ARCQCheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_preserves_parameters_and_metadata(self):
        torch.manual_seed(8600)
        policy = ARCQPolicy(hidden_dim=16, message_passing_layers=1)
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = Path(directory) / "model.pt"
            save_arcq_checkpoint(
                path,
                policy,
                hidden_dim=16,
                message_passing_layers=1,
                training_state={"episodes_completed": 7},
            )
            restored, metadata = load_arcq_checkpoint(path)
        self.assertEqual(metadata["method"], "ARC-Q")
        self.assertEqual(
            metadata["training_state"]["episodes_completed"],
            7,
        )
        for expected, actual in zip(
            policy.parameters(), restored.parameters(), strict=True
        ):
            self.assertTrue(torch.equal(expected, actual))

    def test_repository_configs_are_valid_and_keep_training_topology_fixed(self):
        smoke = load_training_config("configs/arcq_smoke.yaml")
        training = load_training_config("configs/arcq_train.yaml")
        self.assertEqual(smoke.run.episode_count, 2)
        self.assertGreater(training.run.episode_count, 2)
        self.assertIsInstance(training.run.topology_seed, int)
        self.assertEqual(training.ppo.gamma, 1.0)
        training_seeds = set(range(
            training.run.random_seed + 1,
            training.run.random_seed + training.run.episode_count + 1,
        ))
        validation_seeds = set(range(
            training.run.validation_seed,
            training.run.validation_seed
            + training.run.validation_episode_count,
        ))
        self.assertTrue(training_seeds.isdisjoint(validation_seeds))

    def test_training_and_validation_seeds_must_be_disjoint(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            TrainingRunConfig(
                episode_count=10,
                random_seed=100,
                validation_episode_count=2,
                validation_seed=105,
            )

    def test_incompatible_checkpoint_schema_fails_before_state_loading(self):
        torch.manual_seed(8601)
        policy = ARCQPolicy(hidden_dim=16, message_passing_layers=1)
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = Path(directory) / "model.pt"
            save_arcq_checkpoint(
                path,
                policy,
                hidden_dim=16,
                message_passing_layers=1,
                training_state={},
            )
            payload = torch.load(path, weights_only=False)
            payload["schema_version"] = 1
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "schema"):
                load_arcq_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
