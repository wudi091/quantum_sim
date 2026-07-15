from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from routing_rl.config import CurriculumStage, PPOConfig
from routing_rl.trainer import PPOTrainer


class TinyBatchEnv:
    def __init__(self):
        self.selected = False
        self.reset_seeds = []

    def set_curriculum(self, stage):
        self.stage = stage

    def _observation(self):
        return {
            "global_features": np.asarray([0.0, float(self.selected)], dtype=np.float32),
            "request_features": np.asarray([[1.0, 0.5]], dtype=np.float32),
            "request_mask": np.asarray([True]),
            "candidate_features": np.asarray([[1.0, 0.2, 0.1]], dtype=np.float32),
            "action_mask": np.asarray([not self.selected, True]),
        }

    def reset(self, seed=None, options=None):
        self.reset_seeds.append(seed)
        self.selected = False
        return self._observation(), {}

    def step(self, action):
        if action == 0:
            if self.selected:
                raise AssertionError("masked candidate was selected twice")
            self.selected = True
            return self._observation(), 0.0, False, False, {"duration": 0}
        if action == 1:
            return self._observation(), 1.0, True, False, {
                "duration": 1,
                "completed": 1,
                "sum_delay": 1.0,
            }
        raise AssertionError(f"invalid action {action}")


class TrainerSmokeTest(unittest.TestCase):
    def test_checkpoint_and_history_are_written(self):
        config = PPOConfig(
            hidden_dim=16,
            rollout_steps=8,
            ppo_epochs=1,
            minibatch_size=4,
            checkpoint_every=1,
            curriculum=(CurriculumStage("tiny", 2, 2, 1, 1, 1),),
        )
        with tempfile.TemporaryDirectory() as directory:
            env = TinyBatchEnv()
            trainer = PPOTrainer(env, config, Path(directory))
            history = trainer.train()
            self.assertEqual(len(history), 1)
            self.assertTrue((Path(directory) / "history.json").exists())
            self.assertTrue((Path(directory) / "checkpoint.pt").exists())
            self.assertTrue((Path(directory) / "checkpoint_000001.pt").exists())
            training_seeds = [seed for seed in env.reset_seeds[1:] if seed is not None]
            self.assertEqual(len(training_seeds), len(set(training_seeds)))


if __name__ == "__main__":
    unittest.main()
