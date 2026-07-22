from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

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

    def test_overall_and_high_hop_checkpoints_are_independent_of_selection_score(self):
        config = PPOConfig(
            hidden_dim=16,
            rollout_steps=4,
            ppo_epochs=1,
            minibatch_size=4,
            checkpoint_every=10,
            evaluate_every=1,
            early_stopping_patience=1,
            curriculum=(CurriculumStage("tiny", 2, 2, 2, 1, 1),),
        )

        def evaluator(model, update):
            if update == 1:
                return {
                    "completion_rate": 0.2,
                    "pair_throughput": 0.1,
                    "high_hop_completion_rate": 0.3,
                    "selection_score": 0.5,
                }
            return {
                "completion_rate": 0.4,
                "pair_throughput": 0.2,
                "high_hop_completion_rate": 0.4,
                "selection_score": 0.4,
            }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            trainer = PPOTrainer(TinyBatchEnv(), config, output, evaluator=evaluator)
            history = trainer.train()

            self.assertEqual(len(history), 2)
            self.assertTrue(history[0]["best_evaluation"])
            self.assertTrue(history[0]["best_overall_evaluation"])
            self.assertTrue(history[0]["best_high_hop_evaluation"])
            self.assertNotIn("best_evaluation", history[1])
            self.assertTrue(history[1]["best_overall_evaluation"])
            self.assertTrue(history[1]["best_high_hop_evaluation"])
            self.assertEqual(history[1]["evaluations_without_improvement"], 1)
            self.assertTrue(history[1]["early_stopping"])

            overall = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
            high_hop = torch.load(
                output / "best_highhop.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(overall["update"], 2)
            self.assertEqual(high_hop["update"], 2)

    def test_early_stopping_warmup_resets_stagnation_counter(self):
        config = PPOConfig(
            hidden_dim=16,
            rollout_steps=4,
            ppo_epochs=1,
            minibatch_size=4,
            checkpoint_every=10,
            evaluate_every=1,
            early_stopping_patience=1,
            early_stopping_min_updates=2,
            curriculum=(CurriculumStage("tiny", 2, 2, 4, 1, 1),),
        )

        with tempfile.TemporaryDirectory() as directory:
            trainer = PPOTrainer(
                TinyBatchEnv(),
                config,
                Path(directory),
                evaluator=lambda model, update: {
                    "completion_rate": 0.25,
                    "pair_throughput": 0.125,
                },
            )
            history = trainer.train()

            # Updates 1-2 are the warmup window.  Stagnation observed there
            # must not trigger an immediate stop at the warmup boundary.
            self.assertEqual(len(history), 3)
            self.assertEqual(history[1]["evaluations_without_improvement"], 0)
            self.assertNotIn("early_stopping", history[1])
            self.assertEqual(history[2]["evaluations_without_improvement"], 1)
            self.assertTrue(history[2]["early_stopping"])

    def test_legacy_evaluator_still_selects_overall_best(self):
        config = PPOConfig(
            hidden_dim=16,
            rollout_steps=4,
            ppo_epochs=1,
            minibatch_size=4,
            checkpoint_every=10,
            evaluate_every=1,
            curriculum=(CurriculumStage("tiny", 2, 2, 1, 1, 1),),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            trainer = PPOTrainer(
                TinyBatchEnv(), config, output,
                evaluator=lambda model, update: {
                    "completion_rate": 0.25,
                    "pair_throughput": 0.125,
                },
            )
            history = trainer.train()
            self.assertTrue(history[0]["best_evaluation"])
            self.assertTrue(history[0]["best_overall_evaluation"])
            self.assertTrue((output / "best.pt").exists())
            self.assertFalse((output / "best_highhop.pt").exists())


if __name__ == "__main__":
    unittest.main()
