from __future__ import annotations

import unittest

import torch

from batchswap_rl.baselines import RandomValidPolicy, run_policy
from batchswap_rl.env import make_env
from batchswap_rl.evaluate import (
    LearnedController,
    NamedBaselineController,
    paired_evaluation,
)
from batchswap_rl.model import DynamicPlanActorCritic


class EvaluationTest(unittest.TestCase):
    def test_random_valid_policy_completes(self):
        env = make_env(stage=0, seed=9)
        result = run_policy(env, RandomValidPolicy(seed=9), seed=9)
        self.assertTrue(result["terminated"])
        self.assertEqual(result["completed"], len(env.instance.requests))

    def test_paired_summary_contains_required_metrics(self):
        env = make_env(stage=0, seed=4)
        observation, _ = env.reset(seed=4)
        model = DynamicPlanActorCritic(
            observation["candidate_features"].shape[1],
            observation["global_features"].shape[0],
            hidden_dim=16,
            request_feature_dim=observation["request_features"].shape[1],
        )
        result = paired_evaluation(
            lambda seed: make_env(stage=0, seed=seed),
            {
                "rl": LearnedController(model, torch.device("cpu")),
                "greedy": NamedBaselineController("greedy"),
                "qddca": NamedBaselineController("qddca"),
                "random": NamedBaselineController("random"),
            },
            [20, 21],
        )
        required = {
            "completion", "timeout_rate", "pending_rate",
            "mean_ttl_capped_delay", "mean_success_delay",
            "p95_success_delay", "success_makespan",
            "episode_end_time", "return",
        }
        for metrics in result["core_summary"].values():
            self.assertEqual(set(metrics), required)


if __name__ == "__main__":
    unittest.main()
