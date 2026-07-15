from __future__ import annotations

import unittest

import numpy as np
import torch

from routing_rl.config import PPOConfig
from routing_rl.model import DynamicPlanActorCritic
from routing_rl.ppo import (
    PolicyObservation,
    RolloutBuffer,
    Transition,
    act,
    collate_transitions,
    parse_observation,
    ppo_update,
)


class MaskedPPOTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.device = torch.device("cpu")
        self.model = DynamicPlanActorCritic(4, 3, hidden_dim=32, request_feature_dim=2)

    def observation(self, legal=(True, False, True)):
        return {
            "candidate_features": np.asarray(
                [[1.0, 0.0, 0.2, 0.1], [0.0, 1.0, 0.4, 0.3], [1.0, 1.0, 0.1, 0.7]],
                dtype=np.float32,
            ),
            "global_features": np.asarray([0.2, 0.4, 0.6], dtype=np.float32),
            "request_features": np.asarray([[0.2, 1.0], [0.8, 0.3]], dtype=np.float32),
            "request_mask": np.asarray([True, True]),
            "action_mask": np.asarray([*legal, True]),
        }

    def test_illegal_plan_never_sampled_and_stop_is_last(self):
        parsed = parse_observation(self.observation())
        self.assertEqual(parsed.stop_action, 3)
        for _ in range(100):
            action, _, _ = act(self.model, parsed, self.device)
            self.assertIn(action, (0, 2, 3))

    def test_all_plans_masked_still_allows_stop(self):
        parsed = parse_observation(self.observation((False, False, False)))
        action, _, _ = act(self.model, parsed, self.device)
        self.assertEqual(action, parsed.stop_action)

    def test_padding_remaps_each_local_stop_to_final_column(self):
        short_raw = self.observation((True, False, False))
        short_raw["candidate_features"] = short_raw["candidate_features"][:1]
        short_raw["action_mask"] = np.asarray([True, True])
        short = parse_observation(short_raw)
        long = parse_observation(self.observation())
        transitions = [
            Transition(short, short.stop_action, 0.0, 0.0, 0.0, 0.0, False, False),
            Transition(long, long.stop_action, 0.0, 0.0, 0.0, 0.0, True, True),
        ]
        batch = collate_transitions(
            transitions,
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.arange(2),
            self.device,
        )
        self.assertEqual(batch.action_mask.shape[1], 4)
        self.assertEqual(batch.actions.tolist(), [3, 3])
        self.assertEqual(batch.action_mask[0].tolist(), [True, False, False, True])

    def test_gae_does_not_cross_episode_boundary(self):
        parsed = parse_observation(self.observation())
        rollout = RolloutBuffer(
            [
                Transition(parsed, 3, 0.0, 0.0, 1.0, 10.0, True, True),
                Transition(parsed, 3, 0.0, 0.0, 7.0, 0.0, True, True),
            ]
        )
        config = PPOConfig(normalize_advantage=False)
        advantages, returns = rollout.advantages_and_returns(config)
        np.testing.assert_allclose(advantages, [1.0, 7.0])
        np.testing.assert_allclose(returns, advantages)

    def test_zero_duration_plan_selection_does_not_discount_stop_reward(self):
        parsed = parse_observation(self.observation())
        rollout = RolloutBuffer(
            [
                Transition(parsed, 0, 0.0, 0.0, 0.0, 0.0, False, False, duration=0.0),
                Transition(parsed, 3, 0.0, 0.0, 2.0, 0.0, True, True, duration=1.0),
            ]
        )
        advantages, _ = rollout.advantages_and_returns(
            PPOConfig(normalize_advantage=False)
        )
        np.testing.assert_allclose(advantages, [2.0, 2.0])

    def test_truncation_bootstraps_but_cuts_the_next_episode_trace(self):
        parsed = parse_observation(self.observation())
        rollout = RolloutBuffer(
            [
                Transition(parsed, 3, 0.0, 0.0, 1.0, 10.0, False, True),
                Transition(parsed, 3, 0.0, 0.0, 100.0, 0.0, True, True),
            ]
        )
        advantages, _ = rollout.advantages_and_returns(
            PPOConfig(gamma=0.9, normalize_advantage=False)
        )
        np.testing.assert_allclose(advantages, [10.0, 100.0])

    def test_single_ppo_update_is_finite(self):
        parsed = parse_observation(self.observation())
        transitions = []
        for _ in range(16):
            action, log_prob, value = act(self.model, parsed, self.device)
            transitions.append(
                Transition(parsed, action, log_prob, value, 0.5, 0.0, True, True)
            )
        rollout = RolloutBuffer(transitions)
        config = PPOConfig(
            hidden_dim=32, ppo_epochs=2, minibatch_size=8, normalize_advantage=True
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        metrics = ppo_update(
            self.model, optimizer, rollout, config, self.device, np.random.default_rng(0)
        )
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
