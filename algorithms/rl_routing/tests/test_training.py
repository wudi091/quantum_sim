import math
import unittest

import torch

from algorithms.rl_routing.policy import ARCQPolicy
from algorithms.rl_routing.rollout import collect_episode
from algorithms.rl_routing.training import PPOConfig, PPOTrainer
from algorithms.routing_core.execution import OnlineExecutionConfig
from qnet_core.planning_spec import RequestSpec
from qnet_core.spec import EpisodeSpec, PhysicalConfig


def make_episode(seed):
    return EpisodeSpec(
        seed=seed,
        nodes=(0, 1, 2),
        edges=((0, 1), (1, 2), (0, 2)),
        requests=(
            RequestSpec("r0", 0, 2, ttl=4),
            RequestSpec("r1", 0, 2, ttl=4),
        ),
        horizon=4,
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


class PPOTrainerTests(unittest.TestCase):
    def test_one_update_is_finite_and_changes_parameters(self):
        torch.manual_seed(8400)
        policy = ARCQPolicy(hidden_dim=16, message_passing_layers=1)
        environment_config = OnlineExecutionConfig(
            decision_interval=2,
            path_candidate_count=2,
            construction_kinds=("balanced",),
            purification_kinds=("none",),
        )
        rollouts = tuple(
            collect_episode(
                policy,
                make_episode(seed),
                environment_config,
            )
            for seed in (8401, 8402)
        )
        before = [
            parameter.detach().clone() for parameter in policy.parameters()
        ]
        trainer = PPOTrainer(
            policy,
            PPOConfig(
                learning_rate=1e-3,
                update_epochs=1,
                minibatch_size=8,
            ),
        )
        diagnostics = trainer.update(rollouts, shuffle_seed=8403)
        self.assertEqual(diagnostics.sample_count, 4)
        self.assertTrue(math.isfinite(diagnostics.policy_loss))
        self.assertTrue(math.isfinite(diagnostics.value_loss))
        self.assertLess(diagnostics.maximum_reward_identity_error, 1e-9)
        self.assertTrue(any(
            not torch.equal(previous, current.detach())
            for previous, current in zip(
                before, policy.parameters(), strict=True
            )
        ))

    def test_discounting_cannot_silently_change_the_delay_objective(self):
        with self.assertRaises(ValueError):
            PPOConfig(gamma=0.99)


if __name__ == "__main__":
    unittest.main()
