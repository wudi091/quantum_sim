from __future__ import annotations

import unittest

import numpy as np

from batchswap_rl.baselines import GreedyPolicy, QDDCAPolicy, run_policy
from batchswap_rl.config import CurriculumStage, default_curriculum
from batchswap_rl.env import (
    BatchSwapEnv,
    BatchSwapInstance,
    EnvConfig,
    RequestSpec,
    edge_key,
)


def instance(*requests: RequestSpec, arrivals: dict[int, tuple[tuple[str, str], ...]]):
    edges = tuple(sorted({edge for request in requests for edge in request.edges}))
    return BatchSwapInstance(0, tuple(requests), edges, arrivals)


class BatchSwapEnvTests(unittest.TestCase):
    def test_default_long_curriculum_uses_one_hundred_requests(self):
        stage = default_curriculum()[-1]
        self.assertEqual(stage.min_requests, 100)
        self.assertEqual(stage.max_requests, 100)

    def test_fixed_shapes_and_max_half_short_candidates(self):
        request = RequestSpec("r0", tuple(f"n{i}" for i in range(9)))
        arrivals = {0: request.edges}
        env = BatchSwapEnv(EnvConfig(max_requests=4, max_candidates_per_request=3,
                                     max_hops=10, request_count=1),
                           instance=instance(request, arrivals=arrivals))
        obs, _ = env.reset()
        plans = [plan for plan in env.current_plans if plan is not None]
        self.assertEqual([(p.kind, p.progress) for p in plans],
                         [("max", 8), ("half", 4), ("short", 1)])
        self.assertEqual(obs["candidate_features"].shape, (12, env.candidate_feature_dim))
        self.assertEqual(obs["request_features"].shape, (4, env.request_feature_dim))
        self.assertEqual(obs["action_mask"].shape, (13,))
        self.assertTrue(obs["action_mask"][-1])

    def test_sequential_mask_enforces_request_edge_and_node_constraints(self):
        # Distinct elementary edges, but both long plans swap at shared node x.
        first = RequestSpec("r0", ("a", "x", "b"))
        second = RequestSpec("r1", ("c", "x", "d"))
        arrivals = {0: first.edges + second.edges}
        env = BatchSwapEnv(EnvConfig(max_requests=2, max_hops=4, request_count=2,
                                     node_capacity=1),
                           instance=instance(first, second, arrivals=arrivals))
        obs, _ = env.reset()
        action0 = 0
        obs, reward, terminated, truncated, info = env.step(action0)
        self.assertEqual(reward, 0.0)
        self.assertEqual(info["duration"], 0)
        self.assertFalse(terminated or truncated)
        self.assertFalse(obs["action_mask"][:3].any())  # same request <= 1
        self.assertFalse(obs["action_mask"][3])  # second max plan uses x
        self.assertTrue(obs["action_mask"][-1])

    def test_swap_depth_advances_real_subslots(self):
        request = RequestSpec("r0", tuple(f"n{i}" for i in range(9)))
        env = BatchSwapEnv(EnvConfig(max_requests=1, max_hops=8, request_count=1),
                           instance=instance(request, arrivals={0: request.edges}))
        obs, _ = env.reset()
        obs, _, _, _, select_info = env.step(0)
        self.assertEqual(select_info["duration"], 0)
        obs, _, terminated, truncated, info = env.step(env.stop_action)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["duration"], 3)  # ceil(log2(8))
        self.assertEqual(info["time"], 3)
        self.assertEqual(info["swaps"], 7)

    def test_waiting_never_drops_request(self):
        request = RequestSpec("r0", ("s", "d"))
        edge = edge_key("s", "d")
        env = BatchSwapEnv(EnvConfig(max_requests=1, max_hops=2, request_count=1,
                                     max_subslots=20),
                           instance=instance(request, arrivals={5: (edge,)}))
        obs, _ = env.reset()
        for expected_time in range(1, 6):
            obs, _, terminated, truncated, info = env.step(env.stop_action)
            self.assertFalse(terminated or truncated)
            self.assertEqual(info["time"], expected_time)
            self.assertEqual(info["active"], 1)
        action = int(np.flatnonzero(obs["action_mask"][:-1])[0])
        obs, _, _, _, _ = env.step(action)
        obs, _, terminated, truncated, info = env.step(env.stop_action)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["completed"], 1)

    def test_shared_edge_inventory_cannot_be_double_spent(self):
        first = RequestSpec("r0", ("s", "d"))
        second = RequestSpec("r1", ("s", "d"))
        shared = edge_key("s", "d")
        env = BatchSwapEnv(EnvConfig(max_requests=2, max_hops=2, request_count=2),
                           instance=instance(first, second, arrivals={0: (shared,)}))
        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(0)
        self.assertFalse(obs["action_mask"][3])
        obs, _, _, _, info = env.step(env.stop_action)
        self.assertEqual(info["elementary_now"], 1)
        self.assertEqual(info["completed_now"], 1)

    def test_truncation_keeps_pending_request_instead_of_dropping_it(self):
        request = RequestSpec("r0", ("s", "d"))
        edge = edge_key("s", "d")
        env = BatchSwapEnv(EnvConfig(max_requests=1, max_hops=2, request_count=1,
                                     max_subslots=2),
                           instance=instance(request, arrivals={10: (edge,)}))
        obs, _ = env.reset()
        obs, _, terminated, truncated, _ = env.step(env.stop_action)
        self.assertFalse(terminated or truncated)
        obs, _, terminated, truncated, info = env.step(env.stop_action)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["active"], 1)
        self.assertEqual(info["completed"], 0)
        self.assertEqual(env.frontier["r0"], 0)

    def test_curriculum_changes_workload_not_tensor_bounds(self):
        env = BatchSwapEnv(EnvConfig(max_requests=30, max_hops=50))
        stage = CurriculumStage("medium", 5, 15, 1, 5, 10)
        env.set_curriculum(stage)
        obs, _ = env.reset(seed=7)
        self.assertEqual(len(env.instance.requests), 10)
        self.assertTrue(all(5 <= request.hops <= 15 for request in env.instance.requests))
        self.assertEqual(obs["request_features"].shape[0], 30)
        self.assertEqual(obs["candidate_features"].shape[0], 90)

    def test_reset_seed_stream_varies_and_is_reproducible(self):
        config = EnvConfig(max_requests=4, max_hops=5, request_count=4,
                           min_hops=2, curriculum_max_hops=5,
                           generation_probability=0.5)

        def episode_signatures():
            env = BatchSwapEnv(config)
            env.reset(seed=17)
            signatures = []
            for _ in range(3):
                env.reset()
                signatures.append((env.instance.seed, env.instance.requests,
                                   tuple(sorted(env.instance.arrivals.items()))))
            return signatures

        first = episode_signatures()
        second = episode_signatures()
        self.assertEqual([row[0] for row in first], [18, 19, 20])
        self.assertEqual(first, second)
        self.assertGreater(len({row[2] for row in first}), 1)

    def test_nonlearning_baselines_complete_without_labels(self):
        config = EnvConfig(max_requests=4, max_hops=5, request_count=4,
                           min_hops=2, curriculum_max_hops=5,
                           generation_probability=1.0, max_subslots=100)
        for policy in (GreedyPolicy(), QDDCAPolicy()):
            result = run_policy(BatchSwapEnv(config), policy, seed=3)
            self.assertTrue(result["terminated"])
            self.assertFalse(result["truncated"])
            self.assertEqual(result["completed"], 4)


if __name__ == "__main__":
    unittest.main()
