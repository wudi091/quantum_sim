import unittest

from qnet_core.env import SharedRoutingEnv
from qnet_core.gym_env import GymConfig, SequenceGymEnv
from qnet_core.planners import GreedyPlanner, QDDCAPlanner, RandomPlanner
from qnet_core.scenario import ScenarioConfig
from qnet_core.spec import EpisodeSpec, PhysicalConfig, RequestSpec


class SequenceEnvironmentTests(unittest.TestCase):
    def test_three_node_request_uses_shared_generation_and_settlement(self):
        env = SharedRoutingEnv(EpisodeSpec(
            seed=11,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r0", 0, 2, ttl=4),),
            horizon=8,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                memory_capacity=2,
            ),
        ))
        snapshot = env.snapshot()
        self.assertGreaterEqual(len(snapshot.candidates), 2)
        plan = next(item for item in snapshot.candidates if item.completes_request)
        result = env.commit((plan.plan_id,))
        self.assertEqual(result["completed_now"], 1)
        self.assertEqual(result["metrics"]["completion_rate"], 1.0)
        self.assertTrue(env.done)

    def test_same_seed_produces_same_initial_snapshot(self):
        spec = EpisodeSpec(
            seed=13,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r0", 0, 2, ttl=4),),
            horizon=8,
            physical=PhysicalConfig(generation_probability=0.5),
        )
        left, right = SharedRoutingEnv(spec).snapshot(), SharedRoutingEnv(spec).snapshot()
        self.assertEqual(left.resources, right.resources)
        self.assertEqual(left.candidates, right.candidates)

    def test_all_planners_only_select_from_same_snapshot(self):
        spec = EpisodeSpec(
            seed=17,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r0", 0, 2, ttl=4),),
            horizon=8,
            physical=PhysicalConfig(generation_probability=1.0),
        )
        env = SharedRoutingEnv(spec)
        snapshot = env.snapshot()
        before = snapshot.candidates
        valid = {plan.plan_id for plan in snapshot.candidates}
        for planner in (GreedyPlanner(), QDDCAPlanner(), RandomPlanner(3)):
            planner.reset(spec.seed)
            self.assertLessEqual(set(planner.select(snapshot)), valid)
            self.assertEqual(snapshot.candidates, before)

    def test_gym_wrapper_only_advances_on_stop(self):
        env = SequenceGymEnv(GymConfig(
            max_requests=2,
            max_candidates_per_request=3,
            max_hops=3,
            scenario=ScenarioConfig(
                request_count=1, min_hops=2, max_hops=2, ttl=4, horizon=6,
                physical=PhysicalConfig(generation_probability=1.0, swap_probability=1.0),
            ),
            seed=19,
        ))
        observation, _ = env.reset(seed=19)
        action = int(next(index for index, legal in enumerate(observation["action_mask"][:-1]) if legal))
        _, _, _, _, info = env.step(action)
        self.assertEqual(info["phase"], "select")
        self.assertEqual(env.core.time, 0)
        _, _, terminated, truncated, info = env.step(env.stop_action)
        self.assertFalse(truncated)
        self.assertGreaterEqual(env.core.time, 1)
        self.assertEqual(info["phase"], "execute")

    def test_completion_after_deadline_is_timeout(self):
        env = SharedRoutingEnv(EpisodeSpec(
            seed=23,
            nodes=(0, 1, 2, 3),
            edges=((0, 1), (1, 2), (2, 3)),
            requests=(RequestSpec("r0", 0, 3, ttl=1),),
            horizon=4,
            physical=PhysicalConfig(generation_probability=1.0, swap_probability=1.0),
        ))
        plan = next(item for item in env.snapshot().candidates if item.completes_request)
        result = env.commit((plan.plan_id,))
        self.assertEqual(result["metrics"]["completion_rate"], 0.0)
        self.assertEqual(result["metrics"]["timeout_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
