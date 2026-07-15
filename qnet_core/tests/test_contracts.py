import unittest

from qnet_core.planner_api import COMMIT, PlanDescriptor, PlanningSnapshot
from qnet_core.scenario import ScenarioConfig, make_episode
from qnet_core.spec import EpisodeSpec, PhysicalConfig, RequestSpec


class SharedContractTests(unittest.TestCase):
    def test_poisson_arrivals_are_seeded_and_horizon_covers_deadlines(self):
        config = ScenarioConfig(request_count=20, ttl=7, horizon=7, arrival_rate=0.5)
        first = make_episode(config, 123)
        second = make_episode(config, 123)
        arrivals = [request.arrival for request in first.requests]
        self.assertEqual(arrivals, [request.arrival for request in second.requests])
        self.assertEqual(arrivals, sorted(arrivals))
        self.assertEqual(first.horizon, arrivals[-1] + config.ttl)

    def test_episode_spec_rejects_invalid_probability(self):
        with self.assertRaises(ValueError):
            EpisodeSpec(
                seed=0, nodes=(0, 1), edges=((0, 1),), requests=(), horizon=1,
                physical=PhysicalConfig(generation_probability=0),
            )

    def test_planner_snapshot_is_immutable_shape(self):
        request = RequestSpec("r0", 0, 1, ttl=4)
        self.assertEqual(request.deadline, 4)
        self.assertEqual(COMMIT, -1)
        self.assertTrue(PlanningSnapshot(0, (), (), (), (), {}).action_mask == ())


if __name__ == "__main__":
    unittest.main()
