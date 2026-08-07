import unittest

import networkx as nx

from qnet_core.planner_api import COMMIT, PlanDescriptor, PlanningSnapshot
from qnet_core.scenario import ScenarioConfig, make_episode
from qnet_core.spec import EpisodeSpec, PhysicalConfig, RequestSpec


class SharedContractTests(unittest.TestCase):
    def test_parallel_corridor_batch_is_seeded_and_has_equal_routes(self):
        config = ScenarioConfig(
            request_count=4,
            min_hops=3,
            max_hops=3,
            topology_mode="parallel_corridors",
            parallel_corridors=2,
            batch_mode=True,
            ttl=8,
            horizon=8,
        )
        first = make_episode(config, 123)
        second = make_episode(config, 123)
        self.assertEqual(first, second)
        self.assertEqual(
            {(request.source, request.destination, request.arrival)
             for request in first.requests},
            {(0, 1, 0)},
        )
        graph = nx.Graph(first.edges)
        self.assertEqual(
            list(nx.all_simple_paths(graph, 0, 1)),
            [[0, 2, 3, 1], [0, 4, 5, 1]],
        )

    def test_poisson_arrivals_are_seeded_and_horizon_covers_deadlines(self):
        config = ScenarioConfig(request_count=20, ttl=7, horizon=7, arrival_rate=0.5)
        first = make_episode(config, 123)
        second = make_episode(config, 123)
        arrivals = [request.arrival for request in first.requests]
        self.assertEqual(arrivals, [request.arrival for request in second.requests])
        self.assertEqual(arrivals, sorted(arrivals))
        self.assertEqual(first.horizon, arrivals[-1] + config.ttl)

    def test_generated_requests_use_seeded_distributed_endpoints(self):
        config = ScenarioConfig(request_count=100, min_hops=2, max_hops=50)
        first = make_episode(config, 321)
        second = make_episode(config, 321)
        self.assertEqual(first.requests, second.requests)
        graph = nx.Graph(first.edges)
        self.assertEqual(graph.number_of_nodes(), 200)
        self.assertTrue(nx.is_connected(graph))
        self.assertGreater(graph.number_of_edges(), graph.number_of_nodes() - 1)
        self.assertGreaterEqual(nx.diameter(graph), 50)
        distances = sorted(
            nx.shortest_path_length(graph, request.source, request.destination)
            for request in first.requests
        )
        expected = sorted(
            2 + round(48 * index / 99)
            for index in range(100)
        )
        self.assertEqual(distances, expected)
        self.assertGreater(len({request.source for request in first.requests}), 1)
        self.assertTrue(any(request.source < request.destination for request in first.requests))
        self.assertTrue(any(request.source > request.destination for request in first.requests))
        self.assertTrue(all(
            request.source in first.nodes and request.destination in first.nodes
            for request in first.requests
        ))

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
