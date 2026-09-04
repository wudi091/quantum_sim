import unittest
from unittest.mock import patch

import networkx as nx

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

    def test_generated_batch_is_seeded_and_simultaneous(self):
        config = ScenarioConfig(request_count=20, ttl=7, horizon=7)
        first = make_episode(config, 123)
        second = make_episode(config, 123)
        arrivals = [request.arrival for request in first.requests]
        self.assertEqual(arrivals, [request.arrival for request in second.requests])
        self.assertEqual(set(arrivals), {0})
        self.assertEqual(first.horizon, config.horizon)
        self.assertEqual(
            {request.deadline for request in first.requests},
            {config.ttl},
        )

    def test_topology_seed_separates_graph_and_request_randomness(self):
        config = ScenarioConfig(
            request_count=20,
            min_hops=None,
            max_hops=None,
            topology_nodes=24,
            topology_mode="random_regular",
            random_regular_degree=4,
            endpoint_mode="uniform_random",
            ttl=8,
            horizon=8,
        )
        first = make_episode(config, 501, topology_seed=77)
        second = make_episode(config, 502, topology_seed=77)
        unseen = make_episode(config, 501, topology_seed=78)
        self.assertEqual(first.nodes, second.nodes)
        self.assertEqual(first.edges, second.edges)
        self.assertNotEqual(first.requests, second.requests)
        self.assertNotEqual(first.edges, unseen.edges)

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

    def test_waxman_batch_has_simultaneous_arrivals_and_distributed_endpoints(self):
        config = ScenarioConfig(
            request_count=12,
            min_hops=2,
            max_hops=5,
            topology_nodes=20,
            ttl=8,
            horizon=8,
        )
        episode = make_episode(config, 777)
        self.assertEqual({request.arrival for request in episode.requests}, {0})
        self.assertGreater(
            len({
                (request.source, request.destination)
                for request in episode.requests
            }),
            1,
        )
        graph = nx.Graph(episode.edges)
        expected_hops = sorted(
            config.min_hops + round(
                (config.max_hops - config.min_hops) * index
                / (config.request_count - 1)
            )
            for index in range(config.request_count)
        )
        actual_hops = sorted(
            nx.shortest_path_length(
                graph, request.source, request.destination
            )
            for request in episode.requests
        )
        self.assertEqual(actual_hops, expected_hops)

    def test_waxman_uniform_random_endpoints_only_require_connectivity(self):
        config = ScenarioConfig(
            request_count=200,
            topology_nodes=32,
            waxman_alpha=0.2,
            waxman_beta=0.7,
            waxman_add_mst=False,
            endpoint_mode="uniform_random",
            ttl=16,
            horizon=24,
        )
        episode = make_episode(config, 3101)
        self.assertEqual(len(episode.nodes), 32)
        graph = nx.Graph(episode.edges)
        self.assertTrue(nx.is_connected(graph))
        self.assertGreater(graph.number_of_edges(), graph.number_of_nodes() - 1)
        distances = []
        for request in episode.requests:
            self.assertNotEqual(request.source, request.destination)
            distances.append(nx.shortest_path_length(
                graph, request.source, request.destination
            ))
        self.assertGreater(len(set(distances)), 1)
        self.assertGreater(len({
            (request.source, request.destination)
            for request in episode.requests
        }), 1)

    def test_waxman_uniform_random_mode_rejects_disconnected_topology(self):
        config = ScenarioConfig(
            request_count=1,
            topology_nodes=32,
            topology_attempts=2,
            waxman_add_mst=False,
            endpoint_mode="uniform_random",
        )
        disconnected = nx.Graph()
        disconnected.add_nodes_from(range(32))
        disconnected.add_edges_from((index, index + 1) for index in range(15))
        with patch(
            "qnet_core.scenario.nx.waxman_graph",
            return_value=disconnected,
        ) as generator:
            with self.assertRaises(RuntimeError):
                make_episode(config, 7)
        self.assertEqual(generator.call_count, 2)

    def test_barabasi_albert_batch_is_seeded_and_distance_stratified(self):
        config = ScenarioConfig(
            request_count=24,
            min_hops=2,
            max_hops=4,
            topology_nodes=96,
            topology_mode="barabasi_albert",
            barabasi_attachment=2,
            ttl=8,
            horizon=8,
        )
        first = make_episode(config, 919)
        second = make_episode(config, 919)
        self.assertEqual(first, second)
        graph = nx.Graph(first.edges)
        self.assertTrue(nx.is_connected(graph))
        self.assertEqual(len(first.nodes), 96)
        expected_hops = sorted(
            config.min_hops + round(
                (config.max_hops - config.min_hops) * index
                / (config.request_count - 1)
            )
            for index in range(config.request_count)
        )
        actual_hops = sorted(
            nx.shortest_path_length(
                graph, request.source, request.destination
            )
            for request in first.requests
        )
        self.assertEqual(actual_hops, expected_hops)

    def test_barabasi_albert_rejects_invalid_attachment_count(self):
        config = ScenarioConfig(
            request_count=1,
            topology_nodes=8,
            topology_mode="barabasi_albert",
            barabasi_attachment=8,
        )
        with self.assertRaises(ValueError):
            make_episode(config, 1)

    def test_erdos_renyi_batch_is_seeded_and_distance_stratified(self):
        config = ScenarioConfig(
            request_count=24,
            min_hops=4,
            max_hops=4,
            topology_nodes=64,
            topology_mode="erdos_renyi",
            erdos_renyi_mean_degree=6.0,
            ttl=8,
            horizon=8,
        )
        first = make_episode(config, 44000)
        second = make_episode(config, 44000)
        self.assertEqual(first, second)
        graph = nx.Graph(first.edges)
        self.assertTrue(nx.is_connected(graph))
        self.assertEqual(len(first.nodes), 64)
        self.assertEqual({
            nx.shortest_path_length(
                graph, request.source, request.destination
            )
            for request in first.requests
        }, {4})

    def test_random_regular_batch_is_seeded_and_distance_stratified(self):
        config = ScenarioConfig(
            request_count=24,
            min_hops=4,
            max_hops=4,
            topology_nodes=64,
            topology_mode="random_regular",
            random_regular_degree=4,
            ttl=8,
            horizon=8,
        )
        first = make_episode(config, 44100)
        second = make_episode(config, 44100)
        self.assertEqual(first, second)
        graph = nx.Graph(first.edges)
        self.assertTrue(nx.is_connected(graph))
        self.assertEqual(len(first.nodes), 64)
        self.assertEqual({
            nx.shortest_path_length(
                graph, request.source, request.destination
            )
            for request in first.requests
        }, {4})

    def test_random_regular_rejects_odd_degree_sum(self):
        config = ScenarioConfig(
            request_count=1,
            topology_nodes=9,
            topology_mode="random_regular",
            random_regular_degree=3,
        )
        with self.assertRaises(ValueError):
            make_episode(config, 1)

    def test_episode_spec_rejects_invalid_probability(self):
        with self.assertRaises(ValueError):
            EpisodeSpec(
                seed=0, nodes=(0, 1), edges=((0, 1),), requests=(), horizon=1,
                physical=PhysicalConfig(generation_probability=0),
            )

if __name__ == "__main__":
    unittest.main()
