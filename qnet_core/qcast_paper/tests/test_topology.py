import math
import random
import unittest

import networkx as nx

from qnet_core.qcast_paper.topology import (
    AuthorTopologyConfig, calibrate_alpha, generate_author_topology_with_metadata,
)


class TopologyTests(unittest.TestCase):
    def test_alpha_calibration_hits_target(self):
        alpha = calibrate_alpha((1.0, 2.0, 3.0), 0.7)
        value = sum(math.exp(-alpha * distance) for distance in (1.0, 2.0, 3.0)) / 3
        self.assertAlmostEqual(value, 0.7, delta=0.002)

    def test_author_topology_is_connected_and_heterogeneous(self):
        result = generate_author_topology_with_metadata(
            AuthorTopologyConfig(node_count=20, average_degree=4, target_link_probability=0.6),
            random.Random(19900111),
        )
        graph = nx.Graph()
        graph.add_nodes_from(result.topology.nodes)
        graph.add_edges_from((edge.u, edge.v) for edge in result.topology.edges)
        self.assertTrue(nx.is_connected(graph))
        self.assertEqual(len(result.positions), 20)
        widths = {edge.width for edge in result.topology.edges}
        self.assertTrue(widths <= set(range(3, 8)))
        self.assertGreater(len(widths), 1)

    def test_reference_resource_statistics_match_author_scale(self):
        result = generate_author_topology_with_metadata(
            AuthorTopologyConfig(), random.Random(19900111),
        )
        graph = nx.Graph()
        graph.add_nodes_from(result.topology.nodes)
        graph.add_edges_from((item.u, item.v) for item in result.topology.edges)
        physical_channels = len(result.topology.channels)
        average_neighbours = 2 * graph.number_of_edges() / graph.number_of_nodes()
        # Official smoke: about 2700 physical channels and 6.46 neighbours.
        self.assertGreater(physical_channels, 2300)
        self.assertLess(physical_channels, 3100)
        self.assertAlmostEqual(average_neighbours, 6.46, delta=0.8)


if __name__ == "__main__":
    unittest.main()
