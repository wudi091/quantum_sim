import math
import unittest

from qnet_core.qcast_paper.allocation import (
    allocate_recovery_paths, eda_fixed_width, geda_allocate, width_first_path,
)
from qnet_core.qcast_paper.ext import expected_throughput, propagate_distribution
from qnet_core.qcast_paper.model import EdgeSpec, QCastTopology, SDPair


class ExtTests(unittest.TestCase):
    def test_heterogeneous_single_lane_uses_q_h_minus_one(self):
        value = expected_throughput((0.5, 0.25), 1, 0.9)
        self.assertAlmostEqual(value, 0.5 * 0.25 * 0.9)

    def test_one_hop_has_no_swap_penalty(self):
        self.assertAlmostEqual(expected_throughput((0.6,), 2, 0.2), 2 * 0.6 * 0.2**0)

    def test_distribution_is_normalized(self):
        previous = [0.2, 0.5, 0.3]
        updated = propagate_distribution(previous, 0.7, 2)
        self.assertAlmostEqual(sum(updated), 1.0)
        self.assertTrue(all(value >= 0 for value in updated))


class AllocationTests(unittest.TestCase):
    @staticmethod
    def topology(node_qubits, edges, **kwargs):
        return QCastTopology(node_qubits, edges, **kwargs)

    def test_width_first_scans_from_max_and_rejects_interior_capacity(self):
        topology = self.topology(
            {0: 3, 1: 2, 2: 3},
            [EdgeSpec(0, 1, 3, 0.8), EdgeSpec(1, 2, 3, 0.8)],
            swap_probability=1.0,
        )
        candidate = width_first_path(topology.residual(), 0, 2)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.width, 1)

    def test_eda_is_fixed_width(self):
        topology = self.topology(
            {0: 2, 1: 4, 2: 2, 3: 4},
            [
                EdgeSpec(0, 1, 2, 0.9), EdgeSpec(1, 2, 2, 0.9),
                EdgeSpec(0, 3, 2, 0.6), EdgeSpec(3, 2, 2, 0.6),
            ],
            swap_probability=1.0,
        )
        candidate = eda_fixed_width(topology.residual(), 0, 2, 2)
        self.assertEqual(candidate.width, 2)
        self.assertEqual(candidate.path, (0, 1, 2))

    def test_geda_updates_residual_and_allocates_next_pair(self):
        topology = self.topology(
            {0: 2, 1: 4, 2: 2, 3: 2},
            [
                EdgeSpec(0, 1, 2, 0.9), EdgeSpec(1, 2, 2, 0.9),
                EdgeSpec(1, 3, 2, 0.8),
            ],
            swap_probability=1.0,
        )
        result = geda_allocate(topology, [SDPair(0, 2), SDPair(0, 3)], path_cap=4)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].pair, SDPair(0, 2))
        self.assertEqual(result[0].width, 2)

    def test_recovery_paths_use_residual_channels(self):
        topology = self.topology(
            {0: 4, 1: 8, 2: 4, 3: 8},
            [
                EdgeSpec(0, 1, 3, 0.9), EdgeSpec(1, 2, 3, 0.9),
                EdgeSpec(1, 3, 3, 0.8), EdgeSpec(3, 2, 3, 0.8),
            ],
            swap_probability=1.0,
            link_state_range=2,
        )
        residual = topology.residual()
        major_candidate = width_first_path(residual, 0, 2)
        channels = residual.reserve(major_candidate.path, major_candidate.width)
        from qnet_core.qcast_paper.model import MajorReservation
        major = MajorReservation(SDPair(0, 2), major_candidate.path, major_candidate.width,
                                 major_candidate.expected_throughput, channels)
        recovery = allocate_recovery_paths(residual, major, 2)
        self.assertTrue(recovery)
        self.assertTrue(all(item.channels for item in recovery))


if __name__ == "__main__":
    unittest.main()

