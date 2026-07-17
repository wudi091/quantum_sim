import random
import unittest

from qnet_core.qcast_paper.model import EdgeSpec, MajorReservation, QCastTopology, SDPair
from qnet_core.qcast_paper.recovery import connected, recover_lane, recovery_loop_edges, xor_edges
from qnet_core.qcast_paper.simulator import SimulationConfig, run_experiment, run_slot, sample_sd_pairs
from qnet_core.qcast_paper.allocation import allocate_recovery_paths, width_first_path


class RecoveryAndSimulatorTests(unittest.TestCase):
    def test_xor_loop_connectivity(self):
        major = (0, 1, 2)
        self.assertTrue(connected(xor_edges({(0, 1)}, {(1, 2)}), 0, 2))
        topology = QCastTopology(
            {0: 4, 1: 8, 2: 4, 3: 8},
            [EdgeSpec(0, 1, 2, 1.0), EdgeSpec(1, 2, 2, 1.0),
             EdgeSpec(0, 3, 2, 1.0), EdgeSpec(3, 2, 2, 1.0)],
            swap_probability=1.0,
        )
        residual = topology.residual()
        candidate = width_first_path(residual, 0, 2)
        channels = residual.reserve(candidate.path, candidate.width)
        major_res = MajorReservation(SDPair(0, 2), candidate.path, candidate.width,
                                     candidate.ext, channels)
        recovery = allocate_recovery_paths(residual, major_res, 2)
        self.assertTrue(recovery)
        self.assertTrue(recovery_loop_edges(major, recovery[0]))

    def test_sample_pairs_are_distinct(self):
        topology = QCastTopology({index: 4 for index in range(8)},
                                 [EdgeSpec(index, index + 1, 2, 1.0) for index in range(7)])
        pairs = sample_sd_pairs(topology, 3, random.Random(5))
        endpoints = [node for pair in pairs for node in (pair.source, pair.destination)]
        self.assertEqual(len(endpoints), len(set(endpoints)))

    def test_run_slot_resets_and_reports_width_throughput(self):
        topology = QCastTopology(
            {0: 4, 1: 8, 2: 4},
            [EdgeSpec(0, 1, 2, 1.0), EdgeSpec(1, 2, 2, 1.0)],
            swap_probability=1.0,
        )
        result = run_slot(topology, [SDPair(0, 2)], random.Random(2),
                          config=SimulationConfig(recovery=False))
        self.assertGreaterEqual(result.throughput, 1)
        self.assertLessEqual(result.throughput, 2)
        self.assertEqual(result.successful_pairs, 1)

    def test_author_p4_falls_back_to_major_and_uses_successful_low_id_link(self):
        topology = QCastTopology(
            {0: 4, 1: 8, 2: 4},
            [EdgeSpec(0, 1, 2, 1.0), EdgeSpec(1, 2, 2, 1.0)],
            swap_probability=1.0,
        )
        residual = topology.residual()
        candidate = width_first_path(residual, 0, 2)
        channels = residual.reserve(candidate.path, candidate.width)
        major = MajorReservation(SDPair(0, 2), candidate.path, candidate.width,
                                 candidate.ext, channels)
        # One successful channel per edge: source P4 can deliver one of the two
        # width lanes despite marking each edge broken.
        outcomes = {ref: ref.channel_id % 2 == 0 for ref in channels}
        used = set()
        first = recover_lane(major, (), outcomes, 0, swap_probability=1.0,
                             used_channels=used)
        second = recover_lane(major, (), outcomes, 1, swap_probability=1.0,
                              used_channels=used)
        self.assertTrue(first.success)
        self.assertFalse(second.success)

    def test_experiment_pair_mean_is_distinct_pair_count(self):
        def factory(index, rng):
            del index, rng
            return QCastTopology(
                {0: 8, 1: 12, 2: 8},
                [EdgeSpec(0, 1, 2, 1.0), EdgeSpec(1, 2, 2, 1.0)],
                swap_probability=1.0,
            )
        result = run_experiment(
            factory, 1, topology_count=1, slots_per_topology=2, seed=1,
            config=SimulationConfig(recovery=False),
        )
        self.assertLessEqual(result["successful_pairs_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
