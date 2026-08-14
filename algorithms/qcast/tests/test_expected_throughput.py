import unittest

from algorithms.qcast.expected_throughput import (
    expected_throughput,
    propagate_distribution,
)


class ExpectedThroughputTests(unittest.TestCase):
    def test_heterogeneous_single_lane_uses_q_h_minus_one(self):
        value = expected_throughput((0.5, 0.25), 1, 0.9)
        self.assertAlmostEqual(value, 0.5 * 0.25 * 0.9)

    def test_one_hop_has_no_swap_penalty(self):
        self.assertAlmostEqual(
            expected_throughput((0.6,), 2, 0.2),
            2 * 0.6,
        )

    def test_distribution_is_normalized(self):
        updated = propagate_distribution([0.2, 0.5, 0.3], 0.7, 2)
        self.assertAlmostEqual(sum(updated), 1.0)
        self.assertTrue(all(value >= 0 for value in updated))


if __name__ == "__main__":
    unittest.main()
