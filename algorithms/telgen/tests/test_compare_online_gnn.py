import unittest

from algorithms.telgen.compare_online_gnn import (
    _resolve_construction_space,
)


class OnlineGNNComparisonConfigurationTests(unittest.TestCase):
    def test_adaptive_construction_uses_the_requested_tree_union(self):
        self.assertEqual(
            _resolve_construction_space(5, None),
            ((), 5, "adaptive_swap_tree_selection"),
        )

    def test_fixed_construction_keeps_one_seen_swap_tree_kind(self):
        self.assertEqual(
            _resolve_construction_space(5, 3),
            (("swap_tree_3",), None, "fixed_swap_tree_3"),
        )

    def test_fixed_construction_rejects_an_out_of_range_tree(self):
        with self.assertRaises(ValueError):
            _resolve_construction_space(5, 5)


if __name__ == "__main__":
    unittest.main()
