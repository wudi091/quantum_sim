import unittest

import numpy as np

from algorithms.telgen.comparison_methods import (
    FORMAL_METHOD_ORDER,
    SCALABLE_METHOD_ORDER,
)
from algorithms.telgen.plot_paper_figures import (
    FIGURE_SPECS,
    METHOD_ORDER,
    ROUTING_METHOD_ORDER,
    _parse_numbers,
)
from algorithms.telgen.plot_utils import (
    bootstrap_mean_ci,
    configure_paper_style,
    ecdf,
    method_style,
)


class PaperFigureTests(unittest.TestCase):
    def test_figures_use_the_frozen_method_sets(self):
        self.assertEqual(METHOD_ORDER, FORMAL_METHOD_ORDER)
        self.assertEqual(ROUTING_METHOD_ORDER, SCALABLE_METHOD_ORDER)
        self.assertNotIn("qcast", METHOD_ORDER)

    def test_reference_paper_style_uses_dual_encoding(self):
        configure_paper_style()
        identities = {
            (
                method_style(method).color,
                method_style(method).marker,
                method_style(method).linestyle,
                method_style(method).hatch,
            )
            for method in FORMAL_METHOD_ORDER
        }
        self.assertEqual(len(identities), len(FORMAL_METHOD_ORDER))

    def test_registry_contains_ten_unique_figures(self):
        self.assertEqual(len(FIGURE_SPECS), 10)
        self.assertEqual(len({spec.number for spec in FIGURE_SPECS}), 10)
        self.assertEqual(len({spec.stem for spec in FIGURE_SPECS}), 10)

    def test_bootstrap_is_deterministic_and_contains_mean(self):
        first = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], samples=500, seed=7)
        second = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], samples=500, seed=7)
        self.assertEqual(first, second)
        mean, low, high = first
        self.assertEqual(mean, 2.5)
        self.assertLessEqual(low, mean)
        self.assertGreaterEqual(high, mean)

    def test_ecdf_is_monotonic_and_normalized(self):
        x, y = ecdf([3.0, 1.0, 2.0])
        np.testing.assert_array_equal(x, np.asarray([1.0, 2.0, 3.0]))
        self.assertTrue(np.all(np.diff(y) > 0))
        self.assertEqual(float(y[-1]), 1.0)

    def test_figure_selection_parser(self):
        self.assertEqual(_parse_numbers("1,3,10"), {1, 3, 10})
        self.assertIsNone(_parse_numbers(None))
        with self.assertRaisesRegex(ValueError, "unknown figure"):
            _parse_numbers("11")


if __name__ == "__main__":
    unittest.main()
