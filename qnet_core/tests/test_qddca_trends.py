import unittest
from unittest.mock import patch

from qnet_core.qddca_trends import OFFICIAL, _meets_threshold, _shape_stats, run_suite


class QDDCATrendTests(unittest.TestCase):
    def test_shape_stats_reports_direction_and_normalized_bias(self):
        stats = _shape_stats([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        self.assertAlmostEqual(stats["spearman"], 1.0)
        self.assertAlmostEqual(stats["direction_agreement"], 1.0)
        self.assertAlmostEqual(stats["normalized_bias"], 0.0)
        self.assertAlmostEqual(stats["normalized_mae"], 0.0)

    def test_suite_rejects_empty_seed_set(self):
        with self.assertRaises(ValueError):
            run_suite(0)

    def test_rank_threshold_tolerates_float_roundoff_only(self):
        self.assertTrue(_meets_threshold(0.7999999999999999, 0.8))
        self.assertFalse(_meets_threshold(0.799, 0.8))

    def test_window_throughput_correlation_is_required_for_overall_pass(self):
        window_throughput = {1: 1.0, 5: 5.0, 10: 4.0, 20: 3.0, 30: 2.0}
        window_drop = {1: 0.0, 5: 1.0, 10: 2.0, 20: 3.0, 30: 4.0}

        def fake_case(window, max_try, reroute, seed):
            del seed
            if not reroute:
                return {"throughput": 1.0, "drop": 10.0, "cv": 1.0,
                        "completed": 1.0, "time": 1.0}
            if window == OFFICIAL["retry"]["window"]:
                throughput = {1: 1.0, 5: 3.0, 10: 4.0}[max_try]
                drop = {1: 3.0, 5: 2.0, 10: 1.0}[max_try]
                return {"throughput": throughput, "drop": drop, "cv": 0.3,
                        "completed": throughput, "time": 1.0}
            index = OFFICIAL["window"]["x"].index(window)
            return {
                "throughput": window_throughput[window],
                "drop": window_drop[window],
                "cv": OFFICIAL["window"]["cv"][index],
                "completed": window_throughput[window],
                "time": 1.0,
            }

        with patch("qnet_core.qddca_trends.run_case", side_effect=fake_case):
            validation = run_suite(1)["validation"]
        self.assertLess(validation["window_throughput_spearman"], 0.8)
        self.assertFalse(validation["overall_pass"])


if __name__ == "__main__":
    unittest.main()
