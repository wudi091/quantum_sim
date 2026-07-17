import copy
import unittest

from qnet_core.qcast_paper.validate import (
    OFFICIAL_COMMIT,
    PAPER_TARGETS,
    spearman,
    validate_sweep_payload,
)


def paper_payload():
    return {
        "source_commit": OFFICIAL_COMMIT,
        "slots_per_point": 1000,
        "sweeps": copy.deepcopy(PAPER_TARGETS["sweeps"]),
        "recovery": copy.deepcopy(PAPER_TARGETS["recovery"]),
    }


AUTHOR_NAMES = {
    "QCAST": "Online", "QPASS_CR": "CR", "Greedy": "Greedy_H", "SLMP": "SL",
}


def paper_rows():
    rows = []
    common = {"n": 100, "p": 0.6, "q": 0.9, "k": 3, "m": 10,
              "slots": 1000, "topologies": 10, "success_pairs": 1.0}
    for parameter, sweep in PAPER_TARGETS["sweeps"].items():
        for algorithm, values in sweep["algorithms"].items():
            for x_value, throughput in zip(sweep["x"], values):
                rows.append({
                    **common, "fig": f"throughput-{parameter}", "parameter": parameter,
                    "value": x_value, "algorithm": AUTHOR_NAMES[algorithm],
                    "throughput": throughput,
                })
    for algorithm, levels in PAPER_TARGETS["recovery"].items():
        for state, throughput in levels.items():
            rows.append({
                **common, "fig": "recovery", "parameter": "recovery", "value": state,
                "algorithm": AUTHOR_NAMES[algorithm], "throughput": throughput,
            })
    return rows


class QCastSweepValidatorTests(unittest.TestCase):
    def test_exact_vector_targets_pass_all_three_judgements(self):
        report = validate_sweep_payload(paper_payload())
        self.assertTrue(report["config_pass"])
        self.assertTrue(report["trend_pass"])
        self.assertTrue(report["value_pass"])
        self.assertTrue(report["overall_pass"])
        self.assertEqual(report["target_provenance"], PAPER_TARGETS["provenance"])

    def test_uniform_ten_percent_scaling_preserves_trends_and_values(self):
        payload = paper_payload()
        for sweep in payload["sweeps"].values():
            for algorithm, values in sweep["algorithms"].items():
                sweep["algorithms"][algorithm] = [value * 1.1 for value in values]
        for levels in payload["recovery"].values():
            levels["on"] *= 1.1
            levels["off"] *= 1.1
        report = validate_sweep_payload(payload)
        self.assertTrue(report["trend_pass"])
        self.assertTrue(report["value_pass"])

    def test_official_row_list_and_author_algorithm_names_are_supported(self):
        rows = paper_rows()
        report = validate_sweep_payload(rows)
        self.assertEqual(report["input_schema"], "official_row_list")
        self.assertTrue(report["overall_pass"])
        self.assertIsNone(report["config"]["source_commit_matches"])
        self.assertEqual(report["config"]["row_protocol"]["row_count"], len(rows))

    def test_row_list_uses_per_row_topology_protocol_for_config_pass(self):
        rows = paper_rows()
        rows[0]["topologies"] = 1
        report = validate_sweep_payload(rows)
        self.assertFalse(report["config_pass"])
        self.assertTrue(any("topologies" in error for error in report["config"]["errors"]))

    def test_wrong_protocol_and_length_fail_configuration_without_crashing(self):
        payload = paper_payload()
        payload["slots_per_point"] = 100
        payload["sweeps"]["q"]["algorithms"]["QCAST"].pop()
        report = validate_sweep_payload(payload)
        self.assertFalse(report["config_pass"])
        self.assertFalse(report["trend_pass"])
        self.assertFalse(report["value_pass"])
        self.assertGreaterEqual(len(report["config"]["errors"]), 2)

    def test_reversed_p_curve_fails_trend_and_value(self):
        payload = paper_payload()
        payload["sweeps"]["p"]["algorithms"]["QCAST"].reverse()
        report = validate_sweep_payload(payload)
        curve = report["sweeps"]["p"]["algorithms"]["QCAST"]
        self.assertLess(curve["spearman"], 0.0)
        self.assertFalse(curve["trend_pass"])
        self.assertFalse(report["trend_pass"])
        self.assertFalse(report["value_pass"])

    def test_recovery_regression_is_reported_separately(self):
        payload = paper_payload()
        payload["recovery"]["QCAST"] = {"on": 10.0, "off": 12.0}
        report = validate_sweep_payload(payload)
        self.assertFalse(report["recovery"]["QCAST"]["trend_pass"])
        self.assertFalse(report["trend_pass"])

    def test_ordering_mismatch_is_visible_per_point(self):
        payload = paper_payload()
        payload["sweeps"]["n"]["algorithms"]["SLMP"][0] = 100.0
        report = validate_sweep_payload(payload)
        first = report["sweeps"]["n"]["ordering"]["points"][0]
        self.assertFalse(first["matches"])
        self.assertEqual(first["observed"][0], "SLMP")
        self.assertAlmostEqual(report["sweeps"]["n"]["ordering"]["match_rate"], 0.8)

    def test_spearman_supports_ties_without_scipy(self):
        self.assertAlmostEqual(spearman([1, 2, 2, 4], [10, 20, 20, 40]), 1.0)


if __name__ == "__main__":
    unittest.main()
