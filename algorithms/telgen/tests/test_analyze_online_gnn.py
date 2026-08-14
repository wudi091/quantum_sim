import unittest

from algorithms.telgen.analyze_online_gnn import (
    _additional_hard_gates,
    _convert_payload,
)


class OnlineGNNAnalysisTests(unittest.TestCase):
    def test_converts_audited_gnn_report_for_paired_analysis(self):
        payload = {
            "schema_version": 1,
            "experiment": "paired_online_gnn_milp_qcast",
            "comparison_contract": {
                "paired_episode_spec": True,
                "independent_persistent_executors": True,
                "gnn_calls_milp_online": False,
                "qcast_uses_gnn_or_milp": False,
            },
            "configuration": {"skip_milp": True},
            "milp_config": None,
            "scenario": {"topology_mode": "waxman"},
            "trials": [{
                "seed": 1,
                "episode": {"seed": 1},
                "methods": {
                    "gnn": {"metrics": {}, "violations": []},
                    "qcast": {"metrics": {}, "violations": []},
                },
            }],
        }
        converted = _convert_payload(payload)
        self.assertEqual(converted["trials"][0]["telgen"], payload["trials"][0]["methods"]["gnn"])
        self.assertEqual(converted["trials"][0]["qcast"], payload["trials"][0]["methods"]["qcast"])

    def test_rejects_report_without_exact_episode(self):
        payload = {
            "schema_version": 1,
            "experiment": "paired_online_gnn_milp_qcast",
            "comparison_contract": {
                "paired_episode_spec": True,
                "independent_persistent_executors": True,
                "gnn_calls_milp_online": False,
                "qcast_uses_gnn_or_milp": False,
            },
            "configuration": {"skip_milp": True},
            "milp_config": None,
            "scenario": {},
            "trials": [{"seed": 1, "methods": {}}],
        }
        with self.assertRaisesRegex(ValueError, "EpisodeSpec"):
            _convert_payload(payload)

    def test_nominal_slot_overrun_is_not_an_unsafe_schedule_failure(self):
        method = {
            "metrics": {},
            "violations": [{"code": "slot_completion_overrun"}],
        }
        totals = _additional_hard_gates({
            "case": {"trials": [{
                "methods": {"gnn": method, "qcast": method},
            }]},
        })
        self.assertEqual(totals["gnn_unsafe_schedule_violation_count"], 0.0)
        self.assertEqual(totals["qcast_unsafe_schedule_violation_count"], 0.0)
        self.assertEqual(
            totals["gnn_nominal_completion_overrun_count"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
