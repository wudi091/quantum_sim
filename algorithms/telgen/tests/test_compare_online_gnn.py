import unittest
from types import SimpleNamespace

from algorithms.telgen.compare_online_gnn import (
    build_parser,
    _method_payload,
    _nearest_rank_percentile,
    _resolve_construction_space,
    _routing_baseline_configs,
)
from algorithms.telgen.comparison_methods import (
    COMPARISON_PROFILES,
    FORMAL_METHOD_ORDER,
    SCALABLE_METHOD_ORDER,
    ordered_present_methods,
    validate_profile_methods,
)


class OnlineGNNComparisonConfigurationTests(unittest.TestCase):
    def test_formal_method_set_is_frozen_without_qcast(self):
        self.assertEqual(
            FORMAL_METHOD_ORDER,
            ("gnn", "milp", "qpass", "greedy"),
        )
        self.assertEqual(
            SCALABLE_METHOD_ORDER,
            ("gnn", "qpass", "greedy"),
        )
        self.assertEqual(
            ordered_present_methods({"greedy", "gnn", "qpass"}),
            SCALABLE_METHOD_ORDER,
        )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            ordered_present_methods({"gnn", "qcast"})

    def test_profiles_do_not_allow_arbitrary_method_subsets(self):
        self.assertEqual(
            set(COMPARISON_PROFILES),
            {"formal", "scalable", "construction_ablation"},
        )
        self.assertEqual(
            validate_profile_methods(
                "scalable",
                {"gnn", "qpass", "greedy"},
            ),
            SCALABLE_METHOD_ORDER,
        )
        with self.assertRaisesRegex(ValueError, "requires"):
            validate_profile_methods("scalable", {"gnn", "qpass"})

    def test_cli_defaults_to_the_full_formal_profile(self):
        args = build_parser().parse_args([
            "--checkpoint",
            "model.pt",
            "--output",
            "out",
        ])
        self.assertEqual(args.comparison_profile, "formal")
        self.assertFalse(hasattr(args, "skip_qcast"))
        self.assertFalse(hasattr(args, "skip_qpass"))
        self.assertFalse(hasattr(args, "skip_greedy"))

    def test_cli_accepts_qcast_style_fixed_arrival_rounds(self):
        args = build_parser().parse_args([
            "--checkpoint",
            "model.pt",
            "--output",
            "out",
            "--arrival-rounds",
            "20",
            "--requests-per-batch",
            "5",
        ])
        self.assertIsNone(args.requests)
        self.assertEqual(args.arrival_rounds, 20)
        self.assertEqual(args.requests_per_batch, 5)

    def test_cli_rejects_request_count_and_arrival_rounds_together(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args([
                "--checkpoint",
                "model.pt",
                "--output",
                "out",
                "--requests",
                "20",
                "--arrival-rounds",
                "4",
            ])

    def test_profile_order_is_stable_for_serialized_reports(self):
        self.assertEqual(
            list(COMPARISON_PROFILES["formal"]),
            ["gnn", "milp", "qpass", "greedy"],
        )
        self.assertEqual(
            list(COMPARISON_PROFILES["scalable"]),
            ["gnn", "qpass", "greedy"],
        )

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

    def test_routing_baselines_share_the_online_window_contract(self):
        configs = _routing_baseline_configs(
            decision_interval=4,
            path_candidate_count=4,
        )
        self.assertEqual(set(configs), {"qpass", "greedy"})
        for algorithm, config in configs.items():
            self.assertEqual(config.algorithm, algorithm)
            self.assertEqual(config.decision_interval, 4)
            self.assertEqual(config.path_candidate_count, 4)
            self.assertEqual(config.construction_kind, "left_deep")

    def test_p95_decision_time_uses_the_same_nearest_rank_rule(self):
        self.assertEqual(
            _nearest_rank_percentile([0.1, 0.4, 0.2, 0.3], 95.0),
            0.4,
        )

    def test_method_payload_records_construction_usage_without_changing_metrics(self):
        result = SimpleNamespace(
            metrics={"completed_requests": 2.0},
            decisions=(
                SimpleNamespace(decision_seconds=0.1),
                SimpleNamespace(decision_seconds=0.3),
            ),
            attempts=(
                SimpleNamespace(construction_kind="swap_tree_0", success=True),
                SimpleNamespace(construction_kind="swap_tree_1", success=False),
                SimpleNamespace(construction_kind="swap_tree_1", success=True),
            ),
            violations=(),
        )

        payload = _method_payload(result, wall_seconds=1.25)

        self.assertEqual(payload["metrics"]["completed_requests"], 2.0)
        self.assertEqual(payload["metrics"]["p95_decision_seconds"], 0.3)
        self.assertEqual(
            payload["construction_usage"]["attempt_counts"],
            {"swap_tree_0": 1, "swap_tree_1": 2},
        )
        self.assertEqual(
            payload["construction_usage"]["successful_attempt_counts"],
            {"swap_tree_0": 1, "swap_tree_1": 1},
        )


if __name__ == "__main__":
    unittest.main()
