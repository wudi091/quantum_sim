from __future__ import annotations

import unittest

from routing_rl.train import make_config, parse_args
from routing_rl.large_scale import (
    SETTINGS,
    build_args as build_large_scale_args,
    prepare_initialization,
)
from routing_rl.small_scale import (
    SETTINGS as SMALL_SETTINGS,
    assess_direction,
    build_args as build_small_scale_args,
)


class TrainConfigTest(unittest.TestCase):
    def test_default_training_is_single_full_range_stage(self):
        args = parse_args([])
        config = make_config(args)
        self.assertEqual(len(config.curriculum), 1)
        stage = config.curriculum[0]
        self.assertEqual(stage.name, "full")
        self.assertEqual((stage.min_hops, stage.max_hops), (2, 50))
        self.assertEqual(stage.max_requests, 100)
        self.assertEqual(config.total_updates, 1000)
        self.assertEqual(args.request_ttl, 64)
        self.assertEqual(args.swap_probability, 0.95)
        self.assertIsNone(args.topology_nodes)
        self.assertEqual(args.waxman_alpha, 0.05)
        self.assertEqual(args.waxman_beta, 0.02)
        self.assertEqual(config.reward.potential_coef, 0.1)
        self.assertEqual(config.reward.completion_bonus, 1.0)
        self.assertEqual(config.reward.failure_coef, 0.1)
        self.assertEqual(config.gamma, 0.99)
        self.assertEqual(config.value_coef, 0.5)
        self.assertFalse(config.anneal_learning_rate)
        self.assertEqual(config.early_stopping_patience, 0)
        self.assertEqual(args.high_hop_evaluation_episodes, 0)
        self.assertEqual(args.high_hop_min_hops, 41)
        self.assertFalse(args.select_high_hop)

    def test_high_hop_selection_flags_are_configurable(self):
        args = parse_args([
            "--high-hop-evaluation-episodes", "7",
            "--high-hop-min-hops", "37",
            "--select-high-hop",
        ])
        self.assertEqual(args.high_hop_evaluation_episodes, 7)
        self.assertEqual(args.high_hop_min_hops, 37)
        self.assertTrue(args.select_high_hop)

    def test_learning_rate_annealing_is_opt_in(self):
        config = make_config(parse_args(["--anneal-learning-rate"]))
        self.assertTrue(config.anneal_learning_rate)

    def test_gamma_is_configurable(self):
        config = make_config(parse_args(["--gamma", "0.97"]))
        self.assertEqual(config.gamma, 0.97)

    def test_value_coefficient_is_configurable(self):
        config = make_config(parse_args(["--value-coef", "0.1"]))
        self.assertEqual(config.value_coef, 0.1)

    def test_early_stopping_patience_is_configurable(self):
        config = make_config(parse_args(["--early-stopping-patience", "4"]))
        self.assertEqual(config.early_stopping_patience, 4)

    def test_legacy_curriculum_remains_opt_in(self):
        args = parse_args(["--curriculum"])
        config = make_config(args)
        self.assertEqual([stage.name for stage in config.curriculum], ["short", "medium", "long"])

    def test_smoke_keeps_single_stage_and_one_update(self):
        args = parse_args(["--smoke", "--min-hops", "2", "--max-hops", "50"])
        config = make_config(args)
        self.assertEqual(len(config.curriculum), 1)
        self.assertEqual(config.curriculum[0].updates, 1)
        self.assertLessEqual(config.rollout_steps, 64)
        self.assertLessEqual(config.minibatch_size, 32)

    def test_large_scale_preset_is_formal_full_range_run(self):
        args = build_large_scale_args(SETTINGS)
        config = make_config(args)
        stage = config.curriculum[0]
        self.assertEqual((stage.min_hops, stage.max_hops), (2, 50))
        self.assertEqual(stage.max_requests, 20)
        self.assertEqual(config.total_updates, 1000)
        self.assertEqual(config.rollout_steps, 512)
        self.assertEqual(config.ppo_epochs, 4)
        self.assertEqual(config.entropy_coef, 0.001)
        self.assertEqual(args.topology_nodes, 200)
        self.assertEqual(args.high_hop_evaluation_episodes, 10)
        self.assertEqual(config.early_stopping_patience, 10)
        self.assertEqual(config.early_stopping_min_updates, 300)
        self.assertIsNone(args.init_checkpoint)
        self.assertTrue(config.anneal_learning_rate)

    def test_large_scale_missing_warm_start_falls_back_to_scratch(self):
        args = build_large_scale_args(SETTINGS)
        args.init_checkpoint = args.output / "missing.pt"
        self.assertFalse(prepare_initialization(args))
        self.assertIsNone(args.init_checkpoint)

    def test_small_scale_pilot_is_full_range_scratch_without_curriculum(self):
        args = build_small_scale_args(SMALL_SETTINGS)
        config = make_config(args)
        stage = config.curriculum[0]
        self.assertEqual((stage.min_hops, stage.max_hops), (2, 50))
        self.assertEqual(stage.max_requests, 5)
        self.assertEqual(config.total_updates, 30)
        self.assertFalse(args.curriculum)
        self.assertIsNone(args.init_checkpoint)
        self.assertEqual(config.rollout_steps, 128)
        self.assertEqual(config.ppo_epochs, 4)

    def test_direction_gate_requires_overall_gain_and_stability(self):
        initial = {
            "pair_throughput": 0.05,
            "completion_rate": 0.20,
            "high_hop_completion_rate": 0.01,
        }
        learned = {
            "pair_throughput": 0.07,
            "completion_rate": 0.27,
            "high_hop_completion_rate": 0.01,
        }
        history = [
            {"evaluation": {"pair_throughput": 0.065}},
            {"evaluation": {"pair_throughput": 0.070}},
            {"evaluation": {"pair_throughput": 0.068}},
        ]
        report = assess_direction(initial, learned, history)
        self.assertTrue(report["passed"])

        learned["completion_rate"] = 0.23
        report = assess_direction(initial, learned, history)
        self.assertFalse(report["passed"])

    def test_early_stopping_minimum_updates_is_configurable(self):
        config = make_config(parse_args(["--early-stopping-min-updates", "300"]))
        self.assertEqual(config.early_stopping_min_updates, 300)


if __name__ == "__main__":
    unittest.main()
