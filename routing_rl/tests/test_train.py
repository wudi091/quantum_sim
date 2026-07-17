from __future__ import annotations

import unittest

from routing_rl.train import make_config, parse_args


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
        self.assertFalse(config.anneal_learning_rate)
        self.assertEqual(config.early_stopping_patience, 0)

    def test_learning_rate_annealing_is_opt_in(self):
        config = make_config(parse_args(["--anneal-learning-rate"]))
        self.assertTrue(config.anneal_learning_rate)

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


if __name__ == "__main__":
    unittest.main()
