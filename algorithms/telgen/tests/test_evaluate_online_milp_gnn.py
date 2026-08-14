import unittest

from algorithms.telgen.evaluate_online_milp_gnn import (
    _checkpoint_seen_episode_seeds,
)


class FrozenGNNEvaluationTests(unittest.TestCase):
    def test_checkpoint_seen_seeds_cover_all_recorded_splits(self):
        checkpoint = {
            "split": {
                "train_seeds": [3, 1],
                "validation_seeds": [5],
                "test_seeds": [7, 6],
            }
        }
        self.assertEqual(
            _checkpoint_seen_episode_seeds(checkpoint),
            (1, 3, 5, 6, 7),
        )

    def test_checkpoint_seen_seeds_require_complete_provenance(self):
        with self.assertRaises(ValueError):
            _checkpoint_seen_episode_seeds({"split": {"train_seeds": []}})


if __name__ == "__main__":
    unittest.main()
