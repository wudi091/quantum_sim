import tempfile
import unittest
from pathlib import Path

from algorithms.baselines.online import (
    OnlineBaselineConfig,
    run_online_baseline,
    save_online_baseline_result,
)
from algorithms.baselines.planner import BASELINE_ALGORITHMS
from qnet_core.planning_spec import RequestSpec
from qnet_core.spec import EpisodeSpec, PhysicalConfig


def deterministic_physical(**overrides):
    values = dict(
        generation_probability=1.0,
        swap_probability=1.0,
        detector_efficiency=1.0,
        bsm_success_probability=1.0,
        quantum_distance_m=1.0,
        slot_duration_ps=1_000_000,
        memory_capacity=2,
        node_memory_capacity=4,
        max_width=1,
    )
    values.update(overrides)
    return PhysicalConfig(**values)


class OnlineBaselineTests(unittest.TestCase):
    def test_all_baselines_hide_future_requests_and_obey_the_schedule(self):
        spec = EpisodeSpec(
            seed=2200,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2), (0, 2)),
            requests=(
                RequestSpec("r0", 0, 2, arrival=0, ttl=8),
                RequestSpec("r1", 0, 2, arrival=3, ttl=5),
            ),
            horizon=8,
            physical=deterministic_physical(),
        )
        for algorithm in BASELINE_ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                result = run_online_baseline(
                    spec,
                    OnlineBaselineConfig(
                        algorithm=algorithm,
                        decision_interval=4,
                        path_candidate_count=2,
                    ),
                )
                self.assertEqual(
                    [item.decision_slot for item in result.decisions],
                    [0, 4],
                )
                self.assertEqual(
                    result.decisions[0].visible_request_ids,
                    ("r0",),
                )
                self.assertNotIn(
                    "r1",
                    result.decisions[0].eligible_request_ids,
                )
                self.assertEqual(result.violations, ())
                self.assertEqual(
                    result.metrics["schedule_violation_count"],
                    0.0,
                )
                self.assertEqual(result.metrics["completed_requests"], 2.0)

    def test_best_fifo_history_is_carried_across_decision_windows(self):
        spec = EpisodeSpec(
            seed=2201,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(
                RequestSpec("r0", 0, 1, arrival=0, ttl=6),
                RequestSpec("r1", 0, 1, arrival=2, ttl=4),
            ),
            horizon=6,
            physical=deterministic_physical(),
        )
        result = run_online_baseline(
            spec,
            OnlineBaselineConfig(
                algorithm="best_fifo",
                decision_interval=2,
                path_candidate_count=1,
            ),
        )
        self.assertIsNone(
            result.decisions[0].planner_state_average_before
        )
        learned_average = (
            result.decisions[0].planner_state_average_after
        )
        self.assertIsNotNone(learned_average)
        self.assertEqual(
            result.decisions[1].planner_state_average_before,
            learned_average,
        )

    def test_result_writer_creates_versioned_and_latest_files(self):
        spec = EpisodeSpec(
            seed=2202,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r0", 0, 1, ttl=2),),
            horizon=2,
            physical=deterministic_physical(),
        )
        result = run_online_baseline(
            spec,
            OnlineBaselineConfig(
                algorithm="greedy",
                decision_interval=1,
                path_candidate_count=1,
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            paths = save_online_baseline_result(result, temporary)
            for path in (
                paths.json_path,
                paths.csv_path,
                paths.latest_json_path,
                paths.latest_csv_path,
            ):
                self.assertTrue(Path(path).is_file())


if __name__ == "__main__":
    unittest.main()
