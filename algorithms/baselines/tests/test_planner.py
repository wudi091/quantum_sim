import unittest

from algorithms.baselines.planner import (
    BASELINE_ALGORITHMS,
    plan_baseline_window,
)
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


class BaselinePlannerTests(unittest.TestCase):
    def test_every_baseline_returns_a_feasible_shared_resource_plan(self):
        spec = EpisodeSpec(
            seed=2100,
            nodes=(0, 1, 2, 3),
            edges=((0, 1), (1, 3), (0, 2), (2, 3)),
            requests=(
                RequestSpec("r0", 0, 3, ttl=6),
                RequestSpec("r1", 0, 3, ttl=6),
                RequestSpec("r2", 1, 2, ttl=6),
            ),
            horizon=6,
            physical=deterministic_physical(),
        )
        for algorithm in BASELINE_ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                record = plan_baseline_window(
                    spec,
                    algorithm=algorithm,
                    window_start_slot=0,
                    window_end_slot=3,
                    completion_end_slot=6,
                    path_candidate_count=4,
                )
                self.assertTrue(record.solution.feasibility.feasible)
                request_ids = [
                    variable.request_id
                    for variable in record.solution.selected_variables
                ]
                self.assertEqual(len(request_ids), len(set(request_ids)))

    def test_greedy_and_qpass_prefer_the_direct_route(self):
        spec = EpisodeSpec(
            seed=2101,
            nodes=(0, 1, 2),
            edges=((0, 2), (0, 1), (1, 2)),
            requests=(RequestSpec("r0", 0, 2, ttl=5),),
            horizon=5,
            physical=deterministic_physical(
                generation_probability=0.8,
                swap_probability=0.9,
            ),
        )
        for algorithm in ("greedy", "qpass"):
            with self.subTest(algorithm=algorithm):
                record = plan_baseline_window(
                    spec,
                    algorithm=algorithm,
                    window_start_slot=0,
                    window_end_slot=5,
                    path_candidate_count=2,
                )
                self.assertEqual(
                    record.solution.selected_variables[0].route_nodes,
                    (0, 2),
                )

    def test_qpath_and_qleap_use_purification_only_when_required(self):
        spec = EpisodeSpec(
            seed=2102,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(
                RequestSpec(
                    "r0",
                    0,
                    1,
                    ttl=8,
                    required_fidelity=0.8,
                ),
            ),
            horizon=8,
            physical=deterministic_physical(
                initial_fidelity=0.8,
                swap_degradation=1.0,
                node_memory_capacity=8,
            ),
        )
        for algorithm in ("qpath", "qleap"):
            with self.subTest(algorithm=algorithm):
                record = plan_baseline_window(
                    spec,
                    algorithm=algorithm,
                    window_start_slot=0,
                    window_end_slot=8,
                    path_candidate_count=1,
                )
                self.assertEqual(
                    record.solution.selected_variables[0].purification_kind,
                    "elementary_once",
                )
                self.assertGreaterEqual(
                    record.solution.selected_variables[0].expected_fidelity,
                    0.8,
                )

    def test_best_fifo_can_serve_a_later_cheap_request_before_fifo_head(self):
        spec = EpisodeSpec(
            seed=2103,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(
                RequestSpec("r0", 0, 2, ttl=4),
                RequestSpec("r1", 0, 1, ttl=4),
            ),
            horizon=4,
            physical=deterministic_physical(
                generation_probability=0.8,
                swap_probability=0.9,
            ),
        )
        strict = plan_baseline_window(
            spec,
            algorithm="strict_fifo",
            window_start_slot=0,
            window_end_slot=1,
            completion_end_slot=4,
            path_candidate_count=1,
        )
        best = plan_baseline_window(
            spec,
            algorithm="best_fifo",
            window_start_slot=0,
            window_end_slot=1,
            completion_end_slot=4,
            path_candidate_count=1,
        )
        self.assertEqual(
            strict.solution.selected_variables[0].request_id,
            "r0",
        )
        self.assertEqual(
            best.solution.selected_variables[0].request_id,
            "r1",
        )
        self.assertLess(
            best.selected_path_costs[0],
            strict.selected_path_costs[0],
        )


if __name__ == "__main__":
    unittest.main()
