from __future__ import annotations

import unittest

from qnet_core.order_core import (
    OrderAwareBatchEnv,
    OrderBatchProblem,
    OrderCoreConfig,
    OrderPlan,
    simulate_order_batch,
)
from qnet_core.order_milp import (
    DeterministicPathMilpPlanner,
    DeterministicPathOrderMilpPlanner,
    UnsupportedOrderMilpProblem,
)
from qnet_core.order_scenarios import (
    make_order_counterexample,
    make_seeded_hotspot_problem,
)


class DeterministicOrderMilpTests(unittest.TestCase):
    @staticmethod
    def _solve_same_snapshot(problem):
        snapshot = OrderAwareBatchEnv(problem).snapshot()
        before = snapshot.candidates
        path = DeterministicPathMilpPlanner()
        order = DeterministicPathOrderMilpPlanner()
        path_ids = path.select(snapshot)
        order_ids = order.select(snapshot)
        return snapshot, before, path, order, path_ids, order_ids

    def test_counterexample_milp_matches_environment_and_finds_c_first(self):
        problem = make_order_counterexample(hotspot_capacity=2)
        snapshot, before, path, order, path_ids, order_ids = (
            self._solve_same_snapshot(problem)
        )
        path_result = simulate_order_batch(problem, path_ids)
        order_result = simulate_order_batch(problem, order_ids)
        selected_main = next(
            plan for plan in problem.candidates
            if plan.plan_id in order_ids and plan.request_id == "R1"
        )

        self.assertIs(snapshot.candidates, before)
        self.assertEqual(path.last_objective, path_result.completed_count)
        self.assertEqual(order.last_objective, order_result.completed_count)
        self.assertTrue(path.last_solution.certified_optimal)
        self.assertTrue(order.last_solution.certified_optimal)
        self.assertEqual(path_result.completed_count, 2)
        self.assertEqual(order_result.completed_count, 3)
        self.assertGreaterEqual(order.last_objective, path.last_objective)
        self.assertEqual(selected_main.swap_order[0], "C")

    def test_roomy_control_has_no_order_gap(self):
        problem = make_order_counterexample(hotspot_capacity=4)
        _, _, path, order, path_ids, order_ids = self._solve_same_snapshot(
            problem
        )

        self.assertEqual(
            path.last_objective,
            simulate_order_batch(problem, path_ids).completed_count,
        )
        self.assertEqual(
            order.last_objective,
            simulate_order_batch(problem, order_ids).completed_count,
        )
        self.assertEqual(path.last_objective, 3)
        self.assertEqual(order.last_objective, 3)

    def test_seeded_hotspots_preserve_milp_dominance(self):
        for seed in range(10):
            problem = make_seeded_hotspot_problem(
                seed, hotspot_capacity=2
            )
            _, _, path, order, path_ids, order_ids = (
                self._solve_same_snapshot(problem)
            )
            with self.subTest(seed=seed):
                self.assertEqual(
                    path.last_objective,
                    simulate_order_batch(problem, path_ids).completed_count,
                )
                self.assertEqual(
                    order.last_objective,
                    simulate_order_batch(problem, order_ids).completed_count,
                )
                self.assertGreaterEqual(
                    order.last_objective, path.last_objective
                )

    def test_stochastic_problem_is_not_mislabeled_exact(self):
        problem = make_order_counterexample(
            generation_probability=0.8,
            swap_probability=0.9,
        )
        planner = DeterministicPathOrderMilpPlanner()

        with self.assertRaises(UnsupportedOrderMilpProblem):
            planner.select(OrderAwareBatchEnv(problem).snapshot())

    def test_non_certifiable_partial_generation_domain_is_rejected(self):
        problem = make_order_counterexample(hotspot_capacity=3)

        with self.assertRaisesRegex(
            UnsupportedOrderMilpProblem,
            "hotspot memory must be even",
        ):
            DeterministicPathOrderMilpPlanner().select(
                OrderAwareBatchEnv(problem).snapshot()
            )

    def test_previous_partial_generation_counterexample_is_rejected(self):
        problem = OrderBatchProblem.create(
            candidates=(
                OrderPlan(
                    "r1", "r1", ("H", "B", "C"), ("B",), priority=0
                ),
                OrderPlan(
                    "r2", "r2", ("L", "H", "R"), ("H",), priority=1
                ),
                OrderPlan(
                    "r3", "r3", ("L", "X"), (), priority=2
                ),
            ),
            node_capacity={
                "H": 2,
                "B": 2,
                "C": 1,
                "L": 1,
                "R": 1,
                "X": 1,
            },
            config=OrderCoreConfig(
                slot_duration_ps=2_000,
                generation_interval_ps=1_000,
                swap_service_ps=1_000,
                memory_reset_ps=100,
                generation_probability=1.0,
                swap_probability=1.0,
                edge_capacity=1,
                bsm_capacity_per_node=1,
            ),
            required_requests=("r1",),
            preloaded_requests=("r1",),
        )

        with self.assertRaisesRegex(
            UnsupportedOrderMilpProblem,
            "multi-swap main path",
        ):
            DeterministicPathOrderMilpPlanner().select(
                OrderAwareBatchEnv(problem).snapshot()
            )


if __name__ == "__main__":
    unittest.main()
