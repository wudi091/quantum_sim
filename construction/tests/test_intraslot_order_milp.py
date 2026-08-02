from __future__ import annotations

import unittest

from construction.intraslot_order_milp import solve_counterexample_milp
from construction.reproduce_intraslot_generation import run_case


class IntraSlotOrderMilpTests(unittest.TestCase):
    def test_global_optimum_releases_c_first(self) -> None:
        result = solve_counterexample_milp()

        self.assertEqual(result.completed_requests, 3)
        self.assertEqual(
            set(result.optimal_orders),
            {
                ("C", "B", "D"),
                ("C", "D", "B"),
            },
        )
        self.assertIn(result.selected_order, result.optimal_orders)
        self.assertEqual(
            result.waiting_completion_round,
            {"R2": 2, "R3": 3},
        )

    def test_every_order_matches_fixed_automatic_simulator(self) -> None:
        result = solve_counterexample_milp()

        for order, outcome in result.order_outcomes.items():
            with self.subTest(order=order):
                simulated = run_case(order, c_capacity=2)
                self.assertEqual(
                    outcome.completed_requests,
                    simulated.completed_count,
                )

    def test_extra_memory_removes_order_gap(self) -> None:
        result = solve_counterexample_milp(hotspot_capacity=4)

        self.assertEqual(result.completed_requests, 3)
        self.assertEqual(
            set(result.optimal_orders),
            set(result.order_outcomes),
        )


if __name__ == "__main__":
    unittest.main()
