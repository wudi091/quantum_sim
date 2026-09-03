import unittest

import numpy as np
from scipy.optimize import linprog

from algorithms.telgen import (
    build_delay_model,
    expand_construction_candidates,
    evaluate_expected_censored_delay,
)
from qnet_core.construction_catalog import build_route_construction_catalogue
from qnet_core.planning_spec import PlanningSpec, RequestSpec


class OptimizationModelTests(unittest.TestCase):
    def setUp(self):
        self.spec = PlanningSpec(
            seed=1,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(
                RequestSpec("r0", 0, 1, ttl=4),
                RequestSpec("r1", 0, 1, ttl=4),
            ),
            horizon=4,
        )
        self.capacities = {
            "link:0-1": 2,
            "genlane:0-1": 2,
            "memory:0": 4,
            "memory:1": 4,
            "bsm:0": 1,
            "bsm:1": 1,
        }
        candidates = build_route_construction_catalogue(
            self.spec,
            candidate_count=1,
            construction_kinds=("balanced",),
            purification_kinds=("none",),
        )
        self.variables = expand_construction_candidates(
            self.spec,
            candidates,
            self.capacities,
        ).variables

    def test_delay_model_contains_request_and_resource_time_constraints(self):
        model = build_delay_model(
            self.variables,
            self.capacities,
            request_censoring_latencies={"r0": 4.0, "r1": 4.0},
        )
        kinds = {item.kind for item in model.ub_constraints}
        self.assertEqual(kinds, {"request", "resource_time"})
        expected = np.asarray([
            item.expected_success_probability
            * (item.completion_latency - 4.0)
            for item in self.variables
        ])
        self.assertTrue(np.allclose(model.objective, expected))
        self.assertEqual(model.name, "minimize_expected_censored_completion_latency")
        self.assertEqual(model.objective_constant, 8.0)
        self.assertEqual(len(model.eq_constraints), 0)
        self.assertAlmostEqual(
            evaluate_expected_censored_delay(
                model,
                np.zeros(len(self.variables), dtype=float),
            ),
            8.0,
        )

    def test_unserved_requests_are_represented_by_the_constant_penalty(self):
        model = build_delay_model(
            (),
            self.capacities,
            request_censoring_latencies={"r0": 4.0, "r1": 2.0},
        )
        self.assertEqual(model.objective_constant, 6.0)
        self.assertEqual(model.request_censoring_latency_map, {
            "r0": 4.0,
            "r1": 2.0,
        })

    def test_reserved_usage_reduces_resource_rhs(self):
        model = build_delay_model(
            self.variables,
            self.capacities,
            reserved_usage={("link:0-1", 0): 1},
            request_censoring_latencies={"r0": 4.0, "r1": 4.0},
        )
        link_row = next(
            item
            for item in model.ub_constraints
            if item.resource_id == "link:0-1" and item.slot == 0
        )
        self.assertEqual(link_row.rhs, 1.0)

    def test_lp_solution_matches_the_single_stage_objective(self):
        model = build_delay_model(
            self.variables,
            self.capacities,
            request_censoring_latencies={"r0": 4.0, "r1": 4.0},
        )
        result = linprog(
            model.objective,
            A_ub=model.a_ub,
            b_ub=model.b_ub,
            bounds=[(0.0, 1.0)] * len(model.variable_ids),
            method="highs",
        )
        self.assertTrue(result.success, result.message)
        self.assertIsNotNone(result.x)
        point = np.asarray(result.x, dtype=float)
        self.assertLessEqual(
            float(np.max(model.a_ub @ point - model.b_ub)),
            1e-8,
        )
        self.assertAlmostEqual(
            evaluate_expected_censored_delay(model, point),
            model.objective_constant + float(model.objective @ point),
            places=8,
        )
        # The direct LP must choose both independent requests at the earliest
        # feasible completion in this non-conflicting fixture.
        self.assertAlmostEqual(float(np.sum(point)), 2.0, places=8)


if __name__ == "__main__":
    unittest.main()
