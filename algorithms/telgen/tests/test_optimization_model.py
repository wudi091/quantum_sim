import unittest

import numpy as np

from algorithms.telgen import (
    build_stage_one_model,
    build_stage_two_model,
    expand_construction_candidates,
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
                RequestSpec("r0", 0, 1, ttl=1),
                RequestSpec("r1", 0, 1, ttl=1),
            ),
            horizon=1,
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

    def test_stage_one_contains_request_and_resource_time_constraints(self):
        model = build_stage_one_model(self.variables, self.capacities)
        kinds = {item.kind for item in model.ub_constraints}
        self.assertEqual(kinds, {"request", "resource_time"})
        self.assertTrue(np.allclose(
            model.objective,
            -np.asarray([
                item.expected_success_probability for item in self.variables
            ]),
        ))

    def test_stage_two_fixes_expected_throughput(self):
        completed_mass = self.variables[0].expected_success_probability
        model = build_stage_two_model(
            self.variables,
            self.capacities,
            completed_mass,
        )
        expected = np.asarray([
            item.expected_success_probability for item in self.variables
        ])
        self.assertTrue(np.allclose(model.a_eq.toarray()[0], expected))
        self.assertEqual(model.b_eq.tolist(), [completed_mass])
        self.assertEqual(model.eq_constraints[0].kind, "throughput_equality")

    def test_reserved_usage_reduces_resource_rhs(self):
        model = build_stage_one_model(
            self.variables,
            self.capacities,
            reserved_usage={("link:0-1", 0): 1},
        )
        link_row = next(
            item
            for item in model.ub_constraints
            if item.resource_id == "link:0-1" and item.slot == 0
        )
        self.assertEqual(link_row.rhs, 1.0)


if __name__ == "__main__":
    unittest.main()
