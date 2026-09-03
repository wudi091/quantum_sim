import unittest
from dataclasses import replace

import numpy as np

from algorithms.telgen import (
    ConstructionAwareMILPOracle,
    DiscreteSolveResult,
    NominalConstructionSchedule,
    ResourceSlotUsage,
    TimeExpandedCandidate,
    expand_construction_candidates,
    has_numerically_zero_mip_gap,
    is_numerically_optimal_result,
)
from qnet_core.construction_catalog import build_route_construction_catalogue
from qnet_core.planning_spec import PlanningSpec, RequestSpec


def three_request_bases():
    spec = PlanningSpec(
        seed=9,
        nodes=(0, 1, 2, 3, 4, 5),
        edges=((0, 1), (2, 3), (4, 5)),
        requests=(
            RequestSpec("r0", 0, 1, ttl=1),
            RequestSpec("r1", 2, 3, ttl=1),
            RequestSpec("r2", 4, 5, ttl=1),
        ),
        horizon=1,
    )
    candidates = build_route_construction_catalogue(
        spec,
        candidate_count=1,
        construction_kinds=("balanced",),
    )
    return {candidate.request_id: candidate for candidate in candidates}


def manual_variable(base, resources):
    usage = tuple(sorted(
        ResourceSlotUsage(resource, 0, 1)
        for resource in resources
    ))
    schedule = NominalConstructionSchedule(
        candidate_id=base.candidate_id,
        operation_slots=((base.dag.operations[0].op_id, 0),),
        duration_slots=1,
        resource_usage=usage,
    )
    return TimeExpandedCandidate(
        variable_id=f"{base.request_id}@0",
        base_candidate=base,
        start_slot=0,
        completion_slot=1,
        completion_latency=1,
        expected_fidelity=1.0,
        resource_usage=usage,
        nominal_schedule=schedule,
    )


class MILPOracleTests(unittest.TestCase):
    def test_numerical_gap_certification_accepts_only_optimal_roundoff(self):
        self.assertTrue(has_numerically_zero_mip_gap(1.8577558830020406e-12))
        self.assertTrue(has_numerically_zero_mip_gap(2.519004916667029e-8))
        self.assertFalse(has_numerically_zero_mip_gap(1e-6))
        self.assertFalse(has_numerically_zero_mip_gap(None))
        result = DiscreteSolveResult(
            solve_name="test",
            success=True,
            status=0,
            message="optimal",
            primal=np.zeros(1, dtype=float),
            objective_value=1.0,
            mip_gap=1.8577558830020406e-12,
            mip_node_count=0,
            mip_dual_bound=1.0 + 3e-14,
        )
        self.assertTrue(is_numerically_optimal_result(result))
        self.assertFalse(is_numerically_optimal_result(DiscreteSolveResult(
            **{
                **result.__dict__,
                "mip_dual_bound": result.objective_value + 1e-4,
            }
        )))
        self.assertFalse(is_numerically_optimal_result(DiscreteSolveResult(
            **{**result.__dict__, "status": 1, "message": "time limit"}
        )))

    def test_numerical_certification_accepts_highs_optimal_roundoff(self):
        result = DiscreteSolveResult(
            solve_name="minimize_expected_censored_completion_latency",
            success=True,
            status=0,
            message="Optimization terminated successfully. (HiGHS Status 7: Optimal)",
            primal=np.zeros(1, dtype=float),
            objective_value=33.74161920000001,
            mip_gap=2.519004916667029e-8,
            mip_node_count=0,
            mip_dual_bound=33.741618350046814,
        )
        self.assertTrue(is_numerically_optimal_result(result))

    def test_triangle_set_packing_selects_one_feasible_request(self):
        bases = three_request_bases()
        variables = (
            manual_variable(bases["r0"], ("a", "b")),
            manual_variable(bases["r1"], ("b", "c")),
            manual_variable(bases["r2"], ("a", "c")),
        )
        solution = ConstructionAwareMILPOracle().solve(
            variables,
            {"a": 1, "b": 1, "c": 1},
            request_censoring_latencies={
                "r0": 2.0,
                "r1": 2.0,
                "r2": 2.0,
            },
        )
        self.assertEqual(solution.completed_request_count, 1)
        self.assertEqual(len(solution.selected_variables), 1)
        self.assertEqual(sum(solution.final_values.values()), 1)

    def test_oracle_consumes_real_time_expansion(self):
        spec = PlanningSpec(
            seed=10,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(
                RequestSpec("r0", 0, 1, ttl=1),
                RequestSpec("r1", 0, 1, ttl=1),
            ),
            horizon=1,
        )
        capacities = {
            "link:0-1": 1,
            "genlane:0-1": 1,
            "memory:0": 2,
            "memory:1": 2,
            "bsm:0": 1,
            "bsm:1": 1,
        }
        candidates = build_route_construction_catalogue(
            spec,
            candidate_count=1,
            construction_kinds=("balanced",),
        )
        expansion = expand_construction_candidates(
            spec,
            candidates,
            capacities,
        )
        solution = ConstructionAwareMILPOracle().solve(
            expansion,
            capacities,
            request_censoring_latencies={"r0": 2.0, "r1": 2.0},
        )
        self.assertEqual(solution.completed_request_count, 1)
        self.assertEqual(len(solution.selected_variables), 1)

    def test_running_reservation_reduces_available_capacity(self):
        bases = three_request_bases()
        variables = (
            manual_variable(bases["r0"], ("shared",)),
            manual_variable(bases["r1"], ("shared",)),
        )
        solution = ConstructionAwareMILPOracle().solve(
            variables,
            {"shared": 2},
            reserved_usage={("shared", 0): 1},
            request_censoring_latencies={"r0": 2.0, "r1": 2.0},
        )
        self.assertEqual(solution.completed_request_count, 1)
        resource_row = next(
            descriptor
            for descriptor in solution.model.ub_constraints
            if descriptor.kind == "resource_time"
        )
        self.assertEqual(resource_row.rhs, 1.0)

    def test_oracle_exposes_one_solve_and_actual_censored_delay(self):
        bases = three_request_bases()
        variables = (
            manual_variable(bases["r0"], ("a",)),
            manual_variable(bases["r1"], ("b",)),
        )
        solution = ConstructionAwareMILPOracle().solve(
            variables,
            {"a": 1, "b": 1},
            request_censoring_latencies={"r0": 4.0, "r1": 4.0},
        )
        self.assertEqual(solution.result.solve_name,
                         "minimize_expected_censored_completion_latency")
        self.assertEqual(solution.completed_request_count, 2)
        self.assertAlmostEqual(solution.total_completion_latency, 2.0)
        self.assertEqual(
            set(solution.__dataclass_fields__),
            {
                "variables",
                "model",
                "result",
                "completed_request_count",
                "expected_completed_request_mass",
                "total_completion_latency",
            },
        )

    def test_delay_objective_prefers_earlier_completion_under_conflict(self):
        bases = three_request_bases()
        early = manual_variable(bases["r0"], ("shared",))
        late = replace(
            manual_variable(bases["r1"], ("shared",)),
            completion_slot=3,
            completion_latency=3,
        )
        solution = ConstructionAwareMILPOracle().solve(
            (early, late),
            {"shared": 1},
            request_censoring_latencies={"r0": 5.0, "r1": 5.0},
        )
        self.assertEqual(
            {variable.request_id for variable in solution.selected_variables},
            {"r0"},
        )
        self.assertAlmostEqual(solution.total_completion_latency, 6.0)


if __name__ == "__main__":
    unittest.main()
