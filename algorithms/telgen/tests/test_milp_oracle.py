import unittest

import numpy as np

from algorithms.telgen import (
    ConstructionAwareMILPOracle,
    DiscreteStageResult,
    NominalConstructionSchedule,
    ResourceSlotUsage,
    TimeExpandedCandidate,
    expand_construction_candidates,
    has_numerically_zero_mip_gap,
    is_numerically_optimal_stage,
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
        self.assertFalse(has_numerically_zero_mip_gap(1e-6))
        self.assertFalse(has_numerically_zero_mip_gap(None))
        stage = DiscreteStageResult(
            stage_name="test",
            success=True,
            status=0,
            message="optimal",
            primal=np.zeros(1, dtype=float),
            objective_value=1.0,
            mip_gap=1.8577558830020406e-12,
            mip_node_count=0,
            mip_dual_bound=1.0 + 3e-14,
        )
        self.assertTrue(is_numerically_optimal_stage(stage))
        self.assertFalse(is_numerically_optimal_stage(DiscreteStageResult(
            **{
                **stage.__dict__,
                "mip_dual_bound": stage.objective_value + 1e-4,
            }
        )))
        self.assertFalse(is_numerically_optimal_stage(DiscreteStageResult(
            **{**stage.__dict__, "status": 1, "message": "time limit"}
        )))

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
        )
        self.assertEqual(solution.completed_request_count, 1)
        resource_row = next(
            descriptor
            for descriptor in solution.stage_one_model.ub_constraints
            if descriptor.kind == "resource_time"
        )
        self.assertEqual(resource_row.rhs, 1.0)


if __name__ == "__main__":
    unittest.main()
