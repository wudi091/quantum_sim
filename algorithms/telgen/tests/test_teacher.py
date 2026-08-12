import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from algorithms.telgen import (
    ConstructionAwareLPTeacher,
    NominalConstructionSchedule,
    ResourceSlotUsage,
    TimeExpandedCandidate,
    TimeExpansionResult,
    save_teacher_solution,
)
from qnet_core.construction_catalog import build_route_construction_catalogue
from qnet_core.planning_spec import PlanningSpec, RequestSpec


def one_hop_bases():
    spec = PlanningSpec(
        seed=3,
        nodes=(0, 1, 2),
        edges=((0, 1), (1, 2)),
        requests=(
            RequestSpec("slow", 0, 1, ttl=5),
            RequestSpec("fast", 1, 2, ttl=5),
        ),
        horizon=5,
    )
    bases = build_route_construction_catalogue(
        spec,
        candidate_count=1,
        construction_kinds=("balanced",),
    )
    return {candidate.request_id: candidate for candidate in bases}


def manual_variable(
    base,
    *,
    variable_id,
    completion,
    resource="shared",
    success_probability=1.0,
):
    usage = (ResourceSlotUsage(resource, 0, 1),)
    schedule = NominalConstructionSchedule(
        candidate_id=base.candidate_id,
        operation_slots=((base.dag.operations[0].op_id, 0),),
        duration_slots=completion,
        resource_usage=usage,
    )
    return TimeExpandedCandidate(
        variable_id=variable_id,
        base_candidate=base,
        start_slot=0,
        completion_slot=completion,
        completion_latency=completion,
        expected_fidelity=1.0,
        resource_usage=usage,
        nominal_schedule=schedule,
        expected_success_probability=success_probability,
    )


class TeacherTests(unittest.TestCase):
    def test_teacher_prefers_higher_expected_completion_over_lower_latency(self):
        bases = one_hop_bases()
        variables = (
            manual_variable(
                bases["slow"],
                variable_id="reliable@0",
                completion=3,
                success_probability=0.9,
            ),
            manual_variable(
                bases["fast"],
                variable_id="fragile@0",
                completion=1,
                success_probability=0.4,
            ),
        )

        solution = ConstructionAwareLPTeacher().solve(
            variables, {"shared": 1}
        )

        self.assertAlmostEqual(solution.stage_one_completed_mass, 0.9, places=6)
        self.assertAlmostEqual(solution.completed_request_mass, 0.9, places=6)
        self.assertGreater(solution.final_values["reliable@0"], 1.0 - 1e-6)
        self.assertLess(solution.final_values["fragile@0"], 1e-6)
        self.assertAlmostEqual(solution.total_completion_latency, 2.7, places=6)

    def test_stage_two_throughput_row_uses_success_probabilities(self):
        bases = one_hop_bases()
        variables = (
            manual_variable(
                bases["slow"],
                variable_id="slow@0",
                completion=3,
                success_probability=0.8,
            ),
            manual_variable(
                bases["fast"],
                variable_id="fast@0",
                completion=1,
                success_probability=0.6,
            ),
        )

        solution = ConstructionAwareLPTeacher().solve(
            variables, {"shared": 1}
        )

        expected = np.asarray([
            variable.expected_success_probability
            for variable in solution.variables
        ])
        np.testing.assert_allclose(
            solution.stage_two_lp.a_eq.toarray()[0], expected
        )
        np.testing.assert_allclose(
            solution.stage_two_lp.objective,
            expected * np.asarray([
                variable.completion_latency for variable in solution.variables
            ]),
        )

    def test_lexicographic_teacher_keeps_throughput_then_selects_faster_request(self):
        bases = one_hop_bases()
        variables = (
            manual_variable(
                bases["slow"], variable_id="slow@0", completion=3
            ),
            manual_variable(
                bases["fast"], variable_id="fast@0", completion=1
            ),
        )
        solution = ConstructionAwareLPTeacher().solve(
            variables, {"shared": 1}
        )
        self.assertAlmostEqual(solution.stage_one_completed_mass, 1.0, places=6)
        self.assertAlmostEqual(solution.completed_request_mass, 1.0, places=6)
        values = solution.final_values
        self.assertGreater(values["fast@0"], 1.0 - 1e-6)
        self.assertLess(values["slow@0"], 1e-6)
        self.assertAlmostEqual(solution.total_completion_latency, 1.0, places=6)

    def test_teacher_records_each_ipm_stage_trajectory(self):
        bases = one_hop_bases()
        variables = (
            manual_variable(
                bases["slow"], variable_id="slow@0", completion=3
            ),
            manual_variable(
                bases["fast"], variable_id="fast@0", completion=1
            ),
        )
        solution = ConstructionAwareLPTeacher().solve(
            variables, {"shared": 1}
        )
        self.assertGreaterEqual(solution.stage_one.primal_trajectory.shape[0], 2)
        self.assertGreaterEqual(solution.stage_two.primal_trajectory.shape[0], 2)
        self.assertEqual(solution.stage_one.primal_trajectory.shape[1], 2)
        self.assertEqual(solution.stage_two.primal_trajectory.shape[1], 2)
        self.assertLess(solution.stage_two.max_violation_trajectory[-1], 1e-7)
        self.assertEqual(solution.stage_one.solver_backend, "trajectory_ipm")
        self.assertTrue(solution.stage_one.trajectory_complete)

    def test_highs_backend_preserves_the_same_lexicographic_solution(self):
        bases = one_hop_bases()
        variables = (
            manual_variable(
                bases["slow"], variable_id="slow@0", completion=3
            ),
            manual_variable(
                bases["fast"], variable_id="fast@0", completion=1
            ),
        )
        solution = ConstructionAwareLPTeacher(
            solver_backend="highs_ipm"
        ).solve(variables, {"shared": 1})

        self.assertAlmostEqual(solution.completed_request_mass, 1.0, places=6)
        self.assertGreater(solution.final_values["fast@0"], 1.0 - 1e-6)
        self.assertLess(solution.final_values["slow@0"], 1e-6)
        self.assertEqual(solution.stage_one.solver_backend, "highs_ipm")
        self.assertFalse(solution.stage_one.trajectory_complete)
        self.assertEqual(solution.stage_one.primal_trajectory.shape, (1, 2))

    def test_request_and_resource_time_rows_are_both_present(self):
        bases = one_hop_bases()
        variables = (
            manual_variable(
                bases["slow"], variable_id="slow@0", completion=3
            ),
            manual_variable(
                bases["fast"], variable_id="fast@0", completion=1
            ),
        )
        solution = ConstructionAwareLPTeacher().solve(
            variables, {"shared": 1}
        )
        kinds = {item.kind for item in solution.stage_one_lp.ub_constraints}
        self.assertEqual(kinds, {"request", "resource_time"})
        self.assertEqual(
            solution.stage_two_lp.eq_constraints[0].kind,
            "throughput_equality",
        )

    def test_empty_candidate_set_is_a_valid_zero_solution(self):
        solution = ConstructionAwareLPTeacher().solve((), {})
        self.assertEqual(solution.variables, ())
        self.assertEqual(solution.completed_request_mass, 0.0)
        self.assertEqual(solution.stage_one.primal_trajectory.shape, (1, 0))
        self.assertEqual(solution.stage_two.primal_trajectory.shape, (1, 0))

    def test_training_record_round_trips_as_npz(self):
        bases = one_hop_bases()
        variable = manual_variable(
            bases["fast"], variable_id="fast@0", completion=1
        )
        expansion = TimeExpansionResult(
            variables=(variable,),
            schedules=(variable.nominal_schedule,),
        )
        solution = ConstructionAwareLPTeacher().solve(
            expansion, {"shared": 1}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_teacher_solution(
                solution, Path(directory) / "teacher_trace.npz"
            )
            with np.load(path) as payload:
                metadata = json.loads(str(payload["metadata"]))
                self.assertEqual(
                    metadata["variables"][0]["variable_id"], "fast@0"
                )
                self.assertEqual(
                    metadata["variables"][0]["expected_success_probability"],
                    1.0,
                )
                self.assertIn("stage_one_trajectory", payload.files)
                self.assertEqual(metadata["matrix_storage"], "csr_v1")
                self.assertIn("stage_one_a_ub_data", payload.files)
                self.assertIn("stage_two_a_eq_data", payload.files)

    def test_real_time_expansion_enforces_batch_link_contention(self):
        spec = PlanningSpec(
            seed=5,
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
        bases = build_route_construction_catalogue(
            spec,
            candidate_count=1,
            construction_kinds=("balanced",),
        )
        from algorithms.telgen import expand_construction_candidates

        expansion = expand_construction_candidates(spec, bases, capacities)
        solution = ConstructionAwareLPTeacher().solve(expansion, capacities)
        self.assertEqual(len(expansion.variables), 2)
        self.assertAlmostEqual(solution.completed_request_mass, 1.0, places=7)
        self.assertLess(
            solution.stage_two.max_violation_trajectory[-1],
            1e-7,
        )

    def test_teacher_can_select_two_disjoint_same_slot_swaps(self):
        spec = PlanningSpec(
            seed=21,
            nodes=(0, 1, 2, 3, 4, 5),
            edges=((0, 1), (1, 2), (3, 4), (4, 5)),
            requests=(
                RequestSpec("r0", 0, 2, ttl=2),
                RequestSpec("r1", 3, 5, ttl=2),
            ),
            horizon=2,
        )
        capacities = {
            "link:0-1": 1,
            "link:1-2": 1,
            "link:3-4": 1,
            "link:4-5": 1,
            "genlane:0-1": 1,
            "genlane:1-2": 1,
            "genlane:3-4": 1,
            "genlane:4-5": 1,
            **{f"memory:{node}": 4 for node in spec.nodes},
            **{f"bsm:{node}": 1 for node in spec.nodes},
        }
        bases = build_route_construction_catalogue(
            spec,
            candidate_count=1,
            construction_kinds=("balanced",),
        )
        from algorithms.telgen import expand_construction_candidates

        expansion = expand_construction_candidates(spec, bases, capacities)
        solution = ConstructionAwareLPTeacher().solve(expansion, capacities)

        self.assertEqual(len(expansion.variables), 2)
        self.assertAlmostEqual(solution.completed_request_mass, 2.0, places=7)
        self.assertTrue(all(
            value > 1.0 - 1e-6
            for value in solution.final_values.values()
        ))

    def test_reserved_usage_reduces_the_lp_resource_rhs(self):
        bases = one_hop_bases()
        variables = (
            manual_variable(
                bases["slow"], variable_id="slow@0", completion=3
            ),
            manual_variable(
                bases["fast"], variable_id="fast@0", completion=1
            ),
        )
        solution = ConstructionAwareLPTeacher().solve(
            variables,
            {"shared": 2},
            reserved_usage={("shared", 0): 1},
        )
        resource = next(
            item for item in solution.stage_one_lp.ub_constraints
            if item.kind == "resource_time"
        )
        self.assertEqual(resource.rhs, 1.0)
        self.assertAlmostEqual(solution.completed_request_mass, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
