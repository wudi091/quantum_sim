import unittest

from algorithms.routing_core.packing import (
    greedy_feasible_projection,
    validate_packing_selection,
)
from algorithms.routing_core.time_expansion import (
    NominalConstructionSchedule,
    ResourceSlotUsage,
    TimeExpandedCandidate,
)
from qnet_core.construction_catalog import build_route_construction_catalogue
from qnet_core.planning_spec import PlanningSpec, RequestSpec


def _request_bases():
    spec = PlanningSpec(
        seed=12,
        nodes=(0, 1, 2, 3, 4, 5),
        edges=((0, 1), (2, 3), (4, 5)),
        requests=(
            RequestSpec("r0", 0, 1, ttl=8),
            RequestSpec("r1", 2, 3, ttl=8),
            RequestSpec("r2", 4, 5, ttl=8),
        ),
        horizon=8,
    )
    return {
        candidate.request_id: candidate
        for candidate in build_route_construction_catalogue(
            spec,
            candidate_count=1,
            construction_kinds=("balanced",),
        )
    }


def _variable(base, resources, *, token="0"):
    usage = tuple(sorted(
        ResourceSlotUsage(resource, 0, 1) for resource in resources
    ))
    schedule = NominalConstructionSchedule(
        candidate_id=base.candidate_id,
        operation_slots=((base.dag.operations[0].op_id, 0),),
        duration_slots=1,
        resource_usage=usage,
    )
    return TimeExpandedCandidate(
        variable_id=f"{base.request_id}@{token}",
        base_candidate=base,
        start_slot=0,
        completion_slot=1,
        completion_latency=1,
        expected_fidelity=1.0,
        resource_usage=usage,
        nominal_schedule=schedule,
    )


class PackingTests(unittest.TestCase):
    def test_projection_is_one_score_order_scan(self):
        bases = _request_bases()
        variables = (
            _variable(bases["r0"], ("a", "b")),
            _variable(bases["r1"], ("a",)),
            _variable(bases["r2"], ("b",)),
        )
        result = greedy_feasible_projection(
            variables,
            {"a": 1, "b": 1},
            {"r0@0": 0.9, "r1@0": 0.8, "r2@0": 0.8},
        )

        self.assertEqual(result.completed_request_count, 1)
        self.assertEqual(result.selected_variables[0].request_id, "r0")
        self.assertEqual(result.strategy, "score_order_greedy")

    def test_projection_selects_one_candidate_per_request(self):
        bases = _request_bases()
        variables = (
            _variable(bases["r0"], ("a",), token="first"),
            _variable(bases["r0"], ("b",), token="second"),
        )
        result = greedy_feasible_projection(
            variables,
            {"a": 1, "b": 1},
            {"r0@first": 0.9, "r0@second": 0.8},
        )

        self.assertEqual(result.completed_request_count, 1)
        self.assertEqual(result.selected_variables[0].variable_id, "r0@first")

    def test_projection_respects_reserved_usage(self):
        bases = _request_bases()
        variables = (
            _variable(bases["r0"], ("shared",)),
            _variable(bases["r1"], ("shared",)),
        )
        result = greedy_feasible_projection(
            variables,
            {"shared": 2},
            {"r0@0": 0.9, "r1@0": 0.8},
            reserved_usage={("shared", 0): 1},
        )

        self.assertEqual(result.completed_request_count, 1)
        self.assertTrue(result.feasibility.feasible)

    def test_validator_rejects_duplicate_requests(self):
        bases = _request_bases()
        first = _variable(bases["r0"], ("a",), token="first")
        second = _variable(bases["r0"], ("b",), token="second")

        report = validate_packing_selection(
            (first, second),
            {"a": 1, "b": 1},
        )

        self.assertFalse(report.feasible)
        self.assertIn("request selected twice: r0", report.violations)


if __name__ == "__main__":
    unittest.main()
