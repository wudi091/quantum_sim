import unittest

from algorithms.telgen import (
    ConstructionAwareMILPOracle,
    HardConstraintDecoder,
    NominalConstructionSchedule,
    ResourceSlotUsage,
    TimeExpandedCandidate,
    build_teacher_batch_record,
    compare_decoder_and_milp,
    validate_decoded_selection,
)
from qnet_core.construction_catalog import build_route_construction_catalogue
from qnet_core.planning_spec import PlanningSpec, RequestSpec
from qnet_core.scenario import ScenarioConfig


def request_bases():
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
    candidates = build_route_construction_catalogue(
        spec,
        candidate_count=1,
        construction_kinds=("balanced",),
    )
    return {candidate.request_id: candidate for candidate in candidates}


def variable(base, resources, *, completion=1, token="0"):
    usage = tuple(sorted(
        ResourceSlotUsage(resource, 0, 1)
        for resource in resources
    ))
    schedule = NominalConstructionSchedule(
        candidate_id=base.candidate_id,
        operation_slots=((base.dag.operations[0].op_id, 0),),
        duration_slots=completion,
        resource_usage=usage,
    )
    return TimeExpandedCandidate(
        variable_id=f"{base.request_id}@{token}",
        base_candidate=base,
        start_slot=0,
        completion_slot=completion,
        completion_latency=completion,
        expected_fidelity=1.0,
        resource_usage=usage,
        nominal_schedule=schedule,
    )


class HardDecoderTests(unittest.TestCase):
    def test_one_drop_local_search_replaces_one_blocker_with_two_requests(self):
        bases = request_bases()
        variables = (
            variable(bases["r0"], ("a", "b")),
            variable(bases["r1"], ("a",)),
            variable(bases["r2"], ("b",)),
        )
        scores = {
            "r0@0": 0.9,
            "r1@0": 0.8,
            "r2@0": 0.8,
        }
        decoded = HardConstraintDecoder(
            beam_width=1,
            random_restarts=0,
        ).decode(
            variables,
            {"a": 1, "b": 1},
            scores,
        )
        self.assertEqual(decoded.completed_request_count, 2)
        self.assertEqual(set(decoded.selected_by_request), {"r1", "r2"})
        self.assertGreaterEqual(decoded.local_search_iterations, 1)
        self.assertTrue(decoded.feasibility.feasible)

    def test_same_throughput_replacement_reduces_completion_latency(self):
        bases = request_bases()
        slow = variable(
            bases["r0"], ("shared",), completion=5, token="slow"
        )
        fast = variable(
            bases["r1"], ("shared",), completion=1, token="fast"
        )
        decoded = HardConstraintDecoder().decode(
            (slow, fast),
            {"shared": 1},
            {"r0@slow": 0.9, "r1@fast": 0.8},
        )
        self.assertEqual(decoded.completed_request_count, 1)
        self.assertEqual(decoded.selected_variables[0].request_id, "r1")
        self.assertEqual(decoded.total_completion_latency, 1.0)

    def test_decoder_matches_milp_on_augmenting_example(self):
        bases = request_bases()
        variables = (
            variable(bases["r0"], ("a", "b")),
            variable(bases["r1"], ("a",)),
            variable(bases["r2"], ("b",)),
        )
        capacities = {"a": 1, "b": 1}
        decoded = HardConstraintDecoder().decode(
            variables,
            capacities,
            {"r0@0": 0.9, "r1@0": 0.8, "r2@0": 0.8},
        )
        discrete = ConstructionAwareMILPOracle().solve(
            variables,
            capacities,
        )
        report = compare_decoder_and_milp(decoded, discrete)
        self.assertTrue(report.decoder_feasible)
        self.assertTrue(report.throughput_is_optimal)
        self.assertEqual(report.throughput_absolute_loss, 0)
        self.assertEqual(report.latency_absolute_gap, 0.0)

    def test_validator_rejects_duplicate_request_selection(self):
        bases = request_bases()
        first = variable(bases["r0"], ("a",), token="first")
        second = variable(bases["r0"], ("b",), token="second")
        report = validate_decoded_selection(
            (first, second),
            {"a": 1, "b": 1},
        )
        self.assertFalse(report.feasible)
        self.assertIn("request selected twice", report.violations[0])

    def test_declared_request_without_candidate_is_reported_rejected(self):
        bases = request_bases()
        only = variable(bases["r0"], ("only",))
        decoded = HardConstraintDecoder().decode(
            (only,),
            {"only": 1},
            {"r0@0": 1.0},
            request_ids=("r0", "r1", "r2"),
        )
        self.assertEqual(decoded.rejected_request_ids, ("r1", "r2"))

    def test_decoder_accounts_for_running_resource_reservations(self):
        bases = request_bases()
        variables = (
            variable(bases["r0"], ("shared",)),
            variable(bases["r1"], ("shared",)),
        )
        decoded = HardConstraintDecoder().decode(
            variables,
            {"shared": 2},
            {"r0@0": 0.9, "r1@0": 0.8},
            reserved_usage={("shared", 0): 1},
        )
        self.assertEqual(decoded.completed_request_count, 1)
        self.assertTrue(decoded.feasibility.feasible)

    def test_real_static_batch_matches_small_milp_throughput(self):
        record = build_teacher_batch_record(
            ScenarioConfig(
                request_count=8,
                min_hops=2,
                max_hops=5,
                ttl=6,
                horizon=6,
            ),
            seed=100,
            path_candidate_count=1,
        )
        decoded = HardConstraintDecoder().decode(
            record.expansion,
            record.capacities,
            record.solution.final_values,
            request_ids=tuple(
                request.id for request in record.episode.requests
            ),
        )
        discrete = ConstructionAwareMILPOracle().solve(
            record.expansion,
            record.capacities,
        )
        report = compare_decoder_and_milp(decoded, discrete)
        self.assertTrue(report.decoder_feasible)
        self.assertTrue(report.throughput_is_optimal)
        self.assertEqual(report.latency_absolute_gap, 0.0)

    def test_large_candidate_set_uses_lp_support_multistart(self):
        bases = request_bases()
        variables = tuple(
            variable(
                bases[request_id],
                (f"resource:{request_id}:{index}",),
                completion=index + 1,
                token=str(index),
            )
            for request_id in ("r0", "r1", "r2")
            for index in range(4)
        )
        capacities = {
            f"resource:{request_id}:{index}": 1
            for request_id in ("r0", "r1", "r2")
            for index in range(4)
        }
        scores = {
            item.variable_id: (1.0 if item.variable_id.endswith("@0") else 0.0)
            for item in variables
        }
        decoded = HardConstraintDecoder(
            scalable_variable_threshold=10,
            scalable_random_restarts=2,
        ).decode(variables, capacities, scores)

        self.assertEqual(decoded.completed_request_count, 3)
        self.assertEqual(decoded.search_strategy, "lp_support_multistart")
        self.assertEqual(decoded.support_variable_count, 3)
        self.assertTrue(decoded.feasibility.feasible)


if __name__ == "__main__":
    unittest.main()
