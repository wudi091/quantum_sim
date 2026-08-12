from dataclasses import replace
import unittest

from algorithms.telgen.milp_oracle import ConstructionAwareMILPOracle
from algorithms.telgen.time_expansion import (
    NominalConstructionSchedule,
    ResourceSlotUsage,
    TimeExpandedCandidate,
)
from algorithms.telgen.validate_construction_milp import (
    ConstructionMILPTrial,
    aggregate_trials,
    compare_construction_policies,
)
from qnet_core.construction_catalog import build_route_construction_catalogue
from qnet_core.planning_spec import PlanningSpec, RequestSpec


def _base_candidates():
    spec = PlanningSpec(
        seed=5,
        nodes=(0, 1, 2, 3),
        edges=((0, 1), (2, 3)),
        requests=(
            RequestSpec("r0", 0, 1, ttl=1),
            RequestSpec("r1", 2, 3, ttl=1),
        ),
        horizon=1,
    )
    candidates = build_route_construction_catalogue(
        spec,
        candidate_count=1,
        construction_kinds=("left_deep",),
    )
    return {candidate.request_id: candidate for candidate in candidates}


def _variable(base, policy, resource):
    candidate = replace(
        base,
        candidate_id=f"{base.request_id}:{policy}",
        construction_kind=policy,
    )
    operation_id = candidate.dag.operations[0].op_id
    usage = (ResourceSlotUsage(resource, 0, 1),)
    schedule = NominalConstructionSchedule(
        candidate_id=candidate.candidate_id,
        operation_slots=((operation_id, 0),),
        duration_slots=1,
        resource_usage=usage,
    )
    return TimeExpandedCandidate(
        variable_id=f"{candidate.candidate_id}@slot:0",
        base_candidate=candidate,
        start_slot=0,
        completion_slot=1,
        completion_latency=1,
        expected_fidelity=1.0,
        resource_usage=usage,
        nominal_schedule=schedule,
    )


class ConstructionMILPValidationTests(unittest.TestCase):
    def test_mixing_policies_can_beat_every_fixed_policy(self):
        bases = _base_candidates()
        variables = (
            _variable(bases["r0"], "tree_a", "a"),
            _variable(bases["r1"], "tree_a", "a"),
            _variable(bases["r0"], "tree_b", "b"),
            _variable(bases["r1"], "tree_b", "b"),
        )
        comparison = compare_construction_policies(
            variables,
            {"a": 1, "b": 1},
            fixed_policies=("tree_a", "tree_b"),
            oracle=ConstructionAwareMILPOracle(),
        )
        self.assertEqual(
            comparison.construction_aware.completed_request_count,
            2,
        )
        self.assertEqual(comparison.best_fixed.completed_request_count, 1)
        self.assertEqual(comparison.completed_request_delta, 1)
        self.assertEqual(
            set(
                comparison.construction_aware.selected_construction_kinds
            ),
            {"tree_a", "tree_b"},
        )

        aggregate = aggregate_trials(
            (ConstructionMILPTrial(
                seed=5,
                node_count=4,
                edge_count=2,
                request_count=2,
                candidate_count=4,
                variable_count=4,
                rejected_candidate_count=0,
                comparison=comparison,
            ),),
            bootstrap_samples=100,
            randomization_samples=100,
            statistics_seed=9,
        )
        self.assertTrue(aggregate["all_milp_solutions_exact"])
        self.assertFalse(aggregate["advantage_validated"])

    def test_fixed_policies_must_exactly_partition_the_aware_set(self):
        bases = _base_candidates()
        variables = (
            _variable(bases["r0"], "tree_a", "a"),
            _variable(bases["r1"], "tree_b", "b"),
        )
        with self.assertRaisesRegex(ValueError, "do not partition"):
            compare_construction_policies(
                variables,
                {"a": 1, "b": 1},
                fixed_policies=("tree_a",),
            )


if __name__ == "__main__":
    unittest.main()
