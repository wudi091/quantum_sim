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
    _markdown_summary,
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

    def test_count_validation_rejects_non_unit_candidate_weights(self):
        bases = _base_candidates()
        weighted = replace(
            _variable(bases["r0"], "tree_a", "a"),
            expected_success_probability=0.8,
        )
        with self.assertRaisesRegex(ValueError, "unit candidate weights"):
            compare_construction_policies(
                (weighted,),
                {"a": 1},
                fixed_policies=("tree_a",),
            )

    def test_markdown_uses_configured_tree_count_and_nominal_wording(self):
        payload = {
            "validation_config": {"swap_tree_count": 2},
            "aggregate": {
                "trial_count": 1,
                "construction_aware_mean_completed_requests": 2.0,
                "best_fixed_mean_completed_requests": 1.0,
                "mean_completed_request_delta": 1.0,
                "relative_completed_request_gain": 1.0,
                "completed_request_delta_bootstrap_95_ci": [1.0, 1.0],
                "completed_request_delta_randomization_p_value": 0.01,
                "strict_win_count": 1,
                "tie_count": 0,
                "loss_count": 0,
                "mixed_construction_solution_rate": 1.0,
                "all_milp_solutions_exact": True,
                "advantage_validated": True,
            },
        }
        markdown = _markdown_summary(payload)
        self.assertIn("2 种固定交换树", markdown)
        self.assertIn("名义可接纳数", markdown)
        self.assertIn("不等价于物理完成请求数", markdown)


if __name__ == "__main__":
    unittest.main()
