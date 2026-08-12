import unittest

from algorithms.telgen.validate_construction_physics import (
    ConstructionPhysicalTrial,
    PhysicalPolicyOutcome,
    aggregate_physical_trials,
)


def _outcome(
    policy: str,
    *,
    planned: int,
    completed: int,
    latency: float,
    violations=(),
):
    unsafe = sum(code != "slot_completion_overrun" for code in violations)
    return PhysicalPolicyOutcome(
        policy=policy,
        planned_selected_requests=planned,
        completed_requests=completed,
        completion_retention=completed / planned if planned else None,
        mean_censored_latency_slots=latency,
        p95_completion_latency_slots=latency,
        schedule_violation_count=len(violations),
        unsafe_schedule_violation_count=unsafe,
        schedule_violation_codes=tuple(violations),
        physical_failure_count=planned - completed,
        fidelity_violation_count=0,
        physical_backend_rejection_count=0,
        post_completion_validation_failure_count=0,
        peak_physical_memory_usage=1,
    )


def _trial(seed: int, aware, fixed):
    return ConstructionPhysicalTrial(
        planning_seed=seed,
        physical_seed=50_000 + seed,
        node_count=8,
        edge_count=10,
        request_count=4,
        candidate_count=20,
        variable_count=20,
        best_fixed_policy="swap_tree_0",
        construction_aware=aware,
        best_fixed=fixed,
        completed_request_delta=(
            aware.completed_requests - fixed.completed_requests
        ),
        censored_latency_delta_slots=(
            fixed.mean_censored_latency_slots
            - aware.mean_censored_latency_slots
        ),
    )


class ConstructionPhysicalValidationTests(unittest.TestCase):
    def test_aggregate_uses_physical_completions_and_pooled_retention(self):
        trials = (
            _trial(
                1,
                _outcome("construction_aware", planned=4, completed=3, latency=2),
                _outcome("swap_tree_0", planned=3, completed=2, latency=3),
            ),
            _trial(
                2,
                _outcome("construction_aware", planned=2, completed=2, latency=1),
                _outcome("swap_tree_0", planned=2, completed=1, latency=2),
            ),
        )

        aggregate = aggregate_physical_trials(
            trials,
            bootstrap_samples=100,
            randomization_samples=100,
            statistics_seed=9,
        )

        self.assertEqual(
            aggregate["construction_aware_mean_completed_requests"], 2.5
        )
        self.assertEqual(aggregate["best_fixed_mean_completed_requests"], 1.5)
        self.assertEqual(aggregate["mean_completed_request_delta"], 1.0)
        self.assertAlmostEqual(
            aggregate["construction_aware_completion_retention"], 5 / 6
        )
        self.assertAlmostEqual(
            aggregate["best_fixed_completion_retention"], 3 / 5
        )
        self.assertTrue(aggregate["hard_gates_valid"])

    def test_completion_overrun_is_nonfatal_but_other_violation_is_fatal(self):
        nonfatal = _trial(
            1,
            _outcome(
                "construction_aware",
                planned=1,
                completed=1,
                latency=1,
                violations=("slot_completion_overrun",),
            ),
            _outcome("swap_tree_0", planned=1, completed=1, latency=1),
        )
        fatal = _trial(
            2,
            _outcome(
                "construction_aware",
                planned=1,
                completed=0,
                latency=4,
                violations=("launch_rejected",),
            ),
            _outcome("swap_tree_0", planned=1, completed=1, latency=1),
        )

        nonfatal_aggregate = aggregate_physical_trials(
            (nonfatal,),
            bootstrap_samples=10,
            randomization_samples=10,
            statistics_seed=4,
        )
        fatal_aggregate = aggregate_physical_trials(
            (fatal,),
            bootstrap_samples=10,
            randomization_samples=10,
            statistics_seed=4,
        )
        self.assertTrue(nonfatal_aggregate["hard_gates_valid"])
        self.assertFalse(fatal_aggregate["hard_gates_valid"])


if __name__ == "__main__":
    unittest.main()
