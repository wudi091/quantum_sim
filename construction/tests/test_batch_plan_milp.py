from __future__ import annotations

import unittest

from construction.batch_plan_milp import (
    TimedPlanCandidate,
    solve_joint_counterexample_milp,
    solve_time_indexed_batch_milp,
)


class BatchPlanMilpTests(unittest.TestCase):
    def test_required_request_is_selected(self) -> None:
        candidates = (
            TimedPlanCandidate(
                request_id="required",
                candidate_id="large",
                path=("A", "B"),
                swap_order=(),
                duration=2,
                memory_profile={"A": (1, 1), "B": (1, 1)},
                bsm_profile={},
            ),
            TimedPlanCandidate(
                request_id="optional",
                candidate_id="small",
                path=("C", "D"),
                swap_order=(),
                duration=1,
                memory_profile={"C": (1,), "D": (1,)},
                bsm_profile={},
            ),
        )
        solution = solve_time_indexed_batch_milp(
            candidates,
            memory_capacity={"A": 1, "B": 1, "C": 1, "D": 1},
            bsm_capacity={},
            horizon=2,
            required_requests=("required",),
        )

        self.assertIn("required", solution.selected)

    def test_joint_counterexample_selects_all_and_c_first(self) -> None:
        solution = solve_joint_counterexample_milp()

        self.assertEqual(solution.completed_requests, 3)
        self.assertEqual(set(solution.selected), {"R1", "R2", "R3"})
        self.assertEqual(solution.selected["R1"].candidate.swap_order[0], "C")
        self.assertEqual(solution.selected["R1"].start, 0)
        self.assertEqual(solution.selected["R2"].start, 1)
        self.assertEqual(solution.selected["R3"].start, 2)

    def test_joint_model_selects_an_alternate_path_around_hotspot(self) -> None:
        candidates = (
            TimedPlanCandidate(
                request_id="blocker",
                candidate_id="hold-c",
                path=("A", "C", "B"),
                swap_order=("C",),
                duration=2,
                memory_profile={"A": (1, 1), "C": (2, 2), "B": (1, 1)},
                bsm_profile={},
                allowed_starts=(0,),
                priority=0,
            ),
            TimedPlanCandidate(
                request_id="routed",
                candidate_id="via-c",
                path=("S", "C", "T"),
                swap_order=("C",),
                duration=1,
                memory_profile={"S": (1,), "C": (2,), "T": (1,)},
                bsm_profile={"C": (1,)},
                priority=1,
            ),
            TimedPlanCandidate(
                request_id="routed",
                candidate_id="via-d",
                path=("S", "D", "T"),
                swap_order=("D",),
                duration=1,
                memory_profile={"S": (1,), "D": (2,), "T": (1,)},
                bsm_profile={"D": (1,)},
                priority=1,
            ),
        )
        solution = solve_time_indexed_batch_milp(
            candidates,
            memory_capacity={
                "A": 1,
                "B": 1,
                "C": 2,
                "D": 2,
                "S": 1,
                "T": 1,
            },
            bsm_capacity={"C": 1, "D": 1},
            horizon=2,
        )

        self.assertEqual(solution.completed_requests, 2)
        self.assertEqual(
            solution.selected["routed"].candidate.candidate_id,
            "via-d",
        )

    def test_exclusive_current_epr_cannot_be_double_claimed(self) -> None:
        candidates = tuple(
            TimedPlanCandidate(
                request_id=request_id,
                candidate_id="direct",
                path=(request_id, "T"),
                swap_order=(),
                duration=1,
                memory_profile={request_id: (1,), "T": (1,)},
                bsm_profile={},
                exclusive_resources=frozenset({"epr-shared"}),
            )
            for request_id in ("R1", "R2")
        )
        solution = solve_time_indexed_batch_milp(
            candidates,
            memory_capacity={"R1": 1, "R2": 1, "T": 2},
            bsm_capacity={},
            horizon=1,
        )

        self.assertEqual(solution.completed_requests, 1)


if __name__ == "__main__":
    unittest.main()
