import unittest

from algorithms.qddca import QDDCAPlanner
from qnet_core.planner_api import PlanDescriptor, PlanFeedback, PlanningSnapshot


def plan(plan_id, request_id, reached, remaining, kind="allocation"):
    return PlanDescriptor(
        plan_id=plan_id,
        request_id=request_id,
        route_nodes=(0, reached) if kind == "allocation" else (0,),
        reached_node=reached,
        elementary_pair_ids=(),
        swap_actions=(),
        duration=0,
        remaining_hops=remaining,
        completes_request=False,
        kind=kind,
    )


class QDDCAPlannerTests(unittest.TestCase):
    def snapshot(self, candidates, feedback=()):
        return PlanningSnapshot(
            time=0,
            requests=(
                {
                    "id": "r0",
                    "source": 0,
                    "destination": 5,
                    "frontier": 0,
                    "initial_hops": 3.0,
                    "shortest_hops": 3,
                    "shortest_next_hop": 1,
                    "fidelity_hop_bound": 3,
                    "arrival": 0,
                    "completed_at": None,
                    "expired_at": None,
                },
            ),
            resources=(),
            candidates=tuple(candidates),
            action_mask=tuple(True for _ in candidates),
            metrics={},
            phase="allocate",
            feedback=tuple(feedback),
        )

    def test_hard_and_soft_bounds_prune_candidates(self):
        planner = QDDCAPlanner(max_try=5, seed=7)
        planner.reset(7)
        candidates = [
            plan("short", "r0", 1, 2),
            plan("detour", "r0", 2, 3),
            plan("too-long", "r0", 3, 4),
            plan("drop", "r0", 0, 3, kind="drop"),
        ]
        selected = planner.select(self.snapshot(candidates))
        self.assertEqual(selected, ("short",))
        self.assertNotIn("too-long", selected)

    def test_rejected_request_updates_neighbor_acceptance_history(self):
        planner = QDDCAPlanner(max_try=3, seed=11)
        planner.reset(11)
        first = plan("first", "r0", 1, 2)
        selected = planner.select(self.snapshot([first]))
        self.assertEqual(selected, ("first",))
        feedback = PlanFeedback(
            feedback_id=1,
            time=0,
            phase="allocate",
            plan_id="first",
            request_id="r0",
            reached_node=1,
            accepted=False,
            succeeded=False,
            reason="resource_rejected",
        )
        planner.select(self.snapshot([plan("second", "r0", 1, 2)], (feedback,)))
        self.assertEqual(list(planner.history[(0, 1)]), [False])
        self.assertEqual(planner.attempts["r0"], 3)

    def test_exhausted_attempts_select_virtual_drop(self):
        planner = QDDCAPlanner(max_try=1, seed=13)
        planner.reset(13)
        first = plan("first", "r0", 1, 2)
        self.assertEqual(planner.select(self.snapshot([first])), ("first",))
        feedback = PlanFeedback(
            feedback_id=1,
            time=0,
            phase="allocate",
            plan_id="first",
            request_id="r0",
            reached_node=1,
            accepted=False,
            succeeded=False,
            reason="resource_rejected",
        )
        drop = plan("drop", "r0", 0, 3, kind="drop")
        self.assertEqual(planner.select(self.snapshot([drop], (feedback,))), ("drop",))


if __name__ == "__main__":
    unittest.main()
