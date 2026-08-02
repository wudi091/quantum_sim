import unittest
from unittest.mock import patch

import qnet_core.planners as planners_module
from qnet_core.env import SharedRoutingEnv
from qnet_core.planner_api import PlanDescriptor, PlanningSnapshot
from qnet_core.planners import OptimalPlanner, QDDCAPlanner
from qnet_core.spec import EpisodeSpec, PhysicalConfig, RequestSpec


def _request_row(request_id: str, shortest_hops: int) -> dict[str, object]:
    return {
        "id": request_id,
        "deadline": 8,
        "shortest_hops": shortest_hops,
        "delivered_pairs": 0,
        "demand_pairs": 1,
    }


def _plan(
    plan_id: str,
    request_id: str,
    pair_ids: tuple[str, ...],
    *,
    remaining_hops: int,
    completes: bool,
) -> PlanDescriptor:
    reached = 100 + len(pair_ids)
    return PlanDescriptor(
        plan_id=plan_id,
        request_id=request_id,
        route_nodes=(0, reached),
        reached_node=reached,
        elementary_pair_ids=pair_ids,
        swap_actions=(),
        duration=1,
        remaining_hops=remaining_hops,
        completes_request=completes,
    )


class OptimalPlannerTests(unittest.TestCase):
    @staticmethod
    def _single_candidate_snapshot() -> PlanningSnapshot:
        return PlanningSnapshot(
            time=0,
            requests=(_request_row("r0", 1),),
            resources=(),
            candidates=(
                _plan(
                    "r0-complete", "r0", ("e0",),
                    remaining_hops=0, completes=True,
                ),
            ),
            action_mask=(True,),
            metrics={},
        )

    def test_requests_and_verifies_zero_mip_gap(self):
        real_milp = planners_module.milp

        with patch.object(
            planners_module, "milp", wraps=real_milp
        ) as mocked_milp:
            selected = OptimalPlanner().select(
                self._single_candidate_snapshot()
            )

        self.assertEqual(selected, ("r0-complete",))
        self.assertEqual(
            mocked_milp.call_args.kwargs["options"]["mip_rel_gap"],
            0.0,
        )

    def test_nonzero_reported_mip_gap_is_rejected(self):
        real_milp = planners_module.milp

        def inexact_milp(*args, **kwargs):
            result = real_milp(*args, **kwargs)
            result.mip_gap = 1e-4
            return result

        with patch.object(
            planners_module, "milp", side_effect=inexact_milp
        ):
            with self.assertRaisesRegex(
                RuntimeError, "not proven optimal"
            ):
                OptimalPlanner().select(self._single_candidate_snapshot())

    def test_open_reported_dual_bound_is_rejected(self):
        real_milp = planners_module.milp

        def open_bound_milp(*args, **kwargs):
            result = real_milp(*args, **kwargs)
            result.mip_gap = 0.0
            result.mip_dual_bound = result.fun - 1.0
            return result

        with patch.object(
            planners_module, "milp", side_effect=open_bound_milp
        ):
            with self.assertRaisesRegex(
                RuntimeError, "lacks a closed objective bound"
            ):
                OptimalPlanner().select(self._single_candidate_snapshot())

    def test_completion_count_dominates_partial_progress(self):
        candidates = (
            _plan(
                "r0-far-partial", "r0", ("shared", "r0-extra"),
                remaining_hops=0, completes=False,
            ),
            _plan(
                "r1-complete", "r1", ("shared",),
                remaining_hops=0, completes=True,
            ),
            _plan(
                "r2-complete", "r2", ("independent",),
                remaining_hops=0, completes=True,
            ),
        )
        snapshot = PlanningSnapshot(
            time=0,
            requests=(
                _request_row("r0", 8),
                _request_row("r1", 1),
                _request_row("r2", 1),
            ),
            resources=(),
            candidates=candidates,
            action_mask=(True, True, True),
            metrics={},
        )

        selected = set(OptimalPlanner().select(snapshot))

        self.assertEqual(selected, {"r1-complete", "r2-complete"})

    def test_progress_dominates_work_cost_after_equal_completions(self):
        candidates = (
            _plan(
                "near", "r0", ("e0",),
                remaining_hops=3, completes=False,
            ),
            _plan(
                "far", "r0", ("e0", "e1", "e2"),
                remaining_hops=1, completes=False,
            ),
        )
        snapshot = PlanningSnapshot(
            time=0,
            requests=(_request_row("r0", 4),),
            resources=(),
            candidates=candidates,
            action_mask=(True, True),
            metrics={},
        )

        self.assertEqual(OptimalPlanner().select(snapshot), ("far",))

    def test_qddca_and_optimal_share_core_but_differ_by_planner_action(self):
        spec = EpisodeSpec(
            seed=211,
            nodes=(0, 1, 2, 3, 4, 5),
            edges=((0, 1), (1, 2), (3, 4), (4, 5)),
            requests=(
                RequestSpec("r0", 0, 2, ttl=1),
                RequestSpec("r1", 3, 5, ttl=1),
            ),
            horizon=2,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                memory_capacity=2,
            ),
        )

        optimal_env = SharedRoutingEnv(spec)
        optimal = OptimalPlanner()
        optimal.reset(spec.seed)
        optimal_env.commit(optimal.select(optimal_env.snapshot()))

        qddca_env = SharedRoutingEnv(spec)
        qddca = QDDCAPlanner()
        qddca.reset(spec.seed)
        qddca_env.commit(qddca.select(qddca_env.snapshot()))

        self.assertEqual(optimal_env.metrics()["completion_rate"], 1.0)
        self.assertEqual(qddca_env.metrics()["completion_rate"], 0.0)
        self.assertEqual(optimal_env.time, qddca_env.time)


if __name__ == "__main__":
    unittest.main()
