from __future__ import annotations

import unittest

import qnet_core.order_milp as order_milp_module

from qnet_core.order_core import (
    OrderAwareBatchEnv,
    OrderBatchProblem,
    OrderCoreConfig,
    OrderLinkSpec,
    OrderPlan,
    simulate_order_batch,
)
from qnet_core.order_milp import MilpNominalPathOrderPlanner
from qnet_core.order_waxman import (
    WaxmanOrderConfig,
    make_waxman_order_episode,
)


class NominalLazyMilpTests(unittest.TestCase):
    def test_executor_feasibility_is_not_downward_closed(self):
        config = WaxmanOrderConfig(
            node_count=10,
            average_degree=3,
            target_link_probability=0.6,
            request_count=12,
            arrival_rate=4.0,
            episode_steps=3,
            request_ttl_slots=3,
            min_hops=2,
            max_hops=4,
            candidate_paths=2,
            order_variants_per_path=2,
            candidate_request_cap=None,
            node_memory_cap=3,
            slot_duration_ps=3_000,
            generation_interval_ps=1_000,
            swap_service_ps=1_000,
            memory_reset_ps=100,
            swap_probability=0.9,
            bsm_capacity_per_node=2,
            epr_ttl_slots=3,
        )
        episode = make_waxman_order_episode(config, seed=1)
        pending = tuple(
            request.request_id for request in episode.requests
        )
        request_ids = episode.eligible_request_ids(pending, slot=1)
        self.assertEqual(
            request_ids,
            ("r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"),
        )
        problem = episode.problem_for_slot(
            request_ids, slot=1, physics_seed=0,
        )
        feasible_superset = (
            "t1:r0:p0:o0",
            "t1:r2:p0:o0",
            "t1:r3:p0:o0",
            "t1:r6:p1:o0",
            "t1:r7:p0:o0",
        )
        infeasible_subset = tuple(
            plan_id for plan_id in feasible_superset
            if ":r2:" not in plan_id
        )

        superset_result = simulate_order_batch(
            problem, feasible_superset, record_traces=False,
        )
        subset_result = simulate_order_batch(
            problem, infeasible_subset, record_traces=False,
        )

        # Removing r2 lets r6 occupy node 0 one generation epoch earlier.
        # That delays r7's first unblocked draw until the last epoch, where it
        # fails.  Thus executor feasibility is not downward-closed: a core or
        # hyperedge cut that also rejects supersets of this failed subset is
        # unsound.  Lazy MILP cuts must exclude only the exact assignment.
        self.assertEqual(
            superset_result.completed,
            ("r0", "r2", "r3", "r6", "r7"),
        )
        self.assertEqual(superset_result.failed, ())
        self.assertEqual(superset_result.missed, ())
        self.assertEqual(subset_result.completed, ("r0", "r3", "r6"))
        self.assertEqual(subset_result.failed, ())
        self.assertEqual(subset_result.missed, ("r7",))
        r7_plan = next(
            plan for plan in problem.candidates
            if plan.plan_id == "t1:r7:p0:o0"
        )
        self.assertTrue(
            order_milp_module._plan_hard_possible_in_scenario(
                problem, r7_plan, 0
            )
        )

    def test_static_resource_bound_reduces_infeasible_top_cardinality(self):
        plans = (
            OrderPlan("r1:fixed", "r1", (0, 1, 2), (1,), priority=0),
            OrderPlan("r2:fixed", "r2", (3, 1, 4), (1,), priority=1),
        )
        problem = OrderBatchProblem.create(
            candidates=plans,
            node_capacity={0: 1, 1: 2, 2: 1, 3: 1, 4: 1},
            links=tuple(
                OrderLinkSpec(left, right, generation_probability=1.0)
                for left, right in ((0, 1), (1, 2), (1, 3), (1, 4))
            ),
            config=OrderCoreConfig(
                slot_duration_ps=1_000,
                generation_interval_ps=1_000,
                swap_service_ps=1_000,
                memory_reset_ps=0,
                generation_probability=1.0,
                swap_probability=1.0,
                bsm_capacity_per_node=2,
            ),
        )
        planner = MilpNominalPathOrderPlanner((0,))

        selected = planner.select(OrderAwareBatchEnv(problem).snapshot())

        # Both requests are individually completable, but the sole generation
        # epoch cannot admit both through the two-slot hotspot.  The safe node
        # throughput row proves a static upper bound of one before executor
        # enumeration.  Deterministic search prefers the lower-priority-rank
        # request and its sole candidate.
        self.assertEqual(selected, ("r1:fixed",))
        self.assertEqual(planner.last_objective, 1)
        self.assertTrue(planner.last_proven_optimal)
        self.assertEqual(planner.last_solution.cuts, 0)
        self.assertEqual(planner.last_solution.milp_solves, 1)
        self.assertEqual(planner.last_solution.static_upper_bound, 1)
        self.assertEqual(planner.last_solution.enumerated_assignments, 1)
        self.assertEqual(planner.last_solution.evaluations, 1)


if __name__ == "__main__":
    unittest.main()
