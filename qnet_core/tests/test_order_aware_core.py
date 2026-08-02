import unittest

from qnet_core.order_core import (
    OrderAwareBatchEnv,
    OrderBatchProblem,
    OrderCoreConfig,
    OrderPlan,
    OrderStoredPair,
    simulate_order_batch,
)
from qnet_core.order_benchmark import run_suite
from qnet_core.order_planners import (
    QCASTFixedOrderPlanner,
    QDDCAFixedOrderPlanner,
    SAAPathOrderPlanner,
    SAAPathPlanner,
)
from qnet_core.order_scenarios import make_order_counterexample


class OrderAwareCoreTests(unittest.TestCase):
    @staticmethod
    def _run(problem, planner):
        planner.reset(problem.config.seed)
        env = OrderAwareBatchEnv(problem)
        snapshot = env.snapshot()
        before = snapshot.candidates
        selected = tuple(planner.select(snapshot))
        result = env.commit(selected)
        return snapshot, before, selected, result

    def test_counterexample_isolates_swap_order_gain(self):
        problem = make_order_counterexample(hotspot_capacity=2)
        outputs = {
            planner.name: self._run(problem, planner)
            for planner in (
                QDDCAFixedOrderPlanner(),
                QCASTFixedOrderPlanner(),
                SAAPathPlanner(),
                SAAPathOrderPlanner(),
            )
        }

        self.assertEqual(outputs["qddca_fixed"][3].completed_count, 2)
        self.assertEqual(outputs["qcast_fixed"][3].completed_count, 2)
        self.assertEqual(outputs["saa_path"][3].completed_count, 2)
        self.assertEqual(outputs["saa_path_order"][3].completed_count, 3)
        selected = outputs["saa_path_order"][2]
        main = next(
            plan for plan in problem.candidates
            if plan.plan_id in selected and plan.request_id == "R1"
        )
        self.assertEqual(main.swap_order[0], "C")
        self.assertEqual(
            outputs["saa_path_order"][3].completion_time_ps,
            {"R1": 3000, "R2": 2000, "R3": 3000},
        )

    def test_roomy_hotspot_removes_order_gap(self):
        problem = make_order_counterexample(hotspot_capacity=4)
        counts = []
        for planner in (
            QDDCAFixedOrderPlanner(),
            QCASTFixedOrderPlanner(),
            SAAPathPlanner(),
            SAAPathOrderPlanner(),
        ):
            counts.append(self._run(problem, planner)[3].completed_count)
        self.assertEqual(counts, [3, 3, 3, 3])

    def test_physical_event_intervals_replace_controller_subrounds(self):
        problem = make_order_counterexample(hotspot_capacity=2)
        selected = QDDCAFixedOrderPlanner().select(
            OrderAwareBatchEnv(problem).snapshot()
        )
        result = simulate_order_batch(problem, selected)
        generation_times = tuple(
            trace.time_ps for trace in result.traces
            if trace.generation_events
        )

        self.assertEqual(generation_times, (0, 1000, 2000))
        self.assertFalse(hasattr(problem.config, "rounds_per_slot"))
        self.assertEqual(problem.config.slot_duration_ps, 3000)
        self.assertEqual(problem.config.generation_interval_ps, 1000)

    def test_parallel_swap_group_is_not_silently_serialized(self):
        config = OrderCoreConfig(
            slot_duration_ps=2_000,
            generation_interval_ps=1_000,
            swap_service_ps=1_000,
            memory_reset_ps=0,
        )
        parallel = OrderPlan(
            "parallel", "r", tuple("ABCDE"), tuple("BDC"),
            swap_groups=(("B", "D"), ("C",)),
        )
        sequential = OrderPlan(
            "sequential", "r", tuple("ABCDE"), tuple("BDC"),
        )

        parallel_result = simulate_order_batch(
            OrderBatchProblem.create(
                candidates=(parallel,),
                node_capacity={node: 2 for node in "ABCDE"},
                config=config,
                required_requests=("r",),
                preloaded_requests=("r",),
            ),
            ("parallel",),
        )
        sequential_result = simulate_order_batch(
            OrderBatchProblem.create(
                candidates=(sequential,),
                node_capacity={node: 2 for node in "ABCDE"},
                config=config,
                required_requests=("r",),
                preloaded_requests=("r",),
            ),
            ("sequential",),
        )

        self.assertEqual(parallel_result.completed, ("r",))
        self.assertEqual(parallel_result.completion_time_ps, {"r": 2_000})
        self.assertEqual(sequential_result.missed, ("r",))
        first_swap_nodes = {
            event.middle
            for event in parallel_result.traces[0].swap_events
            if event.status == "success"
        }
        self.assertEqual(first_swap_nodes, {"B", "D"})

    def test_planners_cannot_mutate_the_shared_snapshot(self):
        problem = make_order_counterexample(hotspot_capacity=2)
        for planner in (
            QDDCAFixedOrderPlanner(),
            QCASTFixedOrderPlanner(),
            SAAPathPlanner(),
            SAAPathOrderPlanner(),
        ):
            with self.subTest(planner=planner.name):
                snapshot, before, selected, _ = self._run(problem, planner)
                self.assertEqual(snapshot.candidates, before)
                self.assertLessEqual(set(selected), {
                    plan.plan_id for plan in snapshot.candidates
                })

    def test_environment_rejects_second_commit(self):
        problem = make_order_counterexample()
        env = OrderAwareBatchEnv(problem)
        selected = QDDCAFixedOrderPlanner().select(env.snapshot())
        env.commit(selected)
        with self.assertRaises(RuntimeError):
            env.commit(selected)

    def test_seeded_suite_reports_only_order_gap_under_low_memory(self):
        result = run_suite(seeds=3, hotspot_capacities=(2, 4))
        constrained = result["cases"]["hotspot_capacity_2"]
        roomy = result["cases"]["hotspot_capacity_4"]

        self.assertEqual(
            constrained["gaps"]["milp_path_minus_qddca_fixed"], 0.0
        )
        self.assertEqual(
            constrained["gaps"]["qcast_fixed_minus_qddca_fixed"], 0.0
        )
        self.assertGreater(
            constrained["gaps"]["milp_path_order_minus_milp_path"], 0.0
        )
        self.assertEqual(
            roomy["gaps"]["milp_path_order_minus_milp_path"], 0.0
        )
        self.assertIn("MILP", result["model"]["batch_optimizer"])
        for case in result["cases"].values():
            for name in ("milp_path", "milp_path_order"):
                self.assertTrue(all(
                    row["milp_objective_matches_execution"]
                    for row in case["rows"][name]
                ))
                self.assertTrue(all(
                    row["milp_certified_optimal"]
                    for row in case["rows"][name]
                ))

    def test_qcast_fixed_prefers_higher_ext_path(self):
        plans = (
            OrderPlan(
                "request:short", "request",
                ("A", "B", "C"), ("B",),
            ),
            OrderPlan(
                "request:long", "request",
                ("A", "D", "E", "C"), ("D", "E"),
            ),
        )
        problem = OrderBatchProblem.create(
            candidates=plans,
            node_capacity={node: 2 for node in "ABCDE"},
            config=OrderCoreConfig(
                generation_probability=0.8,
                swap_probability=0.9,
            ),
        )

        selected = QCASTFixedOrderPlanner().select(
            OrderAwareBatchEnv(problem).snapshot()
        )

        self.assertEqual(selected, ("request:short",))

    def test_saa_path_can_choose_a_fixed_order_detour(self):
        plans = (
            OrderPlan(
                "main:fixed", "main",
                ("A", "B", "H", "C", "D"),
                ("B", "H", "C"), priority=0,
            ),
            OrderPlan(
                "request:hotspot", "request",
                ("X", "H", "Y"), ("H",), priority=1,
            ),
            OrderPlan(
                "request:detour", "request",
                ("X", "J", "Y"), ("J",), priority=1,
            ),
        )
        problem = OrderBatchProblem.create(
            candidates=plans,
            node_capacity={node: 2 for node in "ABHCDXYJ"},
            config=OrderCoreConfig(
                slot_duration_ps=2000,
                generation_interval_ps=1000,
                swap_service_ps=1000,
                memory_reset_ps=100,
            ),
            required_requests=("main",),
            preloaded_requests=("main",),
        )

        selected = SAAPathPlanner().select(
            OrderAwareBatchEnv(problem).snapshot()
        )

        self.assertIn("request:detour", selected)
        self.assertNotIn("request:hotspot", selected)

    def test_swap_randomness_is_common_across_orders(self):
        base = make_order_counterexample(
            hotspot_capacity=2,
            swap_probability=0.55,
        )
        main_plans = tuple(
            plan for plan in base.candidates if plan.request_id == "R1"
        )
        for physics_seed in range(12):
            problem = base.with_physics_seed(physics_seed)
            outcomes = {
                simulate_order_batch(
                    problem, (plan.plan_id,), record_traces=False
                ).completed_count
                for plan in main_plans
            }
            with self.subTest(physics_seed=physics_seed):
                self.assertEqual(len(outcomes), 1)

    @staticmethod
    def _unfinished_inventory_problem(slot_id=0, initial_inventory=()):
        return OrderBatchProblem.create(
            candidates=(OrderPlan(
                plan_id=f"r-slot-{slot_id}",
                request_id="r",
                path=("A", "B", "C", "D", "E"),
                swap_order=("B", "C", "D"),
                decision_slot=slot_id,
                arrival_slot=0,
            ),),
            node_capacity={node: 2 for node in "ABCDE"},
            initial_inventory=initial_inventory,
            config=OrderCoreConfig(
                slot_duration_ps=1_000,
                generation_interval_ps=1_000,
                swap_service_ps=1_000,
                memory_reset_ps=0,
                epr_ttl_slots=3,
                slot_id=slot_id,
            ),
        )

    def test_unconsumed_elementary_pairs_cross_slot_boundary(self):
        first = self._unfinished_inventory_problem()
        result = simulate_order_batch(first, ("r-slot-0",))

        self.assertEqual(result.missed, ("r",))
        self.assertEqual(
            {pair.elementary_edge for pair in result.remaining_inventory},
            {("C", "D"), ("D", "E")},
        )
        self.assertTrue(all(
            pair.born_slot == 0 and pair.expires_slot == 3
            for pair in result.remaining_inventory
        ))

    def test_carried_inventory_is_claimed_without_duplicate_generation(self):
        first = self._unfinished_inventory_problem()
        carried = simulate_order_batch(
            first, ("r-slot-0",)
        ).remaining_inventory
        second = OrderBatchProblem.create(
            candidates=(OrderPlan(
                plan_id="next",
                request_id="next",
                path=("C", "D", "E"),
                swap_order=("D",),
                arrival_slot=1,
                decision_slot=1,
            ),),
            node_capacity={node: 2 for node in "CDE"},
            initial_inventory=carried,
            config=OrderCoreConfig(
                slot_duration_ps=1_000,
                generation_interval_ps=1_000,
                swap_service_ps=1_000,
                memory_reset_ps=0,
                slot_id=1,
            ),
        )

        result = simulate_order_batch(second, ("next",))

        self.assertEqual(result.completed, ("next",))
        self.assertFalse(any(
            event.status == "success"
            for trace in result.traces
            for event in trace.generation_events
        ))
        self.assertEqual(result.remaining_inventory, ())

    def test_problem_rejects_expired_inventory(self):
        with self.assertRaisesRegex(ValueError, "alive"):
            self._unfinished_inventory_problem(
                slot_id=3,
                initial_inventory=(OrderStoredPair(
                    "expired", "C", "D", born_slot=0, expires_slot=3,
                ),),
            )


if __name__ == "__main__":
    unittest.main()
