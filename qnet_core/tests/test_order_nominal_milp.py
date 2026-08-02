from __future__ import annotations

from dataclasses import replace
from itertools import product
from math import ceil
import unittest
from unittest.mock import patch

import qnet_core.order_milp as order_milp_module

from qnet_core.order_core import (
    OrderAwareBatchEnv,
    OrderBatchProblem,
    OrderCoreConfig,
    OrderLinkSpec,
    OrderPlan,
    OrderStoredPair,
    simulate_order_batch,
)
from qnet_core.order_milp import (
    MilpNominalPathOrderPlanner,
    MilpNominalPathPlanner,
    MilpStaticPathOrderPlanner,
    MilpStaticPathPlanner,
)
from qnet_core.order_waxman import (
    WaxmanOrderConfig,
    make_waxman_order_episode,
)


def _general_problem(*, direct_probability: float | None = None):
    if direct_probability is not None:
        return OrderBatchProblem.create(
            candidates=(OrderPlan("r:p", "r", (0, 1), ()),),
            node_capacity={0: 1, 1: 1},
            links=(OrderLinkSpec(0, 1, generation_probability=direct_probability),),
            config=OrderCoreConfig(
                slot_duration_ps=1_000,
                generation_interval_ps=1_000,
                swap_service_ps=1_000,
                generation_probability=direct_probability,
                swap_probability=0.9,
            ),
        )

    candidates = (
        OrderPlan("r1:fixed", "r1", (0, 1, 2, 3), (1, 2), priority=0),
        OrderPlan("r1:reverse", "r1", (0, 1, 2, 3), (2, 1), priority=0),
        OrderPlan("r2:fixed", "r2", (4, 1, 2, 5), (1, 2), priority=1),
        OrderPlan("r2:reverse", "r2", (4, 1, 2, 5), (2, 1), priority=1),
    )
    edges = {(u, v) if u < v else (v, u)
             for plan in candidates
             for u, v in zip(plan.path, plan.path[1:])}
    return OrderBatchProblem.create(
        candidates=candidates,
        node_capacity={node: 2 for node in range(6)},
        links=tuple(
            OrderLinkSpec(*edge, capacity=1, generation_probability=0.65)
            for edge in sorted(edges)
        ),
        initial_inventory=(OrderStoredPair("stored-01", 0, 1, 0, 3),),
        config=OrderCoreConfig(
            slot_duration_ps=4_000,
            generation_interval_ps=1_000,
            swap_service_ps=1_000,
            memory_reset_ps=100,
            generation_probability=0.65,
            swap_probability=0.9,
            edge_capacity=1,
            bsm_capacity_per_node=1,
            epr_ttl_slots=3,
            slot_id=0,
            seed=999,
        ),
        name="general-shared-edge-inventory",
    )


def _finite_scenario_optimum(problem, *, allow_orders, seeds, threshold):
    snapshot = OrderAwareBatchEnv(problem).snapshot()
    eligible = tuple(
        plan for plan in snapshot.candidates
        if allow_orders or plan.is_fixed_order
    )
    request_ids = tuple(sorted({plan.request_id for plan in eligible}))
    by_request = {
        request_id: tuple(
            plan for plan in eligible if plan.request_id == request_id
        )
        for request_id in request_ids
    }
    required = ceil(threshold * len(seeds))
    best = -1
    for choices in product(*((None, *by_request[r]) for r in request_ids)):
        plans = tuple(plan for plan in choices if plan is not None)
        ids = tuple(plan.plan_id for plan in plans)
        counts = {plan.request_id: 0 for plan in plans}
        for seed in seeds:
            result = simulate_order_batch(
                snapshot.problem.with_physics_seed(seed), ids,
                record_traces=False,
            )
            for request_id in result.completed:
                counts[request_id] += 1
        if all(value >= required for value in counts.values()):
            best = max(best, len(ids))
    return best


class NominalOrderMilpTests(unittest.TestCase):
    def test_static_snapshot_milp_is_direct_and_model_optimal(self):
        snapshot = OrderAwareBatchEnv(
            _general_problem(direct_probability=1.0)
        ).snapshot()
        planner = MilpStaticPathOrderPlanner((0,))

        with patch.object(
            order_milp_module,
            "simulate_order_batch",
            side_effect=AssertionError(
                "static planning must not invoke the physical executor"
            ),
        ):
            selected = planner.select(snapshot)

        self.assertEqual(selected, ("r:p",))
        self.assertEqual(planner.last_objective, 1)
        self.assertTrue(planner.last_solution.proven_optimal)
        self.assertFalse(planner.last_solution.certified_optimal)
        self.assertEqual(planner.last_evaluations, 0)
        self.assertIn("cp-sat", planner.last_solution.backend)

    def test_static_path_order_model_dominates_path_only_model(self):
        snapshot = OrderAwareBatchEnv(_general_problem()).snapshot()
        path = MilpStaticPathPlanner((3, 7), chance_threshold=0.5)
        order = MilpStaticPathOrderPlanner(
            (3, 7), chance_threshold=0.5
        )

        path.select(snapshot)
        order.select(snapshot)

        self.assertGreaterEqual(order.last_objective, path.last_objective)
        self.assertEqual(path.last_solution.required_scenarios, 1)
        self.assertEqual(order.last_solution.required_scenarios, 1)

    def test_static_snapshot_probability_screen_can_reject_a_plan(self):
        snapshot = OrderAwareBatchEnv(
            _general_problem(direct_probability=0.0)
        ).snapshot()
        planner = MilpStaticPathOrderPlanner((0,))

        selected = planner.select(snapshot)

        self.assertEqual(selected, ())
        self.assertEqual(planner.last_objective, 0)
        self.assertEqual(planner.last_solution.filtered_candidates, 1)

    def test_decimal_chance_threshold_does_not_overceil(self):
        planner = MilpNominalPathOrderPlanner(
            range(100), chance_threshold=0.07
        )

        self.assertEqual(planner.required_scenarios, 7)

    def test_master_requests_and_verifies_zero_mip_gap(self):
        snapshot = OrderAwareBatchEnv(
            _general_problem(direct_probability=1.0)
        ).snapshot()
        real_milp = order_milp_module.milp
        planner = MilpNominalPathOrderPlanner((0,))

        with patch.object(
            order_milp_module, "milp", wraps=real_milp
        ) as mocked_milp:
            selected = planner.select(snapshot)

        self.assertEqual(selected, ("r:p",))
        self.assertEqual(mocked_milp.call_count, 1)
        for call in mocked_milp.call_args_list:
            self.assertEqual(call.kwargs["options"]["mip_rel_gap"], 0.0)
        self.assertEqual(planner.last_solution.milp_solves, 1)
        self.assertIn("cp-sat", planner.last_solution.backend)

    def test_nonzero_reported_mip_gap_is_rejected(self):
        snapshot = OrderAwareBatchEnv(
            _general_problem(direct_probability=1.0)
        ).snapshot()
        real_milp = order_milp_module.milp

        def inexact_milp(*args, **kwargs):
            result = real_milp(*args, **kwargs)
            result.mip_gap = 1e-4
            return result

        with patch.object(
            order_milp_module, "milp", side_effect=inexact_milp
        ):
            with self.assertRaisesRegex(
                RuntimeError, "not proven optimal"
            ):
                MilpNominalPathOrderPlanner((0,)).select(snapshot)

    def test_open_reported_dual_bound_is_rejected(self):
        snapshot = OrderAwareBatchEnv(
            _general_problem(direct_probability=1.0)
        ).snapshot()
        real_milp = order_milp_module.milp

        def open_bound_milp(*args, **kwargs):
            result = real_milp(*args, **kwargs)
            result.mip_gap = 0.0
            result.mip_dual_bound = result.fun - 1.0
            return result

        with patch.object(
            order_milp_module, "milp", side_effect=open_bound_milp
        ):
            with self.assertRaisesRegex(
                RuntimeError, "lacks a closed objective bound"
            ):
                MilpNominalPathOrderPlanner((0,)).select(snapshot)

    def test_matches_bruteforce_on_general_stochastic_snapshot(self):
        problem = _general_problem()
        snapshot = OrderAwareBatchEnv(problem).snapshot()
        seeds = (3, 7)
        planner = MilpNominalPathOrderPlanner(
            seeds, chance_threshold=0.5
        )

        selected = planner.select(snapshot)

        self.assertEqual(
            planner.last_objective,
            _finite_scenario_optimum(
                problem, allow_orders=True, seeds=seeds, threshold=0.5
            ),
        )
        self.assertEqual(planner.last_objective, len(selected))
        self.assertTrue(planner.last_proven_optimal)
        self.assertTrue(planner.last_solution.proven_optimal)
        self.assertFalse(planner.last_solution.certified_optimal)

    def test_path_projection_uses_only_canonical_orders(self):
        snapshot = OrderAwareBatchEnv(_general_problem()).snapshot()
        planner = MilpNominalPathPlanner((3,))
        selected = planner.select(snapshot)
        lookup = {plan.plan_id: plan for plan in snapshot.candidates}

        self.assertTrue(all(lookup[plan_id].is_fixed_order for plan_id in selected))
        self.assertEqual(planner.last_objective, len(selected))

    def test_probabilities_are_not_replaced_by_one(self):
        impossible = MilpNominalPathOrderPlanner((0,))
        possible = MilpNominalPathOrderPlanner((0,))

        impossible_ids = impossible.select(
            OrderAwareBatchEnv(_general_problem(direct_probability=0.0)).snapshot()
        )
        possible_ids = possible.select(
            OrderAwareBatchEnv(_general_problem(direct_probability=1.0)).snapshot()
        )

        self.assertEqual(impossible_ids, ())
        self.assertEqual(impossible.last_objective, 0)
        self.assertEqual(possible_ids, ("r:p",))
        self.assertEqual(possible.last_objective, 1)

    def test_inactive_candidate_rejects_the_snapshot(self):
        expired = OrderPlan(
            "r:expired",
            "r",
            (0, 1),
            (),
            arrival_slot=0,
            decision_slot=2,
            deadline_slot=2,
        )
        problem = OrderBatchProblem.create(
            candidates=(expired,),
            node_capacity={0: 1, 1: 1},
            links=(OrderLinkSpec(0, 1, generation_probability=1.0),),
            config=OrderCoreConfig(
                slot_duration_ps=1_000,
                generation_interval_ps=1_000,
                swap_service_ps=1_000,
                generation_probability=1.0,
                swap_probability=1.0,
                slot_id=2,
            ),
        )
        snapshot = OrderAwareBatchEnv(problem).snapshot()

        for planner_type in (
            MilpNominalPathPlanner,
            MilpNominalPathOrderPlanner,
        ):
            with self.subTest(planner=planner_type.__name__):
                with self.assertRaisesRegex(
                    ValueError,
                    r"inactive candidates.*r:expired\(decision_slot=2, "
                    r"deadline_slot=2\)",
                ):
                    planner_type((0,)).select(snapshot)

    def test_seed_zero_formal_catalogue_filters_only_proven_impossibility(self):
        config = WaxmanOrderConfig(
            node_count=20,
            average_degree=4,
            request_count=100,
            arrival_rate=100 / 30,
            episode_steps=30,
            request_ttl_slots=5,
            min_hops=2,
            max_hops=6,
            candidate_paths=4,
            order_variants_per_path=4,
            candidate_request_cap=None,
            node_memory_cap=2,
            epr_ttl_slots=3,
            slot_duration_ps=4_000,
            generation_interval_ps=1_000,
            swap_service_ps=1_000,
            memory_reset_ps=100,
            swap_probability=0.9,
        )
        episode = make_waxman_order_episode(config, seed=0)
        pending = tuple(request.request_id for request in episode.requests)
        request_ids = episode.eligible_request_ids(pending, slot=0)
        self.assertEqual(request_ids, ("r0", "r1", "r2", "r3"))
        self.assertTrue(all(
            len(episode.paths[request_id]) == 4
            for request_id in request_ids
        ))
        problem = episode.problem_for_slot(
            request_ids, slot=0, physics_seed=0
        )
        snapshot = OrderAwareBatchEnv(problem).snapshot()
        raw_ids = tuple(plan.plan_id for plan in snapshot.candidates)

        fixed = MilpNominalPathPlanner((0,))
        joint = MilpNominalPathOrderPlanner((0,))
        fixed_ids = fixed.select(snapshot)
        joint_ids = joint.select(snapshot)

        self.assertEqual(
            tuple(plan.plan_id for plan in snapshot.candidates), raw_ids
        )
        self.assertEqual(fixed.last_solution.eligible_candidates, 16)
        self.assertEqual(joint.last_solution.eligible_candidates, 33)
        self.assertEqual(fixed.last_solution.filtered_candidates, 6)
        self.assertEqual(joint.last_solution.filtered_candidates, 16)
        self.assertEqual(fixed.last_solution.static_upper_bound, 3)
        self.assertEqual(joint.last_solution.static_upper_bound, 3)
        self.assertEqual(fixed.last_objective, 3)
        self.assertEqual(joint.last_objective, 3)
        self.assertNotIn("r1", {
            next(
                plan.request_id for plan in snapshot.candidates
                if plan.plan_id == plan_id
            )
            for plan_id in (*fixed_ids, *joint_ids)
        })

    def test_matching_inventory_prevents_unsafe_generation_filter(self):
        plan = OrderPlan("r:p", "r", (0, 1), ())
        without_inventory = OrderBatchProblem.create(
            candidates=(plan,),
            node_capacity={0: 1, 1: 1},
            links=(OrderLinkSpec(0, 1, generation_probability=0.0),),
            config=OrderCoreConfig(
                slot_duration_ps=1_000,
                generation_interval_ps=1_000,
                swap_service_ps=1_000,
                generation_probability=0.0,
                swap_probability=1.0,
            ),
        )
        with_inventory = replace(
            without_inventory,
            initial_inventory=(
                OrderStoredPair("stored", 0, 1, 0, 3),
            ),
        )

        self.assertFalse(
            order_milp_module._plan_hard_possible_in_scenario(
                without_inventory, plan, 0
            )
        )
        self.assertTrue(
            order_milp_module._plan_hard_possible_in_scenario(
                with_inventory, plan, 0
            )
        )
        selected = MilpNominalPathOrderPlanner((0,)).select(
            OrderAwareBatchEnv(with_inventory).snapshot()
        )
        self.assertEqual(selected, ("r:p",))

    def test_preloaded_request_does_not_require_generation_draws(self):
        plan = OrderPlan("r:p", "r", (0, 1, 2), (1,))
        problem = OrderBatchProblem.create(
            candidates=(plan,),
            node_capacity={0: 1, 1: 2, 2: 1},
            links=(
                OrderLinkSpec(0, 1, generation_probability=0.0),
                OrderLinkSpec(1, 2, generation_probability=0.0),
            ),
            required_requests=("r",),
            preloaded_requests=("r",),
            config=OrderCoreConfig(
                slot_duration_ps=1_000,
                generation_interval_ps=1_000,
                swap_service_ps=1_000,
                memory_reset_ps=0,
                generation_probability=0.0,
                swap_probability=1.0,
                bsm_capacity_per_node=1,
            ),
        )
        planner = MilpNominalPathOrderPlanner((0,))

        self.assertTrue(
            order_milp_module._plan_hard_possible_in_scenario(
                problem, plan, 0
            )
        )
        selected = planner.select(OrderAwareBatchEnv(problem).snapshot())
        self.assertEqual(selected, ("r:p",))
        self.assertEqual(planner.last_solution.filtered_candidates, 0)
        self.assertEqual(planner.last_solution.static_upper_bound, 1)
        self.assertTrue(planner.last_solution.proven_optimal)

    def test_fixed_swap_failure_is_hard_filtered(self):
        plan = OrderPlan("r:p", "r", (0, 1, 2), (1,))
        problem = OrderBatchProblem.create(
            candidates=(plan,),
            node_capacity={0: 1, 1: 2, 2: 1},
            links=(
                OrderLinkSpec(0, 1, generation_probability=1.0),
                OrderLinkSpec(1, 2, generation_probability=1.0),
            ),
            config=OrderCoreConfig(
                slot_duration_ps=1_000,
                generation_interval_ps=1_000,
                swap_service_ps=1_000,
                generation_probability=1.0,
                swap_probability=0.0,
            ),
        )
        planner = MilpNominalPathOrderPlanner((0,))

        self.assertFalse(
            order_milp_module._plan_hard_possible_in_scenario(
                problem, plan, 0
            )
        )
        self.assertEqual(
            planner.select(OrderAwareBatchEnv(problem).snapshot()), ()
        )
        self.assertEqual(planner.last_solution.filtered_candidates, 1)

    def test_chance_filter_counts_possible_scenarios(self):
        problem = _general_problem(direct_probability=0.5)
        plan = problem.candidates[0]
        successful: list[int] = []
        failed: list[int] = []
        for seed in range(100):
            result = simulate_order_batch(
                problem.with_physics_seed(seed),
                (plan.plan_id,),
                record_traces=False,
            )
            (successful if result.completed else failed).append(seed)
            if len(successful) >= 2 and len(failed) >= 2:
                break
        self.assertGreaterEqual(len(successful), 2)
        self.assertGreaterEqual(len(failed), 2)

        rejected = MilpNominalPathOrderPlanner(
            (successful[0], failed[0], failed[1]),
            chance_threshold=0.5,
        )
        accepted = MilpNominalPathOrderPlanner(
            (successful[0], successful[1], failed[0]),
            chance_threshold=0.5,
        )
        snapshot = OrderAwareBatchEnv(problem).snapshot()

        self.assertEqual(rejected.select(snapshot), ())
        self.assertEqual(rejected.last_solution.filtered_candidates, 1)
        self.assertEqual(accepted.select(snapshot), ("r:p",))
        self.assertEqual(accepted.last_solution.filtered_candidates, 0)

    def test_known_feasible_incumbent_is_revalidated(self):
        feasible_snapshot = OrderAwareBatchEnv(
            _general_problem(direct_probability=1.0)
        ).snapshot()
        impossible_snapshot = OrderAwareBatchEnv(
            _general_problem(direct_probability=0.0)
        ).snapshot()

        with self.assertRaisesRegex(ValueError, "ineligible plan IDs"):
            MilpNominalPathOrderPlanner((0,)).select_with_incumbent(
                feasible_snapshot, ("missing-plan",)
            )
        with self.assertRaisesRegex(ValueError, "physical scenario oracle"):
            MilpNominalPathOrderPlanner((0,)).select_with_incumbent(
                impossible_snapshot, ("r:p",)
            )

        equal = MilpNominalPathOrderPlanner((0,))
        self.assertEqual(
            equal.select_with_incumbent(feasible_snapshot, ("r:p",)),
            ("r:p",),
        )
        self.assertEqual(equal.last_solution.static_upper_bound, 1)
        self.assertEqual(equal.last_solution.enumerated_assignments, 1)
        self.assertEqual(equal.last_solution.evaluations, 1)

        required_problem = OrderBatchProblem.create(
            candidates=(OrderPlan("r:p", "r", (0, 1), ()),),
            node_capacity={0: 1, 1: 1},
            links=(OrderLinkSpec(0, 1, generation_probability=1.0),),
            required_requests=("r",),
            config=OrderCoreConfig(
                slot_duration_ps=1_000,
                generation_interval_ps=1_000,
                swap_service_ps=1_000,
                generation_probability=1.0,
                swap_probability=1.0,
            ),
        )
        with self.assertRaisesRegex(ValueError, "omits required requests"):
            MilpNominalPathOrderPlanner((0,)).select_with_incumbent(
                OrderAwareBatchEnv(required_problem).snapshot(), ()
            )

    def test_late_edge_uses_its_selected_order_deadline(self):
        plan = OrderPlan("r:p", "r", (0, 1, 2, 3), (1, 2))
        late_edge = plan.elementary_edges[-1]
        planning_seed = None
        probability = None
        for seed in range(1_000):
            draws = tuple(
                order_milp_module._planner_uniform(
                    seed, "generation", 0, "r", late_edge, attempt
                )
                for attempt in range(3)
            )
            if draws[2] < min(draws[:2]):
                planning_seed = seed
                probability = (draws[2] + min(draws[:2])) / 2
                break
        self.assertIsNotNone(planning_seed)
        assert probability is not None
        problem = OrderBatchProblem.create(
            candidates=(plan,),
            node_capacity={0: 1, 1: 2, 2: 2, 3: 1},
            links=(
                OrderLinkSpec(0, 1, generation_probability=1.0),
                OrderLinkSpec(1, 2, generation_probability=1.0),
                OrderLinkSpec(2, 3, generation_probability=probability),
            ),
            config=OrderCoreConfig(
                slot_duration_ps=3_000,
                generation_interval_ps=1_000,
                swap_service_ps=1_000,
                memory_reset_ps=0,
                generation_probability=probability,
                swap_probability=1.0,
                slot_id=0,
            ),
        )

        self.assertEqual(
            order_milp_module._edge_generation_attempt_limit(
                problem, plan, late_edge
            ),
            3,
        )
        self.assertTrue(
            order_milp_module._plan_hard_possible_in_scenario(
                problem, plan, planning_seed
            )
        )
        result = simulate_order_batch(
            problem.with_physics_seed(planning_seed),
            (plan.plan_id,),
            record_traces=False,
        )
        self.assertEqual(result.completed, ("r",))

    def test_hybrid_optimizer_matches_raw_bruteforce_grid(self):
        problem = _general_problem()
        snapshot = OrderAwareBatchEnv(problem).snapshot()
        for planner_type, allow_orders in (
            (MilpNominalPathPlanner, False),
            (MilpNominalPathOrderPlanner, True),
        ):
            for seeds, threshold in (
                ((0,), 1.0),
                ((3, 7), 0.5),
                ((1, 5, 9), 2 / 3),
            ):
                with self.subTest(
                    planner=planner_type.__name__,
                    seeds=seeds,
                    threshold=threshold,
                ):
                    planner = planner_type(
                        seeds, chance_threshold=threshold
                    )
                    selected = planner.select(snapshot)
                    expected = _finite_scenario_optimum(
                        problem,
                        allow_orders=allow_orders,
                        seeds=seeds,
                        threshold=threshold,
                    )
                    self.assertEqual(planner.last_objective, expected)
                    self.assertEqual(len(selected), expected)
                    self.assertTrue(planner.last_solution.proven_optimal)

    def test_incumbent_optimizer_matches_raw_bruteforce(self):
        problem = _general_problem()
        snapshot = OrderAwareBatchEnv(problem).snapshot()
        for seeds, threshold in (
            ((0,), 1.0),
            ((3, 7), 0.5),
            ((1, 5, 9), 2 / 3),
        ):
            with self.subTest(seeds=seeds, threshold=threshold):
                fixed = MilpNominalPathPlanner(
                    seeds, chance_threshold=threshold
                )
                fixed_ids = fixed.select(snapshot)
                joint = MilpNominalPathOrderPlanner(
                    seeds, chance_threshold=threshold
                )
                incumbent_ids = joint.select_with_incumbent(
                    snapshot, fixed_ids
                )
                expected = _finite_scenario_optimum(
                    problem,
                    allow_orders=True,
                    seeds=seeds,
                    threshold=threshold,
                )

                self.assertEqual(len(incumbent_ids), expected)
                self.assertEqual(joint.last_objective, expected)
                self.assertTrue(joint.last_solution.proven_optimal)

        # The lower bound must not suppress a strictly better noncanonical
        # action.  Seed 15 has fixed optimum one and joint optimum two.
        fixed = MilpNominalPathPlanner((15,))
        fixed_ids = fixed.select(snapshot)
        joint = MilpNominalPathOrderPlanner((15,))
        joint_ids = joint.select_with_incumbent(snapshot, fixed_ids)
        self.assertEqual(len(fixed_ids), 1)
        self.assertEqual(len(joint_ids), 2)
        self.assertGreater(len(joint_ids), len(fixed_ids))

    def test_parallel_oracle_matches_serial_order_and_scenario_counts(self):
        snapshot = OrderAwareBatchEnv(_general_problem()).snapshot()
        serial = MilpNominalPathOrderPlanner(
            (1, 2),
            chance_threshold=0.5,
            oracle_workers=1,
            oracle_batch_size=4,
        )
        expected_ids = serial.select(snapshot)
        expected_counts = serial.last_solution.scenario_completion_counts

        for workers in (2, 4):
            with self.subTest(workers=workers):
                parallel = MilpNominalPathOrderPlanner(
                    (1, 2),
                    chance_threshold=0.5,
                    oracle_workers=workers,
                    oracle_batch_size=4,
                )
                selected = parallel.select(snapshot)

                self.assertEqual(selected, expected_ids)
                self.assertEqual(parallel.last_objective, serial.last_objective)
                self.assertEqual(
                    parallel.last_solution.scenario_completion_counts,
                    expected_counts,
                )
                self.assertEqual(parallel.last_solution.static_upper_bound, 2)
                # No cardinality-two assignment is oracle-feasible.  Reaching
                # the one-request optimum therefore requires exhausting that
                # complete higher-cardinality CP-SAT enumeration first.
                self.assertEqual(len(selected), 1)
                self.assertTrue(parallel.last_solution.proven_optimal)

    def test_parallel_oracle_flushes_final_partial_batch_before_returning(self):
        snapshot = OrderAwareBatchEnv(_general_problem()).snapshot()
        serial = MilpNominalPathOrderPlanner(
            (15,), oracle_workers=1, oracle_batch_size=8
        )
        expected = serial.select(snapshot)
        parallel = MilpNominalPathOrderPlanner(
            (15,), oracle_workers=2, oracle_batch_size=8
        )

        selected = parallel.select(snapshot)

        # Cardinality two has only four CP-SAT assignments, fewer than the
        # configured batch size.  Its feasible noncanonical action can be
        # found only by flushing the callback's final partial batch.
        self.assertEqual(
            expected, ("r1:reverse", "r2:fixed")
        )
        self.assertEqual(selected, expected)
        self.assertEqual(parallel.last_objective, 2)
        self.assertTrue(parallel.last_solution.proven_optimal)

    def test_parallel_oracle_terminates_spawn_pool_on_exception(self):
        snapshot = OrderAwareBatchEnv(_general_problem()).snapshot()

        class ExplodingPool:
            def __init__(self) -> None:
                self.closed = False
                self.terminated = False
                self.joined = False

            def map(self, function, batches, *, chunksize):
                del function, batches, chunksize
                raise RuntimeError("synthetic worker failure")

            def close(self) -> None:
                self.closed = True

            def terminate(self) -> None:
                self.terminated = True

            def join(self) -> None:
                self.joined = True

        pool = ExplodingPool()

        class FakeSpawnContext:
            def Pool(self, **kwargs):
                self.kwargs = kwargs
                return pool

        context = FakeSpawnContext()
        planner = MilpNominalPathOrderPlanner(
            (2,), oracle_workers=2, oracle_batch_size=4
        )
        with patch.object(
            order_milp_module,
            "_nominal_spawn_context",
            return_value=context,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "synthetic worker failure"
            ):
                planner.select(snapshot)

        self.assertEqual(context.kwargs["processes"], 2)
        self.assertIs(
            context.kwargs["initializer"],
            order_milp_module._init_nominal_oracle_worker,
        )
        self.assertFalse(pool.closed)
        self.assertTrue(pool.terminated)
        self.assertTrue(pool.joined)

    def test_parallel_oracle_constructor_rejects_invalid_sizes(self):
        with self.assertRaisesRegex(ValueError, "oracle_workers"):
            MilpNominalPathOrderPlanner(oracle_workers=0)
        with self.assertRaisesRegex(ValueError, "oracle_batch_size"):
            MilpNominalPathOrderPlanner(oracle_batch_size=0)

    def test_four_by_four_waxman_snapshots_match_raw_bruteforce(self):
        config = WaxmanOrderConfig(
            node_count=10,
            average_degree=4,
            request_count=8,
            arrival_rate=2.0,
            episode_steps=4,
            request_ttl_slots=3,
            min_hops=2,
            max_hops=4,
            candidate_paths=4,
            order_variants_per_path=4,
            candidate_request_cap=None,
            node_memory_cap=2,
            slot_duration_ps=4_000,
            generation_interval_ps=1_000,
            swap_service_ps=1_000,
            memory_reset_ps=100,
            swap_probability=0.9,
            bsm_capacity_per_node=1,
            epr_ttl_slots=3,
        )
        for seed in (0, 2):
            episode = make_waxman_order_episode(config, seed)
            pending = tuple(
                request.request_id for request in episode.requests
            )
            request_ids = episode.eligible_request_ids(pending, slot=0)
            problem = episode.problem_for_slot(
                request_ids, slot=0, physics_seed=0
            )
            snapshot = OrderAwareBatchEnv(problem).snapshot()
            for planner_type, allow_orders in (
                (MilpNominalPathPlanner, False),
                (MilpNominalPathOrderPlanner, True),
            ):
                with self.subTest(
                    seed=seed, planner=planner_type.__name__
                ):
                    planner = planner_type((0,))
                    selected = planner.select(snapshot)
                    expected = _finite_scenario_optimum(
                        problem,
                        allow_orders=allow_orders,
                        seeds=(0,),
                        threshold=1.0,
                    )
                    self.assertEqual(len(selected), expected)
                    self.assertEqual(planner.last_objective, expected)

            fixed = MilpNominalPathPlanner((0,))
            fixed_ids = fixed.select(snapshot)
            incumbent_joint = MilpNominalPathOrderPlanner((0,))
            incumbent_ids = incumbent_joint.select_with_incumbent(
                snapshot, fixed_ids
            )
            incumbent_expected = _finite_scenario_optimum(
                problem,
                allow_orders=True,
                seeds=(0,),
                threshold=1.0,
            )
            self.assertEqual(len(incumbent_ids), incumbent_expected)
            self.assertEqual(
                incumbent_joint.last_objective, incumbent_expected
            )

    def test_planning_scenarios_ignore_hidden_environment_seed(self):
        problem = _general_problem()
        left = OrderAwareBatchEnv(replace(
            problem, config=replace(problem.config, seed=123)
        )).snapshot()
        right = OrderAwareBatchEnv(replace(
            problem, config=replace(problem.config, seed=456)
        )).snapshot()
        first = MilpNominalPathOrderPlanner((17,))
        second = MilpNominalPathOrderPlanner((17,))

        self.assertEqual(first.select(left), second.select(right))
        self.assertEqual(left.problem.config.seed, 0)
        self.assertEqual(right.problem.config.seed, 0)


if __name__ == "__main__":
    unittest.main()
