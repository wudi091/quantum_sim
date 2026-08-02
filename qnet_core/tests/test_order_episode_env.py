from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from qnet_core.order_core import OrderAwareBatchEnv, OrderLinkSpec
from qnet_core.order_episode_env import OrderEpisodeEnv
from qnet_core.order_gym_env import OrderGymConfig
from qnet_core.order_waxman import (
    WaxmanOrderConfig,
    WaxmanOrderEpisode,
    WaxmanOrderRequest,
    make_waxman_order_episode,
)
from qnet_core.order_waxman_benchmark import run_planner_episode


def _gym_config(
    *,
    max_nodes: int = 4,
    max_edges: int = 4,
    max_requests: int = 2,
    max_candidates: int = 4,
    max_hops: int = 2,
) -> OrderGymConfig:
    return OrderGymConfig(
        max_nodes=max_nodes,
        max_edges=max_edges,
        max_requests=max_requests,
        max_candidates=max_candidates,
        max_hops=max_hops,
    )


def _batch_episode() -> WaxmanOrderEpisode:
    """Two requests at slot 0, an empty slot, then one request at slot 2."""

    config = WaxmanOrderConfig(
        node_count=4,
        average_degree=2,
        target_link_probability=1.0,
        request_count=3,
        arrival_rate=1.0,
        request_ttl_slots=2,
        min_hops=1,
        max_hops=2,
        candidate_paths=1,
        order_variants_per_path=1,
        candidate_request_cap=2,
        node_memory_cap=2,
        slot_duration_ps=1_000,
        generation_interval_ps=1_000,
        swap_service_ps=1_000,
        memory_reset_ps=0,
        swap_probability=1.0,
        epr_ttl_slots=3,
    )
    return WaxmanOrderEpisode(
        seed=11,
        config=config,
        nodes=(0, 1, 2, 3),
        links=(
            OrderLinkSpec(0, 1, generation_probability=1.0),
            OrderLinkSpec(1, 2, generation_probability=1.0),
            OrderLinkSpec(2, 3, generation_probability=1.0),
        ),
        node_capacities=((0, 2), (1, 2), (2, 2), (3, 2)),
        positions=(
            (0, (0.0, 0.0)),
            (1, (1.0, 0.0)),
            (2, (2.0, 0.0)),
            (3, (3.0, 0.0)),
        ),
        requests=(
            WaxmanOrderRequest("r0", 0, 1, 0, 2, 1),
            WaxmanOrderRequest("r1", 2, 3, 0, 2, 1),
            WaxmanOrderRequest("r2", 1, 2, 2, 4, 1),
        ),
        request_paths=(
            ("r0", ((0, 1),)),
            ("r1", ((2, 3),)),
            ("r2", ((1, 2),)),
        ),
        topology_beta=1.0,
        link_alpha=1.0,
        horizon_slots=4,
    )


def _pruned_episode(*, cap: int | None) -> WaxmanOrderEpisode:
    """Three simultaneous requests with an optional considered-set cap."""

    base = _batch_episode()
    return replace(
        base,
        config=replace(base.config, candidate_request_cap=cap),
        requests=tuple(
            replace(request, arrival_slot=0, deadline_slot=2)
            for request in base.requests
        ),
        horizon_slots=2,
    )


def _inventory_episode() -> WaxmanOrderEpisode:
    """A slot is too short to swap, so generated pairs must carry forward."""

    config = WaxmanOrderConfig(
        node_count=3,
        average_degree=2,
        target_link_probability=1.0,
        request_count=1,
        arrival_rate=1.0,
        request_ttl_slots=3,
        min_hops=2,
        max_hops=2,
        candidate_paths=1,
        order_variants_per_path=1,
        candidate_request_cap=1,
        node_memory_cap=2,
        slot_duration_ps=500,
        generation_interval_ps=1_000,
        swap_service_ps=1_000,
        memory_reset_ps=0,
        swap_probability=1.0,
        epr_ttl_slots=3,
    )
    return WaxmanOrderEpisode(
        seed=12,
        config=config,
        nodes=(0, 1, 2),
        links=(
            OrderLinkSpec(0, 1, generation_probability=1.0),
            OrderLinkSpec(1, 2, generation_probability=1.0),
        ),
        node_capacities=((0, 2), (1, 2), (2, 2)),
        positions=((0, (0.0, 0.0)), (1, (1.0, 0.0)), (2, (2.0, 0.0))),
        requests=(WaxmanOrderRequest("carry", 0, 2, 0, 3, 2),),
        request_paths=(("carry", ((0, 1, 2),)),),
        topology_beta=1.0,
        link_alpha=1.0,
        horizon_slots=3,
    )


def _order_choice_episode() -> WaxmanOrderEpisode:
    """One request with two distinct complete swap-order candidates."""

    config = WaxmanOrderConfig(
        node_count=4,
        average_degree=2,
        target_link_probability=1.0,
        request_count=1,
        arrival_rate=1.0,
        request_ttl_slots=1,
        min_hops=3,
        max_hops=3,
        candidate_paths=1,
        order_variants_per_path=2,
        candidate_request_cap=1,
        node_memory_cap=2,
        slot_duration_ps=3_000,
        generation_interval_ps=1_000,
        swap_service_ps=1_000,
        memory_reset_ps=0,
        swap_probability=1.0,
        epr_ttl_slots=2,
    )
    return WaxmanOrderEpisode(
        seed=13,
        config=config,
        nodes=(0, 1, 2, 3),
        links=(
            OrderLinkSpec(0, 1, generation_probability=1.0),
            OrderLinkSpec(1, 2, generation_probability=1.0),
            OrderLinkSpec(2, 3, generation_probability=1.0),
        ),
        node_capacities=((0, 2), (1, 2), (2, 2), (3, 2)),
        positions=(
            (0, (0.0, 0.0)),
            (1, (1.0, 0.0)),
            (2, (2.0, 0.0)),
            (3, (3.0, 0.0)),
        ),
        requests=(WaxmanOrderRequest("order", 0, 3, 0, 1, 3),),
        request_paths=(("order", ((0, 1, 2, 3),)),),
        topology_beta=1.0,
        link_alpha=1.0,
        horizon_slots=1,
    )


class OrderEpisodeEnvTests(unittest.TestCase):
    @staticmethod
    def _request_ids(env: OrderEpisodeEnv) -> set[str]:
        if env.planning_snapshot is None:
            return set()
        return {plan.request_id for plan in env.planning_snapshot.candidates}

    @staticmethod
    def _plan_ids(
        env: OrderEpisodeEnv,
        request_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        if env.planning_snapshot is None:
            if request_ids:
                raise AssertionError("an empty slot has no selectable plans")
            return ()
        plans_by_request = {}
        for plan in env.planning_snapshot.candidates:
            plans_by_request.setdefault(plan.request_id, plan.plan_id)
        return tuple(plans_by_request[request_id] for request_id in request_ids)

    def _action(
        self,
        env: OrderEpisodeEnv,
        request_ids: tuple[str, ...] = (),
    ) -> np.ndarray:
        action = env.action_for_plan_ids(self._plan_ids(env, request_ids))
        self.assertEqual(action.shape, (env.gym_config.max_candidates,))
        return action

    def _assert_observation_contract(
        self,
        observation: dict[str, np.ndarray],
        config: OrderGymConfig,
    ) -> None:
        self.assertIn("episode_features", observation)
        self.assertIn("candidate_mask", observation)
        self.assertIn("action_mask", observation)
        self.assertEqual(observation["episode_features"].ndim, 1)
        self.assertEqual(
            observation["candidate_mask"].shape,
            (config.max_candidates,),
        )
        self.assertEqual(
            observation["action_mask"].shape,
            (config.max_candidates,),
        )

    def _assert_step_info_contract(self, info: dict[str, object]) -> None:
        for key in (
            "slot",
            "duration_ps",
            "eligible_request_ids",
            "considered_request_ids",
            "pruned_request_ids",
            "batch_request_ids",
            "completed",
            "expired",
            "selected_plan_ids",
            "inventory_start",
            "inventory_end",
            "physics_seed_visible",
        ):
            self.assertIn(key, info)
        self.assertIs(info["physics_seed_visible"], False)

    def test_full_active_backlog_is_default_and_pruning_is_explicit(self):
        config = _gym_config(
            max_requests=3,
            max_candidates=3,
            max_hops=2,
        )
        full = OrderEpisodeEnv(config, physics_seed_root=49_900)
        _, full_info = full.reset(options={"episode": _pruned_episode(cap=None)})
        self.assertEqual(
            full.current_eligible_request_ids, ("r0", "r1", "r2")
        )
        self.assertEqual(
            full.current_considered_request_ids,
            full.current_eligible_request_ids,
        )
        self.assertEqual(full.current_pruned_request_ids, ())
        self.assertEqual(full_info["considered_request_count"], 3)

        capped = OrderEpisodeEnv(config, physics_seed_root=49_901)
        _, reset_info = capped.reset(options={"episode": _pruned_episode(cap=2)})
        self.assertEqual(
            capped.current_eligible_request_ids, ("r0", "r1", "r2")
        )
        self.assertEqual(capped.current_considered_request_ids, ("r0", "r1"))
        self.assertEqual(capped.current_pruned_request_ids, ("r2",))
        self.assertEqual(reset_info["eligible_request_count"], 3)
        self.assertEqual(reset_info["considered_request_count"], 2)
        self.assertEqual(reset_info["pruned_request_count"], 1)

        _, _, _, _, step_info = capped.step(capped.action_for_plan_ids(()))
        self.assertEqual(step_info["eligible_request_ids"], ("r0", "r1", "r2"))
        self.assertEqual(step_info["considered_request_ids"], ("r0", "r1"))
        self.assertEqual(step_info["pruned_request_ids"], ("r2",))

    def test_episode_has_multiple_steps_and_each_step_is_one_slot(self) -> None:
        episode = _batch_episode()
        config = _gym_config()
        env = OrderEpisodeEnv(config, physics_seed_root=50_000)
        observation, reset_info = env.reset(options={"episode": episode})

        self.assertEqual(env.current_slot, 0)
        self.assertFalse(reset_info["physics_seed_visible"])
        self._assert_observation_contract(observation, config)

        actions = (
            self._action(env, ("r0", "r1")),
            None,
            None,
            None,
        )
        for executed_slot in range(episode.horizon_slots):
            action = actions[executed_slot]
            if action is None:
                request_ids = ("r2",) if executed_slot == 2 else ()
                action = self._action(env, request_ids)
            observation, _, terminated, truncated, info = env.step(action)
            self._assert_step_info_contract(info)
            self._assert_observation_contract(observation, config)
            self.assertEqual(info["slot"], executed_slot)
            self.assertEqual(info["duration_ps"], episode.config.slot_duration_ps)
            self.assertEqual(env.current_slot, executed_slot + 1)
            self.assertFalse(truncated)
            self.assertEqual(
                terminated,
                executed_slot + 1 == episode.horizon_slots,
            )

    def test_topology_is_fixed_across_slot_problems(self) -> None:
        episode = _batch_episode()
        env = OrderEpisodeEnv(_gym_config(), physics_seed_root=50_100)
        env.reset(options={"episode": episode})

        first_problem = env.current_problem
        self.assertIsNotNone(first_problem)
        static_topology = (
            first_problem.node_capacities,
            first_problem.links,
            first_problem.physical_edges,
        )
        self.assertIs(env.episode, episode)

        env.step(self._action(env, ("r0", "r1")))
        self.assertEqual(env.current_slot, 1)
        self.assertIsNone(env.current_problem)
        env.step(self._action(env))

        self.assertEqual(env.current_slot, 2)
        self.assertIs(env.episode, episode)
        self.assertIsNotNone(env.current_problem)
        self.assertEqual(
            (
                env.current_problem.node_capacities,
                env.current_problem.links,
                env.current_problem.physical_edges,
            ),
            static_topology,
        )

    def test_requests_become_visible_only_at_their_arrival_slot(self) -> None:
        env = OrderEpisodeEnv(_gym_config(), physics_seed_root=50_200)
        observation, _ = env.reset(options={"episode": _batch_episode()})

        self.assertEqual(self._request_ids(env), {"r0", "r1"})
        self.assertNotIn("r2", self._request_ids(env))
        self.assertTrue(np.any(observation["candidate_mask"]))
        for plan in env.planning_snapshot.candidates:
            self.assertEqual(plan.decision_slot, 0)
            self.assertLessEqual(plan.arrival_slot, 0)
            self.assertGreater(plan.deadline_slot, 0)

        observation, _, _, _, _ = env.step(
            self._action(env, ("r0", "r1"))
        )
        self.assertEqual(env.current_slot, 1)
        self.assertIsNone(env.planning_snapshot)
        self.assertFalse(np.any(observation["candidate_mask"]))

        env.step(self._action(env))
        self.assertEqual(env.current_slot, 2)
        self.assertEqual(self._request_ids(env), {"r2"})
        for plan in env.planning_snapshot.candidates:
            self.assertEqual(plan.arrival_slot, 2)
            self.assertEqual(plan.decision_slot, 2)
            self.assertGreater(plan.deadline_slot, 2)

    def test_inventory_is_carried_to_the_next_step_without_duplication(self) -> None:
        episode = _inventory_episode()
        config = _gym_config(
            max_nodes=3,
            max_edges=2,
            max_requests=1,
            max_candidates=1,
            max_hops=2,
        )
        env = OrderEpisodeEnv(config, physics_seed_root=50_300)
        env.reset(options={"episode": episode})

        _, _, terminated, truncated, info = env.step(
            self._action(env, ("carry",))
        )
        self._assert_step_info_contract(info)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        carried = tuple(env.inventory)
        self.assertEqual(len(carried), 2)
        self.assertEqual({pair.born_slot for pair in carried}, {0})
        self.assertEqual({pair.expires_slot for pair in carried}, {3})
        self.assertIsNotNone(env.current_problem)
        self.assertEqual(env.current_problem.initial_inventory, carried)

        first_pair_ids = {pair.pair_id for pair in carried}
        env.step(self._action(env, ("carry",)))
        second = tuple(env.inventory)
        self.assertEqual(len(second), 2)
        self.assertEqual({pair.pair_id for pair in second}, first_pair_ids)

    def test_same_seed_replays_the_same_exogenous_flow(self) -> None:
        workload = WaxmanOrderConfig(
            node_count=20,
            average_degree=4,
            request_count=5,
            arrival_rate=2.0,
            request_ttl_slots=3,
            min_hops=2,
            max_hops=4,
            candidate_paths=1,
            order_variants_per_path=1,
            candidate_request_cap=2,
            node_memory_cap=3,
            slot_duration_ps=2_000,
        )
        config = _gym_config(
            max_nodes=24,
            max_edges=128,
            max_requests=2,
            max_candidates=4,
            max_hops=4,
        )
        left = OrderEpisodeEnv(
            config,
            workload_config=workload,
            physics_seed_root=60_000,
        )
        right = OrderEpisodeEnv(
            config,
            workload_config=workload,
            physics_seed_root=60_000,
        )
        left_observation, left_info = left.reset(seed=17)
        right_observation, right_info = right.reset(seed=17)

        self.assertEqual(left.episode, right.episode)
        self.assertEqual(left_info, right_info)
        for key in ("episode_features", "candidate_mask", "action_mask"):
            np.testing.assert_array_equal(
                left_observation[key], right_observation[key]
            )

        for _ in range(left.episode.horizon_slots):
            left_action = left.action_for_plan_ids(())
            right_action = right.action_for_plan_ids(())
            np.testing.assert_array_equal(left_action, right_action)
            left_step = left.step(left_action)
            right_step = right.step(right_action)
            left_observation, left_reward, left_done, left_truncated, left_info = (
                left_step
            )
            (
                right_observation,
                right_reward,
                right_done,
                right_truncated,
                right_info,
            ) = right_step
            self.assertEqual(left_reward, right_reward)
            self.assertEqual(left_done, right_done)
            self.assertEqual(left_truncated, right_truncated)
            self.assertEqual(left_info, right_info)
            self.assertEqual(left.inventory, right.inventory)
            for key in ("episode_features", "candidate_mask", "action_mask"):
                np.testing.assert_array_equal(
                    left_observation[key], right_observation[key]
                )

    def test_default_physics_root_is_not_derived_from_episode_seed(self) -> None:
        episode = _batch_episode()
        with patch(
            "qnet_core.order_episode_env.secrets.randbits",
            side_effect=(101_001, 202_002),
        ):
            left = OrderEpisodeEnv(_gym_config())
            right = OrderEpisodeEnv(_gym_config())

        left.reset(options={"episode": episode})
        right.reset(options={"episode": episode})

        self.assertEqual(left.episode.seed, right.episode.seed)
        self.assertNotEqual(
            left.current_problem.config.seed,
            right.current_problem.config.seed,
        )
        self.assertEqual(left.planning_snapshot.problem.config.seed, 0)
        self.assertEqual(right.planning_snapshot.problem.config.seed, 0)

    def test_fixed_horizon_settles_100_requests_after_exactly_30_steps(
        self,
    ) -> None:
        workload = WaxmanOrderConfig(
            node_count=20,
            average_degree=6,
            request_count=100,
            arrival_rate=100 / 30,
            episode_steps=30,
            request_ttl_slots=10,
            min_hops=2,
            max_hops=6,
            candidate_paths=4,
            order_variants_per_path=4,
            node_memory_cap=4,
        )
        gym_config = _gym_config(
            max_nodes=24,
            max_edges=128,
            max_requests=100,
            max_candidates=1_600,
            max_hops=6,
        )
        episode = make_waxman_order_episode(workload, seed=9)
        env = OrderEpisodeEnv(gym_config, physics_seed_root=60_100)
        env.reset(options={"episode": episode})

        for slot in range(30):
            action = env.action_for_plan_ids(())
            _, _, terminated, truncated, _ = env.step(action)
            self.assertFalse(truncated)
            self.assertEqual(env.current_slot, slot + 1)
            self.assertEqual(terminated, slot == 29)

        metrics = env.metrics()
        self.assertEqual(metrics["episode_steps"], 30)
        self.assertEqual(metrics["completed_count"], 0)
        self.assertEqual(metrics["timeout_count"], 100)
        self.assertEqual(
            metrics["deadline_timeout_count"]
            + metrics["horizon_timeout_count"],
            100,
        )
        self.assertFalse(env.pending_request_ids)

    def test_batch_action_is_committed_atomically_once_per_slot(self) -> None:
        episode = _batch_episode()
        config = _gym_config()
        env = OrderEpisodeEnv(config, physics_seed_root=50_400)
        env.reset(options={"episode": episode})
        selected = self._plan_ids(env, ("r0", "r1"))
        action = env.action_for_plan_ids(selected)

        self.assertEqual(action.shape, (config.max_candidates,))
        self.assertEqual(int(np.count_nonzero(action)), 2)
        _, _, terminated, truncated, info = env.step(action)

        self._assert_step_info_contract(info)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(env.current_slot, 1)
        self.assertEqual(info["slot"], 0)
        self.assertEqual(info["duration_ps"], episode.config.slot_duration_ps)
        self.assertEqual(set(info["selected_plan_ids"]), set(selected))
        self.assertEqual(set(info["batch_request_ids"]), {"r0", "r1"})
        self.assertEqual(set(info["completed"]), {"r0", "r1"})

    def test_multi_hot_rejects_padding_and_two_orders_for_one_request(
        self,
    ) -> None:
        padded_env = OrderEpisodeEnv(_gym_config(), physics_seed_root=50_450)
        padded_env.reset(options={"episode": _batch_episode()})
        padded = np.zeros(padded_env.gym_config.max_candidates, dtype=np.int8)
        padded[-1] = 1
        with self.assertRaisesRegex(ValueError, "padded candidate"):
            padded_env.step(padded)

        order_config = _gym_config(
            max_nodes=4,
            max_edges=3,
            max_requests=1,
            max_candidates=2,
            max_hops=3,
        )
        order_env = OrderEpisodeEnv(order_config, physics_seed_root=50_451)
        order_env.reset(options={"episode": _order_choice_episode()})
        self.assertEqual(len(order_env.candidates), 2)
        duplicate_request = np.ones(2, dtype=np.int8)
        with self.assertRaisesRegex(ValueError, "at most one"):
            order_env.step(duplicate_request)

    def test_slot_execution_exactly_matches_direct_single_slot_core(self) -> None:
        episode = _batch_episode()
        env = OrderEpisodeEnv(_gym_config(), physics_seed_root=50_500)
        env.reset(options={"episode": episode})
        selected = self._plan_ids(env, ("r0", "r1"))
        current_problem = env.current_problem
        self.assertIsNotNone(current_problem)

        direct = OrderAwareBatchEnv(current_problem).commit(selected)
        _, _, _, _, info = env.step(env.action_for_plan_ids(selected))

        self.assertEqual(info["selected_plan_ids"], direct.selected_plan_ids)
        self.assertEqual(info["completed"], direct.completed)
        self.assertEqual(info["failed"], direct.failed)
        self.assertEqual(info["missed"], direct.missed)
        self.assertEqual(info["inventory_end"], direct.remaining_inventory)
        self.assertEqual(env.inventory, direct.remaining_inventory)
        self.assertEqual(env.last_result, direct)

    def test_terminal_observation_masks_actions_and_rejects_another_step(
        self,
    ) -> None:
        episode = _batch_episode()
        config = _gym_config()
        env = OrderEpisodeEnv(config, physics_seed_root=50_600)
        env.reset(options={"episode": episode})

        env.step(self._action(env, ("r0", "r1")))
        env.step(self._action(env))
        env.step(self._action(env, ("r2",)))
        observation, _, terminated, truncated, _ = env.step(
            self._action(env)
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(np.any(observation["action_mask"]))
        with self.assertRaisesRegex(RuntimeError, "termination"):
            env.step(np.zeros(config.max_candidates, dtype=np.int8))

    def test_benchmark_resets_planner_once_and_uses_one_call_per_active_slot(
        self,
    ) -> None:
        class CountingPlanner:
            def __init__(self) -> None:
                self.reset_seeds: list[int] = []
                self.selected_slots: list[int] = []
                self.snapshot_physics_seeds: list[int] = []
                self.last_objective: int | None = None
                self.last_solution = None

            def reset(self, episode_seed: int) -> None:
                self.reset_seeds.append(episode_seed)

            def select(self, snapshot) -> tuple[str, ...]:
                self.selected_slots.append(snapshot.problem.config.slot_id)
                self.snapshot_physics_seeds.append(snapshot.problem.config.seed)
                first_by_request = {}
                for plan in snapshot.candidates:
                    first_by_request.setdefault(plan.request_id, plan.plan_id)
                self.last_objective = len(first_by_request)
                self.last_solution = type(
                    "NominalSolution", (), {"certified_optimal": True}
                )()
                return tuple(first_by_request.values())

        episode = _batch_episode()
        planner = CountingPlanner()

        planner_seed = 70_001
        row = run_planner_episode(
            episode,
            planner,
            physics_seed_root=80_001,
            planner_seed=planner_seed,
        )

        self.assertEqual(planner.reset_seeds, [planner_seed])
        self.assertNotEqual(planner_seed, episode.seed)
        self.assertEqual(planner.selected_slots, [0, 2])
        self.assertEqual(planner.snapshot_physics_seeds, [0, 0])
        self.assertEqual(row["planner_calls"], 2)
        self.assertEqual(row["episode_steps"], episode.horizon_slots)
        self.assertEqual(row["milp_model_objective_sum"], 3.0)
        self.assertEqual(row["milp_executor_completed_count"], 3)
        self.assertEqual(row["executor_completed_count"], 3)
        self.assertEqual(row["milp_model_minus_executor_completed"], 0.0)


if __name__ == "__main__":
    unittest.main()
