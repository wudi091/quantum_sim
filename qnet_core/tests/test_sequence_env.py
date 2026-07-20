import unittest

from qnet_core.env import SharedRoutingEnv
from qnet_core.gym_env import GymConfig, SequenceGymEnv
from qnet_core.planners import GreedyPlanner, QCASTPlanner, QDDCAPlanner, RandomPlanner
from qnet_core.reward import RewardConfig
from qnet_core.scenario import ScenarioConfig
from qnet_core.spec import EpisodeSpec, PhysicalConfig, RequestSpec


class SequenceEnvironmentTests(unittest.TestCase):
    @staticmethod
    def _action_reaching(env: SequenceGymEnv, node: int) -> int:
        return next(
            action for action, plan in enumerate(env.slots)
            if plan is not None and plan.reached_node == node
        )

    @staticmethod
    def _one_hop_action(env: SequenceGymEnv) -> int:
        return next(
            action for action, plan in enumerate(env.slots)
            if plan is not None and len(plan.route_nodes) == 2
        )

    @staticmethod
    def _completion_action(env: SequenceGymEnv) -> int:
        return next(
            action for action, plan in enumerate(env.slots)
            if plan is not None and plan.completes_request
        )

    def test_three_node_request_uses_shared_generation_and_settlement(self):
        env = SharedRoutingEnv(EpisodeSpec(
            seed=11,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r0", 0, 2, ttl=4),),
            horizon=8,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                memory_capacity=2,
            ),
        ))
        snapshot = env.snapshot()
        self.assertGreaterEqual(len(snapshot.candidates), 2)
        plan = next(item for item in snapshot.candidates if item.completes_request)
        result = env.commit((plan.plan_id,))
        self.assertEqual(result["completed_now"], 1)
        self.assertEqual(result["metrics"]["completion_rate"], 1.0)
        self.assertTrue(env.done)

    def test_same_seed_produces_same_initial_snapshot(self):
        spec = EpisodeSpec(
            seed=13,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r0", 0, 2, ttl=4),),
            horizon=8,
            physical=PhysicalConfig(generation_probability=0.5),
        )
        left, right = SharedRoutingEnv(spec).snapshot(), SharedRoutingEnv(spec).snapshot()
        self.assertEqual(left.resources, right.resources)
        self.assertEqual(left.candidates, right.candidates)

    def test_partial_plan_reports_frontier_progress_without_completion(self):
        env = SharedRoutingEnv(EpisodeSpec(
            seed=29,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r0", 0, 2, ttl=8),),
            horizon=8,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                memory_capacity=2,
            ),
        ))
        snapshot = env.snapshot()
        plan = next(
            item for item in snapshot.candidates
            if item.reached_node == 1 and not item.completes_request
        )
        result = env.commit((plan.plan_id,))
        self.assertEqual(result["completed_now"], 0)
        self.assertEqual(result["successful_plans_now"], 1)
        self.assertEqual(result["partial_plan_successes_now"], 1)
        self.assertEqual(result["progress_hops_now"], 1.0)
        self.assertEqual(env.requests["r0"].frontier, 1)
        self.assertEqual(env.metrics()["progress_hops"], 1.0)

    def test_failed_extension_reports_lost_frontier_progress(self):
        env = SharedRoutingEnv(EpisodeSpec(
            seed=31,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r0", 0, 2, ttl=8),),
            horizon=8,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=0.0,
                memory_capacity=2,
            ),
        ))
        first = next(
            item for item in env.snapshot().candidates
            if item.reached_node == 1 and not item.completes_request
        )
        env.commit((first.plan_id,))
        extension = next(
            item for item in env.snapshot().candidates
            if item.request_id == "r0" and item.reached_node == 2
        )
        result = env.commit((extension.plan_id,))
        self.assertEqual(result["failed_now"], 1)
        self.assertEqual(result["progress_hops_now"], -1.0)
        self.assertEqual(env.requests["r0"].frontier, 0)
        self.assertEqual(env.metrics()["progress_hops"], 0.0)

    def test_late_destination_plan_is_not_partial_success(self):
        env = SharedRoutingEnv(EpisodeSpec(
            seed=37,
            nodes=(0, 1, 2, 3),
            edges=((0, 1), (1, 2), (2, 3)),
            requests=(RequestSpec("r0", 0, 3, ttl=1),),
            horizon=4,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                memory_capacity=2,
            ),
        ))
        plan = next(item for item in env.snapshot().candidates if item.completes_request)
        result = env.commit((plan.plan_id,))
        self.assertEqual(result["successful_plans_now"], 1)
        self.assertEqual(result["completed_now"], 0)
        self.assertEqual(result["partial_plan_successes_now"], 0)
        self.assertEqual(result["expired_now"], 1)

    def test_batch_keeps_positive_and_lost_progress_separate(self):
        env = SharedRoutingEnv(EpisodeSpec(
            seed=41,
            nodes=(0, 1, 2, 3, 4),
            edges=((0, 1), (2, 3), (3, 4)),
            requests=(
                RequestSpec("short", 0, 1, ttl=8),
                RequestSpec("long", 2, 4, ttl=8),
            ),
            horizon=8,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=0.0,
                memory_capacity=2,
            ),
        ))
        first = next(
            item for item in env.snapshot().candidates
            if item.request_id == "long" and item.reached_node == 3
        )
        env.commit((first.plan_id,))
        snapshot = env.snapshot()
        short = next(item for item in snapshot.candidates if item.request_id == "short")
        extension = next(
            item for item in snapshot.candidates
            if item.request_id == "long" and item.reached_node == 4
        )
        result = env.commit((short.plan_id, extension.plan_id))
        self.assertEqual(result["progress_hops_now"], 0.0)
        self.assertEqual(result["positive_progress_hops_now"], 1.0)
        self.assertEqual(result["lost_progress_hops_now"], 1.0)

    def test_all_planners_only_select_from_same_snapshot(self):
        spec = EpisodeSpec(
            seed=17,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r0", 0, 2, ttl=4),),
            horizon=8,
            physical=PhysicalConfig(generation_probability=1.0),
        )
        env = SharedRoutingEnv(spec)
        snapshot = env.snapshot()
        before = snapshot.candidates
        valid = {plan.plan_id for plan in snapshot.candidates}
        for planner in (GreedyPlanner(), QCASTPlanner(), QDDCAPlanner(), RandomPlanner(3)):
            planner.reset(spec.seed)
            self.assertLessEqual(set(planner.select(snapshot)), valid)
            self.assertEqual(snapshot.candidates, before)

    def test_gym_wrapper_only_advances_on_stop(self):
        env = SequenceGymEnv(GymConfig(
            max_requests=2,
            max_candidates_per_request=3,
            max_hops=3,
            scenario=ScenarioConfig(
                request_count=1, min_hops=2, max_hops=2, ttl=4, horizon=6,
                arrival_rate=100.0,
                physical=PhysicalConfig(generation_probability=1.0, swap_probability=1.0),
            ),
            seed=19,
        ))
        observation, _ = env.reset(seed=19)
        action = int(next(index for index, legal in enumerate(observation["action_mask"][:-1]) if legal))
        _, _, _, _, info = env.step(action)
        self.assertEqual(info["phase"], "select")
        self.assertEqual(info["duration"], 0.0)
        self.assertEqual(env.core.time, 0)
        _, _, terminated, truncated, info = env.step(env.stop_action)
        self.assertFalse(truncated)
        self.assertGreaterEqual(env.core.time, 1)
        self.assertEqual(info["phase"], "execute")

    def test_gym_wrapper_allows_empty_wait_commit(self):
        env = SequenceGymEnv(GymConfig(
            max_requests=1,
            max_candidates_per_request=3,
            max_hops=2,
            scenario=ScenarioConfig(
                request_count=1, min_hops=2, max_hops=2, ttl=4, horizon=6,
                physical=PhysicalConfig(generation_probability=1.0, swap_probability=1.0),
            ),
            seed=191,
        ))
        observation, _ = env.reset(seed=191)
        self.assertTrue(observation["action_mask"][env.stop_action])
        _, _, _, _, info = env.step(env.stop_action)
        self.assertEqual(info["phase"], "execute")
        self.assertEqual(info["planning_slots"], 0)
        self.assertEqual(env.core.time, 1)

    def test_candidate_exposes_remaining_hops(self):
        env = SequenceGymEnv(GymConfig(
            max_requests=1,
            max_candidates_per_request=3,
            max_hops=3,
            scenario=ScenarioConfig(
                request_count=1, min_hops=3, max_hops=3, ttl=8, horizon=8,
                arrival_rate=100.0,
                physical=PhysicalConfig(generation_probability=1.0, swap_probability=1.0),
            ),
            seed=193,
        ))
        observation, _ = env.reset(seed=193)
        action = self._one_hop_action(env)
        plan = env.slots[action]
        assert plan is not None
        self.assertAlmostEqual(
            float(observation["candidate_features"][action, 9]),
            plan.remaining_hops / env.config.max_hops,
        )

    def test_potential_reward_values_partial_frontier_progress(self):
        env = SequenceGymEnv(GymConfig(
            max_requests=1,
            max_candidates_per_request=3,
            max_hops=2,
            scenario=ScenarioConfig(
                request_count=1, min_hops=2, max_hops=2, ttl=8, horizon=8,
                physical=PhysicalConfig(generation_probability=1.0, swap_probability=1.0),
            ),
            seed=43,
            reward=RewardConfig(
                potential_coef=1.0, completion_bonus=0.0,
                makespan_coef=0.0, failure_coef=0.0, timeout_coef=0.0,
            ),
        ))
        env.reset(seed=43)
        action = self._one_hop_action(env)
        _, selection_reward, _, _, info = env.step(action)
        self.assertEqual(selection_reward, 0.0)
        self.assertEqual(info["duration"], 0.0)
        _, reward, _, _, info = env.step(env.stop_action)
        self.assertAlmostEqual(reward, env.config.discount_gamma)
        self.assertEqual(info["progress_hops_now"], 1.0)
        self.assertAlmostEqual(info["progress_potential_delta"], 1.0)
        self.assertAlmostEqual(info["reward_progress"], env.config.discount_gamma)

    def test_potential_reward_penalizes_failed_frontier_reset(self):
        env = SequenceGymEnv(GymConfig(
            max_requests=1,
            max_candidates_per_request=3,
            max_hops=2,
            scenario=ScenarioConfig(
                request_count=1, min_hops=2, max_hops=2, ttl=8, horizon=8,
                arrival_rate=100.0,
                physical=PhysicalConfig(generation_probability=1.0, swap_probability=0.0),
            ),
            seed=47,
            reward=RewardConfig(
                potential_coef=1.0, completion_bonus=0.0,
                makespan_coef=0.0, failure_coef=0.0, timeout_coef=0.0,
            ),
        ))
        env.reset(seed=47)
        env.step(self._one_hop_action(env))
        env.step(env.stop_action)
        env.step(self._completion_action(env))
        _, reward, _, _, info = env.step(env.stop_action)
        self.assertAlmostEqual(reward, -1.0)
        self.assertEqual(info["lost_progress_hops_now"], 1.0)

    def test_discounted_potential_shaping_telescopes_across_frontier_reset(self):
        gamma = 0.9
        env = SequenceGymEnv(GymConfig(
            max_requests=1,
            max_candidates_per_request=3,
            max_hops=2,
            scenario=ScenarioConfig(
                request_count=1, min_hops=2, max_hops=2, ttl=8, horizon=8,
                arrival_rate=100.0,
                physical=PhysicalConfig(
                    generation_probability=1.0, swap_probability=0.0,
                ),
            ),
            seed=47,
            reward=RewardConfig(
                potential_coef=1.0, completion_bonus=0.0,
                makespan_coef=0.0, failure_coef=0.0, timeout_coef=0.0,
            ),
            discount_gamma=gamma,
        ))
        env.reset(seed=47)
        env.step(self._one_hop_action(env))
        _, advance_reward, _, _, advance_info = env.step(env.stop_action)
        env.step(self._completion_action(env))
        _, reset_reward, _, _, reset_info = env.step(env.stop_action)

        self.assertAlmostEqual(advance_info["progress_potential_delta"], 1.0)
        self.assertAlmostEqual(reset_info["progress_potential_delta"], -1.0)
        self.assertAlmostEqual(advance_reward, gamma)
        self.assertAlmostEqual(reset_reward, -1.0)
        self.assertAlmostEqual(
            advance_reward + gamma ** advance_info["duration"] * reset_reward,
            0.0,
        )

    def test_timeout_terminal_cancels_unfinished_progress(self):
        env = SequenceGymEnv(GymConfig(
            max_requests=1,
            max_candidates_per_request=3,
            max_hops=3,
            scenario=ScenarioConfig(
                request_count=1, min_hops=3, max_hops=3, ttl=1, horizon=4,
                arrival_rate=100.0,
                physical=PhysicalConfig(generation_probability=1.0, swap_probability=1.0),
            ),
            seed=59,
            reward=RewardConfig(
                potential_coef=1.0, completion_bonus=5.0,
                makespan_coef=0.0, timeout_coef=2.0,
            ),
        ))
        env.reset(seed=59)
        env.step(self._completion_action(env))
        _, reward, terminated, truncated, info = env.step(env.stop_action)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(reward, -2.0)
        self.assertEqual(info["progress_potential_after"], 0.0)

    def test_completion_after_deadline_is_timeout(self):
        env = SharedRoutingEnv(EpisodeSpec(
            seed=23,
            nodes=(0, 1, 2, 3),
            edges=((0, 1), (1, 2), (2, 3)),
            requests=(RequestSpec("r0", 0, 3, ttl=1),),
            horizon=4,
            physical=PhysicalConfig(generation_probability=1.0, swap_probability=1.0),
        ))
        plan = next(item for item in env.snapshot().candidates if item.completes_request)
        result = env.commit((plan.plan_id,))
        self.assertEqual(result["metrics"]["completion_rate"], 0.0)
        self.assertEqual(result["metrics"]["timeout_rate"], 1.0)

    def test_qcast_width_two_delivers_two_pairs_through_public_phases(self):
        env = SharedRoutingEnv(EpisodeSpec(
            seed=71,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r0", 0, 2, ttl=8, demand_pairs=2),),
            horizon=8,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                memory_capacity=2,
                node_memory_capacity=4,
                max_width=2,
            ),
        ), candidate_count=6)
        planner = QCASTPlanner()
        planner.reset(71)
        allocation = env.snapshot()
        self.assertEqual(allocation.phase, "allocate")
        selected = planner.select(allocation)
        self.assertEqual(len(selected), 1)
        self.assertEqual(env._candidates[selected[0]].width, 2)
        allocation_result = env.commit(selected)
        self.assertEqual(allocation_result["duration"], 0.0)
        recovery = env.snapshot()
        self.assertEqual(recovery.phase, "recover")
        selected = planner.select(recovery)
        self.assertEqual(env._candidates[selected[0]].width, 2)
        result = env.commit(selected)
        self.assertEqual(result["delivered_pairs_now"], 2)
        self.assertEqual(env.metrics()["delivered_pairs"], 2.0)
        self.assertEqual(env.metrics()["completion_rate"], 1.0)
        self.assertEqual(env.phase, "allocate")

    def test_width_claims_enforce_internal_node_memory(self):
        env = SharedRoutingEnv(EpisodeSpec(
            seed=73,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r0", 0, 2, ttl=8, demand_pairs=2),),
            horizon=8,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                memory_capacity=2,
                node_memory_capacity=3,
                max_width=2,
            ),
        ), candidate_count=6)
        snapshot = env.snapshot()
        wide = next(
            plan.plan_id for plan in snapshot.candidates
            if plan.completes_request and plan.width == 2
        )
        with self.assertRaisesRegex(ValueError, "node 1 memory capacity exceeded"):
            env.commit((wide,))

    def test_recovery_can_use_shared_surplus_detour(self):
        env = SharedRoutingEnv(EpisodeSpec(
            seed=79,
            nodes=(0, 1, 2, 3),
            edges=((0, 1), (1, 3), (0, 2), (2, 3)),
            requests=(
                RequestSpec("r0", 0, 3, ttl=8),
                RequestSpec("r1", 0, 3, ttl=8),
            ),
            horizon=8,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                memory_capacity=2,
                node_memory_capacity=4,
                max_width=2,
            ),
        ), candidate_count=8)
        snapshot = env.snapshot()
        first = next(
            plan.plan_id for plan in snapshot.candidates
            if plan.request_id == "r0" and plan.route_nodes == (0, 1, 3) and plan.width == 1
        )
        second = next(
            plan.plan_id for plan in snapshot.candidates
            if plan.request_id == "r1" and plan.route_nodes == (0, 2, 3) and plan.width == 1
        )
        env.commit((first, second))
        failed_primary = next(
            pair_id for pair_id in env._allocated_pair_ids
            if set(env.backend.pairs[pair_id].endpoints) == {0, 1}
        )
        env.backend.discard_pair(failed_primary)
        env._prepared_time = None
        env._candidates = {}
        recovery = env.snapshot()
        detour = next(
            plan.plan_id for plan in recovery.candidates
            if plan.request_id == "r0" and plan.route_nodes == (0, 2, 3)
        )
        result = env.commit((detour,))
        self.assertEqual(result["delivered_pairs_now"], 1)
        self.assertEqual(env.requests["r0"].completed_at, 1)

    def test_gym_exposes_width_phase_and_claim_conflicts(self):
        env = SequenceGymEnv(GymConfig(
            max_requests=1,
            max_candidates_per_request=6,
            max_hops=2,
            scenario=ScenarioConfig(
                request_count=1, min_hops=2, max_hops=2, ttl=8, horizon=8,
                arrival_rate=100.0,
                physical=PhysicalConfig(
                    generation_probability=1.0,
                    swap_probability=1.0,
                    memory_capacity=2,
                    node_memory_capacity=4,
                    max_width=2,
                ),
                demand_pairs=2,
            ),
            seed=83,
        ))
        observation, _ = env.reset(seed=83)
        self.assertEqual(float(observation["global_features"][8]), 1.0)
        wide = next(
            action for action, plan in enumerate(env.slots)
            if plan is not None and plan.completes_request and plan.width == 2
        )
        self.assertEqual(float(observation["candidate_features"][wide, 11]), 1.0)
        observation, _, _, _, _ = env.step(wide)
        self.assertFalse(any(observation["action_mask"][:-1]))
        observation, _, _, _, info = env.step(env.stop_action)
        self.assertEqual(info["phase_after"], "recover")
        self.assertEqual(float(observation["global_features"][9]), 1.0)
        recovery = next(
            action for action, plan in enumerate(env.slots)
            if plan is not None and plan.width == 2
        )
        self.assertEqual(float(observation["candidate_features"][recovery, 10]), 1.0)


if __name__ == "__main__":
    unittest.main()
