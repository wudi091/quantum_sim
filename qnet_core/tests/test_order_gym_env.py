import unittest

import numpy as np

from qnet_core.order_core import (
    OrderAwareBatchEnv,
    OrderBatchProblem,
    OrderCoreConfig,
    OrderPlan,
    OrderStoredPair,
)
from qnet_core.order_gym_env import OrderGymConfig, OrderGymEnv
from qnet_core.order_planners import SAAPathOrderPlanner
from qnet_core.order_scenarios import make_order_counterexample
from qnet_core.order_scenarios import make_seeded_hotspot_problem


class OrderGymEnvTests(unittest.TestCase):
    def setUp(self):
        self.problem = make_order_counterexample(hotspot_capacity=2, seed=913)
        self.env = OrderGymEnv(OrderGymConfig(
            max_nodes=16,
            max_edges=16,
            max_requests=4,
            max_candidates=16,
            max_hops=5,
        ))

    def test_observation_exposes_graph_and_complete_order_tensors(self):
        observation, info = self.env.reset(options={"problem": self.problem})

        self.assertEqual(info["phase"], "reset")
        self.assertEqual(observation["node_features"].shape, (16, 8))
        self.assertEqual(observation["edge_index"].shape, (2, 16))
        self.assertEqual(observation["candidate_features"].shape, (16, 10))
        self.assertEqual(observation["candidate_path_nodes"].shape, (16, 6))
        self.assertEqual(observation["candidate_order_nodes"].shape, (16, 4))
        self.assertTrue(np.any(observation["candidate_order_mask"]))
        self.assertEqual(self.env.planning_snapshot.problem.config.seed, 0)
        self.assertEqual(self.env.problem.config.seed, 913)

    def test_stop_is_masked_until_required_request_has_an_order(self):
        observation, _ = self.env.reset(options={"problem": self.problem})
        self.assertFalse(observation["action_mask"][self.env.stop_action])

        main = next(
            plan for plan in self.env.candidates if plan.request_id == "R1"
        )
        observation, reward, terminated, truncated, info = self.env.step(
            self.env.action_for_plan(main.plan_id)
        )

        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["duration_ps"], 0)
        self.assertTrue(observation["action_mask"][self.env.stop_action])
        self.assertFalse(any(
            observation["action_mask"][self.env.action_for_plan(plan.plan_id)]
            for plan in self.env.candidates if plan.request_id == "R1"
        ))

    def test_planner_and_future_policy_share_identical_stop_execution(self):
        planner = SAAPathOrderPlanner()

        direct = OrderAwareBatchEnv(self.problem)
        selected = planner.select(direct.snapshot())
        direct_result = direct.commit(selected)

        self.env.reset(options={"problem": self.problem})
        selected = planner.select(self.env.planning_snapshot)
        for plan_id in selected:
            observation, reward, terminated, truncated, info = self.env.step(
                self.env.action_for_plan(plan_id)
            )
            self.assertEqual(reward, 0.0)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["duration_ps"], 0)
        _, reward, terminated, truncated, info = self.env.step(
            self.env.stop_action
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertGreater(reward, 0.0)
        self.assertEqual(info["completed_count"], 3)
        self.assertEqual(self.env.core.result, direct_result)

    def test_terminal_environment_masks_every_action(self):
        self.env.reset(options={"problem": self.problem})
        selected = SAAPathOrderPlanner().select(self.env.planning_snapshot)
        for plan_id in selected:
            self.env.step(self.env.action_for_plan(plan_id))
        observation, _, _, _, _ = self.env.step(self.env.stop_action)

        self.assertFalse(np.any(observation["action_mask"]))
        with self.assertRaises(RuntimeError):
            self.env.step(self.env.stop_action)

    def test_hidden_physics_seed_cannot_change_planner_snapshot(self):
        first = make_seeded_hotspot_problem(
            1,
            generation_probability=0.7,
            swap_probability=0.8,
            physics_seed=111,
        )
        second = make_seeded_hotspot_problem(
            1,
            generation_probability=0.7,
            swap_probability=0.8,
            physics_seed=222,
        )
        left = OrderAwareBatchEnv(first).snapshot()
        right = OrderAwareBatchEnv(second).snapshot()
        planner = SAAPathOrderPlanner((1001, 1002, 1003))

        self.assertEqual(left, right)
        self.assertEqual(planner.select(left), planner.select(right))

    def test_inventory_occupancy_and_edge_readiness_are_observable(self):
        inventory = (
            OrderStoredPair("cd", "C", "D", 0, 3),
            OrderStoredPair("de", "D", "E", 0, 3),
        )
        problem = OrderBatchProblem.create(
            candidates=(OrderPlan(
                "next", "next", ("C", "D", "E"), ("D",),
                arrival_slot=1, decision_slot=1,
            ),),
            node_capacity={node: 2 for node in "CDE"},
            initial_inventory=inventory,
            config=OrderCoreConfig(slot_id=1, seed=777),
        )
        env = OrderGymEnv(OrderGymConfig(
            max_nodes=4,
            max_edges=4,
            max_requests=2,
            max_candidates=4,
            max_hops=2,
        ))
        observation, _ = env.reset(options={"problem": problem})

        occupancy = {
            node: observation["node_features"][env.node_index[node], 2]
            for node in env.nodes
        }
        readiness = {
            elementary_edge: observation["edge_features"][index, 1]
            for index, elementary_edge in enumerate(env.edges)
        }
        self.assertEqual(occupancy, {"C": 0.5, "D": 1.0, "E": 0.5})
        self.assertEqual(readiness, {("C", "D"): 1.0, ("D", "E"): 1.0})
        self.assertEqual(env.planning_snapshot.problem.initial_inventory, inventory)

        hidden_a = OrderAwareBatchEnv(problem.with_physics_seed(1)).snapshot()
        hidden_b = OrderAwareBatchEnv(problem.with_physics_seed(2)).snapshot()
        self.assertEqual(hidden_a, hidden_b)


if __name__ == "__main__":
    unittest.main()
