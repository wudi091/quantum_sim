import itertools
import unittest
from unittest.mock import patch
from dataclasses import replace

import networkx as nx
import numpy as np

from qnet_core.order_core import OrderAwareBatchEnv, OrderLinkSpec
from qnet_core.order_planners import (
    QCASTFixedOrderPlanner,
    QDDCAFixedOrderPlanner,
)
from qnet_core.order_gym_env import OrderGymConfig, OrderGymEnv
from qnet_core.order_episode_env import OrderEpisodeEnv
from qnet_core.order_waxman import (
    WaxmanOrderConfig,
    WaxmanOrderEpisode,
    WaxmanOrderRequest,
    make_waxman_order_episode,
)
from qnet_core.order_waxman_benchmark import (
    main as waxman_benchmark_main,
    run_planner_episode,
    run_suite,
)


class WaxmanOrderTests(unittest.TestCase):
    def setUp(self):
        self.config = WaxmanOrderConfig(
            node_count=30,
            average_degree=4,
            request_count=20,
            arrival_rate=2.0,
            request_ttl_slots=6,
            min_hops=2,
            max_hops=5,
            candidate_paths=2,
            order_variants_per_path=2,
            candidate_request_cap=3,
            node_memory_cap=3,
            slot_duration_ps=5_000,
        )

    def test_core_defaults_use_formal_four_by_four_full_request_scope(self):
        config = WaxmanOrderConfig()

        self.assertEqual(config.candidate_paths, 4)
        self.assertEqual(config.order_variants_per_path, 4)
        self.assertIsNone(config.candidate_request_cap)
        self.assertEqual(
            OrderEpisodeEnv._default_gym_config(config).max_candidates,
            config.request_count * 16,
        )

    def test_waxman_benchmark_cli_defaults_to_formal_four_by_four_scope(self):
        class ParserInspected(Exception):
            pass

        def inspect_defaults(parser, *args, **kwargs):
            defaults = {
                action.dest: action.default for action in parser._actions
            }
            self.assertEqual(defaults["candidate_paths"], 4)
            self.assertEqual(defaults["order_variants"], 4)
            self.assertIsNone(defaults["candidate_request_cap"])
            raise ParserInspected

        with patch(
            "argparse.ArgumentParser.parse_args",
            autospec=True,
            side_effect=inspect_defaults,
        ):
            with self.assertRaises(ParserInspected):
                waxman_benchmark_main()

    def test_default_request_scope_exposes_the_full_active_backlog(self):
        config = replace(self.config, candidate_request_cap=None)
        episode = make_waxman_order_episode(config, seed=17)
        pending = tuple(request.request_id for request in episode.requests)
        by_slot = [
            episode.eligible_request_ids(pending, slot)
            for slot in range(episode.horizon_slots)
        ]
        eligible = max(by_slot, key=len)
        slot = by_slot.index(eligible)

        self.assertGreater(len(eligible), 3)
        self.assertEqual(
            episode.considered_request_ids(pending, slot), eligible
        )
        self.assertEqual(episode.active_request_ids(pending, slot), eligible)

    def test_candidate_request_cap_is_explicit_edf_pruning_only(self):
        episode = make_waxman_order_episode(self.config, seed=17)
        pending = tuple(request.request_id for request in episode.requests)
        slot = next(
            slot for slot in range(episode.horizon_slots)
            if len(episode.eligible_request_ids(pending, slot)) > 3
        )
        eligible = episode.eligible_request_ids(pending, slot)

        self.assertEqual(
            episode.considered_request_ids(pending, slot), eligible[:3]
        )
        # The compatibility name now means the full active set, never the
        # old silently truncated candidate prefix.
        self.assertEqual(episode.active_request_ids(pending, slot), eligible)

    def test_waxman_poisson_episode_is_reproducible_and_real(self):
        first = make_waxman_order_episode(self.config, seed=17)
        second = make_waxman_order_episode(self.config, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(len(first.nodes), 30)
        self.assertEqual(len(first.requests), 20)
        arrivals = [request.arrival_slot for request in first.requests]
        self.assertEqual(arrivals, sorted(arrivals))
        self.assertGreater(arrivals[-1], arrivals[0])
        self.assertTrue(all(
            request.source != request.destination
            and self.config.min_hops <= request.shortest_hops
            <= self.config.max_hops
            for request in first.requests
        ))
        graph = nx.Graph()
        graph.add_nodes_from(first.nodes)
        graph.add_edges_from(link.elementary_edge for link in first.links)
        self.assertTrue(nx.is_connected(graph))
        self.assertGreater(len({
            round(link.generation_probability, 6) for link in first.links
        }), 1)

    def test_fixed_30_step_conditional_poisson_workload(self):
        config = WaxmanOrderConfig(
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
            candidate_request_cap=4,
            node_memory_cap=4,
        )

        first = make_waxman_order_episode(config, seed=7)
        replay = make_waxman_order_episode(config, seed=7)
        different = make_waxman_order_episode(config, seed=8)

        self.assertEqual(first, replay)
        self.assertEqual(first.horizon_slots, 30)
        self.assertEqual(len(first.requests), 100)
        arrivals = [request.arrival_slot for request in first.requests]
        self.assertEqual(arrivals, sorted(arrivals))
        self.assertTrue(all(0 <= slot < 30 for slot in arrivals))
        self.assertNotEqual(
            (first.links, first.positions, first.requests, first.request_paths),
            (
                different.links,
                different.positions,
                different.requests,
                different.request_paths,
            ),
        )
        self.assertTrue(all(
            1 <= len(paths) <= 4 for _, paths in first.request_paths
        ))

    def test_four_paths_have_up_to_four_complete_unique_orders(self):
        paths = (
            (0, 1, 2, 3, 19),
            (0, 4, 5, 6, 19),
            (0, 7, 8, 9, 19),
            (0, 10, 11, 12, 19),
        )
        config = WaxmanOrderConfig(
            node_count=20,
            average_degree=2,
            request_count=1,
            arrival_rate=1.0,
            episode_steps=1,
            request_ttl_slots=1,
            min_hops=4,
            max_hops=4,
            candidate_paths=4,
            order_variants_per_path=4,
            candidate_request_cap=None,
            node_memory_cap=8,
            swap_probability=1.0,
        )
        links = tuple(
            OrderLinkSpec(left, right, generation_probability=1.0)
            for path in paths
            for left, right in zip(path, path[1:])
        )
        episode = WaxmanOrderEpisode(
            seed=31,
            config=config,
            nodes=tuple(range(20)),
            links=links,
            node_capacities=tuple((node, 8) for node in range(20)),
            positions=tuple((node, (float(node), 0.0)) for node in range(20)),
            requests=(WaxmanOrderRequest("r", 0, 19, 0, 1, 4),),
            request_paths=(("r", paths),),
            topology_beta=1.0,
            link_alpha=1.0,
            horizon_slots=1,
        )

        problem = episode.problem_for_slot(("r",), 0, physics_seed=99)

        self.assertEqual(config.candidate_paths, 4)
        self.assertEqual(config.order_variants_per_path, 4)
        self.assertIsNone(config.candidate_request_cap)
        self.assertEqual(len(problem.candidates), 16)
        for path in paths:
            schedules = {
                plan.schedule_key
                for plan in problem.candidates
                if plan.path == path
            }
            self.assertEqual(len(schedules), 4)
            for plan in (
                value for value in problem.candidates if value.path == path
            ):
                self.assertEqual(len(plan.swap_order), len(path) - 2)
                self.assertEqual(set(plan.swap_order), set(path[1:-1]))

        snapshot = OrderAwareBatchEnv(problem).snapshot()
        lookup = {plan.plan_id: plan for plan in snapshot.candidates}
        for planner in (QDDCAFixedOrderPlanner(), QCASTFixedOrderPlanner()):
            selected = planner.select(snapshot)
            self.assertEqual(len(selected), 1)
            self.assertTrue(lookup[selected[0]].is_fixed_order)

    def test_explicit_exhaustive_catalogue_has_all_seven_group_schedules(self):
        path = (0, 1, 2, 3, 4)
        config = WaxmanOrderConfig(
            node_count=5,
            average_degree=2,
            request_count=1,
            arrival_rate=1.0,
            episode_steps=1,
            request_ttl_slots=1,
            min_hops=4,
            max_hops=4,
            candidate_paths=1,
            order_variants_per_path=None,
            node_memory_cap=4,
            swap_probability=1.0,
        )
        episode = WaxmanOrderEpisode(
            seed=32,
            config=config,
            nodes=path,
            links=tuple(
                OrderLinkSpec(left, right, generation_probability=1.0)
                for left, right in zip(path, path[1:])
            ),
            node_capacities=tuple((node, 4) for node in path),
            positions=tuple(
                (node, (float(node), 0.0)) for node in path
            ),
            requests=(WaxmanOrderRequest("r", 0, 4, 0, 1, 4),),
            request_paths=(("r", (path,)),),
            topology_beta=1.0,
            link_alpha=1.0,
            horizon_slots=1,
        )

        problem = episode.problem_for_slot(("r",), 0, physics_seed=100)

        self.assertIsNone(config.order_variants_per_path)
        self.assertIsNone(config.candidate_request_cap)
        self.assertEqual(config.max_swap_orders_per_path, 7)
        self.assertEqual(len(problem.candidates), 7)
        self.assertEqual(
            {plan.swap_order for plan in problem.candidates},
            set(itertools.permutations(path[1:-1])),
        )
        self.assertEqual(
            {plan.schedule.groups for plan in problem.candidates},
            {
                ((1, 3), (2,)),
                *(tuple((node,) for node in order)
                  for order in itertools.permutations(path[1:-1])),
            },
        )
        self.assertEqual(problem.candidates[0].schedule.groups, ((1, 3), (2,)))
        self.assertTrue(problem.candidates[0].is_fixed_order)
        self.assertEqual(
            OrderEpisodeEnv._default_gym_config(config).max_candidates,
            7,
        )
        undersized = OrderEpisodeEnv(
            OrderGymConfig(
                max_nodes=5,
                max_edges=4,
                max_requests=1,
                max_candidates=6,
                max_hops=4,
            ),
            config,
        )
        with self.assertRaisesRegex(
            ValueError, "candidate catalogue exceeds max_candidates"
        ):
            undersized.reset(options={"episode": episode})

    def test_slot_problem_exposes_full_topology_and_request_timing(self):
        episode = make_waxman_order_episode(self.config, seed=3)
        pending = tuple(request.request_id for request in episode.requests)
        slot = next(
            value for value in range(episode.horizon_slots)
            if episode.active_request_ids(pending, value)
        )
        request_ids = episode.active_request_ids(pending, slot)
        problem = episode.problem_for_slot(
            request_ids, slot, physics_seed=999,
        )
        env = OrderGymEnv(OrderGymConfig(
            max_nodes=32,
            max_edges=256,
            max_requests=3,
            max_candidates=12,
            max_hops=5,
        ))
        observation, _ = env.reset(options={"problem": problem})

        self.assertEqual(set(problem.physical_edges), {
            link.elementary_edge for link in episode.links
        })
        self.assertEqual(observation["edge_features"].shape, (256, 6))
        self.assertEqual(observation["request_features"].shape, (3, 10))
        visible_probabilities = observation["edge_features"][
            observation["edge_mask"], 5
        ]
        self.assertGreater(np.ptp(visible_probabilities), 0.0)
        for plan in problem.candidates:
            self.assertEqual(plan.decision_slot, slot)
            self.assertLessEqual(plan.arrival_slot, slot)
            self.assertGreater(plan.deadline_slot, slot)

    def test_small_rolling_suite_settles_all_requests(self):
        config = WaxmanOrderConfig(
            node_count=20,
            average_degree=4,
            request_count=6,
            arrival_rate=2.0,
            request_ttl_slots=5,
            min_hops=2,
            max_hops=4,
            candidate_paths=1,
            order_variants_per_path=2,
            candidate_request_cap=2,
            node_memory_cap=3,
            slot_duration_ps=4_000,
        )
        result = run_suite(
            episodes=1,
            base_seed=23,
            config=config,
            oracle_rollouts=1,
            planner_names=("qddca_fixed", "qcast_fixed"),
        )

        self.assertEqual(result["config"]["request_count"], 6)
        self.assertEqual(result["episodes"], 1)
        self.assertEqual(result["episode_seeds"], (23,))
        self.assertEqual(len(result["planner_seeds"]), 1)
        self.assertNotEqual(result["planner_seeds"][0], 23)
        self.assertNotIn("physics_seed_roots", result)
        self.assertIn("physics_seed_isolation", result["model"])
        self.assertIn("milp_nominal_objective", result["model"])
        self.assertEqual(len(result["topologies"][0]["requests"]), 6)
        for rows in result["rows"].values():
            row = rows[0]
            self.assertEqual(
                row["completed_count"] + row["timeout_count"], 6
            )

    def test_rolling_benchmark_expires_inventory_at_fixed_ttl(self):
        config = WaxmanOrderConfig(
            node_count=3,
            average_degree=2,
            request_count=1,
            arrival_rate=1.0,
            request_ttl_slots=1,
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
            epr_ttl_slots=1,
        )
        episode = WaxmanOrderEpisode(
            seed=0,
            config=config,
            nodes=(0, 1, 2),
            links=(
                OrderLinkSpec(0, 1),
                OrderLinkSpec(1, 2),
            ),
            node_capacities=((0, 2), (1, 2), (2, 2)),
            positions=((0, (0.0, 0.0)), (1, (0.5, 0.0)), (2, (1.0, 0.0))),
            requests=(WaxmanOrderRequest("r0", 0, 2, 0, 1, 2),),
            request_paths=(("r0", ((0, 1, 2),)),),
            topology_beta=1.0,
            link_alpha=1.0,
            horizon_slots=2,
        )

        row = run_planner_episode(
            episode,
            QDDCAFixedOrderPlanner(),
            physics_seed_root=90_001,
            planner_seed=90_002,
        )

        self.assertEqual(row["completed_count"], 0)
        self.assertEqual(row["timeout_count"], 1)
        self.assertEqual(row["max_inventory_pairs"], 2)
        self.assertEqual(row["expired_inventory_pairs"], 2)
        self.assertEqual(row["remaining_inventory_pairs"], 0)

    def test_suite_shares_hidden_physics_root_but_not_episode_seed(self):
        calls: list[dict[str, int]] = []

        def fake_run(episode, planner, **kwargs):
            del planner
            calls.append({
                "episode_seed": episode.seed,
                "physics_seed_root": kwargs["physics_seed_root"],
                "planner_seed": kwargs["planner_seed"],
            })
            return {
                "episode_seed": episode.seed,
                "completed_count": 0,
                "completion_rate": 0.0,
            }

        config = WaxmanOrderConfig(
            node_count=20,
            average_degree=4,
            request_count=2,
            arrival_rate=1.0,
            request_ttl_slots=2,
            min_hops=2,
            max_hops=3,
            candidate_paths=1,
            order_variants_per_path=1,
            candidate_request_cap=1,
            node_memory_cap=2,
            episode_steps=2,
        )
        with patch(
            "qnet_core.order_waxman_benchmark.run_planner_episode",
            side_effect=fake_run,
        ):
            result = run_suite(
                episodes=1,
                base_seed=31,
                physics_seed_base=710_000,
                planner_seed_base=720_000,
                config=config,
                oracle_rollouts=1,
                planner_names=("qddca_fixed", "qcast_fixed"),
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["episode_seed"], 31)
        self.assertEqual(
            calls[0]["physics_seed_root"],
            calls[1]["physics_seed_root"],
        )
        self.assertEqual(calls[0]["planner_seed"], calls[1]["planner_seed"])
        self.assertNotEqual(calls[0]["planner_seed"], 31)
        self.assertNotEqual(
            calls[0]["physics_seed_root"], calls[0]["planner_seed"]
        )
        self.assertNotIn("physics_seed_roots", result)

    def test_seed_streams_are_stable_across_episode_chunks(self):
        calls: list[tuple[int, int, int]] = []

        def fake_run(episode, planner, **kwargs):
            del planner
            calls.append((
                episode.seed,
                kwargs["physics_seed_root"],
                kwargs["planner_seed"],
            ))
            return {
                "episode_seed": episode.seed,
                "completed_count": 0,
                "completion_rate": 0.0,
            }

        config = WaxmanOrderConfig(
            node_count=20,
            average_degree=4,
            request_count=2,
            arrival_rate=1.0,
            request_ttl_slots=2,
            min_hops=2,
            max_hops=3,
            candidate_paths=1,
            order_variants_per_path=1,
            candidate_request_cap=1,
            node_memory_cap=2,
            episode_steps=2,
        )
        with patch(
            "qnet_core.order_waxman_benchmark.run_planner_episode",
            side_effect=fake_run,
        ):
            run_suite(
                episodes=2,
                base_seed=10,
                physics_seed_base=750_000,
                planner_seed_base=760_000,
                config=config,
                planner_names=("qddca_fixed",),
            )
            full_episode_11 = calls[-1]
            calls.clear()
            run_suite(
                episodes=1,
                base_seed=11,
                physics_seed_base=750_000,
                planner_seed_base=760_000,
                config=config,
                planner_names=("qddca_fixed",),
            )

        self.assertEqual(full_episode_11, calls[0])

    def test_nominal_milp_reports_model_and_executor_outcomes_separately(
        self,
    ) -> None:
        config = WaxmanOrderConfig(
            node_count=10,
            average_degree=4,
            request_count=4,
            arrival_rate=4 / 3,
            episode_steps=3,
            request_ttl_slots=3,
            min_hops=2,
            max_hops=3,
            candidate_paths=2,
            order_variants_per_path=2,
            candidate_request_cap=2,
            node_memory_cap=2,
            slot_duration_ps=4_000,
            generation_interval_ps=1_000,
            swap_service_ps=1_000,
            memory_reset_ps=100,
            swap_probability=0.9,
            epr_ttl_slots=2,
        )
        result = run_suite(
            episodes=1,
            base_seed=0,
            physics_seed_base=730_000,
            planner_seed_base=740_000,
            config=config,
            oracle_rollouts=1,
            planner_names=(
                "milp_nominal_path",
                "milp_nominal_path_order",
            ),
        )

        for name in (
            "milp_nominal_path",
            "milp_nominal_path_order",
        ):
            row = result["rows"][name][0]
            self.assertEqual(
                row["milp_model_objective_slots"], row["planner_calls"]
            )
            self.assertEqual(
                row["milp_model_optimal_slots"], row["planner_calls"]
            )
            self.assertEqual(
                row["milp_executor_completed_count"],
                row["executor_completed_count"],
            )
            self.assertIn("milp_model_objective_sum", row)
            self.assertIn("milp_model_minus_executor_completed", row)


if __name__ == "__main__":
    unittest.main()
