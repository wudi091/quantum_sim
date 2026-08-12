import unittest

import networkx as nx

from algorithms.qcast.online import OnlineQCASTConfig
from algorithms.telgen.compare_online import (
    _endpoint_mode_for_hops,
    _resolve_endpoint_configuration,
    _resolve_temporal_configuration,
    build_parser,
    run_online_comparison,
)
from algorithms.telgen.online import OnlineTELGENConfig
from qnet_core.scenario import ScenarioConfig, make_episode
from qnet_core.spec import PhysicalConfig


class OnlineComparisonConfigurationTests(unittest.TestCase):
    def test_default_benchmark_uses_fixed_four_hop_pilot_configuration(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.requests, 100)
        self.assertEqual(args.nodes, 64)
        self.assertEqual(args.paths, 4)
        self.assertEqual(args.construction_plans, 5)
        self.assertFalse(args.uniform_random_endpoints)
        self.assertEqual(
            _resolve_endpoint_configuration(
                args.min_hops,
                args.max_hops,
                args.uniform_random_endpoints,
            ),
            (4, 4, "distance_stratified"),
        )
        timing = _resolve_temporal_configuration(
            output=args.output,
            ttl=args.ttl,
            horizon=args.horizon,
            decision_interval=args.decision_interval,
            request_count=args.requests,
            requests_per_batch=args.requests_per_batch,
        )
        self.assertEqual(
            timing.output,
            "results/telgen_qcast_waxman_fixed4_periodic",
        )
        self.assertEqual(timing.ttl, 16)
        self.assertEqual(timing.horizon, 52)
        self.assertEqual(timing.decision_interval, 4)

    def test_periodic_timing_requires_final_ttl_drain(self):
        invalid = (
            {"horizon": 51},
            {"decision_interval": 0},
            {"ttl": 0},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                values = dict(
                    output=None,
                    ttl=None,
                    horizon=None,
                    decision_interval=None,
                    request_count=100,
                    requests_per_batch=10,
                )
                values.update(overrides)
                with self.assertRaises(ValueError):
                    _resolve_temporal_configuration(**values)

    def test_parser_accepts_fixed_hop_bounds(self):
        args = build_parser().parse_args([
            "--min-hops", "4",
            "--max-hops", "4",
        ])
        self.assertEqual(args.min_hops, 4)
        self.assertEqual(args.max_hops, 4)
        self.assertEqual(
            _endpoint_mode_for_hops(args.min_hops, args.max_hops),
            "distance_stratified",
        )

    def test_explicit_random_endpoint_mode_disables_default_hop_bounds(self):
        self.assertEqual(
            _resolve_endpoint_configuration(None, None, True),
            (None, None, "uniform_random"),
        )

    def test_random_endpoint_mode_rejects_explicit_hop_bounds(self):
        with self.assertRaises(ValueError):
            _resolve_endpoint_configuration(4, 4, True)

    def test_hop_bounds_must_be_valid_and_paired(self):
        invalid = (
            (4, None),
            (None, 4),
            (0, 4),
            (5, 4),
        )
        for min_hops, max_hops in invalid:
            with self.subTest(min_hops=min_hops, max_hops=max_hops):
                with self.assertRaises(ValueError):
                    _endpoint_mode_for_hops(min_hops, max_hops)

    def test_fixed_four_hop_workload_is_strict(self):
        episode = make_episode(
            ScenarioConfig(
                request_count=100,
                min_hops=4,
                max_hops=4,
                topology_nodes=64,
                waxman_alpha=0.15,
                waxman_beta=0.45,
                topology_attempts=128,
                waxman_add_mst=False,
                endpoint_mode="distance_stratified",
                ttl=16,
                horizon=24,
            ),
            seed=3101,
        )
        graph = nx.Graph(episode.edges)
        self.assertEqual(len(episode.requests), 100)
        self.assertEqual(
            {
                nx.shortest_path_length(
                    graph,
                    request.source,
                    request.destination,
                )
                for request in episode.requests
            },
            {4},
        )

    def test_periodic_batch_arrivals_include_a_partial_final_batch(self):
        episode = make_episode(
            ScenarioConfig(
                request_count=23,
                min_hops=4,
                max_hops=4,
                topology_nodes=64,
                waxman_alpha=0.15,
                waxman_beta=0.45,
                topology_attempts=128,
                waxman_add_mst=False,
                endpoint_mode="distance_stratified",
                ttl=16,
                horizon=28,
                arrival_batch_size=10,
                arrival_interval=4,
            ),
            seed=3101,
        )
        self.assertEqual(episode.horizon, 28)
        self.assertEqual(
            [request.arrival for request in episode.requests],
            [0] * 10 + [4] * 10 + [8] * 3,
        )
        self.assertEqual(
            [request.deadline for request in episode.requests],
            [16] * 10 + [20] * 10 + [24] * 3,
        )

    def test_scenario_rejects_horizon_before_the_final_arrival_ttl(self):
        with self.assertRaisesRegex(ValueError, "arrival's TTL"):
            make_episode(
                ScenarioConfig(
                    request_count=11,
                    min_hops=2,
                    max_hops=2,
                    topology_mode="parallel_corridors",
                    ttl=4,
                    horizon=7,
                    arrival_batch_size=10,
                    arrival_interval=4,
                ),
                seed=3101,
            )

    def test_periodic_batch_executes_recurring_decisions(self):
        horizon = 6
        report = run_online_comparison(
            ScenarioConfig(
                request_count=1,
                min_hops=2,
                max_hops=2,
                ttl=horizon,
                horizon=horizon,
                physical=PhysicalConfig(
                    generation_probability=1.0,
                    swap_probability=1.0,
                    detector_efficiency=1.0,
                    bsm_success_probability=1.0,
                    quantum_distance_m=1.0,
                    node_memory_capacity=4,
                ),
                topology_mode="parallel_corridors",
                parallel_corridors=2,
            ),
            seeds=1,
            seed_start=1305,
            telgen_config=OnlineTELGENConfig(
                decision_interval=2,
                path_candidate_count=1,
                construction_kinds=("balanced",),
                purification_kinds=("none",),
            ),
            qcast_config=OnlineQCASTConfig(
                decision_interval=2,
                path_candidate_count=1,
                construction_kind="left_deep",
                purification_kind="none",
            ),
        )
        trial = report.trials[0]
        self.assertEqual(trial.episode.horizon, horizon)
        self.assertEqual({request.arrival for request in trial.episode.requests}, {0})
        self.assertEqual(
            [item.decision_slot for item in trial.telgen.decisions],
            [0, 2, 4],
        )
        self.assertEqual(
            [item.window_end_slot for item in trial.telgen.decisions],
            [2, 4, 6],
        )
        self.assertEqual(
            [item.decision_slot for item in trial.qcast.decisions],
            [0, 2, 4],
        )
        self.assertEqual(
            [item.window_end_slot for item in trial.qcast.decisions],
            [2, 4, 6],
        )

    def test_comparison_defaults_to_scenario_arrival_interval(self):
        horizon = 6
        report = run_online_comparison(
            ScenarioConfig(
                request_count=1,
                min_hops=2,
                max_hops=2,
                ttl=horizon,
                horizon=horizon,
                physical=PhysicalConfig(
                    generation_probability=1.0,
                    swap_probability=1.0,
                    detector_efficiency=1.0,
                    bsm_success_probability=1.0,
                    quantum_distance_m=1.0,
                    node_memory_capacity=4,
                ),
                topology_mode="parallel_corridors",
                parallel_corridors=2,
                arrival_batch_size=1,
                arrival_interval=2,
            ),
            seeds=1,
            seed_start=1306,
        )
        trial = report.trials[0]
        self.assertEqual(report.telgen_config.decision_interval, 2)
        self.assertEqual(report.qcast_config.decision_interval, 2)
        self.assertEqual(
            [item.decision_slot for item in trial.telgen.decisions],
            [0, 2, 4],
        )
        self.assertEqual(
            [item.decision_slot for item in trial.qcast.decisions],
            [0, 2, 4],
        )

    def test_static_scenario_defaults_to_one_full_horizon_decision(self):
        horizon = 6
        report = run_online_comparison(
            ScenarioConfig(
                request_count=1,
                min_hops=2,
                max_hops=2,
                ttl=horizon,
                horizon=horizon,
                physical=PhysicalConfig(
                    generation_probability=1.0,
                    swap_probability=1.0,
                    detector_efficiency=1.0,
                    bsm_success_probability=1.0,
                    quantum_distance_m=1.0,
                    node_memory_capacity=4,
                ),
                topology_mode="parallel_corridors",
                parallel_corridors=2,
            ),
            seeds=1,
            seed_start=1307,
        )

        trial = report.trials[0]
        self.assertEqual(report.telgen_config.decision_interval, horizon)
        self.assertEqual(report.qcast_config.decision_interval, horizon)
        self.assertEqual(
            [item.decision_slot for item in trial.telgen.decisions],
            [0],
        )
        self.assertEqual(
            [item.decision_slot for item in trial.qcast.decisions],
            [0],
        )

    def test_periodic_comparison_rejects_arrival_decision_mismatch(self):
        scenario = ScenarioConfig(
            request_count=2,
            min_hops=2,
            max_hops=2,
            ttl=6,
            horizon=8,
            topology_mode="parallel_corridors",
            parallel_corridors=2,
            arrival_batch_size=1,
            arrival_interval=2,
        )

        with self.assertRaisesRegex(ValueError, "arrival interval"):
            run_online_comparison(
                scenario,
                seeds=1,
                telgen_config=OnlineTELGENConfig(
                    decision_interval=1,
                    path_candidate_count=1,
                    construction_kinds=("balanced",),
                    purification_kinds=("none",),
                ),
                qcast_config=OnlineQCASTConfig(
                    decision_interval=1,
                    path_candidate_count=1,
                ),
            )


if __name__ == "__main__":
    unittest.main()
