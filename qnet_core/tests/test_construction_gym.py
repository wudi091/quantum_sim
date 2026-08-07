import unittest

from algorithms.caappo import ShortestPathLeftDeepPolicy
from qnet_core.construction_catalog import build_route_construction_catalogue
from qnet_core.construction_gym import ConstructionBatchEnv
from qnet_core.planning_spec import RequestSpec
from qnet_core.spec import EpisodeSpec, PhysicalConfig


class ConstructionBatchEnvTests(unittest.TestCase):
    def test_reset_restores_pristine_ready_operations(self):
        spec = EpisodeSpec(
            seed=400,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r0", 0, 2, ttl=20),),
            horizon=20,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                memory_capacity=2,
                node_memory_capacity=4,
                quantum_distance_m=1.0,
            ),
        )
        candidates = build_route_construction_catalogue(spec.planning, candidate_count=1)
        selected = ShortestPathLeftDeepPolicy().select(candidates)
        env = ConstructionBatchEnv(spec, selected)

        initial = env.reset()
        initial_ready = tuple(operation.op_id for operation in initial.ready_operations)
        state = initial
        while not state.terminated:
            state = env.step(state.ready_operations if state.ready_operations else ())

        reset = env.reset()
        self.assertEqual(
            initial_ready,
            tuple(operation.op_id for operation in reset.ready_operations),
        )

    def test_event_env_launches_operations_and_settles_request(self):
        spec = EpisodeSpec(
            seed=401,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r0", 0, 2, ttl=20),),
            horizon=20,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                memory_capacity=2,
                node_memory_capacity=4,
                quantum_distance_m=1.0,
            ),
        )
        candidates = build_route_construction_catalogue(spec.planning, candidate_count=1)
        selected = ShortestPathLeftDeepPolicy().select(candidates)
        env = ConstructionBatchEnv(spec, selected)
        state = env.reset()
        self.assertEqual(len(state.ready_operations), 2)
        while not state.terminated:
            ready = state.ready_operations
            state = env.step(ready if ready else ())
        self.assertEqual(env.metrics()["completed_requests"], 1.0)
        self.assertEqual(env.metrics()["risk_count"], 0.0)
        self.assertGreater(env.metrics()["peak_memory_usage"], 0.0)
        self.assertEqual(
            env.metrics()["peak_memory_usage"],
            env.metrics()["peak_physical_memory_usage"],
        )
        self.assertGreaterEqual(
            env.metrics()["peak_reserved_memory_units"],
            env.metrics()["peak_physical_memory_usage"],
        )
        self.assertGreater(
            env.metrics()["physical_memory_time_unit_ps"], 0.0
        )
        self.assertEqual(env.metrics()["physical_failure_count"], 0.0)
        self.assertEqual(env.metrics()["fidelity_violation_count"], 0.0)
        self.assertTrue(env.event_trace)
        self.assertEqual(
            state.info["cumulative_flow_cost_ps"],
            env.metrics()["censored_flow_time_ps"],
        )

    def test_failure_reward_matches_horizon_censored_flow_time(self):
        spec = EpisodeSpec(
            seed=403,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(
                RequestSpec(
                    "r0",
                    0,
                    1,
                    ttl=20,
                    required_fidelity=0.99,
                ),
            ),
            horizon=20,
            physical=PhysicalConfig(
                generation_probability=1.0,
                initial_fidelity=0.8,
                memory_capacity=1,
                node_memory_capacity=1,
                quantum_distance_m=1.0,
            ),
        )
        candidates = build_route_construction_catalogue(
            spec.planning, candidate_count=1
        )
        selected = ShortestPathLeftDeepPolicy().select(candidates)
        terminal = next(iter(selected.values())).dag.operations[-1]
        self.assertEqual(terminal.required_fidelity, 0.99)
        env = ConstructionBatchEnv(spec, selected)
        state = env.reset()

        state = env.step(state.ready_operations)

        self.assertTrue(state.terminated)
        self.assertEqual(state.info["risk_count"], 1)
        self.assertEqual(
            state.info["cumulative_flow_cost_ps"],
            env.horizon_ps,
        )
        self.assertEqual(
            state.info["cumulative_flow_cost_ps"],
            env.metrics()["censored_flow_time_ps"],
        )
        self.assertEqual(env.metrics()["fidelity_violation_count"], 1.0)
        self.assertEqual(env.metrics()["fidelity_check_count"], 1.0)
        self.assertEqual(env.metrics()["physical_failure_count"], 0.0)
        self.assertAlmostEqual(state.reward, -2.0)

    def test_metrics_does_not_settle_live_requests(self):
        spec = EpisodeSpec(
            seed=405,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r0", 0, 1, ttl=20),),
            horizon=20,
            physical=PhysicalConfig(
                generation_probability=1.0,
                quantum_distance_m=1.0,
            ),
        )
        candidates = build_route_construction_catalogue(
            spec.planning, candidate_count=1
        )
        env = ConstructionBatchEnv(
            spec, ShortestPathLeftDeepPolicy().select(candidates)
        )
        env.reset()

        metrics = env.metrics()

        self.assertEqual(metrics["risk_count"], 1.0)
        self.assertEqual(env._settled, {})

    def test_arrival_after_horizon_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "arrival cannot exceed"):
            EpisodeSpec(
                seed=407,
                nodes=(0, 1),
                edges=((0, 1),),
                requests=(RequestSpec("r0", 0, 1, arrival=20),),
                horizon=10,
                physical=PhysicalConfig(
                    generation_probability=1.0,
                    quantum_distance_m=1.0,
                ),
            )


if __name__ == "__main__":
    unittest.main()
