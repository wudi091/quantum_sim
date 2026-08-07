import unittest
from dataclasses import replace

from qnet_core.construction_api import (
    ConstructionDAG,
    ConstructionOperation,
    LogicalSegment,
    OperationKind,
    ResourceDemand,
)
from qnet_core.construction_catalog import (
    RouteConstructionCandidate,
    build_route_construction_catalogue,
)
from qnet_core.construction_evaluate import run_joint_plan_baseline
from qnet_core.construction_executor import ConstructionDAGExecutor
from qnet_core.joint_construction_gym import JointConstructionBatchEnv, JointPhase
from qnet_core.planning_spec import RequestSpec
from qnet_core.runtime import make_sequence_construction_executor
from qnet_core.spec import EpisodeSpec, PhysicalConfig


def _one_hop_generation(
    request_id: str,
    operation_id: str,
    segment_id: str,
    required_fidelity: float = 0.0,
) -> ConstructionOperation:
    demand = ResourceDemand.from_mapping({
        "link:0-1": 1,
        "genlane:0-1": 1,
        "memory:0": 1,
        "memory:1": 1,
    })
    hold = ResourceDemand.from_mapping({
        "link:0-1": 1,
        "memory:0": 1,
        "memory:1": 1,
    })
    return ConstructionOperation(
        operation_id,
        request_id,
        OperationKind.GEN,
        output_segment_id=segment_id,
        output_endpoints=(0, 1),
        resource_demand=demand,
        output_resource_hold=hold,
        required_fidelity=required_fidelity,
        retry_limit=1,
    )


class ConstructionRegressionTests(unittest.TestCase):
    def test_fixed_evaluator_is_idempotent_for_catalogue_candidates(self):
        spec = EpisodeSpec(
            seed=900,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r", 0, 1),),
            horizon=10,
            physical=PhysicalConfig(
                generation_probability=1.0,
                memory_capacity=1,
                node_memory_capacity=1,
                quantum_distance_m=1.0,
            ),
        )
        candidate = build_route_construction_catalogue(
            spec.planning, candidate_count=1, construction_kinds=("left_deep",)
        )[0]

        first = run_joint_plan_baseline(spec, {"r": candidate})
        second = run_joint_plan_baseline(spec, {"r": candidate})

        self.assertEqual(dict(first.metrics), dict(second.metrics))
        self.assertEqual(
            tuple((event.event_kind, event.success, event.physical_time_ps)
                  for event in first.event_trace),
            tuple((event.event_kind, event.success, event.physical_time_ps)
                  for event in second.event_trace),
        )
        self.assertEqual(first.settlements, second.settlements)

    def test_multi_pair_catalogue_serializes_terminal_deliveries(self):
        spec = EpisodeSpec(
            seed=901,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r", 0, 1, demand_pairs=2),),
            horizon=10,
        )
        candidate = build_route_construction_catalogue(
            spec.planning, candidate_count=1, construction_kinds=("left_deep",)
        )[0]
        self.assertEqual(candidate.demand_pairs, 2)
        self.assertEqual(len(candidate.all_terminal_segment_ids), 2)
        self.assertEqual(
            sum(operation.kind == OperationKind.RELEASE for operation in candidate.dag.operations),
            1,
        )

    def test_repair_cannot_lower_request_fidelity(self):
        spec = EpisodeSpec(
            seed=902,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r", 0, 1, required_fidelity=1.0),),
            horizon=10,
            physical=PhysicalConfig(
                generation_probability=1.0,
                initial_fidelity=0.8,
                node_memory_capacity=2,
                quantum_distance_m=1.0,
            ),
        )
        operation = _one_hop_generation("r", "bad", "bad-segment", 1.0)
        candidate = RouteConstructionCandidate(
            "r:custom",
            "r",
            (0, 1),
            "custom",
            ConstructionDAG("r", (operation,)),
            "bad-segment",
        )
        env = JointConstructionBatchEnv(spec, (candidate,))
        env.reset()
        state = env.admit({"r": candidate})
        state = env.step(state.ready_operations)
        self.assertEqual(state.phase, JointPhase.REPAIR)
        options = env.repair_options("r")
        self.assertEqual(options[0][0].required_fidelity, 1.0)
        with self.assertRaisesRegex(ValueError, "cannot lower request"):
            env.repair("r", (replace(options[0][0], required_fidelity=0.0),))

    def test_deadline_and_expiration_are_timestamped_at_boundaries(self):
        spec = EpisodeSpec(
            seed=903,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r", 0, 1, ttl=1),),
            horizon=4,
            physical=PhysicalConfig(
                generation_probability=1.0,
                node_memory_capacity=2,
                quantum_distance_m=1.0,
            ),
        )
        generation = _one_hop_generation("r", "g", "s")
        release = ConstructionOperation(
            "release",
            "r",
            OperationKind.RELEASE,
            predecessors=("g",),
            input_segment_ids=("s",),
            duration_ps=2_000_000,
        )
        candidate = RouteConstructionCandidate(
            "r:deadline",
            "r",
            (0, 1),
            "custom",
            ConstructionDAG("r", (generation, release)),
            "missing-terminal",
        )
        env = JointConstructionBatchEnv(spec, (candidate,))
        env.reset()
        state = env.admit({"r": candidate})
        state = env.step(state.ready_operations)
        state = env.step(state.ready_operations)
        self.assertEqual(state.observation.physical_time_ps, 1_000_000)
        deadline_events = [
            event for event in env.core._event_log if event.event_kind == "deadline"
        ]
        self.assertEqual(len(deadline_events), 1)
        self.assertEqual(deadline_events[0].physical_time_ps, 1_000_000)

    def test_evaluator_supports_multiple_deliveries(self):
        spec = EpisodeSpec(
            seed=904,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r", 0, 1, demand_pairs=2, required_fidelity=0.5),),
            horizon=20,
            physical=PhysicalConfig(
                generation_probability=1.0,
                initial_fidelity=0.9,
                node_memory_capacity=1,
                memory_capacity=1,
                quantum_distance_m=1.0,
            ),
        )
        candidate = build_route_construction_catalogue(
            spec.planning, candidate_count=1, construction_kinds=("left_deep",)
        )[0]
        result = run_joint_plan_baseline(spec, {"r": candidate})
        self.assertEqual(result.metrics["delivered_pairs"], 2.0)
        self.assertEqual(result.metrics["completed_requests"], 1.0)

    def test_fixed_evaluator_settles_at_deadline_boundary(self):
        spec = EpisodeSpec(
            seed=905,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r", 0, 1, ttl=1),),
            horizon=5,
            physical=PhysicalConfig(
                generation_probability=1.0,
                node_memory_capacity=1,
                memory_capacity=1,
                quantum_distance_m=1.0,
            ),
        )
        operation = replace(
            _one_hop_generation("r", "g", "s"), duration_ps=2_000_000
        )
        candidate = RouteConstructionCandidate(
            "r:deadline-evaluator",
            "r",
            (0, 1),
            "custom",
            ConstructionDAG("r", (operation,)),
            "s",
        )
        result = run_joint_plan_baseline(spec, {"r": candidate})
        self.assertEqual(result.settlements[0].settlement_time, 1_000_000)
        self.assertTrue(any(
            event.event_kind == "deadline" and event.physical_time_ps == 1_000_000
            for event in result.event_trace
        ))

    def test_fixed_evaluator_settles_on_memory_expiration(self):
        spec = EpisodeSpec(
            seed=906,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r", 0, 2),),
            horizon=10,
            physical=PhysicalConfig(
                generation_probability=1.0,
                memory_lifetime=1,
                node_memory_capacity=4,
                quantum_distance_m=1.0,
            ),
        )
        generation = _one_hop_generation("r", "g", "s")
        delay = ConstructionOperation(
            "delay",
            "r",
            OperationKind.RELEASE,
            predecessors=("g",),
            duration_ps=2_000_000,
        )
        candidate = RouteConstructionCandidate(
            "r:expiration-evaluator",
            "r",
            (0, 1, 2),
            "custom",
            ConstructionDAG("r", (generation, delay)),
            "missing-terminal",
        )
        result = run_joint_plan_baseline(spec, {"r": candidate})
        expiration = next(
            event for event in result.event_trace
            if event.failure_cause == "expiration"
        )
        self.assertEqual(
            result.settlements[0].settlement_time, expiration.physical_time_ps
        )

    def test_fixed_evaluator_waits_for_expiration_without_inflight_work(self):
        spec = EpisodeSpec(
            seed=9061,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r", 0, 1),),
            horizon=10,
            physical=PhysicalConfig(
                generation_probability=1.0,
                memory_lifetime=1,
                node_memory_capacity=2,
                memory_capacity=1,
                quantum_distance_m=1.0,
            ),
        )
        generation = _one_hop_generation("r", "g", "s")
        candidate = RouteConstructionCandidate(
            "r:resident-expiration",
            "r",
            (0, 1),
            "custom",
            ConstructionDAG("r", (generation,)),
            "missing-terminal",
        )

        result = run_joint_plan_baseline(spec, {"r": candidate})

        expiration = next(
            event for event in result.event_trace
            if event.event_kind == "expiration"
        )
        self.assertFalse(result.settlements[0].success)
        self.assertEqual(
            result.settlements[0].settlement_time,
            expiration.physical_time_ps,
        )

    def test_fixed_evaluator_releases_late_output_after_deadline_settlement(self):
        spec = EpisodeSpec(
            seed=907,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(
                RequestSpec("r0", 0, 1, ttl=1),
                RequestSpec("r1", 0, 1, arrival=3),
            ),
            horizon=6,
            physical=PhysicalConfig(
                generation_probability=1.0,
                node_memory_capacity=1,
                memory_capacity=1,
                quantum_distance_m=1.0,
            ),
        )
        late = RouteConstructionCandidate(
            "r0:late",
            "r0",
            (0, 1),
            "custom",
            ConstructionDAG("r0", (
                replace(_one_hop_generation("r0", "g0", "s0"), duration_ps=2_000_000),
            )),
            "s0",
        )
        next_request = RouteConstructionCandidate(
            "r1:on-time",
            "r1",
            (0, 1),
            "custom",
            ConstructionDAG("r1", (_one_hop_generation("r1", "g1", "s1"),)),
            "s1",
        )
        result = run_joint_plan_baseline(
            spec, {"r0": late, "r1": next_request}
        )
        self.assertFalse(result.settlements[0].success)
        self.assertTrue(result.settlements[1].success)

    def test_fixed_evaluator_caps_future_deadline_at_horizon(self):
        spec = EpisodeSpec(
            seed=908,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r", 0, 1, ttl=100),),
            horizon=5,
        )
        candidate = RouteConstructionCandidate(
            "r:empty",
            "r",
            (0, 1),
            "custom",
            ConstructionDAG("r", ()),
            "missing-terminal",
        )
        result = run_joint_plan_baseline(spec, {"r": candidate})
        self.assertFalse(result.settlements[0].success)
        self.assertEqual(
            result.settlements[0].settlement_time,
            spec.horizon * spec.physical.slot_duration_ps,
        )

    def test_batch_env_caps_inflight_deadline_boundary_at_horizon(self):
        spec = EpisodeSpec(
            seed=909,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r", 0, 1, ttl=100),),
            horizon=5,
            physical=PhysicalConfig(
                generation_probability=1.0,
                node_memory_capacity=1,
                memory_capacity=1,
                quantum_distance_m=1.0,
            ),
        )
        operation = replace(
            _one_hop_generation("r", "g", "s"), duration_ps=10_000_000
        )
        candidate = RouteConstructionCandidate(
            "r:long",
            "r",
            (0, 1),
            "custom",
            ConstructionDAG("r", (operation,)),
            "s",
        )
        env = JointConstructionBatchEnv(spec, (candidate,))
        env.reset()
        state = env.admit({"r": candidate})
        state = env.step(state.ready_operations)
        self.assertTrue(state.terminated)
        self.assertEqual(env.metrics()["risk_count"], 1.0)

    def test_retry_limit_bounds_repair_options(self):
        operation = replace(
            _one_hop_generation("r", "g", "s"),
            retry_limit=1,
            success_probability=0.0,
        )
        executor = ConstructionDAGExecutor(
            (ConstructionDAG("r", (operation,)),),
            {
                "link:0-1": 1,
                "genlane:0-1": 1,
                "memory:0": 1,
                "memory:1": 1,
            },
            seed=909,
        )
        executor.launch((operation,))
        executor.advance_to_next_event()
        options = executor.repair_options("r")
        self.assertEqual(len(options), 1)
        executor.repair("r", options[0])
        retry = options[0][0]
        executor.launch((retry,))
        executor.advance_to_next_event()
        self.assertEqual(executor.repair_options("r"), ())

    def test_sequence_retry_limit_bounds_repair_options(self):
        spec = EpisodeSpec(
            seed=910,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r", 0, 1),),
            horizon=10,
            physical=PhysicalConfig(
                generation_probability=1.0,
                detector_efficiency=0.0,
                node_memory_capacity=1,
                memory_capacity=1,
                quantum_distance_m=1.0,
            ),
        )
        operation = _one_hop_generation("r", "g", "s")
        executor = make_sequence_construction_executor(
            spec, (ConstructionDAG("r", (operation,)),)
        )
        executor.launch((operation,))
        executor.advance_to_next_event()
        options = executor.repair_options("r")
        self.assertEqual(len(options), 1)
        executor.repair("r", options[0])
        executor.launch(options[0])
        executor.advance_to_next_event()
        self.assertEqual(executor.repair_options("r"), ())

    def test_sequence_failed_swap_rebuilds_consumed_prefix(self):
        spec = EpisodeSpec(
            seed=911,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r", 0, 2),),
            horizon=20,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=0.0,
                detector_efficiency=1.0,
                node_memory_capacity=2,
                memory_capacity=1,
                quantum_distance_m=1.0,
            ),
        )
        candidate = build_route_construction_catalogue(
            spec.planning,
            candidate_count=1,
            construction_kinds=("left_deep",),
        )[0]
        executor = make_sequence_construction_executor(spec, (candidate.dag,))
        generations = executor.ready_operations()
        self.assertEqual(len(generations), 2)
        executor.launch(generations)
        generation_batch = executor.advance_to_next_event()
        self.assertTrue(all(event.success for event in generation_batch.events))

        swap = executor.ready_operations()
        self.assertEqual(len(swap), 1)
        executor.launch(swap)
        failed_batch = executor.advance_to_next_event()
        self.assertEqual(len(failed_batch.events), 1)
        self.assertFalse(failed_batch.events[0].success)
        self.assertEqual(
            set(failed_batch.events[0].consumed_segment_ids),
            {operation.output_segment_id for operation in generations},
        )

        options = executor.repair_options("r")
        self.assertEqual(len(options), 1)
        repair = options[0]
        self.assertEqual(
            [operation.kind for operation in repair],
            [OperationKind.GEN, OperationKind.GEN, OperationKind.SWAP],
        )
        rebuilt_ids = {
            operation.output_segment_id
            for operation in repair
            if operation.kind == OperationKind.GEN
        }
        self.assertTrue(
            set(repair[-1].input_segment_ids).issubset(rebuilt_ids)
        )
        executor.repair("r", repair)
        executor.launch(executor.ready_operations())
        retry_batch = executor.advance_to_next_event()
        self.assertTrue(all(event.success for event in retry_batch.events))
        retry_swap = executor.ready_operations()
        self.assertEqual(len(retry_swap), 1)
        self.assertEqual(retry_swap[0].kind, OperationKind.SWAP)

    def test_post_completion_capacity_is_checked_against_resident_holds(self):
        resident = ResourceDemand.from_mapping({
            "link:0-1": 1,
            "memory:0": 1,
            "memory:1": 1,
        })
        initial = LogicalSegment(
            "resident",
            "r",
            0,
            1,
            0,
            held_resources=resident,
        )
        operation = _one_hop_generation("r", "g", "new")
        operation = replace(
            operation,
            output_resource_hold=ResourceDemand.from_mapping({
                "link:0-1": 1,
                "memory:0": 2,
                "memory:1": 2,
            }),
        )
        executor = ConstructionDAGExecutor(
            (ConstructionDAG("r", (operation,)),),
            {
                "link:0-1": 2,
                "genlane:0-1": 1,
                "memory:0": 2,
                "memory:1": 2,
            },
            initial_segments=(initial,),
        )
        with self.assertRaisesRegex(ValueError, "post-completion capacity"):
            executor.launch((operation,))


if __name__ == "__main__":
    unittest.main()
