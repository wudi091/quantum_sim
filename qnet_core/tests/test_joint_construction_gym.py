import unittest

from algorithms.caappo import ShortestPathLeftDeepPolicy
from qnet_core.construction_api import (
    ConstructionDAG,
    ConstructionOperation,
    OperationKind,
    ResourceDemand,
)
from qnet_core.construction_catalog import RouteConstructionCandidate
from qnet_core.construction_catalog import build_route_construction_catalogue
from qnet_core.joint_construction_gym import JointConstructionBatchEnv, JointPhase
from qnet_core.planning_spec import RequestSpec
from qnet_core.spec import EpisodeSpec, PhysicalConfig


class JointConstructionBatchEnvTests(unittest.TestCase):
    def test_route_and_construction_are_an_admission_transition(self):
        spec = EpisodeSpec(
            seed=501,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r0", 0, 2, ttl=20),),
            horizon=20,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                node_memory_capacity=4,
                quantum_distance_m=1.0,
            ),
        )
        catalogue = build_route_construction_catalogue(
            spec.planning, candidate_count=2
        )
        env = JointConstructionBatchEnv(spec, catalogue)
        initial = env.reset()
        self.assertEqual(initial.phase, JointPhase.ADMISSION)
        self.assertIsNone(initial.observation)
        chosen = {
            "r0": ShortestPathLeftDeepPolicy().select(catalogue)["r0"]
        }
        admitted = env.admit(chosen)
        self.assertEqual(admitted.phase, JointPhase.EXECUTION)
        self.assertEqual(admitted.info["event_kind"], "admission")
        self.assertIsNotNone(admitted.observation)
        while not admitted.terminated:
            admitted = env.step(
                admitted.ready_operations if admitted.ready_operations else ()
            )
        self.assertEqual(env.phase, JointPhase.TERMINAL)
        self.assertEqual(env.metrics()["completed_requests"], 1.0)

    def test_settlement_releases_pair_for_next_request_on_capacity_one_link(self):
        spec = EpisodeSpec(
            seed=502,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(
                RequestSpec("r0", 0, 1),
                RequestSpec("r1", 0, 1),
            ),
            horizon=20,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                memory_capacity=1,
                node_memory_capacity=1,
                quantum_distance_m=1.0,
            ),
        )
        catalogue = build_route_construction_catalogue(
            spec.planning, candidate_count=1, construction_kinds=("left_deep",)
        )
        by_request = {candidate.request_id: candidate for candidate in catalogue}
        env = JointConstructionBatchEnv(spec, catalogue)
        env.reset()
        admitted = env.admit(by_request)
        first_operation = next(
            operation for operation in admitted.ready_operations
            if operation.request_id == "r0"
        )
        first = env.step((first_operation,))
        self.assertFalse(first.terminated)
        self.assertEqual(tuple(env.core.executor.snapshot().segments), ())
        second_operation = next(
            operation for operation in first.ready_operations
            if operation.request_id == "r1"
        )
        second = env.step((second_operation,))
        self.assertTrue(second.terminated)
        self.assertEqual(env.metrics()["completed_requests"], 2.0)

    def test_physical_failure_enters_repair_then_drop(self):
        spec = EpisodeSpec(
            seed=503,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r0", 0, 1, ttl=20),),
            horizon=20,
            physical=PhysicalConfig(
                generation_probability=1.0,
                detector_efficiency=0.0,
                node_memory_capacity=1,
                quantum_distance_m=1.0,
            ),
        )
        catalogue = build_route_construction_catalogue(
            spec.planning, candidate_count=1
        )
        env = JointConstructionBatchEnv(spec, catalogue)
        env.reset()
        admitted = env.admit({"r0": catalogue[0]})
        failed = env.step(admitted.ready_operations)
        self.assertEqual(failed.phase, JointPhase.REPAIR)
        self.assertEqual(env.repairable_requests, ("r0",))
        dropped = env.drop("r0")
        self.assertTrue(dropped.terminated)
        self.assertEqual(env.metrics()["risk_count"], 1.0)
        self.assertEqual(
            env.metrics()["censored_flow_time_ps"],
            spec.horizon * spec.physical.slot_duration_ps,
        )

    def test_failed_request_with_pending_operation_must_drain_before_drop(self):
        spec = EpisodeSpec(
            seed=505,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r0", 0, 1, ttl=150),),
            horizon=150,
            physical=PhysicalConfig(
                generation_probability=1.0,
                detector_efficiency=0.0,
                node_memory_capacity=1,
                quantum_distance_m=1.0,
            ),
        )
        generation = ConstructionOperation(
            "r0:gen", "r0", OperationKind.GEN,
            output_segment_id="r0:seg", output_endpoints=(0, 1),
            resource_demand=ResourceDemand.from_mapping({
                "link:0-1": 1, "genlane:0-1": 1,
                "memory:0": 1, "memory:1": 1,
            }),
            output_resource_hold=ResourceDemand.from_mapping({
                "link:0-1": 1, "memory:0": 1, "memory:1": 1,
            }),
        )
        release = ConstructionOperation(
            "r0:release", "r0", OperationKind.RELEASE, duration_ps=80_000_000
        )
        dag = ConstructionDAG("r0", (generation, release))
        candidate = RouteConstructionCandidate(
            "r0:custom", "r0", (0, 1), "custom", dag, "r0:seg"
        )
        env = JointConstructionBatchEnv(spec, (candidate,))
        env.reset()
        admitted = env.admit({"r0": candidate})
        failed = env.step(admitted.ready_operations)
        self.assertEqual(failed.phase, JointPhase.EXECUTION)
        self.assertTrue(env.core.executor.has_in_flight)
        with self.assertRaisesRegex(RuntimeError, "only legal in REPAIR|in-flight"):
            env.drop("r0")
        self.assertFalse(failed.terminated)
        self.assertEqual(env.metrics()["event_count"], 1.0)
        drained = env.step(())
        self.assertEqual(drained.phase, JointPhase.REPAIR)
        dropped = env.drop("r0")
        self.assertTrue(dropped.terminated)
        self.assertFalse(env.core.executor.has_in_flight)

    def test_settled_request_is_not_repairable_after_mixed_batch_outcome(self):
        spec = EpisodeSpec(
            seed=507,
            nodes=(0, 1, 2, 3),
            edges=((0, 1), (2, 3)),
            requests=(
                RequestSpec("r0", 0, 1, required_fidelity=0.7),
                RequestSpec("r1", 2, 3, required_fidelity=0.7),
            ),
            horizon=100,
            physical=PhysicalConfig(
                generation_probability=1.0,
                initial_fidelity=0.99,
                max_width=2,
                node_memory_capacity=4,
                quantum_distance_m=1.0,
            ),
        )
        demand = ResourceDemand.from_mapping({
            "link:0-1": 1,
            "genlane:0-1": 1,
            "memory:0": 1,
            "memory:1": 1,
        })
        bad = ConstructionOperation(
            "r0:bad", "r0", OperationKind.GEN,
            output_segment_id="r0:bad-seg", output_endpoints=(0, 1),
            resource_demand=demand, output_resource_hold=ResourceDemand.from_mapping({
                "link:0-1": 1, "memory:0": 1, "memory:1": 1,
            }), required_fidelity=1.0,
        )
        good = ConstructionOperation(
            "r0:good", "r0", OperationKind.GEN,
            output_segment_id="r0:good-seg", output_endpoints=(0, 1),
            resource_demand=demand, output_resource_hold=ResourceDemand.from_mapping({
                "link:0-1": 1, "memory:0": 1, "memory:1": 1,
            }), required_fidelity=0.5,
        )
        r0 = RouteConstructionCandidate(
            "r0:custom", "r0", (0, 1), "custom",
            ConstructionDAG("r0", (bad, good)), "r0:good-seg"
        )
        r1 = RouteConstructionCandidate(
            "r1:custom", "r1", (2, 3), "custom",
            ConstructionDAG("r1", (ConstructionOperation(
                "r1:release", "r1", OperationKind.RELEASE,
                duration_ps=1,
            ),)), "r1:missing-terminal"
        )
        env = JointConstructionBatchEnv(spec, (r0, r1))
        env.reset()
        admitted = env.admit({"r0": r0, "r1": r1})
        outcome = env.step(tuple(
            operation for operation in admitted.ready_operations
            if operation.request_id == "r0"
        ))
        self.assertEqual(outcome.phase, JointPhase.EXECUTION)
        self.assertEqual(env.repairable_requests, ())
        self.assertEqual(env.metrics()["completed_requests"], 1.0)

    def test_late_terminal_event_clears_prior_repairable_marker(self):
        spec = EpisodeSpec(
            seed=509,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(
                RequestSpec("r0", 0, 1, required_fidelity=0.7),
                RequestSpec("r1", 1, 2, required_fidelity=0.7),
            ),
            horizon=100,
            physical=PhysicalConfig(
                generation_probability=1.0,
                memory_lifetime=1000,
                max_width=2,
                node_memory_capacity=4,
                quantum_distance_m=1.0,
            ),
        )
        demand = ResourceDemand.from_mapping({
            "link:0-1": 1,
            "genlane:0-1": 1,
            "memory:0": 1,
            "memory:1": 1,
        })
        bad = ConstructionOperation(
            "r0:bad", "r0", OperationKind.GEN,
            output_segment_id="r0:bad-late", output_endpoints=(0, 1),
            resource_demand=demand,
            output_resource_hold=ResourceDemand.from_mapping({
                "link:0-1": 1, "memory:0": 1, "memory:1": 1,
            }),
            required_fidelity=1.0,
        )
        good = ConstructionOperation(
            "r0:good", "r0", OperationKind.GEN,
            output_segment_id="r0:good-late", output_endpoints=(0, 1),
            resource_demand=demand,
            output_resource_hold=ResourceDemand.from_mapping({
                "link:0-1": 1, "memory:0": 1, "memory:1": 1,
            }),
            required_fidelity=0.5,
            duration_ps=5_000_000,
        )
        r0 = RouteConstructionCandidate(
            "r0:late", "r0", (0, 1), "custom",
            ConstructionDAG("r0", (bad, good)), "r0:good-late"
        )
        r1 = RouteConstructionCandidate(
            "r1:late", "r1", (1, 2), "custom",
            ConstructionDAG("r1", (ConstructionOperation(
                "r1:release", "r1", OperationKind.RELEASE, duration_ps=1,
            ),)), "r1:missing-terminal"
        )
        env = JointConstructionBatchEnv(spec, (r0, r1))
        env.reset()
        admitted = env.admit({"r0": r0, "r1": r1})
        first = env.step(tuple(
            operation for operation in admitted.ready_operations
            if operation.request_id == "r0"
        ))
        self.assertEqual(first.phase, JointPhase.EXECUTION)
        self.assertEqual(env.repairable_requests, ("r0",))
        second = env.step(())
        self.assertEqual(second.phase, JointPhase.EXECUTION)
        self.assertEqual(env.repairable_requests, ())
        self.assertEqual(env.metrics()["completed_requests"], 1.0)

    def test_terminal_waits_for_settled_request_operations_to_drain(self):
        spec = EpisodeSpec(
            seed=511,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r0", 0, 1, required_fidelity=0.5),),
            horizon=100,
            physical=PhysicalConfig(
                generation_probability=1.0,
                quantum_distance_m=1.0,
            ),
        )
        generation = ConstructionOperation(
            "r0:terminal", "r0", OperationKind.GEN,
            output_segment_id="r0:terminal-seg", output_endpoints=(0, 1),
            resource_demand=ResourceDemand.from_mapping({
                "link:0-1": 1, "genlane:0-1": 1,
                "memory:0": 1, "memory:1": 1,
            }),
            output_resource_hold=ResourceDemand.from_mapping({
                "link:0-1": 1, "memory:0": 1, "memory:1": 1,
            }),
            required_fidelity=0.5,
        )
        release = ConstructionOperation(
            "r0:late-release", "r0", OperationKind.RELEASE,
            duration_ps=5_000_000,
        )
        candidate = RouteConstructionCandidate(
            "r0:drain", "r0", (0, 1), "custom",
            ConstructionDAG("r0", (generation, release)), "r0:terminal-seg"
        )
        env = JointConstructionBatchEnv(spec, (candidate,))
        env.reset()
        admitted = env.admit({"r0": candidate})
        delivered = env.step(admitted.ready_operations)
        self.assertFalse(delivered.terminated)
        self.assertEqual(delivered.phase, JointPhase.EXECUTION)
        self.assertTrue(env.core.executor.has_in_flight)
        drained = env.step(())
        self.assertTrue(drained.terminated)
        self.assertEqual(drained.phase, JointPhase.TERMINAL)
        self.assertFalse(env.core.executor.has_in_flight)
        self.assertTrue(env.core.executor.terminated)

    def test_segment_expiration_enters_repair(self):
        spec = EpisodeSpec(
            seed=513,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r0", 0, 2),),
            horizon=10,
            physical=PhysicalConfig(
                generation_probability=1.0,
                memory_lifetime=1,
                node_memory_capacity=4,
                quantum_distance_m=1.0,
            ),
        )
        generation = ConstructionOperation(
            "r0:g0", "r0", OperationKind.GEN,
            output_segment_id="r0:s0", output_endpoints=(0, 1),
            resource_demand=ResourceDemand.from_mapping({
                "link:0-1": 1, "genlane:0-1": 1,
                "memory:0": 1, "memory:1": 1,
            }),
            output_resource_hold=ResourceDemand.from_mapping({
                "link:0-1": 1, "memory:0": 1, "memory:1": 1,
            }),
        )
        delay = ConstructionOperation(
            "r0:delay", "r0", OperationKind.RELEASE,
            predecessors=("r0:g0",), duration_ps=2_000_000,
        )
        candidate = RouteConstructionCandidate(
            "r0:expiration", "r0", (0, 1, 2), "custom",
            ConstructionDAG("r0", (generation, delay)), "r0:missing-terminal"
        )
        env = JointConstructionBatchEnv(spec, (candidate,))
        env.reset()
        admitted = env.admit({"r0": candidate})
        generated = env.step(admitted.ready_operations)
        expired = env.step(generated.ready_operations)
        self.assertEqual(expired.phase, JointPhase.REPAIR)
        self.assertEqual(env.repairable_requests, ("r0",))
        self.assertTrue(any(
            event.failure_cause == "expiration"
            for event in env.core._event_log
        ))


if __name__ == "__main__":
    unittest.main()
