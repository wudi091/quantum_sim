import unittest
from dataclasses import replace
from unittest.mock import patch

from algorithms.caappo import ShortestPathLeftDeepPolicy
from qnet_core.construction_api import (
    ConstructionDAG,
    ConstructionLaunchRejected,
    ConstructionOperation,
    OperationKind,
    RepairKind,
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

    def test_admission_rejects_intrinsically_memory_infeasible_dag(self):
        spec = EpisodeSpec(
            seed=5021,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r0", 0, 2),),
            horizon=20,
            physical=PhysicalConfig(
                generation_probability=1.0,
                memory_capacity=1,
                node_memory_capacity=1,
                quantum_distance_m=1.0,
            ),
        )
        catalogue = build_route_construction_catalogue(
            spec.planning,
            candidate_count=1,
            construction_kinds=("left_deep",),
        )
        env = JointConstructionBatchEnv(spec, catalogue)
        env.reset()
        self.assertEqual(env.legal_admission_candidates("r0"), ())

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
        self.assertEqual(env.metrics()["physical_failure_count"], 1.0)
        self.assertEqual(
            env.metrics()["generation_protocol_attempt_count"], 1.0
        )
        self.assertEqual(
            env.metrics()["generation_physical_failure_count"], 1.0
        )
        self.assertEqual(env.metrics()["physical_backend_rejection_count"], 0.0)
        self.assertEqual(env.metrics()["executor_rejection_count"], 0.0)
        self.assertEqual(
            env.metrics()["censored_flow_time_ps"],
            spec.horizon * spec.physical.slot_duration_ps,
        )

    def test_repair_options_enforce_retry_limit_and_lineage(self):
        spec = EpisodeSpec(
            seed=5031,
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
            spec.planning, candidate_count=1, construction_kinds=("left_deep",)
        )
        env = JointConstructionBatchEnv(spec, catalogue)
        env.reset()
        admitted = env.admit({"r0": catalogue[0]})
        failed = env.step(admitted.ready_operations)
        self.assertEqual(failed.phase, JointPhase.REPAIR)
        choices = env.repair_choices("r0")
        self.assertEqual(len(choices), 1)
        self.assertEqual(choices[0].kind, RepairKind.RETRY)
        self.assertNotEqual(
            choices[0].terminal_segment_ids,
            env.core.terminal_segment_ids("r0"),
        )
        options = env.repair_options("r0")
        self.assertEqual(len(options), 1)
        repaired = env.repair("r0", options[0])
        self.assertEqual(
            env.core.terminal_segment_ids("r0"),
            choices[0].terminal_segment_ids,
        )
        failed_again = env.step(repaired.ready_operations)
        self.assertEqual(failed_again.phase, JointPhase.REPAIR)
        self.assertEqual(env.repair_options("r0"), ())

    def test_failed_swap_can_reroute_to_alternative_catalogue_path(self):
        spec = EpisodeSpec(
            seed=5032,
            nodes=(0, 1, 2, 3, 4),
            edges=((0, 1), (1, 2), (2, 3), (3, 4), (0, 4)),
            requests=(RequestSpec("r0", 0, 4, ttl=20),),
            horizon=20,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=0.0,
                node_memory_capacity=4,
                quantum_distance_m=1.0,
            ),
        )
        catalogue = build_route_construction_catalogue(
            spec.planning,
            candidate_count=2,
            construction_kinds=("balanced",),
        )
        original = next(
            candidate for candidate in catalogue
            if candidate.route_nodes == (0, 1, 2, 3, 4)
        )
        direct = next(
            candidate for candidate in catalogue
            if candidate.route_nodes == (0, 4)
        )
        env = JointConstructionBatchEnv(spec, catalogue)
        env.reset()
        admitted = env.admit({"r0": original})
        generated = env.step(admitted.ready_operations)
        failed = env.step(generated.ready_operations)

        self.assertEqual(failed.phase, JointPhase.REPAIR)
        failed_state = next(
            state for state in failed.observation.dag_states
            if state.request_id == "r0"
        )
        stale_old_ids = (
            set(failed_state.operation_ids)
            - set(failed_state.completed)
            - set(failed_state.dead)
        )
        self.assertTrue(stale_old_ids)
        choices = env.repair_choices("r0")
        reroute = next(
            choice for choice in choices
            if choice.kind == RepairKind.REROUTE
            and choice.candidate_id == direct.candidate_id
        )
        self.assertTrue(all(
            operation.dag_version == 1 for operation in reroute.operations
        ))
        self.assertTrue(set(reroute.terminal_segment_ids).issubset({
            operation.output_segment_id for operation in reroute.operations
        }))
        self.assertEqual(
            sum(operation.kind == OperationKind.RELEASE for operation in reroute.operations),
            2,
        )

        repaired = env.repair_choice("r0", reroute)
        self.assertEqual(repaired.info["repair_kind"], RepairKind.REROUTE)
        self.assertEqual(env.selected["r0"].candidate_id, direct.candidate_id)
        repaired_state = next(
            state for state in repaired.observation.dag_states
            if state.request_id == "r0"
        )
        self.assertTrue(stale_old_ids.issubset(set(repaired_state.dead)))
        self.assertTrue(all(
            operation.kind == OperationKind.RELEASE
            for operation in repaired.ready_operations
        ))
        released = env.step(repaired.ready_operations)
        self.assertTrue(all(
            operation.kind == OperationKind.GEN
            for operation in released.ready_operations
        ))
        completed = env.step(released.ready_operations)
        self.assertTrue(completed.terminated)
        self.assertEqual(env.metrics()["completed_requests"], 1.0)
        self.assertEqual(env.metrics()["risk_count"], 0.0)

    def test_failed_swap_can_generate_out_of_catalogue_route_at_repair(self):
        spec = EpisodeSpec(
            seed=50321,
            nodes=(0, 1, 2, 3),
            edges=((0, 1), (1, 3), (0, 2), (2, 3), (0, 3)),
            requests=(RequestSpec("r0", 0, 3, ttl=30),),
            horizon=30,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=0.0,
                node_memory_capacity=6,
                quantum_distance_m=1.0,
            ),
        )
        all_candidates = build_route_construction_catalogue(
            spec.planning,
            candidate_count=None,
            construction_kinds=("balanced",),
        )
        original = all_candidates[0]
        env = JointConstructionBatchEnv(
            spec,
            (original,),
            dynamic_repair_paths=2,
            dynamic_repair_construction_kinds=("balanced",),
        )
        env.reset()
        admitted = env.admit({"r0": original})
        generated = env.step(admitted.ready_operations)
        failed = env.step(generated.ready_operations)
        self.assertEqual(failed.phase, JointPhase.REPAIR)

        choices = env.repair_choices("r0")
        dynamic_direct = next(
            choice for choice in choices
            if choice.kind == RepairKind.REROUTE
            and choice.route_nodes == (0, 3)
        )
        self.assertTrue(dynamic_direct.candidate_id.startswith(
            "r0:dynamic:path:"
        ))
        repaired = env.repair_choice("r0", dynamic_direct)
        self.assertEqual(env.selected["r0"].route_nodes, (0, 3))
        self.assertEqual(
            [operation.kind for operation in repaired.ready_operations],
            [OperationKind.GEN],
        )
        terminal = env.step(repaired.ready_operations)
        self.assertTrue(terminal.terminated)
        self.assertEqual(env.metrics()["completed_requests"], 1.0)
        self.assertEqual(env.metrics()["risk_count"], 0.0)

    def test_repeated_reroute_excludes_all_previously_attempted_routes(self):
        spec = EpisodeSpec(
            seed=50322,
            nodes=(0, 1, 2, 3, 4),
            edges=(
                (0, 1), (1, 4),
                (0, 2), (2, 4),
                (0, 3), (3, 4),
            ),
            requests=(RequestSpec("r0", 0, 4, ttl=60),),
            horizon=60,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=0.0,
                node_memory_capacity=6,
                quantum_distance_m=1.0,
            ),
        )
        initial = build_route_construction_catalogue(
            spec.planning,
            candidate_count=1,
            construction_kinds=("balanced",),
        )[0]
        env = JointConstructionBatchEnv(
            spec,
            (initial,),
            max_route_repairs=2,
            dynamic_repair_paths=2,
            dynamic_repair_construction_kinds=("balanced",),
        )
        env.reset()
        failed = env.step(
            env.admit({"r0": initial}).ready_operations
        )
        failed = env.step(failed.ready_operations)
        self.assertEqual(failed.phase, JointPhase.REPAIR)

        first_choices = env.repair_choices("r0")
        first_reroutes = [
            choice for choice in first_choices if choice.kind == RepairKind.REROUTE
        ]
        self.assertEqual(len(first_reroutes), 2)
        first = first_reroutes[0]
        first_route = first.route_nodes
        repaired = env.repair_choice("r0", first)
        failed = env.step(repaired.ready_operations)
        failed = env.step(failed.ready_operations)
        self.assertEqual(failed.phase, JointPhase.REPAIR)

        second_routes = {
            choice.route_nodes
            for choice in env.repair_choices("r0")
            if choice.kind == RepairKind.REROUTE
        }
        self.assertNotIn(initial.route_nodes, second_routes)
        self.assertNotIn(first_route, second_routes)
        self.assertEqual(len(second_routes), 1)

    def test_same_route_alternative_construction_remains_a_repair_choice(self):
        spec = EpisodeSpec(
            seed=50324,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r0", 0, 2, ttl=30),),
            horizon=30,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=0.0,
                node_memory_capacity=4,
                quantum_distance_m=1.0,
            ),
        )
        catalogue = build_route_construction_catalogue(
            spec.planning,
            candidate_count=1,
            construction_kinds=("balanced", "left_deep"),
        )
        selected = next(
            candidate for candidate in catalogue
            if candidate.construction_kind == "balanced"
        )
        env = JointConstructionBatchEnv(
            spec,
            catalogue,
            dynamic_repair_paths=0,
        )
        env.reset()
        failed = env.step(env.admit({"r0": selected}).ready_operations)
        failed = env.step(failed.ready_operations)
        self.assertEqual(failed.phase, JointPhase.REPAIR)
        same_route = next(
            choice for choice in env.repair_choices("r0")
            if choice.kind == RepairKind.REROUTE
            and choice.route_nodes == selected.route_nodes
            and choice.construction_kind == "left_deep"
        )
        self.assertEqual(same_route.route_nodes, selected.route_nodes)

    def test_dynamic_repair_catalogue_is_request_local(self):
        spec = EpisodeSpec(
            seed=50323,
            nodes=(0, 1, 2, 3, 4, 5, 6, 7),
            edges=(
                (0, 1), (1, 2), (0, 3), (3, 2),
                (4, 5), (5, 6), (4, 7), (7, 6),
            ),
            requests=(
                RequestSpec("r0", 0, 2, ttl=30),
                RequestSpec("r1", 4, 6, ttl=30),
            ),
            horizon=30,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=0.0,
                node_memory_capacity=6,
                quantum_distance_m=1.0,
            ),
        )
        initial = build_route_construction_catalogue(
            spec.planning,
            candidate_count=1,
            construction_kinds=("balanced",),
        )
        env = JointConstructionBatchEnv(
            spec,
            initial,
            dynamic_repair_paths=1,
            dynamic_repair_construction_kinds=("balanced",),
        )
        env.reset()
        admitted = env.admit({candidate.request_id: candidate for candidate in initial})
        generated = env.step(admitted.ready_operations)
        failed_r0 = next(
            operation for operation in generated.ready_operations
            if operation.request_id == "r0"
        )
        failed = env.step((failed_r0,))
        self.assertEqual(failed.phase, JointPhase.REPAIR)
        self.assertEqual(env.repairable_requests, ("r0",))
        choices = env.repair_choices("r0")
        self.assertTrue(choices)
        self.assertTrue(all(choice.request_id == "r0" for choice in choices))
        self.assertTrue(all(
            choice.route_nodes is None
            or set(choice.route_nodes).issubset({0, 1, 2, 3})
            for choice in choices
        ))

    def test_repeated_retry_advances_terminal_segment_lineage(self):
        spec = EpisodeSpec(
            seed=5033,
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
        base = build_route_construction_catalogue(
            spec.planning,
            candidate_count=1,
            construction_kinds=("left_deep",),
        )[0]
        operation = replace(base.dag.operations[0], retry_limit=2)
        candidate = replace(
            base,
            dag=ConstructionDAG("r0", (operation,)),
            terminal_segment_id=operation.output_segment_id,
            terminal_segment_ids=(operation.output_segment_id,),
        )
        env = JointConstructionBatchEnv(spec, (candidate,))
        env.reset()
        failed = env.step(env.admit({"r0": candidate}).ready_operations)

        first = env.repair_choices("r0")[0]
        failed_again = env.step(
            env.repair_choice("r0", first).ready_operations
        )
        self.assertEqual(failed_again.phase, JointPhase.REPAIR)
        second = env.repair_choices("r0")[0]
        self.assertEqual(
            env.core.terminal_segment_ids("r0"),
            first.terminal_segment_ids,
        )
        self.assertNotEqual(
            second.terminal_segment_ids,
            first.terminal_segment_ids,
        )

        failed_third = env.step(
            env.repair_choice("r0", second).ready_operations
        )
        self.assertEqual(failed_third.phase, JointPhase.REPAIR)
        self.assertEqual(env.repair_choices("r0"), ())

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
        self.assertEqual(env.metrics()["expiration_count"], 1.0)
        self.assertGreater(
            env.metrics()["physical_memory_time_unit_slots"], 0.0
        )

    def test_executor_launch_rejection_is_an_observable_repair_event(self):
        spec = EpisodeSpec(
            seed=514,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r0", 0, 1, ttl=20),),
            horizon=20,
            physical=PhysicalConfig(
                generation_probability=1.0,
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

        with patch.object(
            env.core.executor,
            "launch",
            side_effect=ConstructionLaunchRejected(
                "synthetic scheduler rejection"
            ),
        ):
            rejected = env.step(admitted.ready_operations)

        self.assertEqual(rejected.phase, JointPhase.REPAIR)
        self.assertEqual(env.metrics()["executor_rejection_count"], 1.0)
        self.assertEqual(
            env.metrics()["executor_launch_batch_attempt_count"], 1.0
        )
        self.assertTrue(any(
            event.failure_cause == "executor_launch_rejection"
            for event in env.core.event_trace
        ))

    def test_executor_internal_launch_error_remains_fail_fast(self):
        spec = EpisodeSpec(
            seed=515,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r0", 0, 1, ttl=20),),
            horizon=20,
            physical=PhysicalConfig(
                generation_probability=1.0,
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

        with patch.object(
            env.core.executor,
            "launch",
            side_effect=RuntimeError("injected backend invariant failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "invariant failure"):
                env.step(admitted.ready_operations)

        self.assertEqual(env.metrics()["executor_rejection_count"], 0.0)
        self.assertFalse(any(
            event.failure_cause == "executor_launch_rejection"
            for event in env.core.event_trace
        ))


if __name__ == "__main__":
    unittest.main()
