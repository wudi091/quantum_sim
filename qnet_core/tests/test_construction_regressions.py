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
    )


class ConstructionRegressionTests(unittest.TestCase):
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
