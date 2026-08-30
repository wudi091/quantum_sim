import unittest

from algorithms.qcast.online_planner import (
    QCASTAllocation,
    QCASTRecoveryPathPlan,
)
from algorithms.qcast.recovery import (
    QCASTRecoveryPolicy,
    _select_recovery_paths,
)
from qnet_core.construction_api import (
    ConstructionDAG,
    ConstructionOperation,
    ConstructionSnapshot,
    DAGState,
    ExecutionEvent,
    LogicalSegment,
    OperationKind,
)
from qnet_core.construction_catalog import RouteConstructionCandidate
from qnet_core.construction_plans import left_deep_path_dag
from qnet_core.planning_spec import RequestSpec
from qnet_core.scheduled_execution import ScheduledEventDisposition
from qnet_core.spec import EpisodeSpec, PhysicalConfig


class QCASTRecoveryPolicyTests(unittest.TestCase):
    @staticmethod
    def _recovery(
        recovery_id: str,
        start: int,
        end: int,
    ) -> QCASTRecoveryPathPlan:
        route_nodes = tuple(range(start, end + 1))
        return QCASTRecoveryPathPlan(
            recovery_id=recovery_id,
            major_start_index=start,
            major_end_index=end,
            route_nodes=route_nodes,
            generation_operation_ids=tuple(
                f"{recovery_id}:gen:{index}"
                for index in range(len(route_nodes) - 1)
            ),
            segment_ids=tuple(
                f"{recovery_id}:segment:{index}"
                for index in range(len(route_nodes) - 1)
            ),
        )

    def test_recovery_paths_may_overlap_only_on_healthy_major_edges(self):
        left = self._recovery("left", 0, 2)
        right = self._recovery("right", 1, 3)

        selected = _select_recovery_paths((0, 2), (left, right))

        self.assertEqual(selected, (left, right))

    def test_recovery_paths_cannot_both_claim_the_same_broken_edge(self):
        left = self._recovery("left", 0, 2)
        right = self._recovery("right", 1, 3)

        selected = _select_recovery_paths((0, 1, 2), (left, right))

        self.assertIsNone(selected)

    def test_direct_recovery_segment_completes_without_a_swap_suffix(self):
        request_id = "r0"
        episode = EpisodeSpec(
            seed=1400,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2), (0, 2)),
            requests=(RequestSpec(request_id, 0, 2, ttl=4),),
            horizon=4,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                detector_efficiency=1.0,
                bsm_success_probability=1.0,
                quantum_distance_m=1.0,
                slot_duration_ps=1_000_000,
            ),
        )
        base = left_deep_path_dag(request_id, (0, 1, 2))
        recovery_operation = ConstructionOperation(
            op_id=f"{request_id}:recovery:gen",
            request_id=request_id,
            kind=OperationKind.GEN,
            output_segment_id=f"{request_id}:recovery:segment",
            output_endpoints=(0, 2),
            ordinal=len(base.operations),
        )
        dag = ConstructionDAG(
            request_id,
            base.operations + (recovery_operation,),
        )
        major_generations = tuple(
            operation
            for operation in base.operations
            if operation.kind == OperationKind.GEN
        )
        candidate = RouteConstructionCandidate(
            candidate_id=f"{request_id}:candidate",
            request_id=request_id,
            route_nodes=(0, 1, 2),
            construction_kind="left_deep",
            dag=dag,
            terminal_segment_id=base.operations[-1].output_segment_id or "",
            terminal_segment_ids=(base.operations[-1].output_segment_id or "",),
        )
        recovery = QCASTRecoveryPathPlan(
            recovery_id=f"{request_id}:recovery",
            major_start_index=0,
            major_end_index=2,
            route_nodes=(0, 2),
            generation_operation_ids=(recovery_operation.op_id,),
            segment_ids=(recovery_operation.output_segment_id or "",),
        )
        allocation = QCASTAllocation(
            candidate=candidate,
            expected_throughput=1.0,
            width=1,
            major_generation_operation_ids=tuple(
                operation.op_id for operation in major_generations
            ),
            major_segment_ids=tuple(
                operation.output_segment_id or ""
                for operation in major_generations
            ),
            recovery_paths=(recovery,),
        )
        completed = (recovery_operation.op_id,)
        dead = tuple(operation.op_id for operation in major_generations)
        snapshot = ConstructionSnapshot(
            physical_time_ps=1,
            horizon_ps=episode.horizon * episode.physical.slot_duration_ps,
            dag_states=(DAGState(
                request_id=request_id,
                version=0,
                operation_ids=tuple(operation.op_id for operation in dag.operations),
                completed=completed,
                dead=dead,
                committed_prefix=completed,
            ),),
            operations=dag.operations,
            segments=(LogicalSegment(
                segment_id=recovery.segment_ids[0],
                request_id=request_id,
                left=0,
                right=2,
                born_time_ps=1,
            ),),
        )
        event = ExecutionEvent(
            event_id="event:0",
            operation_id=recovery_operation.op_id,
            request_id=request_id,
            attempt_id="attempt:0",
            event_kind=OperationKind.GEN.lower(),
            physical_time_ps=1,
            success=True,
            output_segment_id=recovery.segment_ids[0],
            output_fidelity=1.0,
        )
        policy = QCASTRecoveryPolicy(episode)
        policy.register(allocation)

        responses = tuple(policy.on_event_batch(
            (event,),
            snapshot,
            (request_id,),
        ))

        self.assertEqual(len(responses), 1)
        self.assertEqual(
            responses[0].disposition,
            ScheduledEventDisposition.COMPLETE,
        )
        self.assertEqual(
            responses[0].completion_segment_id,
            recovery.segment_ids[0],
        )
        self.assertTrue(policy.decisions[0].repaired)
        self.assertEqual(policy.decisions[0].repaired_route_nodes, (0, 2))


if __name__ == "__main__":
    unittest.main()
