import unittest

from qnet_core.construction_api import (
    ConstructionDAG,
    ConstructionOperation,
    LogicalSegment,
    OperationKind,
    ResourceDemand,
)
from dataclasses import replace
from qnet_core.construction_decoder import (
    CapacityFeasibilityOracle,
    canonical_decode,
    canonical_decode_ready_set,
    feasible_operation_indices,
)
from qnet_core.construction_executor import ConstructionDAGExecutor
from qnet_core.construction_metrics import (
    RequestSettlement,
    censored_flow_time,
    event_accounted_flow_time,
)
from qnet_core.construction_plans import balanced_path_dag, left_deep_path_dag


class ConstructionContractTests(unittest.TestCase):
    def test_demand_is_canonical_and_dag_rejects_cycles(self):
        demand = ResourceDemand((("bsm:1", 1), ("memory:0", 2)))
        self.assertEqual(demand.entries, (("bsm:1", 1), ("memory:0", 2)))
        operation = ConstructionOperation(
            "a", "r", OperationKind.GEN,
            output_segment_id="s", output_endpoints=(0, 1),
        )
        dag = ConstructionDAG("r", (operation,))
        self.assertEqual(dag.ready_ids(set()), ("a",))
        with self.assertRaises(ValueError):
            ConstructionDAG("r", (
                ConstructionOperation("x", "r", OperationKind.RELEASE, predecessors=("y",)),
                ConstructionOperation("y", "r", OperationKind.RELEASE, predecessors=("x",)),
            ))

    def test_canonical_decoder_masks_capacity_incrementally(self):
        operations = tuple(
            ConstructionOperation(
                op_id=f"o{i}", request_id="r", kind=OperationKind.RELEASE,
                resource_demand=ResourceDemand.from_mapping({"bsm": 1}), ordinal=i,
            ) for i in range(3)
        )
        dag = ConstructionDAG("r", operations)
        oracle = CapacityFeasibilityOracle({"bsm": 2})
        self.assertEqual(
            canonical_decode(operations, dag, set(), oracle, True, (0, 1)),
            operations[:2],
        )
        with self.assertRaises(ValueError):
            canonical_decode(operations, dag, set(), oracle, True, (0, 1, 2))
        with self.assertRaises(ValueError):
            canonical_decode(operations, dag, set(), oracle, False, ())

    def test_incremental_mask_does_not_hide_individually_feasible_choice(self):
        same_request = tuple(
            ConstructionOperation(
                op_id=f"r:release:{i}",
                request_id="r",
                kind=OperationKind.RELEASE,
                resource_demand=ResourceDemand.from_mapping({"bsm": 1}),
                ordinal=i,
            )
            for i in range(2)
        )
        oracle = CapacityFeasibilityOracle({"bsm": 1})
        dag = ConstructionDAG("r", same_request)
        self.assertEqual(
            feasible_operation_indices(
                same_request,
                dag,
                set(),
                oracle,
            ),
            (True, True),
        )
        cross_request = (
            same_request[0],
            ConstructionOperation(
                op_id="other:release",
                request_id="other",
                kind=OperationKind.RELEASE,
                resource_demand=ResourceDemand.from_mapping({"bsm": 1}),
            ),
        )
        self.assertEqual(
            canonical_decode_ready_set(cross_request, oracle, True, (0,)),
            (cross_request[1],),
        )

    def test_snapshot_is_read_only(self):
        dag = ConstructionDAG("r", (
            ConstructionOperation("g", "r", OperationKind.GEN,
                                   output_segment_id="s", output_endpoints=(0, 1)),
        ))
        executor = ConstructionDAGExecutor((dag,), {"link:0-1": 1}, seed=3, horizon_ps=10)
        before = executor.snapshot()
        after = executor.snapshot()
        self.assertEqual(before, after)
        self.assertEqual(executor.time, 0)
        self.assertEqual(executor.event_log, [])

    def test_left_deep_and_balanced_same_path_have_distinct_event_traces(self):
        route = (0, 1, 2, 3, 4)
        capacities = {
            "link:0-1": 1, "link:1-2": 1, "link:2-3": 1, "link:3-4": 1,
            "genlane:0-1": 1, "genlane:1-2": 1,
            "genlane:2-3": 1, "genlane:3-4": 1,
            "bsm:1": 1, "bsm:2": 1, "bsm:3": 1,
            "memory:0": 2, "memory:1": 2, "memory:2": 2,
            "memory:3": 2, "memory:4": 2,
        }

        def run(dag):
            executor = ConstructionDAGExecutor((dag,), capacities, seed=7, horizon_ps=30)
            while executor.ready_operations():
                executor.launch(executor.ready_operations())
                executor.advance_to_next_event()
            return executor

        left = run(left_deep_path_dag("r", route))
        balanced = run(balanced_path_dag("r", route))
        left_times = [event.physical_time_ps for event in left.event_log]
        balanced_times = [event.physical_time_ps for event in balanced.event_log]
        self.assertEqual(left.event_log[-1].output_segment_id, "r:seg:left:3")
        self.assertEqual(balanced.event_log[-1].output_segment_id, "r:seg:balanced:2")
        self.assertNotEqual(left_times, balanced_times)
        self.assertGreater(left.event_log[-1].physical_time_ps, balanced.event_log[-1].physical_time_ps)

    def test_failed_branch_keeps_surviving_prefix_for_repair(self):
        generation = ConstructionOperation(
            "g", "r", OperationKind.GEN,
            output_segment_id="s", output_endpoints=(0, 1),
        )
        failed = ConstructionOperation(
            "bad", "r", OperationKind.SWAP,
            predecessors=("g",), input_segment_ids=("s",),
            output_segment_id="out", output_endpoints=(0, 2),
            success_probability=0.0,
        )
        dag = ConstructionDAG("r", (generation, failed))
        executor = ConstructionDAGExecutor((dag,), {"link": 1}, seed=1, horizon_ps=10)
        executor.launch((generation,))
        executor.advance_to_next_event()
        executor.launch((failed,))
        batch = executor.advance_to_next_event()
        self.assertFalse(batch.events[0].success)
        self.assertIn("s", {segment.segment_id for segment in executor.available_segments()})
        repair = ConstructionOperation(
            "repair", "r", OperationKind.SWAP,
            predecessors=("g",), input_segment_ids=("s",),
            output_segment_id="out2", output_endpoints=(0, 2),
            dag_version=dag.version + 1,
        )
        executor.repair("r", (repair,))
        executor.launch((repair,))
        self.assertTrue(executor.advance_to_next_event().events[0].success)

    def test_repair_is_atomic_when_a_new_operation_is_invalid(self):
        base = ConstructionOperation(
            "base",
            "r",
            OperationKind.GEN,
            output_segment_id="s",
            output_endpoints=(0, 1),
        )
        dag = ConstructionDAG("r", (base,))
        before = dag.state()
        valid = ConstructionOperation(
            "repair-valid",
            "r",
            OperationKind.RELEASE,
            predecessors=("base",),
            dag_version=1,
        )
        invalid = ConstructionOperation(
            "repair-invalid",
            "r",
            OperationKind.RELEASE,
            predecessors=("missing",),
            dag_version=1,
        )

        with self.assertRaisesRegex(ValueError, "unknown predecessor"):
            dag.repair((valid, invalid))

        self.assertEqual(dag.state(), before)
        self.assertEqual(tuple(operation.op_id for operation in dag.operations), ("base",))

    def test_reroute_supersedes_only_uncommitted_old_operations(self):
        failed = ConstructionOperation(
            "failed", "r", OperationKind.RELEASE, ordinal=0
        )
        stale = ConstructionOperation(
            "stale", "r", OperationKind.RELEASE, ordinal=1
        )
        dag = ConstructionDAG("r", (failed, stale))
        executor = ConstructionDAGExecutor((dag,), {}, horizon_ps=10)
        dag.mark_started("failed", set())
        dag.mark_dead("failed")
        replacement = ConstructionOperation(
            "reroute", "r", OperationKind.RELEASE,
            ordinal=2, dag_version=1,
        )

        executor.repair(
            "r", (replacement,), supersede_uncommitted=True
        )

        self.assertIn("failed", dag.dead)
        self.assertIn("stale", dag.dead)
        self.assertNotIn("reroute", dag.dead)
        self.assertEqual(
            tuple(operation.op_id for operation in executor.ready_operations()),
            ("reroute",),
        )

    def test_censored_flow_time_matches_event_accounting(self):
        settlements = (
            RequestSettlement("success", 0, 3, True),
            RequestSettlement("failed", 1, 2, False),
        )
        self.assertEqual(censored_flow_time(settlements, 5), 7)
        self.assertEqual(
            event_accounted_flow_time(((0, 3, 1), (1, 2, 1)), (settlements[1],), 5),
            7,
        )

    def test_horizon_timeout_settles_pending_operations(self):
        operation = ConstructionOperation(
            "late", "r", OperationKind.RELEASE,
            duration_ps=5,
        )
        executor = ConstructionDAGExecutor((ConstructionDAG("r", (operation,)),), {}, horizon_ps=2)
        executor.launch((operation,))
        batch = executor.advance_to_next_event()
        self.assertTrue(batch.terminal)
        self.assertEqual(batch.events[0].failure_cause, "horizon_timeout")
        self.assertEqual(batch.physical_time_ps, 2)

    def test_launch_rejects_forged_operation_payload(self):
        operation = ConstructionOperation(
            "release", "r", OperationKind.RELEASE, duration_ps=1
        )
        dag = ConstructionDAG("r", (operation,))
        executor = ConstructionDAGExecutor((dag,), {}, horizon_ps=10)
        forged = replace(operation, duration_ps=2)
        with self.assertRaisesRegex(ValueError, "canonical operation"):
            executor.launch((forged,))

    def test_executor_rejects_cross_dag_output_segment_collision(self):
        first = ConstructionOperation(
            "g0", "r0", OperationKind.GEN,
            output_segment_id="shared", output_endpoints=(0, 1),
        )
        second = ConstructionOperation(
            "g1", "r1", OperationKind.GEN,
            output_segment_id="shared", output_endpoints=(1, 2),
        )
        with self.assertRaisesRegex(ValueError, "shared by multiple operations"):
            ConstructionDAGExecutor(
                (ConstructionDAG("r0", (first,)), ConstructionDAG("r1", (second,))),
                {"link:0-1": 2, "link:1-2": 2},
            )

    def test_executor_rejects_cross_dag_operation_id_collision(self):
        first = ConstructionOperation("shared", "r0", OperationKind.RELEASE)
        second = ConstructionOperation("shared", "r1", OperationKind.RELEASE)
        with self.assertRaisesRegex(ValueError, "operation id is shared"):
            ConstructionDAGExecutor(
                (ConstructionDAG("r0", (first,)), ConstructionDAG("r1", (second,))),
                {},
            )

    def test_executor_rechecks_cross_dag_ids_after_direct_dag_mutation(self):
        first = ConstructionOperation("shared", "r0", OperationKind.RELEASE)
        second = ConstructionOperation("base", "r1", OperationKind.RELEASE)
        executor = ConstructionDAGExecutor(
            (ConstructionDAG("r0", (first,)), ConstructionDAG("r1", (second,))),
            {},
        )
        executor.dags["r1"].add_operation(
            ConstructionOperation("shared", "r1", OperationKind.RELEASE)
        )
        with self.assertRaisesRegex(ValueError, "operation id is shared"):
            executor.ready_operations()


if __name__ == "__main__":
    unittest.main()
