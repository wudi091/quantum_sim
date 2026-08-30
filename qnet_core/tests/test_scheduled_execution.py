import unittest
from dataclasses import replace
from types import SimpleNamespace

from qnet_core.construction_api import (
    ConstructionDAG,
    ConstructionOperation,
    ConstructionSnapshot,
    InFlightOperation,
    LogicalSegment,
    OperationKind,
    ResourceDemand,
)
from qnet_core.construction_catalog import build_route_construction_catalogue
from qnet_core.planning_spec import RequestSpec
from qnet_core.scheduled_execution import (
    ConstructionBatchSchedule,
    PersistentConstructionScheduler,
    ScheduledEventDisposition,
    ScheduledEventResponse,
    ScheduledRequestPlan,
    _in_flight_dependency_blocked_operation_ids,
    run_scheduled_construction_plan,
)
from qnet_core.spec import EpisodeSpec, PhysicalConfig


class ScheduledExecutionTests(unittest.TestCase):
    def test_event_policy_can_complete_from_an_existing_terminal_segment(self):
        spec = EpisodeSpec(
            seed=1199,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r0", 0, 1, ttl=4),),
            horizon=4,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                detector_efficiency=1.0,
                bsm_success_probability=1.0,
                quantum_distance_m=1.0,
                slot_duration_ps=1_000_000,
                node_memory_capacity=4,
            ),
        )
        candidate = build_route_construction_catalogue(
            spec.planning,
            candidate_count=1,
            construction_kinds=("balanced",),
        )[0]
        first = candidate.dag.operations[0]
        later = replace(
            first,
            op_id="r0:later",
            predecessors=(first.op_id,),
            output_segment_id="r0:later:segment",
            ordinal=first.ordinal + 1,
        )
        plan = ScheduledRequestPlan(
            request_id="r0",
            candidate_id="policy-completion",
            route_nodes=(0, 1),
            construction_kind="balanced",
            dag=ConstructionDAG("r0", (first, later)),
            terminal_segment_ids=(later.output_segment_id or "",),
            start_slot=0,
            completion_slot=3,
            operation_slots=((first.op_id, 0), (later.op_id, 2)),
        )

        class CompleteFirstGeneration:
            @staticmethod
            def on_event_batch(events, snapshot, active_request_ids):
                del snapshot, active_request_ids
                event = next(item for item in events if item.operation_id == first.op_id)
                return (ScheduledEventResponse(
                    request_id=event.request_id,
                    disposition=ScheduledEventDisposition.COMPLETE,
                    completion_segment_id=event.output_segment_id,
                ),)

        scheduler = PersistentConstructionScheduler(
            spec,
            event_policy=CompleteFirstGeneration(),
        )
        scheduler.submit((plan,))

        update = scheduler.advance_to_slot(1)

        self.assertEqual(len(update.outcomes), 1)
        self.assertTrue(update.outcomes[0].success)
        self.assertEqual(scheduler.completed_request_ids, ("r0",))

    def test_start_slot_and_rejected_requests_survive_the_physical_boundary(self):
        spec = EpisodeSpec(
            seed=1200,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(
                RequestSpec("r0", 0, 1, ttl=4),
                RequestSpec("r1", 0, 1, ttl=4),
            ),
            horizon=4,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                detector_efficiency=1.0,
                bsm_success_probability=1.0,
                quantum_distance_m=1.0,
                node_memory_capacity=2,
            ),
        )
        candidate = next(
            item
            for item in build_route_construction_catalogue(
                spec.planning,
                candidate_count=1,
                construction_kinds=("balanced",),
            )
            if item.request_id == "r0"
        )
        operation = candidate.dag.operations[0]
        schedule = ConstructionBatchSchedule(
            horizon_slots=spec.horizon,
            requests=(ScheduledRequestPlan(
                request_id="r0",
                candidate_id=candidate.candidate_id,
                route_nodes=candidate.route_nodes,
                construction_kind=candidate.construction_kind,
                dag=candidate.dag,
                terminal_segment_ids=candidate.all_terminal_segment_ids,
                start_slot=2,
                completion_slot=3,
                operation_slots=((operation.op_id, 2),),
            ),),
            rejected_request_ids=("r1",),
        )

        result = run_scheduled_construction_plan(spec, schedule)

        self.assertEqual(result.metrics["planned_selected_requests"], 1.0)
        self.assertEqual(result.metrics["completed_requests"], 1.0)
        self.assertEqual(result.metrics["schedule_adherence"], 1.0)
        self.assertEqual(result.violations, ())
        self.assertEqual(len(result.launches), 1)
        self.assertGreaterEqual(
            result.launches[0].actual_time_ps,
            2 * spec.physical.slot_duration_ps,
        )
        self.assertLess(
            result.launches[0].actual_time_ps,
            3 * spec.physical.slot_duration_ps,
        )
        settlements = {item.request_id: item for item in result.settlements}
        self.assertTrue(settlements["r0"].success)
        self.assertFalse(settlements["r1"].success)

    def test_persistent_scheduler_accepts_a_plan_after_idle_wait(self):
        spec = EpisodeSpec(
            seed=1201,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r0", 0, 1, ttl=4),),
            horizon=4,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                detector_efficiency=1.0,
                bsm_success_probability=1.0,
                quantum_distance_m=1.0,
            ),
        )
        candidate = build_route_construction_catalogue(
            spec.planning,
            candidate_count=1,
            construction_kinds=("balanced",),
        )[0]
        operation = candidate.dag.operations[0]
        scheduler = PersistentConstructionScheduler(spec)
        scheduler.advance_to_slot(1)
        plan = ScheduledRequestPlan(
            request_id="r0",
            candidate_id=candidate.candidate_id,
            route_nodes=candidate.route_nodes,
            construction_kind=candidate.construction_kind,
            dag=candidate.dag,
            terminal_segment_ids=candidate.all_terminal_segment_ids,
            start_slot=1,
            completion_slot=2,
            operation_slots=((operation.op_id, 1),),
        )
        scheduler.submit((plan,))
        update = scheduler.advance_to_slot(2)
        self.assertEqual(len(update.outcomes), 1)
        self.assertTrue(update.outcomes[0].success)
        self.assertEqual(scheduler.completed_request_ids, ("r0",))
        self.assertEqual(scheduler.cleanup_request_ids, ())

    def test_persistent_scheduler_rejects_route_endpoint_mismatch(self):
        execution_spec = EpisodeSpec(
            seed=1202,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r0", 0, 2, ttl=4),),
            horizon=4,
            physical=PhysicalConfig(quantum_distance_m=1.0),
        )
        catalogue_spec = EpisodeSpec(
            seed=1202,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r0", 0, 1, ttl=4),),
            horizon=4,
            physical=execution_spec.physical,
        )
        candidate = build_route_construction_catalogue(
            catalogue_spec.planning,
            candidate_count=1,
            construction_kinds=("balanced",),
        )[0]
        operation = candidate.dag.operations[0]
        plan = ScheduledRequestPlan(
            request_id="r0",
            candidate_id=candidate.candidate_id,
            route_nodes=candidate.route_nodes,
            construction_kind=candidate.construction_kind,
            dag=candidate.dag,
            terminal_segment_ids=candidate.all_terminal_segment_ids,
            start_slot=0,
            completion_slot=1,
            operation_slots=((operation.op_id, 0),),
        )
        with self.assertRaisesRegex(ValueError, "endpoints mismatch"):
            PersistentConstructionScheduler(execution_spec).submit((plan,))

    def test_scheduled_plan_rejects_terminal_endpoint_mismatch(self):
        spec = EpisodeSpec(
            seed=1205,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r0", 0, 2, ttl=4),),
            horizon=4,
        )
        candidate = build_route_construction_catalogue(
            spec.planning,
            candidate_count=1,
            construction_kinds=("balanced",),
        )[0]
        terminal_id = candidate.all_terminal_segment_ids[0]
        operations = tuple(
            replace(
                operation,
                output_endpoints=(0, 1),
            )
            if operation.output_segment_id == terminal_id
            else operation
            for operation in candidate.dag.operations
        )
        bad_dag = ConstructionDAG("r0", operations)
        slots = tuple(sorted(
            (operation.op_id, 0 if ":gen:" in operation.op_id else 1)
            for operation in bad_dag.operations
        ))
        with self.assertRaisesRegex(ValueError, "terminal segment endpoints"):
            ScheduledRequestPlan(
                request_id="r0",
                candidate_id="bad-terminal",
                route_nodes=candidate.route_nodes,
                construction_kind=candidate.construction_kind,
                dag=bad_dag,
                terminal_segment_ids=(terminal_id,),
                start_slot=0,
                completion_slot=2,
                operation_slots=slots,
            )

    def test_persistent_scheduler_rejects_terminal_demand_mismatch(self):
        execution_spec = EpisodeSpec(
            seed=1203,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r0", 0, 1, ttl=4, demand_pairs=2),),
            horizon=4,
            physical=PhysicalConfig(quantum_distance_m=1.0),
        )
        catalogue_spec = EpisodeSpec(
            seed=1203,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r0", 0, 1, ttl=4, demand_pairs=1),),
            horizon=4,
            physical=execution_spec.physical,
        )
        candidate = build_route_construction_catalogue(
            catalogue_spec.planning,
            candidate_count=1,
            construction_kinds=("balanced",),
        )[0]
        operation = candidate.dag.operations[0]
        plan = ScheduledRequestPlan(
            request_id="r0",
            candidate_id=candidate.candidate_id,
            route_nodes=candidate.route_nodes,
            construction_kind=candidate.construction_kind,
            dag=candidate.dag,
            terminal_segment_ids=candidate.all_terminal_segment_ids,
            start_slot=0,
            completion_slot=1,
            operation_slots=((operation.op_id, 0),),
        )
        with self.assertRaisesRegex(ValueError, "terminal demand mismatch"):
            PersistentConstructionScheduler(execution_spec).submit((plan,))

    def test_physical_reservation_carries_inflight_input_holds_forward(self):
        spec = EpisodeSpec(
            seed=1204,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r0", 0, 2, ttl=4),),
            horizon=4,
            physical=PhysicalConfig(
                quantum_distance_m=1.0,
                slot_duration_ps=1_000,
            ),
        )
        scheduler = PersistentConstructionScheduler(spec)
        held = ResourceDemand.from_mapping({
            "link:0-1": 1,
            "memory:0": 1,
            "memory:1": 1,
        })
        snapshot = ConstructionSnapshot(
            physical_time_ps=1_000,
            horizon_ps=4_000,
            segments=(LogicalSegment(
                "segment",
                "r0",
                0,
                1,
                0,
                held_resources=held,
            ),),
            reservations=(
                ("bsm:1", 1),
                ("link:0-1", 1),
                ("memory:0", 1),
                ("memory:1", 1),
            ),
            in_flight=(InFlightOperation(
                operation_id="swap",
                request_id="r0",
                attempt_id="swap:attempt:1",
                start_time_ps=1_000,
                completion_time_ps=3_000,
                reserved_resources=(("bsm:1", 1),),
                input_segment_ids=("segment",),
            ),),
            resource_capacities=(
                ("bsm:1", 1),
                ("link:0-1", 2),
                ("memory:0", 2),
                ("memory:1", 2),
            ),
        )
        scheduler.executor = SimpleNamespace(
            physical_time_ps=1_000,
            snapshot=lambda: snapshot,
        )
        usage = scheduler.physical_reservations_by_request(4)["r0"]
        self.assertEqual(usage[("link:0-1", 2)], 1)
        self.assertEqual(usage[("memory:0", 2)], 1)
        self.assertEqual(usage[("memory:1", 2)], 1)
        self.assertEqual(usage[("bsm:1", 2)], 1)

    def test_overdue_ready_work_keeps_planned_slot_priority(self):
        spec = EpisodeSpec(
            seed=1206,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(
                RequestSpec("z_old", 0, 1, ttl=4),
                RequestSpec("a_new", 0, 1, ttl=4),
            ),
            horizon=4,
            physical=PhysicalConfig(quantum_distance_m=1.0),
        )
        scheduler = PersistentConstructionScheduler(spec)
        old = ConstructionOperation(
            "z_old:release",
            "z_old",
            OperationKind.RELEASE,
        )
        new = ConstructionOperation(
            "a_new:release",
            "a_new",
            OperationKind.RELEASE,
        )
        operations = {old.op_id: old, new.op_id: new}
        scheduler._due = {new.op_id: new, old.op_id: old}
        scheduler._planned_slot_by_operation = {
            old.op_id: 1,
            new.op_id: 3,
        }
        scheduler.executor = SimpleNamespace(
            ready_operations=lambda allowed: tuple(sorted(
                (operations[operation_id] for operation_id in allowed),
                key=lambda operation: operation.canonical_key,
            )),
        )

        self.assertEqual(scheduler._ready_due_operations(), (old,))

    def test_inflight_ancestor_suppresses_duplicate_launch_overrun(self):
        generation = ConstructionOperation(
            "r0:gen",
            "r0",
            OperationKind.GEN,
            output_segment_id="r0:elementary",
            output_endpoints=(0, 1),
        )
        middle_swap = ConstructionOperation(
            "r0:swap:middle",
            "r0",
            OperationKind.SWAP,
            predecessors=(generation.op_id,),
            input_segment_ids=(generation.output_segment_id,),
            output_segment_id="r0:middle",
            output_endpoints=(0, 2),
        )
        terminal_swap = ConstructionOperation(
            "r0:swap:terminal",
            "r0",
            OperationKind.SWAP,
            predecessors=(middle_swap.op_id,),
            input_segment_ids=(middle_swap.output_segment_id,),
            output_segment_id="r0:terminal",
            output_endpoints=(0, 3),
        )
        operations = {
            operation.op_id: operation
            for operation in (generation, middle_swap, terminal_swap)
        }

        blocked = _in_flight_dependency_blocked_operation_ids(
            {
                middle_swap.op_id: middle_swap,
                terminal_swap.op_id: terminal_swap,
            },
            operations,
            {generation.op_id},
        )

        self.assertEqual(
            blocked,
            frozenset((middle_swap.op_id, terminal_swap.op_id)),
        )

    def test_unrelated_inflight_work_does_not_hide_launch_overrun(self):
        running = ConstructionOperation(
            "r0:gen",
            "r0",
            OperationKind.GEN,
            output_segment_id="r0:segment",
            output_endpoints=(0, 1),
        )
        due = ConstructionOperation(
            "r1:gen",
            "r1",
            OperationKind.GEN,
            output_segment_id="r1:segment",
            output_endpoints=(2, 3),
        )

        blocked = _in_flight_dependency_blocked_operation_ids(
            {due.op_id: due},
            {running.op_id: running, due.op_id: due},
            {running.op_id},
        )

        self.assertEqual(blocked, frozenset())

    def test_inflight_predecessor_only_reports_completion_overrun(self):
        spec = EpisodeSpec(
            seed=1207,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r0", 0, 2, ttl=4),),
            horizon=4,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                detector_efficiency=1.0,
                bsm_success_probability=1.0,
                classical_delay_ps=100_000,
                quantum_distance_m=1.0,
                slot_duration_ps=1_000,
            ),
        )
        candidate = build_route_construction_catalogue(
            spec.planning,
            candidate_count=1,
            construction_kinds=("balanced",),
        )[0]
        operation_slots = tuple(sorted(
            (
                operation.op_id,
                0 if operation.kind == OperationKind.GEN else 1,
            )
            for operation in candidate.dag.operations
        ))
        plan = ScheduledRequestPlan(
            request_id="r0",
            candidate_id=candidate.candidate_id,
            route_nodes=candidate.route_nodes,
            construction_kind=candidate.construction_kind,
            dag=candidate.dag,
            terminal_segment_ids=candidate.all_terminal_segment_ids,
            start_slot=0,
            completion_slot=2,
            operation_slots=operation_slots,
        )
        scheduler = PersistentConstructionScheduler(spec)
        scheduler.submit((plan,))

        update = scheduler.advance_to_slot(2)

        codes = [violation.code for violation in update.violations]
        self.assertIn("slot_completion_overrun", codes)
        self.assertNotIn("slot_launch_overrun", codes)

    def test_unrelated_inflight_operation_keeps_launch_overrun_hard(self):
        spec = EpisodeSpec(
            seed=1208,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(
                RequestSpec("r0", 0, 1, ttl=4),
                RequestSpec("r1", 0, 1, ttl=4),
            ),
            horizon=4,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                detector_efficiency=1.0,
                bsm_success_probability=1.0,
                classical_delay_ps=100_000,
                quantum_distance_m=1.0,
                slot_duration_ps=1_000,
            ),
        )
        candidates = build_route_construction_catalogue(
            spec.planning,
            candidate_count=1,
            construction_kinds=("balanced",),
        )
        plans = []
        operation_ids = {}
        for candidate in candidates:
            operation = candidate.dag.operations[0]
            start_slot = 0 if candidate.request_id == "r0" else 1
            operation_ids[candidate.request_id] = operation.op_id
            plans.append(ScheduledRequestPlan(
                request_id=candidate.request_id,
                candidate_id=candidate.candidate_id,
                route_nodes=candidate.route_nodes,
                construction_kind=candidate.construction_kind,
                dag=candidate.dag,
                terminal_segment_ids=candidate.all_terminal_segment_ids,
                start_slot=start_slot,
                completion_slot=start_slot + 1,
                operation_slots=((operation.op_id, start_slot),),
            ))
        scheduler = PersistentConstructionScheduler(spec)
        scheduler.submit(tuple(sorted(plans, key=lambda plan: plan.request_id)))

        update = scheduler.advance_to_slot(2)

        launch_overruns = [
            violation.operation_id
            for violation in update.violations
            if violation.code == "slot_launch_overrun"
        ]
        self.assertEqual(launch_overruns, [operation_ids["r1"]])


if __name__ == "__main__":
    unittest.main()
