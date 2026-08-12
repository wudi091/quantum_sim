import unittest
from dataclasses import replace
from unittest.mock import patch

from qnet_core.command_api import ResourceClaim
from qnet_core.construction_api import (
    ConstructionDAG,
    ConstructionOperation,
    LogicalSegment,
    OperationKind,
    ResourceDemand,
)
from qnet_core.construction_decoder import CapacityFeasibilityOracle
from qnet_core.construction_catalog import build_route_construction_catalogue
from qnet_core.construction_metrics import execution_event_metrics
from qnet_core.construction_plans import balanced_path_dag, left_deep_path_dag
from qnet_core.sequence_backend import PreparedGeneration, SequenceBackend
from qnet_core.sequence_construction_executor import SequenceConstructionExecutor
from qnet_core.resource_catalog import build_resource_capacities
from qnet_core.spec import EpisodeSpec, PhysicalConfig, RequestSpec


class SequenceConstructionExecutorTests(unittest.TestCase):
    @staticmethod
    def _disjoint_swap_executor(seed: int = 109):
        spec = EpisodeSpec(
            seed=seed,
            nodes=(0, 1, 2, 3, 4, 5),
            edges=((0, 1), (1, 2), (3, 4), (4, 5)),
            requests=(),
            horizon=100,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                memory_capacity=2,
                node_memory_capacity=4,
                quantum_distance_m=1.0,
            ),
        )
        capacities = {
            "link:0-1": 1,
            "link:1-2": 1,
            "link:3-4": 1,
            "link:4-5": 1,
            "genlane:0-1": 1,
            "genlane:1-2": 1,
            "genlane:3-4": 1,
            "genlane:4-5": 1,
            "bsm:1": 1,
            "bsm:4": 1,
            **{f"swapnode:{node}": 1 for node in spec.nodes},
            **{f"memory:{node}": 4 for node in spec.nodes},
        }
        executor = SequenceConstructionExecutor(
            (
                left_deep_path_dag("r0", (0, 1, 2)),
                left_deep_path_dag("r1", (3, 4, 5)),
            ),
            SequenceBackend(spec),
            capacities,
            horizon_ps=500_000,
        )
        return executor

    def test_native_bbpssw_can_raise_a_one_hop_request_above_threshold(self):
        spec = EpisodeSpec(
            seed=1,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec(
                "r", 0, 1, required_fidelity=0.82
            ),),
            horizon=10,
            physical=PhysicalConfig(
                initial_fidelity=0.8,
                swap_degradation=1.0,
                generation_probability=1.0,
                swap_probability=1.0,
                memory_capacity=2,
                node_memory_capacity=2,
                memory_lifetime=1000,
                quantum_distance_m=1.0,
                slot_duration_ps=1_000_000,
            ),
        )
        candidate = build_route_construction_catalogue(
            spec.planning,
            candidate_count=1,
            construction_kinds=("balanced",),
            purification_kinds=("elementary_once",),
        )[0]
        executor = SequenceConstructionExecutor(
            (candidate.dag,),
            SequenceBackend(spec),
            build_resource_capacities(spec),
        )
        events = []
        for _ in range(3):
            executor.launch(executor.ready_operations())
            events.extend(executor.advance_to_next_event().events)

        terminal = {
            segment.segment_id: segment
            for segment in executor.available_segments()
        }[candidate.terminal_segment_id]
        self.assertEqual(
            [event.event_kind for event in events],
            ["gen", "gen", "purify"],
        )
        self.assertTrue(all(event.success for event in events))
        self.assertGreaterEqual(terminal.fidelity, 0.82)

    def test_backend_preparation_rejection_is_not_stochastic_failure(self):
        backend = self._backend()
        dag = left_deep_path_dag("r", (0, 1))
        executor = SequenceConstructionExecutor(
            (dag,), backend, self._capacities(), horizon_ps=200_000
        )

        def reject_generation(claims, allocation_id):
            return tuple(
                PreparedGeneration(
                    claim,
                    allocation_id,
                    f"rejected-{index}",
                    None,
                    backend.physical_time_ps,
                    "physical_backend_rejection",
                )
                for index, claim in enumerate(claims)
            )

        with patch.object(
            backend, "begin_generation", side_effect=reject_generation
        ):
            executor.launch(executor.ready_operations())
            batch = executor.advance_to_next_event()

        self.assertEqual(batch.events[0].failure_cause,
                         "physical_backend_rejection")
        metrics = execution_event_metrics(batch.events)
        self.assertEqual(metrics["physical_backend_rejection_count"], 1.0)
        self.assertEqual(metrics["physical_failure_count"], 0.0)
        self.assertEqual(metrics["generation_event_count"], 1.0)
        self.assertEqual(metrics["generation_protocol_attempt_count"], 0.0)
        self.assertEqual(metrics["physical_protocol_attempt_count"], 0.0)

    @staticmethod
    def _backend() -> SequenceBackend:
        return SequenceBackend(EpisodeSpec(
            seed=101,
            nodes=(0, 1, 2, 3, 4),
            edges=((0, 1), (1, 2), (2, 3), (3, 4)),
            requests=(),
            horizon=100,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                memory_capacity=2,
                node_memory_capacity=4,
                quantum_distance_m=1.0,
            ),
        ))

    @staticmethod
    def _capacities():
        capacities = {
            "link:0-1": 1,
            "link:1-2": 1,
            "link:2-3": 1,
            "link:3-4": 1,
            "genlane:0-1": 1,
            "genlane:1-2": 1,
            "genlane:2-3": 1,
            "genlane:3-4": 1,
            "bsm:1": 1,
            "bsm:2": 1,
            "bsm:3": 1,
        }
        capacities.update({f"swapnode:{node}": 1 for node in range(5)})
        capacities.update({f"memory:{node}": 4 for node in range(5)})
        return capacities

    def test_logical_only_initial_segments_are_rejected(self):
        dag = left_deep_path_dag("r", (0, 1))
        segment = LogicalSegment("initial", "r", 0, 1, 0)
        with self.assertRaisesRegex(ValueError, "logical-only initial_segments"):
            SequenceConstructionExecutor(
                (dag,),
                self._backend(),
                self._capacities(),
                initial_segments=(segment,),
                horizon_ps=200_000,
            )

    def test_sequence_generation_is_started_as_one_physical_batch(self):
        dag = left_deep_path_dag("r", (0, 1, 2))
        executor = SequenceConstructionExecutor((dag,), self._backend(), self._capacities(), horizon_ps=200_000)
        generations = executor.ready_operations()
        self.assertEqual(len(generations), 2)
        executor.launch(generations)
        before = executor.snapshot()
        self.assertEqual(len(before.in_flight), 2)
        batch = executor.advance_to_next_event()
        self.assertEqual(len(batch.events), 2)
        self.assertTrue(all(event.success for event in batch.events))
        self.assertEqual(len(executor.available_segments()), 2)
        self.assertEqual(batch.physical_time_ps, executor.physical_time_ps)
        self.assertEqual(
            batch.physical_time_ps,
            executor.snapshot().physical_time_ps,
        )
        self.assertEqual(executor.backend.resources()[0].born, batch.physical_time_ps)

    def test_short_logical_event_does_not_complete_swap_before_messages_arrive(self):
        swap_dag = left_deep_path_dag("r0", (0, 1, 2))
        release = ConstructionOperation(
            "r1:release", "r1", OperationKind.RELEASE, duration_ps=1
        )
        executor = SequenceConstructionExecutor(
            (swap_dag, ConstructionDAG("r1", (release,))),
            self._backend(),
            self._capacities(),
            horizon_ps=200_000,
        )
        generations = tuple(
            operation for operation in executor.ready_operations()
            if operation.request_id == "r0"
        )
        executor.launch(generations)
        executor.advance_to_next_event()
        swap = next(
            operation for operation in executor.ready_operations()
            if operation.kind == OperationKind.SWAP
        )
        executor.launch((swap, release))

        first = executor.advance_to_next_event()
        self.assertEqual(
            [(event.operation_id, event.success) for event in first.events],
            [(release.op_id, True)],
        )
        self.assertTrue(executor.has_in_flight)
        self.assertIn(swap.op_id, {
            item.operation_id for item in executor.snapshot().in_flight
        })

        second = executor.advance_to_next_event()
        self.assertEqual(len(second.events), 1)
        self.assertEqual(second.events[0].operation_id, swap.op_id)
        self.assertTrue(second.events[0].success)
        self.assertFalse(executor.has_in_flight)

    def test_same_path_constructions_have_different_sequence_completion_times(self):
        route = (0, 1, 2, 3, 4)

        def run(dag):
            executor = SequenceConstructionExecutor((dag,), self._backend(), self._capacities(), horizon_ps=1_000_000)
            while executor.ready_operations():
                executor.launch(executor.ready_operations())
                executor.advance_to_next_event()
            return executor

        left = run(left_deep_path_dag("r", route, sequential_generation=True))
        balanced = run(balanced_path_dag("r", route))
        self.assertNotEqual(left.event_log[-1].physical_time_ps, balanced.event_log[-1].physical_time_ps)
        self.assertGreater(left.event_log[-1].physical_time_ps, balanced.event_log[-1].physical_time_ps)
        self.assertTrue(all(event.success for event in left.event_log + balanced.event_log))

    def test_same_edge_multi_request_generation_uses_distinct_lanes(self):
        backend = SequenceBackend(EpisodeSpec(
            seed=103,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(),
            horizon=100,
            physical=PhysicalConfig(
                generation_probability=1.0,
                memory_capacity=2,
                node_memory_capacity=2,
                max_width=2,
                quantum_distance_m=1.0,
            ),
        ))
        dags = (
            left_deep_path_dag("r0", (0, 1)),
            left_deep_path_dag("r1", (0, 1)),
        )
        executor = SequenceConstructionExecutor(
            dags,
            backend,
            {
                "link:0-1": 2,
                "genlane:0-1": 2,
                "memory:0": 2,
                "memory:1": 2,
            },
            horizon_ps=200_000,
        )

        executor.launch(executor.ready_operations())
        batch = executor.advance_to_next_event()

        self.assertEqual(len(batch.events), 2)
        self.assertTrue(all(event.success for event in batch.events))
        self.assertEqual(
            {resource.lane for resource in backend.resources()},
            {0, 1},
        )
        self.assertEqual(len(executor.available_segments()), 2)

    def test_same_edge_generation_across_epochs_uses_distinct_lanes(self):
        backend = SequenceBackend(EpisodeSpec(
            seed=104,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(),
            horizon=100,
            physical=PhysicalConfig(
                generation_probability=1.0,
                memory_capacity=2,
                node_memory_capacity=2,
                max_width=2,
                quantum_distance_m=1.0,
            ),
        ))
        dags = (
            left_deep_path_dag("r0", (0, 1)),
            left_deep_path_dag("r1", (0, 1)),
        )
        executor = SequenceConstructionExecutor(
            dags,
            backend,
            {
                "link:0-1": 2,
                "genlane:0-1": 2,
                "memory:0": 2,
                "memory:1": 2,
            },
            horizon_ps=200_000,
        )

        first = next(
            operation
            for operation in executor.ready_operations()
            if operation.request_id == "r0"
        )
        executor.launch((first,))
        second = next(
            operation
            for operation in executor.ready_operations()
            if operation.request_id == "r1"
        )
        executor.launch((second,))
        events = []
        while executor.has_in_flight:
            events.extend(executor.advance_to_next_event().events)

        self.assertEqual(len(events), 2)
        self.assertTrue(all(event.success for event in events))
        self.assertEqual(
            {resource.lane for resource in backend.resources()},
            {0, 1},
        )

    def test_disjoint_swap_nodes_execute_in_parallel_across_multiple_seeds(self):
        for seed in range(109, 117):
            with self.subTest(seed=seed):
                executor = self._disjoint_swap_executor(seed)
                executor.launch(executor.ready_operations())
                generation_batch = executor.advance_to_next_event()
                self.assertEqual(len(generation_batch.events), 4)
                self.assertTrue(all(
                    event.success for event in generation_batch.events
                ))

                swaps = executor.ready_operations()
                self.assertEqual(len(swaps), 2)
                self.assertEqual(
                    {
                        resource
                        for operation in swaps
                        for resource, amount in operation.resource_demand.items()
                        if amount
                    },
                    {
                        "bsm:1",
                        "bsm:4",
                        "swapnode:0",
                        "swapnode:1",
                        "swapnode:2",
                        "swapnode:3",
                        "swapnode:4",
                        "swapnode:5",
                    },
                )
                executor.launch(swaps)
                in_flight = executor.snapshot()
                self.assertEqual(len(in_flight.in_flight), 2)
                self.assertEqual(dict(in_flight.reservations)["bsm:1"], 1)
                self.assertEqual(dict(in_flight.reservations)["bsm:4"], 1)
                self.assertTrue(all(
                    dict(in_flight.reservations)[f"swapnode:{node}"] == 1
                    for node in range(6)
                ))

                swap_batch = executor.advance_to_next_event()
                self.assertEqual(len(swap_batch.events), 2)
                self.assertTrue(all(event.success for event in swap_batch.events))
                self.assertEqual(len(executor.available_segments()), 2)

    def test_same_middle_node_swap_pair_is_rejected(self):
        segments = {
            "a": LogicalSegment("a", "r0", 0, 1, 0),
            "b": LogicalSegment("b", "r0", 1, 2, 0),
            "c": LogicalSegment("c", "r1", 3, 1, 0),
            "d": LogicalSegment("d", "r1", 1, 4, 0),
        }
        swaps = (
            ConstructionOperation(
                "s0", "r0", OperationKind.SWAP,
                input_segment_ids=("a", "b"),
                output_segment_id="o0",
                output_endpoints=(0, 2),
                resource_demand=ResourceDemand.from_mapping({"bsm:1": 1}),
            ),
            ConstructionOperation(
                "s1", "r1", OperationKind.SWAP,
                input_segment_ids=("c", "d"),
                output_segment_id="o1",
                output_endpoints=(3, 4),
                resource_demand=ResourceDemand.from_mapping({"bsm:1": 1}),
            ),
        )
        from qnet_core.sequence_scheduler import SequenceConcurrencyScheduler

        scheduler = SequenceConcurrencyScheduler(
            {"bsm:1": 1},
            supports_concurrent_swaps=True,
        )
        result = scheduler.validate(swaps, segments=tuple(segments.values()))
        self.assertFalse(result.feasible)
        self.assertEqual(
            result.reason,
            "concurrent swaps have shared physical node",
        )

    def test_swaps_sharing_endpoint_node_are_rejected(self):
        segments = (
            LogicalSegment("a", "r0", 0, 1, 0),
            LogicalSegment("b", "r0", 1, 2, 0),
            LogicalSegment("c", "r1", 2, 3, 0),
            LogicalSegment("d", "r1", 3, 4, 0),
        )
        swaps = (
            ConstructionOperation(
                "s0", "r0", OperationKind.SWAP,
                input_segment_ids=("a", "b"),
                output_segment_id="o0",
                output_endpoints=(0, 2),
                resource_demand=ResourceDemand.from_mapping({"bsm:1": 1}),
            ),
            ConstructionOperation(
                "s1", "r1", OperationKind.SWAP,
                input_segment_ids=("c", "d"),
                output_segment_id="o1",
                output_endpoints=(2, 4),
                resource_demand=ResourceDemand.from_mapping({"bsm:3": 1}),
            ),
        )
        from qnet_core.sequence_scheduler import SequenceConcurrencyScheduler

        scheduler = SequenceConcurrencyScheduler(
            {"bsm:1": 1, "bsm:3": 1},
            supports_concurrent_swaps=True,
        )
        result = scheduler.validate(swaps, segments=segments)
        self.assertFalse(result.feasible)
        self.assertEqual(
            result.reason,
            "concurrent swaps have shared physical node",
        )

    def test_generation_batch_rolls_back_if_later_prepare_raises(self):
        backend = self._backend()
        before_protocol_counts = {
            node: len(router.protocols) for node, router in backend.nodes.items()
        }
        real_prepare = backend._prepare_generation
        calls = 0

        def fail_second(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected prepare failure")
            return real_prepare(*args)

        with patch.object(backend, "_prepare_generation", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "injected prepare failure"):
                backend.begin_generation(
                    (ResourceClaim(0, 1, 0), ResourceClaim(2, 3, 0)),
                    "allocation",
                )

        self.assertFalse(backend.pairs)
        self.assertEqual(
            {node: len(router.protocols) for node, router in backend.nodes.items()},
            before_protocol_counts,
        )
        self.assertEqual(backend.node_free_slots(0), 4)
        self.assertEqual(backend.node_free_slots(1), 4)

    def test_swap_launch_exception_restores_ready_state_and_reservations(self):
        dag = left_deep_path_dag("r", (0, 1, 2))
        backend = self._backend()
        executor = SequenceConstructionExecutor(
            (dag,), backend, self._capacities(), horizon_ps=200_000
        )
        executor.launch(executor.ready_operations())
        executor.advance_to_next_event()
        swap = executor.ready_operations()[0]
        counter_before = backend._counter

        with patch.object(
            backend._EntanglementSwappingA,
            "create",
            side_effect=RuntimeError("injected swap failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected swap failure"):
                executor.launch((swap,))

        self.assertEqual(backend._counter, counter_before)
        self.assertFalse(executor.has_in_flight)
        self.assertEqual(executor.ready_operations(), (swap,))
        self.assertNotIn(swap.op_id, dag.started)
        self.assertNotIn(swap.op_id, dag.dead)
        self.assertTrue(all(
            resource.reserved_by is None for resource in backend.resources()
        ))

    def test_snapshot_tracks_resident_memory_and_releases_swap_inputs(self):
        dag = left_deep_path_dag("r", (0, 1, 2))
        backend = self._backend()
        executor = SequenceConstructionExecutor(
            (dag,), backend, self._capacities(), horizon_ps=200_000
        )
        executor.launch(executor.ready_operations())
        executor.advance_to_next_event()

        generated = executor.snapshot()
        self.assertEqual(dict(generated.reservations)["memory:1"], 2)
        self.assertEqual(dict(generated.reservations)["link:0-1"], 1)
        self.assertNotIn("genlane:0-1", dict(generated.reservations))
        backend_state = dict(generated.backend_state)
        self.assertIn("node_memory", backend_state)
        self.assertIn("pair_reservations", backend_state)
        self.assertIn("protocol_arbiter", backend_state)
        self.assertIn("timeline_pending_event_count", backend_state)

        executor.launch(executor.ready_operations())
        in_flight = executor.snapshot()
        self.assertEqual(dict(in_flight.reservations)["bsm:1"], 1)
        executor.advance_to_next_event()
        swapped = executor.snapshot()
        reservations = dict(swapped.reservations)
        self.assertEqual(reservations["memory:0"], 1)
        self.assertEqual(reservations["memory:2"], 1)
        self.assertNotIn("memory:1", reservations)
        self.assertNotIn("link:0-1", reservations)
        self.assertNotIn("link:1-2", reservations)

    def test_cross_epoch_launch_is_rejected_by_conservative_backend(self):
        backend = SequenceBackend(EpisodeSpec(
            seed=107,
            nodes=(0, 1, 2, 3, 4),
            edges=((0, 1), (1, 2), (3, 4)),
            requests=(),
            horizon=100,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                node_memory_capacity=4,
                quantum_distance_m=1.0,
            ),
        ))
        first = left_deep_path_dag("r0", (0, 1, 2))
        second = left_deep_path_dag("r1", (3, 4))
        capacities = self._capacities()
        capacities.update({
            "link:3-4": 1,
            "genlane:3-4": 1,
            "memory:3": 4,
            "memory:4": 4,
        })
        executor = SequenceConstructionExecutor(
            (first, second), backend, capacities, horizon_ps=200_000
        )
        first_generation = tuple(
            operation for operation in executor.ready_operations()
            if operation.request_id == "r0"
        )
        executor.launch(first_generation)
        executor.advance_to_next_event()
        swap = next(
            operation for operation in executor.dags["r0"].operations
            if operation.kind == "SWAP"
        )
        executor.launch((swap,))
        second_generation = next(
            operation for operation in executor.dags["r1"].operations
            if operation.kind == "GEN"
        )
        with self.assertRaisesRegex(
            ValueError,
            "protocol arbiter rejected launch: operations are in flight",
        ):
            executor.launch((second_generation,))
        executor.advance_to_next_event()
        self.assertFalse(executor.has_in_flight)

    def test_horizon_timeout_settles_all_remaining_operations(self):
        backend = self._backend()
        operations = (
            ConstructionOperation(
                "short", "r", OperationKind.RELEASE, duration_ps=1
            ),
            ConstructionOperation(
                "late", "r", OperationKind.RELEASE, duration_ps=2
            ),
        )
        executor = SequenceConstructionExecutor(
            (ConstructionDAG("r", operations),),
            backend,
            self._capacities(),
            horizon_ps=1,
        )
        executor.launch(operations)
        batch = executor.advance_to_next_event()
        self.assertTrue(batch.terminal)
        self.assertEqual(len(batch.events), 2)
        self.assertFalse(executor.has_in_flight)
        self.assertTrue(executor.terminated)
        self.assertEqual(
            {event.failure_cause for event in batch.events},
            {"", "horizon_timeout"},
        )

    def test_snapshot_refreshes_logical_fidelity_after_wait(self):
        backend = self._backend()
        dag = left_deep_path_dag("r", (0, 1))
        executor = SequenceConstructionExecutor(
            (dag,), backend, self._capacities(), horizon_ps=200_000_000
        )
        executor.launch(executor.ready_operations())
        executor.advance_to_next_event()
        initial = executor.snapshot().segments[0].fidelity
        executor.wait_until(executor.physical_time_ps + 50_000_000)
        current = executor.snapshot().segments[0].fidelity
        self.assertLess(current, initial)

    def test_physical_completion_before_nominal_window_beats_horizon_timeout(self):
        backend = SequenceBackend(EpisodeSpec(
            seed=109,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(),
            horizon=100,
            physical=PhysicalConfig(
                generation_probability=1.0,
                quantum_distance_m=1000.0,
            ),
        ))
        dag = left_deep_path_dag("r", (0, 1))
        executor = SequenceConstructionExecutor(
            (dag,),
            backend,
            {
                "link:0-1": 1,
                "genlane:0-1": 1,
                "memory:0": 2,
                "memory:1": 2,
            },
            horizon_ps=20_000_000,
        )
        executor.launch(executor.ready_operations())

        batch = executor.advance_to_next_event()

        self.assertTrue(batch.events[0].success)
        self.assertLess(batch.physical_time_ps, executor.horizon_ps)
        self.assertEqual(batch.physical_time_ps, executor.snapshot().physical_time_ps)

    def test_launch_rejects_forged_operation_payload(self):
        dag = left_deep_path_dag("r", (0, 1))
        backend = self._backend()
        executor = SequenceConstructionExecutor(
            (dag,), backend, self._capacities(), horizon_ps=200_000
        )
        operation = executor.ready_operations()[0]
        forged = replace(operation, output_endpoints=(1, 2))
        with self.assertRaisesRegex(ValueError, "canonical operation"):
            executor.launch((forged,))
        self.assertEqual(executor.ready_operations(), (operation,))
        self.assertFalse(executor.has_in_flight)

    def test_launch_rejects_missing_physical_output_hold(self):
        operation = left_deep_path_dag("r", (0, 1)).operations[0]
        forged_dag_operation = replace(operation, output_resource_hold=ResourceDemand())
        executor = SequenceConstructionExecutor(
            (ConstructionDAG("r", (forged_dag_operation,)),),
            self._backend(),
            self._capacities(),
            horizon_ps=200_000,
        )
        with self.assertRaisesRegex(ValueError, "output resource hold is incomplete"):
            executor.launch((forged_dag_operation,))

    def test_launch_rejects_physical_output_hold_amount_above_one(self):
        backend = SequenceBackend(EpisodeSpec(
            seed=115,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(),
            horizon=100,
            physical=PhysicalConfig(
                generation_probability=1.0,
                memory_capacity=2,
                node_memory_capacity=2,
                max_width=2,
                quantum_distance_m=1.0,
            ),
        ))
        first = left_deep_path_dag("r0", (0, 1)).operations[0]
        second = left_deep_path_dag("r1", (0, 1)).operations[0]
        second = replace(
            second,
            output_resource_hold=ResourceDemand.from_mapping({
                "link:0-1": 1,
                "memory:0": 2,
                "memory:1": 2,
            }),
        )
        capacities = {
            "link:0-1": 2,
            "genlane:0-1": 2,
            "memory:0": 2,
            "memory:1": 2,
        }
        executor = SequenceConstructionExecutor(
            (ConstructionDAG("r0", (first,)), ConstructionDAG("r1", (second,))),
            backend,
            capacities,
            horizon_ps=200_000,
        )
        executor.launch((first,))
        executor.advance_to_next_event()
        snapshot_oracle = CapacityFeasibilityOracle.from_snapshot(executor.snapshot())
        self.assertFalse(snapshot_oracle.check((second,)).feasible)
        with self.assertRaisesRegex(ValueError, "must reserve exactly one"):
            executor.launch((second,))

    def test_launch_rejects_nonphysical_swap_output_hold(self):
        dag = left_deep_path_dag("r", (0, 1, 2))
        swap = next(operation for operation in dag.operations if operation.kind == OperationKind.SWAP)
        forged = replace(
            swap,
            output_resource_hold=ResourceDemand.from_mapping({
                "memory:0": 1,
                "memory:2": 1,
                "memory:1": 1,
            }),
        )
        dag = ConstructionDAG(
            "r",
            tuple(forged if operation.op_id == swap.op_id else operation for operation in dag.operations),
        )
        executor = SequenceConstructionExecutor(
            (dag,), self._backend(), self._capacities(), horizon_ps=200_000
        )
        executor.launch(tuple(operation for operation in executor.ready_operations()))
        executor.advance_to_next_event()
        with self.assertRaisesRegex(ValueError, "contains non-physical resources"):
            executor.launch(executor.ready_operations())

    def test_launch_rejects_missing_swap_node_mutex(self):
        dag = left_deep_path_dag("r", (0, 1, 2))
        swap = next(
            operation
            for operation in dag.operations
            if operation.kind == OperationKind.SWAP
        )
        demand = swap.resource_demand.as_dict()
        del demand["swapnode:0"]
        forged = replace(
            swap,
            resource_demand=ResourceDemand.from_mapping(demand),
        )
        dag = ConstructionDAG(
            "r",
            tuple(
                forged if operation.op_id == swap.op_id else operation
                for operation in dag.operations
            ),
        )
        executor = SequenceConstructionExecutor(
            (dag,), self._backend(), self._capacities(), horizon_ps=200_000
        )
        executor.launch(executor.ready_operations())
        executor.advance_to_next_event()
        with self.assertRaisesRegex(ValueError, "missing swapnode:0"):
            executor.launch(executor.ready_operations())

    def test_swap_output_endpoints_must_match_physical_outer_nodes(self):
        dag = left_deep_path_dag("r", (0, 1, 2))
        swap = next(operation for operation in dag.operations if operation.kind == OperationKind.SWAP)
        wrong_swap = replace(
            swap,
            output_endpoints=(1, 2),
            output_resource_hold=ResourceDemand.from_mapping({
                "memory:1": 1,
                "memory:2": 1,
            }),
        )
        dag = ConstructionDAG(
            "r",
            tuple(wrong_swap if operation.op_id == swap.op_id else operation for operation in dag.operations),
        )
        executor = SequenceConstructionExecutor(
            (dag,), self._backend(), self._capacities(), horizon_ps=200_000
        )
        generations = tuple(operation for operation in executor.ready_operations())
        executor.launch(generations)
        executor.advance_to_next_event()
        with self.assertRaisesRegex(ValueError, "output endpoints do not match"):
            executor.launch(executor.ready_operations())

    def test_launch_rejects_cross_request_segment_consumption(self):
        first = left_deep_path_dag("r0", (0, 1)).operations[0]
        release = ConstructionOperation(
            "r1:release", "r1", OperationKind.RELEASE,
            input_segment_ids=(first.output_segment_id or "",),
        )
        executor = SequenceConstructionExecutor(
            (ConstructionDAG("r0", (first,)), ConstructionDAG("r1", (release,))),
            self._backend(), self._capacities(), horizon_ps=200_000,
        )
        executor.launch((first,))
        executor.advance_to_next_event()
        with self.assertRaisesRegex(ValueError, "another request's segment"):
            executor.launch((release,))

    def test_dynamic_dag_operation_id_collision_is_rejected(self):
        first = left_deep_path_dag("r0", (0, 1)).operations[0]
        second = left_deep_path_dag("r1", (2, 3)).operations[0]
        executor = SequenceConstructionExecutor(
            (ConstructionDAG("r0", (first,)), ConstructionDAG("r1", (second,))),
            self._backend(), self._capacities(), horizon_ps=200_000,
        )
        executor.dags["r1"].add_operation(
            replace(second, op_id=first.op_id, output_segment_id="r1:extra")
        )
        with self.assertRaisesRegex(ValueError, "operation id is shared"):
            executor.ready_operations()

    def test_dynamic_dag_output_id_collision_is_rejected(self):
        first = left_deep_path_dag("r0", (0, 1)).operations[0]
        second = left_deep_path_dag("r1", (2, 3)).operations[0]
        executor = SequenceConstructionExecutor(
            (ConstructionDAG("r0", (first,)), ConstructionDAG("r1", (second,))),
            self._backend(), self._capacities(), horizon_ps=200_000,
        )
        executor.dags["r1"].add_operation(
            replace(second, op_id="r1:extra", output_segment_id=first.output_segment_id)
        )
        with self.assertRaisesRegex(ValueError, "output segment id is shared"):
            executor.ready_operations()

    def test_declared_duration_is_a_logical_completion_lower_bound(self):
        backend = self._backend()
        duration = 2 * backend.generation_duration_ps
        operation = replace(
            left_deep_path_dag("r", (0, 1)).operations[0],
            duration_ps=duration,
        )
        executor = SequenceConstructionExecutor(
            (ConstructionDAG("r", (operation,)),),
            backend,
            self._capacities(),
            horizon_ps=duration + backend.generation_duration_ps,
        )
        executor.launch((operation,))
        batch = executor.advance_to_next_event()
        self.assertTrue(batch.events[0].success)
        self.assertGreaterEqual(batch.physical_time_ps, duration)
        self.assertEqual(batch.physical_time_ps, executor.event_log[-1].physical_time_ps)

    def test_pending_epoch_refreshes_unrelated_segment_fidelity(self):
        backend = SequenceBackend(EpisodeSpec(
            seed=113,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(),
            horizon=200,
            physical=PhysicalConfig(
                generation_probability=1.0,
                memory_lifetime=1000,
                quantum_distance_m=1.0,
            ),
        ))
        generation = left_deep_path_dag("r", (0, 1)).operations[0]
        first_release = ConstructionOperation(
            "release:short", "r", OperationKind.RELEASE,
            duration_ps=50_000_000,
        )
        second_release = ConstructionOperation(
            "release:long", "r", OperationKind.RELEASE,
            duration_ps=100_000_000,
        )
        dag = ConstructionDAG("r", (generation, first_release, second_release))
        executor = SequenceConstructionExecutor(
            (dag,),
            backend,
            {
                "link:0-1": 1,
                "genlane:0-1": 1,
                "memory:0": 2,
                "memory:1": 2,
            },
            horizon_ps=150_000_000,
        )
        executor.launch((generation,))
        executor.advance_to_next_event()
        initial = executor.snapshot().segments[0].fidelity
        executor.launch((first_release, second_release))
        executor.advance_to_next_event()
        refreshed = executor.snapshot().segments[0].fidelity
        self.assertLess(refreshed, initial)

    def test_executor_rejects_cross_dag_output_segment_collision(self):
        first = left_deep_path_dag("r0", (0, 1)).operations[0]
        second = left_deep_path_dag("r1", (0, 1)).operations[0]
        first = replace(first, output_segment_id="shared")
        second = replace(second, output_segment_id="shared")
        with self.assertRaisesRegex(ValueError, "shared by multiple operations"):
            SequenceConstructionExecutor(
                (ConstructionDAG("r0", (first,)), ConstructionDAG("r1", (second,))),
                self._backend(),
                self._capacities(),
                horizon_ps=200_000,
            )

    def test_dynamic_dag_can_be_unregistered_and_reused(self):
        executor = SequenceConstructionExecutor(
            (),
            self._backend(),
            self._capacities(),
            horizon_ps=200_000,
        )
        dag = left_deep_path_dag("r", (0, 1))
        executor.register_dag(dag)
        executor.launch(executor.ready_operations())
        batch = executor.advance_to_next_event()
        self.assertTrue(batch.events[0].success)
        executor.release_request("r")
        executor.unregister_dag("r")
        self.assertNotIn("r", executor.dags)
        executor.register_dag(dag)
        self.assertIn("r", executor.dags)


if __name__ == "__main__":
    unittest.main()
