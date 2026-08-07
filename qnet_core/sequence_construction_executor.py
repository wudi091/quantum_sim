"""Event-driven construction executor backed by SeQUeNCe.

The class intentionally exposes only the neutral construction contracts.  The
SeQUeNCe adapter is kept behind ``SequenceBackend`` and is never imported by a
planner or an algorithm package.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping

from .command_api import ResourceClaim, SwapAction
from .construction_api import (
    ConstructionDAG,
    ConstructionOperation,
    ConstructionSnapshot,
    ExecutionEvent,
    ExecutionEventBatch,
    InFlightOperation,
    LogicalSegment,
    OperationKind,
    ResourceDemand,
)
from .construction_decoder import CapacityFeasibilityOracle
from .construction_repair import generate_repair_options
from .sequence_backend import PreparedGeneration, PreparedSwap, SequenceBackend
from .sequence_protocol_arbiter import ProtocolRequest
from .sequence_scheduler import SequenceConcurrencyScheduler


@dataclass(frozen=True)
class _Pending:
    operation: ConstructionOperation
    attempt_id: str
    completion_time_ps: int
    logical_completion_time_ps: int
    generation: PreparedGeneration | None = None
    swap: PreparedSwap | None = None


class SequenceConstructionExecutor:
    """Construction DAG state machine using SeQUeNCe for physical effects."""

    def __init__(
        self,
        dags: Iterable[ConstructionDAG],
        backend: SequenceBackend,
        capacities: Mapping[str, int],
        initial_segments: Iterable[LogicalSegment] = (),
        horizon_ps: int | None = None,
    ):
        # Executors own mutable DAG state.  Rebuild each graph so a catalogue
        # can be evaluated repeatedly and independently across environments.
        dag_list = tuple(dag.clone() for dag in dags)
        self.dags = {dag.request_id: dag for dag in dag_list}
        if len(self.dags) != len(dag_list) or not dag_list:
            raise ValueError("construction DAG request IDs must be unique and non-empty")
        self.backend = backend
        self.capacities = {str(key): int(value) for key, value in capacities.items()}
        self.oracle = CapacityFeasibilityOracle(self.capacities)
        self.scheduler = SequenceConcurrencyScheduler(
            self.capacities,
            supports_inter_epoch_launch=getattr(
                backend, "supports_inter_epoch_launch", False
            ),
            supports_mixed_operation_concurrency=getattr(
                backend, "supports_mixed_operation_concurrency", False
            ),
            supports_concurrent_swaps=getattr(
                backend, "supports_concurrent_swaps", False
            ),
        )
        self._request_required_fidelity = {
            request.id: float(request.required_fidelity)
            for request in backend.spec.requests
        }
        self._request_endpoints = {
            request.id: frozenset((request.source, request.destination))
            for request in backend.spec.requests
        }
        self.horizon_ps = int(
            backend.spec.horizon * backend.spec.physical.slot_duration_ps
            if horizon_ps is None else horizon_ps
        )
        if self.horizon_ps < backend.physical_time_ps:
            raise ValueError("horizon_ps precedes current physical time")
        initial_segment_list = tuple(initial_segments)
        if initial_segment_list:
            raise ValueError(
                "SeQUeNCe executor does not accept logical-only initial_segments"
            )
        self._segments: dict[str, LogicalSegment] = {}
        self._initial_segment_ids = frozenset(self._segments)
        self._output_owners: dict[str, str] = {}
        self._operation_owners: dict[str, str] = {}
        self._refresh_global_registry()
        self._physical_by_segment: dict[str, str] = {}
        self._pending: dict[str, _Pending] = {}
        self._generation_lanes: dict[tuple[int, int], set[int]] = {}
        self._expired_segments: list[LogicalSegment] = []
        self._attempts: dict[str, int] = {}
        self._counter = 0
        self._terminated = False
        self.event_log: list[ExecutionEvent] = []

    def _effective_required_fidelity(self, operation: ConstructionOperation) -> float:
        """Apply the request-level delivery threshold to terminal outputs.

        A repair DAG is supplied by the planner, so its DTO threshold cannot
        be trusted as the request's service-level contract.  Intermediate
        segments keep their operation-local threshold; only an output whose
        endpoints equal the request endpoints is a terminal delivery gate.
        """

        request_threshold = self._request_required_fidelity.get(operation.request_id)
        if request_threshold is None or operation.output_endpoints is None:
            return operation.required_fidelity
        if frozenset(operation.output_endpoints) != self._request_endpoints.get(
            operation.request_id, frozenset()
        ):
            return operation.required_fidelity
        return max(operation.required_fidelity, request_threshold)

    @property
    def physical_time_ps(self) -> int:
        return self.backend.physical_time_ps

    @property
    def time(self) -> int:
        return self.physical_time_ps

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def has_in_flight(self) -> bool:
        return bool(self._pending)

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter:08d}"

    def _refresh_global_registry(self) -> None:
        """Rebuild cross-DAG identity indexes before accepting mutable DAG state."""

        operation_owners: dict[str, str] = {}
        output_owners: dict[str, str] = {}
        for dag in self.dags.values():
            for operation in dag.operations:
                if operation.request_id != dag.request_id:
                    raise ValueError(
                        f"operation belongs to the wrong request DAG: {operation.op_id}"
                    )
                owner = operation_owners.get(operation.op_id)
                if owner is not None:
                    raise ValueError(
                        f"operation id is shared by multiple DAGs: {operation.op_id}"
                    )
                operation_owners[operation.op_id] = dag.request_id
                output_id = operation.output_segment_id
                if output_id is None:
                    continue
                output_owner = output_owners.get(output_id)
                if output_owner is not None:
                    raise ValueError(
                        f"output segment id is shared by multiple operations: {output_id}"
                    )
                output_owners[output_id] = operation.op_id
        conflicts = self._initial_segment_ids.intersection(output_owners)
        if conflicts:
            raise ValueError(
                f"initial segment id conflicts with a DAG output: {sorted(conflicts)[0]}"
            )
        self._operation_owners = operation_owners
        self._output_owners = output_owners

    def _available_segment_ids(self) -> set[str]:
        consumed = {
            segment_id
            for pending in self._pending.values()
            for segment_id in pending.operation.input_segment_ids
        }
        physically_owned = {
            segment_id
            for pending in self._pending.values()
            if pending.operation.kind in {OperationKind.GEN, OperationKind.SWAP}
            for segment_id in pending.operation.input_segment_ids
        }
        # Never synchronize while a protocol owns its input pair.  Use the
        # backend's read-only view for unrelated pairs so expiration remains
        # visible during another operation's in-flight window.
        if self._pending:
            for segment_id, physical_id in tuple(self._physical_by_segment.items()):
                if segment_id in physically_owned:
                    continue
                resource = self.backend.resource_without_sync(physical_id)
                if resource is None:
                    self._physical_by_segment.pop(segment_id, None)
                    expired = self._segments.pop(segment_id, None)
                    if expired is not None:
                        self._expired_segments.append(expired)
                    self.backend.discard_pair(physical_id)
                elif self._segments[segment_id].fidelity != resource.fidelity:
                    self._segments[segment_id] = replace(
                        self._segments[segment_id], fidelity=resource.fidelity
                    )
            return set(self._segments) - consumed
        # At a decision boundary it is safe to synchronize and refresh both
        # fidelity and expiration state from SeQUeNCe.
        resources = {resource.pair_id: resource for resource in self.backend.resources()}
        for segment_id, physical_id in tuple(self._physical_by_segment.items()):
            resource = resources.get(physical_id)
            if resource is None:
                self._physical_by_segment.pop(segment_id, None)
                expired = self._segments.pop(segment_id, None)
                if expired is not None:
                    self._expired_segments.append(expired)
            elif self._segments[segment_id].fidelity != resource.fidelity:
                self._segments[segment_id] = replace(
                    self._segments[segment_id], fidelity=resource.fidelity
                )
        return set(self._segments) - consumed

    def _drain_expiration_events(self) -> tuple[ExecutionEvent, ...]:
        if not self._expired_segments:
            return ()
        expired = tuple(sorted(
            self._expired_segments, key=lambda segment: segment.segment_id
        ))
        self._expired_segments.clear()
        surviving = tuple(sorted(self._segments))
        events = []
        for segment in expired:
            event = ExecutionEvent(
                event_id=self._next_id("event"),
                operation_id=segment.source_operation_id or segment.segment_id,
                request_id=segment.request_id,
                attempt_id=(
                    f"{segment.source_operation_id or segment.segment_id}:expiration:"
                    f"{self.physical_time_ps}"
                ),
                event_kind="expiration",
                physical_time_ps=self.physical_time_ps,
                success=False,
                failure_cause="expiration",
                consumed_segment_ids=(segment.segment_id,),
                surviving_segment_ids=surviving,
                released_resources=segment.held_resources.entries,
                in_flight_operation_ids=tuple(sorted(self._pending)),
            )
            events.append(event)
            self.event_log.append(event)
        return tuple(events)

    def _next_expiration_time_ps(self) -> int | None:
        """Return the earliest physical memory expiration after ``now``."""

        state = dict(self.backend.construction_state())
        raw_events = state.get("expiration_events", ())
        times = []
        for item in raw_events:
            if len(item) < 2:
                continue
            try:
                timestamp = int(item[1])
            except (TypeError, ValueError):
                continue
            if timestamp > self.physical_time_ps:
                times.append(timestamp)
        return min(times) if times else None

    def next_expiration_time_ps(self) -> int | None:
        """Expose the next physical expiration as a neutral event boundary."""

        return self._next_expiration_time_ps()

    def available_segments(self) -> tuple[LogicalSegment, ...]:
        return tuple(
            sorted((self._segments[segment_id] for segment_id in self._available_segment_ids()),
                   key=lambda segment: segment.segment_id)
        )

    def ready_operations(self) -> tuple[ConstructionOperation, ...]:
        self._refresh_global_registry()
        if self._pending and not getattr(
            self.backend, "supports_inter_epoch_launch", False
        ):
            return ()
        available = self._available_segment_ids()
        operations = tuple(sorted(
            (
                operation
                for dag in self.dags.values()
                for operation in dag.operations
                if operation.op_id in dag.ready_ids(available, set(self._pending))
            ),
            key=lambda operation: operation.canonical_key,
        ))
        if not getattr(self.backend, "supports_concurrent_swaps", False):
            swaps = [operation for operation in operations if operation.kind == OperationKind.SWAP]
            pending_swaps = [
                pending.operation
                for pending in self._pending.values()
                if pending.operation.kind == OperationKind.SWAP
            ]
            if pending_swaps:
                operations = tuple(
                    operation for operation in operations
                    if operation.kind != OperationKind.SWAP
                )
            elif len(swaps) > 1:
                first = swaps[0]
                operations = tuple(
                    operation for operation in operations
                    if operation.kind != OperationKind.SWAP or operation.op_id == first.op_id
                )
        if not getattr(self.backend, "supports_mixed_operation_concurrency", False):
            generations = [operation for operation in operations if operation.kind == OperationKind.GEN]
            swaps = [operation for operation in operations if operation.kind == OperationKind.SWAP]
            if generations and swaps:
                operations = tuple(generations)
        return self.scheduler.pack(
            operations,
            pending_operations=tuple(
                pending.operation for pending in self._pending.values()
            ),
            segments=tuple(self._segments.values()),
        )

    def _usage(self, demands: Iterable[ResourceDemand] = ()) -> dict[str, int]:
        usage: dict[str, int] = {}
        for segment in self._segments.values():
            for resource, amount in segment.held_resources.items():
                usage[resource] = usage.get(resource, 0) + amount
        for pending in self._pending.values():
            for resource, amount in pending.operation.resource_demand.items():
                usage[resource] = usage.get(resource, 0) + amount
        for demand in demands:
            for resource, amount in demand.items():
                usage[resource] = usage.get(resource, 0) + amount
        return usage

    def _post_completion_usage(
        self, operations: Iterable[ConstructionOperation]
    ) -> dict[str, int]:
        consumed = {
            segment_id
            for operation in operations
            for segment_id in operation.input_segment_ids
        }
        pending_consumed = {
            segment_id
            for pending in self._pending.values()
            for segment_id in pending.operation.input_segment_ids
        }
        usage: dict[str, int] = {}
        for segment_id, segment in self._segments.items():
            if segment_id in consumed or segment_id in pending_consumed:
                continue
            for resource, amount in segment.held_resources.items():
                usage[resource] = usage.get(resource, 0) + amount
        for pending in self._pending.values():
            for resource, amount in pending.operation.output_resource_hold.items():
                usage[resource] = usage.get(resource, 0) + amount
        for operation in operations:
            for resource, amount in operation.output_resource_hold.items():
                usage[resource] = usage.get(resource, 0) + amount
        return usage

    def _protocol_requests(
        self, operations: Iterable[ConstructionOperation]
    ) -> tuple[ProtocolRequest, ...]:
        segments = {
            segment.segment_id: segment for segment in self._segments.values()
        }
        requests = []
        for operation in operations:
            request = ProtocolRequest.from_operation(operation, segments)
            if request is not None:
                requests.append(request)
        return tuple(requests)

    def _validate_launch(self, operations: tuple[ConstructionOperation, ...]) -> None:
        self._refresh_global_registry()
        if self._terminated:
            raise RuntimeError("executor is terminated")
        if not operations:
            raise ValueError("launch requires at least one operation")
        if len({operation.op_id for operation in operations}) != len(operations):
            raise ValueError("duplicate operation in launch set")
        available = self._available_segment_ids()
        for operation in operations:
            dag = self.dags.get(operation.request_id)
            if dag is None:
                raise ValueError(f"unknown request DAG: {operation.request_id}")
            canonical = dag.operation(operation.op_id)
            if canonical != operation:
                raise ValueError(
                    f"operation object does not match DAG canonical operation: {operation.op_id}"
                )
            if operation.op_id not in dag.ready_ids(available, set(self._pending)):
                raise ValueError(f"operation is not ready: {operation.op_id}")
            self._validate_output_hold(operation)
            for segment_id in operation.input_segment_ids:
                segment = self._segments.get(segment_id)
                if segment is not None and segment.request_id != operation.request_id:
                    raise ValueError(
                        "operation cannot consume another request's segment"
                    )
            if operation.kind == OperationKind.GEN and operation.output_endpoints is None:
                raise ValueError("GEN requires output endpoints")
            if operation.kind == OperationKind.SWAP:
                if any(segment_id not in self._physical_by_segment for segment_id in operation.input_segment_ids):
                    raise ValueError(f"SWAP input segment is not physical: {operation.op_id}")
                action = self._make_swap_action(operation)
                left_resource = self.backend.resource(action.left_pair_id)
                right_resource = self.backend.resource(action.right_pair_id)
                if left_resource is None or right_resource is None:
                    raise ValueError("SWAP input segment is no longer physical")
                expected_outer = {
                    next(node for node in left_resource.endpoints if node != action.middle),
                    next(node for node in right_resource.endpoints if node != action.middle),
                }
                if set(operation.output_endpoints or ()) != expected_outer:
                    raise ValueError(
                        f"SWAP output endpoints do not match physical outer endpoints: {operation.op_id}"
                    )
                if not self.backend.can_begin_swap(action):
                    raise ValueError(f"physical backend rejected swap: {operation.op_id}")
                expected_bsm = f"bsm:{action.middle}"
                if operation.resource_demand.get(expected_bsm) < 1:
                    raise ValueError(
                        f"SWAP resource demand must reserve {expected_bsm}"
                    )
            if operation.kind == OperationKind.GEN:
                assert operation.output_endpoints is not None
                left, right = operation.output_endpoints
                edge = f"{min(left, right)}-{max(left, right)}"
                expected = (
                    f"link:{edge}",
                    f"genlane:{edge}",
                    f"memory:{left}",
                    f"memory:{right}",
                )
                if any(
                    operation.resource_demand.get(resource) < 1
                    for resource in expected
                ):
                    raise ValueError(
                        f"GEN resource demand is incomplete for edge {edge}"
                    )
        arbiter = getattr(self.backend, "protocol_arbiter", None)
        if arbiter is not None:
            arbiter_result = arbiter.validate(
                self._protocol_requests(operations),
                active=self._protocol_requests(
                    pending.operation for pending in self._pending.values()
                ),
            )
            if not arbiter_result.feasible:
                raise ValueError(
                    f"protocol arbiter rejected launch: {arbiter_result.reason}"
                )
        scheduler_result = self.scheduler.validate(
            operations,
            pending_operations=tuple(
                pending.operation for pending in self._pending.values()
            ),
            segments=tuple(self._segments.values()),
        )
        if not scheduler_result.feasible:
            raise ValueError(f"scheduler rejected launch: {scheduler_result.reason}")
        feasibility = self.oracle.check(operations)
        if not feasibility.feasible:
            raise ValueError(feasibility.reason)
        for resource, amount in self._usage(operation.resource_demand for operation in operations).items():
            if amount > self.capacities.get(resource, 0):
                raise ValueError(f"capacity exceeded: {resource}")
        for resource, amount in self._post_completion_usage(operations).items():
            if amount > self.capacities.get(resource, 0):
                raise ValueError(f"post-completion capacity exceeded: {resource}")
        inputs = [segment_id for operation in operations for segment_id in operation.input_segment_ids]
        if len(inputs) != len(set(inputs)):
            raise ValueError("input segment consumed twice in launch set")
        output_ids = [
            operation.output_segment_id
            for operation in operations
            if operation.output_segment_id is not None
        ]
        if len(output_ids) != len(set(output_ids)):
            raise ValueError("output segment produced twice in launch set")
        pending_output_ids = {
            pending.operation.output_segment_id
            for pending in self._pending.values()
            if pending.operation.output_segment_id is not None
        }
        for output_id in output_ids:
            if output_id in self._segments:
                raise ValueError(f"output segment id is already live: {output_id}")
            if output_id in pending_output_ids:
                raise ValueError(f"output segment id is already in flight: {output_id}")

    def _validate_output_hold(self, operation: ConstructionOperation) -> None:
        hold = operation.output_resource_hold
        for resource, amount in hold.items():
            if amount > self.capacities.get(resource, 0):
                raise ValueError(f"output resource hold exceeds capacity: {resource}")
        if operation.kind == OperationKind.RELEASE:
            if hold:
                raise ValueError("RELEASE cannot declare an output resource hold")
            return
        if operation.output_endpoints is None:
            raise ValueError(f"{operation.kind} requires output endpoints")
        left, right = operation.output_endpoints
        required = {f"memory:{left}": 1, f"memory:{right}": 1}
        if operation.kind == OperationKind.GEN:
            edge = f"{min(left, right)}-{max(left, right)}"
            required[f"link:{edge}"] = 1
        actual = hold.as_dict()
        missing = [resource for resource, amount in required.items() if actual.get(resource) != amount]
        extra = sorted(set(actual).difference(required))
        if missing:
            resource = missing[0]
            if actual.get(resource, 0) < required[resource]:
                raise ValueError(
                    f"{operation.kind} output resource hold is incomplete: {resource}"
                )
            raise ValueError(
                f"{operation.kind} output resource hold must reserve exactly one: {resource}"
            )
        if extra:
            raise ValueError(
                f"{operation.kind} output resource hold contains non-physical resources: {extra[0]}"
            )

    def _generation_claims(
        self, operations: Iterable[ConstructionOperation]
    ) -> dict[str, ResourceClaim]:
        lane_by_edge: dict[tuple[int, int], set[int]] = {
            edge: set(lanes) for edge, lanes in self._generation_lanes.items()
        }
        for pending in self._pending.values():
            if pending.operation.kind != OperationKind.GEN:
                continue
            if pending.generation is None:
                continue
            edge = tuple(sorted(pending.generation.claim.endpoints))
            lane_by_edge.setdefault(edge, set()).add(pending.generation.claim.lane)
        claims: dict[str, ResourceClaim] = {}
        topology_edges = {
            tuple(sorted(edge)) for edge in self.backend.spec.edges
        }
        for operation in sorted(operations, key=lambda item: item.canonical_key):
            if operation.kind != OperationKind.GEN:
                continue
            if operation.output_endpoints is None:
                raise ValueError("GEN requires output endpoints")
            edge = tuple(sorted(operation.output_endpoints))
            if edge not in topology_edges:
                raise ValueError(f"GEN references non-topology edge {edge}")
            used = lane_by_edge.setdefault(edge, set())
            lane = next(
                (candidate for candidate in range(self.backend.spec.physical.max_width)
                 if candidate not in used),
                None,
            )
            if lane is None:
                raise ValueError("GEN launch exceeds physical max_width")
            used.add(lane)
            claims[operation.op_id] = ResourceClaim(edge[0], edge[1], lane)
        return claims

    def _release_generation_lane(self, pending: _Pending) -> None:
        if pending.operation.kind != OperationKind.GEN or pending.generation is None:
            return
        edge = tuple(sorted(pending.generation.claim.endpoints))
        lanes = self._generation_lanes.get(edge)
        if lanes is None:
            return
        lanes.discard(pending.generation.claim.lane)
        if not lanes:
            self._generation_lanes.pop(edge, None)

    def _make_swap_action(self, operation: ConstructionOperation) -> SwapAction:
        physical_ids = tuple(self._physical_by_segment[segment_id] for segment_id in operation.input_segment_ids)
        if len(physical_ids) != 2 or operation.output_endpoints is None:
            raise ValueError("SWAP requires two input segments and output endpoints")
        left_resource = self.backend.resource(physical_ids[0])
        right_resource = self.backend.resource(physical_ids[1])
        if left_resource is None or right_resource is None:
            raise ValueError("SWAP input segment is no longer physical")
        intersection = set(left_resource.endpoints) & set(right_resource.endpoints)
        if len(intersection) != 1:
            raise ValueError("SWAP inputs must share exactly one middle node")
        return SwapAction(
            operation.request_id,
            next(iter(intersection)),
            physical_ids[0],
            physical_ids[1],
        )

    def launch(self, feasible_set: Iterable[ConstructionOperation]) -> tuple[str, ...]:
        operations = tuple(feasible_set)
        self._validate_launch(operations)
        now = self.physical_time_ps
        available_at_launch = self._available_segment_ids()
        attempt_ids: dict[str, str] = {}
        for operation in operations:
            attempt = self._attempts.get(operation.op_id, 0) + 1
            attempt_ids[operation.op_id] = f"{operation.op_id}:attempt:{attempt}"

        generation_ops = [operation for operation in operations if operation.kind == OperationKind.GEN]
        generation_claims = self._generation_claims(generation_ops)
        prepared_generation_items: tuple[PreparedGeneration, ...] = ()
        prepared_generations: dict[ResourceClaim, PreparedGeneration] = {}
        prepared_swaps: list[PreparedSwap] = []
        marked_started: list[ConstructionOperation] = []
        staged_pending: dict[str, _Pending] = {}
        prepared_swap_by_operation: dict[str, PreparedSwap] = {}
        allocation_id = None
        try:
            # Commit only the neutral DAG start markers first.  If a caller
            # injects a failure here, no SeQUeNCe protocol has been started.
            for operation in sorted(operations, key=lambda item: item.canonical_key):
                self.dags[operation.request_id].mark_started(
                    operation.op_id, available_at_launch
                )
                marked_started.append(operation)

            swap_actions = {
                operation.op_id: self._make_swap_action(operation)
                for operation in operations
                if operation.kind == OperationKind.SWAP
            }
            # Start SWAP protocols before GEN protocols.  SeQUeNCe's
            # timeline can otherwise process a newly scheduled generation
            # event before the already-prepared BSM handshake, even when the
            # two operations use disjoint physical nodes.
            for operation in sorted(operations, key=lambda item: item.canonical_key):
                if operation.kind != OperationKind.SWAP:
                    continue
                attempt_id = attempt_ids[operation.op_id]
                swap = self.backend.begin_swap(
                    swap_actions[operation.op_id], attempt_id
                )
                if swap is None:
                    raise ValueError(
                        f"physical backend rejected swap: {operation.op_id}"
                    )
                prepared_swaps.append(swap)
                prepared_swap_by_operation[operation.op_id] = swap

            if generation_ops:
                allocation_id = f"generation-{self._counter + 1:08d}"
                prepared_generation_items = self.backend.begin_generation(
                    tuple(generation_claims.values()), allocation_id
                )
                prepared_generations = {
                    item.claim: item for item in prepared_generation_items
                }
                if set(prepared_generations) != set(generation_claims.values()):
                    raise RuntimeError("physical backend returned incomplete generation handles")

            for operation in sorted(operations, key=lambda item: item.canonical_key):
                attempt_id = attempt_ids[operation.op_id]
                generation = None
                swap = None
                if operation.kind == OperationKind.GEN:
                    claim = generation_claims[operation.op_id]
                    generation = prepared_generations[claim]
                    duration = self.backend.generation_duration_ps
                elif operation.kind == OperationKind.SWAP:
                    swap = prepared_swap_by_operation[operation.op_id]
                    duration = self.backend.swap_duration_ps
                else:
                    duration = operation.duration_ps
                staged_pending[operation.op_id] = _Pending(
                    operation,
                    attempt_id,
                    now + max(operation.duration_ps, duration),
                    now + operation.duration_ps,
                    generation,
                    swap,
                )

        except Exception:
            for operation in reversed(marked_started):
                self.dags[operation.request_id].rollback_started(operation.op_id)
            for prepared_swap in reversed(prepared_swaps):
                self.backend.cancel_swap(prepared_swap)
            self.backend.cancel_generation(prepared_generation_items)
            raise

        if allocation_id is not None:
            self._counter += 1
        for operation in operations:
            self._attempts[operation.op_id] = (
                self._attempts.get(operation.op_id, 0) + 1
            )
        self._pending.update(staged_pending)
        for operation in generation_ops:
            prepared = staged_pending[operation.op_id].generation
            if prepared is not None:
                edge = tuple(sorted(prepared.claim.endpoints))
                self._generation_lanes.setdefault(edge, set()).add(prepared.claim.lane)
        return tuple(attempt_ids[operation.op_id] for operation in operations)

    def snapshot(self) -> ConstructionSnapshot:
        self._refresh_global_registry()
        in_flight = tuple(
            InFlightOperation(
                operation_id=pending.operation.op_id,
                request_id=pending.operation.request_id,
                attempt_id=pending.attempt_id,
                start_time_ps=pending.completion_time_ps - max(
                    pending.operation.duration_ps,
                    self.backend.generation_duration_ps if pending.operation.kind == OperationKind.GEN
                    else self.backend.swap_duration_ps if pending.operation.kind == OperationKind.SWAP
                    else pending.operation.duration_ps,
                ),
                completion_time_ps=pending.completion_time_ps,
                reserved_resources=pending.operation.resource_demand.entries,
                input_segment_ids=pending.operation.input_segment_ids,
            )
            for pending in sorted(self._pending.values(), key=lambda item: item.operation.op_id)
        )
        pending_events = tuple(
            (pending.attempt_id, pending.completion_time_ps, pending.operation.kind)
            for pending in sorted(self._pending.values(), key=lambda item: item.attempt_id)
        )
        reservations = tuple(sorted(self._usage().items()))
        arrivals = tuple(sorted(
            (
                request.id,
                request.arrival * self.backend.spec.physical.slot_duration_ps,
            )
            for request in self.backend.spec.requests
        ))
        deadlines = tuple(sorted(
            (
                request.id,
                request.deadline * self.backend.spec.physical.slot_duration_ps,
            )
            for request in self.backend.spec.requests
            if request.deadline is not None
        ))
        return ConstructionSnapshot(
            physical_time_ps=self.physical_time_ps,
            horizon_ps=self.horizon_ps,
            dag_states=tuple(dag.state() for dag in sorted(self.dags.values(), key=lambda dag: dag.request_id)),
            operations=tuple(sorted(
                (
                    operation
                    for dag in self.dags.values()
                    for operation in dag.operations
                ),
                key=lambda operation: operation.canonical_key,
            )),
            segments=tuple(sorted(self._segments.values(), key=lambda segment: segment.segment_id)),
            reservations=reservations,
            in_flight=in_flight,
            pending_events=pending_events,
            arrivals=arrivals,
            deadlines=deadlines,
            resource_capacities=tuple(sorted(self.capacities.items())),
            backend_state=self.backend.construction_state(),
        )

    def _finish_pending(self, pending: _Pending, forced_failure_cause: str = "") -> ExecutionEvent:
        operation = pending.operation
        dag = self.dags[operation.request_id]
        success = False
        failure_cause = ""
        consumed: tuple[str, ...] = ()
        surviving: tuple[str, ...] = ()
        output_segment_id = None
        output_fidelity = None
        physical_pair_id = None
        self._release_generation_lane(pending)
        if operation.kind == OperationKind.GEN:
            claim = pending.generation.claim if pending.generation is not None else None
            outcomes = self.backend.finish_generation((pending.generation,)) if pending.generation else {}
            physical_pair_id = outcomes.get(claim) if claim is not None else None
            success = physical_pair_id is not None
            if pending.generation is not None:
                # The generation allocation is an atomic launch reservation;
                # after the terminal event the resulting segments are owned by
                # the construction DAG and must be available to SWAP.
                self.backend.release_allocation(pending.generation.allocation_id)
        elif operation.kind == OperationKind.SWAP:
            physical_pair_id = self.backend.finish_swap(pending.swap) if pending.swap else None
            success = physical_pair_id is not None
            consumed = tuple(operation.input_segment_ids)
            for segment_id in consumed:
                self._physical_by_segment.pop(segment_id, None)
                self._segments.pop(segment_id, None)
        elif operation.kind == OperationKind.RELEASE:
            for segment_id in operation.input_segment_ids:
                physical_id = self._physical_by_segment.pop(segment_id, None)
                self._segments.pop(segment_id, None)
                if physical_id is not None:
                    self.backend.discard_pair(physical_id)
            success = True
            consumed = tuple(operation.input_segment_ids)
        if success and operation.output_segment_id is not None:
            resource = self.backend.resource(physical_pair_id) if physical_pair_id else None
            if resource is None:
                success = False
                failure_cause = "physical_output_missing"
            elif any(
                self._usage().get(resource_id, 0) + amount
                > self.capacities.get(resource_id, 0)
                for resource_id, amount in operation.output_resource_hold.items()
            ):
                self.backend.discard_pair(physical_pair_id)
                success = False
                failure_cause = "post_completion_capacity"
            elif resource.fidelity < self._effective_required_fidelity(operation):
                self.backend.discard_pair(physical_pair_id)
                success = False
                failure_cause = "fidelity_reject"
            else:
                output_segment_id = operation.output_segment_id
                output_fidelity = resource.fidelity
                self._physical_by_segment[output_segment_id] = physical_pair_id
                physical_endpoints = (resource.left, resource.right)
                self._segments[output_segment_id] = LogicalSegment(
                    output_segment_id,
                    operation.request_id,
                    physical_endpoints[0],
                    physical_endpoints[1],
                    self.physical_time_ps,
                    resource.fidelity,
                    operation.request_id,
                    operation.op_id,
                    operation.output_resource_hold,
                )
        if forced_failure_cause:
            if output_segment_id is not None:
                physical_id = self._physical_by_segment.pop(output_segment_id, None)
                self._segments.pop(output_segment_id, None)
                if physical_id is not None:
                    self.backend.discard_pair(physical_id)
            success = False
            output_segment_id = None
            output_fidelity = None
            failure_cause = forced_failure_cause
        if success:
            dag.mark_completed(operation.op_id)
        else:
            dag.mark_dead(operation.op_id)
            surviving = tuple(sorted(self._available_segment_ids()))
            if not failure_cause:
                failure_cause = "physical_failure"
        event = ExecutionEvent(
            event_id=self._next_id("event"),
            operation_id=operation.op_id,
            request_id=operation.request_id,
            attempt_id=pending.attempt_id,
            event_kind=operation.kind.lower(),
            physical_time_ps=self.physical_time_ps,
            success=success,
            failure_cause=failure_cause,
            consumed_segment_ids=consumed,
            surviving_segment_ids=surviving,
            output_segment_id=output_segment_id,
            output_fidelity=output_fidelity,
            released_resources=operation.resource_demand.entries,
            in_flight_operation_ids=tuple(sorted(self._pending)),
        )
        self.event_log.append(event)
        return event

    def advance_to_next_event(self, boundary_ps: int | None = None) -> ExecutionEventBatch:
        if self._terminated:
            return ExecutionEventBatch(self.physical_time_ps, (), 0, terminal=True)
        self._available_segment_ids()
        expiration_events = self._drain_expiration_events()
        if expiration_events:
            return ExecutionEventBatch(
                self.physical_time_ps,
                expiration_events,
                0,
                terminal=False,
            )
        if not self._pending:
            self._terminated = True
            return ExecutionEventBatch(self.physical_time_ps, (), 0, terminal=True)
        pending_all = tuple(self._pending.values())
        nominal_next_time = min(pending.completion_time_ps for pending in pending_all)
        boundaries = [nominal_next_time, self.horizon_ps]
        expiration_time = self._next_expiration_time_ps()
        if expiration_time is not None:
            boundaries.append(expiration_time)
        if boundary_ps is not None:
            boundary_ps = int(boundary_ps)
            if boundary_ps < self.physical_time_ps or boundary_ps > self.horizon_ps:
                raise ValueError("boundary time must lie in [current_time, horizon]")
            boundaries.append(boundary_ps)
        target_time = min(boundaries)
        previous = self.physical_time_ps
        self.backend.run_prepared_protocols(
            (
                pending.generation for pending in pending_all
                if pending.generation is not None
            ),
            (pending.swap for pending in pending_all if pending.swap is not None),
            deadline_ps=target_time,
        )
        physical_pending = tuple(
            pending for pending in pending_all
            if pending.swap is not None
            or (pending.generation is not None and pending.generation.context is not None)
        )
        def physical_complete(pending: _Pending) -> bool:
            if pending.generation is not None:
                return (
                    pending.generation.context is None
                    or self.backend.prepared_complete(
                        generations=(pending.generation,)
                    )
                )
            if pending.swap is not None:
                return self.backend.prepared_complete(swaps=(pending.swap,))
            return True

        all_physical_complete = all(
            physical_complete(pending)
            for pending in physical_pending
        )
        def collect_due(event_time: int) -> list[_Pending]:
            return [
                pending for pending in pending_all
                if physical_complete(pending)
                and pending.logical_completion_time_ps <= event_time
            ]

        # A protocol can terminate before the logical operation duration.  In
        # that case wait only until the logical boundary; if the protocol is
        # still running, use its nominal physical window to make progress.
        event_time = self.backend.physical_time_ps
        due = collect_due(event_time)
        if not due and event_time < target_time:
            if all_physical_complete:
                future_logical = [
                    pending.logical_completion_time_ps
                    for pending in pending_all
                    if pending.logical_completion_time_ps > self.backend.physical_time_ps
                ]
                if future_logical:
                    logical_target = min(future_logical)
                    self.backend.advance_physical_to(
                        min(logical_target, target_time, self.horizon_ps),
                        synchronize=False,
                    )
            else:
                self.backend.advance_physical_to(target_time, synchronize=False)
            event_time = self.backend.physical_time_ps
            due = collect_due(event_time)
        events: list[ExecutionEvent] = []
        if not due and event_time >= self.horizon_ps:
            timed_out = tuple(sorted(self._pending.values(), key=lambda item: item.attempt_id))
            self._pending.clear()
            for pending in timed_out:
                events.append(self._finish_pending(pending, "horizon_timeout"))
            self._available_segment_ids()
            events.extend(self._drain_expiration_events())
            self._terminated = True
            return ExecutionEventBatch(
                physical_time_ps=event_time,
                events=tuple(sorted(events, key=lambda event: event.event_id)),
                duration_ps=max(0, event_time - previous),
                terminal=True,
            )
        for pending in sorted(due, key=lambda item: item.attempt_id):
            self._pending.pop(pending.operation.op_id, None)
            events.append(self._finish_pending(pending))
        if event_time >= self.horizon_ps and self._pending:
            timed_out = tuple(sorted(self._pending.values(), key=lambda item: item.attempt_id))
            self._pending.clear()
            for pending in timed_out:
                events.append(self._finish_pending(pending, "horizon_timeout"))
        self._available_segment_ids()
        events.extend(self._drain_expiration_events())
        if event_time >= self.horizon_ps:
            self._terminated = True
        batch = ExecutionEventBatch(
            physical_time_ps=event_time,
            events=tuple(sorted(events, key=lambda event: event.event_id)),
            duration_ps=max(0, event_time - previous),
            terminal=self._terminated and not self._pending,
        )
        if batch.physical_time_ps != self.physical_time_ps:
            raise RuntimeError("event batch and physical backend time diverged")
        return batch

    def repair(
        self,
        request_id: str,
        operations: tuple[ConstructionOperation, ...],
        *,
        supersede_uncommitted: bool = False,
    ) -> None:
        self._refresh_global_registry()
        if request_id not in self.dags:
            raise KeyError(request_id)
        if any(
            pending.operation.request_id == request_id
            for pending in self._pending.values()
        ):
            raise RuntimeError("cannot repair a request with in-flight operations")
        dag = self.dags[request_id]
        if supersede_uncommitted and dag.started:
            raise RuntimeError("cannot reroute a request with started operations")
        obsolete = tuple(
            operation.op_id
            for operation in dag.operations
            if operation.op_id not in dag.completed
            and operation.op_id not in dag.dead
        ) if supersede_uncommitted else ()
        available = self._available_segment_ids()
        outputs = {
            operation.output_segment_id: operation
            for operation in operations
            if operation.output_segment_id is not None
        }
        existing_outputs = set(self._segments)
        pending_outputs = {
            pending.operation.output_segment_id
            for pending in self._pending.values()
            if pending.operation.output_segment_id is not None
        }
        other_dag_outputs = {
            operation.output_segment_id
            for current_dag in self.dags.values()
            if current_dag.request_id != request_id
            for operation in current_dag.operations
            if operation.output_segment_id is not None
        }
        for output_id in outputs:
            if output_id in existing_outputs:
                raise ValueError(f"repair output segment id is already live: {output_id}")
            if output_id in pending_outputs:
                raise ValueError(f"repair output segment id is already in flight: {output_id}")
            if output_id in other_dag_outputs:
                raise ValueError(f"repair output segment id is owned by another DAG: {output_id}")
        for operation in operations:
            owner = self._operation_owners.get(operation.op_id)
            if owner is not None:
                raise ValueError(f"repair operation id already exists: {operation.op_id}")
            self._validate_output_hold(operation)
            if (
                operation.output_endpoints is not None
                and frozenset(operation.output_endpoints)
                == self._request_endpoints.get(request_id, frozenset())
                and operation.required_fidelity
                < self._request_required_fidelity.get(request_id, operation.required_fidelity)
            ):
                raise ValueError(
                    "repair terminal operation cannot lower request required_fidelity"
                )
            if any(predecessor in dag.dead for predecessor in operation.predecessors):
                raise ValueError("repair operation cannot depend on a dead predecessor")
            for segment_id in operation.input_segment_ids:
                if segment_id in available:
                    if self._segments[segment_id].request_id != request_id:
                        raise ValueError("repair cannot consume another request's segment")
                    continue
                producer = outputs.get(segment_id)
                if producer is None or producer.op_id not in operation.predecessors:
                    raise ValueError(
                        f"repair input segment is not surviving or newly produced: {segment_id}"
                    )
        dag.repair(operations)
        if obsolete:
            dag.mark_obsolete(obsolete)
        self._operation_owners.update({operation.op_id: request_id for operation in operations})

    def repair_options(
        self, request_id: str
    ) -> tuple[tuple[ConstructionOperation, ...], ...]:
        if request_id not in self.dags:
            raise KeyError(request_id)
        dag = self.dags[request_id]
        return generate_repair_options(
            dag,
            self._available_segment_ids(),
            next_version=dag.version + 1,
            ordinal_start=max(
                (operation.ordinal for operation in dag.operations), default=0
            ) + 1,
            required_fidelity_for=self._effective_required_fidelity,
        )

    def release_segment(self, segment_id: str) -> LogicalSegment | None:
        segment = self._segments.get(segment_id)
        if segment is None:
            return None
        if any(
            segment_id in pending.operation.input_segment_ids
            for pending in self._pending.values()
        ):
            raise RuntimeError(f"cannot release segment used by an in-flight operation: {segment_id}")
        self._segments.pop(segment_id, None)
        physical_id = self._physical_by_segment.pop(segment_id, None)
        if physical_id is not None:
            self.backend.discard_pair(physical_id)
        return segment

    def release_request(self, request_id: str) -> tuple[str, ...]:
        if request_id not in self.dags:
            raise KeyError(request_id)
        released: list[str] = []
        for segment_id, segment in tuple(self._segments.items()):
            if segment.request_id != request_id:
                continue
            if any(
                segment_id in pending.operation.input_segment_ids
                for pending in self._pending.values()
            ):
                continue
            self.release_segment(segment_id)
            released.append(segment_id)
        return tuple(sorted(released))

    def terminate(self) -> None:
        if self._pending:
            raise RuntimeError("cannot terminate while operations are in flight")
        self._terminated = True

    def wait_until(self, target_time_ps: int) -> ExecutionEventBatch:
        if self._pending:
            raise RuntimeError("wait_until cannot skip in-flight operations")
        if target_time_ps < self.physical_time_ps or target_time_ps > self.horizon_ps:
            raise ValueError("target time must lie in [current_time, horizon]")
        expiration_time = self._next_expiration_time_ps()
        target_time_ps = min(
            target_time_ps,
            expiration_time if expiration_time is not None else target_time_ps,
        )
        duration = target_time_ps - self.physical_time_ps
        self.backend.advance_physical_to(target_time_ps)
        self._available_segment_ids()
        expiration_events = self._drain_expiration_events()
        self._terminated = target_time_ps >= self.horizon_ps
        return ExecutionEventBatch(
            target_time_ps,
            expiration_events,
            duration,
            terminal=self._terminated,
        )
