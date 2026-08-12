"""Simulator-neutral contracts for event-driven entanglement construction.

The planning layer depends on these value objects only.  A physical adapter
may use SeQUeNCe (or another simulator) internally, but no simulator object is
allowed to cross this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Protocol, Sequence


class OperationKind:
    """Stable operation names shared by planners and physical adapters."""

    GEN = "GEN"
    PURIFY = "PURIFY"
    SWAP = "SWAP"
    RELEASE = "RELEASE"


class RepairKind:
    """Stable repair action names visible to planners."""

    RETRY = "RETRY"
    REROUTE = "REROUTE"


class ConstructionLaunchRejected(ValueError):
    """Expected rejection of a submitted launch set by executor validation."""


@dataclass(frozen=True)
class ResourceDemand:
    """A canonical, additive resource-demand vector.

    Keys are opaque strings such as ``memory:node:1`` or ``bsm:1``.  Keeping
    them opaque is what lets the planner remain independent from the physical
    simulator's concrete resource classes.
    """

    entries: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        normalized = tuple(sorted((str(key), int(value)) for key, value in self.entries))
        if len({key for key, _ in normalized}) != len(normalized):
            raise ValueError("resource demand contains duplicate keys")
        if any(not key for key, _ in normalized):
            raise ValueError("resource demand keys must be non-empty")
        if any(value < 0 for _, value in normalized):
            raise ValueError("resource demand values must be non-negative")
        object.__setattr__(self, "entries", normalized)

    @classmethod
    def from_mapping(cls, values: Mapping[str, int]) -> "ResourceDemand":
        return cls(tuple(values.items()))

    def get(self, key: str, default: int = 0) -> int:
        for current, value in self.entries:
            if current == key:
                return value
        return default

    def items(self) -> tuple[tuple[str, int], ...]:
        return self.entries

    def as_dict(self) -> dict[str, int]:
        return dict(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)


@dataclass(frozen=True)
class LogicalSegment:
    """A simulator-independent view of one currently usable entangled segment."""

    segment_id: str
    request_id: str
    left: int
    right: int
    born_time_ps: int
    fidelity: float = 1.0
    owner_request: str | None = None
    source_operation_id: str | None = None
    held_resources: ResourceDemand = field(default_factory=ResourceDemand)

    def __post_init__(self) -> None:
        if not self.segment_id or not self.request_id:
            raise ValueError("segment_id and request_id must be non-empty")
        if self.left == self.right:
            raise ValueError("a logical segment must connect distinct nodes")
        if self.born_time_ps < 0:
            raise ValueError("born_time_ps cannot be negative")
        if not 0.0 <= self.fidelity <= 1.0:
            raise ValueError("fidelity must be in [0, 1]")

    @property
    def endpoints(self) -> tuple[int, int]:
        return self.left, self.right


@dataclass(frozen=True)
class ConstructionOperation:
    """One node in a request construction DAG."""

    op_id: str
    request_id: str
    kind: str
    predecessors: tuple[str, ...] = ()
    input_segment_ids: tuple[str, ...] = ()
    output_segment_id: str | None = None
    output_endpoints: tuple[int, int] | None = None
    resource_demand: ResourceDemand = field(default_factory=ResourceDemand)
    output_resource_hold: ResourceDemand = field(default_factory=ResourceDemand)
    duration_ps: int = 1
    success_probability: float = 1.0
    required_fidelity: float = 0.0
    retry_limit: int = 0
    retry_root_id: str | None = None
    retry_attempt: int = 0
    ordinal: int = 0
    dag_version: int = 0

    def __post_init__(self) -> None:
        if not self.op_id or not self.request_id:
            raise ValueError("op_id and request_id must be non-empty")
        if self.kind not in {
            OperationKind.GEN,
            OperationKind.PURIFY,
            OperationKind.SWAP,
            OperationKind.RELEASE,
        }:
            raise ValueError(f"unsupported operation kind: {self.kind}")
        if len(set(self.predecessors)) != len(self.predecessors):
            raise ValueError("operation predecessors must be unique")
        if len(set(self.input_segment_ids)) != len(self.input_segment_ids):
            raise ValueError("operation inputs must be unique")
        if self.duration_ps < 1:
            raise ValueError("operation duration_ps must be positive")
        if not 0.0 <= self.success_probability <= 1.0:
            raise ValueError("success_probability must be in [0, 1]")
        if not 0.0 <= self.required_fidelity <= 1.0:
            raise ValueError("required_fidelity must be in [0, 1]")
        if (
            self.retry_limit < 0
            or self.retry_attempt < 0
            or self.ordinal < 0
            or self.dag_version < 0
        ):
            raise ValueError(
                "retry_limit, retry_attempt, ordinal, and dag_version must be non-negative"
            )
        if self.retry_attempt > self.retry_limit:
            raise ValueError("retry_attempt cannot exceed retry_limit")
        if self.retry_attempt > 0 and not self.retry_root_id:
            raise ValueError("retry operations must declare retry_root_id")
        if self.output_endpoints is not None:
            left, right = self.output_endpoints
            if left == right:
                raise ValueError("output segment endpoints must be distinct")
        if self.kind == OperationKind.GEN and self.output_segment_id is None:
            raise ValueError("GEN must declare an output segment")
        if self.kind == OperationKind.PURIFY:
            if len(self.input_segment_ids) != 2:
                raise ValueError("PURIFY must consume exactly two input segments")
            if self.output_segment_id is None:
                raise ValueError("PURIFY must declare an output segment")
        if self.kind == OperationKind.SWAP and self.output_segment_id is None:
            raise ValueError("SWAP must declare an output segment")
        if self.kind == OperationKind.RELEASE and self.output_segment_id is not None:
            raise ValueError("RELEASE cannot create an output segment")

    @property
    def canonical_key(self) -> tuple[object, ...]:
        """Injective structural key used by the canonical set decoder."""

        return (
            self.request_id,
            self.dag_version,
            self.ordinal,
            self.kind,
            self.op_id,
        )


@dataclass(frozen=True)
class ConstructionRepairChoice:
    """One planner-visible repair branch.

    DROP remains the implicit categorical action at index zero.  Every value
    represented here appends a fresh operation suffix to the immutable DAG
    prefix; a REROUTE choice also carries the replacement catalogue metadata
    and terminal segment IDs required by request settlement.
    """

    choice_id: str
    request_id: str
    kind: str
    operations: tuple[ConstructionOperation, ...]
    candidate_id: str | None = None
    route_nodes: tuple[int, ...] = ()
    construction_kind: str = ""
    terminal_segment_ids: tuple[str, ...] = ()
    purification_kind: str = "none"

    def __post_init__(self) -> None:
        if not self.choice_id or not self.request_id:
            raise ValueError("repair choice and request IDs must be non-empty")
        if self.kind not in {RepairKind.RETRY, RepairKind.REROUTE}:
            raise ValueError(f"unsupported repair kind: {self.kind}")
        if not self.operations:
            raise ValueError("repair choice requires at least one operation")
        if any(operation.request_id != self.request_id for operation in self.operations):
            raise ValueError("repair choice operations belong to another request")
        if len({operation.op_id for operation in self.operations}) != len(self.operations):
            raise ValueError("repair choice operation IDs must be unique")
        if len(set(self.terminal_segment_ids)) != len(self.terminal_segment_ids):
            raise ValueError("repair terminal segment IDs must be unique")
        if self.kind == RepairKind.REROUTE:
            if not self.candidate_id or len(self.route_nodes) < 2:
                raise ValueError("reroute choice requires candidate and route metadata")
            if not self.construction_kind or not self.terminal_segment_ids:
                raise ValueError("reroute choice requires construction and terminal metadata")
            if not self.purification_kind:
                raise ValueError("reroute choice requires purification metadata")
            produced = {
                operation.output_segment_id
                for operation in self.operations
                if operation.output_segment_id is not None
            }
            if not set(self.terminal_segment_ids).issubset(produced):
                raise ValueError("reroute terminal segments must be produced by the repair suffix")


@dataclass(frozen=True)
class DAGState:
    """Immutable execution view of one construction DAG."""

    request_id: str
    version: int
    operation_ids: tuple[str, ...]
    completed: tuple[str, ...] = ()
    started: tuple[str, ...] = ()
    dead: tuple[str, ...] = ()
    committed_prefix: tuple[str, ...] = ()


@dataclass(frozen=True)
class InFlightOperation:
    operation_id: str
    request_id: str
    attempt_id: str
    start_time_ps: int
    completion_time_ps: int
    reserved_resources: tuple[tuple[str, int], ...]
    input_segment_ids: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionEvent:
    """Physical outcome translated into a neutral event record."""

    event_id: str
    operation_id: str
    request_id: str
    attempt_id: str
    event_kind: str
    physical_time_ps: int
    success: bool
    failure_cause: str = ""
    consumed_segment_ids: tuple[str, ...] = ()
    surviving_segment_ids: tuple[str, ...] = ()
    output_segment_id: str | None = None
    output_fidelity: float | None = None
    released_resources: tuple[tuple[str, int], ...] = ()
    in_flight_operation_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionEventBatch:
    physical_time_ps: int
    events: tuple[ExecutionEvent, ...]
    duration_ps: int
    terminal: bool = False

    def __post_init__(self) -> None:
        if self.physical_time_ps < 0 or self.duration_ps < 0:
            raise ValueError("event time and duration must be non-negative")
        if tuple(sorted(self.events, key=lambda event: event.event_id)) != self.events:
            raise ValueError("events must be sorted by event_id")


@dataclass(frozen=True)
class ConstructionSnapshot:
    """Planner-visible neutral event-process snapshot at one decision epoch.

    ``reservations`` includes both in-flight operation demands and resources
    held by resident logical segments.  ``backend_state`` contains immutable,
    simulator-neutral summaries only.  A backend must separately establish
    whether those summaries are Markov-sufficient; this DTO does not assert it.
    """

    physical_time_ps: int
    horizon_ps: int
    dag_states: tuple[DAGState, ...] = ()
    operations: tuple[ConstructionOperation, ...] = ()
    segments: tuple[LogicalSegment, ...] = ()
    reservations: tuple[tuple[str, int], ...] = ()
    in_flight: tuple[InFlightOperation, ...] = ()
    pending_events: tuple[tuple[str, int, str], ...] = ()
    arrivals: tuple[tuple[str, int], ...] = ()
    deadlines: tuple[tuple[str, int], ...] = ()
    settled_request_ids: tuple[str, ...] = ()
    resource_capacities: tuple[tuple[str, int], ...] = ()
    backend_state: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if self.physical_time_ps < 0 or self.horizon_ps < self.physical_time_ps:
            raise ValueError("snapshot time must lie within the horizon")
        if tuple(sorted(self.dag_states, key=lambda state: state.request_id)) != self.dag_states:
            raise ValueError("dag_states must be request-id sorted")
        if tuple(sorted(self.operations, key=lambda item: item.canonical_key)) != self.operations:
            raise ValueError("operations must be canonical-key sorted")
        if len({operation.op_id for operation in self.operations}) != len(self.operations):
            raise ValueError("snapshot operations must have unique IDs")
        if tuple(sorted(set(self.settled_request_ids))) != self.settled_request_ids:
            raise ValueError("settled_request_ids must be unique and sorted")
        if tuple(sorted(self.segments, key=lambda segment: segment.segment_id)) != self.segments:
            raise ValueError("segments must be segment-id sorted")
        if tuple(sorted(self.in_flight, key=lambda item: item.operation_id)) != self.in_flight:
            raise ValueError("in_flight must be operation-id sorted")


class ConstructionExecutor(Protocol):
    """Neutral event-executor interface consumed by planning code."""

    def snapshot(self) -> ConstructionSnapshot: ...

    def register_dag(self, dag: ConstructionDAG) -> None: ...

    def unregister_dag(self, request_id: str) -> None: ...

    def ready_operations(
        self,
        allowed_operation_ids: Iterable[str] | None = None,
    ) -> Sequence[ConstructionOperation]: ...

    def launch(self, feasible_set: Iterable[ConstructionOperation]) -> tuple[str, ...]: ...

    def advance_to_next_event(
        self, boundary_ps: int | None = None
    ) -> ExecutionEventBatch: ...

    def repair(
        self,
        request_id: str,
        operations: tuple[ConstructionOperation, ...],
        *,
        supersede_uncommitted: bool = False,
    ) -> None: ...

    def repair_options(
        self, request_id: str
    ) -> Sequence[tuple[ConstructionOperation, ...]]: ...

    def release_segment(self, segment_id: str) -> LogicalSegment | None: ...

    def release_request(self, request_id: str) -> tuple[str, ...]: ...

    def terminate(self) -> None: ...

    @property
    def terminated(self) -> bool: ...

    @property
    def has_in_flight(self) -> bool: ...

    def wait_until(self, target_time_ps: int) -> ExecutionEventBatch: ...

    def next_expiration_time_ps(self) -> int | None: ...


class ConstructionDAG:
    """Mutable execution state for a single request's construction DAG."""

    def __init__(self, request_id: str, operations: tuple[ConstructionOperation, ...] = (), version: int = 0):
        if not request_id:
            raise ValueError("request_id must be non-empty")
        self.request_id = request_id
        self.version = int(version)
        if self.version < 0:
            raise ValueError("version must be non-negative")
        self._operations: dict[str, ConstructionOperation] = {}
        self.completed: set[str] = set()
        self.started: set[str] = set()
        self.dead: set[str] = set()
        self._add_many(operations)

    def clone(self) -> "ConstructionDAG":
        """Return a pristine executable copy with the same operation graph.

        The operation graph is the immutable catalogue definition, while
        ``started``, ``completed`` and ``dead`` are executor-owned state.  A
        fresh DAG is therefore preferable to ``copy.deepcopy`` at execution
        boundaries: it preserves the definition and deliberately resets the
        mutable execution markers.
        """

        return ConstructionDAG(
            self.request_id,
            self.operations,
            version=self.version,
        )

    def _add_many(self, operations: tuple[ConstructionOperation, ...]) -> None:
        staged = dict(self._operations)
        output_ids = {
            operation.output_segment_id
            for operation in staged.values()
            if operation.output_segment_id is not None
        }
        for operation in operations:
            if operation.request_id != self.request_id:
                raise ValueError("operation belongs to another request")
            if operation.op_id in staged:
                raise ValueError(f"duplicate operation: {operation.op_id}")
            if (
                operation.output_segment_id is not None
                and operation.output_segment_id in output_ids
            ):
                raise ValueError(
                    f"duplicate output segment: {operation.output_segment_id}"
                )
            staged[operation.op_id] = operation
            if operation.output_segment_id is not None:
                output_ids.add(operation.output_segment_id)
        self._assert_acyclic(staged)
        self._operations = staged

    def _assert_acyclic(
        self,
        operations: Mapping[str, ConstructionOperation] | None = None,
    ) -> None:
        operations = self._operations if operations is None else operations
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(op_id: str) -> None:
            if op_id in visiting:
                raise ValueError("construction DAG contains a cycle")
            if op_id in visited:
                return
            visiting.add(op_id)
            for predecessor in operations[op_id].predecessors:
                if predecessor not in operations:
                    raise ValueError(f"unknown predecessor: {predecessor}")
                visit(predecessor)
            visiting.remove(op_id)
            visited.add(op_id)

        for op_id in operations:
            visit(op_id)

    @property
    def operations(self) -> tuple[ConstructionOperation, ...]:
        return tuple(sorted(self._operations.values(), key=lambda operation: operation.canonical_key))

    def operation(self, op_id: str) -> ConstructionOperation:
        try:
            return self._operations[op_id]
        except KeyError as exc:
            raise KeyError(f"unknown operation: {op_id}") from exc

    def add_operation(self, operation: ConstructionOperation, increment_version: bool = True) -> None:
        self._add_many((operation,))
        if increment_version:
            self.version += 1

    def mark_started(self, op_id: str, available_segment_ids: set[str] | None = None) -> None:
        self._check_live(op_id)
        if available_segment_ids is None:
            available_segment_ids = set()
        if op_id not in self.ready_ids(available_segment_ids):
            raise ValueError(f"operation is not ready: {op_id}")
        self.started.add(op_id)

    def mark_completed(self, op_id: str) -> None:
        self._check_live(op_id)
        if op_id not in self.started:
            raise ValueError(f"operation was not started: {op_id}")
        self.completed.add(op_id)
        self.started.discard(op_id)

    def rollback_started(self, op_id: str) -> None:
        """Return an uncommitted launch to the ready state.

        Executors use this only while rolling back a failed atomic launch.
        A completed or dead operation can never be reopened through this
        method.
        """

        self._check_live(op_id)
        if op_id not in self.started:
            raise ValueError(f"operation was not started: {op_id}")
        self.started.remove(op_id)

    def mark_dead(self, op_id: str) -> None:
        self._check_live(op_id)
        self.dead.add(op_id)
        self.started.discard(op_id)

    def mark_obsolete(self, operation_ids: Iterable[str]) -> None:
        """Retire uncommitted operations after an atomic reroute.

        Completed operations are the immutable prefix and started operations
        are owned by the physical executor, so neither may be superseded.
        """

        targets = tuple(operation_ids)
        unknown = [op_id for op_id in targets if op_id not in self._operations]
        if unknown:
            raise KeyError(f"unknown operation: {unknown[0]}")
        if any(op_id in self.completed for op_id in targets):
            raise ValueError("completed operation cannot be superseded")
        if any(op_id in self.started for op_id in targets):
            raise RuntimeError("started operation cannot be superseded")
        self.dead.update(targets)

    def _check_live(self, op_id: str) -> None:
        if op_id not in self._operations:
            raise KeyError(f"unknown operation: {op_id}")
        if op_id in self.completed or op_id in self.dead:
            raise ValueError(f"operation already terminal: {op_id}")

    def ready_ids(self, available_segment_ids: set[str], reserved_ids: set[str] | None = None) -> tuple[str, ...]:
        reserved_ids = reserved_ids or set()
        ready: list[ConstructionOperation] = []
        for operation in self.operations:
            if operation.op_id in self.completed or operation.op_id in self.started or operation.op_id in self.dead:
                continue
            if operation.op_id in reserved_ids:
                continue
            if not set(operation.predecessors).issubset(self.completed):
                continue
            if not set(operation.input_segment_ids).issubset(available_segment_ids):
                continue
            ready.append(operation)
        return tuple(operation.op_id for operation in ready)

    @property
    def committed_prefix(self) -> tuple[str, ...]:
        return tuple(sorted(self.completed | self.started))

    def state(self) -> DAGState:
        return DAGState(
            request_id=self.request_id,
            version=self.version,
            operation_ids=tuple(operation.op_id for operation in self.operations),
            completed=tuple(sorted(self.completed)),
            started=tuple(sorted(self.started)),
            dead=tuple(sorted(self.dead)),
            committed_prefix=self.committed_prefix,
        )

    def repair(self, operations: tuple[ConstructionOperation, ...]) -> None:
        """Append a new DAG version after a failed branch.

        Existing completed/started operations remain immutable.  New
        operations must depend only on existing operations or surviving
        segments, and are tagged with the incremented DAG version.
        """

        if not operations:
            raise ValueError("repair requires at least one operation")
        next_version = self.version + 1
        for operation in operations:
            if operation.dag_version != next_version:
                raise ValueError("repair operations must carry the new DAG version")
        self._add_many(operations)
        self.version = next_version
