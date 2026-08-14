"""Nominal resource--time expansion for construction-aware candidates.

The planning layer treats one construction operation as one synchronized
planning round.  A construction DAG is scheduled as early as its dependencies
allow, producing a relative schedule.  Each feasible start slot then shifts
that schedule into the batch horizon.

This module consumes only simulator-neutral catalogue objects and opaque
resource keys.  It never imports or mutates SeQUeNCe state.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Mapping, Sequence

from qnet_core.construction_api import ConstructionOperation
from qnet_core.construction_catalog import RouteConstructionCandidate
from qnet_core.planning_spec import PlanningSpec


@dataclass(frozen=True, order=True)
class ResourceSlotUsage:
    """Amount of one opaque resource occupied in one planning slot."""

    resource_id: str
    slot: int
    amount: int

    def __post_init__(self) -> None:
        if not self.resource_id:
            raise ValueError("resource_id must be non-empty")
        if self.slot < 0:
            raise ValueError("resource slot cannot be negative")
        if self.amount < 1:
            raise ValueError("resource usage amount must be positive")


@dataclass(frozen=True)
class NominalConstructionSchedule:
    """Earliest dependency-respecting schedule relative to slot zero."""

    candidate_id: str
    operation_slots: tuple[tuple[str, int], ...]
    duration_slots: int
    resource_usage: tuple[ResourceSlotUsage, ...]

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if self.duration_slots < 1:
            raise ValueError("duration_slots must be positive")
        if tuple(sorted(self.operation_slots)) != self.operation_slots:
            raise ValueError("operation_slots must be operation-id sorted")
        if tuple(sorted(self.resource_usage)) != self.resource_usage:
            raise ValueError("resource_usage must be canonical and sorted")
        if any(slot >= self.duration_slots for _, slot in self.operation_slots):
            raise ValueError("operation lies outside the nominal duration")
        if any(item.slot >= self.duration_slots for item in self.resource_usage):
            raise ValueError("resource usage lies outside the nominal duration")


@dataclass(frozen=True)
class TimeExpandedCandidate:
    """One binary choice: request, route, construction tree, and start slot."""

    variable_id: str
    base_candidate: RouteConstructionCandidate
    start_slot: int
    completion_slot: int
    completion_latency: int
    expected_fidelity: float | None
    resource_usage: tuple[ResourceSlotUsage, ...]
    nominal_schedule: NominalConstructionSchedule
    expected_success_probability: float = 1.0

    def __post_init__(self) -> None:
        if not self.variable_id:
            raise ValueError("variable_id must be non-empty")
        if self.start_slot < 0:
            raise ValueError("start_slot cannot be negative")
        if self.completion_slot <= self.start_slot:
            raise ValueError("completion_slot must follow start_slot")
        if self.completion_latency < 1:
            raise ValueError("completion_latency must be positive")
        if self.expected_fidelity is not None and not 0.0 <= self.expected_fidelity <= 1.0:
            raise ValueError("expected_fidelity must be in [0, 1]")
        if not 0.0 <= self.expected_success_probability <= 1.0:
            raise ValueError("expected_success_probability must be in [0, 1]")
        if tuple(sorted(self.resource_usage)) != self.resource_usage:
            raise ValueError("resource_usage must be canonical and sorted")

    @property
    def candidate_id(self) -> str:
        return self.base_candidate.candidate_id

    @property
    def request_id(self) -> str:
        return self.base_candidate.request_id

    @property
    def route_nodes(self) -> tuple[int, ...]:
        return self.base_candidate.route_nodes

    @property
    def construction_kind(self) -> str:
        return self.base_candidate.construction_kind

    @property
    def purification_kind(self) -> str:
        return self.base_candidate.purification_kind

    @property
    def duration_slots(self) -> int:
        return self.nominal_schedule.duration_slots


@dataclass(frozen=True, order=True)
class CandidateRejection:
    candidate_id: str
    reason: str


@dataclass(frozen=True)
class TimeExpansionResult:
    """Feasible time-expanded variables and deterministic rejection reasons."""

    variables: tuple[TimeExpandedCandidate, ...]
    schedules: tuple[NominalConstructionSchedule, ...]
    rejections: tuple[CandidateRejection, ...] = ()


def normalize_reserved_usage(
    reserved_usage: Mapping[tuple[str, int], int] | None,
    resource_capacities: Mapping[str, int],
) -> dict[tuple[str, int], int]:
    """Validate an existing resource--slot reservation catalogue.

    Online replanning keeps the ordinary ``resource_id -> capacity`` mapping
    unchanged and supplies already committed usage separately.  This helper
    is shared by expansion, MILP assembly, autoregressive feasibility masking,
    and physical-plan compilation so every layer applies the same residual-
    capacity semantics.
    """

    capacities = {str(key): int(value) for key, value in resource_capacities.items()}
    normalized: dict[tuple[str, int], int] = {}
    if reserved_usage is None:
        return normalized
    for raw_key, raw_amount in reserved_usage.items():
        if not isinstance(raw_key, tuple) or len(raw_key) != 2:
            raise ValueError("reserved usage keys must be (resource_id, slot) tuples")
        resource_id = str(raw_key[0])
        slot = int(raw_key[1])
        amount = int(raw_amount)
        if not resource_id:
            raise ValueError("reserved resource IDs must be non-empty")
        if slot < 0:
            raise ValueError("reserved resource slots cannot be negative")
        if amount < 0:
            raise ValueError("reserved resource usage cannot be negative")
        if resource_id not in capacities:
            raise ValueError(f"missing capacity for reserved resource: {resource_id}")
        if amount > capacities[resource_id]:
            raise ValueError(
                f"reserved usage exceeds capacity: {resource_id}@{slot}: "
                f"{amount}>{capacities[resource_id]}"
            )
        if amount:
            normalized[(resource_id, slot)] = amount
    return normalized


def _topological_operations(
    operations: Sequence[ConstructionOperation],
) -> tuple[tuple[ConstructionOperation, ...], dict[str, tuple[str, ...]]]:
    by_id = {operation.op_id: operation for operation in operations}
    if len(by_id) != len(operations):
        raise ValueError("construction candidate contains duplicate operation IDs")
    producer_by_segment = {
        operation.output_segment_id: operation.op_id
        for operation in operations
        if operation.output_segment_id is not None
    }
    dependencies: dict[str, tuple[str, ...]] = {}
    for operation in operations:
        required = list(operation.predecessors)
        for segment_id in operation.input_segment_ids:
            producer_id = producer_by_segment.get(segment_id)
            if producer_id is not None and producer_id not in required:
                required.append(producer_id)
        dependencies[operation.op_id] = tuple(required)
    successors: dict[str, list[str]] = {op_id: [] for op_id in by_id}
    indegree = {op_id: 0 for op_id in by_id}
    for operation in operations:
        for predecessor in dependencies[operation.op_id]:
            if predecessor not in by_id:
                raise ValueError(f"unknown predecessor: {predecessor}")
            successors[predecessor].append(operation.op_id)
            indegree[operation.op_id] += 1
    ready = [
        (by_id[op_id].canonical_key, op_id)
        for op_id, degree in indegree.items()
        if degree == 0
    ]
    heapq.heapify(ready)
    ordered: list[ConstructionOperation] = []
    while ready:
        _, op_id = heapq.heappop(ready)
        ordered.append(by_id[op_id])
        for successor in sorted(successors[op_id]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(
                    ready, (by_id[successor].canonical_key, successor)
                )
    if len(ordered) != len(operations):
        raise ValueError("construction candidate contains a dependency cycle")
    return tuple(ordered), dependencies


def _add_usage(
    usage: dict[tuple[str, int], int],
    resource_id: str,
    slot: int,
    amount: int,
) -> None:
    if amount <= 0:
        return
    key = (resource_id, slot)
    usage[key] = usage.get(key, 0) + int(amount)


def _serialize_shared_swap_nodes(
    ordered: Sequence[ConstructionOperation],
    dependencies: Mapping[str, set[str]],
) -> dict[str, int]:
    """Schedule swaps so one physical node handles at most one per round."""

    operation_slot: dict[str, int] = {}
    swap_nodes_by_slot: dict[int, set[int]] = {}
    for operation in ordered:
        earliest = (
            0
            if not dependencies[operation.op_id]
            else 1 + max(
                operation_slot[item] for item in dependencies[operation.op_id]
            )
        )
        if operation.kind != "SWAP":
            operation_slot[operation.op_id] = earliest
            continue
        nodes = {
            int(resource_id.removeprefix("swapnode:"))
            for resource_id, amount in operation.resource_demand.items()
            if amount > 0 and resource_id.startswith("swapnode:")
        }
        if not nodes:
            raise ValueError(
                f"SWAP operation does not declare physical node mutexes: "
                f"{operation.op_id}"
            )
        slot = earliest
        while nodes.intersection(swap_nodes_by_slot.get(slot, set())):
            slot += 1
        operation_slot[operation.op_id] = slot
        swap_nodes_by_slot.setdefault(slot, set()).update(nodes)
    return operation_slot


def build_nominal_schedule(
    candidate: RouteConstructionCandidate,
) -> NominalConstructionSchedule:
    """Compile a construction DAG into an earliest-start relative schedule.

    Every operation occupies one planning round.  Output resource holds begin
    in the following round and remain occupied through the round in which the
    unique consumer executes.  Terminal output resources are assumed to be
    settled and released when the request completes.
    """

    ordered, dependencies = _topological_operations(candidate.dag.operations)
    if not ordered:
        raise ValueError("construction candidate must contain operations")
    producer_by_segment: dict[str, ConstructionOperation] = {}
    consumers_by_segment: dict[str, list[ConstructionOperation]] = {}
    for operation in ordered:
        if operation.output_segment_id is not None:
            if operation.output_segment_id in producer_by_segment:
                raise ValueError(
                    f"segment has multiple producers: {operation.output_segment_id}"
                )
            producer_by_segment[operation.output_segment_id] = operation
        for segment_id in operation.input_segment_ids:
            consumers_by_segment.setdefault(segment_id, []).append(operation)

    operation_slot = _serialize_shared_swap_nodes(
        ordered,
        dependencies,
    )

    usage: dict[tuple[str, int], int] = {}
    for operation in ordered:
        slot = operation_slot[operation.op_id]
        for resource_id, amount in operation.resource_demand.items():
            _add_usage(usage, resource_id, slot, amount)

        output_segment = operation.output_segment_id
        if output_segment is None or not operation.output_resource_hold:
            continue
        consumers = consumers_by_segment.get(output_segment, [])
        if len(consumers) > 1:
            raise ValueError(
                f"segment has multiple consumers: {output_segment}"
            )
        if not consumers:
            continue
        consumer_slot = operation_slot[consumers[0].op_id]
        if consumer_slot <= slot:
            raise ValueError(
                f"segment consumer is not scheduled after producer: {output_segment}"
            )
        for held_slot in range(slot + 1, consumer_slot + 1):
            for resource_id, amount in operation.output_resource_hold.items():
                _add_usage(usage, resource_id, held_slot, amount)

    terminal_producers = []
    for segment_id in candidate.all_terminal_segment_ids:
        try:
            terminal_producers.append(producer_by_segment[segment_id])
        except KeyError as exc:
            raise ValueError(
                f"terminal segment has no producer: {segment_id}"
            ) from exc
    duration_slots = 1 + max(
        operation_slot[operation.op_id] for operation in terminal_producers
    )
    if any(slot >= duration_slots for slot in operation_slot.values()):
        raise ValueError("construction contains operations after terminal completion")

    resource_usage = tuple(
        ResourceSlotUsage(resource_id, slot, amount)
        for (resource_id, slot), amount in sorted(usage.items())
    )
    return NominalConstructionSchedule(
        candidate_id=candidate.candidate_id,
        operation_slots=tuple(sorted(operation_slot.items())),
        duration_slots=duration_slots,
        resource_usage=resource_usage,
    )


def expand_construction_candidates(
    spec: PlanningSpec,
    candidates: Sequence[RouteConstructionCandidate],
    resource_capacities: Mapping[str, int],
    *,
    fidelity_estimates: Mapping[str, float] | None = None,
    success_probability_estimates: Mapping[str, float] | None = None,
    reserved_usage: Mapping[tuple[str, int], int] | None = None,
    window_start_slot: int | None = None,
    window_end_slot: int | None = None,
    completion_end_slot: int | None = None,
) -> TimeExpansionResult:
    """Shift every nominal construction schedule across its feasible starts.

    Time-window and optional fidelity checks are performed before MILP assembly.
    Construction dependency feasibility is already encoded by the nominal DAG
    schedule.  Missing resource capacities are treated as modelling errors,
    while intrinsically over-capacity candidates are rejected deterministically.
    """

    capacities = {str(key): int(value) for key, value in resource_capacities.items()}
    if any(not key for key in capacities):
        raise ValueError("resource capacity keys must be non-empty")
    if any(value < 1 for value in capacities.values()):
        raise ValueError("resource capacities must be positive")
    reservations = normalize_reserved_usage(reserved_usage, capacities)
    if any(slot >= spec.horizon for _, slot in reservations):
        raise ValueError("reserved resource slot lies outside the planning horizon")
    if window_start_slot is None:
        first_window_slot = 0
    else:
        first_window_slot = int(window_start_slot)
    if window_end_slot is None:
        last_start_window_slot = spec.horizon
    else:
        last_start_window_slot = int(window_end_slot)
    if completion_end_slot is None:
        last_completion_window_slot = last_start_window_slot
    else:
        last_completion_window_slot = int(completion_end_slot)
    if not 0 <= first_window_slot < last_start_window_slot <= spec.horizon:
        raise ValueError("start window must lie inside the episode horizon")
    if not (
        last_start_window_slot
        <= last_completion_window_slot
        <= spec.horizon
    ):
        raise ValueError(
            "completion boundary must follow the start window and lie inside "
            "the episode horizon"
        )
    requests = {request.id: request for request in spec.requests}
    if len(requests) != len(spec.requests):
        raise ValueError("request IDs must be unique")
    unnecessary_purification_groups: set[
        tuple[str, tuple[int, ...], str]
    ] = set()
    if fidelity_estimates is not None:
        for candidate in candidates:
            request = requests.get(candidate.request_id)
            estimate = fidelity_estimates.get(candidate.candidate_id)
            if (
                request is not None
                and estimate is not None
                and candidate.purification_kind == "none"
                and float(estimate) >= request.required_fidelity
            ):
                unnecessary_purification_groups.add((
                    candidate.request_id,
                    candidate.route_nodes,
                    candidate.construction_kind,
                ))

    schedules: list[NominalConstructionSchedule] = []
    variables: list[TimeExpandedCandidate] = []
    rejections: list[CandidateRejection] = []
    seen_candidate_ids: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        if candidate.candidate_id in seen_candidate_ids:
            raise ValueError(f"duplicate candidate ID: {candidate.candidate_id}")
        seen_candidate_ids.add(candidate.candidate_id)
        if candidate.request_id not in requests:
            raise ValueError(
                f"candidate belongs to unknown request: {candidate.request_id}"
            )
        request = requests[candidate.request_id]
        schedule = build_nominal_schedule(candidate)
        schedules.append(schedule)

        expected_fidelity = None
        if fidelity_estimates is not None:
            if candidate.candidate_id not in fidelity_estimates:
                raise ValueError(
                    f"missing fidelity estimate: {candidate.candidate_id}"
                )
            expected_fidelity = float(fidelity_estimates[candidate.candidate_id])
            if not 0.0 <= expected_fidelity <= 1.0:
                raise ValueError("fidelity estimates must be in [0, 1]")
            if expected_fidelity < request.required_fidelity:
                rejections.append(
                    CandidateRejection(candidate.candidate_id, "fidelity")
                )
                continue
            if (
                candidate.purification_kind != "none"
                and (
                    candidate.request_id,
                    candidate.route_nodes,
                    candidate.construction_kind,
                ) in unnecessary_purification_groups
            ):
                rejections.append(CandidateRejection(
                    candidate.candidate_id,
                    "purification_unnecessary",
                ))
                continue

        expected_success_probability = 1.0
        if success_probability_estimates is not None:
            if candidate.candidate_id not in success_probability_estimates:
                raise ValueError(
                    f"missing success probability estimate: {candidate.candidate_id}"
                )
            expected_success_probability = float(
                success_probability_estimates[candidate.candidate_id]
            )
            if not 0.0 <= expected_success_probability <= 1.0:
                raise ValueError("success probability estimates must be in [0, 1]")

        intrinsic_failure: str | None = None
        for item in schedule.resource_usage:
            if item.resource_id not in capacities:
                raise ValueError(
                    f"missing capacity for resource: {item.resource_id}"
                )
            if item.amount > capacities[item.resource_id]:
                intrinsic_failure = f"capacity:{item.resource_id}"
                break
        if intrinsic_failure is not None:
            rejections.append(
                CandidateRejection(candidate.candidate_id, intrinsic_failure)
            )
            continue

        deadline = request.deadline
        last_completion = (
            last_completion_window_slot
            if deadline is None
            else min(last_completion_window_slot, deadline)
        )
        first_start = max(request.arrival, first_window_slot)
        last_start = min(
            last_start_window_slot - 1,
            last_completion - schedule.duration_slots,
        )
        if last_start < first_start:
            rejections.append(
                CandidateRejection(candidate.candidate_id, "time_window")
            )
            continue

        feasible_start_count = 0
        reservation_blocked = False
        for start_slot in range(first_start, last_start + 1):
            absolute_usage = tuple(
                ResourceSlotUsage(
                    item.resource_id,
                    start_slot + item.slot,
                    item.amount,
                )
                for item in schedule.resource_usage
            )
            if any(
                item.amount
                + reservations.get((item.resource_id, item.slot), 0)
                > capacities[item.resource_id]
                for item in absolute_usage
            ):
                reservation_blocked = True
                continue
            completion_slot = start_slot + schedule.duration_slots
            variables.append(TimeExpandedCandidate(
                variable_id=f"{candidate.candidate_id}@slot:{start_slot}",
                base_candidate=candidate,
                start_slot=start_slot,
                completion_slot=completion_slot,
                completion_latency=completion_slot - request.arrival,
                expected_fidelity=expected_fidelity,
                resource_usage=absolute_usage,
                nominal_schedule=schedule,
                expected_success_probability=expected_success_probability,
            ))
            feasible_start_count += 1
        if feasible_start_count == 0 and reservation_blocked:
            rejections.append(CandidateRejection(
                candidate.candidate_id,
                "reserved_capacity",
            ))

    return TimeExpansionResult(
        variables=tuple(sorted(variables, key=lambda item: item.variable_id)),
        schedules=tuple(sorted(schedules, key=lambda item: item.candidate_id)),
        rejections=tuple(sorted(rejections)),
    )
