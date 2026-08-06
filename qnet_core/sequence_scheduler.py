"""Resource-aware launch scheduler for the SeQUeNCe construction executor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .construction_api import ConstructionOperation, LogicalSegment, OperationKind


@dataclass(frozen=True)
class SchedulerValidation:
    feasible: bool
    reason: str = ""


class SequenceConcurrencyScheduler:
    """Validate and greedily pack construction operations at one epoch.

    The scheduler is deliberately simulator-neutral.  It reasons about the
    resource demands and segment holds declared by operations; the executor
    separately performs SeQUeNCe protocol preflight and launch.  Pending
    operations are included in every check, which makes inter-epoch launch
    explicit rather than an implicit backend flag.
    """

    def __init__(
        self,
        capacities: Mapping[str, int],
        *,
        supports_inter_epoch_launch: bool = True,
        supports_mixed_operation_concurrency: bool = True,
        supports_concurrent_swaps: bool = True,
    ):
        self.capacities = {str(key): int(value) for key, value in capacities.items()}
        if any(value < 0 for value in self.capacities.values()):
            raise ValueError("scheduler capacities must be non-negative")
        self.supports_inter_epoch_launch = bool(supports_inter_epoch_launch)
        self.supports_mixed_operation_concurrency = bool(
            supports_mixed_operation_concurrency
        )
        self.supports_concurrent_swaps = bool(supports_concurrent_swaps)

    @staticmethod
    def _add_usage(target: dict[str, int], values: Mapping[str, int]) -> None:
        for resource, amount in values.items():
            target[resource] = target.get(resource, 0) + int(amount)

    @classmethod
    def _segment_usage(
        cls, segments: Iterable[LogicalSegment]
    ) -> dict[str, int]:
        usage: dict[str, int] = {}
        for segment in segments:
            cls._add_usage(usage, segment.held_resources.as_dict())
        return usage

    @classmethod
    def _pending_usage(
        cls, pending_operations: Iterable[ConstructionOperation]
    ) -> dict[str, int]:
        usage: dict[str, int] = {}
        for operation in pending_operations:
            cls._add_usage(usage, operation.resource_demand.as_dict())
        return usage

    @staticmethod
    def _operation_nodes(
        operation: ConstructionOperation,
        segments: Mapping[str, LogicalSegment],
    ) -> frozenset[int]:
        """Return physical endpoint/middle nodes touched by an operation.

        A SWAP's ``output_endpoints`` omit its middle node, so its input
        segments are included as well.  This keeps the scheduler neutral
        while still preventing SeQUeNCe protocols from racing on a shared
        memory or BSM node.
        """

        nodes: set[int] = set(operation.output_endpoints or ())
        for segment_id in operation.input_segment_ids:
            segment = segments.get(segment_id)
            if segment is not None:
                nodes.update(segment.endpoints)
        return frozenset(nodes)

    def validate(
        self,
        operations: Sequence[ConstructionOperation],
        *,
        pending_operations: Sequence[ConstructionOperation] = (),
        segments: Sequence[LogicalSegment] = (),
    ) -> SchedulerValidation:
        operations = tuple(operations)
        pending_operations = tuple(pending_operations)
        if not operations:
            return SchedulerValidation(False, "launch requires at least one operation")
        if pending_operations and not self.supports_inter_epoch_launch:
            return SchedulerValidation(False, "operations are in flight")
        swaps = [operation for operation in operations if operation.kind == OperationKind.SWAP]
        generations = [operation for operation in operations if operation.kind == OperationKind.GEN]
        pending_swaps = [
            operation
            for operation in pending_operations
            if operation.kind == OperationKind.SWAP
        ]
        if (len(swaps) > 1 or (pending_swaps and swaps)) and not self.supports_concurrent_swaps:
            return SchedulerValidation(False, "concurrent swaps are disabled")
        if swaps and generations and not self.supports_mixed_operation_concurrency:
            return SchedulerValidation(False, "mixed generation/swap launch is disabled")

        segment_index = {segment.segment_id: segment for segment in segments}
        # SeQUeNCe's generation and swapping protocols mutate endpoint memory
        # state during the same timeline window.  Mixed launches remain valid
        # only when their physical node scopes are disjoint.
        for operation in operations:
            operation_nodes = self._operation_nodes(operation, segment_index)
            for pending in pending_operations:
                if (
                    operation.kind != pending.kind
                    and {operation.kind, pending.kind}
                    == {OperationKind.GEN, OperationKind.SWAP}
                    and not self.supports_mixed_operation_concurrency
                ):
                    return SchedulerValidation(
                        False,
                        "operations are in flight: mixed generation/swap launch is disabled",
                    )
                pending_nodes = self._operation_nodes(pending, segment_index)
                if (
                    operation_nodes.intersection(pending_nodes)
                    and operation.kind != pending.kind
                    and {operation.kind, pending.kind}
                    == {OperationKind.GEN, OperationKind.SWAP}
                ):
                    return SchedulerValidation(
                        False, "mixed generation/swap launch has shared physical node"
                    )
            if operation.kind == OperationKind.SWAP:
                for other in operations:
                    if other is operation or other.kind != OperationKind.SWAP:
                        continue
                    if operation_nodes.intersection(
                        self._operation_nodes(other, segment_index)
                    ):
                        return SchedulerValidation(
                            False, "concurrent swaps have shared physical node"
                        )
        if swaps and generations:
            for swap in swaps:
                swap_nodes = self._operation_nodes(swap, segment_index)
                for generation in generations:
                    if swap_nodes.intersection(
                        self._operation_nodes(generation, segment_index)
                    ):
                        return SchedulerValidation(
                            False, "mixed generation/swap launch has shared physical node"
                        )

        pending_inputs = {
            segment_id
            for operation in pending_operations
            for segment_id in operation.input_segment_ids
        }
        inputs = [
            segment_id
            for operation in operations
            for segment_id in operation.input_segment_ids
        ]
        if len(inputs) != len(set(inputs)):
            return SchedulerValidation(False, "input segment consumed twice")
        if pending_inputs.intersection(inputs):
            return SchedulerValidation(False, "input segment is already in flight")

        base_usage = self._segment_usage(segments)
        self._add_usage(base_usage, self._pending_usage(pending_operations))
        launch_usage = dict(base_usage)
        for operation in operations:
            self._add_usage(launch_usage, operation.resource_demand.as_dict())
        for resource, amount in launch_usage.items():
            if amount > self.capacities.get(resource, 0):
                return SchedulerValidation(False, f"capacity exceeded: {resource}")

        consumed = set(inputs)
        pending_consumed = {
            segment_id
            for operation in pending_operations
            for segment_id in operation.input_segment_ids
        }
        completion_consumed = consumed | pending_consumed
        completion_usage = self._segment_usage(
            segment
            for segment in segments
            if segment.segment_id not in completion_consumed
        )
        for operation in (*pending_operations, *operations):
            self._add_usage(completion_usage, operation.output_resource_hold.as_dict())
        for resource, amount in completion_usage.items():
            if amount > self.capacities.get(resource, 0):
                return SchedulerValidation(
                    False, f"post-completion capacity exceeded: {resource}"
                )
        return SchedulerValidation(True)

    def pack(
        self,
        ready_operations: Sequence[ConstructionOperation],
        *,
        pending_operations: Sequence[ConstructionOperation] = (),
        segments: Sequence[LogicalSegment] = (),
    ) -> tuple[ConstructionOperation, ...]:
        """Return a deterministic maximal feasible subset of ready operations."""
        chosen: list[ConstructionOperation] = []
        for operation in sorted(
            ready_operations, key=lambda item: item.canonical_key
        ):
            trial = tuple(chosen) + (operation,)
            if self.validate(
                trial,
                pending_operations=pending_operations,
                segments=segments,
            ).feasible:
                chosen.append(operation)
        return tuple(chosen)


__all__ = ["SchedulerValidation", "SequenceConcurrencyScheduler"]
