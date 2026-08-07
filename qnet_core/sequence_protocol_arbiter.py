"""Backend-owned arbitration for SeQUeNCe construction protocols.

The resource scheduler checks capacity and logical segment holds.  This
arbiter owns the separate protocol-family contract: which GEN/SWAP launches
may coexist on the SeQUeNCe timeline, which physical node scopes may overlap,
and whether an input pair is already claimed by an in-flight protocol.
Requests are neutral value objects; no SeQUeNCe protocol instance crosses this
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .construction_api import ConstructionOperation, LogicalSegment, OperationKind


@dataclass(frozen=True)
class ProtocolRequest:
    """Neutral physical scope required by one GEN or SWAP operation."""

    operation_id: str
    kind: str
    physical_nodes: frozenset[int]
    input_segment_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.operation_id:
            raise ValueError("protocol operation_id must be non-empty")
        if self.kind not in {OperationKind.GEN, OperationKind.SWAP}:
            raise ValueError("protocol requests only support GEN and SWAP")
        if not self.physical_nodes:
            raise ValueError("protocol physical_nodes must be non-empty")
        if len(set(self.input_segment_ids)) != len(self.input_segment_ids):
            raise ValueError("protocol input segment IDs must be unique")

    @classmethod
    def from_operation(
        cls,
        operation: ConstructionOperation,
        segments: Mapping[str, LogicalSegment],
    ) -> "ProtocolRequest | None":
        """Build a request from a neutral operation and current segments."""

        if operation.kind not in {OperationKind.GEN, OperationKind.SWAP}:
            return None
        nodes = set(operation.output_endpoints or ())
        for segment_id in operation.input_segment_ids:
            segment = segments.get(segment_id)
            if segment is not None:
                nodes.update(segment.endpoints)
        return cls(
            operation_id=operation.op_id,
            kind=operation.kind,
            physical_nodes=frozenset(nodes),
            input_segment_ids=tuple(operation.input_segment_ids),
        )


@dataclass(frozen=True)
class ArbiterValidation:
    feasible: bool
    reason: str = ""


class SequenceProtocolArbiter:
    """Validate protocol coexistence against explicit backend capabilities.

    The arbiter is intentionally conservative.  A backend must opt in to
    mixed GEN/SWAP launches and concurrent SWAPs; shared SWAP nodes remain
    rejected even when concurrent SWAP support is enabled.  This policy is
    independent of resource capacity and is checked before any physical
    protocol is started.
    """

    def __init__(
        self,
        *,
        supports_inter_epoch_launch: bool,
        supports_mixed_operation_concurrency: bool,
        supports_concurrent_swaps: bool,
    ):
        self.supports_inter_epoch_launch = bool(supports_inter_epoch_launch)
        self.supports_mixed_operation_concurrency = bool(
            supports_mixed_operation_concurrency
        )
        self.supports_concurrent_swaps = bool(supports_concurrent_swaps)

    def state(self) -> tuple[tuple[str, object], ...]:
        """Return a neutral, serializable capability declaration."""

        return (
            ("supports_inter_epoch_launch", self.supports_inter_epoch_launch),
            (
                "supports_mixed_operation_concurrency",
                self.supports_mixed_operation_concurrency,
            ),
            ("supports_concurrent_swaps", self.supports_concurrent_swaps),
        )

    @staticmethod
    def _normalize(
        requests: Iterable[ProtocolRequest],
    ) -> tuple[ProtocolRequest, ...]:
        normalized = tuple(requests)
        ids = [request.operation_id for request in normalized]
        if len(ids) != len(set(ids)):
            raise ValueError("protocol operation IDs must be unique")
        return normalized

    @staticmethod
    def _mixed(
        left: Sequence[ProtocolRequest],
        right: Sequence[ProtocolRequest],
    ) -> bool:
        kinds = {request.kind for request in (*left, *right)}
        return kinds == {OperationKind.GEN, OperationKind.SWAP}

    @staticmethod
    def _swap_node_conflict(
        left: Sequence[ProtocolRequest],
        right: Sequence[ProtocolRequest],
    ) -> bool:
        swaps_left = [request for request in left if request.kind == OperationKind.SWAP]
        swaps_right = [request for request in right if request.kind == OperationKind.SWAP]
        for first_index, first in enumerate(swaps_left):
            for second in swaps_left[first_index + 1:]:
                if first.physical_nodes.intersection(second.physical_nodes):
                    return True
        for first in swaps_left:
            for second in swaps_right:
                if first.physical_nodes.intersection(second.physical_nodes):
                    return True
        for first_index, first in enumerate(swaps_right):
            for second in swaps_right[first_index + 1:]:
                if first.physical_nodes.intersection(second.physical_nodes):
                    return True
        return False

    @staticmethod
    def _mixed_node_conflict(
        left: Sequence[ProtocolRequest],
        right: Sequence[ProtocolRequest],
    ) -> bool:
        for first in (*left, *right):
            for second in (*left, *right):
                if first is second or first.kind == second.kind:
                    continue
                if first.physical_nodes.intersection(second.physical_nodes):
                    return True
        return False

    def validate(
        self,
        requests: Iterable[ProtocolRequest],
        *,
        active: Iterable[ProtocolRequest] = (),
    ) -> ArbiterValidation:
        """Check a prospective launch against active protocol leases."""

        incoming = self._normalize(requests)
        active_requests = self._normalize(active)
        if not incoming:
            return ArbiterValidation(True)
        active_ids = {request.operation_id for request in active_requests}
        if active_ids.intersection(request.operation_id for request in incoming):
            return ArbiterValidation(False, "protocol operation is already active")
        incoming_inputs = [
            segment_id
            for request in incoming
            for segment_id in request.input_segment_ids
        ]
        active_inputs = {
            segment_id
            for request in active_requests
            for segment_id in request.input_segment_ids
        }
        if len(incoming_inputs) != len(set(incoming_inputs)):
            return ArbiterValidation(False, "input segment consumed twice")
        if active_inputs.intersection(incoming_inputs):
            return ArbiterValidation(False, "input segment is already in flight")
        if active_requests and not self.supports_inter_epoch_launch:
            return ArbiterValidation(False, "operations are in flight")
        if self._mixed(incoming, ()) and not self.supports_mixed_operation_concurrency:
            return ArbiterValidation(False, "mixed generation/swap launch is disabled")
        if self._mixed(active_requests, incoming) and not self.supports_mixed_operation_concurrency:
            return ArbiterValidation(
                False,
                "operations are in flight: mixed generation/swap launch is disabled",
            )
        incoming_swaps = sum(
            request.kind == OperationKind.SWAP for request in incoming
        )
        active_swaps = sum(
            request.kind == OperationKind.SWAP for request in active_requests
        )
        if (
            (incoming_swaps > 1 or (active_swaps and incoming_swaps))
            and not self.supports_concurrent_swaps
        ):
            return ArbiterValidation(False, "concurrent swaps are disabled")
        if self._swap_node_conflict(incoming, ()):
            return ArbiterValidation(False, "concurrent swaps have shared physical node")
        if self._swap_node_conflict(active_requests, incoming):
            return ArbiterValidation(
                False,
                "concurrent swaps have shared physical node",
            )
        if self._mixed_node_conflict(incoming, ()):
            return ArbiterValidation(
                False,
                "mixed generation/swap launch has shared physical node",
            )
        if self._mixed_node_conflict(active_requests, incoming):
            return ArbiterValidation(
                False,
                "mixed generation/swap launch has shared physical node",
            )
        return ArbiterValidation(True)


__all__ = [
    "ArbiterValidation",
    "ProtocolRequest",
    "SequenceProtocolArbiter",
]
