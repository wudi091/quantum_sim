"""Conservative SeQUeNCe-BDS fidelity bounds exposed as neutral numbers.

This module does not execute a test instance or inspect future stochastic
outcomes.  It mirrors the configured Bell-diagonal/Werner transformations at
the physical boundary and deliberately charges a full coarse slot of storage
between producer and consumer rounds.  The resulting value is a conservative
candidate-feasibility input for planning, not a replacement for final
SeQUeNCe evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from .construction_api import ConstructionDAG, OperationKind
from .spec import PhysicalConfig


FIDELITY_MODEL_NAME = "sequence_bds_conservative_v2_bbpssw"


@dataclass(frozen=True)
class ConstructionFidelityBound:
    model_name: str
    lower_bound: float
    terminal_bounds: tuple[tuple[str, float], ...]
    expired_segment_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("fidelity model name must be non-empty")
        if not 0.0 <= self.lower_bound <= 1.0:
            raise ValueError("fidelity lower bound must lie in [0, 1]")
        if tuple(sorted(self.terminal_bounds)) != self.terminal_bounds:
            raise ValueError("terminal fidelity bounds must be sorted")
        if any(not 0.0 <= value <= 1.0 for _, value in self.terminal_bounds):
            raise ValueError("terminal fidelity bounds must lie in [0, 1]")
        if tuple(sorted(set(self.expired_segment_ids))) != self.expired_segment_ids:
            raise ValueError("expired segment IDs must be unique and sorted")


@dataclass(frozen=True)
class _SegmentBound:
    fidelity: float
    produced_slot: int
    alive: bool = True


def werner_swap_fidelity(
    left_fidelity: float,
    right_fidelity: float,
    gate_fidelity: float,
) -> float:
    """Match SeQUeNCe's twirled BDS swap with measurement fidelity one."""

    for name, value in (
        ("left_fidelity", left_fidelity),
        ("right_fidelity", right_fidelity),
        ("gate_fidelity", gate_fidelity),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    correct_bell_weight = (
        left_fidelity * right_fidelity
        + (1.0 - left_fidelity) * (1.0 - right_fidelity) / 3.0
    )
    result = (
        gate_fidelity * correct_bell_weight
        + (1.0 - gate_fidelity) / 4.0
    )
    return min(1.0, max(0.0, float(result)))


def werner_bbpssw_result(
    kept_fidelity: float,
    measured_fidelity: float,
    gate_fidelity: float,
) -> tuple[float, float]:
    """Match SeQUeNCe's twirled BDS BBPSSW result.

    Returns ``(success_probability, conditional_output_fidelity)``.  The
    configured routers use unit measurement fidelity and the same gate
    fidelity at both endpoints, which reduces SeQUeNCe's BDS equations to the
    expression below.
    """

    for name, value in (
        ("kept_fidelity", kept_fidelity),
        ("measured_fidelity", measured_fidelity),
        ("gate_fidelity", gate_fidelity),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    if kept_fidelity <= 0.5 or measured_fidelity <= 0.5:
        return 0.0, 0.0

    kept_error = (1.0 - kept_fidelity) / 3.0
    measured_error = (1.0 - measured_fidelity) / 3.0
    kept_a = kept_fidelity + kept_error
    measured_a = measured_fidelity + measured_error
    gate_product = gate_fidelity * gate_fidelity
    success_probability = (
        0.5
        + gate_product
        * (
            kept_a * measured_a
            + (1.0 - kept_a) * (1.0 - measured_a)
        )
        - gate_product / 2.0
    )
    numerator = (
        gate_product
        * (
            kept_fidelity * measured_fidelity
            + kept_error * measured_error
        )
        + (1.0 - gate_product) / 8.0
    )
    if success_probability <= 0.0:
        return 0.0, 0.0
    output_fidelity = numerator / success_probability
    return (
        min(1.0, max(0.0, float(success_probability))),
        min(1.0, max(0.0, float(output_fidelity))),
    )


def werner_storage_fidelity_lower_bound(
    fidelity: float,
    storage_slots: int,
    memory_lifetime_slots: int,
) -> float:
    """Apply symmetric two-memory BDS decoherence for a coarse-slot hold."""

    if not 0.0 <= fidelity <= 1.0:
        raise ValueError("fidelity must lie in [0, 1]")
    if storage_slots < 0:
        raise ValueError("storage_slots cannot be negative")
    if memory_lifetime_slots < 1:
        raise ValueError("memory_lifetime_slots must be positive")
    if storage_slots == 0:
        return float(fidelity)
    werner_weight = (4.0 * fidelity - 1.0) / 3.0
    # SeQUeNCe's default BDS MemoryArray uses equal X/Y/Z error rates.
    # Each endpoint memory applies the channel, hence the factor of two.
    decay = math.exp(
        -8.0 * float(storage_slots)
        / (3.0 * float(memory_lifetime_slots))
    )
    result = (1.0 + 3.0 * werner_weight * decay) / 4.0
    return min(1.0, max(0.0, float(result)))


def estimate_sequence_bds_fidelity_lower_bound(
    physical: PhysicalConfig,
    dag: ConstructionDAG,
    terminal_segment_ids: tuple[str, ...],
    operation_slots: Mapping[str, int],
    *,
    max_storage_slots: int | None = None,
) -> ConstructionFidelityBound:
    """Estimate a construction's conditional terminal-fidelity lower bound.

    All success probabilities are intentionally excluded: fidelity feasibility
    asks whether a successful construction can meet its service threshold.
    Stochastic completion probability remains a separate physical metric.
    """

    if not terminal_segment_ids:
        raise ValueError("at least one terminal segment is required")
    if len(set(terminal_segment_ids)) != len(terminal_segment_ids):
        raise ValueError("terminal segment IDs must be unique")
    if max_storage_slots is not None and max_storage_slots < 1:
        raise ValueError("max_storage_slots must be positive when supplied")
    slots = {str(operation_id): int(slot) for operation_id, slot in operation_slots.items()}
    operation_ids = {operation.op_id for operation in dag.operations}
    if set(slots) != operation_ids:
        raise ValueError("operation slots must cover the DAG exactly")
    if any(slot < 0 for slot in slots.values()):
        raise ValueError("operation slots cannot be negative")

    segments: dict[str, _SegmentBound] = {}
    completed: set[str] = set()
    pending = list(dag.operations)
    expired: set[str] = set()

    def stored(segment_id: str, target_slot: int) -> _SegmentBound:
        segment = segments[segment_id]
        storage_slots = target_slot - segment.produced_slot
        if storage_slots < 0:
            raise ValueError("segment is consumed before it is produced")
        physically_expired = storage_slots >= physical.memory_lifetime
        policy_expired = (
            max_storage_slots is not None
            and storage_slots > max_storage_slots
        )
        alive = segment.alive and not physically_expired and not policy_expired
        if not alive:
            expired.add(segment_id)
            return _SegmentBound(0.0, target_slot, False)
        return _SegmentBound(
            werner_storage_fidelity_lower_bound(
                segment.fidelity,
                storage_slots,
                physical.memory_lifetime,
            ),
            target_slot,
            True,
        )

    while pending:
        progressed = False
        for operation in tuple(pending):
            if not set(operation.predecessors).issubset(completed):
                continue
            if any(
                segment_id not in segments
                for segment_id in operation.input_segment_ids
            ):
                continue
            slot = slots[operation.op_id]
            output_fidelity: float | None = None
            output_alive = True
            if operation.kind == OperationKind.GEN:
                output_fidelity = float(physical.initial_fidelity)
            elif operation.kind == OperationKind.PURIFY:
                if len(operation.input_segment_ids) != 2:
                    raise ValueError("PURIFY fidelity estimation requires two inputs")
                inputs = tuple(
                    stored(segment_id, slot)
                    for segment_id in operation.input_segment_ids
                )
                output_alive = all(item.alive for item in inputs)
                if output_alive:
                    success_probability, output_fidelity = werner_bbpssw_result(
                        inputs[0].fidelity,
                        inputs[1].fidelity,
                        physical.swap_degradation,
                    )
                    output_alive = success_probability > 0.0
                else:
                    output_fidelity = 0.0
            elif operation.kind == OperationKind.SWAP:
                if len(operation.input_segment_ids) != 2:
                    raise ValueError("SWAP fidelity estimation requires two inputs")
                inputs = tuple(
                    stored(segment_id, slot)
                    for segment_id in operation.input_segment_ids
                )
                output_alive = all(item.alive for item in inputs)
                output_fidelity = (
                    werner_swap_fidelity(
                        inputs[0].fidelity,
                        inputs[1].fidelity,
                        physical.swap_degradation,
                    )
                    if output_alive
                    else 0.0
                )
            elif operation.kind != OperationKind.RELEASE:
                raise ValueError(f"unsupported construction operation: {operation.kind}")

            if operation.output_segment_id is not None:
                if output_fidelity is None:
                    raise ValueError("output operation has no fidelity estimate")
                segments[operation.output_segment_id] = _SegmentBound(
                    output_fidelity,
                    slot,
                    output_alive,
                )
            completed.add(operation.op_id)
            pending.remove(operation)
            progressed = True
        if not progressed:
            raise ValueError("construction fidelity graph cannot be resolved")

    terminal_bounds = []
    for segment_id in terminal_segment_ids:
        if segment_id not in segments:
            raise ValueError(f"unknown terminal segment: {segment_id}")
        producer_slot = segments[segment_id].produced_slot
        terminal = stored(segment_id, producer_slot + 1)
        terminal_bounds.append((segment_id, terminal.fidelity))
    ordered = tuple(sorted(terminal_bounds))
    return ConstructionFidelityBound(
        model_name=FIDELITY_MODEL_NAME,
        lower_bound=min(value for _, value in ordered),
        terminal_bounds=ordered,
        expired_segment_ids=tuple(sorted(expired)),
    )
