"""Neutral construction-DAG repair candidate generation.

Repair candidates are expressed entirely with construction DTOs.  A failed
SWAP may have consumed its input segments, so a retry must first rebuild the
missing logical prefix from the original producer operations.  Rebuild
operations are deliberately single-use (``retry_limit=0``); the outer failed
operation owns the bounded retry lineage and a subsequent physical failure is
handled by the normal DROP path.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from .construction_api import ConstructionDAG, ConstructionOperation, OperationKind


def _fresh_id(base: str, existing: set[str]) -> str:
    candidate = base
    suffix = 1
    while candidate in existing:
        candidate = f"{base}:{suffix}"
        suffix += 1
    existing.add(candidate)
    return candidate


def generate_repair_options(
    dag: ConstructionDAG,
    available_segments: set[str],
    *,
    next_version: int,
    ordinal_start: int,
    required_fidelity_for: Callable[[ConstructionOperation], float] | None = None,
) -> tuple[tuple[ConstructionOperation, ...], ...]:
    """Return bounded retry/drop alternatives for all currently dead operations.

    A normal retry is returned when every input segment survives.  For a dead
    SWAP with missing inputs, the helper recursively reconstructs those inputs
    from their GEN/SWAP producer definitions and appends a new SWAP attempt.
    The generated prefix is topologically ordered and consumes only fresh
    operation/segment IDs.
    """
    if next_version != dag.version + 1:
        raise ValueError("repair version must be the next DAG version")
    if ordinal_start < 0:
        raise ValueError("ordinal_start must be non-negative")
    available = set(available_segments)
    operations = dag.operations
    operation_ids = {operation.op_id for operation in operations}
    producers = {
        operation.output_segment_id: operation
        for operation in operations
        if operation.output_segment_id is not None
    }
    required_fidelity_for = required_fidelity_for or (
        lambda operation: operation.required_fidelity
    )
    options: list[tuple[ConstructionOperation, ...]] = []

    for dead in sorted(
        (operation for operation in operations if operation.op_id in dag.dead),
        key=lambda operation: operation.canonical_key,
    ):
        next_attempt = dead.retry_attempt + 1
        if next_attempt > dead.retry_limit:
            continue
        retry_root_id = dead.retry_root_id or dead.op_id
        if any(
            operation.retry_root_id == retry_root_id
            and operation.retry_attempt == next_attempt
            for operation in operations
        ):
            continue

        staged: list[ConstructionOperation] = []
        replacements: dict[str, str] = {}
        visiting: set[str] = set()
        next_ordinal = ordinal_start

        def rebuild(segment_id: str) -> str:
            nonlocal next_ordinal
            if segment_id in available:
                return segment_id
            if segment_id in replacements:
                return replacements[segment_id]
            producer = producers.get(segment_id)
            if producer is None or producer.kind not in {
                OperationKind.GEN,
                OperationKind.SWAP,
            }:
                raise ValueError(f"cannot rebuild missing segment: {segment_id}")
            if producer.op_id in visiting:
                raise ValueError("repair producer graph contains a cycle")
            visiting.add(producer.op_id)
            rebuilt_inputs = tuple(rebuild(item) for item in producer.input_segment_ids)
            producer_base = f"{retry_root_id}:rebuild:{next_attempt}:{producer.op_id}"
            operation_id = _fresh_id(producer_base, operation_ids)
            output_id = _fresh_id(
                f"{segment_id}:repair:{next_attempt}",
                {item.output_segment_id for item in operations if item.output_segment_id}
                | {item.output_segment_id for item in staged if item.output_segment_id},
            )
            predecessors = tuple(
                operation_id
                for operation_id in (
                    item.op_id
                    for item in staged
                    if item.output_segment_id in rebuilt_inputs
                )
            )
            rebuilt = replace(
                producer,
                op_id=operation_id,
                input_segment_ids=rebuilt_inputs,
                output_segment_id=output_id,
                predecessors=predecessors,
                required_fidelity=required_fidelity_for(producer),
                retry_limit=0,
                retry_root_id=None,
                retry_attempt=0,
                dag_version=next_version,
                ordinal=next_ordinal,
            )
            next_ordinal += 1
            staged.append(rebuilt)
            replacements[segment_id] = output_id
            visiting.remove(producer.op_id)
            return output_id

        try:
            if dead.kind == OperationKind.SWAP:
                retry_inputs = tuple(rebuild(item) for item in dead.input_segment_ids)
            elif not set(dead.input_segment_ids).issubset(available):
                continue
            else:
                retry_inputs = dead.input_segment_ids
        except ValueError:
            continue

        retry_id = _fresh_id(
            f"{retry_root_id}:retry:{next_attempt}", operation_ids
        )
        retry_output = (
            None
            if dead.output_segment_id is None
            else _fresh_id(
                f"{dead.output_segment_id}:retry:{next_attempt}",
                {item.output_segment_id for item in operations if item.output_segment_id}
                | {item.output_segment_id for item in staged if item.output_segment_id},
            )
        )
        staged_ids = tuple(item.op_id for item in staged)
        completed_predecessors = tuple(
            predecessor
            for predecessor in dead.predecessors
            if predecessor in dag.completed
        )
        retry = replace(
            dead,
            op_id=retry_id,
            input_segment_ids=retry_inputs,
            output_segment_id=retry_output,
            predecessors=staged_ids + completed_predecessors,
            required_fidelity=required_fidelity_for(dead),
            ordinal=next_ordinal,
            dag_version=next_version,
            retry_root_id=retry_root_id,
            retry_attempt=next_attempt,
        )
        options.append(tuple(staged) + (retry,))
    return tuple(options)


__all__ = ["generate_repair_options"]
