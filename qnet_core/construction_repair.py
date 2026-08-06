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
from typing import Callable, Iterable, Sequence

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


def rebase_route_repair_dag(
    source_dag: ConstructionDAG,
    terminal_segment_ids: Sequence[str],
    *,
    next_version: int,
    ordinal_start: int,
    option_prefix: str,
    existing_operation_ids: Iterable[str] = (),
    existing_output_segment_ids: Iterable[str] = (),
    release_segment_ids: Sequence[str] = (),
) -> tuple[tuple[ConstructionOperation, ...], tuple[str, ...]]:
    """Clone a catalogue DAG as a fresh reroute suffix.

    The source candidate is a template only.  Every operation and produced
    segment receives a fresh ID, all dependencies are remapped, and retry
    lineage is cleared so a reroute cannot silently reset the failed branch's
    bounded retry counter.
    """

    if next_version < 1:
        raise ValueError("route repair version must be positive")
    if ordinal_start < 0:
        raise ValueError("ordinal_start must be non-negative")
    if not option_prefix:
        raise ValueError("route repair option prefix must be non-empty")
    terminals = tuple(terminal_segment_ids)
    if not terminals or len(set(terminals)) != len(terminals):
        raise ValueError("route repair terminals must be unique and non-empty")

    source_operations = {operation.op_id: operation for operation in source_dag.operations}
    produced_segments = {
        operation.output_segment_id
        for operation in source_operations.values()
        if operation.output_segment_id is not None
    }
    if not set(terminals).issubset(produced_segments):
        raise ValueError("route repair terminal is not produced by the source DAG")

    ordered: list[ConstructionOperation] = []
    emitted: set[str] = set()
    pending = dict(source_operations)
    while pending:
        ready = sorted(
            (
                operation
                for operation in pending.values()
                if set(operation.predecessors).issubset(emitted)
            ),
            key=lambda operation: operation.canonical_key,
        )
        if not ready:
            raise ValueError("route repair source DAG is not topologically orderable")
        for operation in ready:
            ordered.append(operation)
            emitted.add(operation.op_id)
            pending.pop(operation.op_id)

    used_operation_ids = set(existing_operation_ids)
    used_output_ids = set(existing_output_segment_ids)
    release_ids: list[str] = []
    release_operations: list[ConstructionOperation] = []
    for index, segment_id in enumerate(tuple(dict.fromkeys(release_segment_ids))):
        if not segment_id:
            raise ValueError("route repair release segment ID must be non-empty")
        operation_id = _fresh_id(
            f"{option_prefix}:release:{index}:{segment_id}",
            used_operation_ids,
        )
        release_ids.append(operation_id)
        release_operations.append(ConstructionOperation(
            op_id=operation_id,
            request_id=source_dag.request_id,
            kind=OperationKind.RELEASE,
            input_segment_ids=(segment_id,),
            duration_ps=1,
            ordinal=ordinal_start + index,
            dag_version=next_version,
        ))
    operation_id_map: dict[str, str] = {}
    output_id_map: dict[str, str] = {}
    for index, operation in enumerate(ordered):
        operation_id_map[operation.op_id] = _fresh_id(
            f"{option_prefix}:op:{index}:{operation.op_id}",
            used_operation_ids,
        )
        if operation.output_segment_id is not None:
            output_id_map[operation.output_segment_id] = _fresh_id(
                f"{option_prefix}:segment:{index}:{operation.output_segment_id}",
                used_output_ids,
            )

    rebased_candidate = tuple(
        replace(
            operation,
            op_id=operation_id_map[operation.op_id],
            predecessors=(
                tuple(
                    operation_id_map[predecessor]
                    for predecessor in operation.predecessors
                )
                if operation.predecessors
                else tuple(release_ids)
            ),
            input_segment_ids=tuple(
                output_id_map.get(segment_id, segment_id)
                for segment_id in operation.input_segment_ids
            ),
            output_segment_id=(
                None
                if operation.output_segment_id is None
                else output_id_map[operation.output_segment_id]
            ),
            retry_limit=0,
            retry_root_id=None,
            retry_attempt=0,
            ordinal=ordinal_start + len(release_operations) + index,
            dag_version=next_version,
        )
        for index, operation in enumerate(ordered)
    )
    return (
        tuple(release_operations) + rebased_candidate,
        tuple(output_id_map[segment_id] for segment_id in terminals),
    )


__all__ = ["generate_repair_options", "rebase_route_repair_dag"]
