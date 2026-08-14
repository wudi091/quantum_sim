"""Compile discrete plans and validate them through SeQUeNCe."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Mapping, Sequence

from qnet_core.scheduled_execution import (
    ConstructionBatchSchedule,
    ScheduledConstructionEvaluation,
    ScheduledRequestPlan,
    run_scheduled_construction_plan,
)
from qnet_core.spec import EpisodeSpec

from .packing import validate_packing_selection
from .time_expansion import TimeExpandedCandidate


def _compile_selected_schedule(
    selected_variables: Sequence[TimeExpandedCandidate],
    rejected_request_ids: Iterable[str],
    *,
    horizon_slots: int,
) -> ConstructionBatchSchedule:
    if horizon_slots < 1:
        raise ValueError("horizon_slots must be positive")

    requests = []
    for variable in sorted(
        selected_variables,
        key=lambda item: item.request_id,
    ):
        absolute_operation_slots = tuple(sorted(
            (
                operation_id,
                variable.start_slot + relative_slot,
            )
            for operation_id, relative_slot
            in variable.nominal_schedule.operation_slots
        ))
        candidate = variable.base_candidate
        requests.append(ScheduledRequestPlan(
            request_id=variable.request_id,
            candidate_id=variable.candidate_id,
            route_nodes=variable.route_nodes,
            construction_kind=variable.construction_kind,
            dag=candidate.dag,
            terminal_segment_ids=candidate.all_terminal_segment_ids,
            start_slot=variable.start_slot,
            completion_slot=variable.completion_slot,
            operation_slots=absolute_operation_slots,
            purification_kind=variable.purification_kind,
        ))
    return ConstructionBatchSchedule(
        horizon_slots=horizon_slots,
        requests=tuple(requests),
        rejected_request_ids=tuple(sorted(set(rejected_request_ids))),
    )


def compile_selected_schedule(
    selected_variables: Sequence[TimeExpandedCandidate],
    request_ids: Iterable[str],
    resource_capacities: Mapping[str, int],
    *,
    horizon_slots: int,
) -> ConstructionBatchSchedule:
    """Compile one exact discrete selection into the neutral schedule DTO."""

    selected = tuple(sorted(
        selected_variables,
        key=lambda item: item.variable_id,
    ))
    declared_requests = tuple(sorted(str(item) for item in request_ids))
    if len(set(declared_requests)) != len(declared_requests):
        raise ValueError("request_ids must be unique")
    selected_requests = {variable.request_id for variable in selected}
    unknown = sorted(selected_requests - set(declared_requests))
    if unknown:
        raise ValueError(f"selected variable belongs to unknown request: {unknown[0]}")
    feasibility = validate_packing_selection(
        selected,
        resource_capacities,
    )
    if not feasibility.feasible:
        raise ValueError(
            "cannot compile an infeasible discrete selection: "
            f"{feasibility.violations[0]}"
        )
    return _compile_selected_schedule(
        selected,
        set(declared_requests) - selected_requests,
        horizon_slots=horizon_slots,
    )


def evaluate_selected_physics(
    spec: EpisodeSpec,
    selected_variables: Sequence[TimeExpandedCandidate],
    resource_capacities: Mapping[str, int],
    *,
    physical_seed: int | None = None,
) -> ScheduledConstructionEvaluation:
    """Run one discrete selection through the shared SeQUeNCe boundary."""

    schedule = compile_selected_schedule(
        selected_variables,
        (request.id for request in spec.requests),
        resource_capacities,
        horizon_slots=spec.horizon,
    )
    physical_spec = spec if physical_seed is None else replace(
        spec, seed=int(physical_seed)
    )
    return run_scheduled_construction_plan(physical_spec, schedule)
