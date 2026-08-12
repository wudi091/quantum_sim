"""Compile hard-decoded TELGEN plans and validate them through SeQUeNCe."""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import fmean
from typing import Iterable, Mapping, Sequence

from qnet_core.scheduled_execution import (
    ConstructionBatchSchedule,
    ScheduledConstructionEvaluation,
    ScheduledRequestPlan,
    run_scheduled_construction_plan,
)
from qnet_core.spec import EpisodeSpec

from .hard_decoder import HardDecoderSolution, validate_decoded_selection
from .time_expansion import TimeExpandedCandidate


@dataclass(frozen=True)
class PhysicalValidationTrial:
    seed: int
    evaluation: ScheduledConstructionEvaluation

    @property
    def completed_requests(self) -> int:
        return int(self.evaluation.metrics["completed_requests"])

    @property
    def schedule_adherent(self) -> bool:
        return bool(self.evaluation.metrics["schedule_adherence"])


@dataclass(frozen=True)
class PhysicalConsistencyReport:
    planned_selected_requests: int
    planned_total_completion_latency_slots: float
    trials: tuple[PhysicalValidationTrial, ...]
    mean_completed_requests: float
    mean_completion_retention: float | None
    mean_censored_latency_slots: float
    schedule_adherence_rate: float


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
    feasibility = validate_decoded_selection(
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


def compile_decoded_schedule(
    decoded: HardDecoderSolution,
    *,
    horizon_slots: int,
) -> ConstructionBatchSchedule:
    """Translate hard-decoder output into the simulator-neutral schedule DTO."""

    if not decoded.feasibility.feasible:
        raise ValueError("cannot compile an infeasible hard-decoder solution")
    return _compile_selected_schedule(
        decoded.selected_variables,
        decoded.rejected_request_ids,
        horizon_slots=horizon_slots,
    )


def evaluate_decoded_physics(
    spec: EpisodeSpec,
    decoded: HardDecoderSolution,
    *,
    physical_seed: int | None = None,
) -> ScheduledConstructionEvaluation:
    """Run one decoded schedule with an optional independent physical seed."""

    schedule = compile_decoded_schedule(decoded, horizon_slots=spec.horizon)
    physical_spec = spec if physical_seed is None else replace(
        spec, seed=int(physical_seed)
    )
    return run_scheduled_construction_plan(physical_spec, schedule)


def evaluate_selected_physics(
    spec: EpisodeSpec,
    selected_variables: Sequence[TimeExpandedCandidate],
    resource_capacities: Mapping[str, int],
    *,
    physical_seed: int | None = None,
) -> ScheduledConstructionEvaluation:
    """Run one exact MILP selection through the same SeQUeNCe boundary."""

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


def validate_decoded_physics(
    spec: EpisodeSpec,
    decoded: HardDecoderSolution,
    physical_seeds: Iterable[int],
) -> PhysicalConsistencyReport:
    """Repeat one fixed nominal plan under independent SeQUeNCe randomness."""

    seeds = tuple(int(seed) for seed in physical_seeds)
    if not seeds:
        raise ValueError("at least one physical seed is required")
    if any(seed < 0 for seed in seeds):
        raise ValueError("physical seeds must be non-negative")
    if len(set(seeds)) != len(seeds):
        raise ValueError("physical seeds must be unique")

    trials = tuple(
        PhysicalValidationTrial(
            seed,
            evaluate_decoded_physics(spec, decoded, physical_seed=seed),
        )
        for seed in seeds
    )
    planned = decoded.completed_request_count
    mean_completed = fmean(trial.completed_requests for trial in trials)
    retention = None if planned == 0 else mean_completed / planned
    slot_duration_ps = spec.physical.slot_duration_ps
    return PhysicalConsistencyReport(
        planned_selected_requests=planned,
        planned_total_completion_latency_slots=(
            decoded.total_completion_latency
        ),
        trials=trials,
        mean_completed_requests=mean_completed,
        mean_completion_retention=retention,
        mean_censored_latency_slots=fmean(
            trial.evaluation.metrics["mean_censored_latency_ps"]
            / slot_duration_ps
            for trial in trials
        ),
        schedule_adherence_rate=fmean(
            float(trial.schedule_adherent) for trial in trials
        ),
    )
