"""Rolling-horizon TELGEN planning on one persistent SeQUeNCe episode."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
import csv
import json
from pathlib import Path
import shutil
from statistics import fmean
from time import perf_counter
from typing import TYPE_CHECKING, Mapping

import numpy as np

from qnet_core.construction_api import ExecutionEvent
from qnet_core.construction_metrics import (
    RequestSettlement,
    censored_flow_time,
    execution_event_metrics,
)
from qnet_core.resource_catalog import build_resource_capacities
from qnet_core.scheduled_execution import (
    PersistentConstructionScheduler,
    PersistentScheduleUpdate,
    ScheduleViolation,
    ScheduledOperationLaunch,
)
from qnet_core.spec import EpisodeSpec

from .dataset import (
    PlanningBatchProblem,
    build_planning_batch_problem,
)
from .milp_oracle import (
    ConstructionAwareMILPOracle,
    DiscreteOracleSolution,
    is_numerically_optimal_result,
)
from .physical_validation import (
    compile_selected_schedule,
)
from .time_expansion import TimeExpandedCandidate

if TYPE_CHECKING:
    from .ipm_policy import OnlineIPMGNNPolicy


@dataclass(frozen=True)
class OnlineTELGENConfig:
    """Planning-window parameters; physical parameters remain in the spec."""

    decision_interval: int = 4
    path_candidate_count: int = 3
    construction_kinds: tuple[str, ...] = ("left_deep", "balanced")
    swap_tree_count: int | None = None
    purification_kinds: tuple[str, ...] = ("none", "elementary_once")
    decision_backend: str = "milp_teacher"
    gnn_checkpoint: str | None = None
    gnn_device: str = "auto"
    milp_time_limit_seconds: float = 60.0
    milp_relative_gap: float = 0.0

    def __post_init__(self) -> None:
        if self.decision_interval < 1:
            raise ValueError("decision_interval must be positive")
        if self.path_candidate_count < 1:
            raise ValueError("path_candidate_count must be positive")
        if not self.construction_kinds and self.swap_tree_count is None:
            raise ValueError("at least one construction policy is required")
        if self.swap_tree_count is not None and self.swap_tree_count < 1:
            raise ValueError("swap_tree_count must be positive")
        if not self.purification_kinds:
            raise ValueError("at least one purification kind is required")
        if self.decision_backend not in {"milp_teacher", "ipm_gnn"}:
            raise ValueError(
                f"unknown online decision backend: {self.decision_backend}"
            )
        if self.milp_time_limit_seconds <= 0:
            raise ValueError("milp_time_limit_seconds must be positive")
        if self.milp_relative_gap < 0:
            raise ValueError("milp_relative_gap cannot be negative")
        if self.gnn_device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"unknown GNN device: {self.gnn_device}")
        if (
            self.decision_backend == "ipm_gnn"
            and not self.gnn_checkpoint
        ):
            raise ValueError("GNN decision backend requires a checkpoint")
        if (
            self.decision_backend == "milp_teacher"
            and self.milp_relative_gap != 0.0
        ):
            raise ValueError(
                "milp_teacher labels require a zero requested relative gap"
            )


@dataclass(frozen=True)
class OnlineDecisionRecord:
    decision_slot: int
    window_end_slot: int
    completion_end_slot: int
    visible_request_ids: tuple[str, ...]
    eligible_request_ids: tuple[str, ...]
    running_request_ids: tuple[str, ...]
    selected_variable_ids: tuple[str, ...]
    deferred_request_ids: tuple[str, ...]
    candidate_count: int
    variable_count: int
    candidate_rejection_count: int
    reserved_resource_slot_count: int
    selected_request_count: int
    selected_expected_completed_mass: float
    selection_strategy: str
    selected_expected_completion_latency: float
    planner_seconds: float
    decision_seconds: float
    decision_backend: str = "milp_teacher"
    policy_inference_seconds: float = 0.0
    milp_mip_gap: float | None = None
    policy_output_feasible: bool | None = None
    policy_invalid_action_index: int | None = None
    policy_invalid_action_reason: str | None = None


@dataclass(frozen=True)
class OnlineAttemptRecord:
    request_id: str
    attempt: int
    decision_slot: int
    variable_id: str
    candidate_id: str
    route_nodes: tuple[int, ...]
    construction_kind: str
    purification_kind: str
    expected_success_probability: float
    planned_start_slot: int
    planned_completion_slot: int
    success: bool | None
    settlement_time_ps: int | None
    failure_cause: str


@dataclass(frozen=True)
class OnlineTELGENResult:
    config: OnlineTELGENConfig
    episode: EpisodeSpec
    episode_seed: int
    horizon_slots: int
    decisions: tuple[OnlineDecisionRecord, ...]
    attempts: tuple[OnlineAttemptRecord, ...]
    settlements: tuple[RequestSettlement, ...]
    launches: tuple[ScheduledOperationLaunch, ...]
    violations: tuple[ScheduleViolation, ...]
    event_trace: tuple[ExecutionEvent, ...]
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class OnlineResultPaths:
    json_path: Path
    csv_path: Path
    latest_json_path: Path
    latest_csv_path: Path


class OnlineTELGENController:
    """Execute TELGEN plans through one persistent SeQUeNCe scheduler."""

    def __init__(
        self,
        spec: EpisodeSpec,
        config: OnlineTELGENConfig | None = None,
        *,
        milp_oracle: ConstructionAwareMILPOracle | None = None,
        ipm_gnn_policy: "OnlineIPMGNNPolicy | None" = None,
    ):
        self.spec = spec
        self.config = config or OnlineTELGENConfig()
        self.milp_oracle = milp_oracle or ConstructionAwareMILPOracle(
            time_limit_seconds=self.config.milp_time_limit_seconds,
            mip_relative_gap=self.config.milp_relative_gap,
        )
        self.ipm_gnn_policy = ipm_gnn_policy
        if self.config.decision_backend == "ipm_gnn":
            if self.ipm_gnn_policy is None:
                from .ipm_policy import OnlineIPMGNNPolicy

                self.ipm_gnn_policy = OnlineIPMGNNPolicy.from_checkpoint(
                    self.config.gnn_checkpoint,
                    device=self.config.gnn_device,
                )
        elif self.ipm_gnn_policy is not None:
            raise ValueError(
                "an IPM GNN policy requires the ipm_gnn decision backend"
            )
        self.capacities = build_resource_capacities(spec)
        self.scheduler = PersistentConstructionScheduler(spec)
        self.requests = {request.id: request for request in spec.requests}
        self._running_variables: dict[str, TimeExpandedCandidate] = {}
        self._attempt_counts: dict[str, int] = {}
        self._attempts: list[OnlineAttemptRecord] = []
        self._active_attempt_index: dict[str, int] = {}
        self._decisions: list[OnlineDecisionRecord] = []
        self._expired_times: dict[str, int] = {}

    def _visible_request_ids(self, slot: int) -> tuple[str, ...]:
        return tuple(sorted(
            request.id
            for request in self.spec.requests
            if request.arrival <= slot
            and request.id not in self.scheduler.completed_request_ids
            and request.id not in self._expired_times
        ))

    def _expire_waiting_requests(self, slot: int) -> None:
        for request in self.spec.requests:
            if request.id in self.scheduler.completed_request_ids:
                continue
            if request.id in self._running_variables:
                continue
            if request.id in self._expired_times:
                continue
            if request.deadline is not None and request.deadline <= slot:
                self._expired_times[request.id] = (
                    request.deadline * self.spec.physical.slot_duration_ps
                )

    def _eligible_requests(self, slot: int) -> tuple[str, ...]:
        return tuple(sorted(
            request.id
            for request in self.spec.requests
            if request.arrival <= slot
            and (request.deadline is None or request.deadline > slot)
            and request.id not in self._running_variables
            and request.id not in self.scheduler.completed_request_ids
            and request.id not in self._expired_times
            and self.scheduler.can_submit(request.id)
        ))

    def _reserved_usage(self, window_end_slot: int) -> dict[tuple[str, int], int]:
        current_slot = self.scheduler.current_slot
        by_request: dict[str, dict[tuple[str, int], int]] = {}
        for request_id, variable in self._running_variables.items():
            request_usage = by_request.setdefault(request_id, {})
            for usage in variable.resource_usage:
                if not current_slot <= usage.slot < window_end_slot:
                    continue
                key = (usage.resource_id, usage.slot)
                request_usage[key] = request_usage.get(key, 0) + usage.amount

        physical_by_request = self.scheduler.physical_reservations_by_request(
            window_end_slot
        )
        for request_id, physical_usage in physical_by_request.items():
            request_usage = by_request.setdefault(request_id, {})
            for key, amount in physical_usage.items():
                request_usage[key] = max(request_usage.get(key, 0), amount)

        reserved: dict[tuple[str, int], int] = {}
        for request_usage in by_request.values():
            for key, amount in request_usage.items():
                resource_id, _ = key
                reserved[key] = min(
                    self.capacities[resource_id],
                    reserved.get(key, 0) + amount,
                )
        return reserved

    def _build_decision_problem(
        self,
        slot: int,
        start_window_end_slot: int,
        eligible_request_ids: tuple[str, ...],
        reserved_usage: Mapping[tuple[str, int], int],
    ) -> PlanningBatchProblem | None:
        if not eligible_request_ids:
            return None
        eligible = set(eligible_request_ids)
        window_episode = replace(
            self.spec,
            requests=tuple(
                request for request in self.spec.requests
                if request.id in eligible
            ),
        )
        return build_planning_batch_problem(
            window_episode,
            window_start_slot=slot,
            window_end_slot=start_window_end_slot,
            completion_end_slot=self.spec.horizon,
            reserved_usage=reserved_usage,
            resource_capacities=self.capacities,
            path_candidate_count=self.config.path_candidate_count,
            construction_kinds=self.config.construction_kinds,
            swap_tree_count=self.config.swap_tree_count,
            purification_kinds=self.config.purification_kinds,
        )

    def _solve_milp_decision(
        self,
        problem: PlanningBatchProblem | None,
    ) -> tuple[DiscreteOracleSolution | None, float]:
        if problem is None or not problem.expansion.variables:
            return None, 0.0
        started = perf_counter()
        solution = self.milp_oracle.solve(
            problem.expansion,
            problem.capacities,
            reserved_usage=problem.reserved_usage_map,
            request_censoring_latencies=(
                problem.request_censoring_latency_map
            ),
        )
        if not is_numerically_optimal_result(solution.result):
            result = solution.result
            raise RuntimeError(
                f"{result.solve_name} did not reach certified numerical "
                f"optimality: status={result.status}, gap={result.mip_gap}, "
                f"objective={result.objective_value}, "
                f"dual_bound={result.mip_dual_bound}, "
                f"message={result.message}"
            )
        return solution, perf_counter() - started

    def _solve_ipm_gnn_decision(
        self,
        problem: PlanningBatchProblem | None,
    ):
        if problem is None or not problem.expansion.variables:
            return None
        if self.ipm_gnn_policy is None:
            raise RuntimeError("IPM GNN backend has no loaded policy")
        return self.ipm_gnn_policy.decide(
            problem.expansion.variables,
            problem.capacities,
            horizon=problem.episode.horizon,
            reserved_usage=problem.reserved_usage_map,
            request_censoring_latencies=(
                problem.request_censoring_latency_map
            ),
        )

    def _register_selected_variables(
        self,
        slot: int,
        selected_variables: tuple[TimeExpandedCandidate, ...],
        eligible_request_ids: tuple[str, ...],
    ) -> None:
        schedule = compile_selected_schedule(
            selected_variables,
            eligible_request_ids,
            self.capacities,
            horizon_slots=self.spec.horizon,
        )
        self.scheduler.submit(schedule.requests)
        for variable in sorted(
            selected_variables,
            key=lambda item: item.request_id,
        ):
            request_id = variable.request_id
            self._running_variables[request_id] = variable
            attempt = self._attempt_counts.get(request_id, 0) + 1
            self._attempt_counts[request_id] = attempt
            self._attempts.append(OnlineAttemptRecord(
                request_id=request_id,
                attempt=attempt,
                decision_slot=slot,
                variable_id=variable.variable_id,
                candidate_id=variable.candidate_id,
                route_nodes=variable.route_nodes,
                construction_kind=variable.construction_kind,
                purification_kind=variable.purification_kind,
                expected_success_probability=(
                    variable.expected_success_probability
                ),
                planned_start_slot=variable.start_slot,
                planned_completion_slot=variable.completion_slot,
                success=None,
                settlement_time_ps=None,
                failure_cause="",
            ))
            self._active_attempt_index[request_id] = len(self._attempts) - 1

    def _process_update(self, update: PersistentScheduleUpdate) -> None:
        for outcome in update.outcomes:
            self._running_variables.pop(outcome.request_id, None)
            index = self._active_attempt_index.pop(outcome.request_id, None)
            if index is None:
                continue
            previous = self._attempts[index]
            self._attempts[index] = replace(
                previous,
                success=outcome.success,
                settlement_time_ps=outcome.settlement_time_ps,
                failure_cause=outcome.failure_cause,
            )

    def _decision(self, slot: int) -> None:
        self._expire_waiting_requests(slot)
        visible = self._visible_request_ids(slot)
        eligible = self._eligible_requests(slot)
        running = tuple(sorted(self._running_variables))
        start_window_end = min(
            self.spec.horizon,
            slot + self.config.decision_interval,
        )
        if start_window_end <= slot:
            return
        reserved = self._reserved_usage(self.spec.horizon)
        started = perf_counter()
        problem = self._build_decision_problem(
            slot,
            start_window_end,
            eligible,
            reserved,
        )
        milp_solution: DiscreteOracleSolution | None = None
        gnn_decision = None
        milp_solve_seconds = 0.0
        if self.config.decision_backend == "milp_teacher":
            milp_solution, milp_solve_seconds = self._solve_milp_decision(
                problem
            )
        elif self.config.decision_backend == "ipm_gnn":
            gnn_decision = self._solve_ipm_gnn_decision(problem)
        selected_ids: tuple[str, ...] = ()
        selected_count = 0
        expected_completed_mass = 0.0
        selected_variables: tuple[TimeExpandedCandidate, ...] = ()
        if milp_solution is not None:
            selected_variables = milp_solution.selected_variables
            selected_ids = tuple(
                variable.variable_id for variable in selected_variables
            )
            selected_count = milp_solution.completed_request_count
            expected_completed_mass = (
                milp_solution.expected_completed_request_mass
            )
            self._register_selected_variables(
                slot,
                selected_variables,
                eligible,
            )
        elif gnn_decision is not None:
            selected_variables = gnn_decision.selection.selected_variables
            selected_ids = gnn_decision.selection.selected_variable_ids
            selected_count = gnn_decision.selection.completed_request_count
            expected_completed_mass = (
                gnn_decision.selection.expected_completed_request_mass
            )
            if gnn_decision.selection.feasible:
                self._register_selected_variables(
                    slot,
                    selected_variables,
                    eligible,
                )
        decision_seconds = perf_counter() - started
        selected_requests = {
            variable.request_id for variable in selected_variables
        }
        self._decisions.append(OnlineDecisionRecord(
            decision_slot=slot,
            window_end_slot=start_window_end,
            completion_end_slot=self.spec.horizon,
            visible_request_ids=visible,
            eligible_request_ids=eligible,
            running_request_ids=running,
            selected_variable_ids=selected_ids,
            deferred_request_ids=tuple(
                request_id for request_id in eligible
                if request_id not in selected_requests
            ),
            candidate_count=(
                0 if problem is None else len(problem.candidates)
            ),
            variable_count=(
                0 if problem is None else len(problem.expansion.variables)
            ),
            candidate_rejection_count=(
                0 if problem is None else len(problem.expansion.rejections)
            ),
            reserved_resource_slot_count=len(reserved),
            selected_request_count=selected_count,
            selected_expected_completed_mass=expected_completed_mass,
            selection_strategy=(
                "milp_exact"
                if milp_solution is not None
                else "gnn_ipm_shared_rounding"
                if gnn_decision is not None
                else "none"
            ),
            selected_expected_completion_latency=(
                milp_solution.total_completion_latency
                if milp_solution is not None
                else (
                    0.0
                    if gnn_decision is None
                    else gnn_decision.selection.total_completion_latency
                )
            ),
            planner_seconds=(
                milp_solve_seconds
                if gnn_decision is None
                else gnn_decision.inference_seconds
            ),
            decision_seconds=decision_seconds,
            decision_backend=self.config.decision_backend,
            policy_inference_seconds=(
                0.0
                if gnn_decision is None
                else gnn_decision.inference_seconds
            ),
            milp_mip_gap=(
                None
                if milp_solution is None
                else milp_solution.result.mip_gap
            ),
            policy_output_feasible=(
                None
                if gnn_decision is None
                else gnn_decision.selection.feasible
            ),
            policy_invalid_action_index=(
                None
                if gnn_decision is None
                else gnn_decision.invalid_action_index
            ),
            policy_invalid_action_reason=(
                None
                if gnn_decision is None
                else gnn_decision.invalid_action_reason
            ),
        ))

    def _settlements(self) -> tuple[RequestSettlement, ...]:
        completed = self.scheduler.completed_times
        horizon_ps = self.spec.horizon * self.spec.physical.slot_duration_ps
        settlements = []
        for request in self.spec.requests:
            arrival_ps = request.arrival * self.spec.physical.slot_duration_ps
            if request.id in completed:
                settlements.append(RequestSettlement(
                    request.id,
                    arrival_ps,
                    completed[request.id],
                    True,
                ))
                continue
            deadline_ps = (
                None
                if request.deadline is None
                else request.deadline * self.spec.physical.slot_duration_ps
            )
            settlements.append(RequestSettlement(
                request.id,
                arrival_ps,
                self._expired_times.get(
                    request.id,
                    horizon_ps if deadline_ps is None else min(horizon_ps, deadline_ps),
                ),
                False,
            ))
        return tuple(settlements)

    @staticmethod
    def _percentile(values: list[int], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(
            len(ordered) - 1,
            max(0, int(round(percentile / 100.0 * (len(ordered) - 1)))),
        )
        return float(ordered[index])

    @staticmethod
    def _gini(values: list[int]) -> float:
        """Return the Gini coefficient of non-negative request delays."""

        if not values:
            return 0.0
        ordered = np.asarray(values, dtype=float)
        if np.any(ordered < 0.0) or not np.all(np.isfinite(ordered)):
            raise ValueError("completion delays must be finite and non-negative")
        total = float(np.sum(ordered))
        if total <= 0.0:
            return 0.0
        count = ordered.size
        return float(
            np.abs(ordered[:, None] - ordered[None, :]).sum()
            / (2.0 * count * total)
        )

    def _metrics(
        self,
        settlements: tuple[RequestSettlement, ...],
    ) -> dict[str, float]:
        horizon_ps = self.spec.horizon * self.spec.physical.slot_duration_ps
        completed = sum(item.success for item in settlements)
        successful_latencies = [
            item.settlement_time - item.arrival_time
            for item in settlements
            if item.success
        ]
        censored_latencies = [
            max(
                0,
                (
                    item.settlement_time
                    if item.success
                    else horizon_ps
                ) - item.arrival_time,
            )
            for item in settlements
        ]
        successful_request_ids = {
            item.request_id for item in settlements if item.success
        }
        final_fidelity_events: dict[str, ExecutionEvent] = {}
        for event in self.scheduler.event_trace:
            if (
                event.request_id not in successful_request_ids
                or not event.success
                or event.output_fidelity is None
            ):
                continue
            previous = final_fidelity_events.get(event.request_id)
            if previous is None or (
                event.physical_time_ps,
                event.event_id,
            ) >= (previous.physical_time_ps, previous.event_id):
                final_fidelity_events[event.request_id] = event
        final_fidelity_losses = [
            max(0.0, 1.0 - float(event.output_fidelity))
            for event in final_fidelity_events.values()
            if event.output_fidelity is not None
        ]
        flow_time = censored_flow_time(settlements, horizon_ps)
        attempted_requests = len(self._attempt_counts)
        total_attempts = len(self._attempts)
        decision_times = [item.decision_seconds for item in self._decisions]
        planner_times = [
            item.planner_seconds for item in self._decisions
            if item.variable_count > 0
        ]
        gnn_policy_decisions = [
            item for item in self._decisions
            if getattr(item, "policy_output_feasible", None) is not None
        ]
        invalid_gnn_decisions = [
            item for item in gnn_policy_decisions
            if not getattr(item, "policy_output_feasible", True)
        ]
        event_metrics = execution_event_metrics(self.scheduler.event_trace)
        return {
            "request_count": float(len(settlements)),
            "completed_requests": float(completed),
            "completion_rate": completed / max(len(settlements), 1),
            "throughput_per_slot": completed / max(self.spec.horizon, 1),
            "censored_flow_time_ps": float(flow_time),
            "mean_censored_latency_ps": flow_time / max(len(settlements), 1),
            "mean_completion_delay_ps": flow_time / max(len(settlements), 1),
            "max_completion_delay_ps": float(max(censored_latencies, default=0)),
            "completion_delay_gini": self._gini(censored_latencies),
            "mean_final_fidelity_loss": (
                0.0
                if not final_fidelity_losses
                else fmean(final_fidelity_losses)
            ),
            "final_fidelity_observation_count": float(
                len(final_fidelity_losses)
            ),
            "mean_success_latency_ps": (
                0.0 if not successful_latencies else fmean(successful_latencies)
            ),
            "p95_completion_latency_ps": self._percentile(
                successful_latencies,
                95.0,
            ),
            "decision_count": float(len(self._decisions)),
            "attempted_request_count": float(attempted_requests),
            "construction_attempt_count": float(total_attempts),
            "retry_count": float(max(0, total_attempts - attempted_requests)),
            "mean_decision_seconds": (
                0.0 if not decision_times else fmean(decision_times)
            ),
            "mean_planner_seconds": (
                0.0 if not planner_times else fmean(planner_times)
            ),
            "mean_policy_inference_seconds": (
                0.0
                if not self._decisions
                else fmean(
                    getattr(item, "policy_inference_seconds", 0.0)
                    for item in self._decisions
                )
            ),
            "gnn_policy_decision_count": float(len(gnn_policy_decisions)),
            "gnn_invalid_decision_count": float(
                len(invalid_gnn_decisions)
            ),
            "gnn_invalid_decision_rate": (
                len(invalid_gnn_decisions)
                / max(len(gnn_policy_decisions), 1)
            ),
            "schedule_violation_count": float(len(self.scheduler.violations)),
            "schedule_adherence": float(not self.scheduler.violations),
            "makespan_ps": float(max(
                (
                    event.physical_time_ps
                    for event in self.scheduler.event_trace
                ),
                default=0,
            )),
            **self.scheduler.memory_metrics(),
            **event_metrics,
        }

    def run(self) -> OnlineTELGENResult:
        while self.scheduler.current_slot < self.spec.horizon:
            slot = self.scheduler.current_slot
            self._decision(slot)
            target = min(
                self.spec.horizon,
                slot + self.config.decision_interval,
            )
            update = self.scheduler.advance_to_slot(target)
            self._process_update(update)
        self._expire_waiting_requests(self.spec.horizon)
        for request_id, index in tuple(self._active_attempt_index.items()):
            previous = self._attempts[index]
            self._attempts[index] = replace(
                previous,
                success=False,
                settlement_time_ps=self.horizon_ps,
                failure_cause="horizon_timeout",
            )
            self._active_attempt_index.pop(request_id, None)
        settlements = self._settlements()
        return OnlineTELGENResult(
            config=self.config,
            episode=self.spec,
            episode_seed=self.spec.seed,
            horizon_slots=self.spec.horizon,
            decisions=tuple(self._decisions),
            attempts=tuple(self._attempts),
            settlements=settlements,
            launches=self.scheduler.launches,
            violations=self.scheduler.violations,
            event_trace=self.scheduler.event_trace,
            metrics=self._metrics(settlements),
        )

    @property
    def horizon_ps(self) -> int:
        return self.spec.horizon * self.spec.physical.slot_duration_ps


def run_online_telgen(
    spec: EpisodeSpec,
    config: OnlineTELGENConfig | None = None,
    *,
    milp_oracle: ConstructionAwareMILPOracle | None = None,
    ipm_gnn_policy: "OnlineIPMGNNPolicy | None" = None,
) -> OnlineTELGENResult:
    return OnlineTELGENController(
        spec,
        config,
        milp_oracle=milp_oracle,
        ipm_gnn_policy=ipm_gnn_policy,
    ).run()


def _json_payload(result: OnlineTELGENResult) -> dict[str, object]:
    return {
        "schema_version": 3,
        "episode_seed": result.episode_seed,
        "horizon_slots": result.horizon_slots,
        "episode": asdict(result.episode),
        "config": asdict(result.config),
        "metrics": dict(result.metrics),
        "decisions": [asdict(item) for item in result.decisions],
        "attempts": [asdict(item) for item in result.attempts],
        "settlements": [asdict(item) for item in result.settlements],
        "launches": [asdict(item) for item in result.launches],
        "violations": [asdict(item) for item in result.violations],
        "events": [asdict(item) for item in result.event_trace],
    }


def save_online_result(
    result: OnlineTELGENResult,
    output_directory: str | Path,
) -> OnlineResultPaths:
    """Write versioned JSON/CSV results plus fixed-name latest copies."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = ""
    collision_index = 1
    while True:
        json_path = output / f"online_telgen_results_{timestamp}{suffix}.json"
        csv_path = output / f"online_telgen_results_{timestamp}{suffix}.csv"
        if not json_path.exists() and not csv_path.exists():
            break
        collision_index += 1
        suffix = f"_{collision_index}"
    latest_json = output / "online_telgen_results.json"
    latest_csv = output / "online_telgen_results.csv"
    json_path.write_text(
        json.dumps(_json_payload(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    settlement_by_request = {
        item.request_id: item for item in result.settlements
    }
    attempts_by_request: dict[str, list[OnlineAttemptRecord]] = {}
    for attempt in result.attempts:
        attempts_by_request.setdefault(attempt.request_id, []).append(attempt)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "request_id",
            "arrival_time_ps",
            "settlement_time_ps",
            "success",
            "latency_ps",
            "attempt_count",
            "last_candidate_id",
            "last_failure_cause",
        ))
        writer.writeheader()
        for request_id in sorted(settlement_by_request):
            settlement = settlement_by_request[request_id]
            attempts = attempts_by_request.get(request_id, [])
            last_attempt = attempts[-1] if attempts else None
            writer.writerow({
                "request_id": request_id,
                "arrival_time_ps": settlement.arrival_time,
                "settlement_time_ps": settlement.settlement_time,
                "success": int(settlement.success),
                "latency_ps": settlement.settlement_time - settlement.arrival_time,
                "attempt_count": len(attempts),
                "last_candidate_id": (
                    "" if last_attempt is None else last_attempt.candidate_id
                ),
                "last_failure_cause": (
                    "" if last_attempt is None else last_attempt.failure_cause
                ),
            })
    shutil.copyfile(json_path, latest_json)
    shutil.copyfile(csv_path, latest_csv)
    return OnlineResultPaths(
        json_path=json_path,
        csv_path=csv_path,
        latest_json_path=latest_json,
        latest_csv_path=latest_csv,
    )
