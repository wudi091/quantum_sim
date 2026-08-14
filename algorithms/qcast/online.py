"""Rolling-horizon Q-CAST baseline on the shared SeQUeNCe executor."""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import fmean
from time import perf_counter
from typing import Mapping

from algorithms.telgen.packing import PackingSolution
from algorithms.telgen.online import (
    OnlineAttemptRecord,
    OnlineTELGENConfig,
    OnlineTELGENController,
)
from qnet_core.construction_api import ExecutionEvent
from qnet_core.construction_metrics import RequestSettlement
from qnet_core.scheduled_execution import ScheduleViolation, ScheduledOperationLaunch
from qnet_core.spec import EpisodeSpec

from .online_planner import QCASTPlanningRecord, plan_qcast_window


@dataclass(frozen=True)
class OnlineQCASTConfig:
    """Planning-window settings for the rolling Q-CAST adaptation."""

    decision_interval: int = 4
    path_candidate_count: int = 3
    construction_kind: str = "left_deep"
    purification_kind: str = "none"

    def __post_init__(self) -> None:
        if self.decision_interval < 1:
            raise ValueError("decision_interval must be positive")
        if self.path_candidate_count < 1:
            raise ValueError("path_candidate_count must be positive")
        if self.construction_kind not in {"left_deep", "balanced"}:
            raise ValueError("unsupported construction_kind")
        if self.purification_kind not in {"none", "elementary_once"}:
            raise ValueError("unsupported purification_kind")


@dataclass(frozen=True)
class OnlineQCASTDecisionRecord:
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
    selected_expected_throughput: float
    selected_request_count: int
    selected_total_completion_latency: float
    planner_seconds: float
    decision_seconds: float

@dataclass(frozen=True)
class OnlineQCASTResult:
    config: OnlineQCASTConfig
    episode: EpisodeSpec
    episode_seed: int
    horizon_slots: int
    decisions: tuple[OnlineQCASTDecisionRecord, ...]
    attempts: tuple[OnlineAttemptRecord, ...]
    settlements: tuple[RequestSettlement, ...]
    launches: tuple[ScheduledOperationLaunch, ...]
    violations: tuple[ScheduleViolation, ...]
    event_trace: tuple[ExecutionEvent, ...]
    metrics: Mapping[str, float]


class OnlineQCASTController(OnlineTELGENController):
    """Reuse the shared online lifecycle while replacing only the planner."""

    def __init__(
        self,
        spec: EpisodeSpec,
        config: OnlineQCASTConfig | None = None,
    ):
        self.qcast_config = config or OnlineQCASTConfig()
        super().__init__(
            spec,
            OnlineTELGENConfig(
                decision_interval=self.qcast_config.decision_interval,
                path_candidate_count=self.qcast_config.path_candidate_count,
                construction_kinds=(self.qcast_config.construction_kind,),
                purification_kinds=(self.qcast_config.purification_kind,),
            ),
        )
        self._decisions: list[OnlineQCASTDecisionRecord] = []

    def _solve_qcast_decision(
        self,
        slot: int,
        start_window_end_slot: int,
        eligible_request_ids: tuple[str, ...],
        reserved_usage: Mapping[tuple[str, int], int],
    ) -> tuple[QCASTPlanningRecord | None, PackingSolution | None]:
        if not eligible_request_ids:
            return None, None
        eligible = set(eligible_request_ids)
        window_episode = replace(
            self.spec,
            requests=tuple(
                request
                for request in self.spec.requests
                if request.id in eligible
            ),
        )
        record = plan_qcast_window(
            window_episode,
            window_start_slot=slot,
            window_end_slot=start_window_end_slot,
            completion_end_slot=self.spec.horizon,
            request_ids=eligible_request_ids,
            resource_capacities=self.capacities,
            reserved_usage=reserved_usage,
            path_candidate_count=self.qcast_config.path_candidate_count,
            construction_kind=self.qcast_config.construction_kind,
            purification_kind=self.qcast_config.purification_kind,
        )
        return record, record.solution

    def _decision(self, slot: int) -> None:
        self._expire_waiting_requests(slot)
        visible = self._visible_request_ids(slot)
        eligible = self._eligible_requests(slot)
        running = tuple(sorted(self._running_variables))
        start_window_end = min(
            self.spec.horizon,
            slot + self.qcast_config.decision_interval,
        )
        if start_window_end <= slot:
            return
        reserved = self._reserved_usage(self.spec.horizon)
        decision_started = perf_counter()
        planner_started = perf_counter()
        record, solution = self._solve_qcast_decision(
            slot,
            start_window_end,
            eligible,
            reserved,
        )
        planner_seconds = perf_counter() - planner_started
        selected_ids: tuple[str, ...] = ()
        selected_count = 0
        if solution is not None:
            self._register_selected_variables(
                slot,
                solution.selected_variables,
                solution.request_ids,
            )
            selected_ids = tuple(
                variable.variable_id
                for variable in solution.selected_variables
            )
            selected_count = solution.completed_request_count
        selected_requests = (
            set() if solution is None else set(solution.selected_by_request)
        )
        decision_seconds = perf_counter() - decision_started
        self._decisions.append(OnlineQCASTDecisionRecord(
            decision_slot=slot,
            window_end_slot=start_window_end,
            completion_end_slot=self.spec.horizon,
            visible_request_ids=visible,
            eligible_request_ids=eligible,
            running_request_ids=running,
            selected_variable_ids=selected_ids,
            deferred_request_ids=tuple(
                request_id
                for request_id in eligible
                if request_id not in selected_requests
            ),
            candidate_count=0 if record is None else len(record.candidates),
            variable_count=(
                0 if record is None else len(record.expansion.variables)
            ),
            candidate_rejection_count=(
                0 if record is None else len(record.expansion.rejections)
            ),
            reserved_resource_slot_count=len(reserved),
            selected_expected_throughput=(
                0.0 if record is None else record.selected_expected_throughput
            ),
            selected_request_count=selected_count,
            selected_total_completion_latency=(
                0.0 if solution is None else solution.total_completion_latency
            ),
            planner_seconds=planner_seconds,
            decision_seconds=decision_seconds,
        ))

    def _metrics(
        self,
        settlements: tuple[RequestSettlement, ...],
    ) -> dict[str, float]:
        metrics = super()._metrics(settlements)
        planner_times = [
            decision.planner_seconds
            for decision in self._decisions
            if decision.eligible_request_ids
        ]
        metrics["mean_qcast_planning_seconds"] = (
            0.0 if not planner_times else fmean(planner_times)
        )
        return metrics

    def run(self) -> OnlineQCASTResult:
        base = super().run()
        return OnlineQCASTResult(
            config=self.qcast_config,
            episode=base.episode,
            episode_seed=base.episode_seed,
            horizon_slots=base.horizon_slots,
            decisions=tuple(self._decisions),
            attempts=base.attempts,
            settlements=base.settlements,
            launches=base.launches,
            violations=base.violations,
            event_trace=base.event_trace,
            metrics=base.metrics,
        )


def run_online_qcast(
    spec: EpisodeSpec,
    config: OnlineQCASTConfig | None = None,
) -> OnlineQCASTResult:
    return OnlineQCASTController(spec, config).run()


__all__ = [
    "OnlineQCASTConfig",
    "OnlineQCASTController",
    "OnlineQCASTDecisionRecord",
    "OnlineQCASTResult",
    "run_online_qcast",
]
