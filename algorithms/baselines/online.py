"""Rolling-horizon non-learning baselines on the shared executor.

Only the planning rule changes between algorithms.  Request visibility,
resource reservations, retries, and physical execution are inherited from
the same persistent SeQUeNCe-backed lifecycle used by TELGEN and Q-CAST.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
import csv
import json
from pathlib import Path
import shutil
from statistics import fmean
from time import perf_counter
from typing import Mapping

from algorithms.telgen.online import (
    OnlineAttemptRecord,
    OnlineTELGENConfig,
    OnlineTELGENController,
)
from algorithms.telgen.packing import PackingSolution
from qnet_core.construction_api import ExecutionEvent
from qnet_core.construction_metrics import RequestSettlement
from qnet_core.scheduled_execution import (
    ScheduleViolation,
    ScheduledOperationLaunch,
)
from qnet_core.spec import EpisodeSpec

from .planner import (
    BASELINE_ALGORITHMS,
    BaselinePlannerState,
    BaselinePlanningRecord,
    plan_baseline_window,
)


class _DisabledMILPOracle:
    """Fail closed if a future refactor accidentally enters the MILP path."""

    def solve(self, *args, **kwargs):
        raise RuntimeError("non-learning baselines must not invoke MILP")


@dataclass(frozen=True)
class OnlineBaselineConfig:
    """Online settings shared by all non-learning planning rules."""

    algorithm: str = "greedy"
    decision_interval: int = 4
    path_candidate_count: int = 4
    construction_kind: str = "left_deep"

    def __post_init__(self) -> None:
        if self.algorithm not in BASELINE_ALGORITHMS:
            raise ValueError(f"unknown baseline algorithm: {self.algorithm}")
        if self.decision_interval < 1:
            raise ValueError("decision_interval must be positive")
        if self.path_candidate_count < 1:
            raise ValueError("path_candidate_count must be positive")
        if self.construction_kind not in {"left_deep", "balanced"}:
            raise ValueError("unsupported construction_kind")


@dataclass(frozen=True)
class OnlineBaselineDecisionRecord:
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
    considered_variable_count: int
    candidate_rejection_count: int
    reserved_resource_slot_count: int
    selected_request_count: int
    selected_total_completion_latency: float
    selected_score: float
    planner_seconds: float
    decision_seconds: float
    planner_state_average_before: float | None = None
    planner_state_average_after: float | None = None


@dataclass(frozen=True)
class OnlineBaselineResult:
    config: OnlineBaselineConfig
    episode: EpisodeSpec
    episode_seed: int
    horizon_slots: int
    decisions: tuple[OnlineBaselineDecisionRecord, ...]
    attempts: tuple[OnlineAttemptRecord, ...]
    settlements: tuple[RequestSettlement, ...]
    launches: tuple[ScheduledOperationLaunch, ...]
    violations: tuple[ScheduleViolation, ...]
    event_trace: tuple[ExecutionEvent, ...]
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class OnlineBaselineResultPaths:
    json_path: Path
    csv_path: Path
    latest_json_path: Path
    latest_csv_path: Path


class OnlineBaselineController(OnlineTELGENController):
    """Run one named baseline without invoking MILP or a learned policy."""

    def __init__(
        self,
        spec: EpisodeSpec,
        config: OnlineBaselineConfig | None = None,
    ) -> None:
        self.baseline_config = config or OnlineBaselineConfig()
        purification_kinds = (
            ("none", "elementary_once")
            if self.baseline_config.algorithm in {"qpath", "qleap"}
            else ("none",)
        )
        super().__init__(
            spec,
            OnlineTELGENConfig(
                decision_interval=self.baseline_config.decision_interval,
                path_candidate_count=(
                    self.baseline_config.path_candidate_count
                ),
                construction_kinds=(
                    self.baseline_config.construction_kind,
                ),
                purification_kinds=purification_kinds,
            ),
            milp_oracle=_DisabledMILPOracle(),
        )
        self._planner_state = BaselinePlannerState()
        self._decisions: list[OnlineBaselineDecisionRecord] = []

    def _solve_baseline_decision(
        self,
        slot: int,
        start_window_end_slot: int,
        eligible_request_ids: tuple[str, ...],
        reserved_usage: Mapping[tuple[str, int], int],
    ) -> tuple[BaselinePlanningRecord | None, PackingSolution | None]:
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
        record = plan_baseline_window(
            window_episode,
            algorithm=self.baseline_config.algorithm,
            window_start_slot=slot,
            window_end_slot=start_window_end_slot,
            completion_end_slot=self.spec.horizon,
            request_ids=eligible_request_ids,
            resource_capacities=self.capacities,
            reserved_usage=reserved_usage,
            path_candidate_count=(
                self.baseline_config.path_candidate_count
            ),
            construction_kind=self.baseline_config.construction_kind,
            planner_state=self._planner_state,
        )
        self._planner_state = record.state_after
        return record, record.solution

    def _decision(self, slot: int) -> None:
        self._expire_waiting_requests(slot)
        visible = self._visible_request_ids(slot)
        eligible = self._eligible_requests(slot)
        running = tuple(sorted(self._running_variables))
        start_window_end = min(
            self.spec.horizon,
            slot + self.baseline_config.decision_interval,
        )
        if start_window_end <= slot:
            return
        reserved = self._reserved_usage(self.spec.horizon)
        decision_started = perf_counter()
        planner_started = perf_counter()
        record, solution = self._solve_baseline_decision(
            slot,
            start_window_end,
            eligible,
            reserved,
        )
        planner_seconds = perf_counter() - planner_started
        selected_variables = (
            () if solution is None else solution.selected_variables
        )
        if solution is not None:
            self._register_selected_variables(
                slot,
                selected_variables,
                solution.request_ids,
            )
        selected_requests = {
            variable.request_id for variable in selected_variables
        }
        decision_seconds = perf_counter() - decision_started
        self._decisions.append(OnlineBaselineDecisionRecord(
            decision_slot=slot,
            window_end_slot=start_window_end,
            completion_end_slot=self.spec.horizon,
            visible_request_ids=visible,
            eligible_request_ids=eligible,
            running_request_ids=running,
            selected_variable_ids=tuple(
                variable.variable_id for variable in selected_variables
            ),
            deferred_request_ids=tuple(
                request_id
                for request_id in eligible
                if request_id not in selected_requests
            ),
            candidate_count=(
                0 if record is None else len(record.problem.candidates)
            ),
            variable_count=(
                0
                if record is None
                else len(record.problem.expansion.variables)
            ),
            considered_variable_count=(
                0 if record is None else len(record.considered_variables)
            ),
            candidate_rejection_count=(
                0
                if record is None
                else len(record.problem.expansion.rejections)
            ),
            reserved_resource_slot_count=len(reserved),
            selected_request_count=len(selected_requests),
            selected_total_completion_latency=(
                0.0
                if solution is None
                else solution.total_completion_latency
            ),
            selected_score=(
                0.0 if record is None else record.selected_score
            ),
            planner_seconds=planner_seconds,
            decision_seconds=decision_seconds,
            planner_state_average_before=(
                None
                if record is None
                else record.state_before.average_path_cost
            ),
            planner_state_average_after=(
                None
                if record is None
                else record.state_after.average_path_cost
            ),
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
        metrics["mean_baseline_planning_seconds"] = (
            0.0 if not planner_times else fmean(planner_times)
        )
        return metrics

    def run(self) -> OnlineBaselineResult:
        base = super().run()
        return OnlineBaselineResult(
            config=self.baseline_config,
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


def run_online_baseline(
    spec: EpisodeSpec,
    config: OnlineBaselineConfig | None = None,
) -> OnlineBaselineResult:
    return OnlineBaselineController(spec, config).run()


def save_online_baseline_result(
    result: OnlineBaselineResult,
    output_directory: str | Path,
) -> OnlineBaselineResultPaths:
    """Save one baseline run as versioned JSON/CSV and latest copies."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"online_{result.config.algorithm}_{timestamp}"
    suffix = ""
    collision_index = 1
    while (output / f"{stem}{suffix}.json").exists() or (
        output / f"{stem}{suffix}.csv"
    ).exists():
        collision_index += 1
        suffix = f"_{collision_index}"
    json_path = output / f"{stem}{suffix}.json"
    csv_path = output / f"{stem}{suffix}.csv"
    latest_json = output / f"online_{result.config.algorithm}.json"
    latest_csv = output / f"online_{result.config.algorithm}.csv"
    payload = {
        "schema_version": 1,
        "method": result.config.algorithm,
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
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
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
                "latency_ps": (
                    settlement.settlement_time - settlement.arrival_time
                ),
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
    return OnlineBaselineResultPaths(
        json_path=json_path,
        csv_path=csv_path,
        latest_json_path=latest_json,
        latest_csv_path=latest_csv,
    )


__all__ = [
    "OnlineBaselineConfig",
    "OnlineBaselineController",
    "OnlineBaselineDecisionRecord",
    "OnlineBaselineResult",
    "OnlineBaselineResultPaths",
    "run_online_baseline",
    "save_online_baseline_result",
]
