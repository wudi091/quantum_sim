"""Joint admission/execution wrapper for construction-aware routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .construction_api import ConstructionOperation, ConstructionSnapshot
from .construction_catalog import (
    RouteConstructionCandidate,
    candidates_by_request,
)
from .construction_gym import ConstructionBatchEnv
from .spec import EpisodeSpec


class JointPhase:
    ADMISSION = "ADMISSION"
    EXECUTION = "EXECUTION"
    REPAIR = "REPAIR"
    TERMINAL = "TERMINAL"


@dataclass(frozen=True)
class JointStep:
    phase: str
    observation: ConstructionSnapshot | None
    ready_operations: tuple[ConstructionOperation, ...]
    admission_candidates: Mapping[str, tuple[RouteConstructionCandidate, ...]]
    reward: float
    terminated: bool
    info: Mapping[str, object]


class JointConstructionBatchEnv:
    """Event SMDP with an explicit joint ``(path, construction)`` admission.

    Admission is a single vector action selecting one catalogue candidate per
    request. Operation-set actions then run through the event-driven
    construction environment.
    """

    def __init__(
        self,
        spec: EpisodeSpec,
        candidates: Sequence[RouteConstructionCandidate],
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        chi: float = 1.0,
    ):
        self.spec = spec
        self._by_request = candidates_by_request(tuple(candidates))
        expected = {request.id for request in spec.requests}
        if set(self._by_request) != expected:
            raise ValueError("candidate catalogue must cover every request exactly")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.chi = float(chi)
        self.phase = JointPhase.ADMISSION
        self._core: ConstructionBatchEnv | None = None
        self.selected: dict[str, RouteConstructionCandidate] = {}
        self._repairable: set[str] = set()

    @property
    def core(self) -> ConstructionBatchEnv:
        if self._core is None:
            raise RuntimeError("admission has not been committed")
        return self._core

    @property
    def admission_candidates(self) -> Mapping[str, tuple[RouteConstructionCandidate, ...]]:
        return self._by_request

    def reset(self) -> JointStep:
        self.phase = JointPhase.ADMISSION
        self._core = None
        self.selected = {}
        self._repairable = set()
        return JointStep(
            JointPhase.ADMISSION,
            None,
            (),
            self._by_request,
            0.0,
            False,
            {"event_kind": "reset", "duration_ps": 0},
        )

    def admit(
        self,
        selection: Mapping[str, RouteConstructionCandidate | str],
    ) -> JointStep:
        if self.phase != JointPhase.ADMISSION:
            raise RuntimeError("admission is only legal in ADMISSION phase")
        expected = {request.id for request in self.spec.requests}
        if set(selection) != expected:
            raise ValueError("admission must select exactly one candidate per request")
        chosen: dict[str, RouteConstructionCandidate] = {}
        for request_id, value in selection.items():
            options = self._by_request[request_id]
            if isinstance(value, str):
                matches = [
                    candidate for candidate in options
                    if candidate.candidate_id == value
                ]
                if len(matches) != 1:
                    raise ValueError(f"unknown candidate for request {request_id}: {value}")
                candidate = matches[0]
            else:
                candidate = value
                if candidate not in options:
                    raise ValueError(f"candidate does not belong to request {request_id}")
            chosen[request_id] = candidate
        self.selected = chosen
        self._core = ConstructionBatchEnv(
            self.spec,
            chosen,
            alpha=self.alpha,
            beta=self.beta,
            chi=self.chi,
            auto_settle_failures=False,
        )
        core_step = self._core.reset()
        self.phase = JointPhase.TERMINAL if core_step.terminated else JointPhase.EXECUTION
        return JointStep(
            self.phase,
            core_step.observation,
            core_step.ready_operations,
            {},
            0.0,
            core_step.terminated,
            {
                "event_kind": "admission",
                "duration_ps": 0,
                "selected_candidates": tuple(
                    (request_id, chosen[request_id].candidate_id)
                    for request_id in sorted(chosen)
                ),
            },
        )

    def step(self, operations: Sequence[ConstructionOperation] = ()) -> JointStep:
        if self.phase == JointPhase.ADMISSION:
            raise RuntimeError("call admit() before execution step")
        if self.phase == JointPhase.TERMINAL:
            raise RuntimeError("joint environment is terminal")
        if self.phase == JointPhase.REPAIR:
            raise RuntimeError("submit repair() or drop() in REPAIR phase")
        core_step = self.core.step(operations)
        settled_ids = set(core_step.info.get("settled_request_ids", ()))
        self._repairable.difference_update(settled_ids)
        if core_step.terminated:
            self.phase = JointPhase.TERMINAL
        elif core_step.info.get("observed_failures_now", 0):
            self._repairable.update(
                request_id
                for request_id in core_step.info.get("observed_failure_request_ids", ())
                if request_id not in settled_ids
            )
            # A failed event can share a launch epoch with longer operations
            # from the same request.  Drain those events before exposing the
            # repair/drop action; drop never leaves a request in flight.
            self.phase = (
                JointPhase.EXECUTION
                if self.core.executor.has_in_flight
                else JointPhase.REPAIR if self._repairable else JointPhase.EXECUTION
            )
        elif self._repairable and not self.core.executor.has_in_flight:
            self.phase = JointPhase.REPAIR
        return JointStep(
            self.phase,
            core_step.observation,
            core_step.ready_operations,
            {},
            core_step.reward,
            core_step.terminated,
            core_step.info,
        )

    @property
    def repairable_requests(self) -> tuple[str, ...]:
        return tuple(sorted(self._repairable))

    def repair(
        self,
        request_id: str,
        operations: tuple[ConstructionOperation, ...],
    ) -> JointStep:
        if self.phase != JointPhase.REPAIR:
            raise RuntimeError("repair is only legal in REPAIR phase")
        if request_id not in self._repairable:
            raise ValueError("request is not awaiting repair")
        core_step = self.core.repair(request_id, operations)
        self._repairable.remove(request_id)
        self.phase = (
            JointPhase.REPAIR if self._repairable else JointPhase.EXECUTION
        )
        return JointStep(
            self.phase,
            core_step.observation,
            core_step.ready_operations,
            {},
            core_step.reward,
            False,
            core_step.info,
        )

    def drop(self, request_id: str) -> JointStep:
        if self.phase != JointPhase.REPAIR:
            raise RuntimeError("drop is only legal in REPAIR phase")
        if request_id not in self._repairable:
            raise ValueError("request is not awaiting repair")
        if any(
            pending.request_id == request_id
            for pending in self.core.executor.snapshot().in_flight
        ):
            raise RuntimeError("cannot drop a request with in-flight operations")
        core_step = self.core.drop(request_id)
        self._repairable.remove(request_id)
        self.phase = (
            JointPhase.TERMINAL
            if core_step.terminated
            else JointPhase.REPAIR if self._repairable else JointPhase.EXECUTION
        )
        return JointStep(
            self.phase,
            core_step.observation,
            core_step.ready_operations,
            {},
            core_step.reward,
            core_step.terminated,
            core_step.info,
        )

    def metrics(self) -> dict[str, float]:
        if self._core is None:
            return {
                "completed_requests": 0.0,
                "completion_rate": 0.0,
                "censored_flow_time_ps": 0.0,
                "risk_count": 0.0,
                "event_count": 0.0,
            }
        return self._core.metrics()
