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
        self._admission_order: tuple[str, ...] = tuple(
            sorted(request.id for request in spec.requests)
        )
        self._admission_index = 0
        self._admission_preview_usage: dict[str, int] = {}

    def _admission_capacities(self) -> dict[str, int]:
        capacities: dict[str, int] = {}
        for raw_u, raw_v in self.spec.edges:
            u, v = sorted((raw_u, raw_v))
            capacities[f"link:{u}-{v}"] = self.spec.physical.memory_capacity
            capacities[f"genlane:{u}-{v}"] = self.spec.physical.max_width
        for node in self.spec.nodes:
            capacities[f"bsm:{node}"] = 1
            degree = sum(node in edge for edge in self.spec.edges)
            capacities[f"memory:{node}"] = (
                self.spec.physical.node_memory_capacity
                if self.spec.physical.node_memory_capacity is not None
                else max(1, degree * self.spec.physical.memory_capacity)
            )
        return capacities

    @staticmethod
    def _candidate_footprint(candidate: RouteConstructionCandidate) -> dict[str, int]:
        """Conservative per-resource footprint used only for admission masking."""

        footprint: dict[str, int] = {}
        for operation in candidate.dag.operations:
            for resource, amount in operation.resource_demand.items():
                footprint[resource] = max(footprint.get(resource, 0), amount)
            for resource, amount in operation.output_resource_hold.items():
                footprint[resource] = max(footprint.get(resource, 0), amount)
        return footprint

    @staticmethod
    def _candidate_has_feasible_schedule(
        candidate: RouteConstructionCandidate,
        capacities: Mapping[str, int],
    ) -> bool:
        """Check whether one sequential topological execution fits capacity.

        Admission does not reserve resources across requests, but each chosen
        construction DAG must be intrinsically executable. This search tracks
        resident output holds, transient launch demand, and input consumption;
        it catches cases such as a two-hop path with only one memory at the
        middle node without rejecting later time sharing between requests.
        """

        operations = tuple(candidate.dag.operations)
        memo: set[tuple[frozenset[str], tuple[tuple[str, tuple[tuple[str, int], ...]], ...]]] = set()

        def visit(
            completed: frozenset[str],
            live_holds: tuple[tuple[str, tuple[tuple[str, int], ...]], ...],
        ) -> bool:
            if len(completed) == len(operations):
                return True
            key = (completed, live_holds)
            if key in memo:
                return False
            memo.add(key)
            live = {segment_id: dict(entries) for segment_id, entries in live_holds}
            usage: dict[str, int] = {}
            for hold in live.values():
                for resource, amount in hold.items():
                    usage[resource] = usage.get(resource, 0) + amount
            available = set(live)
            ready = tuple(sorted(
                (
                    operation
                    for operation in operations
                    if operation.op_id not in completed
                    and set(operation.predecessors).issubset(completed)
                    and set(operation.input_segment_ids).issubset(available)
                ),
                key=lambda operation: operation.canonical_key,
            ))
            for operation in ready:
                launch_usage = dict(usage)
                for resource, amount in operation.resource_demand.items():
                    launch_usage[resource] = launch_usage.get(resource, 0) + amount
                if any(
                    amount > capacities.get(resource, 0)
                    for resource, amount in launch_usage.items()
                ):
                    continue
                next_live = {segment_id: dict(values) for segment_id, values in live.items()}
                for segment_id in operation.input_segment_ids:
                    next_live.pop(segment_id, None)
                if operation.output_segment_id is not None:
                    next_live[operation.output_segment_id] = dict(
                        operation.output_resource_hold.items()
                    )
                post_usage: dict[str, int] = {}
                for hold in next_live.values():
                    for resource, amount in hold.items():
                        post_usage[resource] = post_usage.get(resource, 0) + amount
                if any(
                    amount > capacities.get(resource, 0)
                    for resource, amount in post_usage.items()
                ):
                    continue
                normalized = tuple(sorted(
                    (segment_id, tuple(sorted(values.items())))
                    for segment_id, values in next_live.items()
                ))
                if visit(completed | {operation.op_id}, normalized):
                    return True
            return False

        return visit(frozenset(), ())

    def _admission_observation(self) -> dict[str, object]:
        next_request = (
            self._admission_order[self._admission_index]
            if self._admission_index < len(self._admission_order)
            else None
        )
        legal = {
            request_id: tuple(candidate.candidate_id for candidate in values)
            for request_id, values in self._by_request.items()
        }
        if next_request is not None:
            legal[next_request] = tuple(
                candidate.candidate_id
                for candidate in self.legal_admission_candidates(next_request)
            )
        return {
            "request_order": self._admission_order,
            "next_request_id": next_request,
            "admission_index": self._admission_index,
            "selected_candidate_ids": tuple(
                (request_id, self.selected[request_id].candidate_id)
                for request_id in self._admission_order
                if request_id in self.selected
            ),
            "preview_usage": tuple(sorted(self._admission_preview_usage.items())),
            "legal_candidate_ids": tuple(
                (request_id, tuple(candidate_ids))
                for request_id, candidate_ids in sorted(legal.items())
            ),
        }

    def legal_admission_candidates(
        self, request_id: str
    ) -> tuple[RouteConstructionCandidate, ...]:
        if request_id not in self._by_request:
            raise KeyError(request_id)
        capacities = self._admission_capacities()
        legal = []
        for candidate in self._by_request[request_id]:
            footprint = self._candidate_footprint(candidate)
            # Admission selects plans but does not reserve execution-time
            # resources. Requests may reuse the same unit-capacity link after
            # an earlier request settles, so cumulative preview usage is a
            # policy context feature rather than a residual-capacity mask.
            if all(
                amount <= capacities.get(resource, 0)
                for resource, amount in footprint.items()
            ) and self._candidate_has_feasible_schedule(candidate, capacities):
                legal.append(candidate)
        return tuple(legal)

    @property
    def core(self) -> ConstructionBatchEnv:
        if self._core is None:
            raise RuntimeError("admission has not been committed")
        return self._core

    @property
    def admission_candidates(self) -> Mapping[str, tuple[RouteConstructionCandidate, ...]]:
        return self._by_request

    @property
    def admission_capacities(self) -> Mapping[str, int]:
        """Neutral resource capacities used by the admission mask."""

        return self._admission_capacities()

    def reset(self) -> JointStep:
        self.phase = JointPhase.ADMISSION
        self._core = None
        self.selected = {}
        self._repairable = set()
        self._admission_index = 0
        self._admission_preview_usage = {}
        return JointStep(
            JointPhase.ADMISSION,
            None,
            (),
            self._by_request,
            0.0,
            False,
            {
                "event_kind": "reset",
                "duration_ps": 0,
                "admission_observation": self._admission_observation(),
            },
        )

    def select_admission(
        self,
        request_id: str,
        value: RouteConstructionCandidate | str,
    ) -> JointStep:
        """Commit one autoregressive route/construction choice."""

        if self.phase != JointPhase.ADMISSION:
            raise RuntimeError("admission selection is only legal in ADMISSION phase")
        if self._admission_index >= len(self._admission_order):
            raise RuntimeError("all admission choices have already been selected")
        expected_request = self._admission_order[self._admission_index]
        if request_id != expected_request:
            raise ValueError(
                f"admission choices must follow canonical order; expected {expected_request}"
            )
        options = self._by_request[request_id]
        if isinstance(value, str):
            matches = [candidate for candidate in options if candidate.candidate_id == value]
            if len(matches) != 1:
                raise ValueError(f"unknown candidate for request {request_id}: {value}")
            candidate = matches[0]
        else:
            candidate = value
            if candidate not in options:
                raise ValueError(f"candidate does not belong to request {request_id}")
        if candidate not in self.legal_admission_candidates(request_id):
            raise ValueError(
                f"candidate is infeasible under current admission preview: {candidate.candidate_id}"
            )
        self.selected[request_id] = candidate
        for resource, amount in self._candidate_footprint(candidate).items():
            self._admission_preview_usage[resource] = (
                self._admission_preview_usage.get(resource, 0) + amount
            )
        self._admission_index += 1
        if self._admission_index < len(self._admission_order):
            return JointStep(
                JointPhase.ADMISSION,
                None,
                (),
                self._by_request,
                0.0,
                False,
                {
                    "event_kind": "admission_choice",
                    "duration_ps": 0,
                    "selected_candidate": candidate.candidate_id,
                    "admission_observation": self._admission_observation(),
                },
            )
        return self._commit_admission()

    def _commit_admission(self) -> JointStep:
        self._core = ConstructionBatchEnv(
            self.spec,
            self.selected,
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
                    (request_id, self.selected[request_id].candidate_id)
                    for request_id in self._admission_order
                ),
            },
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
        self.selected = {}
        self._admission_index = 0
        self._admission_preview_usage = {}
        state = self.reset()
        for request_id in self._admission_order:
            state = self.select_admission(request_id, selection[request_id])
        return state

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
        # Expiration is a physical event boundary.  A request may still have
        # an unrelated operation in flight from the same launch epoch; drain
        # that suffix before exposing REPAIR/DROP, because repair is only
        # legal on a quiescent request prefix.
        expiration_request_ids = tuple(
            core_step.info.get("observed_expiration_request_ids", ())
        )
        if (
            expiration_request_ids
            and self.core.executor.has_in_flight
            and not core_step.terminated
        ):
            while self.core.executor.has_in_flight:
                core_step = self.core.step(())
                settled_ids.update(
                    core_step.info.get("settled_request_ids", ())
                )
                self._repairable.difference_update(settled_ids)
        if core_step.terminated:
            self.phase = JointPhase.TERMINAL
        elif core_step.info.get("observed_failures_now", 0) or expiration_request_ids:
            self._repairable.update(
                request_id
                for request_id in tuple(core_step.info.get(
                    "observed_failure_request_ids", ()
                )) + expiration_request_ids
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

    def repair_options(
        self, request_id: str
    ) -> tuple[tuple[ConstructionOperation, ...], ...]:
        if self.phase != JointPhase.REPAIR:
            raise RuntimeError("repair options are only available in REPAIR phase")
        if request_id not in self._repairable:
            raise ValueError("request is not awaiting repair")
        return self.core.repair_options(request_id)

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
                "delivered_pairs": 0.0,
                "completion_rate": 0.0,
                "censored_flow_time_ps": 0.0,
                "risk_count": 0.0,
                "event_count": 0.0,
            }
        return self._core.metrics()
