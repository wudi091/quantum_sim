"""Interactive construction-aware routing environment.

The environment is deliberately policy-agnostic.  It exposes a neutral
planning observation, validates an autoregressive sequence against residual
resource--time capacities, and delegates the accepted schedule to the shared
SeQUeNCe-backed executor.  It never imports a SeQUeNCe class.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Mapping

from algorithms.routing_core.candidates import (
    PlanningBatchProblem,
    build_planning_batch_problem,
)
from algorithms.routing_core.execution import (
    OnlineDecisionRecord,
    OnlineExecutionConfig,
    OnlineExecutionController,
    OnlineExecutionResult,
)
from algorithms.routing_core.time_expansion import TimeExpandedCandidate
from qnet_core.construction_api import ConstructionSnapshot
from qnet_core.planning_spec import RequestSpec
from qnet_core.spec import EpisodeSpec


STOP_ACTION = "__stop__"


@dataclass(frozen=True)
class RoutingObservation:
    """One decision-epoch state exposed to a routing policy."""

    episode_seed: int
    decision_index: int
    slot: int
    horizon_slots: int
    window_end_slot: int
    completion_end_slot: int
    nodes: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    visible_requests: tuple[RequestSpec, ...]
    visible_request_ids: tuple[str, ...]
    eligible_request_ids: tuple[str, ...]
    running_request_ids: tuple[str, ...]
    completed_request_ids: tuple[str, ...]
    expired_request_ids: tuple[str, ...]
    resource_capacities: tuple[tuple[str, int], ...]
    reserved_usage: tuple[tuple[str, int, int], ...]
    physical_snapshot: ConstructionSnapshot
    problem: PlanningBatchProblem | None

    @property
    def variables(self) -> tuple[TimeExpandedCandidate, ...]:
        if self.problem is None:
            return ()
        return self.problem.expansion.variables

    @property
    def capacities(self) -> dict[str, int]:
        return dict(self.resource_capacities)

    @property
    def reserved_usage_map(self) -> dict[tuple[str, int], int]:
        return {
            (resource_id, slot): amount
            for resource_id, slot, amount in self.reserved_usage
        }


@dataclass(frozen=True)
class RoutingAction:
    """A complete autoregressive plan terminated by ``STOP_ACTION``."""

    decision_slot: int
    action_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.decision_slot < 0:
            raise ValueError("decision_slot cannot be negative")
        if not self.action_ids or self.action_ids[-1] != STOP_ACTION:
            raise ValueError("a routing action must end with STOP")
        if STOP_ACTION in self.action_ids[:-1]:
            raise ValueError("STOP can only appear at the end of an action")
        variable_ids = self.action_ids[:-1]
        if len(set(variable_ids)) != len(variable_ids):
            raise ValueError("a variable cannot be selected twice")

    @property
    def variable_ids(self) -> tuple[str, ...]:
        return self.action_ids[:-1]


@dataclass(frozen=True)
class RoutingTransition:
    """One physical decision interval and its delay-aligned reward."""

    observation: RoutingObservation
    action: RoutingAction
    reward: float
    delay_area_slots: float
    terminal_censoring_slots: float
    completed_request_ids: tuple[str, ...]
    expired_request_ids: tuple[str, ...]
    failed_attempt_request_ids: tuple[str, ...]
    next_slot: int
    done: bool


class FeasiblePlanBuilder:
    """Stateful action mask for one autoregressive planning decision.

    The builder applies only hard feasibility conditions: one plan per request
    and every opaque resource--slot capacity.  It does not rank, repair, or
    greedily add actions on behalf of the policy.
    """

    def __init__(self, observation: RoutingObservation):
        self.observation = observation
        self._variables = {
            variable.variable_id: variable
            for variable in observation.variables
        }
        if len(self._variables) != len(observation.variables):
            raise ValueError("observation contains duplicate variable IDs")
        self._capacities = observation.capacities
        self._usage = observation.reserved_usage_map
        self._selected_ids: list[str] = []
        self._selected_requests: set[str] = set()
        self._stopped = False

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def selected_variable_ids(self) -> tuple[str, ...]:
        return tuple(self._selected_ids)

    @property
    def selected_variables(self) -> tuple[TimeExpandedCandidate, ...]:
        return tuple(self._variables[item] for item in self._selected_ids)

    @property
    def current_usage(self) -> Mapping[tuple[str, int], int]:
        return dict(self._usage)

    def remaining_capacity(self, resource_id: str, slot: int) -> int:
        if resource_id not in self._capacities:
            raise KeyError(f"unknown resource: {resource_id}")
        return self._capacities[resource_id] - self._usage.get(
            (resource_id, int(slot)), 0
        )

    @staticmethod
    def _usage_delta(
        variable: TimeExpandedCandidate,
    ) -> dict[tuple[str, int], int]:
        delta: dict[tuple[str, int], int] = {}
        for item in variable.resource_usage:
            key = (item.resource_id, item.slot)
            delta[key] = delta.get(key, 0) + item.amount
        return delta

    def can_select(self, variable_id: str) -> bool:
        if self._stopped:
            return False
        variable = self._variables.get(variable_id)
        if variable is None or variable.request_id in self._selected_requests:
            return False
        for (resource_id, slot), amount in self._usage_delta(variable).items():
            capacity = self._capacities.get(resource_id)
            if capacity is None:
                raise ValueError(f"missing capacity: {resource_id}")
            if self._usage.get((resource_id, slot), 0) + amount > capacity:
                return False
        return True

    def legal_action_ids(self) -> tuple[str, ...]:
        if self._stopped:
            return ()
        legal_variables = tuple(
            variable_id
            for variable_id in sorted(self._variables)
            if self.can_select(variable_id)
        )
        return (*legal_variables, STOP_ACTION)

    def select(self, action_id: str) -> None:
        if self._stopped:
            raise RuntimeError("the autoregressive plan already stopped")
        if action_id == STOP_ACTION:
            self._stopped = True
            return
        if action_id not in self._variables:
            raise KeyError(f"unknown routing action: {action_id}")
        if not self.can_select(action_id):
            raise ValueError(f"infeasible routing action: {action_id}")
        variable = self._variables[action_id]
        self._selected_ids.append(action_id)
        self._selected_requests.add(variable.request_id)
        for key, amount in self._usage_delta(variable).items():
            self._usage[key] = self._usage.get(key, 0) + amount

    def apply(self, action: RoutingAction) -> None:
        if action.decision_slot != self.observation.slot:
            raise ValueError("routing action was produced for a stale slot")
        for action_id in action.action_ids:
            self.select(action_id)
        if not self._stopped:
            raise ValueError("routing action did not terminate")

    def finish(self) -> RoutingAction:
        if not self._stopped:
            self.select(STOP_ACTION)
        return RoutingAction(
            decision_slot=self.observation.slot,
            action_ids=(*self._selected_ids, STOP_ACTION),
        )


class ConstructionAwareRoutingEnvironment(OnlineExecutionController):
    """Step-wise ARC-Q environment backed by one persistent physical run."""

    def __init__(
        self,
        spec: EpisodeSpec,
        config: OnlineExecutionConfig | None = None,
    ) -> None:
        super().__init__(spec, config)
        self._decision_index = 0
        self._cached_observation: RoutingObservation | None = None
        self._transitions: list[RoutingTransition] = []
        self._done = False
        self._final_result: OnlineExecutionResult | None = None

    def _decision(self, slot: int) -> None:
        raise RuntimeError(
            "use observe() and step(); the RL environment is interactive"
        )

    @property
    def done(self) -> bool:
        return self._done

    @property
    def transitions(self) -> tuple[RoutingTransition, ...]:
        return tuple(self._transitions)

    def observe(self) -> RoutingObservation:
        if self._done:
            raise RuntimeError("the episode is already complete")
        if self._cached_observation is not None:
            return self._cached_observation

        slot = self.scheduler.current_slot
        self._expire_waiting_requests(slot)
        visible = self._visible_request_ids(slot)
        eligible = self._eligible_requests(slot)
        running = tuple(sorted(self._running_variables))
        window_end = min(
            self.spec.horizon,
            slot + self.config.decision_interval,
        )
        reserved = self._reserved_usage(self.spec.horizon)
        problem: PlanningBatchProblem | None = None
        if eligible:
            eligible_set = set(eligible)
            planning_episode = replace(
                self.spec,
                requests=tuple(
                    request
                    for request in self.spec.requests
                    if request.id in eligible_set
                ),
            )
            problem = build_planning_batch_problem(
                planning_episode,
                path_candidate_count=self.config.path_candidate_count,
                construction_kinds=self.config.construction_kinds,
                swap_tree_count=self.config.swap_tree_count,
                purification_kinds=self.config.purification_kinds,
                resource_capacities=self.capacities,
                reserved_usage=reserved,
                window_start_slot=slot,
                window_end_slot=window_end,
                completion_end_slot=self.spec.horizon,
            )

        observation = RoutingObservation(
            episode_seed=self.spec.seed,
            decision_index=self._decision_index,
            slot=slot,
            horizon_slots=self.spec.horizon,
            window_end_slot=window_end,
            completion_end_slot=self.spec.horizon,
            nodes=self.spec.nodes,
            edges=self.spec.edges,
            visible_requests=tuple(sorted(
                (
                    request
                    for request in self.spec.requests
                    if request.id in set(visible)
                ),
                key=lambda request: request.id,
            )),
            visible_request_ids=visible,
            eligible_request_ids=eligible,
            running_request_ids=running,
            completed_request_ids=self.scheduler.completed_request_ids,
            expired_request_ids=tuple(sorted(self._expired_times)),
            resource_capacities=tuple(sorted(self.capacities.items())),
            reserved_usage=tuple(sorted(
                (resource_id, usage_slot, amount)
                for (resource_id, usage_slot), amount in reserved.items()
            )),
            physical_snapshot=self.scheduler.executor.snapshot(),
            problem=problem,
        )
        self._cached_observation = observation
        return observation

    def _delay_cost(
        self,
        start_time_ps: int,
        end_time_ps: int,
        newly_expired: tuple[str, ...],
    ) -> tuple[float, float]:
        """Return holding area and terminal censoring charge in slot units."""

        slot_duration = self.spec.physical.slot_duration_ps
        horizon_ps = self.horizon_ps
        completed = self.scheduler.completed_times
        delay_area_ps = 0
        for request in self.spec.requests:
            arrival_ps = request.arrival * slot_duration
            terminal_time = completed.get(request.id)
            if terminal_time is None:
                terminal_time = self._expired_times.get(request.id)
            active_start = max(start_time_ps, arrival_ps)
            active_end = end_time_ps
            if terminal_time is not None:
                active_end = min(active_end, terminal_time)
            if active_end > active_start:
                delay_area_ps += active_end - active_start

        terminal_charge_ps = sum(
            max(0, horizon_ps - self._expired_times[request_id])
            for request_id in newly_expired
        )
        return (
            delay_area_ps / slot_duration,
            terminal_charge_ps / slot_duration,
        )

    def _finish_episode(self) -> None:
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
        self._final_result = OnlineExecutionResult(
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

    def step(
        self,
        action: RoutingAction,
        *,
        planner_seconds: float = 0.0,
    ) -> RoutingTransition:
        if self._done:
            raise RuntimeError("the episode is already complete")
        if planner_seconds < 0.0:
            raise ValueError("planner_seconds cannot be negative")
        observation = self.observe()
        decision_started = perf_counter()
        builder = FeasiblePlanBuilder(observation)
        builder.apply(action)
        selected = builder.selected_variables
        if selected:
            self._register_selected_variables(
                observation.slot,
                selected,
                observation.eligible_request_ids,
            )

        selected_requests = {variable.request_id for variable in selected}
        validation_seconds = perf_counter() - decision_started
        self._decisions.append(OnlineDecisionRecord(
            decision_slot=observation.slot,
            window_end_slot=observation.window_end_slot,
            completion_end_slot=observation.completion_end_slot,
            visible_request_ids=observation.visible_request_ids,
            eligible_request_ids=observation.eligible_request_ids,
            running_request_ids=observation.running_request_ids,
            selected_variable_ids=builder.selected_variable_ids,
            deferred_request_ids=tuple(
                request_id
                for request_id in observation.eligible_request_ids
                if request_id not in selected_requests
            ),
            candidate_count=(
                0
                if observation.problem is None
                else len(observation.problem.candidates)
            ),
            variable_count=len(observation.variables),
            candidate_rejection_count=(
                0
                if observation.problem is None
                else len(observation.problem.expansion.rejections)
            ),
            reserved_resource_slot_count=len(observation.reserved_usage),
            selected_request_count=len(selected_requests),
            selected_expected_completed_mass=sum(
                variable.expected_success_probability for variable in selected
            ),
            selected_expected_completion_latency=sum(
                variable.expected_success_probability
                * variable.completion_latency
                for variable in selected
            ),
            planner_seconds=planner_seconds,
            decision_seconds=planner_seconds + validation_seconds,
            selection_strategy="arcq_autoregressive_policy",
        ))

        start_time_ps = self.scheduler.physical_time_ps
        completed_before = set(self.scheduler.completed_request_ids)
        expired_before = set(self._expired_times)
        update = self.scheduler.advance_to_slot(observation.window_end_slot)
        self._process_update(update)
        self._expire_waiting_requests(observation.window_end_slot)
        end_time_ps = self.scheduler.physical_time_ps

        newly_completed = tuple(sorted(
            set(self.scheduler.completed_request_ids) - completed_before
        ))
        newly_expired = tuple(sorted(set(self._expired_times) - expired_before))
        failed_attempts = tuple(sorted({
            outcome.request_id
            for outcome in update.outcomes
            if not outcome.success
        }))
        delay_area, terminal_charge = self._delay_cost(
            start_time_ps,
            end_time_ps,
            newly_expired,
        )
        request_count = max(1, len(self.spec.requests))
        reward = -(delay_area + terminal_charge) / request_count
        self._done = self.scheduler.current_slot >= self.spec.horizon
        transition = RoutingTransition(
            observation=observation,
            action=action,
            reward=float(reward),
            delay_area_slots=float(delay_area),
            terminal_censoring_slots=float(terminal_charge),
            completed_request_ids=newly_completed,
            expired_request_ids=newly_expired,
            failed_attempt_request_ids=failed_attempts,
            next_slot=self.scheduler.current_slot,
            done=self._done,
        )
        self._transitions.append(transition)
        self._decision_index += 1
        self._cached_observation = None
        if self._done:
            self._finish_episode()
        return transition

    def result(self) -> OnlineExecutionResult:
        if not self._done or self._final_result is None:
            raise RuntimeError("the episode has not finished")
        return self._final_result

    def reward_identity_error(self) -> float:
        """Difference between return and evaluator mean censored latency."""

        result = self.result()
        measured_slots = (
            result.metrics["mean_censored_latency_ps"]
            / self.spec.physical.slot_duration_ps
        )
        return float(measured_slots + sum(
            transition.reward for transition in self._transitions
        ))


__all__ = [
    "STOP_ACTION",
    "ConstructionAwareRoutingEnvironment",
    "FeasiblePlanBuilder",
    "RoutingAction",
    "RoutingObservation",
    "RoutingTransition",
]
