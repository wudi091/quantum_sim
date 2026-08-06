"""Autodiff rollout trainer for construction-aware batch routing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qnet_core.construction_catalog import RouteConstructionCandidate
from qnet_core.construction_decoder import CapacityFeasibilityOracle
from qnet_core.joint_construction_gym import (
    JointConstructionBatchEnv,
    JointPhase,
    JointStep,
)
from qnet_core.spec import EpisodeSpec

from .policy import PolicySample
from .torch_policy import (
    TorchCAAPPOPolicy,
    TorchRouteRecord,
    TorchTransition,
    TorchUpdateStats,
    compute_gae,
)


@dataclass(frozen=True)
class TorchEpisodeTrainingResult:
    reward: float
    discounted_return: float
    metrics: dict[str, float]
    update_stats: TorchUpdateStats | None
    selected_candidates: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _Decision:
    sample: PolicySample
    reward: float
    risk_increment: float
    duration_ps: int


class TorchCAAPPORolloutTrainer:
    """Run masked PPO decisions through the SeQUeNCe-backed joint SMDP."""

    def __init__(
        self,
        policy: TorchCAAPPOPolicy | None = None,
        *,
        risk_limit: float = 0.0,
        gamma_per_slot: float = 1.0,
        gae_lambda: float = 0.95,
        alpha: float = 1.0,
        beta: float = 1.0,
        chi: float = 1.0,
        potential_shaping: bool = True,
        shaping_coef: float = 0.1,
    ):
        if not 0.0 < gamma_per_slot <= 1.0:
            raise ValueError("gamma_per_slot must lie in (0, 1]")
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gae_lambda must lie in [0, 1]")
        self.policy = policy or TorchCAAPPOPolicy()
        self.risk_limit = float(risk_limit)
        self.gamma_per_slot = float(gamma_per_slot)
        self.gae_lambda = float(gae_lambda)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.chi = float(chi)
        self.potential_shaping = bool(potential_shaping)
        self.shaping_coef = float(shaping_coef)

    @staticmethod
    def _potential(state: JointStep) -> float:
        if state.terminated or state.phase == JointPhase.TERMINAL:
            return 0.0
        if state.observation is None:
            return 0.0
        operations = {
            operation.op_id: operation
            for operation in state.observation.operations
        }
        settled_request_ids = {
            str(request_id)
            for request_id in state.info.get("settled_request_ids", ())
        }
        in_flight = {
            item.operation_id: item
            for item in state.observation.in_flight
        }
        critical_paths: list[float] = []
        for dag_state in state.observation.dag_states:
            if dag_state.request_id in settled_request_ids:
                continue
            if dag_state.request_id in state.observation.settled_request_ids:
                continue
            completed = set(dag_state.completed)
            dead = set(dag_state.dead)
            remaining = {
                operation_id
                for operation_id in dag_state.operation_ids
                if operation_id not in completed and operation_id not in dead
            }
            if not remaining:
                continue
            longest: dict[str, float] = {}
            pending = set(remaining)
            while pending:
                progressed = False
                for operation_id in tuple(sorted(pending)):
                    operation = operations.get(operation_id)
                    if operation is None:
                        longest[operation_id] = 0.0
                        pending.remove(operation_id)
                        progressed = True
                        continue
                    predecessors = [
                        predecessor for predecessor in operation.predecessors
                        if predecessor in remaining
                    ]
                    if any(predecessor not in longest for predecessor in predecessors):
                        continue
                    pending_operation = in_flight.get(operation_id)
                    duration = (
                        max(
                            0,
                            pending_operation.completion_time_ps
                            - state.observation.physical_time_ps,
                        )
                        if pending_operation is not None
                        else operation.duration_ps
                    )
                    longest[operation_id] = float(duration) + max(
                        (longest[predecessor] for predecessor in predecessors),
                        default=0.0,
                    )
                    pending.remove(operation_id)
                    progressed = True
                if not progressed:
                    raise RuntimeError("construction DAG contains a cycle")
            critical_paths.append(max(longest.values(), default=0.0))
        if not critical_paths:
            return 0.0
        horizon = max(float(state.observation.horizon_ps), 1.0)
        return -float(sum(min(path / horizon, 1.0) for path in critical_paths))

    def _shaped_reward(
        self,
        before: JointStep,
        after: JointStep,
        duration_ps: int,
        slot_duration_ps: int,
    ) -> float:
        if not self.potential_shaping:
            return float(after.reward)
        discount = self._discount(duration_ps, slot_duration_ps)
        return float(after.reward) + self.shaping_coef * (
            discount * self._potential(after) - self._potential(before)
        )

    @staticmethod
    def _admission_context(
        env: JointConstructionBatchEnv,
        state: JointStep,
        selected: dict[str, RouteConstructionCandidate],
        legal_candidates: tuple[RouteConstructionCandidate, ...],
        request_index: int,
        request_count: int,
    ) -> tuple[float, ...]:
        observation = state.info.get("admission_observation", {})
        preview_usage = dict(observation.get("preview_usage", ()))
        capacities = dict(env.admission_capacities)
        scale = max(1.0, float(sum(capacities.values())))
        request_id = tuple(sorted(env.admission_candidates))[request_index]
        return (
            float(request_index) / max(request_count, 1),
            float(len(selected)) / max(request_count, 1),
            float(sum(preview_usage.values())) / scale,
            float(len(preview_usage)) / max(len(capacities), 1),
            float(sum(candidate.hop_count for candidate in selected.values())),
            float(sum(len(candidate.dag.operations) for candidate in selected.values())),
            float(len(legal_candidates))
            / max(len(env.admission_candidates[request_id]), 1),
            float(request_index == request_count - 1),
        )

    def _select_routes(
        self,
        env: JointConstructionBatchEnv,
        *,
        deterministic: bool,
    ) -> tuple[JointStep, tuple[TorchRouteRecord, ...]]:
        state = env.reset()
        selected: dict[str, RouteConstructionCandidate] = {}
        samples = []
        request_order = tuple(sorted(env.admission_candidates))
        for request_index, request_id in enumerate(request_order):
            legal = env.legal_admission_candidates(request_id)
            if not legal:
                raise ValueError(
                    f"no legal admission candidate for request {request_id}"
                )
            context = self._admission_context(
                env,
                state,
                selected,
                legal,
                request_index,
                len(request_order),
            )
            sample = self.policy.sample_route(
                legal, context, deterministic=deterministic
            )
            selected[request_id] = sample.candidate
            samples.append((sample, tuple(legal)))
            state = env.select_admission(request_id, sample.candidate)
        records = tuple(
            TorchRouteRecord(
                legal,
                sample.index,
                sample.context,
                sample.log_probability,
                0.0,
            )
            for sample, legal in samples
        )
        return state, records

    def _discount(self, duration_ps: int, slot_duration_ps: int) -> float:
        slots = max(0.0, float(duration_ps) / max(slot_duration_ps, 1))
        return float(self.gamma_per_slot ** slots)

    @staticmethod
    def _risk_after(state: JointStep, previous: float) -> tuple[float, float]:
        current = float(state.info.get("risk_count", previous))
        if current + 1e-12 < previous:
            raise RuntimeError("episode risk count must be monotone")
        return current, current - previous

    def run_episode(
        self,
        spec: EpisodeSpec,
        candidates: tuple[RouteConstructionCandidate, ...],
        *,
        deterministic: bool = False,
        update: bool = True,
    ) -> TorchEpisodeTrainingResult:
        env = JointConstructionBatchEnv(
            spec,
            candidates,
            alpha=self.alpha,
            beta=self.beta,
            chi=self.chi,
        )
        state, route_records = self._select_routes(
            env, deterministic=deterministic
        )
        decisions: list[_Decision] = []
        cumulative_risk = 0.0

        while not state.terminated:
            if state.phase == JointPhase.REPAIR:
                before = state
                request_id = env.repairable_requests[0]
                choices = env.repair_choices(request_id)
                if state.observation is None:
                    raise RuntimeError("repair state lacks a construction observation")
                repair = self.policy.sample_repair(
                    state.observation, choices, deterministic=deterministic
                ).sample
                if repair.repair_action > 0 and choices:
                    option_index = repair.repair_action - 1
                    if option_index >= len(choices):
                        raise RuntimeError("repair policy selected an invalid option")
                    state = env.repair_choice(request_id, choices[option_index])
                else:
                    state = env.drop(request_id)
                reward = self._shaped_reward(
                    before, state, int(state.info.get("duration_ps", 0)),
                    spec.physical.slot_duration_ps,
                )
                cumulative_risk, increment = self._risk_after(
                    state, cumulative_risk
                )
                decisions.append(_Decision(
                    repair,
                    reward,
                    increment,
                    int(state.info.get("duration_ps", 0)),
                ))
                continue

            if state.observation is None:
                raise RuntimeError("execution state lacks a construction observation")
            oracle = CapacityFeasibilityOracle.from_snapshot(state.observation)
            before = state
            operation_sample = self.policy.sample_operation(
                state.observation,
                state.ready_operations,
                oracle,
                stop_legal=env.core.stop_legal(),
                deterministic=deterministic,
            ).sample
            operation_by_id = {
                operation.op_id: operation for operation in state.ready_operations
            }
            operations = tuple(
                operation_by_id[operation_id]
                for operation_id in operation_sample.action.operation_ids
            )
            state = env.step(operations)
            reward = self._shaped_reward(
                before, state, int(state.info.get("duration_ps", 0)),
                spec.physical.slot_duration_ps,
            )
            cumulative_risk, increment = self._risk_after(state, cumulative_risk)
            decisions.append(_Decision(
                operation_sample,
                reward,
                increment,
                int(state.info.get("duration_ps", 0)),
            ))

        metrics = env.metrics()
        episode_risk = float(metrics["risk_count"])
        if abs(cumulative_risk - episode_risk) > 1e-9:
            raise RuntimeError("transition risk accounting diverged from episode risk")

        transitions: tuple[TorchTransition, ...] = ()
        discounted_return = 0.0
        if decisions:
            values = np.asarray([
                self.policy.value_for_sample(decision.sample)
                for decision in decisions
            ], dtype=np.float64)
            risk_values = np.asarray([
                self.policy.constraint_for_sample(decision.sample)
                for decision in decisions
            ], dtype=np.float64)
            next_values = np.concatenate((values[1:], np.zeros(1)))
            next_risk_values = np.concatenate((risk_values[1:], np.zeros(1)))
            dones = tuple(
                index == len(decisions) - 1 for index in range(len(decisions))
            )
            discounts = np.asarray([
                self._discount(decision.duration_ps, spec.physical.slot_duration_ps)
                for decision in decisions
            ], dtype=np.float64)
            advantages, value_targets = compute_gae(
                [decision.reward for decision in decisions],
                values,
                next_values,
                dones,
                gae_lambda=self.gae_lambda,
                discounts=discounts,
            )
            risk_advantages, risk_targets = compute_gae(
                [decision.risk_increment for decision in decisions],
                risk_values,
                next_risk_values,
                dones,
                gamma=1.0,
                gae_lambda=self.gae_lambda,
                discounts=np.ones(len(decisions), dtype=np.float64),
            )
            running_discount = 1.0
            for decision, discount in zip(decisions, discounts):
                discounted_return += running_discount * decision.reward
                running_discount *= float(discount)
            transitions = tuple(
                TorchTransition(
                    decision.sample,
                    decision.sample.log_probability,
                    float(advantages[index]),
                    float(value_targets[index]),
                    float(risk_advantages[index]),
                    float(risk_targets[index]),
                    episode_risk,
                    float(np.prod(discounts[:index])) if index else 1.0,
                )
                for index, decision in enumerate(decisions)
            )

        route_records = tuple(
            TorchRouteRecord(
                record.candidates,
                record.index,
                record.context,
                record.old_log_probability,
                discounted_return,
            )
            for record in route_records
        )
        update_stats = None
        if update and not deterministic:
            update_stats = self.policy.update(
                transitions,
                route_records,
                risk_limit=self.risk_limit,
            )
        return TorchEpisodeTrainingResult(
            reward=float(sum(decision.reward for decision in decisions)),
            discounted_return=float(discounted_return),
            metrics=metrics,
            update_stats=update_stats,
            selected_candidates=tuple(
                (request_id, env.selected[request_id].candidate_id)
                for request_id in sorted(env.selected)
            ),
        )

    def train(
        self,
        spec: EpisodeSpec,
        candidates: tuple[RouteConstructionCandidate, ...],
        episodes: int,
    ) -> tuple[TorchEpisodeTrainingResult, ...]:
        if episodes < 1:
            raise ValueError("episodes must be positive")
        return tuple(
            self.run_episode(spec, candidates, deterministic=False, update=True)
            for _ in range(episodes)
        )


__all__ = ["TorchCAAPPORolloutTrainer", "TorchEpisodeTrainingResult"]
