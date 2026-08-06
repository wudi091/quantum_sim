"""Reference rollout loop for the event-driven CAAPPO policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from qnet_core.construction_catalog import RouteConstructionCandidate, candidates_by_request
from qnet_core.construction_decoder import CapacityFeasibilityOracle
from qnet_core.joint_construction_gym import JointConstructionBatchEnv, JointPhase
from qnet_core.spec import EpisodeSpec

from .policy import CAAPPOPolicy, PPOTransition


@dataclass(frozen=True)
class EpisodeTrainingResult:
    reward: float
    metrics: dict[str, float]
    policy_stats: dict[str, float]


class CAAPPORolloutTrainer:
    """Train route/construction decisions against the shared executor."""

    def __init__(
        self,
        policy: CAAPPOPolicy | None = None,
        *,
        risk_limit: float = 0.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        chi: float = 1.0,
    ):
        self.policy = policy or CAAPPOPolicy()
        self.risk_limit = float(risk_limit)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.chi = float(chi)

    def _select_routes(
        self,
        candidates: tuple[RouteConstructionCandidate, ...],
        deterministic: bool,
    ) -> tuple[
        dict[str, RouteConstructionCandidate],
        list[tuple[tuple[RouteConstructionCandidate, ...], int, float]],
    ]:
        selected: dict[str, RouteConstructionCandidate] = {}
        records: list[tuple[tuple[RouteConstructionCandidate, ...], int, float]] = []
        for request_id, values in candidates_by_request(candidates).items():
            candidate, log_probability = self.policy.select_candidate(
                values, deterministic=deterministic
            )
            selected[request_id] = candidate
            records.append((values, values.index(candidate), log_probability))
        return selected, records

    def run_episode(
        self,
        spec: EpisodeSpec,
        candidates: tuple[RouteConstructionCandidate, ...],
        *,
        deterministic: bool = False,
        update: bool = True,
    ) -> EpisodeTrainingResult:
        selected, route_records = self._select_routes(candidates, deterministic)
        env = JointConstructionBatchEnv(
            spec,
            candidates,
            alpha=self.alpha,
            beta=self.beta,
            chi=self.chi,
        )
        env.reset()
        state = env.admit(selected)
        rewards: list[float] = []
        risk_increments: list[float] = []
        cumulative_risk = 0.0
        transitions: list[tuple[object, int]] = []

        def record_outcome() -> None:
            nonlocal cumulative_risk
            rewards.append(state.reward)
            current_risk = float(state.info.get("risk_count", cumulative_risk))
            if current_risk < cumulative_risk:
                raise RuntimeError("episode risk count must be monotone")
            risk_increments.append(current_risk - cumulative_risk)
            cumulative_risk = current_risk

        while not state.terminated:
            if state.phase == JointPhase.REPAIR:
                for request_id in env.repairable_requests:
                    state = env.drop(request_id)
                    record_outcome()
                continue
            if state.observation is None:
                raise RuntimeError("execution state lacks a construction observation")
            oracle = CapacityFeasibilityOracle.from_snapshot(state.observation)
            sample = self.policy.operation_sample(
                state.observation,
                state.ready_operations,
                oracle,
                stop_legal=env.core.stop_legal(),
                deterministic=deterministic,
            )
            operation_by_id = {operation.op_id: operation for operation in state.ready_operations}
            operations = tuple(
                operation_by_id[operation_id]
                for operation_id in sample.action.operation_ids
            )
            state = env.step(operations)
            record_outcome()
            transitions.append((sample, len(rewards) - 1))
        returns: list[float] = [0.0] * len(rewards)
        running = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            running += rewards[index]
            returns[index] = running
        risk_returns: list[float] = [0.0] * len(risk_increments)
        running_risk = 0.0
        for index in range(len(risk_increments) - 1, -1, -1):
            running_risk += risk_increments[index]
            risk_returns[index] = running_risk
        advantages = {
            reward_index: returns[reward_index] - self.policy.value_estimate(sample.feature)
            for sample, reward_index in transitions
        }
        risk_advantages = {
            reward_index: risk_returns[reward_index]
            - self.policy.constraint_estimate(sample.feature)
            for sample, reward_index in transitions
        }
        risk_cost = env.metrics()["risk_count"]
        if abs(running_risk - risk_cost) > 1e-9:
            raise RuntimeError("transition risk accounting diverged from episode risk")
        ppo_transitions = tuple(
            PPOTransition(
                sample.feature,
                0,
                sample.log_probability,
                advantages[reward_index],
                returns[reward_index],
                risk_returns[reward_index],
                sample.ordered_operation_ids,
                sample.operation_ordinals,
                sample.legal_indices,
                sample.selected_indices,
                sample.seed_legal_indices,
                sample.seed_index,
                episode_risk_cost=risk_cost,
                risk_advantage=risk_advantages[reward_index],
            )
            for sample, reward_index in transitions
        )
        policy_stats = self.policy.update(
            ppo_transitions,
            risk_limit=self.risk_limit,
        ) if update and not deterministic else {}
        if update and not deterministic:
            policy_stats.update(
                self.policy.update_routes(
                    route_records,
                    sum(rewards),
                    risk_cost=risk_cost,
                )
            )
        return EpisodeTrainingResult(sum(rewards), env.metrics(), policy_stats)

    def train(
        self,
        spec: EpisodeSpec,
        candidates: tuple[RouteConstructionCandidate, ...],
        episodes: int,
    ) -> tuple[EpisodeTrainingResult, ...]:
        if episodes < 1:
            raise ValueError("episodes must be positive")
        return tuple(
            self.run_episode(spec, candidates, deterministic=False, update=True)
            for _ in range(episodes)
        )
