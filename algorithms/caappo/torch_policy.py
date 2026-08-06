"""PyTorch CAAPPO policy on the neutral construction environment.

The NumPy policy remains the small reference implementation.  This module is
the trainable actor-critic path: action legality is still decided by the
simulator-neutral canonical decoder, while PyTorch owns only scores, critics,
PPO ratios, and the CMDP dual update.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Bernoulli, Categorical

from qnet_core.construction_api import (
    ConstructionOperation,
    ConstructionSnapshot,
    OperationKind,
)
from qnet_core.construction_catalog import RouteConstructionCandidate
from qnet_core.construction_decoder import (
    CapacityFeasibilityOracle,
    canonical_decode_ready_set,
)

from .policy import (
    CAAPPOPolicy,
    PolicyAction,
    PolicySample,
)


@dataclass(frozen=True)
class TorchRouteSample:
    candidate: RouteConstructionCandidate
    index: int
    context: tuple[float, ...]
    feature: np.ndarray
    log_probability: float
    entropy: float


@dataclass(frozen=True)
class TorchOperationSample:
    sample: PolicySample


@dataclass(frozen=True)
class TorchRepairSample:
    sample: PolicySample


@dataclass(frozen=True)
class TorchRouteRecord:
    candidates: tuple[RouteConstructionCandidate, ...]
    index: int
    context: tuple[float, ...]
    old_log_probability: float
    advantage: float


@dataclass(frozen=True)
class TorchTransition:
    sample: PolicySample
    old_log_probability: float
    advantage: float
    return_value: float
    risk_advantage: float
    risk_return: float
    episode_risk_cost: float
    discount_prefix: float = 1.0


@dataclass(frozen=True)
class TorchUpdateStats:
    policy_loss: float
    value_loss: float
    constraint_value_loss: float
    route_policy_loss: float
    entropy: float
    lambda_risk: float
    episode_risk_cost: float


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    next_values: Sequence[float],
    dones: Sequence[bool],
    *,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    discounts: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute finite-horizon GAE and value targets.

    ``values`` and ``next_values`` are state-value predictions at the same
    event epochs.  A terminal transition has zero bootstrap value, even when
    the environment reaches the horizon by timeout settlement.
    """

    if not (0.0 < gamma <= 1.0):
        raise ValueError("gamma must lie in (0, 1]")
    if not (0.0 <= gae_lambda <= 1.0):
        raise ValueError("gae_lambda must lie in [0, 1]")
    size = len(rewards)
    if not (
        len(values) == size
        and len(next_values) == size
        and len(dones) == size
    ):
        raise ValueError("GAE inputs must have equal lengths")
    transition_discounts = (
        np.full(size, gamma, dtype=np.float64)
        if discounts is None
        else np.asarray(discounts, dtype=np.float64)
    )
    if transition_discounts.shape != (size,):
        raise ValueError("discounts must have one entry per transition")
    if np.any(transition_discounts < 0.0) or np.any(transition_discounts > 1.0):
        raise ValueError("transition discounts must lie in [0, 1]")
    advantages = np.zeros(size, dtype=np.float64)
    running = 0.0
    for index in range(size - 1, -1, -1):
        bootstrap = 0.0 if dones[index] else float(next_values[index])
        discount = float(transition_discounts[index])
        delta = float(rewards[index]) + discount * bootstrap - float(values[index])
        running = delta + discount * gae_lambda * (0.0 if dones[index] else running)
        advantages[index] = running
    return advantages, advantages + np.asarray(values, dtype=np.float64)


class TorchRelationAwareDAGEncoder(nn.Module):
    """Trainable one-layer message-passing encoder for construction DAGs."""

    def __init__(self, hidden_dim: int = 32, seed: int = 0):
        super().__init__()
        if hidden_dim < 4:
            raise ValueError("hidden_dim must be at least 4")
        torch.manual_seed(int(seed))
        self.hidden_dim = int(hidden_dim)
        self.self_projection = nn.Linear(9, hidden_dim)
        self.message_projection = nn.Linear(9, hidden_dim)

    @staticmethod
    def _operation_features(
        snapshot: ConstructionSnapshot,
        operations: Sequence[ConstructionOperation],
        device: torch.device,
    ) -> Tensor:
        completed = {
            operation_id
            for state in snapshot.dag_states
            for operation_id in state.completed
        }
        started = {
            operation_id
            for state in snapshot.dag_states
            for operation_id in state.started
        }
        dead = {
            operation_id
            for state in snapshot.dag_states
            for operation_id in state.dead
        }
        settled = set(snapshot.settled_request_ids)
        values = []
        active_operations = tuple(
            operation
            for operation in operations
            if operation.request_id not in settled
        )
        for operation in active_operations:
            values.append((
                float(operation.kind == OperationKind.GEN),
                float(operation.kind == OperationKind.SWAP),
                float(operation.kind == OperationKind.RELEASE),
                float(operation.op_id in completed),
                float(operation.op_id in started),
                float(operation.op_id in dead),
                float(len(operation.predecessors)),
                float(sum(amount for _, amount in operation.resource_demand.items())),
                float(operation.duration_ps),
            ))
        if not values:
            return torch.zeros((0, 9), dtype=torch.float32, device=device)
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    def encode_tensor(
        self,
        snapshot: ConstructionSnapshot,
        operations: Sequence[ConstructionOperation],
    ) -> Tensor:
        device = next(self.parameters()).device
        source = snapshot.operations or tuple(operations)
        source = tuple(
            operation
            for operation in source
            if operation.request_id not in set(snapshot.settled_request_ids)
        )
        features = self._operation_features(snapshot, source, device)
        if not len(source):
            return torch.zeros(self.hidden_dim + 5, dtype=torch.float32, device=device)
        index = {operation.op_id: position for position, operation in enumerate(source)}
        message_rows = []
        for operation in source:
            predecessor_indices = [
                index[pred] for pred in operation.predecessors if pred in index
            ]
            if predecessor_indices:
                message_rows.append(features[predecessor_indices].mean(dim=0))
            else:
                message_rows.append(torch.zeros(9, dtype=torch.float32, device=device))
        messages = torch.stack(message_rows, dim=0)
        hidden = torch.tanh(
            self.self_projection(features) + self.message_projection(messages)
        )
        global_features = torch.as_tensor((
            snapshot.physical_time_ps / max(snapshot.horizon_ps, 1),
            len(snapshot.segments),
            len(snapshot.in_flight),
            len(snapshot.pending_events),
            sum(amount for _, amount in snapshot.reservations),
        ), dtype=torch.float32, device=device)
        return torch.cat((hidden.mean(dim=0), global_features))

    def encode(
        self,
        snapshot: ConstructionSnapshot,
        operations: Sequence[ConstructionOperation],
    ) -> np.ndarray:
        return self.encode_tensor(snapshot, operations).detach().cpu().numpy()


class TorchCAAPPOPolicy(nn.Module):
    """Trainable CAAPPO heads with exact environment-side action masking."""

    admission_context_dim = 8

    def __init__(
        self,
        hidden_dim: int = 32,
        seed: int = 0,
        learning_rate: float = 3e-4,
        device: str | torch.device = "cpu",
        use_dag_state: bool = True,
        use_capacity_context: bool = True,
    ):
        super().__init__()
        if hidden_dim < 4:
            raise ValueError("hidden_dim must be at least 4")
        torch.manual_seed(int(seed))
        self.encoder = TorchRelationAwareDAGEncoder(hidden_dim, seed)
        self.hidden_dim = int(hidden_dim)
        self.state_dim = hidden_dim + 5
        self.route_dim = 5 + self.admission_context_dim
        self.operation_dim = hidden_dim + 1
        self.learning_rate = float(learning_rate)
        self.device = torch.device(device)
        self.use_dag_state = bool(use_dag_state)
        self.use_capacity_context = bool(use_capacity_context)
        self.lambda_risk = 0.0

        self.route_actor = nn.Sequential(
            nn.Linear(self.route_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.operation_actor = nn.Sequential(
            nn.Linear(self.operation_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.repair_actor = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.repair_option_actor = nn.Sequential(
            nn.Linear(self.state_dim + 3, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.value_critic = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.constraint_critic = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _tensor(self, values: np.ndarray | Sequence[float]) -> Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    @staticmethod
    def _candidate_feature(candidate: RouteConstructionCandidate) -> np.ndarray:
        return CAAPPOPolicy._candidate_feature(candidate)

    def _route_features(
        self,
        candidates: Sequence[RouteConstructionCandidate],
        context: Sequence[float],
    ) -> Tensor:
        context_array = np.asarray(tuple(context), dtype=np.float32)
        if context_array.shape != (self.admission_context_dim,):
            raise ValueError(
                f"admission context must have {self.admission_context_dim} entries"
            )
        return self._tensor(np.vstack([
            np.concatenate((self._candidate_feature(candidate), context_array))
            for candidate in candidates
        ]))

    def _state_feature(
        self,
        snapshot: ConstructionSnapshot,
        operations: Sequence[ConstructionOperation],
    ) -> np.ndarray:
        return self._state_tensor(snapshot, operations).detach().cpu().numpy()

    def _state_tensor(
        self,
        snapshot: ConstructionSnapshot,
        operations: Sequence[ConstructionOperation],
    ) -> Tensor:
        if not self.use_dag_state:
            snapshot = replace(snapshot, operations=())
            operations = ()
        feature = self.encoder.encode_tensor(snapshot, operations)
        if not self.use_capacity_context:
            feature = feature.clone()
            feature[-1] = 0.0
        return feature

    def _state_tensor_for_sample(self, sample: PolicySample) -> Tensor:
        if sample.state_snapshot is None:
            return self._tensor(sample.feature)
        operations = tuple(sample.state_operations)
        return self._state_tensor(sample.state_snapshot, operations)

    def value_for_sample(self, sample: PolicySample) -> float:
        with torch.no_grad():
            return float(self.value_tensor(self._state_tensor_for_sample(sample)).item())

    def constraint_for_sample(self, sample: PolicySample) -> float:
        with torch.no_grad():
            return float(
                self.constraint_tensor(self._state_tensor_for_sample(sample)).item()
            )

    def value_tensor(self, feature: np.ndarray | Tensor) -> Tensor:
        values = feature if isinstance(feature, Tensor) else self._tensor(feature)
        return self.value_critic(values).squeeze(-1)

    def constraint_tensor(self, feature: np.ndarray | Tensor) -> Tensor:
        values = feature if isinstance(feature, Tensor) else self._tensor(feature)
        return self.constraint_critic(values).squeeze(-1)

    def value(self, feature: np.ndarray) -> float:
        with torch.no_grad():
            return float(self.value_tensor(feature).item())

    def constraint_value(self, feature: np.ndarray) -> float:
        with torch.no_grad():
            return float(self.constraint_tensor(feature).item())

    def sample_route(
        self,
        candidates: Sequence[RouteConstructionCandidate],
        context: Sequence[float],
        *,
        deterministic: bool = False,
    ) -> TorchRouteSample:
        if not candidates:
            raise ValueError("candidate catalogue is empty")
        features = self._route_features(candidates, context)
        logits = self.route_actor(features).squeeze(-1)
        distribution = Categorical(logits=logits)
        index = int(torch.argmax(logits).item()) if deterministic else int(
            distribution.sample().item()
        )
        action = torch.as_tensor(index, device=self.device)
        return TorchRouteSample(
            candidates[index],
            index,
            tuple(float(value) for value in context),
            features[index].detach().cpu().numpy(),
            float(distribution.log_prob(action).detach().item()),
            float(distribution.entropy().detach().item()),
        )

    def evaluate_route_log_probability(
        self,
        candidates: Sequence[RouteConstructionCandidate],
        index: int,
        context: Sequence[float],
    ) -> Tensor:
        if index < 0 or index >= len(candidates):
            raise ValueError("route sample index out of range")
        logits = self.route_actor(self._route_features(candidates, context)).squeeze(-1)
        return Categorical(logits=logits).log_prob(
            torch.as_tensor(index, dtype=torch.long, device=self.device)
        )

    @staticmethod
    def _ordered_ready_operations(
        snapshot: ConstructionSnapshot,
        candidates: Sequence[ConstructionOperation],
    ) -> tuple[ConstructionOperation, ...]:
        completed = {
            operation_id
            for state in snapshot.dag_states
            for operation_id in state.completed
        }
        started = {
            operation_id
            for state in snapshot.dag_states
            for operation_id in state.started
        }
        dead = {
            operation_id
            for state in snapshot.dag_states
            for operation_id in state.dead
        }
        available = {segment.segment_id for segment in snapshot.segments}
        return tuple(sorted(
            (
                operation for operation in candidates
                if operation.op_id not in completed
                and operation.op_id not in started
                and operation.op_id not in dead
                and set(operation.predecessors).issubset(completed)
                and set(operation.input_segment_ids).issubset(available)
            ),
            key=lambda operation: operation.canonical_key,
        ))

    def sample_operation(
        self,
        snapshot: ConstructionSnapshot,
        candidates: Sequence[ConstructionOperation],
        oracle: CapacityFeasibilityOracle,
        *,
        stop_legal: bool,
        deterministic: bool = False,
    ) -> TorchOperationSample:
        ordered = self._ordered_ready_operations(snapshot, candidates)
        state_tensor = self._state_tensor(snapshot, ordered)
        feature = state_tensor.detach().cpu().numpy()
        if not ordered and not stop_legal:
            raise ValueError("no ready operation and STOP is not legal")
        operation_features = (
            torch.stack([
                torch.cat((state_tensor[: self.hidden_dim], self._tensor((float(operation.ordinal),))))
                for operation in ordered
            ])
            if ordered
            else torch.zeros((0, self.operation_dim), dtype=torch.float32, device=self.device)
        )
        scores = self.operation_actor(operation_features).squeeze(-1)
        selected_indices: list[int] = []
        seed_legal = tuple(
            index for index, operation in enumerate(ordered)
            if oracle.can_add((), operation)
        )
        legal_indices: list[int] = []
        seed_index = -1
        log_probability = torch.zeros((), device=self.device)
        entropy = torch.zeros((), device=self.device)
        if not stop_legal:
            if not seed_legal:
                raise ValueError("no legal operation and STOP is not legal")
            seed_logits = scores[list(seed_legal)]
            seed_distribution = Categorical(logits=seed_logits)
            seed_position = (
                int(torch.argmax(seed_logits).item())
                if deterministic
                else int(seed_distribution.sample().item())
            )
            seed_index = int(seed_legal[seed_position])
            selected_indices.append(seed_index)
            log_probability = log_probability + seed_distribution.log_prob(
                torch.as_tensor(seed_position, device=self.device)
            )
            entropy = entropy + seed_distribution.entropy()
            candidate_indices = range(seed_index + 1, len(ordered))
        else:
            candidate_indices = range(len(ordered))
        for index in candidate_indices:
            operation = ordered[index]
            if not oracle.can_add(
                tuple(ordered[selected] for selected in selected_indices), operation
            ):
                continue
            legal_indices.append(index)
            distribution = Bernoulli(logits=scores[index])
            choose = (
                bool(scores[index].item() >= 0.0)
                if deterministic
                else bool(distribution.sample().item())
            )
            log_probability = log_probability + distribution.log_prob(
                torch.as_tensor(float(choose), device=self.device)
            )
            entropy = entropy + distribution.entropy()
            if choose:
                selected_indices.append(index)
        if not selected_indices and not stop_legal:
            raise ValueError("empty operation set is illegal")
        decoded = canonical_decode_ready_set(
            ordered,
            oracle,
            stop_legal,
            selected_indices,
        )
        sample = PolicySample(
            PolicyAction(None, tuple(operation.op_id for operation in decoded), not decoded),
            float(log_probability.detach().item()),
            float(entropy.detach().item()),
            feature,
            tuple(operation.op_id for operation in ordered),
            tuple(operation.ordinal for operation in ordered),
            tuple(legal_indices),
            tuple(selected_indices),
            seed_legal,
            seed_index,
            state_snapshot=snapshot,
            state_operations=tuple(ordered),
        )
        return TorchOperationSample(sample)

    def evaluate_operation_log_probability(self, sample: PolicySample) -> Tensor:
        feature = self._state_tensor_for_sample(sample)
        if not sample.ordered_operation_ids:
            return torch.zeros((), device=self.device)
        operation_features = torch.stack([
            torch.cat((feature[: self.hidden_dim], self._tensor((float(ordinal),))))
            for ordinal in sample.operation_ordinals
        ])
        scores = self.operation_actor(operation_features).squeeze(-1)
        selected = set(sample.selected_indices)
        log_probability = torch.zeros((), device=self.device)
        if sample.seed_index >= 0:
            seed_legal = tuple(sample.seed_legal_indices)
            seed_distribution = Categorical(logits=scores[list(seed_legal)])
            position = seed_legal.index(sample.seed_index)
            log_probability = log_probability + seed_distribution.log_prob(
                torch.as_tensor(position, dtype=torch.long, device=self.device)
            )
        for index in sorted(set(sample.legal_indices)):
            distribution = Bernoulli(logits=scores[index])
            log_probability = log_probability + distribution.log_prob(
                torch.as_tensor(float(index in selected), device=self.device)
            )
        del feature
        return log_probability

    def sample_repair(
        self,
        snapshot: ConstructionSnapshot,
        repair_options: Sequence[tuple[ConstructionOperation, ...]],
        *,
        deterministic: bool = False,
    ) -> TorchRepairSample:
        operations = tuple(
            operation for option in repair_options for operation in option
        )
        state_tensor = self._state_tensor(snapshot, operations)
        feature = state_tensor.detach().cpu().numpy()
        if not repair_options:
            return TorchRepairSample(PolicySample(
                PolicyAction(None, (), True),
                0.0,
                0.0,
                feature,
                repair_action=0,
                state_snapshot=snapshot,
                state_operations=operations,
            ))
        option_features = tuple(
            tuple(float(value) for value in self._repair_option_feature(option))
            for option in repair_options
        )
        drop_logit = self.repair_actor(state_tensor).squeeze(-1)
        retry_logits = self.repair_option_actor(
            self._repair_option_inputs_tensor(state_tensor, option_features)
        ).squeeze(-1)
        distribution = Categorical(logits=torch.cat((
            drop_logit.reshape(1), retry_logits,
        )))
        action_index = (
            int(torch.argmax(distribution.logits).item())
            if deterministic
            else int(distribution.sample().item())
        )
        action = torch.as_tensor(action_index, dtype=torch.long, device=self.device)
        sample = PolicySample(
            PolicyAction(None, (), action_index == 0),
            float(distribution.log_prob(action).detach().item()),
            float(distribution.entropy().detach().item()),
            feature,
            ordered_operation_ids=tuple(
                operation.op_id for operation in operations
            ),
            repair_action=action_index,
            repair_option_features=option_features,
            state_snapshot=snapshot,
            state_operations=operations,
        )
        return TorchRepairSample(sample)

    def _repair_option_feature(
        self, option: Sequence[ConstructionOperation]
    ) -> np.ndarray:
        return np.asarray((
            float(len(option)),
            float(sum(operation.duration_ps for operation in option)) / 1_000_000.0,
            float(min((operation.ordinal for operation in option), default=0)),
        ), dtype=np.float32)

    def _repair_option_inputs(
        self,
        feature: np.ndarray,
        option_features: Sequence[Sequence[float]],
    ) -> np.ndarray:
        state = np.asarray(feature, dtype=np.float32)
        options = np.asarray(option_features, dtype=np.float32)
        if options.ndim != 2 or options.shape[1] != 3:
            raise ValueError("repair option features must have shape (n, 3)")
        return np.concatenate((
            np.repeat(state[None, :], len(options), axis=0),
            options,
        ), axis=1)

    def _repair_distribution(self, sample: PolicySample) -> Categorical | None:
        if sample.repair_action < 0 or not sample.repair_option_features:
            return None
        state_tensor = self._state_tensor_for_sample(sample)
        drop_logit = self.repair_actor(state_tensor).squeeze(-1)
        retry_logits = self.repair_option_actor(
            self._repair_option_inputs_tensor(state_tensor, sample.repair_option_features)
        ).squeeze(-1)
        return Categorical(logits=torch.cat((drop_logit.reshape(1), retry_logits)))

    def _repair_option_inputs_tensor(
        self,
        feature: Tensor,
        option_features: Sequence[Sequence[float]],
    ) -> Tensor:
        options = self._tensor(option_features)
        if options.ndim != 2 or options.shape[1] != 3:
            raise ValueError("repair option features must have shape (n, 3)")
        return torch.cat((feature.reshape(1, -1).repeat(len(options), 1), options), dim=1)

    def evaluate_repair_log_probability(self, sample: PolicySample) -> Tensor:
        if sample.repair_action < 0:
            return torch.zeros((), device=self.device)
        distribution = self._repair_distribution(sample)
        if distribution is None:
            return torch.zeros((), device=self.device)
        return distribution.log_prob(
            torch.as_tensor(sample.repair_action, dtype=torch.long, device=self.device)
        )

    def evaluate_action_entropy(self, sample: PolicySample) -> Tensor:
        """Return the entropy of the exact masked distribution used to sample."""

        if sample.repair_action >= 0:
            distribution = self._repair_distribution(sample)
            return (
                torch.zeros((), device=self.device)
                if distribution is None
                else distribution.entropy()
            )
        if not sample.ordered_operation_ids:
            return torch.zeros((), device=self.device)
        state_tensor = self._state_tensor_for_sample(sample)
        operation_features = torch.stack([
            torch.cat((state_tensor[: self.hidden_dim], self._tensor((float(ordinal),))))
            for ordinal in sample.operation_ordinals
        ])
        scores = self.operation_actor(operation_features).squeeze(-1)
        entropy = torch.zeros((), device=self.device)
        if sample.seed_index >= 0:
            entropy = entropy + Categorical(
                logits=scores[list(sample.seed_legal_indices)]
            ).entropy()
        for index in sorted(set(sample.legal_indices)):
            entropy = entropy + Bernoulli(logits=scores[index]).entropy()
        return entropy

    def evaluate_action_log_probability(self, sample: PolicySample) -> Tensor:
        if sample.repair_action >= 0:
            return self.evaluate_repair_log_probability(sample)
        return self.evaluate_operation_log_probability(sample)

    def update(
        self,
        transitions: Sequence[TorchTransition],
        route_records: Sequence[TorchRouteRecord],
        *,
        risk_limit: float = 0.0,
        clip_epsilon: float = 0.2,
        epochs: int = 4,
        entropy_coef: float = 1e-3,
        value_coef: float = 0.5,
        constraint_value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ) -> TorchUpdateStats:
        if not transitions and not route_records:
            return TorchUpdateStats(0.0, 0.0, 0.0, 0.0, 0.0, self.lambda_risk, 0.0)
        if epochs < 1:
            raise ValueError("epochs must be positive")
        lambda_snapshot = float(self.lambda_risk)
        episode_cost = float(
            transitions[0].episode_risk_cost if transitions else 0.0
        )
        last_policy = 0.0
        last_value = 0.0
        last_constraint = 0.0
        last_route = 0.0
        last_entropy = 0.0
        for _ in range(epochs):
            policy_losses: list[Tensor] = []
            value_losses: list[Tensor] = []
            constraint_losses: list[Tensor] = []
            route_losses: list[Tensor] = []
            entropies: list[Tensor] = []
            for transition in transitions:
                new_log_probability = self.evaluate_action_log_probability(
                    transition.sample
                )
                old_log_probability = torch.as_tensor(
                    transition.old_log_probability,
                    dtype=torch.float32,
                    device=self.device,
                )
                ratio = torch.exp(torch.clamp(
                    new_log_probability - old_log_probability,
                    min=-20.0,
                    max=20.0,
                ))
                advantage = torch.as_tensor(
                    transition.advantage,
                    dtype=torch.float32,
                    device=self.device,
                )
                risk_advantage = torch.as_tensor(
                    transition.risk_advantage,
                    dtype=torch.float32,
                    device=self.device,
                )
                reward_term = transition.discount_prefix * advantage
                effective = reward_term - lambda_snapshot * risk_advantage
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - clip_epsilon,
                    1.0 + clip_epsilon,
                )
                policy_losses.append(-torch.minimum(
                    ratio * effective,
                    clipped_ratio * effective,
                ))
                value_target = torch.as_tensor(
                    transition.return_value,
                    dtype=torch.float32,
                    device=self.device,
                )
                risk_target = torch.as_tensor(
                    transition.risk_return,
                    dtype=torch.float32,
                    device=self.device,
                )
                value_prediction = self.value_tensor(
                    self._state_tensor_for_sample(transition.sample)
                )
                constraint_prediction = self.constraint_tensor(
                    self._state_tensor_for_sample(transition.sample)
                )
                value_losses.append((value_prediction - value_target) ** 2)
                constraint_losses.append((constraint_prediction - risk_target) ** 2)
                entropies.append(self.evaluate_action_entropy(transition.sample))
            for record in route_records:
                new_log_probability = self.evaluate_route_log_probability(
                    record.candidates,
                    record.index,
                    record.context,
                )
                old_log_probability = torch.as_tensor(
                    record.old_log_probability,
                    dtype=torch.float32,
                    device=self.device,
                )
                ratio = torch.exp(torch.clamp(
                    new_log_probability - old_log_probability,
                    min=-20.0,
                    max=20.0,
                ))
                advantage = torch.as_tensor(
                    record.advantage - lambda_snapshot * episode_cost,
                    dtype=torch.float32,
                    device=self.device,
                )
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - clip_epsilon,
                    1.0 + clip_epsilon,
                )
                route_losses.append(-torch.minimum(
                    ratio * advantage,
                    clipped_ratio * advantage,
                ))
                route_logits = self.route_actor(
                    self._route_features(record.candidates, record.context)
                ).squeeze(-1)
                entropies.append(Categorical(logits=route_logits).entropy())
            policy_mean = torch.stack(policy_losses).mean() if policy_losses else torch.zeros((), device=self.device)
            value_mean = torch.stack(value_losses).mean() if value_losses else torch.zeros((), device=self.device)
            constraint_mean = torch.stack(constraint_losses).mean() if constraint_losses else torch.zeros((), device=self.device)
            route_mean = torch.stack(route_losses).mean() if route_losses else torch.zeros((), device=self.device)
            entropy_mean = torch.stack(entropies).mean() if entropies else torch.zeros((), device=self.device)
            total = (
                policy_mean
                + value_coef * value_mean
                + constraint_value_coef * constraint_mean
                + route_mean
                - entropy_coef * entropy_mean
            )
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            self.optimizer.step()
            last_policy = float(policy_mean.detach().item())
            last_value = float(value_mean.detach().item())
            last_constraint = float(constraint_mean.detach().item())
            last_route = float(route_mean.detach().item())
            last_entropy = float(entropy_mean.detach().item())

        self.lambda_risk = max(
            0.0,
            self.lambda_risk + self.learning_rate * (episode_cost - risk_limit),
        )
        return TorchUpdateStats(
            last_policy,
            last_value,
            last_constraint,
            last_route,
            last_entropy,
            self.lambda_risk,
            episode_cost,
        )


__all__ = [
    "TorchCAAPPOPolicy",
    "TorchRelationAwareDAGEncoder",
    "TorchOperationSample",
    "TorchRepairSample",
    "TorchRouteRecord",
    "TorchRouteSample",
    "TorchTransition",
    "TorchUpdateStats",
    "compute_gae",
]
