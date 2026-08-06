"""NumPy reference implementation of CAAPPO's masked joint policy.

The implementation is intentionally dependency-light.  It provides the
action semantics, relation-aware encoding, clipped objective bookkeeping, and
CMDP dual update; a larger training run may replace the linear parameter
updates with an autodiff backend without changing the environment contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from qnet_core.construction_api import ConstructionOperation, ConstructionSnapshot, OperationKind
from qnet_core.construction_catalog import RouteConstructionCandidate
from qnet_core.construction_decoder import (
    CapacityFeasibilityOracle,
    canonical_decode_ready_set,
)


@dataclass(frozen=True)
class PolicyAction:
    candidate_id: str | None
    operation_ids: tuple[str, ...]
    stop: bool = False


@dataclass(frozen=True)
class PolicySample:
    action: PolicyAction
    log_probability: float
    entropy: float
    feature: np.ndarray
    ordered_operation_ids: tuple[str, ...] = ()
    operation_ordinals: tuple[int, ...] = ()
    legal_indices: tuple[int, ...] = ()
    selected_indices: tuple[int, ...] = ()
    seed_legal_indices: tuple[int, ...] = ()
    seed_index: int = -1
    repair_action: int = -1
    repair_option_features: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class PPOTransition:
    feature: np.ndarray
    action_index: int
    old_log_probability: float
    advantage: float
    return_value: float
    risk_cost: float = 0.0
    ordered_operation_ids: tuple[str, ...] = ()
    operation_ordinals: tuple[int, ...] = ()
    legal_indices: tuple[int, ...] = ()
    selected_indices: tuple[int, ...] = ()
    seed_legal_indices: tuple[int, ...] = ()
    seed_index: int = -1
    episode_risk_cost: float | None = None
    risk_advantage: float | None = None
    repair_action: int = -1


class RelationAwareDAGEncoder:
    """One-layer message-passing encoder over the current operation DAG."""

    def __init__(self, hidden_dim: int = 32, seed: int = 0):
        if hidden_dim < 4:
            raise ValueError("hidden_dim must be at least 4")
        self.hidden_dim = int(hidden_dim)
        rng = np.random.default_rng(seed)
        self._self = rng.normal(0.0, 0.1, (9, hidden_dim))
        self._message = rng.normal(0.0, 0.1, (9, hidden_dim))
        self._bias = np.zeros(hidden_dim, dtype=np.float64)

    @staticmethod
    def _operation_feature(
        operation: ConstructionOperation,
        snapshot: ConstructionSnapshot,
    ) -> np.ndarray:
        completed = {
            op_id for state in snapshot.dag_states for op_id in state.completed
        }
        started = {op_id for state in snapshot.dag_states for op_id in state.started}
        dead = {op_id for state in snapshot.dag_states for op_id in state.dead}
        kind = (
            float(operation.kind == OperationKind.GEN),
            float(operation.kind == OperationKind.SWAP),
            float(operation.kind == OperationKind.RELEASE),
        )
        resource_total = float(sum(amount for _, amount in operation.resource_demand.items()))
        return np.asarray(
            kind + (
                float(operation.op_id in completed),
                float(operation.op_id in started),
                float(operation.op_id in dead),
                float(len(operation.predecessors)),
                resource_total,
                float(operation.duration_ps),
            ),
            dtype=np.float64,
        )

    def encode(
        self,
        snapshot: ConstructionSnapshot,
        operations: Sequence[ConstructionOperation],
    ) -> np.ndarray:
        operations = snapshot.operations or tuple(operations)
        settled = set(snapshot.settled_request_ids)
        if settled:
            operations = tuple(
                operation for operation in operations
                if operation.request_id not in settled
            )
        if not operations:
            return np.zeros(self.hidden_dim + 5, dtype=np.float64)
        features = np.vstack([
            self._operation_feature(operation, snapshot) for operation in operations
        ])
        index = {operation.op_id: i for i, operation in enumerate(operations)}
        messages = np.zeros_like(features)
        for i, operation in enumerate(operations):
            predecessor_indices = [index[pred] for pred in operation.predecessors if pred in index]
            if predecessor_indices:
                messages[i] = features[predecessor_indices].mean(axis=0)
        hidden = np.tanh(features @ self._self + messages @ self._message + self._bias)
        global_features = np.asarray((
            snapshot.physical_time_ps / max(snapshot.horizon_ps, 1),
            len(snapshot.segments),
            len(snapshot.in_flight),
            len(snapshot.pending_events),
            sum(amount for _, amount in snapshot.reservations),
        ), dtype=np.float64)
        return np.concatenate((hidden.mean(axis=0), global_features))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    values = np.exp(shifted)
    total = float(values.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("invalid categorical logits")
    return values / total


class CAAPPOPolicy:
    """Masked joint route/construction policy with a lightweight PPO update."""

    def __init__(self, hidden_dim: int = 32, seed: int = 0, learning_rate: float = 1e-2):
        self.encoder = RelationAwareDAGEncoder(hidden_dim, seed)
        self.rng = np.random.default_rng(seed)
        self.learning_rate = float(learning_rate)
        self.lambda_risk = 0.0
        self._candidate_weights: np.ndarray | None = None
        self._operation_weights: np.ndarray | None = None
        self._value_weights: np.ndarray | None = None
        self._constraint_weights: np.ndarray | None = None
        self._admission_context_dim = 8
        self._repair_weights: np.ndarray | None = None

    @staticmethod
    def _candidate_feature(candidate: RouteConstructionCandidate) -> np.ndarray:
        peak_live = 0
        live = 0
        for operation in candidate.dag.operations:
            live += int(operation.output_segment_id is not None)
            live -= len(operation.input_segment_ids)
            peak_live = max(peak_live, live)
        return np.asarray((
            float(candidate.hop_count),
            float(candidate.construction_kind == "left_deep"),
            float(candidate.construction_kind == "balanced"),
            float(len(candidate.dag.operations)),
            float(peak_live),
        ), dtype=np.float64)

    def _candidate_feature_with_context(
        self,
        candidate: RouteConstructionCandidate,
        context: Sequence[float] | None = None,
    ) -> np.ndarray:
        values = np.zeros(self._admission_context_dim, dtype=np.float64)
        if context is not None:
            context_array = np.asarray(tuple(context), dtype=np.float64)
            if context_array.ndim != 1 or context_array.shape[0] != self._admission_context_dim:
                raise ValueError(
                    f"admission context must have {self._admission_context_dim} entries"
                )
            values = context_array
        return np.concatenate((self._candidate_feature(candidate), values))

    def _ensure_candidate_weights(self, dimension: int) -> None:
        if self._candidate_weights is None:
            self._candidate_weights = self.rng.normal(0.0, 0.05, dimension)
        elif self._candidate_weights.shape[0] != dimension:
            if self._candidate_weights.shape[0] == 5 and dimension == 13:
                self._candidate_weights = np.concatenate((
                    self._candidate_weights,
                    np.zeros(dimension - 5, dtype=np.float64),
                ))
            else:
                raise ValueError("candidate actor dimension changed within a policy")

    def _ensure_operation_weights(self, dimension: int) -> None:
        if self._operation_weights is None:
            self._operation_weights = self.rng.normal(0.0, 0.05, dimension)

    def _ensure_repair_weights(self, dimension: int) -> None:
        if self._repair_weights is None:
            self._repair_weights = self.rng.normal(0.0, 0.05, dimension)
        elif self._repair_weights.shape[0] != dimension:
            raise ValueError("repair actor dimension changed within a policy")

    def _operation_feature(self, feature: np.ndarray, ordinal: int) -> np.ndarray:
        """Feature used by the operation actor (state embedding + ordinal)."""

        return np.concatenate((
            np.asarray(feature[: self.encoder.hidden_dim], dtype=np.float64),
            np.asarray((float(ordinal),), dtype=np.float64),
        ))

    @staticmethod
    def _sigmoid(score: float) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(score, -40.0, 40.0))))

    def select_candidate(
        self,
        candidates: Sequence[RouteConstructionCandidate],
        *,
        deterministic: bool = False,
    ) -> tuple[RouteConstructionCandidate, float]:
        if not candidates:
            raise ValueError("candidate catalogue is empty")
        features = np.vstack([
            self._candidate_feature_with_context(candidate)
            for candidate in candidates
        ])
        self._ensure_candidate_weights(features.shape[1])
        probabilities = _softmax(features @ self._candidate_weights)
        index = int(np.argmax(probabilities) if deterministic else self.rng.choice(len(candidates), p=probabilities))
        return candidates[index], float(np.log(max(probabilities[index], 1e-12)))

    def select_candidate_context(
        self,
        candidates: Sequence[RouteConstructionCandidate],
        context: Sequence[float],
        *,
        deterministic: bool = False,
    ) -> tuple[RouteConstructionCandidate, float]:
        """Autoregressive admission head conditioned on prior selections."""

        # Test and research policies may override the legacy selector.  Keep
        # that contract intact while the default CAAPPO head uses context.
        if type(self).select_candidate is not CAAPPOPolicy.select_candidate:
            return self.select_candidate(candidates, deterministic=deterministic)
        if not candidates:
            raise ValueError("candidate catalogue is empty")
        features = np.vstack([
            self._candidate_feature_with_context(candidate, context)
            for candidate in candidates
        ])
        self._ensure_candidate_weights(features.shape[1])
        assert self._candidate_weights is not None
        probabilities = _softmax(features @ self._candidate_weights)
        index = int(
            np.argmax(probabilities)
            if deterministic
            else self.rng.choice(len(candidates), p=probabilities)
        )
        return candidates[index], float(np.log(max(probabilities[index], 1e-12)))

    def operation_sample(
        self,
        snapshot: ConstructionSnapshot,
        candidates: Sequence[ConstructionOperation],
        oracle: CapacityFeasibilityOracle,
        *,
        stop_legal: bool,
        deterministic: bool = False,
    ) -> PolicySample:
        completed = {
            op_id for state in snapshot.dag_states for op_id in state.completed
        }
        started = {
            op_id for state in snapshot.dag_states for op_id in state.started
        }
        dead = {op_id for state in snapshot.dag_states for op_id in state.dead}
        available_segments = {segment.segment_id for segment in snapshot.segments}
        ordered = tuple(sorted(
            (
                operation for operation in candidates
                if operation.op_id not in completed
                and operation.op_id not in started
                and operation.op_id not in dead
                and set(operation.predecessors).issubset(completed)
                and set(operation.input_segment_ids).issubset(available_segments)
            ),
            key=lambda operation: operation.canonical_key,
        ))
        feature = self.encoder.encode(snapshot, ordered)
        self._ensure_operation_weights(self.encoder.hidden_dim + 1)
        assert self._operation_weights is not None
        scores = np.asarray([
            float(self._operation_feature(feature, operation.ordinal) @ self._operation_weights)
            for operation in ordered
        ], dtype=np.float64)
        selected: list[ConstructionOperation] = []
        selected_indices: list[int] = []
        legal_indices: list[int] = []
        seed_legal_indices = tuple(
            index for index, operation in enumerate(ordered)
            if oracle.can_add((), operation)
        )
        seed_index = -1
        log_probability = 0.0
        entropy = 0.0
        if not stop_legal:
            if not seed_legal_indices:
                raise ValueError("no legal operation and STOP is not legal")
            seed_probabilities = _softmax(scores[list(seed_legal_indices)])
            seed_position = int(
                np.argmax(seed_probabilities)
                if deterministic
                else self.rng.choice(len(seed_legal_indices), p=seed_probabilities)
            )
            seed_index = int(seed_legal_indices[seed_position])
            selected.append(ordered[seed_index])
            selected_indices.append(seed_index)
            seed_probability = float(seed_probabilities[seed_position])
            log_probability = math.log(max(seed_probability, 1e-12))
            entropy = float(-np.sum(
                seed_probabilities * np.log(np.maximum(seed_probabilities, 1e-12))
            ))
            candidate_indices = range(seed_index + 1, len(ordered))
        else:
            candidate_indices = range(len(ordered))

        for index in candidate_indices:
            operation = ordered[index]
            if not oracle.can_add(selected, operation):
                continue
            legal_indices.append(index)
            probability = self._sigmoid(float(scores[index]))
            choose = probability >= 0.5 if deterministic else bool(self.rng.random() < probability)
            log_probability += math.log(max(probability if choose else 1.0 - probability, 1e-12))
            entropy -= probability * math.log(max(probability, 1e-12)) + (1 - probability) * math.log(max(1 - probability, 1e-12))
            if choose:
                selected.append(operation)
                selected_indices.append(index)
        if not selected and not stop_legal:
            raise ValueError("no legal operation and STOP is not legal")
        decoded = canonical_decode_ready_set(
            ordered,
            oracle,
            stop_legal,
            selected_indices,
        )
        action = PolicyAction(
            None,
            tuple(operation.op_id for operation in decoded),
            not decoded,
        )
        return PolicySample(
            action,
            float(log_probability),
            float(entropy),
            feature,
            tuple(operation.op_id for operation in ordered),
            tuple(operation.ordinal for operation in ordered),
            tuple(legal_indices),
            tuple(selected_indices),
            seed_legal_indices,
            seed_index,
        )

    def repair_sample(
        self,
        snapshot: ConstructionSnapshot,
        repair_options: Sequence[tuple[ConstructionOperation, ...]],
        *,
        deterministic: bool = False,
    ) -> PolicySample:
        """Choose retry versus DROP through a learned binary repair head."""

        operations = tuple(
            operation for option in repair_options for operation in option
        )
        feature = self.encoder.encode(snapshot, operations)
        if not repair_options:
            return PolicySample(
                PolicyAction(None, (), True),
                0.0,
                0.0,
                feature,
                repair_action=-1,
            )
        self._ensure_repair_weights(feature.shape[0])
        assert self._repair_weights is not None
        probability = self._sigmoid(float(feature @ self._repair_weights))
        retry = bool(
            probability >= 0.5
            if deterministic
            else self.rng.random() < probability
        )
        log_probability = math.log(
            max(probability if retry else 1.0 - probability, 1e-12)
        )
        entropy = -(
            probability * math.log(max(probability, 1e-12))
            + (1.0 - probability) * math.log(max(1.0 - probability, 1e-12))
        )
        return PolicySample(
            PolicyAction(None, (), not retry),
            float(log_probability),
            float(entropy),
            feature,
            repair_action=1 if retry else 0,
        )

    def joint_sample(
        self,
        snapshot: ConstructionSnapshot,
        candidates: Sequence[RouteConstructionCandidate],
        operations_by_candidate: dict[str, Sequence[ConstructionOperation]],
        oracle: CapacityFeasibilityOracle,
        *,
        stop_legal: bool,
        deterministic: bool = False,
    ) -> PolicySample:
        candidate, route_log_probability = self.select_candidate(
            candidates,
            deterministic=deterministic,
        )
        operation_sample = self.operation_sample(
            snapshot,
            operations_by_candidate.get(candidate.candidate_id, candidate.dag.operations),
            oracle,
            stop_legal=stop_legal,
            deterministic=deterministic,
        )
        return PolicySample(
            PolicyAction(
                candidate.candidate_id,
                operation_sample.action.operation_ids,
                operation_sample.action.stop,
            ),
            route_log_probability + operation_sample.log_probability,
            operation_sample.entropy,
            operation_sample.feature,
            operation_sample.ordered_operation_ids,
            operation_sample.operation_ordinals,
            operation_sample.legal_indices,
            operation_sample.selected_indices,
            operation_sample.seed_legal_indices,
            operation_sample.seed_index,
        )

    def _ensure_value_weights(self, dimension: int) -> None:
        if self._value_weights is None:
            self._value_weights = np.zeros(dimension, dtype=np.float64)
        if self._constraint_weights is None:
            self._constraint_weights = np.zeros(dimension, dtype=np.float64)

    def value_estimate(self, feature: np.ndarray) -> float:
        """Return the current linear value baseline for one observation."""

        self._ensure_value_weights(feature.shape[0])
        assert self._value_weights is not None
        return float(feature @ self._value_weights)

    def constraint_estimate(self, feature: np.ndarray) -> float:
        """Return the current linear cost-to-go baseline for one observation."""

        self._ensure_value_weights(feature.shape[0])
        assert self._constraint_weights is not None
        return float(feature @ self._constraint_weights)

    def _operation_log_probability(
        self,
        feature: np.ndarray,
        operation_ordinals: Sequence[int],
        legal_indices: Sequence[int],
        selected_indices: Sequence[int],
        seed_legal_indices: Sequence[int] = (),
        seed_index: int = -1,
        weights: np.ndarray | None = None,
    ) -> float:
        weights = self._operation_weights if weights is None else weights
        if weights is None:
            return 0.0
        selected = set(selected_indices)
        scores = [
            float(self._operation_feature(feature, ordinal) @ weights)
            for ordinal in operation_ordinals
        ]
        log_probability = 0.0
        if seed_index >= 0:
            if seed_index not in set(seed_legal_indices) or seed_index not in selected:
                return float("-inf")
            seed_probabilities = _softmax(np.asarray(
                [scores[index] for index in seed_legal_indices], dtype=np.float64
            ))
            seed_position = tuple(seed_legal_indices).index(seed_index)
            log_probability += math.log(max(float(seed_probabilities[seed_position]), 1e-12))
        for index in sorted(set(legal_indices)):
            probability = self._sigmoid(scores[index])
            log_probability += math.log(max(
                probability if index in selected else 1.0 - probability,
                1e-12,
            ))
        return float(log_probability)

    def _operation_log_probability_and_gradient(
        self,
        feature: np.ndarray,
        operation_ordinals: Sequence[int],
        legal_indices: Sequence[int],
        selected_indices: Sequence[int],
        seed_legal_indices: Sequence[int],
        seed_index: int,
        weights: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Return log pi and its exact score-function gradient."""

        gradient = np.zeros_like(weights)
        selected = set(selected_indices)
        action_features = [
            self._operation_feature(feature, ordinal)
            for ordinal in operation_ordinals
        ]
        scores = [float(action_feature @ weights) for action_feature in action_features]
        log_probability = 0.0
        if seed_index >= 0:
            seed_legal = tuple(seed_legal_indices)
            if seed_index not in seed_legal or seed_index not in selected:
                return float("-inf"), gradient
            seed_probabilities = _softmax(np.asarray(
                [scores[index] for index in seed_legal], dtype=np.float64
            ))
            seed_position = seed_legal.index(seed_index)
            log_probability += math.log(max(float(seed_probabilities[seed_position]), 1e-12))
            expected = sum(
                float(probability) * action_features[index]
                for probability, index in zip(seed_probabilities, seed_legal)
            )
            gradient += action_features[seed_index] - expected
        for index in sorted(set(legal_indices)):
            probability = self._sigmoid(scores[index])
            choose = 1.0 if index in selected else 0.0
            log_probability += math.log(max(
                probability if choose else 1.0 - probability,
                1e-12,
            ))
            # The log-probability uses a 1e-12 floor.  Once a Bernoulli tail
            # reaches that floor its numerical derivative is zero as well;
            # keeping the analytic unsaturated derivative would disagree with
            # the objective actually optimized.
            if 1e-12 < probability < 1.0 - 1e-12:
                gradient += (choose - probability) * action_features[index]
        return float(log_probability), gradient

    def update(
        self,
        transitions: Sequence[PPOTransition],
        clip_epsilon: float = 0.2,
        risk_limit: float = 0.0,
        lambda_risk_snapshot: float | None = None,
    ) -> dict[str, float]:
        if clip_epsilon <= 0:
            raise ValueError("clip_epsilon must be positive")
        if not transitions:
            return {"policy_loss": 0.0, "value_loss": 0.0, "lambda_risk": self.lambda_risk}
        losses = []
        value_losses = []
        constraint_losses = []
        costs = []
        policy_gradient = None
        repair_gradient = None
        old_lambda = (
            self.lambda_risk
            if lambda_risk_snapshot is None
            else float(lambda_risk_snapshot)
        )
        assert self._operation_weights is not None or self._candidate_weights is not None
        for transition in transitions:
            self._ensure_value_weights(transition.feature.shape[0])
            assert self._value_weights is not None
            assert self._constraint_weights is not None
            value_prediction = float(transition.feature @ self._value_weights)
            constraint_prediction = float(
                transition.feature @ self._constraint_weights
            )
            value_error = transition.return_value - value_prediction
            constraint_error = transition.risk_cost - constraint_prediction
            value_losses.append(value_error ** 2)
            constraint_losses.append(constraint_error ** 2)
            self._value_weights += self.learning_rate * value_error * transition.feature
            self._constraint_weights += (
                self.learning_rate * constraint_error * transition.feature
            )
            if transition.repair_action >= 0:
                self._ensure_repair_weights(transition.feature.shape[0])
                assert self._repair_weights is not None
                repair_probability = self._sigmoid(
                    float(transition.feature @ self._repair_weights)
                )
                choose_retry = float(transition.repair_action == 1)
                log_probability = math.log(max(
                    repair_probability
                    if choose_retry
                    else 1.0 - repair_probability,
                    1e-12,
                ))
                score_gradient = (
                    choose_retry - repair_probability
                ) * transition.feature
                actor_kind = "repair"
            elif (
                transition.ordered_operation_ids
                and self._operation_weights is not None
            ):
                expected_dimension = self.encoder.hidden_dim + 1
                if self._operation_weights.shape[0] != expected_dimension:
                    raise ValueError(
                        "operation actor dimension does not match the encoded action features"
                    )
                log_probability, score_gradient = self._operation_log_probability_and_gradient(
                    transition.feature,
                    transition.operation_ordinals,
                    transition.legal_indices,
                    transition.selected_indices,
                    transition.seed_legal_indices,
                    transition.seed_index,
                    self._operation_weights,
                )
                actor_kind = "operation"
            else:
                log_probability = transition.old_log_probability
                score_gradient = None
                actor_kind = "none"
            ratio = math.exp(max(
                -20.0,
                min(20.0, log_probability - transition.old_log_probability),
            ))
            clipped = min(max(ratio, 1.0 - clip_epsilon), 1.0 + clip_epsilon)
            constraint_advantage = (
                transition.risk_advantage
                if transition.risk_advantage is not None
                else transition.risk_cost - constraint_prediction
            )
            effective_advantage = (
                transition.advantage - old_lambda * constraint_advantage
            )
            surrogate = min(
                ratio * effective_advantage,
                clipped * effective_advantage,
            )
            losses.append(-surrogate)
            costs.append(transition.risk_cost)
            if score_gradient is not None:
                clipped_active = not (
                    effective_advantage >= 0.0 and ratio > 1.0 + clip_epsilon
                ) and not (
                    effective_advantage < 0.0 and ratio < 1.0 - clip_epsilon
                )
                if clipped_active:
                    if actor_kind == "repair":
                        if repair_gradient is None:
                            repair_gradient = np.zeros_like(score_gradient)
                        repair_gradient += effective_advantage * ratio * score_gradient
                    else:
                        if policy_gradient is None:
                            policy_gradient = np.zeros_like(score_gradient)
                        policy_gradient += effective_advantage * ratio * score_gradient
        if policy_gradient is not None and self._operation_weights is not None:
            self._operation_weights += (
                self.learning_rate * policy_gradient / max(len(transitions), 1)
            )
        if repair_gradient is not None and self._repair_weights is not None:
            self._repair_weights += (
                self.learning_rate * repair_gradient / max(len(transitions), 1)
            )
        episode_cost_values = [
            transition.episode_risk_cost
            for transition in transitions
            if transition.episode_risk_cost is not None
        ]
        episode_cost = float(
            episode_cost_values[0]
            if episode_cost_values
            else np.sum(costs)
        )
        self.lambda_risk = max(
            0.0,
            self.lambda_risk
            + self.learning_rate * (episode_cost - risk_limit),
        )
        return {
            "policy_loss": float(np.mean(losses)),
            "value_loss": float(np.mean(value_losses)),
            "constraint_value_loss": float(np.mean(constraint_losses)),
            "lambda_risk": self.lambda_risk,
            "episode_risk_cost": episode_cost,
        }

    def update_routes(
        self,
        records: Sequence[tuple],
        advantage: float,
        clip_epsilon: float = 0.2,
        risk_cost: float = 0.0,
        lambda_risk: float | None = None,
    ) -> dict[str, float]:
        """Update the admission actor on joint episode returns.

        This is the NumPy reference counterpart of the route PPO head.  It
        recomputes the categorical probability of the selected candidate and
        updates the shared candidate scorer; a production implementation can
        replace this method with autodiff without changing the environment.
        """

        if not records:
            return {"route_policy_loss": 0.0}
        losses = []
        effective_advantage = float(
            advantage - (self.lambda_risk if lambda_risk is None else lambda_risk) * risk_cost
        )
        route_gradient = None
        for record in records:
            if len(record) == 3:
                candidates, index, old_log_probability = record
                context = None
            elif len(record) == 4:
                candidates, index, old_log_probability, context = record
            else:
                raise ValueError("route record must contain 3 or 4 fields")
            features = np.vstack([
                self._candidate_feature_with_context(candidate, context)
                for candidate in candidates
            ])
            self._ensure_candidate_weights(features.shape[1])
            assert self._candidate_weights is not None
            probabilities = _softmax(features @ self._candidate_weights)
            if index < 0 or index >= len(candidates):
                raise ValueError("route sample index out of range")
            new_log_probability = math.log(max(float(probabilities[index]), 1e-12))
            ratio = math.exp(max(-20.0, min(20.0, new_log_probability - old_log_probability)))
            clipped = min(max(ratio, 1.0 - clip_epsilon), 1.0 + clip_epsilon)
            surrogate = min(ratio * effective_advantage, clipped * effective_advantage)
            losses.append(-surrogate)
            expected = probabilities @ features
            gradient = features[index] - expected
            clipped_active = not (
                effective_advantage >= 0.0 and ratio > 1.0 + clip_epsilon
            ) and not (
                effective_advantage < 0.0 and ratio < 1.0 - clip_epsilon
            )
            if clipped_active:
                if route_gradient is None:
                    route_gradient = np.zeros_like(gradient)
                route_gradient += effective_advantage * ratio * gradient
        if route_gradient is not None:
            self._candidate_weights += self.learning_rate * route_gradient / max(len(records), 1)
        return {"route_policy_loss": float(np.mean(losses))}
