from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .config import PPOConfig
from .model import DynamicPlanActorCritic


PLAN_KEYS = ("plan_features", "candidate_features", "candidates")
GLOBAL_KEYS = ("global_features", "state_features", "global_state")
MASK_KEYS = ("action_mask", "candidate_mask", "mask")
REQUEST_KEYS = ("request_features", "requests")
REQUEST_MASK_KEYS = ("request_mask", "active_request_mask")
PLAN_REQUEST_KEYS = ("plan_request_index", "candidate_request_index")


@dataclass(frozen=True)
class PolicyObservation:
    plan_features: np.ndarray
    global_features: np.ndarray
    action_mask: np.ndarray
    plan_mask: np.ndarray
    request_features: np.ndarray | None = None
    request_mask: np.ndarray | None = None
    plan_request_index: np.ndarray | None = None

    @property
    def stop_action(self) -> int:
        return int(self.plan_features.shape[0])


def _first_present(observation: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in observation:
            return observation[key]
    raise KeyError(f"observation needs one of {tuple(keys)}; got {tuple(observation)}")


def parse_observation(observation: Mapping[str, Any]) -> PolicyObservation:
    """Normalize the environment observation used by PPO.

    STOP is always the last action.  Environments may supply a plan-only mask
    or a plan-plus-STOP mask; in the former case a legal STOP entry is appended.
    """
    plans = np.asarray(_first_present(observation, PLAN_KEYS), dtype=np.float32).copy()
    global_features = np.asarray(
        _first_present(observation, GLOBAL_KEYS), dtype=np.float32
    ).copy()
    if plans.ndim != 2:
        raise ValueError(f"candidate plan features must be rank 2, got {plans.shape}")
    if global_features.ndim != 1:
        raise ValueError(f"global features must be rank 1, got {global_features.shape}")
    try:
        mask = np.asarray(_first_present(observation, MASK_KEYS), dtype=np.bool_)
    except KeyError:
        mask = np.ones(plans.shape[0] + 1, dtype=np.bool_)
    if mask.ndim != 1:
        raise ValueError("action mask must be rank 1")
    if mask.shape[0] == plans.shape[0]:
        mask = np.concatenate((mask, np.ones(1, dtype=np.bool_)))
    if mask.shape[0] != plans.shape[0] + 1:
        raise ValueError("action mask must have num_plans or num_plans + 1 entries")
    mask = mask.copy()
    if not mask.any():
        raise ValueError("action mask must expose at least one legal action")
    request_features = None
    request_mask = None
    plan_request_index = None
    for key in REQUEST_KEYS:
        if key in observation:
            request_features = np.asarray(observation[key], dtype=np.float32).copy()
            break
    if request_features is not None:
        if request_features.ndim != 2:
            raise ValueError("request features must be rank 2")
        for key in REQUEST_MASK_KEYS:
            if key in observation:
                request_mask = np.asarray(observation[key], dtype=np.bool_).copy()
                break
        if request_mask is None:
            request_mask = np.ones(request_features.shape[0], dtype=np.bool_)
        if request_mask.shape != request_features.shape[:1]:
            raise ValueError("request mask must contain one entry per request row")
    for key in PLAN_REQUEST_KEYS:
        if key in observation:
            plan_request_index = np.asarray(observation[key], dtype=np.int64).copy()
            break
    if plan_request_index is not None and plan_request_index.shape != plans.shape[:1]:
        raise ValueError("plan_request_index must contain one entry per plan row")
    # The environment uses fixed tensor shapes for efficient replay, while the
    # legal candidate set is dynamic.  Excluding masked slots from the set pool
    # prevents unused zero rows and already-conflicting plans from diluting the
    # actor/critic context.
    return PolicyObservation(
        plans, global_features, mask, mask[:-1].copy(), request_features, request_mask,
        plan_request_index,
    )


@dataclass
class Transition:
    observation: PolicyObservation
    action: int
    old_log_prob: float
    old_value: float
    reward: float
    next_value: float
    terminated: bool
    episode_done: bool
    duration: float = 1.0


@dataclass
class RolloutBuffer:
    transitions: list[Transition] = field(default_factory=list)

    def add(self, transition: Transition) -> None:
        if not 0 <= transition.action < transition.observation.action_mask.shape[0]:
            raise ValueError("recorded action is outside the dynamic action space")
        self.transitions.append(transition)

    def __len__(self) -> int:
        return len(self.transitions)

    def advantages_and_returns(self, config: PPOConfig) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros(len(self), dtype=np.float32)
        next_advantage = 0.0
        for index in reversed(range(len(self))):
            item = self.transitions[index]
            duration = max(float(item.duration), 0.0)
            discount = config.gamma**duration
            trace_discount = (config.gamma * config.gae_lambda) ** duration
            bootstrap = 0.0 if item.terminated else item.next_value
            delta = item.reward + discount * bootstrap - item.old_value
            continuation = 0.0 if item.episode_done else 1.0
            next_advantage = delta + trace_discount * continuation * next_advantage
            advantages[index] = next_advantage
        values = np.asarray([item.old_value for item in self.transitions], dtype=np.float32)
        returns = advantages + values
        if config.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / max(float(advantages.std()), 1e-8)
        return advantages, returns


@dataclass
class PaddedBatch:
    plans: Tensor
    globals: Tensor
    plan_mask: Tensor
    action_mask: Tensor
    requests: Tensor | None
    request_mask: Tensor | None
    plan_request_index: Tensor | None
    actions: Tensor
    old_log_probs: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor


def collate_transitions(
    transitions: Sequence[Transition],
    advantages: np.ndarray,
    returns: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> PaddedBatch:
    selected = [transitions[int(index)] for index in indices]
    max_plans = max((item.observation.plan_features.shape[0] for item in selected), default=0)
    # Linear layers support a zero-length plan dimension, but padding to one is
    # friendlier to older PyTorch builds.  The synthetic row is always masked.
    padded_plans = max(max_plans, 1)
    plan_dim = selected[0].observation.plan_features.shape[1]
    global_dim = selected[0].observation.global_features.shape[0]
    plans = np.zeros((len(selected), padded_plans, plan_dim), dtype=np.float32)
    globals_array = np.zeros((len(selected), global_dim), dtype=np.float32)
    plan_mask = np.zeros((len(selected), padded_plans), dtype=np.bool_)
    action_mask = np.zeros((len(selected), padded_plans + 1), dtype=np.bool_)
    actions = np.empty(len(selected), dtype=np.int64)
    has_requests = selected[0].observation.request_features is not None
    requests = None
    request_mask_array = None
    plan_request_index = None
    if has_requests:
        max_requests = max(item.observation.request_features.shape[0] for item in selected)
        request_dim = selected[0].observation.request_features.shape[1]
        requests = np.zeros((len(selected), max(max_requests, 1), request_dim), dtype=np.float32)
        request_mask_array = np.zeros((len(selected), max(max_requests, 1)), dtype=np.bool_)
    if any(item.observation.plan_request_index is not None for item in selected):
        plan_request_index = np.full((len(selected), padded_plans), -1, dtype=np.int64)

    for row, item in enumerate(selected):
        observation = item.observation
        count = observation.plan_features.shape[0]
        plans[row, :count] = observation.plan_features
        globals_array[row] = observation.global_features
        plan_mask[row, :count] = observation.plan_mask
        action_mask[row, :count] = observation.action_mask[:count]
        action_mask[row, -1] = observation.action_mask[count]
        # STOP is local index ``count`` in the unpadded environment action
        # space, but the network's padded STOP is always the final column.
        actions[row] = padded_plans if item.action == count else item.action
        if has_requests:
            if observation.request_features is None or observation.request_mask is None:
                raise ValueError("request features must be consistently present throughout a rollout")
            request_count = observation.request_features.shape[0]
            requests[row, :request_count] = observation.request_features
            request_mask_array[row, :request_count] = observation.request_mask
        if plan_request_index is not None and observation.plan_request_index is not None:
            plan_request_index[row, :count] = observation.plan_request_index

    to_tensor = lambda value, dtype: torch.as_tensor(value, dtype=dtype, device=device)
    return PaddedBatch(
        plans=to_tensor(plans, torch.float32),
        globals=to_tensor(globals_array, torch.float32),
        plan_mask=to_tensor(plan_mask, torch.bool),
        action_mask=to_tensor(action_mask, torch.bool),
        requests=None if requests is None else to_tensor(requests, torch.float32),
        request_mask=None
        if request_mask_array is None
        else to_tensor(request_mask_array, torch.bool),
        plan_request_index=None
        if plan_request_index is None
        else to_tensor(plan_request_index, torch.long),
        actions=to_tensor(actions, torch.long),
        old_log_probs=to_tensor([item.old_log_prob for item in selected], torch.float32),
        old_values=to_tensor([item.old_value for item in selected], torch.float32),
        advantages=to_tensor(advantages[indices], torch.float32),
        returns=to_tensor(returns[indices], torch.float32),
    )


@torch.no_grad()
def act(
    model: DynamicPlanActorCritic,
    observation: Mapping[str, Any] | PolicyObservation,
    device: torch.device,
    deterministic: bool = False,
) -> tuple[int, float, float]:
    parsed = observation if isinstance(observation, PolicyObservation) else parse_observation(observation)
    count = parsed.plan_features.shape[0]
    padded_count = max(count, 1)
    plans = np.zeros((1, padded_count, parsed.plan_features.shape[1]), dtype=np.float32)
    plans[0, :count] = parsed.plan_features
    plan_mask = np.zeros((1, padded_count), dtype=np.bool_)
    plan_mask[0, :count] = parsed.plan_mask
    action_mask = np.zeros((1, padded_count + 1), dtype=np.bool_)
    action_mask[0, :count] = parsed.action_mask[:count]
    action_mask[0, -1] = parsed.action_mask[count]
    requests = None
    request_mask = None
    plan_request_index = None
    if parsed.request_features is not None:
        requests = torch.as_tensor(parsed.request_features[None], device=device)
        request_mask = torch.as_tensor(parsed.request_mask[None], device=device)
    if parsed.plan_request_index is not None:
        values = np.full((1, padded_count), -1, dtype=np.int64)
        values[0, :count] = parsed.plan_request_index
        plan_request_index = torch.as_tensor(values, device=device)
    distribution, value = model.distribution_and_value(
        torch.as_tensor(plans, device=device),
        torch.as_tensor(parsed.global_features[None], device=device),
        torch.as_tensor(plan_mask, device=device),
        torch.as_tensor(action_mask, device=device),
        requests,
        request_mask,
        plan_request_index,
    )
    network_action = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
    network_index = int(network_action.item())
    environment_action = count if network_index == padded_count else network_index
    return environment_action, float(distribution.log_prob(network_action).item()), float(value.item())


def ppo_update(
    model: DynamicPlanActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBuffer,
    config: PPOConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    if not rollout.transitions:
        raise ValueError("cannot update PPO from an empty rollout")
    advantages, returns = rollout.advantages_and_returns(config)
    metrics: list[dict[str, float]] = []
    stop_early = False
    all_indices = np.arange(len(rollout), dtype=np.int64)

    for _ in range(config.ppo_epochs):
        rng.shuffle(all_indices)
        for start in range(0, len(all_indices), config.minibatch_size):
            indices = all_indices[start : start + config.minibatch_size]
            batch = collate_transitions(rollout.transitions, advantages, returns, indices, device)
            distribution, values = model.distribution_and_value(
                batch.plans,
                batch.globals,
                batch.plan_mask,
                batch.action_mask,
                batch.requests,
                batch.request_mask,
                batch.plan_request_index,
            )
            log_probs = distribution.log_prob(batch.actions)
            entropy = distribution.entropy().mean()
            log_ratio = log_probs - batch.old_log_probs
            ratio = log_ratio.exp()
            unclipped = ratio * batch.advantages
            clipped = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * batch.advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()

            if config.value_clip is None:
                value_loss = nn.functional.mse_loss(values, batch.returns)
            else:
                clipped_values = batch.old_values + (values - batch.old_values).clamp(
                    -config.value_clip, config.value_clip
                )
                raw_error = (values - batch.returns).square()
                clipped_error = (clipped_values - batch.returns).square()
                value_loss = 0.5 * torch.maximum(raw_error, clipped_error).mean()

            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > config.clip_ratio).float().mean()
            metrics.append(
                {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                    "approx_kl": float(approx_kl.item()),
                    "clip_fraction": float(clip_fraction.item()),
                    "grad_norm": float(grad_norm.item()),
                }
            )
            if config.target_kl is not None and approx_kl.item() > config.target_kl:
                stop_early = True
                break
        if stop_early:
            break

    return {
        key: float(np.mean([row[key] for row in metrics]))
        for key in metrics[0]
    } | {"epochs_completed": float(len(metrics) / max(1, int(np.ceil(len(rollout) / config.minibatch_size))))}
