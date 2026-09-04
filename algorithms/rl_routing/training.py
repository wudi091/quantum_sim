"""PPO optimization for the delay-aligned ARC-Q policy."""

from __future__ import annotations

from dataclasses import dataclass
import random
from statistics import fmean
from typing import Sequence

import torch
from torch import Tensor

from .policy import ARCQPolicy
from .rollout import EpisodeRollout, PolicyRolloutStep


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    update_epochs: int = 4
    minibatch_size: int = 16
    max_gradient_norm: float = 0.5

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.gamma != 1.0:
            raise ValueError(
                "ARC-Q fixes gamma=1 so return remains exactly delay-aligned"
            )
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must lie in [0, 1]")
        if not 0.0 < self.clip_ratio < 1.0:
            raise ValueError("clip_ratio must lie in (0, 1)")
        if self.value_loss_coefficient < 0.0:
            raise ValueError("value_loss_coefficient cannot be negative")
        if self.entropy_coefficient < 0.0:
            raise ValueError("entropy_coefficient cannot be negative")
        if self.update_epochs < 1 or self.minibatch_size < 1:
            raise ValueError("PPO update counts must be positive")
        if self.max_gradient_norm <= 0.0:
            raise ValueError("max_gradient_norm must be positive")


@dataclass(frozen=True)
class PPODiagnostics:
    sample_count: int
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    gradient_norm: float
    mean_episode_return: float
    mean_completed_requests: float
    mean_censored_latency_slots: float
    maximum_reward_identity_error: float


@dataclass(frozen=True)
class _TrainingSample:
    step: PolicyRolloutStep
    advantage: float
    return_target: float


def _advantages_for_rollout(
    rollout: EpisodeRollout,
    config: PPOConfig,
) -> tuple[list[float], list[float]]:
    advantages = [0.0] * len(rollout.steps)
    return_targets = [0.0] * len(rollout.steps)
    next_value = 0.0
    next_advantage = 0.0
    for index in range(len(rollout.steps) - 1, -1, -1):
        step = rollout.steps[index]
        continuation = 0.0 if step.done else 1.0
        temporal_difference = (
            step.reward
            + config.gamma * next_value * continuation
            - step.old_value
        )
        advantage = (
            temporal_difference
            + config.gamma
            * config.gae_lambda
            * continuation
            * next_advantage
        )
        advantages[index] = advantage
        return_targets[index] = advantage + step.old_value
        next_value = step.old_value
        next_advantage = advantage
    return advantages, return_targets


def _training_samples(
    rollouts: Sequence[EpisodeRollout],
    config: PPOConfig,
) -> list[_TrainingSample]:
    samples: list[_TrainingSample] = []
    for rollout in rollouts:
        advantages, targets = _advantages_for_rollout(rollout, config)
        samples.extend(
            _TrainingSample(step, advantage, target)
            for step, advantage, target in zip(
                rollout.steps, advantages, targets, strict=True
            )
        )
    if not samples:
        raise ValueError("at least one rollout step is required")

    raw_advantages = torch.tensor(
        [sample.advantage for sample in samples], dtype=torch.float32
    )
    if len(samples) > 1:
        deviation = float(raw_advantages.std(unbiased=False).item())
        if deviation > 1e-8:
            mean = float(raw_advantages.mean().item())
            samples = [
                _TrainingSample(
                    sample.step,
                    (sample.advantage - mean) / deviation,
                    sample.return_target,
                )
                for sample in samples
            ]
    return samples


class PPOTrainer:
    def __init__(
        self,
        policy: ARCQPolicy,
        config: PPOConfig | None = None,
    ) -> None:
        self.policy = policy
        self.config = config or PPOConfig()
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate,
        )

    def update(
        self,
        rollouts: Sequence[EpisodeRollout],
        *,
        shuffle_seed: int | None = None,
    ) -> PPODiagnostics:
        samples = _training_samples(rollouts, self.config)
        rng = random.Random(shuffle_seed)
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        approximate_kls: list[float] = []
        clip_fractions: list[float] = []
        gradient_norms: list[float] = []
        self.policy.train()

        for _ in range(self.config.update_epochs):
            indices = list(range(len(samples)))
            rng.shuffle(indices)
            for offset in range(0, len(indices), self.config.minibatch_size):
                batch = [
                    samples[index]
                    for index in indices[
                        offset:offset + self.config.minibatch_size
                    ]
                ]
                batch_policy_losses: list[Tensor] = []
                batch_value_losses: list[Tensor] = []
                batch_entropies: list[Tensor] = []
                batch_kls: list[Tensor] = []
                batch_clipped: list[Tensor] = []

                for sample in batch:
                    evaluation = self.policy.evaluate_action(
                        sample.step.observation,
                        sample.step.action,
                    )
                    old_log_probability = torch.tensor(
                        sample.step.old_log_probability,
                        dtype=evaluation.log_probability.dtype,
                        device=self.policy.device,
                    )
                    advantage = torch.tensor(
                        sample.advantage,
                        dtype=evaluation.log_probability.dtype,
                        device=self.policy.device,
                    )
                    return_target = torch.tensor(
                        sample.return_target,
                        dtype=evaluation.value.dtype,
                        device=self.policy.device,
                    )
                    log_ratio = (
                        evaluation.log_probability - old_log_probability
                    )
                    ratio = log_ratio.exp()
                    clipped_ratio = ratio.clamp(
                        1.0 - self.config.clip_ratio,
                        1.0 + self.config.clip_ratio,
                    )
                    surrogate = torch.minimum(
                        ratio * advantage,
                        clipped_ratio * advantage,
                    )
                    batch_policy_losses.append(-surrogate)
                    batch_value_losses.append(
                        (evaluation.value - return_target).square()
                    )
                    batch_entropies.append(evaluation.entropy)
                    batch_kls.append((ratio - 1.0) - log_ratio)
                    batch_clipped.append(
                        (torch.abs(ratio - 1.0) > self.config.clip_ratio).float()
                    )

                policy_loss = torch.stack(batch_policy_losses).mean()
                value_loss = torch.stack(batch_value_losses).mean()
                entropy = torch.stack(batch_entropies).mean()
                total_loss = (
                    policy_loss
                    + self.config.value_loss_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_gradient_norm,
                )
                self.optimizer.step()

                policy_losses.append(float(policy_loss.detach().item()))
                value_losses.append(float(value_loss.detach().item()))
                entropies.append(float(entropy.detach().item()))
                approximate_kls.append(float(
                    torch.stack(batch_kls).mean().detach().item()
                ))
                clip_fractions.append(float(
                    torch.stack(batch_clipped).mean().detach().item()
                ))
                gradient_norms.append(float(gradient_norm.detach().item()))

        mean_latency_slots = fmean(
            rollout.execution.metrics["mean_censored_latency_ps"]
            / rollout.execution.episode.physical.slot_duration_ps
            for rollout in rollouts
        )
        return PPODiagnostics(
            sample_count=len(samples),
            policy_loss=fmean(policy_losses),
            value_loss=fmean(value_losses),
            entropy=fmean(entropies),
            approximate_kl=fmean(approximate_kls),
            clip_fraction=fmean(clip_fractions),
            gradient_norm=fmean(gradient_norms),
            mean_episode_return=fmean(
                rollout.episode_return for rollout in rollouts
            ),
            mean_completed_requests=fmean(
                rollout.execution.metrics["completed_requests"]
                for rollout in rollouts
            ),
            mean_censored_latency_slots=mean_latency_slots,
            maximum_reward_identity_error=max(
                abs(rollout.reward_identity_error) for rollout in rollouts
            ),
        )


__all__ = ["PPOConfig", "PPODiagnostics", "PPOTrainer"]
