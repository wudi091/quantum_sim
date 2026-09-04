"""ARC-Q rollout collection on the persistent SeQUeNCe environment."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from algorithms.routing_core.execution import (
    OnlineExecutionConfig,
    OnlineExecutionResult,
)
from qnet_core.spec import EpisodeSpec

from .environment import (
    ConstructionAwareRoutingEnvironment,
    RoutingAction,
    RoutingObservation,
)
from .policy import ARCQPolicy


@dataclass(frozen=True)
class PolicyRolloutStep:
    observation: RoutingObservation
    action: RoutingAction
    reward: float
    done: bool
    old_log_probability: float
    old_value: float
    token_count: int


@dataclass(frozen=True)
class EpisodeRollout:
    steps: tuple[PolicyRolloutStep, ...]
    execution: OnlineExecutionResult
    episode_return: float
    reward_identity_error: float


def collect_episode(
    policy: ARCQPolicy,
    episode: EpisodeSpec,
    environment_config: OnlineExecutionConfig,
    *,
    deterministic: bool = False,
) -> EpisodeRollout:
    """Collect one complete online trajectory without retaining autograd."""

    environment = ConstructionAwareRoutingEnvironment(
        episode,
        environment_config,
    )
    was_training = policy.training
    policy.eval()
    steps: list[PolicyRolloutStep] = []
    while not environment.done:
        observation = environment.observe()
        policy_started = perf_counter()
        with torch.no_grad():
            evaluation = policy.sample_action(
                observation,
                deterministic=deterministic,
            )
        policy_seconds = perf_counter() - policy_started
        transition = environment.step(
            evaluation.action,
            policy_seconds=policy_seconds,
        )
        steps.append(PolicyRolloutStep(
            observation=observation,
            action=evaluation.action,
            reward=transition.reward,
            done=transition.done,
            old_log_probability=float(evaluation.log_probability.item()),
            old_value=float(evaluation.value.item()),
            token_count=evaluation.token_count,
        ))
    if was_training:
        policy.train()
    return EpisodeRollout(
        steps=tuple(steps),
        execution=environment.result(),
        episode_return=float(sum(step.reward for step in steps)),
        reward_identity_error=environment.reward_identity_error(),
    )


__all__ = ["EpisodeRollout", "PolicyRolloutStep", "collect_episode"]
