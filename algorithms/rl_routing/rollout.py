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
class PolicyRolloutToken:
    """One policy transition; only STOP advances the physical environment."""

    observation: RoutingObservation
    prefix_action_ids: tuple[str, ...]
    action_id: str
    reward: float
    done: bool
    old_log_probability: float
    old_value: float


@dataclass(frozen=True)
class PolicyRolloutStep:
    observation: RoutingObservation
    action: RoutingAction
    reward: float
    done: bool
    old_log_probability: float
    old_value: float
    token_count: int
    tokens: tuple[PolicyRolloutToken, ...]


@dataclass(frozen=True)
class EpisodeRollout:
    steps: tuple[PolicyRolloutStep, ...]
    execution: OnlineExecutionResult
    episode_return: float
    reward_identity_error: float
    has_value_estimates: bool

    @property
    def tokens(self) -> tuple[PolicyRolloutToken, ...]:
        return tuple(token for step in self.steps for token in step.tokens)


def collect_episode(
    policy: ARCQPolicy,
    episode: EpisodeSpec,
    environment_config: OnlineExecutionConfig,
    *,
    deterministic: bool = False,
    collect_value_estimates: bool = True,
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
                include_value=collect_value_estimates,
            )
        policy_seconds = perf_counter() - policy_started
        transition = environment.step(
            evaluation.action,
            policy_seconds=policy_seconds,
        )
        token_records = tuple(
            PolicyRolloutToken(
                observation=observation,
                prefix_action_ids=token.prefix_action_ids,
                action_id=token.action_id,
                reward=(
                    transition.reward
                    if index == len(evaluation.tokens) - 1
                    else 0.0
                ),
                done=(
                    transition.done
                    if index == len(evaluation.tokens) - 1
                    else False
                ),
                old_log_probability=float(token.log_probability.item()),
                old_value=float(token.value.item()),
            )
            for index, token in enumerate(evaluation.tokens)
        )
        steps.append(PolicyRolloutStep(
            observation=observation,
            action=evaluation.action,
            reward=transition.reward,
            done=transition.done,
            old_log_probability=float(evaluation.log_probability.item()),
            old_value=float(evaluation.value.item()),
            token_count=evaluation.token_count,
            tokens=token_records,
        ))
    if was_training:
        policy.train()
    return EpisodeRollout(
        steps=tuple(steps),
        execution=environment.result(),
        episode_return=float(sum(step.reward for step in steps)),
        reward_identity_error=environment.reward_identity_error(),
        has_value_estimates=collect_value_estimates,
    )


__all__ = [
    "EpisodeRollout",
    "PolicyRolloutStep",
    "PolicyRolloutToken",
    "collect_episode",
]
