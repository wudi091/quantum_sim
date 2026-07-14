"""Non-learning policies for evaluating the pure-RL selector."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .env import BatchSwapEnv


class GreedyPolicy:
    """Prefer completion, then progress per swap-depth, age, and low cost."""

    def reset(self) -> None:
        pass

    def act(self, env: BatchSwapEnv, observation: Mapping[str, np.ndarray]) -> int:
        mask = np.asarray(observation["action_mask"], dtype=bool)
        legal = np.flatnonzero(mask[:env.stop_action])
        if not len(legal):
            return env.stop_action

        def score(action: int) -> tuple[float, ...]:
            plan = env.decode_action(int(action))
            assert plan is not None
            request = env.instance.requests[plan.request_index]
            age = env.time - request.arrival
            return (float(plan.completed),
                    plan.progress / max(plan.swap_depth, 1),
                    float(age), float(plan.progress), -float(len(plan.edges)), -float(action))

        return int(max(legal, key=score))


class QDDCAPolicy:
    """Persistent FIFO baseline restricted to one physical hop per request."""

    def reset(self) -> None:
        pass

    def act(self, env: BatchSwapEnv, observation: Mapping[str, np.ndarray]) -> int:
        mask = np.asarray(observation["action_mask"], dtype=bool)
        legal = []
        for action in np.flatnonzero(mask[:env.stop_action]):
            plan = env.decode_action(int(action))
            if plan is not None and plan.progress == 1:
                legal.append((plan.request_index, int(action)))
        return min(legal)[1] if legal else env.stop_action


class RandomValidPolicy:
    """Uniformly sample a currently legal action, including STOP."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(self.seed if seed is None else seed)

    def act(self, env: BatchSwapEnv, observation: Mapping[str, np.ndarray]) -> int:
        del env
        legal = np.flatnonzero(np.asarray(observation["action_mask"], dtype=bool))
        if not len(legal):
            raise RuntimeError("environment exposed no legal action, including STOP")
        return int(self.rng.choice(legal))


def run_policy(env: BatchSwapEnv, policy: GreedyPolicy | QDDCAPolicy | RandomValidPolicy,
               *, seed: int | None = None) -> dict[str, object]:
    observation, info = env.reset(seed=seed)
    if isinstance(policy, RandomValidPolicy):
        policy.reset(seed)
    else:
        policy.reset()
    terminated = truncated = False
    episode_return = 0.0
    while not (terminated or truncated):
        action = policy.act(env, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
    return {**info, "episode_return": episode_return,
            "terminated": terminated, "truncated": truncated}
