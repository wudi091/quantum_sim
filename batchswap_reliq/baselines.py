"""Non-learning policies for the RELiQ-backed BatchSwap environment."""

from __future__ import annotations

from typing import Mapping
import numpy as np

from .env import BatchSwapReliqEnv


class GreedyPolicy:
    def reset(self) -> None:
        return None

    def act(self, env: BatchSwapReliqEnv, observation: Mapping[str, np.ndarray]) -> int:
        legal = np.flatnonzero(np.asarray(observation["action_mask"], dtype=bool)[:env.stop_action])
        if not len(legal):
            return env.stop_action

        def score(action: int):
            plan = env.decode_action(int(action))
            return (
                float(plan.completed),
                plan.progress / max(plan.swap_depth, 1),
                min(float(link.fidelity) for link in plan.input_links),
                -plan.swap_depth,
                -int(action),
            )

        return int(max(legal, key=score))


class QDDCAPolicy:
    """FIFO local controller preferring the smallest feasible extension.

    The physical environment exposes one farthest-feasible prefix per candidate
    topology route. If no candidate happens to stop after exactly one hop, the
    shortest available prefix is the closest legal analogue of Q-DDCA's local
    one-hop action; it must not idle forever merely because plans are batched.
    """

    def reset(self) -> None:
        return None

    def act(self, env: BatchSwapReliqEnv, observation: Mapping[str, np.ndarray]) -> int:
        mask = np.asarray(observation["action_mask"], dtype=bool)
        choices = []
        for action in np.flatnonzero(mask[:env.stop_action]):
            plan = env.decode_action(int(action))
            if plan is not None:
                request = env.instance.requests[plan.request_index]
                choices.append((
                    request.arrival,
                    plan.progress != 1,
                    plan.progress,
                    plan.swap_depth,
                    plan.request_index,
                    int(action),
                ))
        return min(choices)[-1] if choices else env.stop_action


class RandomValidPolicy:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(self.seed if seed is None else seed)

    def act(self, env: BatchSwapReliqEnv, observation: Mapping[str, np.ndarray]) -> int:
        del env
        legal = np.flatnonzero(np.asarray(observation["action_mask"], dtype=bool))
        return int(self.rng.choice(legal))
