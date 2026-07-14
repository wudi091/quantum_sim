"""Thin compatibility layer for the optional ``batchswap_reliq`` backend.

The RELIQ-backed environment intentionally exposes the same five-value
Gymnasium-style API and observation keys as :mod:`batchswap_rl.env`.  This
adapter is still useful at the boundary: it validates/copies observations,
keeps STOP as the final action, and translates the trainer's curriculum stage
object to the backend's integer stage when necessary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _first(observation: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in observation:
            return observation[key]
    raise KeyError(f"observation missing one of {keys}; got {tuple(observation)}")


def canonicalize_observation(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return the stable keys consumed by ``batchswap_rl.ppo.parse_observation``."""
    candidate = np.asarray(
        _first(observation, "candidate_features", "plan_features", "plans", "candidates"),
        dtype=np.float32,
    ).copy()
    global_features = np.asarray(
        _first(observation, "global_features", "global_state", "state_features", "state"),
        dtype=np.float32,
    ).copy()
    mask = np.asarray(
        _first(observation, "action_mask", "candidate_mask", "mask"), dtype=bool
    ).copy()
    if candidate.ndim != 2 or global_features.ndim != 1 or mask.ndim != 1:
        raise ValueError(
            "RELIQ observation requires candidate_features [K,F], "
            "global_features [G], and action_mask [K or K+1]"
        )
    if mask.shape[0] == candidate.shape[0]:
        mask = np.concatenate((mask, np.ones(1, dtype=bool)))
    if mask.shape[0] != candidate.shape[0] + 1:
        raise ValueError("action_mask must contain one entry per candidate plus STOP")
    if not mask.any():
        raise ValueError("RELIQ action mask must expose at least one legal action")

    result: dict[str, np.ndarray] = {
        "candidate_features": candidate,
        "global_features": global_features,
        "action_mask": mask,
    }
    request_key = next(
        (key for key in ("request_features", "requests") if key in observation), None
    )
    if request_key is not None:
        result["request_features"] = np.asarray(
            observation[request_key], dtype=np.float32
        ).copy()
        request_mask_key = next(
            (key for key in ("request_mask", "active_request_mask") if key in observation),
            None,
        )
        if request_mask_key is not None:
            result["request_mask"] = np.asarray(
                observation[request_mask_key], dtype=bool
            ).copy()
    return result


def _stage_index(stage: Any) -> int:
    if isinstance(stage, (int, np.integer)):
        return int(stage)
    name = str(getattr(stage, "name", "")).lower()
    return {"short": 0, "medium": 1, "long": 2}.get(name, 0)


class ReliqEnvironmentAdapter:
    """Normalize an already-created ``BatchSwapReliqEnv`` instance."""

    def __init__(self, environment: Any, factory: Any | None = None, seed: int = 0):
        self.environment = environment
        self._factory = factory
        self._seed = seed

    def __getattr__(self, name: str) -> Any:
        environment = self.__dict__.get("environment")
        if environment is None:
            raise AttributeError(name)
        return getattr(environment, name)

    def reset(self, **kwargs: Any):
        result = self.environment.reset(**kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("batchswap_reliq.reset must return (observation, info)")
        observation, info = result
        return canonicalize_observation(observation), dict(info)

    def step(self, action: int):
        result = self.environment.step(action)
        if not isinstance(result, tuple) or len(result) != 5:
            raise ValueError("batchswap_reliq.step must return (obs, reward, terminated, truncated, info)")
        observation, reward, terminated, truncated, info = result
        return (
            canonicalize_observation(observation),
            float(reward),
            bool(terminated),
            bool(truncated),
            dict(info),
        )

    def set_curriculum(self, stage: Any) -> None:
        setter = getattr(self.environment, "set_curriculum", None)
        if setter is None:
            if self._factory is not None:
                self.environment = self._factory(stage=_stage_index(stage), seed=self._seed)
            return
        try:
            setter(stage)
        except (TypeError, AttributeError, ValueError):
            setter(_stage_index(stage))


def make_reliq_env(stage: int = 0, seed: int = 0, **kwargs: Any) -> ReliqEnvironmentAdapter:
    """Lazy factory so importing ``batchswap_rl`` does not require RELIQ deps."""
    from batchswap_reliq.env import make_env

    def factory(**values: Any):
        params = dict(kwargs)
        params.update(values)
        return make_env(**params)
    return ReliqEnvironmentAdapter(
        make_env(stage=stage, seed=seed, **kwargs), factory=factory, seed=seed
    )
