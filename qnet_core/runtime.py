"""Composition root wiring the routing layer to the SeQUeNCe backend."""

from __future__ import annotations

from .env import SharedRoutingEnv
from .gym_env import GymConfig, SequenceGymEnv
from .scenario import ScenarioConfig, make_episode
from .sequence_backend import SequenceBackend
from .spec import EpisodeSpec


def make_sequence_env(
    spec: EpisodeSpec,
    candidate_count: int = 3,
) -> SharedRoutingEnv:
    """Build one routing environment with SeQUeNCe as its physical backend."""
    return SharedRoutingEnv(
        spec.planning,
        backend=SequenceBackend(spec),
        candidate_count=candidate_count,
    )


def _make_sequence_core(
    scenario: ScenarioConfig,
    seed: int,
    candidate_count: int,
) -> SharedRoutingEnv:
    return make_sequence_env(
        make_episode(scenario, seed),
        candidate_count=candidate_count,
    )


def make_sequence_gym_env(config: GymConfig | None = None) -> SequenceGymEnv:
    """Build the Gym wrapper without exposing SeQUeNCe to the Gym module."""
    return SequenceGymEnv(config, core_factory=_make_sequence_core)
