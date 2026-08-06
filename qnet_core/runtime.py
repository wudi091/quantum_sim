"""Composition root wiring the routing layer to the SeQUeNCe backend."""

from __future__ import annotations

from .env import SharedRoutingEnv
from .gym_env import GymConfig, SequenceGymEnv
from .scenario import ScenarioConfig, make_episode
from .sequence_backend import SequenceBackend
from .sequence_construction_executor import SequenceConstructionExecutor
from .construction_api import ConstructionDAG
from .spec import EpisodeSpec


def make_sequence_env(
    spec: EpisodeSpec,
    candidate_count: int = 3,
    request_driven_generation: bool = False,
    local_candidates: bool = False,
    best_effort_allocations: bool = False,
) -> SharedRoutingEnv:
    """Build one routing environment with SeQUeNCe as its physical backend."""
    return SharedRoutingEnv(
        spec.planning,
        backend=SequenceBackend(spec),
        candidate_count=candidate_count,
        request_driven_generation=request_driven_generation,
        local_candidates=local_candidates,
        best_effort_allocations=best_effort_allocations,
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


def make_sequence_construction_executor(
    spec: EpisodeSpec,
    dags: tuple[ConstructionDAG, ...],
) -> SequenceConstructionExecutor:
    """Build the event-driven construction layer on top of SeQUeNCe.

    The capacity catalogue is neutral: operation resource keys are strings and
    only the adapter interprets them as link/BSM capacities.
    """

    backend = SequenceBackend(spec)
    capacities: dict[str, int] = {}
    for raw_u, raw_v in spec.edges:
        u, v = sorted((raw_u, raw_v))
        capacities[f"link:{u}-{v}"] = spec.physical.memory_capacity
        capacities[f"genlane:{u}-{v}"] = spec.physical.max_width
    for node in spec.nodes:
        capacities[f"bsm:{node}"] = 1
        degree = sum(node in edge for edge in spec.edges)
        capacities[f"memory:{node}"] = (
            spec.physical.node_memory_capacity
            if spec.physical.node_memory_capacity is not None
            else max(1, degree * spec.physical.memory_capacity)
        )
    return SequenceConstructionExecutor(
        dags,
        backend,
        capacities,
        horizon_ps=spec.horizon * spec.physical.slot_duration_ps,
    )
