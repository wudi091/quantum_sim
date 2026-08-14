"""Composition root wiring construction schedules to SeQUeNCe."""

from __future__ import annotations

from .sequence_backend import SequenceBackend
from .sequence_construction_executor import SequenceConstructionExecutor
from .construction_api import ConstructionDAG
from .resource_catalog import build_resource_capacities
from .spec import EpisodeSpec

def make_sequence_construction_executor(
    spec: EpisodeSpec,
    dags: tuple[ConstructionDAG, ...],
) -> SequenceConstructionExecutor:
    """Build the event-driven construction layer on top of SeQUeNCe.

    The capacity catalogue is neutral: operation resource keys are strings and
    only the adapter interprets them as link/BSM capacities.
    """

    backend = SequenceBackend(spec)
    capacities = build_resource_capacities(spec)
    return SequenceConstructionExecutor(
        dags,
        backend,
        capacities,
        horizon_ps=spec.horizon * spec.physical.slot_duration_ps,
    )
