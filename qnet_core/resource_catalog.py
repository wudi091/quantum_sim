"""Simulator-neutral resource capacities shared by planning and execution."""

from __future__ import annotations

from .spec import EpisodeSpec


def build_resource_capacities(spec: EpisodeSpec) -> dict[str, int]:
    """Build the opaque capacity catalogue exposed to planning components.

    The returned keys deliberately contain no SeQUeNCe objects.  Both the
    construction executor and the LP teacher consume this same catalogue, so
    their resource limits cannot silently drift apart.
    """

    capacities: dict[str, int] = {}
    for raw_u, raw_v in spec.edges:
        u, v = sorted((raw_u, raw_v))
        capacities[f"link:{u}-{v}"] = spec.physical.memory_capacity
        capacities[f"genlane:{u}-{v}"] = spec.physical.max_width
        capacities[f"purify:{u}-{v}"] = 1
    for node in spec.nodes:
        capacities[f"bsm:{node}"] = 1
        degree = sum(node in edge for edge in spec.edges)
        capacities[f"memory:{node}"] = (
            spec.physical.node_memory_capacity
            if spec.physical.node_memory_capacity is not None
            else max(1, degree * spec.physical.memory_capacity)
        )
    return capacities
