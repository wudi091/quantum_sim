"""Independent implementation of the Q-CAST paper model.

This package deliberately does not depend on :mod:`qnet_core.env`.  The
authors' simulator resets all node/channel resources at the end of every
slot, and represents every physical channel explicitly; the classes here
mirror that model.
"""

from .model import (
    Channel, ChannelRef, EdgeSpec, MajorReservation, PathCandidate,
    QCastTopology, RecoveryReservation, ResidualResources, SDPair,
)
from .ext import expected_throughput, ext, propagate_distribution
from .allocation import (
    allocate_recovery_paths, eda_fixed_width, geda_allocate,
    width_first_path,
)
from .recovery import (
    LaneOutcome, connected, recover_lane, recovery_loop_edges, shortest_path_from_edges,
    xor_edges,
)
from .simulator import SimulationConfig, SlotResult, run_experiment, run_slot, sample_sd_pairs
from .topology import (
    AuthorTopologyConfig, AuthorTopologyResult, calibrate_alpha,
    generate_author_topology, generate_author_topology_with_metadata,
)

__all__ = [
    "Channel", "ChannelRef", "EdgeSpec", "MajorReservation",
    "PathCandidate", "QCastTopology", "RecoveryReservation",
    "ResidualResources", "SDPair", "expected_throughput", "ext",
    "propagate_distribution", "allocate_recovery_paths", "eda_fixed_width",
    "geda_allocate", "width_first_path",
    "LaneOutcome", "connected", "recover_lane", "recovery_loop_edges",
    "shortest_path_from_edges", "xor_edges", "SimulationConfig", "SlotResult",
    "run_experiment", "run_slot", "sample_sd_pairs",
    "AuthorTopologyConfig", "AuthorTopologyResult", "calibrate_alpha",
    "generate_author_topology", "generate_author_topology_with_metadata",
]
