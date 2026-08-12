"""Planning-side success estimates for construction candidates.

The estimator consumes only immutable episode configuration and neutral
construction DAGs.  It does not execute SeQUeNCe or inspect any simulator
state.  Generation, purification, and swap outcomes are treated as
independent conditional events, matching the one-shot candidate semantics of
the time-expanded teacher model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from qnet_core.construction_api import OperationKind
from qnet_core.construction_catalog import RouteConstructionCandidate
from qnet_core.fidelity_estimation import (
    werner_bbpssw_result,
    werner_storage_fidelity_lower_bound,
    werner_swap_fidelity,
)
from qnet_core.spec import EpisodeSpec, PhysicalConfig

from .time_expansion import build_nominal_schedule


SUCCESS_PROBABILITY_MODEL_NAME = "sequence_independent_operations_v1"


@dataclass(frozen=True, order=True)
class CandidateSuccessEstimate:
    """Expected one-shot completion probability for one candidate."""

    candidate_id: str
    probability: float
    operation_probabilities: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("success probability must lie in [0, 1]")
        if tuple(sorted(self.operation_probabilities)) != self.operation_probabilities:
            raise ValueError("operation probabilities must be sorted")
        if any(
            not 0.0 <= probability <= 1.0
            for _, probability in self.operation_probabilities
        ):
            raise ValueError("operation probabilities must lie in [0, 1]")


@dataclass(frozen=True)
class _SegmentEstimate:
    fidelity: float
    produced_slot: int
    alive: bool = True


def effective_generation_probability(physical: PhysicalConfig) -> float:
    """Return the per-edge probability used by the SeQUeNCe adapter.

    Each elementary link consists of two half-distance quantum channels.  The
    product of their transmissions is therefore the full-distance attenuation
    term below.
    """

    transmission = 10.0 ** (
        -float(physical.quantum_attenuation_db_per_m)
        * float(physical.quantum_distance_m)
        / 10.0
    )
    probability = (
        float(physical.generation_probability)
        * transmission
        * float(physical.detector_efficiency) ** 2
        * float(physical.bsm_success_probability)
    )
    return min(1.0, max(0.0, probability))


def estimate_candidate_success_probability(
    episode: EpisodeSpec,
    candidate: RouteConstructionCandidate,
) -> CandidateSuccessEstimate:
    """Estimate the probability that every terminal pair is constructed."""

    requests = {request.id: request for request in episode.requests}
    request = requests.get(candidate.request_id)
    if request is None:
        raise ValueError(
            f"candidate belongs to unknown request: {candidate.request_id}"
        )
    schedule = build_nominal_schedule(candidate)
    slots = dict(schedule.operation_slots)
    physical = episode.physical
    generation_probability = effective_generation_probability(physical)
    swap_probability = min(1.0, max(0.0, float(physical.swap_probability)))

    segments: dict[str, _SegmentEstimate] = {}
    completed: set[str] = set()
    pending = list(candidate.dag.operations)
    operation_probabilities: dict[str, float] = {}

    def stored(segment_id: str, target_slot: int) -> _SegmentEstimate:
        segment = segments[segment_id]
        storage_slots = target_slot - segment.produced_slot
        if storage_slots < 0:
            raise ValueError("segment is consumed before it is produced")
        physically_expired = storage_slots >= physical.memory_lifetime
        policy_expired = (
            request.max_storage_slots is not None
            and storage_slots > request.max_storage_slots
        )
        alive = segment.alive and not physically_expired and not policy_expired
        if not alive:
            return _SegmentEstimate(0.0, target_slot, False)
        return _SegmentEstimate(
            werner_storage_fidelity_lower_bound(
                segment.fidelity,
                storage_slots,
                physical.memory_lifetime,
            ),
            target_slot,
            True,
        )

    while pending:
        progressed = False
        for operation in tuple(pending):
            if not set(operation.predecessors).issubset(completed):
                continue
            if any(
                segment_id not in segments
                for segment_id in operation.input_segment_ids
            ):
                continue
            slot = slots[operation.op_id]
            local_probability = 1.0
            output_fidelity: float | None = None
            output_alive = True
            if operation.kind == OperationKind.GEN:
                local_probability = generation_probability
                output_fidelity = float(physical.initial_fidelity)
            elif operation.kind == OperationKind.PURIFY:
                if len(operation.input_segment_ids) != 2:
                    raise ValueError(
                        "PURIFY success estimation requires two inputs"
                    )
                inputs = tuple(
                    stored(segment_id, slot)
                    for segment_id in operation.input_segment_ids
                )
                output_alive = all(item.alive for item in inputs)
                if output_alive:
                    local_probability, output_fidelity = werner_bbpssw_result(
                        inputs[0].fidelity,
                        inputs[1].fidelity,
                        physical.swap_degradation,
                    )
                    output_alive = local_probability > 0.0
                else:
                    local_probability = 0.0
                    output_fidelity = 0.0
            elif operation.kind == OperationKind.SWAP:
                if len(operation.input_segment_ids) != 2:
                    raise ValueError("SWAP success estimation requires two inputs")
                inputs = tuple(
                    stored(segment_id, slot)
                    for segment_id in operation.input_segment_ids
                )
                output_alive = all(item.alive for item in inputs)
                local_probability = swap_probability if output_alive else 0.0
                output_fidelity = (
                    werner_swap_fidelity(
                        inputs[0].fidelity,
                        inputs[1].fidelity,
                        physical.swap_degradation,
                    )
                    if output_alive
                    else 0.0
                )
            elif operation.kind != OperationKind.RELEASE:
                raise ValueError(
                    f"unsupported construction operation: {operation.kind}"
                )

            operation_probabilities[operation.op_id] = local_probability
            if operation.output_segment_id is not None:
                if output_fidelity is None:
                    raise ValueError("output operation has no fidelity estimate")
                segments[operation.output_segment_id] = _SegmentEstimate(
                    output_fidelity,
                    slot,
                    output_alive and local_probability > 0.0,
                )
            completed.add(operation.op_id)
            pending.remove(operation)
            progressed = True
        if not progressed:
            raise ValueError("construction success graph cannot be resolved")

    terminal_alive = True
    for segment_id in candidate.all_terminal_segment_ids:
        if segment_id not in segments:
            raise ValueError(f"unknown terminal segment: {segment_id}")
        segment = segments[segment_id]
        terminal_alive = terminal_alive and stored(
            segment_id, segment.produced_slot + 1
        ).alive
    probability = (
        math.prod(operation_probabilities.values()) if terminal_alive else 0.0
    )
    return CandidateSuccessEstimate(
        candidate_id=candidate.candidate_id,
        probability=min(1.0, max(0.0, float(probability))),
        operation_probabilities=tuple(sorted(operation_probabilities.items())),
    )


def estimate_candidate_success_probabilities(
    episode: EpisodeSpec,
    candidates: Sequence[RouteConstructionCandidate],
) -> tuple[CandidateSuccessEstimate, ...]:
    estimates = []
    seen: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        if candidate.candidate_id in seen:
            raise ValueError(f"duplicate candidate ID: {candidate.candidate_id}")
        seen.add(candidate.candidate_id)
        estimates.append(estimate_candidate_success_probability(episode, candidate))
    return tuple(estimates)


def candidate_success_probability_map(
    episode: EpisodeSpec,
    candidates: Sequence[RouteConstructionCandidate],
) -> dict[str, float]:
    return {
        estimate.candidate_id: estimate.probability
        for estimate in estimate_candidate_success_probabilities(episode, candidates)
    }


__all__ = [
    "CandidateSuccessEstimate",
    "SUCCESS_PROBABILITY_MODEL_NAME",
    "candidate_success_probability_map",
    "effective_generation_probability",
    "estimate_candidate_success_probabilities",
    "estimate_candidate_success_probability",
]
