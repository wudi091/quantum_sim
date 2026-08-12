"""Planning-facing candidate fidelity bounds supplied by the core boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from qnet_core.construction_catalog import RouteConstructionCandidate
from qnet_core.fidelity_estimation import (
    FIDELITY_MODEL_NAME,
    estimate_sequence_bds_fidelity_lower_bound,
)
from qnet_core.spec import EpisodeSpec

from .time_expansion import build_nominal_schedule


@dataclass(frozen=True, order=True)
class CandidateFidelityEstimate:
    candidate_id: str
    lower_bound: float
    terminal_bounds: tuple[tuple[str, float], ...]


def estimate_candidate_fidelity_bounds(
    episode: EpisodeSpec,
    candidates: Sequence[RouteConstructionCandidate],
) -> tuple[CandidateFidelityEstimate, ...]:
    """Estimate every candidate without executing the current test instance."""

    requests = {request.id: request for request in episode.requests}
    estimates = []
    seen: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        if candidate.candidate_id in seen:
            raise ValueError(f"duplicate candidate ID: {candidate.candidate_id}")
        seen.add(candidate.candidate_id)
        request = requests.get(candidate.request_id)
        if request is None:
            raise ValueError(
                f"candidate belongs to unknown request: {candidate.request_id}"
            )
        schedule = build_nominal_schedule(candidate)
        bound = estimate_sequence_bds_fidelity_lower_bound(
            episode.physical,
            candidate.dag,
            candidate.all_terminal_segment_ids,
            dict(schedule.operation_slots),
            max_storage_slots=request.max_storage_slots,
        )
        estimates.append(CandidateFidelityEstimate(
            candidate_id=candidate.candidate_id,
            lower_bound=bound.lower_bound,
            terminal_bounds=bound.terminal_bounds,
        ))
    return tuple(estimates)


def candidate_fidelity_estimate_map(
    episode: EpisodeSpec,
    candidates: Sequence[RouteConstructionCandidate],
) -> dict[str, float]:
    return {
        item.candidate_id: item.lower_bound
        for item in estimate_candidate_fidelity_bounds(episode, candidates)
    }


__all__ = [
    "CandidateFidelityEstimate",
    "FIDELITY_MODEL_NAME",
    "candidate_fidelity_estimate_map",
    "estimate_candidate_fidelity_bounds",
]
