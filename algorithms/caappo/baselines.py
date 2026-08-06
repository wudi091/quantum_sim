"""Rule-based joint route/construction policies used before RL."""

from __future__ import annotations

from typing import Protocol

from qnet_core.construction_catalog import RouteConstructionCandidate, candidates_by_request


class JointPlanPolicy(Protocol):
    def select(
        self,
        candidates: tuple[RouteConstructionCandidate, ...],
    ) -> dict[str, RouteConstructionCandidate]: ...


class ShortestPathLeftDeepPolicy:
    """Shortest route, with left-deep construction as the path baseline."""

    def select(self, candidates):
        grouped = candidates_by_request(candidates)
        result = {}
        for request_id, values in grouped.items():
            result[request_id] = min(
                values,
                key=lambda candidate: (
                    candidate.hop_count,
                    candidate.construction_kind != "left_deep",
                    candidate.candidate_id,
                ),
            )
        return result


class BalancedConstructionPolicy:
    """Shortest route, preferring balanced construction when available."""

    def select(self, candidates):
        grouped = candidates_by_request(candidates)
        result = {}
        for request_id, values in grouped.items():
            result[request_id] = min(
                values,
                key=lambda candidate: (
                    candidate.hop_count,
                    candidate.construction_kind != "balanced",
                    candidate.candidate_id,
                ),
            )
        return result


class MemoryAwareConstructionPolicy:
    """Shortest route with a tie-break on peak DAG live-segment pressure."""

    @staticmethod
    def _pressure(candidate: RouteConstructionCandidate) -> tuple[int, int]:
        live = 0
        peak = 0
        for operation in candidate.dag.operations:
            live += int(operation.output_segment_id is not None)
            live -= len(operation.input_segment_ids)
            peak = max(peak, live)
        return peak, len(candidate.dag.operations)

    def select(self, candidates):
        grouped = candidates_by_request(candidates)
        result = {}
        for request_id, values in grouped.items():
            result[request_id] = min(
                values,
                key=lambda candidate: (
                    candidate.hop_count,
                    self._pressure(candidate),
                    candidate.candidate_id,
                ),
            )
        return result
