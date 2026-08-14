"""Deterministic hard-constraint decoding for continuous candidate scores."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import json
import math
from pathlib import Path
import random
from typing import Mapping, Sequence

from .milp_oracle import DiscreteOracleSolution
from .time_expansion import (
    TimeExpandedCandidate,
    TimeExpansionResult,
    normalize_reserved_usage,
)


@dataclass(frozen=True)
class DecodedFeasibilityReport:
    feasible: bool
    violations: tuple[str, ...]
    selected_request_count: int
    used_resource_slot_count: int


@dataclass(frozen=True)
class HardDecoderSolution:
    variables: tuple[TimeExpandedCandidate, ...]
    scores: tuple[float, ...]
    request_ids: tuple[str, ...]
    selected_variables: tuple[TimeExpandedCandidate, ...]
    feasibility: DecodedFeasibilityReport
    beam_width: int
    random_restarts: int
    local_search_iterations: int
    search_strategy: str = "beam_local_search"
    support_variable_count: int = 0

    @property
    def completed_request_count(self) -> int:
        return len(self.selected_variables)

    @property
    def total_completion_latency(self) -> float:
        return float(sum(
            variable.completion_latency for variable in self.selected_variables
        ))

    @property
    def expected_completed_request_mass(self) -> float:
        return float(sum(
            variable.expected_success_probability
            for variable in self.selected_variables
        ))

    @property
    def expected_total_completion_latency(self) -> float:
        return float(sum(
            variable.expected_success_probability * variable.completion_latency
            for variable in self.selected_variables
        ))

    @property
    def total_selected_score(self) -> float:
        score_by_id = {
            variable.variable_id: score
            for variable, score in zip(self.variables, self.scores)
        }
        return float(sum(
            score_by_id[variable.variable_id]
            for variable in self.selected_variables
        ))

    @property
    def final_values(self) -> dict[str, int]:
        selected = {
            variable.variable_id for variable in self.selected_variables
        }
        return {
            variable.variable_id: int(variable.variable_id in selected)
            for variable in self.variables
        }

    @property
    def selected_by_request(self) -> dict[str, TimeExpandedCandidate]:
        return {
            variable.request_id: variable
            for variable in self.selected_variables
        }

    @property
    def rejected_request_ids(self) -> tuple[str, ...]:
        selected_requests = {
            variable.request_id for variable in self.selected_variables
        }
        return tuple(
            request_id for request_id in self.request_ids
            if request_id not in selected_requests
        )


@dataclass(frozen=True)
class DecoderMILPGapReport:
    variable_count: int
    decoder_feasible: bool
    decoder_completed_request_count: int
    discrete_completed_request_count: int
    decoder_expected_completed_request_mass: float
    discrete_expected_completed_request_mass: float
    throughput_absolute_loss: float
    throughput_relative_loss: float | None
    throughput_is_optimal: bool
    latency_is_comparable: bool
    decoder_total_completion_latency: float
    discrete_total_completion_latency: float
    latency_absolute_gap: float | None
    latency_relative_gap: float | None
    decoder_total_selected_score: float
    beam_width: int
    random_restarts: int
    local_search_iterations: int
    decoder_selected_variable_ids: tuple[str, ...]
    discrete_selected_variable_ids: tuple[str, ...]


def _usage_for(
    variables: Sequence[TimeExpandedCandidate],
) -> dict[tuple[str, int], int]:
    usage: dict[tuple[str, int], int] = {}
    for variable in variables:
        for item in variable.resource_usage:
            key = (item.resource_id, item.slot)
            usage[key] = usage.get(key, 0) + item.amount
    return usage


def validate_decoded_selection(
    selected: Sequence[TimeExpandedCandidate],
    resource_capacities: Mapping[str, int],
    reserved_usage: Mapping[tuple[str, int], int] | None = None,
) -> DecodedFeasibilityReport:
    """Check request uniqueness and every resource--time capacity."""

    capacities = {str(key): int(value) for key, value in resource_capacities.items()}
    violations: list[str] = []
    seen_requests: set[str] = set()
    for variable in selected:
        if variable.request_id in seen_requests:
            violations.append(f"request selected twice: {variable.request_id}")
        seen_requests.add(variable.request_id)
    usage = normalize_reserved_usage(reserved_usage, capacities)
    for key, amount in _usage_for(selected).items():
        usage[key] = usage.get(key, 0) + amount
    for (resource_id, slot), amount in sorted(usage.items()):
        if resource_id not in capacities:
            violations.append(f"missing capacity: {resource_id}")
        elif amount > capacities[resource_id]:
            violations.append(
                f"capacity exceeded: {resource_id}@{slot}: "
                f"{amount}>{capacities[resource_id]}"
            )
    return DecodedFeasibilityReport(
        feasible=not violations,
        violations=tuple(violations),
        selected_request_count=len(seen_requests),
        used_resource_slot_count=len(usage),
    )


class _PackingState:
    def __init__(
        self,
        capacities: Mapping[str, int],
        reserved_usage: Mapping[tuple[str, int], int] | None = None,
    ):
        self.capacities = {str(key): int(value) for key, value in capacities.items()}
        if any(value < 0 for value in self.capacities.values()):
            raise ValueError("resource capacities cannot be negative")
        self.selected: dict[str, TimeExpandedCandidate] = {}
        self.usage = normalize_reserved_usage(reserved_usage, self.capacities)
        self.reserved_usage = dict(self.usage)

    def can_add(self, variable: TimeExpandedCandidate) -> bool:
        if variable.request_id in self.selected:
            return False
        delta: dict[tuple[str, int], int] = {}
        for item in variable.resource_usage:
            if item.resource_id not in self.capacities:
                raise ValueError(f"missing capacity for resource: {item.resource_id}")
            key = (item.resource_id, item.slot)
            delta[key] = delta.get(key, 0) + item.amount
        return all(
            self.usage.get(key, 0) + amount
            <= self.capacities[key[0]]
            for key, amount in delta.items()
        )

    def add(self, variable: TimeExpandedCandidate) -> None:
        if not self.can_add(variable):
            raise ValueError(f"cannot add infeasible candidate: {variable.variable_id}")
        self.selected[variable.request_id] = variable
        for item in variable.resource_usage:
            key = (item.resource_id, item.slot)
            self.usage[key] = self.usage.get(key, 0) + item.amount

    def clone(self) -> "_PackingState":
        cloned = _PackingState(self.capacities, self.reserved_usage)
        cloned.selected = dict(self.selected)
        cloned.usage = dict(self.usage)
        return cloned


def greedy_feasible_projection(
    expanded: TimeExpansionResult | Sequence[TimeExpandedCandidate],
    resource_capacities: Mapping[str, int],
    scores: Mapping[str, float] | Sequence[float],
    *,
    request_ids: Sequence[str] | None = None,
    reserved_usage: Mapping[tuple[str, int], int] | None = None,
    support_tolerance: float = 1e-9,
) -> HardDecoderSolution:
    """Project scores with one deterministic feasibility scan.

    Candidates are visited only in descending score order.  A candidate is
    accepted exactly when its request is still unselected and adding its
    resource--slot usage remains within capacity.  There is deliberately no
    beam search, restart, replacement, or local improvement.  ``variable_id``
    is used only to make equal-score ordering reproducible.
    """

    if support_tolerance < 0.0:
        raise ValueError("support_tolerance cannot be negative")
    raw_variables = (
        expanded.variables
        if isinstance(expanded, TimeExpansionResult)
        else expanded
    )
    variables = tuple(sorted(
        raw_variables,
        key=lambda item: item.variable_id,
    ))
    if isinstance(scores, Mapping):
        missing = [
            variable.variable_id
            for variable in variables
            if variable.variable_id not in scores
        ]
        if missing:
            raise ValueError(f"missing projection score: {missing[0]}")
        ordered_scores = tuple(
            float(scores[variable.variable_id]) for variable in variables
        )
    else:
        ordered_scores = tuple(float(value) for value in scores)
        if len(ordered_scores) != len(variables):
            raise ValueError("projection score vector has the wrong length")
    if any(
        not math.isfinite(score) or not 0.0 <= score <= 1.0
        for score in ordered_scores
    ):
        raise ValueError("projection scores must be finite and lie in [0, 1]")
    score_by_id = {
        variable.variable_id: score
        for variable, score in zip(variables, ordered_scores)
    }

    variable_request_ids = {variable.request_id for variable in variables}
    if request_ids is None:
        ordered_request_ids = tuple(sorted(variable_request_ids))
    else:
        ordered_request_ids = tuple(str(item) for item in request_ids)
        if len(set(ordered_request_ids)) != len(ordered_request_ids):
            raise ValueError("request_ids must be unique")
        missing_requests = variable_request_ids - set(ordered_request_ids)
        if missing_requests:
            raise ValueError(
                "candidate belongs to undeclared request: "
                f"{sorted(missing_requests)[0]}"
            )

    ranked = tuple(sorted(
        (
            variable
            for variable in variables
            if score_by_id[variable.variable_id] > support_tolerance
        ),
        key=lambda variable: (
            -score_by_id[variable.variable_id],
            variable.variable_id,
        ),
    ))
    state = _PackingState(resource_capacities, reserved_usage)
    for variable in ranked:
        if state.can_add(variable):
            state.add(variable)

    selected_variables = tuple(sorted(
        state.selected.values(),
        key=lambda variable: variable.variable_id,
    ))
    feasibility = validate_decoded_selection(
        selected_variables,
        resource_capacities,
        reserved_usage,
    )
    if not feasibility.feasible:
        raise RuntimeError(
            "greedy projection produced an infeasible plan: "
            f"{feasibility.violations[0]}"
        )
    return HardDecoderSolution(
        variables=variables,
        scores=ordered_scores,
        request_ids=ordered_request_ids,
        selected_variables=selected_variables,
        feasibility=feasibility,
        beam_width=1,
        random_restarts=0,
        local_search_iterations=0,
        search_strategy="score_order_greedy",
        support_variable_count=len(ranked),
    )


class HardConstraintDecoder:
    """Beam rounding, deterministic restarts, and bounded local improvement."""

    def __init__(
        self,
        *,
        beam_width: int = 512,
        random_restarts: int = 512,
        random_seed: int = 0,
        max_one_drop_iterations: int = 100,
        max_pair_exchange_iterations: int = 5,
        pair_exchange_request_limit: int = 8,
        pair_exchange_candidate_limit: int = 8,
        scalable_variable_threshold: int = 5_000,
        scalable_random_restarts: int = 32,
        support_tolerance: float = 1e-9,
    ):
        if beam_width < 1:
            raise ValueError("beam_width must be positive")
        if random_restarts < 0:
            raise ValueError("random_restarts cannot be negative")
        if max_one_drop_iterations < 0:
            raise ValueError("max_one_drop_iterations cannot be negative")
        if max_pair_exchange_iterations < 0:
            raise ValueError("max_pair_exchange_iterations cannot be negative")
        if pair_exchange_request_limit < 0:
            raise ValueError("pair_exchange_request_limit cannot be negative")
        if pair_exchange_candidate_limit < 1:
            raise ValueError("pair_exchange_candidate_limit must be positive")
        if scalable_variable_threshold < 1:
            raise ValueError("scalable_variable_threshold must be positive")
        if scalable_random_restarts < 0:
            raise ValueError("scalable_random_restarts cannot be negative")
        if support_tolerance < 0:
            raise ValueError("support_tolerance cannot be negative")
        self.beam_width = int(beam_width)
        self.random_restarts = int(random_restarts)
        self.random_seed = int(random_seed)
        self.max_one_drop_iterations = int(max_one_drop_iterations)
        self.max_pair_exchange_iterations = int(max_pair_exchange_iterations)
        self.pair_exchange_request_limit = int(pair_exchange_request_limit)
        self.pair_exchange_candidate_limit = int(pair_exchange_candidate_limit)
        self.scalable_variable_threshold = int(scalable_variable_threshold)
        self.scalable_random_restarts = int(scalable_random_restarts)
        self.support_tolerance = float(support_tolerance)

    @staticmethod
    def _quality(
        selected: Mapping[str, TimeExpandedCandidate],
        score_by_id: Mapping[str, float],
    ) -> tuple[object, ...]:
        variables = tuple(selected.values())
        return (
            -sum(
                variable.expected_success_probability
                for variable in variables
            ),
            sum(
                variable.expected_success_probability
                * variable.completion_latency
                for variable in variables
            ),
            -sum(score_by_id[variable.variable_id] for variable in variables),
            -len(variables),
            tuple(sorted(variable.variable_id for variable in variables)),
        )

    @staticmethod
    def _state_from(
        selected: Mapping[str, TimeExpandedCandidate],
        capacities: Mapping[str, int],
        reserved_usage: Mapping[tuple[str, int], int] | None = None,
    ) -> _PackingState:
        state = _PackingState(capacities, reserved_usage)
        for variable in sorted(selected.values(), key=lambda item: item.variable_id):
            state.add(variable)
        return state

    @staticmethod
    def _candidate_pressure(
        variable: TimeExpandedCandidate,
        capacities: Mapping[str, int],
    ) -> float:
        return float(sum(
            item.amount / capacities[item.resource_id]
            for item in variable.resource_usage
        ))

    @staticmethod
    def _beam_rank(
        state: _PackingState,
        score_by_id: Mapping[str, float],
    ) -> tuple[object, ...]:
        tight_count = 0
        pressure = 0.0
        for (resource_id, _slot), amount in state.usage.items():
            ratio = amount / state.capacities[resource_id]
            pressure += ratio * ratio
            if amount >= state.capacities[resource_id]:
                tight_count += 1
        variables = tuple(state.selected.values())
        return (
            -sum(
                variable.expected_success_probability
                for variable in variables
            ),
            tight_count,
            pressure,
            sum(
                variable.expected_success_probability
                * variable.completion_latency
                for variable in variables
            ),
            -sum(score_by_id[variable.variable_id] for variable in variables),
            tuple(sorted(variable.variable_id for variable in variables)),
        )

    def _prune_beam(
        self,
        states: Sequence[_PackingState],
        score_by_id: Mapping[str, float],
    ) -> list[_PackingState]:
        unique: dict[tuple[str, ...], _PackingState] = {}
        for state in states:
            key = tuple(sorted(
                variable.variable_id for variable in state.selected.values()
            ))
            unique.setdefault(key, state)
        return sorted(
            unique.values(),
            key=lambda state: self._beam_rank(state, score_by_id),
        )[:self.beam_width]

    def _beam_selection(
        self,
        request_order: Sequence[str],
        candidates_by_request: Mapping[str, Sequence[TimeExpandedCandidate]],
        capacities: Mapping[str, int],
        score_by_id: Mapping[str, float],
        reserved_usage: Mapping[tuple[str, int], int] | None = None,
    ) -> dict[str, TimeExpandedCandidate]:
        beam = [_PackingState(capacities, reserved_usage)]
        for request_id in request_order:
            expanded: list[_PackingState] = []
            for state in beam:
                expanded.append(state)
                for variable in candidates_by_request[request_id]:
                    if state.can_add(variable):
                        child = state.clone()
                        child.add(variable)
                        expanded.append(child)
            beam = self._prune_beam(expanded, score_by_id)
        return dict(min(
            beam,
            key=lambda state: self._quality(state.selected, score_by_id),
        ).selected)

    def _randomized_selection(
        self,
        initial: Mapping[str, TimeExpandedCandidate],
        candidates_by_request: Mapping[str, Sequence[TimeExpandedCandidate]],
        request_mass: Mapping[str, float],
        capacities: Mapping[str, int],
        score_by_id: Mapping[str, float],
        reserved_usage: Mapping[tuple[str, int], int] | None = None,
    ) -> dict[str, TimeExpandedCandidate]:
        best = dict(initial)
        best_quality = self._quality(best, score_by_id)
        rng = random.Random(self.random_seed)
        request_ids = tuple(sorted(candidates_by_request))
        min_pressure = {
            request_id: min(
                self._candidate_pressure(variable, capacities)
                for variable in candidates_by_request[request_id]
            )
            for request_id in request_ids
        }
        min_latency = {
            request_id: min(
                variable.completion_latency
                for variable in candidates_by_request[request_id]
            )
            for request_id in request_ids
        }
        for restart in range(self.random_restarts):
            mode = restart % 4
            noise = {request_id: rng.random() for request_id in request_ids}
            if mode == 0:
                order = sorted(request_ids, key=lambda request_id: (
                    -request_mass[request_id], noise[request_id]
                ))
            elif mode == 1:
                order = sorted(request_ids, key=lambda request_id: (
                    min_pressure[request_id], noise[request_id]
                ))
            elif mode == 2:
                order = sorted(request_ids, key=lambda request_id: (
                    min_latency[request_id], noise[request_id]
                ))
            else:
                order = sorted(request_ids, key=lambda request_id: noise[request_id])
            state = _PackingState(capacities, reserved_usage)
            for request_id in order:
                candidate_noise = {
                    variable.variable_id: rng.random()
                    for variable in candidates_by_request[request_id]
                }
                candidate_mode = (restart + len(state.selected)) % 4
                if candidate_mode == 0:
                    key = lambda variable: (
                        -score_by_id[variable.variable_id],
                        candidate_noise[variable.variable_id],
                    )
                elif candidate_mode == 1:
                    key = lambda variable: (
                        variable.completion_latency,
                        candidate_noise[variable.variable_id],
                    )
                elif candidate_mode == 2:
                    key = lambda variable: (
                        self._candidate_pressure(variable, capacities),
                        candidate_noise[variable.variable_id],
                    )
                else:
                    key = lambda variable: candidate_noise[variable.variable_id]
                for variable in sorted(candidates_by_request[request_id], key=key):
                    if state.can_add(variable):
                        state.add(variable)
                        break
            quality = self._quality(state.selected, score_by_id)
            if quality < best_quality:
                best = dict(state.selected)
                best_quality = quality
        return best

    def _scalable_selection(
        self,
        candidates_by_request: Mapping[str, Sequence[TimeExpandedCandidate]],
        request_mass: Mapping[str, float],
        capacities: Mapping[str, int],
        score_by_id: Mapping[str, float],
        reserved_usage: Mapping[tuple[str, int], int] | None = None,
    ) -> tuple[dict[str, TimeExpandedCandidate], int]:
        """Decode a large LP without cloning a beam over all time shifts.

        Positive LP-support variables provide the first integral proposal.
        Every trial can then use the complete candidate set to admit requests
        omitted by rounding.  Deterministic pressure/latency orders and seeded
        randomized orders reduce ordering bias while every insertion is checked
        against the original resource--time capacities.
        """

        request_ids = tuple(sorted(candidates_by_request))
        pressure = {
            variable.variable_id: self._candidate_pressure(
                variable, capacities
            )
            for variables in candidates_by_request.values()
            for variable in variables
        }
        rankings: dict[
            tuple[str, str], tuple[TimeExpandedCandidate, ...]
        ] = {}
        minimum_pressure: dict[str, float] = {}
        minimum_latency: dict[str, int] = {}
        for request_id, variables in candidates_by_request.items():
            rankings[(request_id, "score")] = tuple(sorted(
                variables,
                key=lambda variable: (
                    -score_by_id[variable.variable_id],
                    -variable.expected_success_probability,
                    variable.completion_latency,
                    pressure[variable.variable_id],
                    variable.variable_id,
                ),
            ))
            rankings[(request_id, "latency")] = tuple(sorted(
                variables,
                key=lambda variable: (
                    -variable.expected_success_probability,
                    variable.expected_success_probability
                    * variable.completion_latency,
                    pressure[variable.variable_id],
                    -score_by_id[variable.variable_id],
                    variable.variable_id,
                ),
            ))
            rankings[(request_id, "pressure")] = tuple(sorted(
                variables,
                key=lambda variable: (
                    -variable.expected_success_probability,
                    pressure[variable.variable_id],
                    variable.completion_latency,
                    -score_by_id[variable.variable_id],
                    variable.variable_id,
                ),
            ))
            minimum_pressure[request_id] = min(
                pressure[variable.variable_id] for variable in variables
            )
            minimum_latency[request_id] = min(
                variable.completion_latency for variable in variables
            )

        support_variables = tuple(sorted(
            (
                variable
                for variables in candidates_by_request.values()
                for variable in variables
                if score_by_id[variable.variable_id] > self.support_tolerance
            ),
            key=lambda variable: (
                -score_by_id[variable.variable_id],
                -variable.expected_success_probability,
                variable.completion_latency,
                variable.variable_id,
            ),
        ))
        orders: list[tuple[str, ...]] = [
            tuple(sorted(request_ids, key=lambda request_id: (
                -request_mass[request_id],
                minimum_pressure[request_id],
                minimum_latency[request_id],
                request_id,
            ))),
            tuple(sorted(request_ids, key=lambda request_id: (
                minimum_pressure[request_id],
                -request_mass[request_id],
                minimum_latency[request_id],
                request_id,
            ))),
            tuple(sorted(request_ids, key=lambda request_id: (
                minimum_latency[request_id],
                minimum_pressure[request_id],
                -request_mass[request_id],
                request_id,
            ))),
        ]
        rng = random.Random(self.random_seed)
        for _ in range(self.scalable_random_restarts):
            noise = {request_id: rng.random() for request_id in request_ids}
            orders.append(tuple(sorted(
                request_ids,
                key=lambda request_id: noise[request_id],
            )))

        best: dict[str, TimeExpandedCandidate] = {}
        best_quality = self._quality(best, score_by_id)
        for order in orders:
            for ranking_kind in ("score", "latency", "pressure"):
                for support_first in (True, False):
                    state = _PackingState(capacities, reserved_usage)
                    if support_first:
                        for variable in support_variables:
                            if state.can_add(variable):
                                state.add(variable)
                    for request_id in order:
                        if request_id in state.selected:
                            continue
                        for variable in rankings[(request_id, ranking_kind)]:
                            if state.can_add(variable):
                                state.add(variable)
                                break
                    quality = self._quality(state.selected, score_by_id)
                    if quality < best_quality:
                        best = dict(state.selected)
                        best_quality = quality
        return best, len(support_variables)

    def _one_drop_improve(
        self,
        selected: dict[str, TimeExpandedCandidate],
        request_order: Sequence[str],
        candidates_by_request: Mapping[str, Sequence[TimeExpandedCandidate]],
        request_mass: Mapping[str, float],
        capacities: Mapping[str, int],
        score_by_id: Mapping[str, float],
        reserved_usage: Mapping[tuple[str, int], int] | None = None,
    ) -> tuple[dict[str, TimeExpandedCandidate], int]:
        iterations = 0
        while iterations < self.max_one_drop_iterations and selected:
            best_quality = self._quality(selected, score_by_id)
            best_selection: dict[str, TimeExpandedCandidate] | None = None
            drop_order = sorted(selected, key=lambda request_id: (
                request_mass.get(request_id, 0.0),
                -selected[request_id].completion_latency,
                request_id,
            ))
            unselected = [
                request_id for request_id in request_order
                if request_id not in selected
            ]
            for dropped in drop_order:
                base = {
                    request_id: variable
                    for request_id, variable in selected.items()
                    if request_id != dropped
                }
                trial = self._state_from(base, capacities, reserved_usage)
                for request_id in (*unselected, dropped):
                    for variable in candidates_by_request[request_id]:
                        if trial.can_add(variable):
                            trial.add(variable)
                            break
                quality = self._quality(trial.selected, score_by_id)
                if quality < best_quality:
                    best_quality = quality
                    best_selection = dict(trial.selected)
            if best_selection is None:
                break
            selected = best_selection
            iterations += 1
        return selected, iterations

    def _candidate_shortlist(
        self,
        candidates: Sequence[TimeExpandedCandidate],
        score_by_id: Mapping[str, float],
    ) -> tuple[TimeExpandedCandidate, ...]:
        rankings = (
            sorted(candidates, key=lambda variable: (
                -score_by_id[variable.variable_id],
                -variable.expected_success_probability,
                variable.completion_latency,
                variable.variable_id,
            )),
            sorted(candidates, key=lambda variable: (
                -variable.expected_success_probability,
                variable.expected_success_probability
                * variable.completion_latency,
                -score_by_id[variable.variable_id],
                variable.variable_id,
            )),
        )
        selected: list[TimeExpandedCandidate] = []
        seen: set[str] = set()
        for variable in itertools.chain.from_iterable(rankings):
            if variable.variable_id in seen:
                continue
            selected.append(variable)
            seen.add(variable.variable_id)
            if len(selected) >= self.pair_exchange_candidate_limit:
                break
        return tuple(selected)

    def _pair_improve(
        self,
        selected: dict[str, TimeExpandedCandidate],
        request_order: Sequence[str],
        candidates_by_request: Mapping[str, Sequence[TimeExpandedCandidate]],
        capacities: Mapping[str, int],
        score_by_id: Mapping[str, float],
        reserved_usage: Mapping[tuple[str, int], int] | None = None,
    ) -> tuple[dict[str, TimeExpandedCandidate], int]:
        iterations = 0
        while iterations < self.max_pair_exchange_iterations and len(selected) >= 2:
            best_quality = self._quality(selected, score_by_id)
            best_selection: dict[str, TimeExpandedCandidate] | None = None
            selected_requests = tuple(sorted(selected))
            unselected = [
                request_id for request_id in request_order
                if request_id not in selected
            ][:self.pair_exchange_request_limit]
            for left_drop, right_drop in itertools.combinations(selected_requests, 2):
                base = {
                    request_id: variable
                    for request_id, variable in selected.items()
                    if request_id not in {left_drop, right_drop}
                }
                base_state = self._state_from(
                    base,
                    capacities,
                    reserved_usage,
                )
                available = tuple(dict.fromkeys((
                    left_drop, right_drop, *unselected
                )))
                for left_request, right_request in itertools.combinations(available, 2):
                    for left_variable in self._candidate_shortlist(
                        candidates_by_request[left_request], score_by_id
                    ):
                        if not base_state.can_add(left_variable):
                            continue
                        left_state = base_state.clone()
                        left_state.add(left_variable)
                        for right_variable in self._candidate_shortlist(
                            candidates_by_request[right_request], score_by_id
                        ):
                            if not left_state.can_add(right_variable):
                                continue
                            trial = left_state.clone()
                            trial.add(right_variable)
                            quality = self._quality(trial.selected, score_by_id)
                            if quality < best_quality:
                                best_quality = quality
                                best_selection = dict(trial.selected)
            if best_selection is None:
                break
            selected = best_selection
            iterations += 1
        return selected, iterations

    def _refine_candidates(
        self,
        selected: dict[str, TimeExpandedCandidate],
        candidates_by_request: Mapping[str, Sequence[TimeExpandedCandidate]],
        capacities: Mapping[str, int],
        score_by_id: Mapping[str, float],
        reserved_usage: Mapping[tuple[str, int], int] | None = None,
    ) -> dict[str, TimeExpandedCandidate]:
        for request_id in sorted(selected):
            current = selected[request_id]
            base = {
                other_id: variable
                for other_id, variable in selected.items()
                if other_id != request_id
            }
            state = self._state_from(base, capacities, reserved_usage)
            feasible = [
                variable for variable in candidates_by_request[request_id]
                if state.can_add(variable)
            ]
            if not feasible:
                continue
            replacement = min(feasible, key=lambda variable: (
                -variable.expected_success_probability,
                variable.expected_success_probability
                * variable.completion_latency,
                -score_by_id[variable.variable_id],
                variable.variable_id,
            ))
            if (
                -replacement.expected_success_probability,
                replacement.expected_success_probability
                * replacement.completion_latency,
                -score_by_id[replacement.variable_id],
                replacement.variable_id,
            ) < (
                -current.expected_success_probability,
                current.expected_success_probability
                * current.completion_latency,
                -score_by_id[current.variable_id],
                current.variable_id,
            ):
                selected[request_id] = replacement
        return selected

    def decode(
        self,
        expanded: TimeExpansionResult | Sequence[TimeExpandedCandidate],
        resource_capacities: Mapping[str, int],
        scores: Mapping[str, float] | Sequence[float],
        *,
        request_ids: Sequence[str] | None = None,
        reserved_usage: Mapping[tuple[str, int], int] | None = None,
    ) -> HardDecoderSolution:
        raw_variables = (
            expanded.variables if isinstance(expanded, TimeExpansionResult)
            else expanded
        )
        variables = tuple(sorted(raw_variables, key=lambda item: item.variable_id))
        if isinstance(scores, Mapping):
            missing = [
                variable.variable_id for variable in variables
                if variable.variable_id not in scores
            ]
            if missing:
                raise ValueError(f"missing decoder score: {missing[0]}")
            ordered_scores = tuple(
                float(scores[variable.variable_id]) for variable in variables
            )
        else:
            ordered_scores = tuple(float(value) for value in scores)
            if len(ordered_scores) != len(variables):
                raise ValueError("decoder score vector has the wrong length")
        if any(not 0.0 <= score <= 1.0 for score in ordered_scores):
            raise ValueError("decoder scores must lie in [0, 1]")
        score_by_id = {
            variable.variable_id: score
            for variable, score in zip(variables, ordered_scores)
        }

        candidates_by_request: dict[str, list[TimeExpandedCandidate]] = {}
        for variable in variables:
            candidates_by_request.setdefault(variable.request_id, []).append(variable)
        for request_id in candidates_by_request:
            candidates_by_request[request_id].sort(key=lambda variable: (
                -score_by_id[variable.variable_id],
                -variable.expected_success_probability,
                variable.completion_latency,
                variable.completion_slot,
                variable.variable_id,
            ))
        variable_request_ids = set(candidates_by_request)
        if request_ids is None:
            ordered_request_ids = tuple(sorted(variable_request_ids))
        else:
            ordered_request_ids = tuple(dict.fromkeys(str(item) for item in request_ids))
            if len(ordered_request_ids) != len(tuple(request_ids)):
                raise ValueError("request_ids must be unique")
            missing_requests = variable_request_ids - set(ordered_request_ids)
            if missing_requests:
                raise ValueError(
                    f"candidate belongs to undeclared request: {sorted(missing_requests)[0]}"
                )
        request_mass = {
            request_id: min(1.0, sum(
                variable.expected_success_probability
                * score_by_id[variable.variable_id]
                for variable in request_variables
            ))
            for request_id, request_variables in candidates_by_request.items()
        }
        request_order = tuple(sorted(candidates_by_request, key=lambda request_id: (
            -request_mass[request_id],
            len(candidates_by_request[request_id]),
            min(
                variable.completion_latency
                for variable in candidates_by_request[request_id]
            ),
            request_id,
        )))

        if len(variables) >= self.scalable_variable_threshold:
            selected, support_variable_count = self._scalable_selection(
                candidates_by_request,
                request_mass,
                resource_capacities,
                score_by_id,
                reserved_usage,
            )
            one_drop_iterations = 0
            pair_iterations = 0
            search_strategy = "lp_support_multistart"
        else:
            selected = self._beam_selection(
                request_order,
                candidates_by_request,
                resource_capacities,
                score_by_id,
                reserved_usage,
            )
            selected = self._randomized_selection(
                selected,
                candidates_by_request,
                request_mass,
                resource_capacities,
                score_by_id,
                reserved_usage,
            )
            selected, one_drop_iterations = self._one_drop_improve(
                selected,
                request_order,
                candidates_by_request,
                request_mass,
                resource_capacities,
                score_by_id,
                reserved_usage,
            )
            selected, pair_iterations = self._pair_improve(
                selected,
                request_order,
                candidates_by_request,
                resource_capacities,
                score_by_id,
                reserved_usage,
            )
            support_variable_count = sum(
                score > self.support_tolerance for score in ordered_scores
            )
            search_strategy = "beam_local_search"
        selected = self._refine_candidates(
            selected,
            candidates_by_request,
            resource_capacities,
            score_by_id,
            reserved_usage,
        )

        selected_variables = tuple(sorted(
            selected.values(), key=lambda variable: variable.variable_id
        ))
        feasibility = validate_decoded_selection(
            selected_variables,
            resource_capacities,
            reserved_usage,
        )
        if not feasibility.feasible:
            raise RuntimeError(
                f"hard decoder produced an infeasible plan: "
                f"{feasibility.violations[0]}"
            )
        return HardDecoderSolution(
            variables=variables,
            scores=ordered_scores,
            request_ids=ordered_request_ids,
            selected_variables=selected_variables,
            feasibility=feasibility,
            beam_width=self.beam_width,
            random_restarts=self.random_restarts,
            local_search_iterations=(one_drop_iterations + pair_iterations),
            search_strategy=search_strategy,
            support_variable_count=support_variable_count,
        )


def compare_decoder_and_milp(
    decoded: HardDecoderSolution,
    discrete: DiscreteOracleSolution,
    *,
    tolerance: float = 1e-7,
) -> DecoderMILPGapReport:
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    decoded_ids = tuple(variable.variable_id for variable in decoded.variables)
    discrete_ids = tuple(variable.variable_id for variable in discrete.variables)
    if decoded_ids != discrete_ids:
        raise ValueError("decoder and MILP use different variables")
    throughput_loss = (
        discrete.expected_completed_request_mass
        - decoded.expected_completed_request_mass
    )
    numerical_tolerance = 10.0 * tolerance
    if throughput_loss < -numerical_tolerance:
        raise ValueError("decoder throughput exceeds the MILP optimum")
    throughput_loss = max(0.0, throughput_loss)
    relative_loss = (
        throughput_loss / discrete.expected_completed_request_mass
        if discrete.expected_completed_request_mass > tolerance else None
    )
    latency_is_comparable = throughput_loss <= numerical_tolerance
    latency_gap: float | None = None
    relative_latency_gap: float | None = None
    if latency_is_comparable:
        raw_gap = (
            decoded.expected_total_completion_latency
            - discrete.total_completion_latency
        )
        if raw_gap < -numerical_tolerance:
            raise ValueError("decoder latency is below the MILP optimum")
        latency_gap = max(0.0, raw_gap)
        relative_latency_gap = (
            latency_gap / discrete.total_completion_latency
            if discrete.total_completion_latency > tolerance else 0.0
        )
    return DecoderMILPGapReport(
        variable_count=len(decoded_ids),
        decoder_feasible=decoded.feasibility.feasible,
        decoder_completed_request_count=decoded.completed_request_count,
        discrete_completed_request_count=discrete.completed_request_count,
        decoder_expected_completed_request_mass=(
            decoded.expected_completed_request_mass
        ),
        discrete_expected_completed_request_mass=(
            discrete.expected_completed_request_mass
        ),
        throughput_absolute_loss=throughput_loss,
        throughput_relative_loss=relative_loss,
        throughput_is_optimal=throughput_loss <= numerical_tolerance,
        latency_is_comparable=latency_is_comparable,
        decoder_total_completion_latency=(
            decoded.expected_total_completion_latency
        ),
        discrete_total_completion_latency=discrete.total_completion_latency,
        latency_absolute_gap=latency_gap,
        latency_relative_gap=relative_latency_gap,
        decoder_total_selected_score=decoded.total_selected_score,
        beam_width=decoded.beam_width,
        random_restarts=decoded.random_restarts,
        local_search_iterations=decoded.local_search_iterations,
        decoder_selected_variable_ids=tuple(
            variable.variable_id for variable in decoded.selected_variables
        ),
        discrete_selected_variable_ids=tuple(
            variable.variable_id for variable in discrete.selected_variables
        ),
    )


def save_decoder_gap_report(
    report: DecoderMILPGapReport,
    path: str | Path,
    *,
    context: Mapping[str, object] | None = None,
) -> Path:
    target = Path(path)
    if target.suffix.lower() != ".json":
        raise ValueError("decoder gap report path must end with .json")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    if context is not None:
        payload["context"] = dict(context)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target
