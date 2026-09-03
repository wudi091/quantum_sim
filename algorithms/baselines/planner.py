"""Simulator-neutral planners for non-learning routing baselines.

The original algorithms use different simulators and execution models.  This
module keeps their path-selection principles while compiling every accepted
choice into the same fixed-construction, resource--time contract used by
TELGEN and Q-CAST.  No function here imports or inspects SeQUeNCe objects.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from statistics import fmean
from typing import Callable, Mapping, Sequence

from algorithms.qcast.online_planner import (
    effective_link_generation_probability,
)
from algorithms.telgen.dataset import (
    PlanningBatchProblem,
    build_planning_batch_problem,
)
from algorithms.telgen.fidelity import candidate_fidelity_estimate_map
from algorithms.telgen.packing import (
    PackingSolution,
    validate_packing_selection,
)
from algorithms.telgen.time_expansion import (
    TimeExpandedCandidate,
    normalize_reserved_usage,
)
from qnet_core.construction_api import OperationKind
from qnet_core.construction_catalog import RouteConstructionCandidate
from qnet_core.spec import EpisodeSpec


BASELINE_ALGORITHMS = (
    "greedy",
    "strict_fifo",
    "best_fifo",
    "qpass",
    "qpath",
    "qleap",
)


@dataclass(frozen=True)
class BaselinePlannerState:
    """Small amount of history required by Best-FIFO."""

    path_cost_sum: float = 0.0
    path_cost_count: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.path_cost_sum) or self.path_cost_sum < 0.0:
            raise ValueError("path_cost_sum must be finite and non-negative")
        if self.path_cost_count < 0:
            raise ValueError("path_cost_count cannot be negative")
        if self.path_cost_count == 0 and self.path_cost_sum != 0.0:
            raise ValueError("empty path-cost history must have zero sum")

    @property
    def average_path_cost(self) -> float | None:
        if self.path_cost_count == 0:
            return None
        return self.path_cost_sum / self.path_cost_count

    def observe(self, costs: Sequence[float]) -> BaselinePlannerState:
        finite = tuple(float(cost) for cost in costs)
        if any(not math.isfinite(cost) or cost < 0.0 for cost in finite):
            raise ValueError("observed path costs must be finite and non-negative")
        return BaselinePlannerState(
            path_cost_sum=self.path_cost_sum + sum(finite),
            path_cost_count=self.path_cost_count + len(finite),
        )


@dataclass(frozen=True)
class BaselinePlanningRecord:
    """One baseline decision over a shared time-expanded planning problem."""

    algorithm: str
    problem: PlanningBatchProblem
    considered_variables: tuple[TimeExpandedCandidate, ...]
    candidate_scores: tuple[tuple[str, float], ...]
    selected_path_costs: tuple[float, ...]
    state_before: BaselinePlannerState
    state_after: BaselinePlannerState
    solution: PackingSolution

    @property
    def selected_score(self) -> float:
        scores = dict(self.candidate_scores)
        return float(sum(
            scores.get(variable.candidate_id, 0.0)
            for variable in self.solution.selected_variables
        ))


class _SelectionState:
    def __init__(
        self,
        capacities: Mapping[str, int],
        reserved_usage: Mapping[tuple[str, int], int] | None,
    ) -> None:
        self.capacities = {
            str(resource_id): int(capacity)
            for resource_id, capacity in capacities.items()
        }
        self.usage = normalize_reserved_usage(
            reserved_usage,
            self.capacities,
        )
        self.selected: dict[str, TimeExpandedCandidate] = {}

    def can_add(self, variable: TimeExpandedCandidate) -> bool:
        if variable.request_id in self.selected:
            return False
        delta: dict[tuple[str, int], int] = {}
        for item in variable.resource_usage:
            if item.resource_id not in self.capacities:
                raise ValueError(
                    f"missing capacity for resource: {item.resource_id}"
                )
            key = (item.resource_id, item.slot)
            delta[key] = delta.get(key, 0) + item.amount
        return all(
            self.usage.get(key, 0) + amount <= self.capacities[key[0]]
            for key, amount in delta.items()
        )

    def add(self, variable: TimeExpandedCandidate) -> None:
        if not self.can_add(variable):
            raise ValueError(
                f"cannot add infeasible baseline choice: {variable.variable_id}"
            )
        self.selected[variable.request_id] = variable
        for item in variable.resource_usage:
            key = (item.resource_id, item.slot)
            self.usage[key] = self.usage.get(key, 0) + item.amount


def _generation_count(candidate: RouteConstructionCandidate) -> int:
    return sum(
        operation.kind == OperationKind.GEN
        for operation in candidate.dag.operations
    )


def _expected_pair_cost(variable: TimeExpandedCandidate) -> float:
    """Expected elementary-pair consumption of one completed request."""

    probability = float(variable.expected_success_probability)
    if probability <= 0.0:
        return math.inf
    return _generation_count(variable.base_candidate) / probability


def _qpath_resource_cost(variable: TimeExpandedCandidate) -> int:
    """Elementary-pair consumption used by the upstream Q-PATH search."""

    return _generation_count(variable.base_candidate)


def _purification_overhead(variable: TimeExpandedCandidate) -> int:
    return max(
        0,
        _generation_count(variable.base_candidate)
        - variable.base_candidate.hop_count,
    )


def _variable_resource_time_cost(variable: TimeExpandedCandidate) -> int:
    return sum(item.amount for item in variable.resource_usage)


def _qpass_creation_rate(
    episode: EpisodeSpec,
    candidate: RouteConstructionCandidate,
) -> float:
    """Width-one inverse path-generation cost used by Q-PASS.

    The upstream CreationRate implementation minimizes the sum of per-link
    inverse-generation costs.  The shared EpisodeSpec currently has one
    homogeneous physical profile for every edge, so this reduces to
    ``p_link / hop_count`` rather than an end-to-end probability product.
    """

    hop_count = candidate.hop_count
    if hop_count < 1:
        return 0.0
    link_probability = effective_link_generation_probability(episode.physical)
    if link_probability <= 0.0:
        return 0.0
    path_generation_cost = hop_count / link_probability
    return 1.0 / path_generation_cost


def _normalise_scores(
    variables: Sequence[TimeExpandedCandidate],
    score_by_variable: Mapping[str, float],
) -> tuple[float, ...]:
    raw = tuple(
        float(score_by_variable.get(variable.variable_id, 0.0))
        for variable in variables
    )
    finite = tuple(value for value in raw if math.isfinite(value))
    if not finite:
        return (0.0,) * len(raw)
    low = min(finite)
    high = max(finite)
    if math.isclose(low, high):
        fill = 1.0 if high > 0.0 else 0.0
        return tuple(fill if math.isfinite(value) else 0.0 for value in raw)
    return tuple(
        0.0
        if not math.isfinite(value)
        else min(1.0, max(0.0, (value - low) / (high - low)))
        for value in raw
    )


def _solution(
    variables: Sequence[TimeExpandedCandidate],
    selected: Sequence[TimeExpandedCandidate],
    request_ids: tuple[str, ...],
    capacities: Mapping[str, int],
    reserved_usage: Mapping[tuple[str, int], int] | None,
    score_by_variable: Mapping[str, float],
    strategy: str,
) -> PackingSolution:
    ordered_variables = tuple(sorted(
        variables,
        key=lambda variable: variable.variable_id,
    ))
    ordered_selected = tuple(sorted(
        selected,
        key=lambda variable: variable.variable_id,
    ))
    feasibility = validate_packing_selection(
        ordered_selected,
        capacities,
        reserved_usage,
    )
    if not feasibility.feasible:
        raise RuntimeError(
            f"{strategy} produced an infeasible plan: "
            f"{feasibility.violations[0]}"
        )
    return PackingSolution(
        variables=ordered_variables,
        scores=_normalise_scores(ordered_variables, score_by_variable),
        request_ids=request_ids,
        selected_variables=ordered_selected,
        feasibility=feasibility,
        strategy=strategy,
    )


def _request_order(
    episode: EpisodeSpec,
    request_ids: tuple[str, ...],
) -> tuple[str, ...]:
    requests = {request.id: request for request in episode.requests}
    return tuple(sorted(
        request_ids,
        key=lambda request_id: (
            requests[request_id].arrival,
            request_id,
        ),
    ))


def _select_by_request_order(
    variables: Sequence[TimeExpandedCandidate],
    request_order: Sequence[str],
    state: _SelectionState,
    variable_key: Callable[[TimeExpandedCandidate], tuple[object, ...]],
) -> None:
    grouped: dict[str, list[TimeExpandedCandidate]] = {}
    for variable in variables:
        grouped.setdefault(variable.request_id, []).append(variable)
    for request_id in request_order:
        for variable in sorted(grouped.get(request_id, ()), key=variable_key):
            if variable.expected_success_probability <= 0.0:
                continue
            if state.can_add(variable):
                state.add(variable)
                break


def _select_best_fifo(
    variables: Sequence[TimeExpandedCandidate],
    request_order: Sequence[str],
    state: _SelectionState,
    historical_average: float | None,
) -> float | None:
    grouped: dict[str, list[TimeExpandedCandidate]] = {}
    for variable in variables:
        grouped.setdefault(variable.request_id, []).append(variable)
    best_costs = {
        request_id: min(
            (
                _expected_pair_cost(variable)
                for variable in request_variables
                if variable.expected_success_probability > 0.0
            ),
            default=math.inf,
        )
        for request_id, request_variables in grouped.items()
    }
    finite_costs = tuple(
        cost for cost in best_costs.values() if math.isfinite(cost)
    )
    threshold = historical_average
    if threshold is None and finite_costs:
        threshold = fmean(finite_costs)

    deferred: list[str] = []
    for request_id in request_order:
        if threshold is None or best_costs.get(request_id, math.inf) > threshold:
            deferred.append(request_id)
            continue
        selected = False
        for variable in sorted(
            grouped.get(request_id, ()),
            key=lambda item: (
                _expected_pair_cost(item),
                item.start_slot,
                item.completion_slot,
                item.variable_id,
            ),
        ):
            if _expected_pair_cost(variable) > threshold:
                break
            if state.can_add(variable):
                state.add(variable)
                selected = True
                break
        if not selected:
            deferred.append(request_id)

    _select_by_request_order(
        variables,
        deferred,
        state,
        lambda item: (
            _expected_pair_cost(item),
            item.start_slot,
            item.completion_slot,
            item.variable_id,
        ),
    )
    return threshold


def _select_qleap(
    variables: Sequence[TimeExpandedCandidate],
    request_order: Sequence[str],
    state: _SelectionState,
    raw_route_fidelity: Mapping[tuple[str, tuple[int, ...]], float],
) -> None:
    """Apply Q-LEAP's max-fidelity route rule before shared allocation.

    The upstream multi-request code gives lower-purification plans higher
    priority.  Within one request, Q-LEAP first searches the route with the
    largest raw end-to-end fidelity and only purifies when the threshold
    requires it.
    """

    grouped: dict[str, list[TimeExpandedCandidate]] = {}
    for variable in variables:
        grouped.setdefault(variable.request_id, []).append(variable)

    def route_fidelity(variable: TimeExpandedCandidate) -> float:
        return raw_route_fidelity.get(
            (variable.request_id, variable.route_nodes),
            variable.expected_fidelity or 0.0,
        )

    request_rank = {request_id: index for index, request_id in enumerate(request_order)}
    ordered_requests = sorted(
        request_order,
        key=lambda request_id: (
            min(
                (
                    _purification_overhead(variable)
                    for variable in grouped.get(request_id, ())
                ),
                default=math.inf,
            ),
            request_rank[request_id],
        ),
    )
    for request_id in ordered_requests:
        for variable in sorted(
            grouped.get(request_id, ()),
            key=lambda item: (
                -route_fidelity(item),
                _purification_overhead(item),
                item.start_slot,
                item.completion_slot,
                item.variable_id,
            ),
        ):
            if variable.expected_success_probability <= 0.0:
                continue
            if state.can_add(variable):
                state.add(variable)
                break


def _validate_request_subset(
    episode: EpisodeSpec,
    request_ids: tuple[str, ...] | None,
) -> tuple[EpisodeSpec, tuple[str, ...]]:
    declared = (
        tuple(request.id for request in episode.requests)
        if request_ids is None
        else tuple(str(request_id) for request_id in request_ids)
    )
    if len(set(declared)) != len(declared):
        raise ValueError("request_ids must be unique")
    known = {request.id for request in episode.requests}
    unknown = set(declared) - known
    if unknown:
        raise ValueError(f"unknown request: {sorted(unknown)[0]}")
    selected = set(declared)
    return replace(
        episode,
        requests=tuple(
            request for request in episode.requests if request.id in selected
        ),
    ), declared


def plan_baseline_window(
    episode: EpisodeSpec,
    *,
    algorithm: str,
    window_start_slot: int,
    window_end_slot: int,
    completion_end_slot: int | None = None,
    request_ids: tuple[str, ...] | None = None,
    resource_capacities: Mapping[str, int] | None = None,
    reserved_usage: Mapping[tuple[str, int], int] | None = None,
    path_candidate_count: int = 4,
    construction_kind: str = "left_deep",
    planner_state: BaselinePlannerState | None = None,
) -> BaselinePlanningRecord:
    """Plan one online window with a named non-learning baseline."""

    if algorithm not in BASELINE_ALGORITHMS:
        raise ValueError(f"unknown baseline algorithm: {algorithm}")
    if path_candidate_count < 1:
        raise ValueError("path_candidate_count must be positive")
    if construction_kind not in {"left_deep", "balanced"}:
        raise ValueError("unsupported construction_kind")

    planning_episode, declared_request_ids = _validate_request_subset(
        episode,
        request_ids,
    )
    purification_kinds = (
        ("none", "elementary_once")
        if algorithm in {"qpath", "qleap"}
        else ("none",)
    )
    problem = build_planning_batch_problem(
        planning_episode,
        window_start_slot=window_start_slot,
        window_end_slot=window_end_slot,
        completion_end_slot=completion_end_slot,
        resource_capacities=resource_capacities,
        reserved_usage=reserved_usage,
        path_candidate_count=path_candidate_count,
        construction_kinds=(construction_kind,),
        purification_kinds=purification_kinds,
    )
    capacities = problem.capacities
    before = planner_state or BaselinePlannerState()
    variables: tuple[TimeExpandedCandidate, ...] = problem.expansion.variables

    fidelity_by_candidate = candidate_fidelity_estimate_map(
        planning_episode,
        problem.candidates,
    )
    raw_route_fidelity: dict[tuple[str, tuple[int, ...]], float] = {}
    for candidate in problem.candidates:
        if candidate.purification_kind != "none":
            continue
        key = (candidate.request_id, candidate.route_nodes)
        raw_route_fidelity[key] = max(
            raw_route_fidelity.get(key, 0.0),
            fidelity_by_candidate[candidate.candidate_id],
        )

    score_by_variable: dict[str, float] = {}
    for variable in variables:
        cost = _expected_pair_cost(variable)
        if algorithm == "greedy":
            score = 1.0 / (1.0 + variable.base_candidate.hop_count)
        elif algorithm in {"strict_fifo", "best_fifo"}:
            score = 0.0 if not math.isfinite(cost) else 1.0 / (1.0 + cost)
        elif algorithm == "qpath":
            score = 1.0 / (1.0 + _qpath_resource_cost(variable))
        elif algorithm == "qpass":
            score = _qpass_creation_rate(
                planning_episode,
                variable.base_candidate,
            )
        else:
            route_fidelity = raw_route_fidelity.get(
                (variable.request_id, variable.route_nodes),
                variable.expected_fidelity or 0.0,
            )
            score = (
                route_fidelity
                * variable.expected_success_probability
                / max(_generation_count(variable.base_candidate), 1)
            )
        score_by_variable[variable.variable_id] = float(score)

    selection = _SelectionState(capacities, reserved_usage)
    request_order = _request_order(planning_episode, declared_request_ids)
    if algorithm == "greedy":
        _select_by_request_order(
            variables,
            request_order,
            selection,
            lambda item: (
                item.base_candidate.hop_count,
                item.start_slot,
                item.completion_slot,
                _variable_resource_time_cost(item),
                item.variable_id,
            ),
        )
    elif algorithm == "strict_fifo":
        _select_by_request_order(
            variables,
            request_order,
            selection,
            lambda item: (
                _expected_pair_cost(item),
                item.start_slot,
                item.completion_slot,
                item.variable_id,
            ),
        )
    elif algorithm == "best_fifo":
        _select_best_fifo(
            variables,
            request_order,
            selection,
            before.average_path_cost,
        )
    else:
        if algorithm == "qpass":
            ranked = sorted(variables, key=lambda item: (
                -score_by_variable[item.variable_id],
                item.base_candidate.hop_count,
                item.start_slot,
                item.completion_slot,
                item.variable_id,
            ))
        elif algorithm == "qpath":
            ranked = sorted(variables, key=lambda item: (
                _qpath_resource_cost(item),
                item.start_slot,
                item.completion_slot,
                -item.expected_success_probability,
                -(item.expected_fidelity or 0.0),
                item.variable_id,
            ))
        else:
            _select_qleap(
                variables,
                request_order,
                selection,
                raw_route_fidelity,
            )
            ranked = []
        for variable in ranked:
            if variable.expected_success_probability <= 0.0:
                continue
            if selection.can_add(variable):
                selection.add(variable)

    selected = tuple(selection.selected.values())
    selected_costs = tuple(
        _expected_pair_cost(variable)
        for variable in sorted(selected, key=lambda item: item.request_id)
    )
    after = (
        before.observe(selected_costs)
        if algorithm == "best_fifo"
        else before
    )
    solution = _solution(
        variables,
        selected,
        declared_request_ids,
        capacities,
        reserved_usage,
        score_by_variable,
        strategy=f"{algorithm}_shared_resource_time_v1",
    )
    candidate_scores = tuple(sorted(
        (
            candidate.candidate_id,
            max(
                (
                    score_by_variable[variable.variable_id]
                    for variable in variables
                    if variable.candidate_id == candidate.candidate_id
                ),
                default=0.0,
            ),
        )
        for candidate in problem.candidates
    ))
    return BaselinePlanningRecord(
        algorithm=algorithm,
        problem=problem,
        considered_variables=tuple(sorted(
            variables,
            key=lambda variable: variable.variable_id,
        )),
        candidate_scores=candidate_scores,
        selected_path_costs=selected_costs,
        state_before=before,
        state_after=after,
        solution=solution,
    )


__all__ = [
    "BASELINE_ALGORITHMS",
    "BaselinePlannerState",
    "BaselinePlanningRecord",
    "plan_baseline_window",
]
