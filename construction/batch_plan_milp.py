"""Joint deterministic request/path/swap-order selection by time-indexed MILP.

Each candidate is a complete plan: one request, one path, and one complete
linear swap order represented by fixed per-window memory and BSM profiles. The
binary variable z[g, t] selects candidate g and starts its fixed automatic
execution profile at physical window t.

This model jointly performs request admission, candidate/path selection, and
swap-order selection. It does not schedule individual HEG attempts. Profiles
must be produced by a fixed executor or a deterministic scenario, so this is
an exact small-instance oracle only under those supplied profiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable, Mapping, TypeAlias

import numpy as np
from scipy.optimize import Bounds, LinearConstraint

from .exact_milp import solve_exact_milp
from .intraslot_simulator import (
    IntraSlotConfig,
    IntraSlotPlan,
    IntraSlotSimulator,
)


Node: TypeAlias = str


@dataclass(frozen=True)
class TimedPlanCandidate:
    """One complete path/swap candidate with a deterministic resource trace."""

    request_id: str
    candidate_id: str
    path: tuple[Node, ...]
    swap_order: tuple[Node, ...]
    duration: int
    memory_profile: Mapping[Node, tuple[int, ...]]
    bsm_profile: Mapping[Node, tuple[int, ...]]
    allowed_starts: tuple[int, ...] | None = None
    exclusive_resources: frozenset[str] = frozenset()
    reward: int = 1
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.request_id or not self.candidate_id:
            raise ValueError("request_id and candidate_id must be non-empty")
        if self.duration < 1:
            raise ValueError("duration must be positive")
        if self.reward < 1:
            raise ValueError("reward must be positive")
        for name, profile in (
            ("memory", self.memory_profile),
            ("BSM", self.bsm_profile),
        ):
            for node, values in profile.items():
                if len(values) != self.duration:
                    raise ValueError(
                        f"{name} profile for {node} has length {len(values)}, "
                        f"expected {self.duration}"
                    )
                if any(value < 0 for value in values):
                    raise ValueError(f"{name} profile cannot be negative")
        if self.allowed_starts is not None:
            if len(set(self.allowed_starts)) != len(self.allowed_starts):
                raise ValueError("allowed_starts cannot contain duplicates")
            if any(start < 0 for start in self.allowed_starts):
                raise ValueError("allowed starts must be non-negative")

    @property
    def label(self) -> str:
        return f"{self.request_id}:{self.candidate_id}"


@dataclass(frozen=True)
class ScheduledPlan:
    candidate: TimedPlanCandidate
    start: int

    @property
    def finish(self) -> int:
        return self.start + self.candidate.duration


@dataclass(frozen=True)
class BatchMilpSolution:
    selected: dict[str, ScheduledPlan]
    completed_requests: int
    total_reward: int
    primary_weight: int
    solver_objective: float


def _profile_value(
    profile: Mapping[Node, tuple[int, ...]],
    node: Node,
    relative_time: int,
) -> int:
    values = profile.get(node)
    return 0 if values is None else values[relative_time]


def solve_time_indexed_batch_milp(
    candidates: tuple[TimedPlanCandidate, ...],
    *,
    memory_capacity: Mapping[Node, int],
    bsm_capacity: Mapping[Node, int],
    horizon: int,
    required_requests: Iterable[str] = (),
) -> BatchMilpSolution:
    """Jointly select requests, candidate paths/orders, and feasible starts."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not candidates:
        return BatchMilpSolution({}, 0, 0, 1, 0.0)

    labels = [candidate.label for candidate in candidates]
    if len(set(labels)) != len(labels):
        raise ValueError("candidate labels must be unique")

    positive_memory_nodes = {
        node
        for candidate in candidates
        for node, values in candidate.memory_profile.items()
        if any(values)
    }
    positive_bsm_nodes = {
        node
        for candidate in candidates
        for node, values in candidate.bsm_profile.items()
        if any(values)
    }
    missing_memory = positive_memory_nodes - set(memory_capacity)
    missing_bsm = positive_bsm_nodes - set(bsm_capacity)
    if missing_memory:
        raise ValueError(f"missing memory capacities: {sorted(missing_memory)}")
    if missing_bsm:
        raise ValueError(f"missing BSM capacities: {sorted(missing_bsm)}")

    variable_keys: list[tuple[int, int]] = []
    for candidate_index, candidate in enumerate(candidates):
        starts = (
            range(horizon - candidate.duration + 1)
            if candidate.allowed_starts is None
            else candidate.allowed_starts
        )
        for start in starts:
            if start + candidate.duration <= horizon:
                variable_keys.append((candidate_index, start))
    if not variable_keys:
        return BatchMilpSolution({}, 0, 0, 1, 0.0)

    variable_index = {
        key: index for index, key in enumerate(variable_keys)
    }
    variable_count = len(variable_keys)
    request_ids = tuple(dict.fromkeys(
        candidate.request_id for candidate in candidates
    ))
    required = frozenset(required_requests)
    unknown_required = required - set(request_ids)
    if unknown_required:
        raise ValueError(
            f"required request has no candidate: {sorted(unknown_required)}"
        )

    max_priority = max(candidate.priority for candidate in candidates)
    secondary_cost = np.zeros(variable_count, dtype=float)
    for index, (candidate_index, start) in enumerate(variable_keys):
        candidate = candidates[candidate_index]
        finish = start + candidate.duration
        urgency = max_priority - candidate.priority + 1
        footprint = sum(
            sum(values) for values in candidate.memory_profile.values()
        ) + sum(sum(values) for values in candidate.bsm_profile.values())
        secondary_cost[index] = finish * urgency + footprint

    max_secondary_by_request = []
    for request_id in request_ids:
        coefficients = [
            secondary_cost[index]
            for index, (candidate_index, _) in enumerate(variable_keys)
            if candidates[candidate_index].request_id == request_id
        ]
        max_secondary_by_request.append(max(coefficients, default=0))
    primary_weight = int(sum(max_secondary_by_request)) + 1

    objective = secondary_cost.copy()
    for index, (candidate_index, _) in enumerate(variable_keys):
        objective[index] -= (
            primary_weight * candidates[candidate_index].reward
        )

    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    # At most one complete candidate and one start time per request.
    for request_id in request_ids:
        row = np.zeros(variable_count, dtype=float)
        for index, (candidate_index, _) in enumerate(variable_keys):
            if candidates[candidate_index].request_id == request_id:
                row[index] = 1
        rows.append(row)
        lower.append(1 if request_id in required else -np.inf)
        upper.append(1)

    # Current EPR ids or other exclusive resources may be claimed once.
    exclusive_resources = {
        resource
        for candidate in candidates
        for resource in candidate.exclusive_resources
    }
    for resource in sorted(exclusive_resources):
        row = np.zeros(variable_count, dtype=float)
        for index, (candidate_index, _) in enumerate(variable_keys):
            if resource in candidates[candidate_index].exclusive_resources:
                row[index] = 1
        rows.append(row)
        lower.append(-np.inf)
        upper.append(1)

    # Time-expanded physical memory and BSM capacities.
    for absolute_time in range(horizon):
        for node, capacity in memory_capacity.items():
            row = np.zeros(variable_count, dtype=float)
            for index, (candidate_index, start) in enumerate(variable_keys):
                candidate = candidates[candidate_index]
                relative_time = absolute_time - start
                if 0 <= relative_time < candidate.duration:
                    row[index] = _profile_value(
                        candidate.memory_profile, node, relative_time
                    )
            rows.append(row)
            lower.append(-np.inf)
            upper.append(capacity)
        for node, capacity in bsm_capacity.items():
            row = np.zeros(variable_count, dtype=float)
            for index, (candidate_index, start) in enumerate(variable_keys):
                candidate = candidates[candidate_index]
                relative_time = absolute_time - start
                if 0 <= relative_time < candidate.duration:
                    row[index] = _profile_value(
                        candidate.bsm_profile, node, relative_time
                    )
            rows.append(row)
            lower.append(-np.inf)
            upper.append(capacity)

    result = solve_exact_milp(
        c=objective,
        integrality=np.ones(variable_count, dtype=int),
        bounds=Bounds(
            np.zeros(variable_count, dtype=float),
            np.ones(variable_count, dtype=float),
        ),
        constraints=LinearConstraint(
            np.vstack(rows),
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
        ),
        options={"disp": False},
    )

    selected: dict[str, ScheduledPlan] = {}
    for index, (candidate_index, start) in enumerate(variable_keys):
        if result.x[index] <= 0.5:
            continue
        candidate = candidates[candidate_index]
        selected[candidate.request_id] = ScheduledPlan(candidate, start)

    return BatchMilpSolution(
        selected=selected,
        completed_requests=len(selected),
        total_reward=sum(
            scheduled.candidate.reward for scheduled in selected.values()
        ),
        primary_weight=primary_weight,
        solver_objective=float(result.fun),
    )


def build_joint_counterexample_candidates(
    *,
    rounds_per_slot: int = 3,
) -> tuple[TimedPlanCandidate, ...]:
    """Build complete R1/R2/R3 candidates for the deterministic example."""

    r1_path = ("A", "B", "C", "D", "E")
    r1_capacity = {node: 2 for node in r1_path}
    candidates: list[TimedPlanCandidate] = []

    for order in permutations(r1_path[1:-1]):
        result = IntraSlotSimulator(
            plans=(IntraSlotPlan("R1", r1_path, order),),
            node_capacity=r1_capacity,
            config=IntraSlotConfig(
                rounds_per_slot=rounds_per_slot,
                generation_probability=1.0,
                swap_probability=1.0,
                edge_capacity=1,
                bsm_capacity_per_node=1,
                seed=0,
            ),
            initially_ready_requests=("R1",),
        ).run()
        memory_profile = {
            node: tuple(
                trace.occupancy_start.get(node, 0)
                for trace in result.traces
            )
            for node in r1_path
        }
        bsm_profile = {
            node: tuple(
                int(any(
                    event.middle == node and event.status == "success"
                    for event in trace.swap_events
                ))
                for trace in result.traces
            )
            for node in r1_path[1:-1]
        }
        candidates.append(TimedPlanCandidate(
            request_id="R1",
            candidate_id="-".join(order),
            path=r1_path,
            swap_order=order,
            duration=rounds_per_slot,
            memory_profile=memory_profile,
            bsm_profile=bsm_profile,
            allowed_starts=(0,),
            priority=0,
        ))

    for priority, (request_id, path) in enumerate((
        ("R2", ("X", "C", "Y")),
        ("R3", ("U", "C", "V")),
    ), start=1):
        left, middle, right = path
        candidates.append(TimedPlanCandidate(
            request_id=request_id,
            candidate_id="-".join(path),
            path=path,
            swap_order=(middle,),
            duration=1,
            memory_profile={
                left: (1,),
                middle: (2,),
                right: (1,),
            },
            bsm_profile={middle: (1,)},
            priority=priority,
        ))

    return tuple(candidates)


def solve_joint_counterexample_milp(
    *,
    hotspot_capacity: int = 2,
    rounds_per_slot: int = 3,
) -> BatchMilpSolution:
    """Jointly select R1/R2/R3 admission and complete R1 swap order."""

    candidates = build_joint_counterexample_candidates(
        rounds_per_slot=rounds_per_slot
    )
    nodes = ("A", "B", "C", "D", "E", "X", "Y", "U", "V")
    memory_capacity = {node: 2 for node in nodes}
    memory_capacity["C"] = hotspot_capacity
    bsm_capacity = {node: 1 for node in nodes}
    return solve_time_indexed_batch_milp(
        candidates,
        memory_capacity=memory_capacity,
        bsm_capacity=bsm_capacity,
        horizon=rounds_per_slot,
    )
