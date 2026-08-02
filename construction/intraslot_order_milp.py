"""Time-indexed 0-1 MILP oracle for the intra-slot memory counterexample.

The controller chooses exactly one complete swap order for request R1.
R1 starts with all elementary EPRs ready on path A-B-C-D-E. Two lower
priority requests, R2 and R3, each need both memories at hotspot C for one
physical execution window; the fixed automatic executor serves R2 before R3
whenever C has enough free memory.

The MILP does not decide individual EPR-generation attempts. It uses the
hotspot occupancy and BSM-use profiles produced by the fixed simulator for
each complete R1 swap order, then determines how many waiting requests that
automatic service discipline can finish. This is an exact oracle for the
deterministic counterexample (p_gen = p_swap = 1), not for the full
stochastic SeQUeNCe model.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import TypeAlias

import numpy as np
from scipy.optimize import Bounds, LinearConstraint

from .exact_milp import solve_exact_milp
from .intraslot_simulator import (
    IntraSlotConfig,
    IntraSlotPlan,
    IntraSlotSimulator,
)


Node: TypeAlias = str
SwapOrder: TypeAlias = tuple[Node, ...]


@dataclass(frozen=True)
class OrderProfile:
    """Fixed R1 use of hotspot memory and BSM across physical windows."""

    order: SwapOrder
    hotspot_occupancy: tuple[int, ...]
    hotspot_bsm: tuple[int, ...]


@dataclass(frozen=True)
class OrderOutcome:
    """Optimal waiting-request schedule when one R1 order is fixed."""

    order: SwapOrder
    completed_requests: int
    waiting_completion_round: dict[str, int]


@dataclass(frozen=True)
class CounterexampleMilpResult:
    """Global deterministic optimum and per-order objective values."""

    selected_order: SwapOrder
    completed_requests: int
    waiting_completion_round: dict[str, int]
    optimal_orders: tuple[SwapOrder, ...]
    order_outcomes: dict[SwapOrder, OrderOutcome]
    profiles: dict[SwapOrder, OrderProfile]


def build_order_profiles(
    *,
    rounds_per_slot: int = 3,
    hotspot: Node = "C",
) -> dict[SwapOrder, OrderProfile]:
    """Derive immutable R1 resource profiles from the fixed executor."""

    path = ("A", "B", "C", "D", "E")
    capacity = {node: 2 for node in path}
    profiles: dict[SwapOrder, OrderProfile] = {}

    for order in permutations(path[1:-1]):
        result = IntraSlotSimulator(
            plans=(IntraSlotPlan("R1", path, order),),
            node_capacity=capacity,
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
        profiles[order] = OrderProfile(
            order=order,
            hotspot_occupancy=tuple(
                trace.occupancy_start.get(hotspot, 0)
                for trace in result.traces
            ),
            hotspot_bsm=tuple(
                int(any(
                    event.middle == hotspot and event.status == "success"
                    for event in trace.swap_events
                ))
                for trace in result.traces
            ),
        )

    return profiles


def _solve_model(
    profiles: dict[SwapOrder, OrderProfile],
    *,
    hotspot_capacity: int,
    waiting_requests: tuple[str, ...],
    waiting_memory: int,
    force_order: SwapOrder | None,
) -> OrderOutcome:
    """Solve one lexicographic completion/latency 0-1 MILP."""

    orders = tuple(profiles)
    rounds = len(next(iter(profiles.values())).hotspot_occupancy)
    order_index = {order: index for index, order in enumerate(orders)}
    waiting_index = {
        (request_id, round_id): len(orders) + request_pos * rounds + round_id
        for request_pos, request_id in enumerate(waiting_requests)
        for round_id in range(rounds)
    }
    variable_count = len(orders) + len(waiting_requests) * rounds

    # Lexicographic objective:
    # 1) maximize the number of completed waiting requests;
    # 2) among equal-completion schedules, finish them as early as possible.
    completion_weight = len(waiting_requests) * rounds + 1
    objective = np.zeros(variable_count, dtype=float)
    for request_id in waiting_requests:
        for round_id in range(rounds):
            objective[waiting_index[(request_id, round_id)]] = (
                -completion_weight + round_id + 1
            )

    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    # Exactly one complete R1 swap order is selected.
    row = np.zeros(variable_count, dtype=float)
    for index in range(len(orders)):
        row[index] = 1
    rows.append(row)
    lower.append(1)
    upper.append(1)

    # Each waiting request can complete at most once.
    for request_id in waiting_requests:
        row = np.zeros(variable_count, dtype=float)
        for round_id in range(rounds):
            row[waiting_index[(request_id, round_id)]] = 1
        rows.append(row)
        lower.append(-np.inf)
        upper.append(1)

    # Hotspot memory and BSM capacity in every physical window.
    for round_id in range(rounds):
        memory_row = np.zeros(variable_count, dtype=float)
        bsm_row = np.zeros(variable_count, dtype=float)
        for order, profile in profiles.items():
            index = order_index[order]
            memory_row[index] = profile.hotspot_occupancy[round_id]
            bsm_row[index] = profile.hotspot_bsm[round_id]
        for request_id in waiting_requests:
            index = waiting_index[(request_id, round_id)]
            memory_row[index] = waiting_memory
            bsm_row[index] = 1
        rows.append(memory_row)
        lower.append(-np.inf)
        upper.append(hotspot_capacity)
        rows.append(bsm_row)
        lower.append(-np.inf)
        upper.append(1)

    # Fixed automatic priority: a lower-priority request may not complete
    # before every higher-priority waiting request has completed.
    for request_pos in range(1, len(waiting_requests)):
        request_id = waiting_requests[request_pos]
        previous_id = waiting_requests[request_pos - 1]
        for round_id in range(rounds):
            row = np.zeros(variable_count, dtype=float)
            for current_round in range(round_id + 1):
                row[waiting_index[(request_id, current_round)]] += 1
            for previous_round in range(round_id):
                row[waiting_index[(previous_id, previous_round)]] -= 1
            rows.append(row)
            lower.append(-np.inf)
            upper.append(0)

    bounds_lower = np.zeros(variable_count, dtype=float)
    bounds_upper = np.ones(variable_count, dtype=float)
    if force_order is not None:
        forced_index = order_index[force_order]
        for index in range(len(orders)):
            value = 1.0 if index == forced_index else 0.0
            bounds_lower[index] = value
            bounds_upper[index] = value

    result = solve_exact_milp(
        c=objective,
        integrality=np.ones(variable_count, dtype=int),
        bounds=Bounds(bounds_lower, bounds_upper),
        constraints=LinearConstraint(
            np.vstack(rows),
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
        ),
        options={"disp": False},
    )

    selected_order = orders[
        max(range(len(orders)), key=lambda index: result.x[index])
    ]
    waiting_completion_round: dict[str, int] = {}
    for request_id in waiting_requests:
        for round_id in range(rounds):
            if result.x[waiting_index[(request_id, round_id)]] > 0.5:
                waiting_completion_round[request_id] = round_id + 1
                break

    return OrderOutcome(
        order=selected_order,
        completed_requests=1 + len(waiting_completion_round),
        waiting_completion_round=waiting_completion_round,
    )


def solve_counterexample_milp(
    *,
    hotspot_capacity: int = 2,
    rounds_per_slot: int = 3,
) -> CounterexampleMilpResult:
    """Prove the global optimum for the deterministic three-request example."""

    profiles = build_order_profiles(rounds_per_slot=rounds_per_slot)
    waiting_requests = ("R2", "R3")
    global_outcome = _solve_model(
        profiles,
        hotspot_capacity=hotspot_capacity,
        waiting_requests=waiting_requests,
        waiting_memory=2,
        force_order=None,
    )
    order_outcomes = {
        order: _solve_model(
            profiles,
            hotspot_capacity=hotspot_capacity,
            waiting_requests=waiting_requests,
            waiting_memory=2,
            force_order=order,
        )
        for order in profiles
    }
    optimum = max(
        outcome.completed_requests for outcome in order_outcomes.values()
    )
    optimal_orders = tuple(
        order for order, outcome in order_outcomes.items()
        if outcome.completed_requests == optimum
    )

    if global_outcome.completed_requests != optimum:
        raise RuntimeError(
            "unconstrained MILP disagrees with per-order optimum enumeration"
        )

    return CounterexampleMilpResult(
        selected_order=global_outcome.order,
        completed_requests=global_outcome.completed_requests,
        waiting_completion_round=global_outcome.waiting_completion_round,
        optimal_orders=optimal_orders,
        order_outcomes=order_outcomes,
        profiles=profiles,
    )
