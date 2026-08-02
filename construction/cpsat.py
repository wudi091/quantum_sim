"""Exact construction-aware batch selection with OR-Tools CP-SAT.

Decision variable x[r, p] selects at most one (P, C) candidate per
request. Node-memory constraints use the deterministic single-slot peak
footprints from :mod:`construction.simulator`.

Objective is lexicographic by coefficient scaling:
1. maximize the number of admitted requests;
2. among equal-completion solutions, minimize aggregate footprint cost.
"""

from __future__ import annotations

from dataclasses import dataclass

from .plan import ConstructionPlan
from .simulator import PlanExecution, simulate_plan


@dataclass(frozen=True)
class ExactSelection:
    plans: dict[str, ConstructionPlan]
    completed_requests: int
    footprint_cost: int
    objective_value: int


def _penalty(execution: PlanExecution) -> int:
    return (
        sum(execution.peak_memory.values())
        + sum(execution.duration_at_peak.values())
        + execution.makespan
    )


def solve_cpsat(
    candidates: dict[str, list[ConstructionPlan]],
    capacity: dict[int, int],
    *,
    time_limit_seconds: float = 10.0,
) -> ExactSelection:
    """Select an exact maximum-throughput set of (P, C) candidates.

    Raises ``ModuleNotFoundError`` with an actionable message when OR-Tools
    is unavailable. Raises ``TimeoutError`` rather than returning an incumbent
    when CP-SAT cannot prove optimality within ``time_limit_seconds``. Nodes
    omitted from ``capacity`` have zero capacity.
    """
    try:
        from ortools.sat.python import cp_model
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise ModuleNotFoundError(
            "OR-Tools is required for solve_cpsat; install requirements.txt"
        ) from exc

    model = cp_model.CpModel()
    rows: list[tuple[str, int, ConstructionPlan, PlanExecution, object]] = []
    request_vars: dict[str, list[object]] = {}

    for request_id, plans in candidates.items():
        for index, plan in enumerate(plans):
            execution = simulate_plan(plan)
            var = model.new_bool_var(f"select__{request_id}__{index}")
            rows.append((request_id, index, plan, execution, var))
            request_vars.setdefault(request_id, []).append(var)

    for variables in request_vars.values():
        model.add(sum(variables) <= 1)

    constrained_nodes = set(capacity)
    for _, _, _, execution, _ in rows:
        constrained_nodes.update(execution.peak_memory)
    for node in constrained_nodes:
        cap = capacity.get(node, 0)
        model.add(
            sum(execution.peak_memory.get(node, 0) * var
                for _, _, _, execution, var in rows)
            <= cap
        )

    penalties = [_penalty(execution) for _, _, _, execution, _ in rows]
    # Any one additional completion outweighs every possible penalty delta.
    completion_weight = sum(penalties) + 1
    objective = sum(
        (completion_weight - penalty) * var
        for (_, _, _, _, var), penalty in zip(rows, penalties)
    )
    model.maximize(objective)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.solve(model)
    if status != cp_model.OPTIMAL:
        status_name = solver.status_name(status)
        if status in (cp_model.FEASIBLE, cp_model.UNKNOWN):
            raise TimeoutError(
                "CP-SAT did not prove optimality within the configured time limit "
                f"(status={status_name})"
            )
        raise RuntimeError(f"CP-SAT failed to solve selection model: {status_name}")

    selected: dict[str, ConstructionPlan] = {}
    total_penalty = 0
    for request_id, _, plan, execution, var in rows:
        if solver.value(var):
            selected[request_id] = plan
            total_penalty += _penalty(execution)
    return ExactSelection(
        plans=selected,
        completed_requests=len(selected),
        footprint_cost=total_penalty,
        objective_value=int(round(solver.objective_value)),
    )
