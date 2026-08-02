"""Greedy construction-aware planner.

For each request, enumerate (P, C) candidates and commit the feasible plan
with the smallest resource footprint. The score prioritizes lower peak
memory, then shorter peak duration and makespan.
"""

from __future__ import annotations

from .enumerator import enumerate_constructions
from .plan import ConstructionPlan
from .simulator import PlanExecution, SlotSimulator, simulate_plan


def _cost(execution: PlanExecution) -> tuple[int, int, int]:
    return (
        sum(execution.peak_memory.values()),
        sum(execution.duration_at_peak.values()),
        execution.makespan,
    )


def greedy_select(
    slot: SlotSimulator,
    candidate_paths: list[tuple[int, ...]],
) -> ConstructionPlan | None:
    """Choose and commit the lowest-cost feasible (P, C), or None."""
    choices: list[tuple[tuple[int, int, int], ConstructionPlan, PlanExecution]] = []
    for path in candidate_paths:
        for plan in enumerate_constructions(path):
            execution = simulate_plan(plan)
            if slot.can_admit(execution):
                choices.append((_cost(execution), plan, execution))
    if not choices:
        return None
    _, plan, execution = min(choices, key=lambda item: (item[0], item[1].kind))
    slot.admit(plan, execution)
    return plan
