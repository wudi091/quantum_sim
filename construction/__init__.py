"""Construction-aware quantum routing kernel (Phase 1, deterministic).

Isolates con_design.md's claim that the *construction plan* C over a path P --
not P alone -- determines the resource footprint and hence the impact on
concurrent requests. Probabilistic generation and cross-slot state are
deferred to Phase 2.
"""

from .plan import ConstructionPlan, Edge, SwapNode, edge, elementary_ref, path_edges
from .enumerator import (
    balanced_plan,
    enumerate_constructions,
    intermediate_plan,
    sequential_plan,
)
from .simulator import PlanExecution, SlotSimulator, plan_footprint, simulate_plan
from .cpsat import ExactSelection, solve_cpsat

__all__ = [
    "ConstructionPlan",
    "Edge",
    "SwapNode",
    "edge",
    "elementary_ref",
    "path_edges",
    "balanced_plan",
    "enumerate_constructions",
    "intermediate_plan",
    "sequential_plan",
    "PlanExecution",
    "SlotSimulator",
    "plan_footprint",
    "simulate_plan",
    "ExactSelection",
    "solve_cpsat",
]
