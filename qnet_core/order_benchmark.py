"""Legacy one-slot hotspot mechanism benchmark, not the training environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Callable

from .order_core import OrderBatchProblem
from .order_gym_env import OrderGymConfig, OrderGymEnv
from .order_milp import (
    DeterministicPathMilpPlanner,
    DeterministicPathOrderMilpPlanner,
)
from .order_planners import (
    QCASTFixedOrderPlanner,
    QDDCAFixedOrderPlanner,
    SAAPathOrderPlanner,
    SAAPathPlanner,
)
from .order_scenarios import make_seeded_hotspot_problem


PlannerFactory = Callable[[], object]


def run_planner(
    problem: OrderBatchProblem,
    planner: object,
) -> dict[str, object]:
    planner.reset(problem.config.seed)
    env = OrderGymEnv(OrderGymConfig(
        max_nodes=64,
        max_edges=128,
        max_requests=16,
        max_candidates=512,
        max_hops=10,
    ))
    env.reset(options={"problem": problem})
    snapshot = env.planning_snapshot
    started = perf_counter()
    selected = tuple(planner.select(snapshot))
    planning_ms = 1000.0 * (perf_counter() - started)
    row = execute_selected(
        problem,
        selected,
        planning_ms=planning_ms,
        planning_simulations=float(getattr(planner, "last_evaluations", 0)),
    )
    objective = getattr(planner, "last_objective", None)
    row["milp_objective"] = objective
    row["milp_objective_matches_execution"] = (
        None if objective is None
        else int(objective) == int(row["completed_count"])
    )
    row["milp_certified_optimal"] = (
        None if objective is None else bool(
            row["milp_objective_matches_execution"]
            and getattr(
                getattr(planner, "last_solution", None),
                "certified_optimal",
                False,
            )
        )
    )
    return row


def execute_selected(
    problem: OrderBatchProblem,
    selected: tuple[str, ...],
    *,
    planning_ms: float,
    planning_simulations: float,
) -> dict[str, object]:
    env = OrderGymEnv(OrderGymConfig(
        max_nodes=64,
        max_edges=128,
        max_requests=16,
        max_candidates=512,
        max_hops=10,
    ))
    env.reset(options={"problem": problem})
    for plan_id in selected:
        env.step(env.action_for_plan(plan_id))
    _, _, terminated, truncated, _ = env.step(env.stop_action)
    if not terminated or truncated or env.core.result is None:
        raise RuntimeError("order Gym environment did not settle after STOP")
    result = env.core.result
    lookup = {plan.plan_id: plan for plan in problem.candidates}
    required_orders = {
        lookup[plan_id].request_id: lookup[plan_id].swap_order
        for plan_id in selected
        if lookup[plan_id].request_id in problem.required_requests
    }
    total_requests = len({plan.request_id for plan in problem.candidates})
    waiting_candidates = tuple(
        plan for plan in problem.candidates
        if plan.request_id not in problem.required_requests
    )
    hotspot_counts: dict[object, int] = {}
    for plan in waiting_candidates:
        for node in plan.path[1:-1]:
            hotspot_counts[node] = hotspot_counts.get(node, 0) + 1
    hotspot = max(
        hotspot_counts,
        key=lambda node: (hotspot_counts[node], repr(node)),
    )
    hotspot_release_positions = [
        order.index(hotspot)
        for order in required_orders.values()
        if hotspot in order
    ]
    completion_times = tuple(result.completion_time_ps.values())
    blocked_memory = sum(
        event.status == "blocked_memory"
        for trace in result.traces
        for event in trace.generation_events
    )
    successful_swaps = sum(
        event.status == "success"
        for trace in result.traces
        for event in trace.swap_events
    )
    return {
        "problem": problem.name,
        "completed_count": result.completed_count,
        "completion_rate": result.completed_count / total_requests,
        "mean_completion_time_ps": (
            sum(completion_times) / len(completion_times)
            if completion_times else 0.0
        ),
        "selected_count": len(selected),
        "hotspot_release_first": float(
            bool(hotspot_release_positions)
            and all(position == 0 for position in hotspot_release_positions)
        ),
        "blocked_memory_events": blocked_memory,
        "successful_swaps": successful_swaps,
        "planning_ms": planning_ms,
        "planning_simulations": planning_simulations,
        "selected_plan_ids": list(selected),
        "required_orders": {
            request_id: list(order)
            for request_id, order in required_orders.items()
        },
    }


def _aggregate(rows: list[dict[str, object]]) -> dict[str, float]:
    numeric = (
        "completed_count",
        "completion_rate",
        "mean_completion_time_ps",
        "selected_count",
        "hotspot_release_first",
        "blocked_memory_events",
        "successful_swaps",
        "planning_ms",
        "planning_simulations",
    )
    return {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in numeric
    }


def run_suite(
    *,
    seeds: int,
    hotspot_capacities: tuple[int, ...] = (2, 4),
    generation_probability: float = 1.0,
    swap_probability: float = 1.0,
    saa_rollouts: int = 1,
    physics_replicates: int = 1,
) -> dict[str, object]:
    if seeds < 1:
        raise ValueError("seeds must be positive")
    if saa_rollouts < 1:
        raise ValueError("saa_rollouts must be positive")
    if physics_replicates < 1:
        raise ValueError("physics_replicates must be positive")
    rollout_seeds = tuple(100_000 + index for index in range(saa_rollouts))
    deterministic = (
        generation_probability == 1.0 and swap_probability == 1.0
    )
    path_name = "milp_path" if deterministic else "saa_path"
    order_name = "milp_path_order" if deterministic else "saa_path_order"
    factories: dict[str, PlannerFactory] = {
        "qddca_fixed": QDDCAFixedOrderPlanner,
        "qcast_fixed": QCASTFixedOrderPlanner,
        path_name: (
            DeterministicPathMilpPlanner
            if deterministic
            else lambda: SAAPathPlanner(rollout_seeds)
        ),
        order_name: (
            DeterministicPathOrderMilpPlanner
            if deterministic
            else lambda: SAAPathOrderPlanner(rollout_seeds)
        ),
    }
    cases: dict[str, object] = {}
    for capacity in hotspot_capacities:
        rows = {name: [] for name in factories}
        for structure_seed in range(seeds):
            planning_problem = make_seeded_hotspot_problem(
                structure_seed,
                hotspot_capacity=capacity,
                generation_probability=generation_probability,
                swap_probability=swap_probability,
                physics_seed=0,
            )
            for name, factory in factories.items():
                planner = factory()
                planner.reset(structure_seed)
                planning_env = OrderGymEnv(OrderGymConfig(
                    max_nodes=64,
                    max_edges=128,
                    max_requests=16,
                    max_candidates=512,
                    max_hops=10,
                ))
                planning_env.reset(options={"problem": planning_problem})
                started = perf_counter()
                selected = tuple(planner.select(
                    planning_env.planning_snapshot
                ))
                planning_ms = 1000.0 * (perf_counter() - started)
                planning_simulations = float(
                    getattr(planner, "last_evaluations", 0)
                )
                for replicate in range(physics_replicates):
                    physics_seed = (
                        1_000_000
                        + structure_seed * physics_replicates
                        + replicate
                    )
                    execution_problem = planning_problem.with_physics_seed(
                        physics_seed
                    )
                    row = execute_selected(
                        execution_problem,
                        selected,
                        planning_ms=planning_ms,
                        planning_simulations=planning_simulations,
                    )
                    objective = getattr(planner, "last_objective", None)
                    row["milp_objective"] = objective
                    row["milp_objective_matches_execution"] = (
                        None if objective is None
                        else int(objective) == int(row["completed_count"])
                    )
                    row["milp_certified_optimal"] = (
                        None if objective is None else bool(
                            row["milp_objective_matches_execution"]
                            and getattr(
                                getattr(planner, "last_solution", None),
                                "certified_optimal",
                                False,
                            )
                        )
                    )
                    row["structure_seed"] = structure_seed
                    row["physics_replicate"] = replicate
                    row["physics_seed"] = physics_seed
                    rows[name].append(row)
        aggregate = {
            name: _aggregate(values) for name, values in rows.items()
        }
        path = aggregate[path_name]["completion_rate"]
        order = aggregate[order_name]["completion_rate"]
        qddca = aggregate["qddca_fixed"]["completion_rate"]
        qcast = aggregate["qcast_fixed"]["completion_rate"]
        cases[f"hotspot_capacity_{capacity}"] = {
            "rows": rows,
            "aggregate": aggregate,
            "gaps": {
                f"{path_name}_minus_qddca_fixed": path - qddca,
                "qcast_fixed_minus_qddca_fixed": qcast - qddca,
                f"{order_name}_minus_{path_name}": order - path,
                f"{order_name}_minus_qddca_fixed": order - qddca,
                f"{order_name}_minus_qcast_fixed": order - qcast,
            },
        }
    return {
        "model": {
            "controller_actions_per_slot": 1,
            "generation_controlled_by_planner": False,
            "timing": (
                "event-driven physical picoseconds: slot duration, automatic "
                "HEG interval, BSM service, and memory reset"
            ),
            "swap_probability": swap_probability,
            "generation_probability": generation_probability,
            "saa_rollouts": None if deterministic else saa_rollouts,
            "batch_optimizer": (
                "deterministic profile-relaxation MILP with executor certificate"
                if deterministic else
                "finite-rollout sample-average candidate optimizer"
            ),
            "planner_display_names": {
                path_name: "MILP-Path" if deterministic else "SAA-Path",
                order_name: (
                    "MILP-Path+Order" if deterministic else "SAA-Path+Order"
                ),
            },
            "physics_replicates_per_structure": physics_replicates,
            "physics_seed_visible_to_planner": False,
            "mechanism_test_environment": "legacy one-slot OrderGymEnv",
            "formal_training_environment": "OrderEpisodeEnv (not used here)",
            "candidate_scope": "small linear-path swap-order catalogue",
            "milp_certificate": (
                "executor completed count attains the MILP upper bound"
                if deterministic else None
            ),
        },
        "seeds": seeds,
        "paired_physics_episodes_per_capacity": seeds * physics_replicates,
        "paired_physics_episodes_total": (
            seeds * physics_replicates * len(hotspot_capacities)
        ),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fixed-path and complete swap-order planners"
    )
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument(
        "--hotspot-capacities", type=int, nargs="+", default=(2, 4)
    )
    parser.add_argument("--p-gen", type=float, default=1.0)
    parser.add_argument("--p-swap", type=float, default=1.0)
    parser.add_argument(
        "--saa-rollouts", "--oracle-rollouts",
        dest="saa_rollouts", type=int, default=1,
    )
    parser.add_argument("--physics-replicates", type=int, default=1)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/order_core_30seed.json"),
    )
    args = parser.parse_args()
    result = run_suite(
        seeds=args.seeds,
        hotspot_capacities=tuple(args.hotspot_capacities),
        generation_probability=args.p_gen,
        swap_probability=args.p_swap,
        saa_rollouts=args.saa_rollouts,
        physics_replicates=args.physics_replicates,
    )
    payload = json.dumps(result, indent=2, default=str)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(json.dumps({
        name: value["aggregate"]
        for name, value in result["cases"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
