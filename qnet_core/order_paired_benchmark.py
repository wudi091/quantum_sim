"""Paired same-snapshot comparison of nominal fixed- and joint-order MILPs.

The ordinary online benchmark lets every planner advance its own environment.
That is necessary for end-to-end policy evaluation, but it does not isolate the
one-slot value of adding complete swap-order candidates: different actions
produce different pending-request and inventory trajectories.

This module evaluates both nominal MILPs on the exact same immutable planning
snapshot before advancing one shared :class:`OrderEpisodeEnv`.  The configured
driver (fixed order by default) alone controls that state trajectory.  Since
the joint catalogue contains every canonical fixed-order candidate, its
one-slot objective must never be smaller on a paired snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Literal

from .order_episode_env import OrderEpisodeEnv
from .order_milp import (
    MilpNominalPathOrderPlanner,
    MilpNominalPathPlanner,
)
from .order_waxman import (
    WaxmanOrderConfig,
    WaxmanOrderEpisode,
    make_waxman_order_episode,
)
from .order_waxman_benchmark import (
    DEFAULT_PHYSICS_SEED_BASE,
    DEFAULT_PLANNER_SEED_BASE,
    _episode_manifest,
    _gym_config,
    _stream_seed,
)


DriverName = Literal["fixed", "joint"]


_SOLUTION_DIAGNOSTIC_FIELDS = (
    "backend",
    "eligible_candidates",
    "filtered_candidates",
    "enumerated_assignments",
    "static_upper_bound",
    "evaluations",
    "milp_solves",
    "cuts",
    "required_scenarios",
)


def _solution_diagnostics(planner: object) -> dict[str, object | None]:
    """Return a JSON-safe, backward-compatible solver diagnostic record."""

    solution = getattr(planner, "last_solution", None)
    diagnostics: dict[str, object | None] = {}
    for field in _SOLUTION_DIAGNOSTIC_FIELDS:
        value = getattr(solution, field, None)
        if value is None:
            diagnostics[field] = None
        elif field == "backend":
            diagnostics[field] = str(value)
        else:
            try:
                diagnostics[field] = int(value)
            except (TypeError, ValueError, OverflowError):
                diagnostics[field] = None

    # Older/custom planners may expose only this legacy counter.
    if diagnostics["evaluations"] is None:
        try:
            diagnostics["evaluations"] = int(
                getattr(planner, "last_evaluations", 0)
            )
        except (TypeError, ValueError, OverflowError):
            diagnostics["evaluations"] = None
    return diagnostics


def _print_slot_progress(
    *,
    episode_seed: int,
    slot_id: int,
    eligible_request_count: int,
    fixed_objective: int,
    joint_objective: int,
    fixed_enumerated_assignments: object | None,
    joint_enumerated_assignments: object | None,
    fixed_ms: float,
    joint_ms: float,
) -> None:
    """Emit one immediately visible, machine-greppable progress line."""

    fixed_enumerated = int(fixed_enumerated_assignments or 0)
    joint_enumerated = int(joint_enumerated_assignments or 0)
    print(
        "paired "
        f"seed={episode_seed} slot={slot_id} "
        f"eligible={eligible_request_count} "
        f"fixed_obj={fixed_objective} joint_obj={joint_objective} "
        f"fixed_enum={fixed_enumerated} joint_enum={joint_enumerated} "
        f"fixed_ms={fixed_ms:.3f} joint_ms={joint_ms:.3f}",
        file=sys.stderr,
        flush=True,
    )


def _objective(planner: object, selected: tuple[str, ...]) -> int:
    raw = getattr(planner, "last_objective", len(selected))
    value = float(raw)
    rounded = round(value)
    if abs(value - rounded) > 1e-9:
        raise RuntimeError(f"paired MILP objective is not integral: {value}")
    objective = int(rounded)
    if objective != len(selected):
        raise RuntimeError(
            "paired MILP objective disagrees with its selected request count: "
            f"objective={objective}, selected={len(selected)}"
        )
    return objective


def _is_proven_optimal(planner: object) -> bool:
    solution = getattr(planner, "last_solution", None)
    return bool(
        getattr(
            solution,
            "proven_optimal",
            getattr(solution, "certified_optimal", False),
        )
    )


def run_paired_episode(
    episode: WaxmanOrderEpisode,
    *,
    physics_seed_root: int,
    planner_seed: int,
    driver: DriverName = "fixed",
    oracle_workers: int = 1,
    fixed_planner: object | None = None,
    joint_planner: object | None = None,
    progress: bool = False,
) -> dict[str, object]:
    """Compare both MILPs on each shared snapshot and execute one driver.

    Optional planner objects are accepted to keep focused contract tests small;
    production callers normally use the two nominal MILPs constructed here.
    """

    if driver not in ("fixed", "joint"):
        raise ValueError("driver must be 'fixed' or 'joint'")
    if oracle_workers < 1:
        raise ValueError("oracle_workers must be positive")
    fixed = fixed_planner or MilpNominalPathPlanner(
        oracle_workers=oracle_workers
    )
    joint = joint_planner or MilpNominalPathOrderPlanner(
        oracle_workers=oracle_workers
    )
    fixed.reset(int(planner_seed))
    joint.reset(int(planner_seed))

    env = OrderEpisodeEnv(
        _gym_config(episode.config),
        episode.config,
        physics_seed_root=int(physics_seed_root),
    )
    env.reset(options={"episode": episode})

    slots: list[dict[str, object]] = []
    terminated = False
    while not terminated:
        slot_id = env.current_slot
        snapshot = env.planning_snapshot
        inventory_start_count = len(env.inventory)
        eligible_request_ids = tuple(env.current_eligible_request_ids)
        considered_request_ids = tuple(env.current_considered_request_ids)
        pruned_request_ids = tuple(env.current_pruned_request_ids)
        eligible_request_count = len(eligible_request_ids)
        considered_request_count = len(considered_request_ids)
        pruned_request_count = len(pruned_request_ids)
        if eligible_request_count != (
            considered_request_count + pruned_request_count
        ):
            raise AssertionError(
                "eligible requests must partition into considered and pruned"
            )

        if snapshot is None:
            snapshot_hash: str | None = None
            fixed_ids: tuple[str, ...] = ()
            joint_ids: tuple[str, ...] = ()
            fixed_objective = 0
            joint_objective = 0
            fixed_ms = 0.0
            joint_ms = 0.0
            fixed_evaluations = 0
            joint_evaluations = 0
            fixed_diagnostics = {
                field: (0 if field == "evaluations" else None)
                for field in _SOLUTION_DIAGNOSTIC_FIELDS
            }
            joint_diagnostics = dict(fixed_diagnostics)
            fixed_optimal = True
            joint_optimal = True
            candidate_count = 0
            fixed_candidate_count = 0
            order_relevant_requests = 0
            fixed_noncanonical = 0
            joint_noncanonical = 0
            fixed_request_ids: tuple[str, ...] = ()
            joint_request_ids: tuple[str, ...] = ()
            delta = 0
        else:
            snapshot_hash = hashlib.sha256(
                repr(snapshot.problem).encode("utf-8")
            ).hexdigest()
            candidate_count = len(snapshot.candidates)
            fixed_candidate_count = sum(
                plan.is_fixed_order for plan in snapshot.candidates
            )
            candidate_request_ids = {
                plan.request_id for plan in snapshot.candidates
            }
            if candidate_request_ids != set(considered_request_ids):
                raise AssertionError(
                    "paired candidate catalogue disagrees with the environment's "
                    "considered request set"
                )
            order_relevant_requests = len({
                plan.request_id
                for plan in snapshot.candidates
                if len(plan.swap_order) >= 2
            })

            started = perf_counter()
            fixed_ids = tuple(fixed.select(snapshot))
            fixed_ms = 1_000.0 * (perf_counter() - started)
            fixed_objective = _objective(fixed, fixed_ids)
            fixed_diagnostics = _solution_diagnostics(fixed)
            fixed_evaluations = int(fixed_diagnostics["evaluations"] or 0)
            fixed_optimal = _is_proven_optimal(fixed)

            started = perf_counter()
            select_with_incumbent = getattr(
                joint, "select_with_incumbent", None
            )
            if callable(select_with_incumbent):
                joint_ids = tuple(
                    select_with_incumbent(snapshot, fixed_ids)
                )
            else:
                # Backward compatibility for focused tests and external custom
                # planners that implement only the original select(snapshot)
                # protocol.
                joint_ids = tuple(joint.select(snapshot))
            joint_ms = 1_000.0 * (perf_counter() - started)
            joint_objective = _objective(joint, joint_ids)
            joint_diagnostics = _solution_diagnostics(joint)
            joint_evaluations = int(joint_diagnostics["evaluations"] or 0)
            joint_optimal = _is_proven_optimal(joint)

            post_solve_hash = hashlib.sha256(
                repr(snapshot.problem).encode("utf-8")
            ).hexdigest()
            if post_solve_hash != snapshot_hash:
                raise AssertionError(
                    "a paired planner mutated the shared planning snapshot"
                )
            if not fixed_optimal or not joint_optimal:
                raise AssertionError(
                    "paired comparison requires both model optima to be "
                    "proven"
                )

            if joint_objective < fixed_objective:
                raise AssertionError(
                    "joint-order objective fell below its fixed-order subset "
                    f"on episode {episode.seed}, slot {slot_id}: "
                    f"joint={joint_objective}, fixed={fixed_objective}"
                )
            delta = joint_objective - fixed_objective

            lookup = {
                plan.plan_id: plan for plan in snapshot.candidates
            }
            fixed_noncanonical = sum(
                not lookup[plan_id].is_fixed_order for plan_id in fixed_ids
            )
            joint_noncanonical = sum(
                not lookup[plan_id].is_fixed_order for plan_id in joint_ids
            )
            if fixed_noncanonical:
                raise AssertionError(
                    "fixed-order planner selected a noncanonical order"
                )
            if delta > 0 and joint_noncanonical == 0:
                raise AssertionError(
                    "positive paired objective gap has no noncanonical "
                    "joint-order selection"
                )
            fixed_request_ids = tuple(
                lookup[plan_id].request_id for plan_id in fixed_ids
            )
            joint_request_ids = tuple(
                lookup[plan_id].request_id for plan_id in joint_ids
            )

        driver_ids = fixed_ids if driver == "fixed" else joint_ids
        action = env.action_for_plan_ids(driver_ids)
        _, _, terminated, truncated, info = env.step(action)
        if truncated:
            raise RuntimeError("OrderEpisodeEnv truncated unexpectedly")

        slots.append({
            "slot": slot_id,
            "decision_slot": snapshot is not None,
            "snapshot_hash": snapshot_hash,
            "eligible_request_ids": eligible_request_ids,
            "considered_request_ids": considered_request_ids,
            "pruned_request_ids": pruned_request_ids,
            "eligible_request_count": eligible_request_count,
            "considered_request_count": considered_request_count,
            "pruned_request_count": pruned_request_count,
            # Backward-compatible alias.  Historical paired outputs used
            # request_count for the post-pruning considered catalogue.
            "request_count": considered_request_count,
            "order_relevant_requests": order_relevant_requests,
            "candidate_count": candidate_count,
            "fixed_candidate_count": fixed_candidate_count,
            "inventory_start_count": inventory_start_count,
            "fixed_objective": fixed_objective,
            "joint_objective": joint_objective,
            "delta": delta,
            "fixed_selected_plan_ids": fixed_ids,
            "joint_selected_plan_ids": joint_ids,
            "fixed_selected_request_ids": fixed_request_ids,
            "joint_selected_request_ids": joint_request_ids,
            "fixed_noncanonical_selected": fixed_noncanonical,
            "joint_noncanonical_selected": joint_noncanonical,
            "same_selected_plan_ids": set(fixed_ids) == set(joint_ids),
            "same_selected_request_ids": (
                set(fixed_request_ids) == set(joint_request_ids)
            ),
            "fixed_planning_ms": fixed_ms,
            "joint_planning_ms": joint_ms,
            "fixed_evaluations": fixed_evaluations,
            "joint_evaluations": joint_evaluations,
            "fixed_backend": fixed_diagnostics["backend"],
            "joint_backend": joint_diagnostics["backend"],
            "fixed_eligible_candidates": fixed_diagnostics[
                "eligible_candidates"
            ],
            "joint_eligible_candidates": joint_diagnostics[
                "eligible_candidates"
            ],
            "fixed_filtered_candidates": fixed_diagnostics[
                "filtered_candidates"
            ],
            "joint_filtered_candidates": joint_diagnostics[
                "filtered_candidates"
            ],
            "fixed_enumerated_assignments": fixed_diagnostics[
                "enumerated_assignments"
            ],
            "joint_enumerated_assignments": joint_diagnostics[
                "enumerated_assignments"
            ],
            "fixed_static_upper_bound": fixed_diagnostics[
                "static_upper_bound"
            ],
            "joint_static_upper_bound": joint_diagnostics[
                "static_upper_bound"
            ],
            "fixed_milp_solves": fixed_diagnostics["milp_solves"],
            "joint_milp_solves": joint_diagnostics["milp_solves"],
            "fixed_cuts": fixed_diagnostics["cuts"],
            "joint_cuts": joint_diagnostics["cuts"],
            "fixed_required_scenarios": fixed_diagnostics[
                "required_scenarios"
            ],
            "joint_required_scenarios": joint_diagnostics[
                "required_scenarios"
            ],
            "fixed_proven_optimal": fixed_optimal,
            "joint_proven_optimal": joint_optimal,
            "driver": driver,
            "driver_selected_plan_ids": driver_ids,
            "driver_completed_count": int(info["completed_count"]),
            "inventory_end_count": len(info["inventory_end"]),
        })
        if progress:
            _print_slot_progress(
                episode_seed=episode.seed,
                slot_id=slot_id,
                eligible_request_count=eligible_request_count,
                fixed_objective=fixed_objective,
                joint_objective=joint_objective,
                fixed_enumerated_assignments=fixed_diagnostics[
                    "enumerated_assignments"
                ],
                joint_enumerated_assignments=joint_diagnostics[
                    "enumerated_assignments"
                ],
                fixed_ms=fixed_ms,
                joint_ms=joint_ms,
            )

    decision_slots = [row for row in slots if row["decision_slot"]]
    positive_gap_slot_ids = tuple(
        int(row["slot"]) for row in slots if int(row["delta"]) > 0
    )
    metrics = env.metrics()
    return {
        "episode_seed": episode.seed,
        "driver": driver,
        "slot_count": len(slots),
        "decision_slot_count": len(decision_slots),
        "fixed_objective_sum": sum(
            int(row["fixed_objective"]) for row in slots
        ),
        "joint_objective_sum": sum(
            int(row["joint_objective"]) for row in slots
        ),
        "delta_sum": sum(int(row["delta"]) for row in slots),
        "mean_delta_per_decision_slot": (
            sum(int(row["delta"]) for row in decision_slots)
            / max(len(decision_slots), 1)
        ),
        "positive_gap_slots": len(positive_gap_slot_ids),
        "positive_gap_slot_ids": positive_gap_slot_ids,
        "fixed_planning_ms_sum": sum(
            float(row["fixed_planning_ms"]) for row in slots
        ),
        "joint_planning_ms_sum": sum(
            float(row["joint_planning_ms"]) for row in slots
        ),
        "fixed_evaluations_sum": sum(
            int(row["fixed_evaluations"]) for row in slots
        ),
        "joint_evaluations_sum": sum(
            int(row["joint_evaluations"]) for row in slots
        ),
        "joint_noncanonical_selected_sum": sum(
            int(row["joint_noncanonical_selected"]) for row in slots
        ),
        "fixed_optimal_slots": sum(
            bool(row["fixed_proven_optimal"]) for row in decision_slots
        ),
        "joint_optimal_slots": sum(
            bool(row["joint_proven_optimal"]) for row in decision_slots
        ),
        "executor_completed_count": int(metrics["completed_count"]),
        "executor_completion_rate": float(metrics["completion_rate"]),
        "environment_metrics": metrics,
        "slots": slots,
    }


def _aggregate(rows: list[dict[str, object]]) -> dict[str, float | int]:
    total_slots = sum(int(row["slot_count"]) for row in rows)
    decision_slots = sum(int(row["decision_slot_count"]) for row in rows)
    fixed_sum = sum(int(row["fixed_objective_sum"]) for row in rows)
    joint_sum = sum(int(row["joint_objective_sum"]) for row in rows)
    delta_sum = joint_sum - fixed_sum
    return {
        "episodes": len(rows),
        "slots": total_slots,
        "decision_slots": decision_slots,
        "fixed_objective_sum": fixed_sum,
        "joint_objective_sum": joint_sum,
        "delta_sum": delta_sum,
        "fixed_objective_mean_per_decision_slot": (
            fixed_sum / max(decision_slots, 1)
        ),
        "joint_objective_mean_per_decision_slot": (
            joint_sum / max(decision_slots, 1)
        ),
        "delta_mean_per_decision_slot": (
            delta_sum / max(decision_slots, 1)
        ),
        "positive_gap_slots": sum(
            int(row["positive_gap_slots"]) for row in rows
        ),
        "fixed_planning_ms_sum": sum(
            float(row["fixed_planning_ms_sum"]) for row in rows
        ),
        "joint_planning_ms_sum": sum(
            float(row["joint_planning_ms_sum"]) for row in rows
        ),
        "fixed_evaluations_sum": sum(
            int(row["fixed_evaluations_sum"]) for row in rows
        ),
        "joint_evaluations_sum": sum(
            int(row["joint_evaluations_sum"]) for row in rows
        ),
        "joint_noncanonical_selected_sum": sum(
            int(row["joint_noncanonical_selected_sum"]) for row in rows
        ),
        "executor_completed_mean_per_episode": sum(
            int(row["executor_completed_count"]) for row in rows
        ) / len(rows),
    }


def run_suite(
    *,
    episodes: int = 3,
    base_seed: int = 0,
    physics_seed_base: int = DEFAULT_PHYSICS_SEED_BASE,
    planner_seed_base: int = DEFAULT_PLANNER_SEED_BASE,
    config: WaxmanOrderConfig = WaxmanOrderConfig(),
    driver: DriverName = "fixed",
    oracle_workers: int = 1,
    progress: bool = False,
) -> dict[str, object]:
    """Run a reproducible paired benchmark over independent episodes."""

    if episodes < 1:
        raise ValueError("episodes must be positive")
    if driver not in ("fixed", "joint"):
        raise ValueError("driver must be 'fixed' or 'joint'")
    if oracle_workers < 1:
        raise ValueError("oracle_workers must be positive")

    episode_seeds = tuple(int(base_seed) + index for index in range(episodes))
    planner_seeds = tuple(
        _stream_seed(planner_seed_base, seed, "planner")
        for seed in episode_seeds
    )
    physics_seed_roots = tuple(
        _stream_seed(physics_seed_base, seed, "hidden-physics")
        for seed in episode_seeds
    )
    workloads = tuple(
        make_waxman_order_episode(config, seed) for seed in episode_seeds
    )
    rows = [
        run_paired_episode(
            episode,
            physics_seed_root=physics_seed_roots[index],
            planner_seed=planner_seeds[index],
            driver=driver,
            oracle_workers=oracle_workers,
            progress=progress,
        )
        for index, episode in enumerate(workloads)
    ]
    return {
        "model": {
            "comparison": "paired same-snapshot nominal MILP",
            "fixed_planner": "MilpNominalPathPlanner",
            "joint_planner": "MilpNominalPathOrderPlanner",
            "dominance_invariant": "joint_objective >= fixed_objective",
            "driver": driver,
            "oracle_workers": int(oracle_workers),
            "driver_semantics": (
                "both planners evaluate one immutable snapshot; only the "
                f"{driver} planner advances the shared OrderEpisodeEnv"
            ),
            "planning_scenarios": [0],
            "hidden_physics_visible_to_planners": False,
            "request_scope": (
                "all_active_pending"
                if config.candidate_request_cap is None
                else "edf_capped_active_pending"
            ),
            "candidate_request_cap": config.candidate_request_cap,
            "pruning_rule": (
                "none"
                if config.candidate_request_cap is None
                else "earliest_deadline_then_arrival_then_request_id"
            ),
            "swap_order_scope": (
                "all_permutations_per_configured_path"
                if config.order_variants_per_path is None
                else "capped_permutations_per_configured_path"
            ),
            "order_variants_per_path_cap": (
                config.order_variants_per_path
            ),
            "max_swap_orders_per_path": config.max_swap_orders_per_path,
        },
        "config": config.__dict__,
        "episode_count": episodes,
        "base_seed": int(base_seed),
        "physics_seed_base": int(physics_seed_base),
        "planner_seed_base": int(planner_seed_base),
        "episode_seeds": episode_seeds,
        "planner_seeds": planner_seeds,
        "topologies": [_episode_manifest(episode) for episode in workloads],
        "rows": rows,
        "aggregate": _aggregate(rows),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed- and joint-order nominal MILPs on identical "
            "per-slot snapshots"
        )
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--physics-seed-base",
        type=int,
        default=DEFAULT_PHYSICS_SEED_BASE,
    )
    parser.add_argument(
        "--planner-seed-base",
        type=int,
        default=DEFAULT_PLANNER_SEED_BASE,
    )
    parser.add_argument("--driver", choices=("fixed", "joint"), default="fixed")
    parser.add_argument(
        "--oracle-workers",
        type=int,
        default=1,
        help=(
            "spawned workers for batched nominal executor validation; "
            "use 1 for the legacy serial callback"
        ),
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="flush one progress line to stderr after every completed slot",
    )
    parser.add_argument("--nodes", type=int, default=20)
    parser.add_argument("--average-degree", type=int, default=4)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=None,
        help="used only when --steps is omitted by programmatic callers",
    )
    parser.add_argument("--ttl", type=int, default=5)
    parser.add_argument("--min-hops", type=int, default=2)
    parser.add_argument("--max-hops", type=int, default=6)
    parser.add_argument("--candidate-paths", type=int, default=4)
    parser.add_argument(
        "--order-variants",
        type=int,
        default=4,
        help=(
            "complete swap-order groups retained per candidate path "
            "(default: 4)"
        ),
    )
    parser.add_argument(
        "--candidate-request-cap",
        type=int,
        default=None,
        help=(
            "optional EDF cap on arrived, unexpired pending requests in each "
            "paired snapshot; omit to use the complete active backlog"
        ),
    )
    parser.add_argument("--node-memory", type=int, default=2)
    parser.add_argument("--epr-ttl", type=int, default=3)
    parser.add_argument("--slot-duration-ps", type=int, default=4_000)
    parser.add_argument("--generation-interval-ps", type=int, default=1_000)
    parser.add_argument("--swap-service-ps", type=int, default=1_000)
    parser.add_argument("--memory-reset-ps", type=int, default=100)
    parser.add_argument("--swap-probability", type=float, default=0.9)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/order_paired_same_snapshot.json"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = WaxmanOrderConfig(
        node_count=args.nodes,
        average_degree=args.average_degree,
        request_count=args.requests,
        arrival_rate=(
            args.arrival_rate
            if args.arrival_rate is not None
            else args.requests / args.steps
        ),
        episode_steps=args.steps,
        request_ttl_slots=args.ttl,
        min_hops=args.min_hops,
        max_hops=args.max_hops,
        candidate_paths=args.candidate_paths,
        order_variants_per_path=args.order_variants,
        candidate_request_cap=args.candidate_request_cap,
        node_memory_cap=args.node_memory,
        epr_ttl_slots=args.epr_ttl,
        slot_duration_ps=args.slot_duration_ps,
        generation_interval_ps=args.generation_interval_ps,
        swap_service_ps=args.swap_service_ps,
        memory_reset_ps=args.memory_reset_ps,
        swap_probability=args.swap_probability,
    )
    result = run_suite(
        episodes=args.episodes,
        base_seed=args.base_seed,
        physics_seed_base=args.physics_seed_base,
        planner_seed_base=args.planner_seed_base,
        config=config,
        driver=args.driver,
        oracle_workers=args.oracle_workers,
        progress=args.progress,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
