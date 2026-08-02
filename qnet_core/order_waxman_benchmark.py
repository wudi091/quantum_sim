"""Multi-step Waxman benchmark on the shared OrderEpisodeEnv.

The formal CLI profile uses 30 independently seeded episodes, not "30 seeds"
as a synonym for environment steps.  Each episode has one fixed topology,
100 conditionally Poisson arrivals, and exactly 30 control-slot steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Callable

from .order_episode_env import OrderEpisodeEnv
from .order_gym_env import OrderGymConfig
from .order_planners import (
    QCASTFixedOrderPlanner,
    QDDCAFixedOrderPlanner,
    SAAPathOrderPlanner,
    SAAPathPlanner,
)
from .order_waxman import (
    WaxmanOrderConfig,
    WaxmanOrderEpisode,
    make_waxman_order_episode,
)


PlannerFactory = Callable[[], object]


DEFAULT_PHYSICS_SEED_BASE = 0x5150_4859_53
DEFAULT_PLANNER_SEED_BASE = 0x5150_4C41_4E


def _stream_seed(base: int, episode_index: int, domain: str) -> int:
    """Derive a reproducible 32-bit seed in an explicitly separate domain."""

    payload = f"{domain}|{int(base)}|{int(episode_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _milp_nominal_path_factory() -> object:
    # Kept local so benchmark utilities that do not select this planner remain
    # importable while the general nominal MILP implementation is optional.
    from .order_milp import MilpNominalPathPlanner

    return MilpNominalPathPlanner()


def _milp_nominal_path_order_factory() -> object:
    from .order_milp import MilpNominalPathOrderPlanner

    return MilpNominalPathOrderPlanner()


def _episode_manifest(episode: WaxmanOrderEpisode) -> dict[str, object]:
    paths = episode.paths
    return {
        "episode_seed": episode.seed,
        "nodes": list(episode.nodes),
        "positions": [
            {"node": node, "x": position[0], "y": position[1]}
            for node, position in episode.positions
        ],
        "links": [
            {
                "left": link.left,
                "right": link.right,
                "capacity": link.capacity,
                "generation_probability": link.generation_probability,
            }
            for link in episode.links
        ],
        "node_capacities": dict(episode.node_capacities),
        "requests": [
            {
                "request_id": request.request_id,
                "source": request.source,
                "destination": request.destination,
                "arrival_slot": request.arrival_slot,
                "deadline_slot": request.deadline_slot,
                "shortest_hops": request.shortest_hops,
                "candidate_paths": [
                    list(path) for path in paths[request.request_id]
                ],
            }
            for request in episode.requests
        ],
        "topology_beta": episode.topology_beta,
        "link_alpha": episode.link_alpha,
        "horizon_slots": episode.horizon_slots,
    }


def _gym_config(config: WaxmanOrderConfig) -> OrderGymConfig:
    return OrderGymConfig(
        max_nodes=max(128, config.node_count),
        max_edges=max(512, config.node_count * config.average_degree),
        # Padding is an episode-level tensor capacity, not a request-pruning
        # rule.  Keep it large enough for the complete active backlog even
        # when an exact benchmark explicitly caps its considered request set.
        max_requests=config.request_count,
        max_candidates=(
            config.request_count
            * config.candidate_paths
            * config.max_swap_orders_per_path
        ),
        max_hops=config.max_hops,
    )


def run_planner_episode(
    episode: WaxmanOrderEpisode,
    planner: object,
    *,
    physics_seed_root: int,
    planner_seed: int,
) -> dict[str, object]:
    """Run one fixed-topology multi-step episode in the shared environment."""

    planning_seconds = 0.0
    planner_calls = 0
    planning_simulations = 0.0
    model_objective_sum = 0.0
    model_objective_slots = 0
    model_optimal_slots = 0
    model_executor_completed = 0
    env = OrderEpisodeEnv(
        _gym_config(episode.config),
        episode.config,
        physics_seed_root=int(physics_seed_root),
    )
    _, _ = env.reset(options={"episode": episode})
    planner.reset(int(planner_seed))
    terminated = False
    while not terminated:
        model_objective: float | None = None
        model_proven_optimal = False
        if env.planning_snapshot is None:
            selected: tuple[str, ...] = ()
        else:
            started = perf_counter()
            selected = tuple(planner.select(env.planning_snapshot))
            planning_seconds += perf_counter() - started
            planner_calls += 1
            planning_simulations += float(
                getattr(planner, "last_evaluations", 0)
            )
            raw_objective = getattr(planner, "last_objective", None)
            if raw_objective is not None:
                model_objective = float(raw_objective)
                solution = getattr(planner, "last_solution", None)
                model_proven_optimal = bool(
                    getattr(
                        solution,
                        "proven_optimal",
                        getattr(solution, "certified_optimal", False),
                    )
                )
        action = env.action_for_plan_ids(selected)
        _, _, terminated, truncated, info = env.step(action)
        if truncated:
            raise RuntimeError("OrderEpisodeEnv truncated unexpectedly")
        if model_objective is not None:
            model_objective_sum += model_objective
            model_objective_slots += 1
            model_optimal_slots += int(model_proven_optimal)
            model_executor_completed += int(info["completed_count"])

    metrics = env.metrics()
    row: dict[str, object] = {
        "episode_seed": episode.seed,
        **metrics,
        # This explicit alias makes it impossible to confuse physical
        # execution with a nominal MILP's per-slot model objective.
        "executor_completed_count": metrics["completed_count"],
        "planner_calls": planner_calls,
        "mean_planning_ms": (
            1000.0 * planning_seconds / max(planner_calls, 1)
        ),
        "planning_simulations": planning_simulations,
    }
    if model_objective_slots:
        row.update({
            "milp_model_objective_sum": model_objective_sum,
            "milp_model_objective_mean_per_solve": (
                model_objective_sum / model_objective_slots
            ),
            "milp_model_objective_slots": model_objective_slots,
            "milp_model_optimal_slots": model_optimal_slots,
            "milp_executor_completed_count": model_executor_completed,
            "milp_model_minus_executor_completed": (
                model_objective_sum - model_executor_completed
            ),
        })
    return row


def _aggregate(rows: list[dict[str, object]]) -> dict[str, float]:
    keys = tuple(
        key for key, value in rows[0].items()
        if key != "episode_seed" and isinstance(value, (int, float))
    )
    return {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in keys
    }


def run_suite(
    *,
    episodes: int | None = None,
    base_seed: int = 0,
    physics_seed_base: int = DEFAULT_PHYSICS_SEED_BASE,
    planner_seed_base: int = DEFAULT_PLANNER_SEED_BASE,
    seeds: int | None = None,
    config: WaxmanOrderConfig = WaxmanOrderConfig(),
    oracle_rollouts: int = 2,
    planner_names: tuple[str, ...] = (
        "qddca_fixed", "qcast_fixed",
        "milp_nominal_path", "milp_nominal_path_order",
    ),
) -> dict[str, object]:
    if episodes is None:
        episode_count = 3 if seeds is None else int(seeds)
    else:
        episode_count = int(episodes)
        if seeds is not None and int(seeds) != episode_count:
            raise ValueError("episodes and legacy seeds alias disagree")
    if episode_count < 1 or oracle_rollouts < 1:
        raise ValueError("episodes and oracle_rollouts must be positive")
    episode_seeds = tuple(
        int(base_seed) + index for index in range(episode_count)
    )
    # Derive streams from the stable episode seed rather than the local list
    # index.  This makes one 30-episode run bit-identical to independently
    # executed chunks such as 0--9, 10--19, and 20--29.
    physics_seed_roots = tuple(
        _stream_seed(physics_seed_base, episode_seed, "hidden-physics")
        for episode_seed in episode_seeds
    )
    planner_seeds = tuple(
        _stream_seed(planner_seed_base, episode_seed, "planner")
        for episode_seed in episode_seeds
    )
    rollout_seeds = tuple(200_000 + index for index in range(oracle_rollouts))
    available: dict[str, PlannerFactory] = {
        "qddca_fixed": QDDCAFixedOrderPlanner,
        "qcast_fixed": QCASTFixedOrderPlanner,
        "milp_nominal_path": _milp_nominal_path_factory,
        "milp_nominal_path_order": _milp_nominal_path_order_factory,
        "saa_path": lambda: SAAPathPlanner(rollout_seeds),
        "saa_path_order": lambda: SAAPathOrderPlanner(rollout_seeds),
        # Read-only compatibility for old commands/checkpoints.  New output
        # should use the corrected SAA names above.
        "optimal_path": lambda: SAAPathPlanner(rollout_seeds),
        "optimal_path_order": lambda: SAAPathOrderPlanner(rollout_seeds),
    }
    unknown = set(planner_names) - set(available)
    if unknown:
        raise ValueError(f"unknown planners: {sorted(unknown)}")
    rows = {name: [] for name in planner_names}
    topology_rows: list[dict[str, object]] = []
    for episode_index, episode_seed in enumerate(episode_seeds):
        episode = make_waxman_order_episode(config, episode_seed)
        topology_rows.append(_episode_manifest(episode))
        for name in planner_names:
            rows[name].append(run_planner_episode(
                episode,
                available[name](),
                physics_seed_root=physics_seed_roots[episode_index],
                planner_seed=planner_seeds[episode_index],
            ))
    aggregate = {
        name: _aggregate(values) for name, values in rows.items()
    }
    gaps: dict[str, float] = {}
    order_name = next(
        (
            name for name in (
                "milp_nominal_path_order",
                "saa_path_order",
                "optimal_path_order",
            )
            if name in aggregate
        ),
        None,
    )
    if order_name is not None:
        order_rate = aggregate[order_name]["completion_rate"]
        for name in planner_names:
            if name != order_name:
                gaps[f"{order_name}_minus_{name}"] = (
                    order_rate - aggregate[name]["completion_rate"]
                )
    return {
        "model": {
            "rl_environment": "OrderEpisodeEnv",
            "episode_semantics": (
                "one fixed topology/request trace per episode; one env.step "
                "advances exactly one control slot"
            ),
            "action_semantics": (
                "one atomic multi-hot batch of complete path/swap-order plans"
            ),
            "topology": "Q-CAST author-compatible Waxman-like generator",
            "request_process": (
                "fixed-count homogeneous Poisson conditioned on the episode "
                "window"
                if config.episode_steps is not None
                else "homogeneous Poisson (exponential inter-arrivals)"
            ),
            "source_destination_sampling": (
                "uniform random connected pairs within hop range"
            ),
            "requests_per_topology": config.request_count,
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
            "gnn_candidate_catalogue": (
                f"up to {config.candidate_paths} configured paths x "
                + (
                    "all complete swap-order permutations"
                    if config.order_variants_per_path is None
                    else (
                        f"up to {config.order_variants_per_path} complete "
                        "swap orders"
                    )
                )
            ),
            "path_only_baseline_projection": (
                "Q-DDCA and Q-CAST score paths with their own rules and use "
                "only the canonical fixed order; they do not optimize the "
                "GNN order catalogue"
            ),
            "controller_actions_per_slot": 1,
            "generation_controlled_by_planner": False,
            "test_physics_seed_visible_to_planner": False,
            "physics_seed_isolation": (
                "one hidden physics_seed_root per episode is shared by every "
                "planner, but is never passed to planner.reset or snapshot"
            ),
            "planner_seed_isolation": (
                "planner.reset receives a separate domain seed independent "
                "of the episode/workload seed and hidden physics stream"
            ),
            "milp_nominal_objective": (
                "maximize requests completed by the current deterministic "
                "nominal slot model; this is recorded separately from "
                "stochastic shared-executor completions"
            ),
            "milp_nominal_planning_scenarios": [0],
            "milp_nominal_required_scenarios": 1,
            "saa_rollouts": oracle_rollouts,
            "batch_optimizer": (
                "planner-specific: nominal MILP for milp_nominal_*; optional "
                "finite-rollout enumeration only for explicitly named saa_*"
            ),
            "batch_optimizers": {
                "milp_nominal_path": "nominal mixed-integer linear program",
                "milp_nominal_path_order": (
                    "nominal mixed-integer linear program over complete "
                    "path/swap-order candidates"
                ),
                "saa_path": "optional finite-rollout SAA enumeration",
                "saa_path_order": "optional finite-rollout SAA enumeration",
            },
            "slot_inventory_boundary": (
                "unconsumed elementary EPRs persist across control slots with "
                "fixed TTL; swapped segments are discarded at slot boundaries"
            ),
        },
        "config": config.__dict__,
        "episodes": episode_count,
        "base_seed": int(base_seed),
        "physics_seed_base": int(physics_seed_base),
        "planner_seed_base": int(planner_seed_base),
        "episode_seeds": episode_seeds,
        # Planner seeds may be published because they contain no physics
        # stream information.  Hidden physics roots are intentionally absent.
        "planner_seeds": planner_seeds,
        "seeds": episode_count,
        "topologies": topology_rows,
        "rows": rows,
        "aggregate": aggregate,
        "gaps": gaps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed-horizon multi-episode Waxman routing benchmark"
    )
    parser.add_argument(
        "--episodes", "--seeds", dest="episodes", type=int, default=30,
        help="number of independent episodes; --seeds is a legacy alias",
    )
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--physics-seed-base", type=int,
        default=DEFAULT_PHYSICS_SEED_BASE,
    )
    parser.add_argument(
        "--planner-seed-base", type=int,
        default=DEFAULT_PLANNER_SEED_BASE,
    )
    parser.add_argument("--nodes", type=int, default=20)
    parser.add_argument("--average-degree", type=int, default=4)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--arrival-rate", type=float, default=None,
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
            "optional EDF cap on arrived, unexpired pending requests exposed "
            "to each planner; omit to expose the complete active backlog"
        ),
    )
    parser.add_argument("--node-memory", type=int, default=2)
    parser.add_argument("--epr-ttl", type=int, default=3)
    parser.add_argument("--slot-duration-ps", type=int, default=4_000)
    parser.add_argument("--generation-interval-ps", type=int, default=1_000)
    parser.add_argument("--swap-service-ps", type=int, default=1_000)
    parser.add_argument("--memory-reset-ps", type=int, default=100)
    parser.add_argument("--swap-probability", type=float, default=0.9)
    parser.add_argument("--oracle-rollouts", type=int, default=2)
    parser.add_argument(
        "--planners", nargs="+",
        choices=(
            "qddca_fixed", "qcast_fixed",
            "milp_nominal_path", "milp_nominal_path_order",
            "saa_path", "saa_path_order",
            "optimal_path", "optimal_path_order",
        ),
        default=(
            "qddca_fixed", "qcast_fixed",
            "milp_nominal_path", "milp_nominal_path_order",
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "results/order_episode_20n_100req_30step_30epi_stress_medium.json"
        ),
    )
    args = parser.parse_args()
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
        oracle_rollouts=args.oracle_rollouts,
        planner_names=tuple(args.planners),
    )
    payload = json.dumps(result, indent=2, default=str)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
