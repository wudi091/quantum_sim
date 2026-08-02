"""Run the formal CON generator-oracle comparison.

The default profile is the workload agreed for the shared online environment:
20 nodes, 100 conditionally Poisson arrivals, 30 control slots, no cap on the
arrived/unexpired request backlog, and a fixed 4-path x 4-schedule cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qnet_core.order_waxman import WaxmanOrderConfig
from qnet_core.order_waxman_benchmark import (
    DEFAULT_PHYSICS_SEED_BASE,
    DEFAULT_PLANNER_SEED_BASE,
)

from .generator_oracle import DEFAULT_GENERATORS, run_generator_oracle_benchmark


def formal_config() -> WaxmanOrderConfig:
    return WaxmanOrderConfig(
        node_count=20,
        average_degree=4,
        target_link_probability=0.6,
        request_count=100,
        arrival_rate=100 / 30,
        episode_steps=30,
        request_ttl_slots=5,
        min_hops=2,
        max_hops=6,
        candidate_paths=4,
        order_variants_per_path=4,
        candidate_request_cap=None,
        node_memory_cap=2,
        epr_ttl_slots=3,
        slot_duration_ps=4_000,
        generation_interval_ps=1_000,
        swap_service_ps=1_000,
        memory_reset_ps=100,
        swap_probability=0.9,
        bsm_capacity_per_node=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Formal offline-generator + online-MILP paired benchmark"
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--generators",
        nargs="+",
        choices=DEFAULT_GENERATORS,
        default=DEFAULT_GENERATORS,
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=("qddca_fixed", "qcast_fixed"),
        default=("qddca_fixed", "qcast_fixed"),
    )
    parser.add_argument("--path-pool-per-pair", type=int, default=8)
    parser.add_argument(
        "--online-selector",
        choices=(
            "reliable_memory_milp",
            "static_relaxation",
            "exact_scenario_oracle",
        ),
        default="reliable_memory_milp",
        help=(
            "reliable_memory_milp is the time-indexed deterministic model; "
            "static_relaxation is the legacy necessary-condition MILP; "
            "exact_scenario_oracle exhaustively validates executor outcomes "
            "and is intended only for small snapshots"
        ),
    )
    parser.add_argument(
        "--reliability-confidence",
        type=float,
        default=0.9,
        help="marginal per-link confidence for reliable EPR supply",
    )
    parser.add_argument("--oracle-workers", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    output = args.output
    if output is None:
        output = Path("algorithms/con_method/benchmarks/results") / (
            "formal_generator_ddca_20n_100req_30step_"
            f"{args.episodes}epi_seed{args.base_seed}.json"
        )

    result = run_generator_oracle_benchmark(
        config=formal_config(),
        episode_seeds=range(args.base_seed, args.base_seed + args.episodes),
        generator_names=tuple(args.generators),
        baseline_names=tuple(args.baselines),
        path_pool_per_pair=args.path_pool_per_pair,
        max_hops=6,
        planning_seeds=(0,),
        online_selector=args.online_selector,
        reliability_confidence=args.reliability_confidence,
        oracle_workers=args.oracle_workers,
        physics_seed_base=DEFAULT_PHYSICS_SEED_BASE,
        planner_seed_base=DEFAULT_PLANNER_SEED_BASE,
        output_path=output,
    )
    print(json.dumps({
        "workload_config": result["workload_config"],
        "aggregate": result["aggregate"],
        "baseline_aggregate": result["baseline_aggregate"],
        "method_deltas": result["method_deltas"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
