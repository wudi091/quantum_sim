"""Run one or all non-learning baselines on the same generated episode."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import shutil
from time import perf_counter

from qnet_core.scenario import ScenarioConfig, make_episode
from qnet_core.spec import PhysicalConfig
from qnet_core.workload import resolve_periodic_arrival_workload

from .online import (
    OnlineBaselineConfig,
    run_online_baseline,
    save_online_baseline_result,
)
from .planner import BASELINE_ALGORITHMS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run simulator-neutral non-learning planners through the shared "
            "online SeQUeNCe environment."
        )
    )
    parser.add_argument(
        "--algorithm",
        choices=("all", *BASELINE_ALGORITHMS),
        default="all",
    )
    parser.add_argument("--output", type=Path, default=Path("results/baselines_online"))
    parser.add_argument("--seed", type=int, default=100)
    workload = parser.add_mutually_exclusive_group()
    workload.add_argument(
        "--requests",
        type=int,
        help="legacy mode: fixed total request count",
    )
    workload.add_argument(
        "--arrival-rounds",
        type=int,
        help=(
            "Q-CAST-style mode: fixed traffic rounds with exactly "
            "--requests-per-batch new requests per round"
        ),
    )
    parser.add_argument("--requests-per-batch", type=int, default=10)
    parser.add_argument("--decision-interval", type=int, default=4)
    parser.add_argument("--ttl", type=int, default=16)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--min-hops", type=int, default=4)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--paths", type=int, default=4)
    parser.add_argument(
        "--construction-kind",
        choices=("left_deep", "balanced"),
        default="left_deep",
    )
    parser.add_argument(
        "--topology-mode",
        choices=(
            "waxman",
            "barabasi_albert",
            "erdos_renyi",
            "random_regular",
            "parallel_corridors",
        ),
        default="waxman",
    )
    parser.add_argument(
        "--endpoint-mode",
        choices=("distance_stratified", "uniform_random", "qcast_random"),
        default="distance_stratified",
    )
    parser.add_argument("--waxman-alpha", type=float, default=0.15)
    parser.add_argument("--waxman-beta", type=float, default=0.45)
    parser.add_argument("--topology-attempts", type=int, default=128)
    parser.add_argument("--waxman-add-mst", action="store_true")
    parser.add_argument("--barabasi-attachment", type=int, default=2)
    parser.add_argument("--erdos-renyi-mean-degree", type=float, default=6.0)
    parser.add_argument("--random-regular-degree", type=int, default=4)
    parser.add_argument("--parallel-corridors", type=int, default=2)
    parser.add_argument("--generation-probability", type=float, default=0.8)
    parser.add_argument("--swap-probability", type=float, default=0.9)
    parser.add_argument("--memory-capacity", type=int, default=2)
    parser.add_argument("--node-memory-capacity", type=int)
    parser.add_argument("--max-width", type=int, default=1)
    parser.add_argument("--quantum-distance-m", type=float, default=1000.0)
    parser.add_argument("--slot-duration-ps", type=int, default=50_000_000)
    return parser


def _save_summary(
    output: Path,
    payload: dict[str, object],
) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned = output / f"non_learning_comparison_{timestamp}.json"
    collision_index = 1
    while versioned.exists():
        collision_index += 1
        versioned = output / (
            f"non_learning_comparison_{timestamp}_{collision_index}.json"
        )
    latest = output / "non_learning_comparison.json"
    versioned.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    shutil.copyfile(versioned, latest)
    return versioned, latest


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workload = resolve_periodic_arrival_workload(
        request_count=args.requests,
        arrival_rounds=args.arrival_rounds,
        requests_per_round=args.requests_per_batch,
        arrival_interval_slots=args.decision_interval,
        ttl_slots=args.ttl,
        horizon_slots=args.horizon,
        default_request_count=100,
    )
    horizon = workload.horizon_slots
    physical = PhysicalConfig(
        generation_probability=args.generation_probability,
        swap_probability=args.swap_probability,
        memory_capacity=args.memory_capacity,
        node_memory_capacity=args.node_memory_capacity,
        max_width=args.max_width,
        quantum_distance_m=args.quantum_distance_m,
        slot_duration_ps=args.slot_duration_ps,
    )
    min_hops = (
        args.min_hops
        if args.endpoint_mode == "distance_stratified"
        else None
    )
    max_hops = (
        args.max_hops
        if args.endpoint_mode == "distance_stratified"
        else None
    )
    scenario = ScenarioConfig(
        request_count=workload.request_count,
        min_hops=min_hops,
        max_hops=max_hops,
        ttl=args.ttl,
        horizon=horizon,
        physical=physical,
        topology_nodes=args.nodes,
        topology_mode=args.topology_mode,
        endpoint_mode=args.endpoint_mode,
        waxman_alpha=args.waxman_alpha,
        waxman_beta=args.waxman_beta,
        topology_attempts=args.topology_attempts,
        waxman_add_mst=args.waxman_add_mst,
        barabasi_attachment=args.barabasi_attachment,
        erdos_renyi_mean_degree=args.erdos_renyi_mean_degree,
        random_regular_degree=args.random_regular_degree,
        parallel_corridors=args.parallel_corridors,
        arrival_batch_size=args.requests_per_batch,
        arrival_interval=args.decision_interval,
    )
    episode = make_episode(scenario, args.seed)
    algorithms = (
        BASELINE_ALGORITHMS
        if args.algorithm == "all"
        else (args.algorithm,)
    )
    methods: dict[str, object] = {}
    for algorithm in algorithms:
        config = OnlineBaselineConfig(
            algorithm=algorithm,
            decision_interval=args.decision_interval,
            path_candidate_count=args.paths,
            construction_kind=args.construction_kind,
        )
        started = perf_counter()
        result = run_online_baseline(episode, config)
        wall_seconds = perf_counter() - started
        paths = save_online_baseline_result(result, args.output)
        methods[algorithm] = {
            "config": asdict(config),
            "metrics": dict(result.metrics),
            "wall_seconds": wall_seconds,
            "violation_count": len(result.violations),
            "result_json": str(paths.json_path),
            "result_csv": str(paths.csv_path),
        }
        print(
            f"{algorithm}: "
            f"completed={int(result.metrics['completed_requests'])}/"
            f"{int(result.metrics['request_count'])} "
            f"violations={len(result.violations)} "
            f"wall_s={wall_seconds:.3f}",
            flush=True,
        )
    summary, _ = _save_summary(args.output, {
        "schema_version": 1,
        "experiment": "paired_non_learning_baselines",
        "comparison_contract": {
            "paired_episode_spec": True,
            "independent_persistent_executors": True,
            "future_requests_hidden": True,
            "milp_called": False,
            "learned_model_called": False,
            "shared_resource_time_contract": True,
            "shared_sequence_physical_backend": True,
        },
        "configuration": {
            **vars(args),
            "output": str(args.output),
            "workload": asdict(workload),
            "resolved_request_count": workload.request_count,
            "resolved_horizon": horizon,
        },
        "scenario": asdict(scenario),
        "episode": asdict(episode),
        "methods": methods,
    })
    print(f"summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
