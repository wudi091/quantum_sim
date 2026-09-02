"""Command-line entry point for one rolling TELGEN execution."""

from __future__ import annotations

import argparse

from qnet_core.scenario import ScenarioConfig, make_episode
from qnet_core.spec import PhysicalConfig

from .online import (
    OnlineTELGENConfig,
    run_online_telgen,
    save_online_milp_dataset,
    save_online_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TELGEN on one periodic micro-batch episode."
    )
    parser.add_argument("--output", default="results/telgen_online")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--min-hops", type=int, default=4)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--requests-per-batch", type=int, default=10)
    parser.add_argument("--decision-interval", type=int, default=4)
    parser.add_argument("--ttl", type=int, default=16)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--paths", type=int, default=4)
    parser.add_argument("--construction-plans", type=int, default=5)
    parser.add_argument(
        "--decision-backend",
        choices=("milp_teacher", "gnn", "ipm_gnn"),
        default="milp_teacher",
    )
    parser.add_argument("--gnn-checkpoint")
    parser.add_argument(
        "--gnn-device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--milp-time-limit-seconds", type=float, default=60.0)
    parser.add_argument(
        "--save-milp-dataset",
        action="store_true",
        help="save one GNN graph/label sample per non-empty MILP boundary",
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
        choices=("distance_stratified", "uniform_random"),
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.requests_per_batch < 1 or args.decision_interval < 1:
        raise ValueError("batch size and decision interval must be positive")
    if args.save_milp_dataset and args.decision_backend != "milp_teacher":
        raise ValueError(
            "--save-milp-dataset requires --decision-backend milp_teacher"
        )
    if (
        args.decision_backend in {"gnn", "ipm_gnn"}
        and not args.gnn_checkpoint
    ):
        raise ValueError(
            f"--decision-backend {args.decision_backend} requires "
            "--gnn-checkpoint"
        )
    arrival_rounds = (
        args.requests + args.requests_per_batch - 1
    ) // args.requests_per_batch
    last_arrival = (arrival_rounds - 1) * args.decision_interval
    horizon = last_arrival + args.ttl if args.horizon is None else args.horizon
    if horizon < last_arrival + args.ttl:
        raise ValueError("horizon must cover the final arrival's TTL")
    physical = PhysicalConfig(
        generation_probability=args.generation_probability,
        swap_probability=args.swap_probability,
        memory_capacity=args.memory_capacity,
        node_memory_capacity=args.node_memory_capacity,
        max_width=args.max_width,
        quantum_distance_m=args.quantum_distance_m,
        slot_duration_ps=args.slot_duration_ps,
    )
    min_hops = None if args.endpoint_mode == "uniform_random" else args.min_hops
    max_hops = None if args.endpoint_mode == "uniform_random" else args.max_hops
    scenario = ScenarioConfig(
        request_count=args.requests,
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
    spec = make_episode(scenario, args.seed)
    result = run_online_telgen(
        spec,
        OnlineTELGENConfig(
            decision_interval=args.decision_interval,
            path_candidate_count=args.paths,
            construction_kinds=(),
            swap_tree_count=args.construction_plans,
            purification_kinds=("none",),
            decision_backend="milp_teacher",
            milp_time_limit_seconds=args.milp_time_limit_seconds,
        )
        if args.decision_backend == "milp_teacher"
        else OnlineTELGENConfig(
                decision_interval=args.decision_interval,
                path_candidate_count=args.paths,
                construction_kinds=(),
                swap_tree_count=args.construction_plans,
                purification_kinds=("none",),
                decision_backend=args.decision_backend,
                gnn_checkpoint=args.gnn_checkpoint,
                gnn_device=args.gnn_device,
            ),
    )
    paths = save_online_result(result, args.output)
    dataset_paths = None
    if args.save_milp_dataset:
        dataset_paths = save_online_milp_dataset(
            result,
            f"{args.output}/milp_dataset_seed_{args.seed:08d}",
        )
    print(
        f"completed={int(result.metrics['completed_requests'])}/"
        f"{int(result.metrics['request_count'])} "
        f"attempts={int(result.metrics['construction_attempt_count'])} "
        f"retries={int(result.metrics['retry_count'])}"
    )
    print(f"json: {paths.json_path}")
    print(f"csv: {paths.csv_path}")
    if dataset_paths is not None:
        print(f"milp manifest: {dataset_paths.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
