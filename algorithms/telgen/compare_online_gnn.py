"""Paired online comparison on one shared generated episode suite."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import shutil
from statistics import fmean
from time import perf_counter

import networkx as nx

from algorithms.baselines.online import (
    OnlineBaselineConfig,
    run_online_baseline,
)
from algorithms.qcast.online import OnlineQCASTConfig, run_online_qcast
from qnet_core.planning_spec import RequestSpec
from qnet_core.scenario import ScenarioConfig, make_episode
from qnet_core.spec import EpisodeSpec, PhysicalConfig

from .online import OnlineTELGENConfig, run_online_telgen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare an online GNN checkpoint against MILP and Q-CAST."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=15000)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--requests-per-batch", type=int, default=5)
    parser.add_argument("--decision-interval", type=int, default=4)
    parser.add_argument("--ttl", type=int, default=16)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--min-hops", type=int, default=4)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument(
        "--endpoint-mode",
        choices=("distance_stratified", "uniform_random"),
        default="distance_stratified",
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
        "--topology-file",
        type=Path,
        help="optional node-link JSON topology; overrides generated topology",
    )
    parser.add_argument("--waxman-alpha", type=float, default=0.15)
    parser.add_argument("--waxman-beta", type=float, default=0.45)
    parser.add_argument("--topology-attempts", type=int, default=128)
    parser.add_argument("--waxman-add-mst", action="store_true")
    parser.add_argument("--barabasi-attachment", type=int, default=2)
    parser.add_argument("--erdos-renyi-mean-degree", type=float, default=6.0)
    parser.add_argument("--random-regular-degree", type=int, default=4)
    parser.add_argument("--parallel-corridors", type=int, default=2)
    parser.add_argument("--paths", type=int, default=4)
    parser.add_argument("--construction-plans", type=int, default=5)
    parser.add_argument(
        "--fixed-swap-tree-index",
        type=int,
        help=(
            "Restrict the GNN candidate set to one swap_tree_k. Omit this "
            "option for adaptive per-request construction selection."
        ),
    )
    parser.add_argument("--gnn-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--milp-time-limit-seconds", type=float, default=300.0)
    parser.add_argument("--skip-milp", action="store_true")
    parser.add_argument(
        "--skip-qcast",
        action="store_true",
        help="run only the configured GNN variant for paired ablations",
    )
    parser.add_argument("--skip-qpass", action="store_true")
    parser.add_argument("--skip-greedy", action="store_true")
    parser.add_argument("--generation-probability", type=float, default=0.8)
    parser.add_argument("--swap-probability", type=float, default=0.9)
    parser.add_argument("--memory-capacity", type=int, default=2)
    parser.add_argument("--node-memory-capacity", type=int)
    parser.add_argument("--quantum-distance-m", type=float, default=1000.0)
    parser.add_argument("--slot-duration-ps", type=int, default=50_000_000)
    parser.add_argument(
        "--time-segments",
        type=int,
        default=0,
        help="include per-segment long-run stability metrics (0 disables)",
    )
    return parser


def _aggregate(trials: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    methods = sorted({method for trial in trials for method in trial["methods"]})
    result = {}
    for method in methods:
        rows = [trial["methods"][method]["metrics"] for trial in trials]
        keys = sorted(set.intersection(*(set(row) for row in rows)))
        result[method] = {
            key: float(fmean(float(row[key]) for row in rows))
            for key in keys
        }
        result[method]["wall_seconds"] = float(fmean(
            float(trial["methods"][method]["wall_seconds"])
            for trial in trials
        ))
    return result


def _save(output: Path, payload: dict[str, object]) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned = output / f"online_gnn_comparison_{timestamp}.json"
    latest = output / "online_gnn_comparison.json"
    versioned.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copyfile(versioned, latest)
    return versioned, latest


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stability_segments(result, segment_count: int) -> list[dict[str, object]]:
    """Summarize one continuous run in equal slot intervals."""

    if segment_count < 1:
        return []
    horizon_slots = int(result.horizon_slots)
    if horizon_slots < 1:
        return []
    edges = [
        (horizon_slots * index) // segment_count
        for index in range(segment_count + 1)
    ]
    slot_duration_ps = int(result.episode.physical.slot_duration_ps)
    settlements = tuple(result.settlements)
    decisions = tuple(result.decisions)
    segments: list[dict[str, object]] = []
    for index, (start_slot, end_slot) in enumerate(
        zip(edges, edges[1:]),
        start=1,
    ):
        if end_slot <= start_slot:
            continue
        start_ps = start_slot * slot_duration_ps
        end_ps = end_slot * slot_duration_ps
        completed = sum(
            1
            for item in settlements
            if item.success and start_ps <= item.settlement_time < end_ps
        )
        planner_times = [
            float(item.planner_seconds)
            for item in decisions
            if start_slot <= item.decision_slot < end_slot
            and item.eligible_request_ids
        ]
        decision_times = [
            float(item.decision_seconds)
            for item in decisions
            if start_slot <= item.decision_slot < end_slot
        ]
        segment_slots = end_slot - start_slot
        segments.append({
            "segment": index,
            "start_slot": start_slot,
            "end_slot": end_slot,
            "completed_requests": float(completed),
            "throughput_per_slot": completed / segment_slots,
            "mean_planner_seconds": (
                0.0 if not planner_times else fmean(planner_times)
            ),
            "mean_decision_seconds": (
                0.0 if not decision_times else fmean(decision_times)
            ),
        })
    return segments


def _method_payload(
    result,
    wall_seconds: float,
    time_segments: int = 0,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "metrics": dict(result.metrics),
        "wall_seconds": wall_seconds,
        "violations": [asdict(item) for item in result.violations],
    }
    if time_segments:
        payload["stability_segments"] = _stability_segments(
            result,
            time_segments,
        )
    return payload


def _topology_file_episode(
    path: Path,
    *,
    seed: int,
    request_count: int,
    requests_per_batch: int,
    arrival_interval: int,
    ttl: int,
    horizon: int,
    physical: PhysicalConfig,
    demand_pairs: int = 1,
) -> EpisodeSpec:
    """Build an episode from a TopoHub node-link JSON file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_nodes = payload.get("nodes")
    raw_links = payload.get("links", payload.get("edges"))
    if not isinstance(raw_nodes, list) or not isinstance(raw_links, list):
        raise ValueError(
            f"topology file must contain node-link 'nodes' and 'edges': {path}"
        )
    node_keys = [
        str(item["id"])
        for item in raw_nodes
        if isinstance(item, dict) and "id" in item
    ]
    if len(node_keys) < 2 or len(set(node_keys)) != len(node_keys):
        raise ValueError(f"topology file has invalid node ids: {path}")
    node_index = {key: index for index, key in enumerate(node_keys)}
    edges: set[tuple[int, int]] = set()
    for item in raw_links:
        if (
            not isinstance(item, dict)
            or "source" not in item
            or "target" not in item
        ):
            raise ValueError(f"topology file has an invalid link: {path}")
        left = node_index.get(str(item["source"]))
        right = node_index.get(str(item["target"]))
        if left is None or right is None or left == right:
            raise ValueError(f"topology link references an unknown node: {path}")
        edges.add((min(left, right), max(left, right)))
    graph = nx.Graph()
    graph.add_nodes_from(range(len(node_keys)))
    graph.add_edges_from(edges)
    if not nx.is_connected(graph):
        raise ValueError(f"topology file must be connected: {path}")

    nodes = tuple(sorted(graph.nodes))
    rng = random.Random((int(seed) << 1) ^ 0x544F504F)
    endpoints = [tuple(rng.sample(nodes, 2)) for _ in range(request_count)]
    arrivals = [
        (index // requests_per_batch) * arrival_interval
        for index in range(request_count)
    ]
    if arrivals[-1] + ttl > horizon:
        raise ValueError(
            "horizon must cover the final topology-file request TTL"
        )
    requests = tuple(
        RequestSpec(
            f"r{index}",
            int(endpoints[index][0]),
            int(endpoints[index][1]),
            arrival=int(arrivals[index]),
            ttl=int(ttl),
            demand_pairs=int(demand_pairs),
        )
        for index in range(request_count)
    )
    return EpisodeSpec(
        seed=int(seed),
        nodes=nodes,
        edges=tuple(sorted(edges)),
        requests=requests,
        horizon=int(horizon),
        physical=physical,
    )


def _resolve_construction_space(
    construction_plans: int,
    fixed_swap_tree_index: int | None,
) -> tuple[tuple[str, ...], int | None, str]:
    if construction_plans < 1:
        raise ValueError("construction-plans must be positive")
    if fixed_swap_tree_index is None:
        return (), construction_plans, "adaptive_swap_tree_selection"
    if not 0 <= fixed_swap_tree_index < construction_plans:
        raise ValueError(
            "fixed-swap-tree-index must lie in "
            "[0, construction-plans)"
        )
    return (
        (f"swap_tree_{fixed_swap_tree_index}",),
        None,
        f"fixed_swap_tree_{fixed_swap_tree_index}",
    )


def _routing_baseline_configs(
    *,
    decision_interval: int,
    path_candidate_count: int,
) -> dict[str, OnlineBaselineConfig]:
    return {
        algorithm: OnlineBaselineConfig(
            algorithm=algorithm,
            decision_interval=decision_interval,
            path_candidate_count=path_candidate_count,
            construction_kind="left_deep",
        )
        for algorithm in ("qpass", "greedy")
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seeds < 1 or args.requests < 1 or args.requests_per_batch < 1:
        raise ValueError("seeds and request counts must be positive")
    if args.time_segments < 0:
        raise ValueError("time-segments cannot be negative")
    construction_kinds, swap_tree_count, construction_policy = (
        _resolve_construction_space(
            args.construction_plans,
            args.fixed_swap_tree_index,
        )
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
        max_width=1,
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
        waxman_alpha=args.waxman_alpha,
        waxman_beta=args.waxman_beta,
        topology_attempts=args.topology_attempts,
        waxman_add_mst=args.waxman_add_mst,
        endpoint_mode=args.endpoint_mode,
        barabasi_attachment=args.barabasi_attachment,
        erdos_renyi_mean_degree=args.erdos_renyi_mean_degree,
        random_regular_degree=args.random_regular_degree,
        parallel_corridors=args.parallel_corridors,
        arrival_batch_size=args.requests_per_batch,
        arrival_interval=args.decision_interval,
    )
    common = dict(
        decision_interval=args.decision_interval,
        path_candidate_count=args.paths,
        construction_kinds=construction_kinds,
        swap_tree_count=swap_tree_count,
        purification_kinds=("none",),
    )
    gnn_config = OnlineTELGENConfig(
        **common,
        decision_backend="ipm_gnn",
        gnn_checkpoint=str(args.checkpoint),
        gnn_device=args.gnn_device,
    )
    milp_config = OnlineTELGENConfig(
        **common,
        decision_backend="milp_teacher",
        milp_time_limit_seconds=args.milp_time_limit_seconds,
        milp_relative_gap=0.0,
    )
    qcast_config = OnlineQCASTConfig(
        decision_interval=args.decision_interval,
        path_candidate_count=args.paths,
        construction_kind="left_deep",
        purification_kind="none",
    )
    routing_baseline_configs = _routing_baseline_configs(
        decision_interval=args.decision_interval,
        path_candidate_count=args.paths,
    )
    trials = []
    started = perf_counter()
    for index, seed in enumerate(range(args.seed_start, args.seed_start + args.seeds)):
        if args.topology_file is None:
            episode = make_episode(scenario, seed)
        else:
            episode = _topology_file_episode(
                args.topology_file,
                seed=seed,
                request_count=args.requests,
                requests_per_batch=args.requests_per_batch,
                arrival_interval=args.decision_interval,
                ttl=args.ttl,
                horizon=horizon,
                physical=physical,
            )
        methods = {}
        gnn_started = perf_counter()
        gnn = run_online_telgen(episode, gnn_config)
        methods["gnn"] = _method_payload(
            gnn,
            perf_counter() - gnn_started,
            args.time_segments,
        )
        if not args.skip_qcast:
            qcast_started = perf_counter()
            qcast = run_online_qcast(episode, qcast_config)
            methods["qcast"] = _method_payload(
                qcast,
                perf_counter() - qcast_started,
                args.time_segments,
            )
        if not args.skip_qpass:
            qpass_started = perf_counter()
            qpass = run_online_baseline(
                episode,
                routing_baseline_configs["qpass"],
            )
            methods["qpass"] = _method_payload(
                qpass,
                perf_counter() - qpass_started,
                args.time_segments,
            )
        if not args.skip_greedy:
            greedy_started = perf_counter()
            greedy = run_online_baseline(
                episode,
                routing_baseline_configs["greedy"],
            )
            methods["greedy"] = _method_payload(
                greedy,
                perf_counter() - greedy_started,
                args.time_segments,
            )
        if not args.skip_milp:
            milp_started = perf_counter()
            milp = run_online_telgen(episode, milp_config)
            methods["milp"] = _method_payload(
                milp,
                perf_counter() - milp_started,
                args.time_segments,
            )
        trials.append({
            "seed": seed,
            "episode": asdict(episode),
            "methods": methods,
        })
        summary = " ".join(
            f"{name}={int(item['metrics']['completed_requests'])}"
            for name, item in methods.items()
        )
        print(
            f"episode={index + 1}/{args.seeds} seed={seed} {summary}",
            flush=True,
        )
    configuration = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    payload = {
        "schema_version": 1,
        "experiment": "paired_online_gnn_routing_baselines",
        "comparison_contract": {
            "paired_episode_spec": True,
            "independent_persistent_executors": True,
            "future_requests_hidden": True,
            "gnn_calls_milp_online": False,
            "qcast_uses_gnn_or_milp": False,
            "qcast_included": not args.skip_qcast,
            "qpass_uses_gnn_or_milp": False,
            "qpass_included": not args.skip_qpass,
            "greedy_uses_gnn_or_milp": False,
            "greedy_included": not args.skip_greedy,
            "primary_metric": "mean_completion_delay_ps",
            "secondary_metrics": [
                "max_completion_delay_ps",
                "mean_final_fidelity_loss",
                "mean_planner_seconds",
                "completion_delay_gini",
            ],
            "throughput_metric": "throughput_per_slot",
            "gnn_construction_policy": construction_policy,
            "gnn_decision_backend": "ipm_gnn",
            "time_segments": args.time_segments,
            "topology_file": (
                None if args.topology_file is None else str(args.topology_file)
            ),
        },
        "configuration": {
            **configuration,
            "checkpoint": str(args.checkpoint),
            "output": str(args.output),
            "resolved_horizon": horizon,
        },
        "checkpoint_sha256": _checkpoint_sha256(args.checkpoint),
        "scenario": asdict(scenario),
        "gnn_config": asdict(gnn_config),
        "milp_config": None if args.skip_milp else asdict(milp_config),
        "qcast_config": None if args.skip_qcast else asdict(qcast_config),
        "qpass_config": (
            None
            if args.skip_qpass
            else asdict(routing_baseline_configs["qpass"])
        ),
        "greedy_config": (
            None
            if args.skip_greedy
            else asdict(routing_baseline_configs["greedy"])
        ),
        "elapsed_seconds": perf_counter() - started,
        "aggregate": _aggregate(trials),
        "trials": trials,
    }
    versioned, _ = _save(args.output, payload)
    for method, metrics in payload["aggregate"].items():
        print(
            f"{method}: completed={metrics['completed_requests']:.3f} "
            f"mean_delay_ps={metrics['mean_completion_delay_ps']:.3f} "
            f"max_delay_ps={metrics['max_completion_delay_ps']:.3f} "
            f"fidelity_loss={metrics['mean_final_fidelity_loss']:.6f} "
            f"gini={metrics['completion_delay_gini']:.4f} "
            f"decision_s={metrics.get('mean_decision_seconds', 0.0):.6f}"
        )
    print(f"json: {versioned}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
