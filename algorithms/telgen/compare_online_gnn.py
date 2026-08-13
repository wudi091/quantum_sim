"""Paired online comparison of GNN, exact MILP, and Q-CAST."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import shutil
from statistics import fmean
from time import perf_counter

from algorithms.qcast.online import OnlineQCASTConfig, run_online_qcast
from qnet_core.scenario import ScenarioConfig, make_episode
from qnet_core.spec import PhysicalConfig

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
    parser.add_argument("--paths", type=int, default=4)
    parser.add_argument("--construction-plans", type=int, default=5)
    parser.add_argument("--gnn-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gnn-decode-threshold", type=float)
    parser.add_argument("--milp-time-limit-seconds", type=float, default=300.0)
    parser.add_argument("--skip-milp", action="store_true")
    parser.add_argument("--generation-probability", type=float, default=0.8)
    parser.add_argument("--swap-probability", type=float, default=0.9)
    parser.add_argument("--memory-capacity", type=int, default=2)
    parser.add_argument("--node-memory-capacity", type=int)
    parser.add_argument("--quantum-distance-m", type=float, default=1000.0)
    parser.add_argument("--slot-duration-ps", type=int, default=50_000_000)
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seeds < 1 or args.requests < 1 or args.requests_per_batch < 1:
        raise ValueError("seeds and request counts must be positive")
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
    scenario = ScenarioConfig(
        request_count=args.requests,
        min_hops=args.min_hops,
        max_hops=args.max_hops,
        ttl=args.ttl,
        horizon=horizon,
        physical=physical,
        topology_nodes=args.nodes,
        topology_mode="waxman",
        waxman_alpha=0.15,
        waxman_beta=0.45,
        topology_attempts=128,
        waxman_add_mst=False,
        endpoint_mode="distance_stratified",
        arrival_batch_size=args.requests_per_batch,
        arrival_interval=args.decision_interval,
    )
    common = dict(
        decision_interval=args.decision_interval,
        path_candidate_count=args.paths,
        construction_kinds=(),
        swap_tree_count=args.construction_plans,
        purification_kinds=("none",),
        teacher_solver_backend="highs_ipm",
    )
    gnn_config = OnlineTELGENConfig(
        **common,
        decision_backend="gnn",
        gnn_checkpoint=str(args.checkpoint),
        gnn_device=args.gnn_device,
        gnn_decode_threshold=args.gnn_decode_threshold,
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
    trials = []
    started = perf_counter()
    for index, seed in enumerate(range(args.seed_start, args.seed_start + args.seeds)):
        episode = make_episode(scenario, seed)
        methods = {}
        gnn_started = perf_counter()
        gnn = run_online_telgen(episode, gnn_config)
        methods["gnn"] = {
            "metrics": dict(gnn.metrics),
            "wall_seconds": perf_counter() - gnn_started,
        }
        qcast_started = perf_counter()
        qcast = run_online_qcast(episode, qcast_config)
        methods["qcast"] = {
            "metrics": dict(qcast.metrics),
            "wall_seconds": perf_counter() - qcast_started,
        }
        if not args.skip_milp:
            milp_started = perf_counter()
            milp = run_online_telgen(episode, milp_config)
            methods["milp"] = {
                "metrics": dict(milp.metrics),
                "wall_seconds": perf_counter() - milp_started,
            }
        trials.append({"seed": seed, "methods": methods})
        summary = " ".join(
            f"{name}={int(item['metrics']['completed_requests'])}"
            for name, item in methods.items()
        )
        print(
            f"episode={index + 1}/{args.seeds} seed={seed} {summary}",
            flush=True,
        )
    payload = {
        "schema_version": 1,
        "experiment": "paired_online_gnn_milp_qcast",
        "configuration": {
            **vars(args),
            "checkpoint": str(args.checkpoint),
            "output": str(args.output),
            "resolved_horizon": horizon,
        },
        "scenario": asdict(scenario),
        "gnn_config": asdict(gnn_config),
        "milp_config": None if args.skip_milp else asdict(milp_config),
        "qcast_config": asdict(qcast_config),
        "elapsed_seconds": perf_counter() - started,
        "aggregate": _aggregate(trials),
        "trials": trials,
    }
    versioned, _ = _save(args.output, payload)
    for method, metrics in payload["aggregate"].items():
        print(
            f"{method}: completed={metrics['completed_requests']:.3f} "
            f"latency_ps={metrics['mean_censored_latency_ps']:.3f} "
            f"decision_s={metrics.get('mean_decision_seconds', 0.0):.6f}"
        )
    print(f"json: {versioned}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
