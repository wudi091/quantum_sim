"""Paired comparison: IPM GNN, LP, Q-PASS, and Greedy.

The IPM GNN and LP share one adaptive construction space and are decoded with
the single minimal decoder in the packing module; Q-PASS and Greedy use their
fixed left-deep construction rule.  All four methods run on the same paired
EpisodeSpec instances through the same persistent SeQUeNCe executor.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import shutil
from statistics import fmean
from time import perf_counter

from algorithms.baselines.online import (
    OnlineBaselineConfig,
    run_online_baseline,
)
from qnet_core.scenario import ScenarioConfig, make_episode
from qnet_core.spec import PhysicalConfig
from qnet_core.workload import resolve_periodic_arrival_workload

from .online import OnlineTELGENConfig, run_online_telgen


METHOD_ORDER = ("ipm_gnn", "lp", "qpass", "greedy")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=15000)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--requests-per-batch", type=int, default=5)
    parser.add_argument("--decision-interval", type=int, default=4)
    parser.add_argument("--ttl", type=int, default=16)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--min-hops", type=int, default=4)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument(
        "--endpoint-mode",
        choices=("distance_stratified", "uniform_random", "qcast_random"),
        default="distance_stratified",
    )
    parser.add_argument(
        "--topology-mode",
        choices=(
            "waxman",
            "barabasi_albert",
            "erdos_renyi",
            "random_regular",
        ),
        default="waxman",
    )
    parser.add_argument("--waxman-alpha", type=float, default=0.15)
    parser.add_argument("--waxman-beta", type=float, default=0.45)
    parser.add_argument("--paths", type=int, default=4)
    parser.add_argument("--construction-plans", type=int, default=5)
    parser.add_argument("--ipm-steps", type=int, default=16)
    parser.add_argument("--generation-probability", type=float, default=0.8)
    parser.add_argument("--swap-probability", type=float, default=0.9)
    parser.add_argument("--memory-capacity", type=int, default=2)
    parser.add_argument("--node-memory-capacity", type=int)
    parser.add_argument("--quantum-distance-m", type=float, default=1000.0)
    parser.add_argument("--slot-duration-ps", type=int, default=50_000_000)
    parser.add_argument("--ipm-gnn-device", choices=("auto", "cpu", "cuda"), default="cpu")
    return parser


def _adaptive_config(args: argparse.Namespace, backend: str) -> OnlineTELGENConfig:
    common = dict(
        decision_interval=args.decision_interval,
        path_candidate_count=args.paths,
        construction_kinds=(),
        swap_tree_count=args.construction_plans,
        purification_kinds=("none",),
    )
    if backend == "lp":
        return OnlineTELGENConfig(**common, decision_backend="lp_teacher")
    return OnlineTELGENConfig(
        **common,
        decision_backend="ipm_gnn",
        ipm_gnn_checkpoint=str(args.checkpoint),
        ipm_gnn_device=args.ipm_gnn_device,
        ipm_steps=args.ipm_steps,
    )


def _baseline_config(args: argparse.Namespace, algorithm: str) -> OnlineBaselineConfig:
    return OnlineBaselineConfig(
        algorithm=algorithm,
        decision_interval=args.decision_interval,
        path_candidate_count=args.paths,
        construction_kind="left_deep",
    )


def _metrics_payload(result) -> dict[str, float]:
    return dict(result.metrics)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seeds < 1:
        raise ValueError("seeds must be positive")
    workload = resolve_periodic_arrival_workload(
        request_count=args.requests,
        arrival_rounds=None,
        requests_per_round=args.requests_per_batch,
        arrival_interval_slots=args.decision_interval,
        ttl_slots=args.ttl,
        horizon_slots=args.horizon,
        default_request_count=20,
    )
    horizon = workload.horizon_slots
    physical = PhysicalConfig(
        generation_probability=args.generation_probability,
        swap_probability=args.swap_probability,
        memory_capacity=args.memory_capacity,
        node_memory_capacity=args.node_memory_capacity,
        max_width=1,
        quantum_distance_m=args.quantum_distance_m,
        slot_duration_ps=args.slot_duration_ps,
    )
    min_hops = args.min_hops if args.endpoint_mode == "distance_stratified" else None
    max_hops = args.max_hops if args.endpoint_mode == "distance_stratified" else None
    scenario = ScenarioConfig(
        request_count=workload.request_count,
        min_hops=min_hops,
        max_hops=max_hops,
        ttl=args.ttl,
        horizon=horizon,
        physical=physical,
        topology_nodes=args.nodes,
        topology_mode=args.topology_mode,
        waxman_alpha=args.waxman_alpha,
        waxman_beta=args.waxman_beta,
        topology_attempts=128,
        waxman_add_mst=True,
        endpoint_mode=args.endpoint_mode,
        arrival_batch_size=args.requests_per_batch,
        arrival_interval=args.decision_interval,
    )
    configs = {
        "ipm_gnn": _adaptive_config(args, "ipm_gnn"),
        "lp": _adaptive_config(args, "lp"),
        "qpass": _baseline_config(args, "qpass"),
        "greedy": _baseline_config(args, "greedy"),
    }
    trials = []
    started = perf_counter()
    for index, seed in enumerate(range(args.seed_start, args.seed_start + args.seeds)):
        episode = make_episode(scenario, seed)
        methods = {}
        for name in METHOD_ORDER:
            method_started = perf_counter()
            if name in {"ipm_gnn", "lp"}:
                result = run_online_telgen(episode, configs[name])
            else:
                result = run_online_baseline(episode, configs[name])
            methods[name] = {
                "metrics": _metrics_payload(result),
                "wall_seconds": perf_counter() - method_started,
            }
        trials.append({"seed": seed, "methods": methods})
        summary = " ".join(
            f"{name}={int(item['metrics']['completed_requests'])}"
            for name, item in methods.items()
        )
        print(f"episode={index + 1}/{args.seeds} seed={seed} {summary}", flush=True)
    aggregate = {}
    for name in METHOD_ORDER:
        rows = [trial["methods"][name]["metrics"] for trial in trials]
        keys = sorted(set.intersection(*(set(row) for row in rows)))
        aggregate[name] = {
            key: float(fmean(float(row[key]) for row in rows))
            for key in keys
        }
        aggregate[name]["wall_seconds"] = float(fmean(
            trial["methods"][name]["wall_seconds"] for trial in trials
        ))
    payload = {
        "schema_version": 1,
        "experiment": "paired_ipm_gnn_lp_qpass_greedy",
        "configuration": {
            **{
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
            },
            "resolved_horizon": horizon,
        },
        "scenario": asdict(scenario),
        "elapsed_seconds": perf_counter() - started,
        "aggregate": aggregate,
        "trials": trials,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned = args.output / f"lp_ipm_gnn_comparison_{timestamp}.json"
    latest = args.output / "lp_ipm_gnn_comparison.json"
    versioned.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(versioned, latest)
    print("\n=== aggregate ===")
    for name in METHOD_ORDER:
        metrics = aggregate[name]
        print(
            f"{name}: completed={metrics['completed_requests']:.3f} "
            f"latency_ps={metrics.get('mean_censored_latency_ps', 0.0):.1f} "
            f"decision_s={metrics.get('mean_decision_seconds', 0.0):.4f}"
        )
    print(f"json: {versioned}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
