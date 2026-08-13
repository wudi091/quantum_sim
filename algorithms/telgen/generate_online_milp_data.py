"""Collect exact online MILP rollout graphs for GNN imitation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version as package_version
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import networkx as nx
import numpy as np
import scipy

from qnet_core.scenario import ScenarioConfig, make_episode
from qnet_core.spec import PhysicalConfig

from .online import (
    OnlineTELGENConfig,
    generate_online_milp_dataset,
    save_online_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect online exact-MILP graph/label episodes."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=12000)
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
    parser.add_argument("--time-limit-seconds", type=float, default=300.0)
    parser.add_argument("--generation-probability", type=float, default=0.8)
    parser.add_argument("--swap-probability", type=float, default=0.9)
    parser.add_argument("--memory-capacity", type=int, default=2)
    parser.add_argument("--node-memory-capacity", type=int)
    parser.add_argument("--quantum-distance-m", type=float, default=1000.0)
    parser.add_argument("--slot-duration-ps", type=int, default=50_000_000)
    return parser


def _write_collection_manifest(
    output: Path,
    payload: dict[str, object],
) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned = output / f"online_milp_dataset_{timestamp}.json"
    latest = output / "online_milp_dataset.json"
    versioned.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    shutil.copyfile(versioned, latest)
    return versioned, latest


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if args.requests_per_batch < 1 or args.decision_interval < 1:
        raise ValueError("batch size and decision interval must be positive")
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
    config = OnlineTELGENConfig(
        decision_interval=args.decision_interval,
        path_candidate_count=args.paths,
        construction_kinds=(),
        swap_tree_count=args.construction_plans,
        purification_kinds=("none",),
        decision_backend="milp_teacher",
        teacher_solver_backend="highs_ipm",
        milp_time_limit_seconds=args.time_limit_seconds,
        milp_relative_gap=0.0,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    total_samples = 0
    started = perf_counter()
    for index in range(args.episodes):
        seed = args.seed_start + index
        episode_started = perf_counter()
        episode = make_episode(scenario, seed)
        episode_directory = args.output / f"episode_{seed:08d}"
        result, dataset_paths = generate_online_milp_dataset(
            episode,
            episode_directory / "dataset",
            config,
        )
        result_paths = save_online_result(result, episode_directory / "rollout")
        relative_manifest = dataset_paths.manifest_path.relative_to(args.output)
        entry = {
            "seed": seed,
            "manifest": relative_manifest.as_posix(),
            "sample_count": len(dataset_paths.sample_paths),
            "skipped_boundary_count": len(result.skipped_milp_boundaries),
            "completed_requests": result.metrics["completed_requests"],
            "request_count": result.metrics["request_count"],
            "mean_decision_seconds": result.metrics["mean_decision_seconds"],
            "elapsed_seconds": perf_counter() - episode_started,
            "rollout_json": result_paths.json_path.relative_to(
                args.output
            ).as_posix(),
        }
        entries.append(entry)
        total_samples += len(dataset_paths.sample_paths)
        print(
            f"episode={index + 1}/{args.episodes} seed={seed} "
            f"samples={entry['sample_count']} "
            f"completed={int(entry['completed_requests'])}/"
            f"{int(entry['request_count'])} "
            f"seconds={entry['elapsed_seconds']:.3f}",
            flush=True,
        )
    try:
        sequence_version = package_version("sequence")
    except PackageNotFoundError:
        sequence_version = "unknown"
    payload = {
        "schema_version": 1,
        "dataset_kind": "online_milp_teacher_collection",
        "configuration": {
            **vars(args),
            "output": str(args.output),
            "resolved_horizon": horizon,
        },
        "scenario": asdict(scenario),
        "online_config": asdict(config),
        "runtime_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "networkx": nx.__version__,
            "sequence": sequence_version,
            "milp_solver": "scipy.optimize.milp/HiGHS",
        },
        "episode_count": len(entries),
        "sample_count": total_samples,
        "elapsed_seconds": perf_counter() - started,
        "episodes": entries,
    }
    versioned, latest = _write_collection_manifest(args.output, payload)
    print(f"samples={total_samples}")
    print(f"manifest: {versioned}")
    print(f"latest: {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
