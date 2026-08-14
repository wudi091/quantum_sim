"""Collect exact online MILP rollout graphs for GNN imitation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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
from qnet_core.spec import EpisodeSpec, PhysicalConfig

from .online import (
    OnlineTELGENConfig,
    generate_online_milp_dataset,
    save_online_result,
)
from .milp_oracle import DiscreteOracleSolveError


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
    parser.add_argument("--time-limit-seconds", type=float, default=300.0)
    parser.add_argument("--time-limit-retries", type=int, default=0)
    parser.add_argument("--time-limit-multiplier", type=float, default=3.0)
    parser.add_argument("--generation-probability", type=float, default=0.8)
    parser.add_argument("--swap-probability", type=float, default=0.9)
    parser.add_argument("--memory-capacity", type=int, default=2)
    parser.add_argument("--node-memory-capacity", type=int)
    parser.add_argument("--quantum-distance-m", type=float, default=1000.0)
    parser.add_argument("--slot-duration-ps", type=int, default=50_000_000)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse validated completed episode directories",
    )
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


def _write_progress_manifest(output: Path, payload: dict[str, object]) -> Path:
    """Atomically checkpoint collection progress after every episode."""

    output.mkdir(parents=True, exist_ok=True)
    latest = output / "online_milp_dataset.json"
    temporary = output / ".online_milp_dataset.json.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(latest)
    return latest


def _normalized_json(value: object) -> object:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _teacher_config_payload(config: OnlineTELGENConfig) -> dict[str, object]:
    payload = _normalized_json(asdict(config))
    keys = (
        "decision_interval",
        "path_candidate_count",
        "construction_kinds",
        "swap_tree_count",
        "purification_kinds",
        "decision_backend",
        "milp_relative_gap",
    )
    return {key: payload[key] for key in keys}


def _time_limit_schedule(
    initial_seconds: float,
    retries: int,
    multiplier: float,
) -> tuple[float, ...]:
    """Return deterministic per-episode exact-MILP retry budgets."""

    if initial_seconds <= 0.0:
        raise ValueError("time limit must be positive")
    if retries < 0:
        raise ValueError("time limit retries cannot be negative")
    if retries and multiplier <= 1.0:
        raise ValueError("time limit multiplier must exceed one")
    return tuple(
        float(initial_seconds) * float(multiplier) ** attempt
        for attempt in range(retries + 1)
    )


def _is_time_limit_failure(error: BaseException) -> bool:
    return isinstance(error, DiscreteOracleSolveError) and (
        "time limit reached" in str(error).lower()
    )


def _load_completed_episode_entry(
    output: Path,
    seed: int,
    config: OnlineTELGENConfig,
    expected_episode: EpisodeSpec,
) -> dict[str, object] | None:
    """Load one completed episode only when its saved provenance is valid."""

    episode_directory = output / f"episode_{seed:08d}"
    dataset_pointer = episode_directory / "dataset" / "manifest.json"
    rollout_pointer = (
        episode_directory / "rollout" / "online_telgen_results.json"
    )
    if not dataset_pointer.is_file() or not rollout_pointer.is_file():
        return None
    dataset_payload = json.loads(dataset_pointer.read_text(encoding="utf-8"))
    rollout_payload = json.loads(rollout_pointer.read_text(encoding="utf-8"))
    if dataset_payload.get("dataset_kind") != "online_milp_teacher_rollout":
        raise RuntimeError(
            f"episode {seed} has an incompatible dataset manifest"
        )
    if int(dataset_payload.get("episode_seed", -1)) != seed:
        raise RuntimeError(f"episode {seed} dataset seed does not match")
    if int(rollout_payload.get("episode_seed", -1)) != seed:
        raise RuntimeError(f"episode {seed} rollout seed does not match")
    saved_config = dataset_payload.get("config")
    if not isinstance(saved_config, dict) or any(
        saved_config.get(key) != value
        for key, value in _teacher_config_payload(config).items()
    ):
        raise RuntimeError(f"episode {seed} configuration does not match")
    expected_environment = {
        "seed": expected_episode.seed,
        "nodes": list(expected_episode.nodes),
        "edges": [list(edge) for edge in expected_episode.edges],
        "horizon": expected_episode.horizon,
        "physical": _normalized_json(asdict(expected_episode.physical)),
    }
    if dataset_payload.get("planning_environment") != expected_environment:
        raise RuntimeError(f"episode {seed} planning environment does not match")
    if rollout_payload.get("episode") != _normalized_json(
        asdict(expected_episode)
    ):
        raise RuntimeError(f"episode {seed} rollout environment does not match")
    version_directory = dataset_payload.get("version_directory")
    resolved_manifest = (
        dataset_pointer
        if version_directory is None
        else dataset_pointer.parent / str(version_directory) / "manifest.json"
    )
    resolved_payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    samples = resolved_payload.get("samples", [])
    if int(resolved_payload.get("sample_count", -1)) != len(samples):
        raise RuntimeError(f"episode {seed} sample count is inconsistent")
    for sample in samples:
        sample_path = resolved_manifest.parent / str(sample["file"])
        if not sample_path.is_file():
            raise RuntimeError(
                f"episode {seed} is missing graph sample {sample_path.name}"
            )
    metrics = rollout_payload.get("metrics", {})
    entry_path = episode_directory / "episode_entry.json"
    elapsed_seconds = None
    solver_attempts = 1
    solved_time_limit = float(saved_config["milp_time_limit_seconds"])
    if entry_path.is_file():
        saved_entry = json.loads(entry_path.read_text(encoding="utf-8"))
        if int(saved_entry.get("seed", -1)) == seed:
            elapsed_seconds = saved_entry.get("elapsed_seconds")
            solver_attempts = int(saved_entry.get("solver_attempts", 1))
            solved_time_limit = float(saved_entry.get(
                "milp_time_limit_seconds",
                solved_time_limit,
            ))
    return {
        "seed": seed,
        "manifest": dataset_pointer.relative_to(output).as_posix(),
        "sample_count": len(samples),
        "skipped_boundary_count": len(
            resolved_payload.get("skipped_boundaries", [])
        ),
        "completed_requests": float(metrics["completed_requests"]),
        "request_count": float(metrics["request_count"]),
        "mean_decision_seconds": float(metrics["mean_decision_seconds"]),
        "elapsed_seconds": elapsed_seconds,
        "solver_attempts": solver_attempts,
        "milp_time_limit_seconds": solved_time_limit,
        "rollout_json": rollout_pointer.relative_to(output).as_posix(),
        "resumed": True,
    }


def _write_episode_entry(
    episode_directory: Path,
    entry: dict[str, object],
) -> None:
    temporary = episode_directory / ".episode_entry.json.tmp"
    temporary.write_text(
        json.dumps(entry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(episode_directory / "episode_entry.json")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if args.requests_per_batch < 1 or args.decision_interval < 1:
        raise ValueError("batch size and decision interval must be positive")
    time_limits = _time_limit_schedule(
        args.time_limit_seconds,
        args.time_limit_retries,
        args.time_limit_multiplier,
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
    config = OnlineTELGENConfig(
        decision_interval=args.decision_interval,
        path_candidate_count=args.paths,
        construction_kinds=(),
        swap_tree_count=args.construction_plans,
        purification_kinds=("none",),
        decision_backend="milp_teacher",
        milp_time_limit_seconds=args.time_limit_seconds,
        milp_relative_gap=0.0,
    )
    try:
        sequence_version = package_version("sequence")
    except PackageNotFoundError:
        sequence_version = "unknown"
    args.output.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    total_samples = 0
    started = perf_counter()

    def collection_payload(*, complete: bool) -> dict[str, object]:
        return {
            "schema_version": 1,
            "dataset_kind": "online_milp_teacher_collection",
            "collection_complete": complete,
            "target_episode_count": args.episodes,
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

    for index in range(args.episodes):
        seed = args.seed_start + index
        episode = make_episode(scenario, seed)
        resumed_entry = (
            _load_completed_episode_entry(
                args.output,
                seed,
                config,
                episode,
            )
            if args.resume
            else None
        )
        if resumed_entry is not None:
            entries.append(resumed_entry)
            total_samples += int(resumed_entry["sample_count"])
            elapsed = resumed_entry["elapsed_seconds"]
            elapsed_text = "unknown" if elapsed is None else f"{float(elapsed):.3f}"
            print(
                f"episode={index + 1}/{args.episodes} seed={seed} "
                f"samples={resumed_entry['sample_count']} "
                f"completed={int(resumed_entry['completed_requests'])}/"
                f"{int(resumed_entry['request_count'])} "
                f"seconds={elapsed_text} resumed=1",
                flush=True,
            )
            _write_progress_manifest(
                args.output,
                collection_payload(complete=False),
            )
            continue
        episode_started = perf_counter()
        episode_directory = args.output / f"episode_{seed:08d}"
        result = None
        dataset_paths = None
        episode_config = config
        solver_attempts = 0
        for attempt_index, time_limit_seconds in enumerate(
            time_limits,
            start=1,
        ):
            solver_attempts = attempt_index
            episode_config = replace(
                config,
                milp_time_limit_seconds=time_limit_seconds,
            )
            try:
                result, dataset_paths = generate_online_milp_dataset(
                    episode,
                    episode_directory / "dataset",
                    episode_config,
                )
                break
            except DiscreteOracleSolveError as error:
                if (
                    not _is_time_limit_failure(error)
                    or attempt_index == len(time_limits)
                ):
                    raise
                print(
                    f"episode={index + 1}/{args.episodes} seed={seed} "
                    f"timeout={time_limit_seconds:.3f}s "
                    f"retry={attempt_index}/{len(time_limits) - 1} "
                    f"next_limit={time_limits[attempt_index]:.3f}s",
                    flush=True,
                )
        if result is None or dataset_paths is None:
            raise RuntimeError("MILP retry loop ended without a result")
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
            "solver_attempts": solver_attempts,
            "milp_time_limit_seconds": (
                episode_config.milp_time_limit_seconds
            ),
            "rollout_json": result_paths.json_path.relative_to(
                args.output
            ).as_posix(),
            "resumed": False,
        }
        _write_episode_entry(episode_directory, entry)
        entries.append(entry)
        total_samples += len(dataset_paths.sample_paths)
        _write_progress_manifest(
            args.output,
            collection_payload(complete=False),
        )
        print(
            f"episode={index + 1}/{args.episodes} seed={seed} "
            f"samples={entry['sample_count']} "
            f"completed={int(entry['completed_requests'])}/"
            f"{int(entry['request_count'])} "
            f"seconds={entry['elapsed_seconds']:.3f}",
            flush=True,
        )
    payload = collection_payload(complete=True)
    versioned, latest = _write_collection_manifest(args.output, payload)
    print(f"samples={total_samples}")
    print(f"manifest: {versioned}")
    print(f"latest: {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
