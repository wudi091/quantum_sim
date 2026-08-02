"""Run the formal CON benchmark over disjoint episode shards in parallel."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

from qnet_core.order_waxman_benchmark import (
    DEFAULT_PHYSICS_SEED_BASE,
    DEFAULT_PLANNER_SEED_BASE,
)

from .generator_oracle import (
    DEFAULT_GENERATORS,
    merge_generator_oracle_benchmark_results,
    run_generator_oracle_benchmark,
)
from .run_formal import formal_config


def _contiguous_shards(
    episode_seeds: tuple[int, ...], workers: int
) -> tuple[tuple[int, ...], ...]:
    workers = min(int(workers), len(episode_seeds))
    width, remainder = divmod(len(episode_seeds), workers)
    shards = []
    start = 0
    for index in range(workers):
        size = width + int(index < remainder)
        shards.append(episode_seeds[start : start + size])
        start += size
    return tuple(shards)


def _run_shard(payload: dict[str, object]) -> dict[str, object]:
    shard = tuple(payload["episode_seeds"])
    output_path = Path(payload["output_path"])
    result = run_generator_oracle_benchmark(
        config=formal_config(),
        episode_seeds=shard,
        generator_names=tuple(payload["generator_names"]),
        baseline_names=tuple(payload["baseline_names"]),
        path_pool_per_pair=int(payload["path_pool_per_pair"]),
        max_hops=6,
        planning_seeds=(0,),
        online_selector="reliable_memory_milp",
        reliability_confidence=float(payload["reliability_confidence"]),
        oracle_workers=1,
        physics_seed_base=DEFAULT_PHYSICS_SEED_BASE,
        planner_seed_base=DEFAULT_PLANNER_SEED_BASE,
        output_path=output_path,
    )
    return {
        "episode_seeds": list(shard),
        "output_path": str(output_path),
        "result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel formal reliable-memory CON benchmark"
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "episode shards evaluated concurrently; keep 1 for the "
            "memory-safe default"
        ),
    )
    parser.add_argument(
        "--reliability-confidence", type=float, default=0.9
    )
    parser.add_argument("--path-pool-per-pair", type=int, default=8)
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not 0.0 < args.reliability_confidence <= 1.0:
        parser.error("--reliability-confidence must lie in (0, 1]")

    episode_seeds = tuple(range(
        args.base_seed, args.base_seed + args.episodes
    ))
    shards = _contiguous_shards(episode_seeds, args.workers)
    partial_dir = args.output.parent / "partials" / args.output.stem
    partial_dir.mkdir(parents=True, exist_ok=True)
    payloads = tuple(
        {
            "episode_seeds": shard,
            "output_path": str(
                partial_dir
                / f"part{index}_seed{shard[0]}-{shard[-1]}.json"
            ),
            "generator_names": tuple(args.generators),
            "baseline_names": tuple(args.baselines),
            "path_pool_per_pair": args.path_pool_per_pair,
            "reliability_confidence": args.reliability_confidence,
        }
        for index, shard in enumerate(shards)
    )
    print(json.dumps({
        "event": "formal_parallel_start",
        "workers": len(shards),
        "shards": [list(shard) for shard in shards],
    }), flush=True)

    completed = []
    with ProcessPoolExecutor(max_workers=len(shards)) as executor:
        future_to_shard = {
            executor.submit(_run_shard, payload): tuple(
                payload["episode_seeds"]
            )
            for payload in payloads
        }
        for future in as_completed(future_to_shard):
            shard = future_to_shard[future]
            value = future.result()
            completed.append(value["result"])
            print(json.dumps({
                "event": "formal_parallel_shard_complete",
                "episode_seeds": list(shard),
                "output_path": value["output_path"],
            }), flush=True)

    result = merge_generator_oracle_benchmark_results(
        completed, output_path=args.output
    )
    print(json.dumps({
        "event": "formal_parallel_complete",
        "episode_count": len(result["episode_seeds"]),
        "ranking": result["ranking"],
        "recommended_generator": result["recommended_generator"],
        "output": str(args.output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
