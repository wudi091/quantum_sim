"""Command-line entry point for generating LP teacher trajectories."""

from __future__ import annotations

import argparse

from qnet_core.scenario import ScenarioConfig

from .dataset import generate_teacher_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate simultaneous-request TELGEN LP teacher records."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--min-hops", type=int, default=2)
    parser.add_argument("--max-hops", type=int, default=5)
    parser.add_argument("--ttl", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--paths", type=int, default=3)
    parser.add_argument(
        "--topology-mode",
        choices=("waxman", "parallel_corridors"),
        default="waxman",
    )
    parser.add_argument("--parallel-corridors", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples < 1:
        raise ValueError("samples must be positive")
    scenario = ScenarioConfig(
        request_count=args.requests,
        min_hops=args.min_hops,
        max_hops=args.max_hops,
        ttl=args.ttl,
        horizon=args.horizon,
        topology_nodes=args.nodes,
        topology_mode=args.topology_mode,
        parallel_corridors=args.parallel_corridors,
    )
    result = generate_teacher_dataset(
        scenario,
        range(args.seed_start, args.seed_start + args.samples),
        args.output,
        path_candidate_count=args.paths,
        overwrite=args.overwrite,
    )
    print(f"generated {len(result.entries)} records")
    print(f"manifest: {result.manifest_path}")
    for entry in result.entries:
        print(
            f"seed={entry.seed} requests={entry.request_count} "
            f"variables={entry.variable_count} "
            f"completed_mass={entry.completed_request_mass:.6f} "
            f"file={entry.file}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
