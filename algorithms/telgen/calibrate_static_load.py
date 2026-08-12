"""Command-line entry point for static batch load calibration."""

from __future__ import annotations

import argparse

from qnet_core.scenario import ScenarioConfig

from .calibration import StaticLoadProfile, generate_static_load_calibration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate low/medium/high static LP-teacher workloads."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--min-hops", type=int, default=2)
    parser.add_argument("--max-hops", type=int, default=5)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--paths", type=int, default=1)
    parser.add_argument("--light-requests", type=int, default=8)
    parser.add_argument("--light-horizon", type=int, default=12)
    parser.add_argument("--medium-requests", type=int, default=24)
    parser.add_argument("--medium-horizon", type=int, default=6)
    parser.add_argument("--heavy-requests", type=int, default=40)
    parser.add_argument("--heavy-horizon", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples < 1:
        raise ValueError("samples must be positive")
    profiles = (
        StaticLoadProfile("light", args.light_requests, args.light_horizon),
        StaticLoadProfile("medium", args.medium_requests, args.medium_horizon),
        StaticLoadProfile("heavy", args.heavy_requests, args.heavy_horizon),
    )
    scenario = ScenarioConfig(
        request_count=max(profile.request_count for profile in profiles),
        min_hops=args.min_hops,
        max_hops=args.max_hops,
        ttl=max(profile.horizon for profile in profiles),
        horizon=max(profile.horizon for profile in profiles),
        topology_nodes=args.nodes,
    )
    result = generate_static_load_calibration(
        scenario,
        range(args.seed_start, args.seed_start + args.samples),
        args.output,
        profiles=profiles,
        path_candidate_count=args.paths,
        overwrite=args.overwrite,
    )
    print(f"generated {len(result.entries)} calibrated records")
    print(f"manifest: {result.manifest_path}")
    print(f"statistics: {result.csv_path}")
    for aggregate in result.aggregates:
        print(
            f"{aggregate.load_profile}: "
            f"completion={aggregate.mean_completion_ratio:.4f} "
            f"fractional_requests={aggregate.mean_fractional_request_ratio:.4f} "
            f"peak_utilization={aggregate.mean_peak_resource_utilization:.4f} "
            f"solve_seconds={aggregate.mean_solve_seconds:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
