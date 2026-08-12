"""Command-line LP-versus-MILP validation for one small static batch."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from qnet_core.scenario import ScenarioConfig

from .dataset import build_teacher_batch_record
from .milp_oracle import (
    ConstructionAwareMILPOracle,
    compare_lp_and_milp,
    save_gap_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the continuous LP teacher with an exact binary MILP."
    )
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--min-hops", type=int, default=2)
    parser.add_argument("--max-hops", type=int, default=5)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--paths", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenario = ScenarioConfig(
        request_count=args.requests,
        min_hops=args.min_hops,
        max_hops=args.max_hops,
        ttl=args.horizon,
        horizon=args.horizon,
        topology_nodes=args.nodes,
    )
    record = build_teacher_batch_record(
        scenario,
        args.seed,
        path_candidate_count=args.paths,
    )
    discrete = ConstructionAwareMILPOracle(
        time_limit_seconds=args.time_limit,
    ).solve(record.expansion, record.capacities)
    report = compare_lp_and_milp(record.solution, discrete)
    payload = asdict(report)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        save_gap_report(
            report,
            args.output,
            context={
                "seed": args.seed,
                "request_count": args.requests,
                "horizon": args.horizon,
                "min_hops": args.min_hops,
                "max_hops": args.max_hops,
                "path_candidate_count": args.paths,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
