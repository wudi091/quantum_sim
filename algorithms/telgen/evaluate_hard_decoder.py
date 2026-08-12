"""Command-line hard-decoder versus MILP evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from qnet_core.scenario import ScenarioConfig

from .dataset import build_teacher_batch_record
from .hard_decoder import (
    HardConstraintDecoder,
    compare_decoder_and_milp,
    save_decoder_gap_report,
)
from .milp_oracle import ConstructionAwareMILPOracle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode LP scores into a feasible plan and compare to MILP."
    )
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--min-hops", type=int, default=2)
    parser.add_argument("--max-hops", type=int, default=5)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--paths", type=int, default=1)
    parser.add_argument("--beam-width", type=int, default=512)
    parser.add_argument("--random-restarts", type=int, default=512)
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
    decoded = HardConstraintDecoder(
        beam_width=args.beam_width,
        random_restarts=args.random_restarts,
    ).decode(
        record.expansion,
        record.capacities,
        record.solution.final_values,
        request_ids=tuple(
            request.id for request in record.episode.requests
        ),
    )
    discrete = ConstructionAwareMILPOracle(
        time_limit_seconds=args.time_limit,
    ).solve(record.expansion, record.capacities)
    report = compare_decoder_and_milp(decoded, discrete)
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    if args.output:
        save_decoder_gap_report(
            report,
            args.output,
            context={
                "seed": args.seed,
                "request_count": args.requests,
                "horizon": args.horizon,
                "min_hops": args.min_hops,
                "max_hops": args.max_hops,
                "path_candidate_count": args.paths,
                "beam_width": args.beam_width,
                "random_restarts": args.random_restarts,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
