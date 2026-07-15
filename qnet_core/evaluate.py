"""Fair planner comparison on identical episode specifications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .env import SharedRoutingEnv
from .planners import GreedyPlanner, QDDCAPlanner, RandomPlanner
from .scenario import ScenarioConfig, make_episode
from .spec import PhysicalConfig


def run_planner(planner: object, scenario: ScenarioConfig, seed: int) -> dict[str, float]:
    spec = make_episode(scenario, seed)
    env = SharedRoutingEnv(spec)
    planner.reset(seed)
    while not env.done:
        snapshot = env.snapshot()
        selected = tuple(planner.select(snapshot))
        env.commit(selected)
    return env.metrics()


def compare(scenario: ScenarioConfig, seeds: int) -> dict[str, object]:
    planners = {
        "greedy": GreedyPlanner(),
        "qddca": QDDCAPlanner(),
        "random": RandomPlanner(0),
    }
    rows = {
        name: [run_planner(planner, scenario, seed) for seed in range(seeds)]
        for name, planner in planners.items()
    }
    aggregate: dict[str, dict[str, float]] = {}
    for name, values in rows.items():
        keys = values[0].keys()
        aggregate[name] = {
            key: sum(row[key] for row in values) / len(values) for key in keys
        }
    return {"scenario": scenario.__dict__, "rows": rows, "aggregate": aggregate}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare planners on one SeQUeNCe setup")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--min-hops", type=int, default=20)
    parser.add_argument("--max-hops", type=int, default=50)
    parser.add_argument("--ttl", type=int, default=32)
    parser.add_argument("--p-gen", type=float, default=0.5)
    parser.add_argument("--p-swap", type=float, default=0.5)
    parser.add_argument("--memory", type=int, default=2)
    parser.add_argument("--arrival-rate", type=float, default=1.0,
                        help="Mean Poisson request arrivals per physical step")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    scenario = ScenarioConfig(
        request_count=args.requests,
        min_hops=args.min_hops,
        max_hops=args.max_hops,
        ttl=args.ttl,
        horizon=args.ttl,
        arrival_rate=args.arrival_rate,
        physical=PhysicalConfig(
            generation_probability=args.p_gen,
            swap_probability=args.p_swap,
            memory_capacity=args.memory,
        ),
    )
    result = compare(scenario, args.seeds)
    payload = json.dumps(result, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
