"""Fair planner comparison on identical episode specifications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from .planners import (
    QCASTPlanner,
    QDDCAPlanner,
)
from .runtime import make_sequence_env
from .scenario import ScenarioConfig, make_episode
from .spec import PhysicalConfig


def run_planner(planner: object, scenario: ScenarioConfig, seed: int) -> dict[str, float]:
    spec = make_episode(scenario, seed)
    env = make_sequence_env(spec)
    planner.reset(seed)
    planning_seconds = 0.0
    planner_calls = 0
    while not env.done:
        snapshot = env.snapshot()
        started = perf_counter()
        selected = tuple(planner.select(snapshot))
        planning_seconds += perf_counter() - started
        planner_calls += 1
        env.commit(selected)
    metrics = env.metrics()
    metrics.update({
        "planner_calls": float(planner_calls),
        "planning_seconds": planning_seconds,
        "mean_planning_ms": 1000.0 * planning_seconds / max(planner_calls, 1),
    })
    return metrics


def compare(
    scenario: ScenarioConfig,
    seeds: int,
    planner_names: tuple[str, ...] | None = None,
) -> dict[str, object]:
    available = {
        "qddca": QDDCAPlanner(),
        "qcast": QCASTPlanner(),
    }
    names = tuple(available) if planner_names is None else planner_names
    unknown = set(names) - set(available)
    if unknown:
        raise ValueError(f"unknown planners: {sorted(unknown)}")
    planners = {name: available[name] for name in names}
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
    parser.add_argument(
        "--planners", nargs="+",
        choices=("qddca", "qcast"),
        default=("qddca", "qcast"),
    )
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
    result = compare(scenario, args.seeds, tuple(args.planners))
    payload = json.dumps(result, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
