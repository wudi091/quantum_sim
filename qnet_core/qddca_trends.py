"""Reproduce Q-DDCA's official qualitative trends on SeQUeNCE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .env import SharedRoutingEnv
from .planners import QDDCAPlanner
from .spec import EpisodeSpec, PhysicalConfig, RequestSpec


OFFICIAL = {
    "retry": {
        "x": [1, 5, 10],
        "throughput": [169, 1452, 2830],
        "drop": [533, 1, 0],
        "window": 20,
        "reroute": True,
    },
    "window": {
        "x": [1, 5, 10, 20, 30],
        "throughput": [487, 1576, 2179, 2032, 1635],
        "drop": [0, 166, 360, 2560, 4995],
        "cv": [0.005, 0.4254, 0.2943, 0.5425, 0.5299],
        "max_try": 5,
        "reroute": True,
    },
    "reroute": {
        "window": 10,
        "max_try": 5,
        "false": {"throughput": 1438, "drop": 1183, "cv": 0.6995},
        "true": {"throughput": 2179, "drop": 360, "cv": 0.2943},
    },
    "source": "Unmodified QDDCA exp1.py/exp2.py, seeds 120/2, qns 0.2.3",
}


FLOW_ENDPOINTS = ((0, 15), (3, 12), (4, 11), (7, 8), (1, 14))


def make_spec(window: int, seed: int, ttl: int = 30) -> EpisodeSpec:
    side = 4
    nodes = tuple(range(side * side))
    edges: list[tuple[int, int]] = []
    for row in range(side):
        for col in range(side):
            node = row * side + col
            if col + 1 < side:
                edges.append((node, node + 1))
            if row + 1 < side:
                edges.append((node, node + side))
    requests = tuple(
        RequestSpec(f"f{flow}-p{packet}", source, destination, ttl=ttl)
        for flow, (source, destination) in enumerate(FLOW_ENDPOINTS)
        for packet in range(window)
    )
    return EpisodeSpec(
        seed=seed,
        nodes=nodes,
        edges=tuple(edges),
        requests=requests,
        horizon=ttl,
        physical=PhysicalConfig(
            generation_probability=0.55,
            swap_probability=0.85,
            memory_capacity=4,
            memory_lifetime=22,
        ),
    )


def run_case(window: int, max_try: int, reroute: bool, seed: int) -> dict[str, float]:
    env = SharedRoutingEnv(make_spec(window, seed), candidate_count=3)
    planner = QDDCAPlanner(max_try=max_try, allow_reroute=reroute, seed=seed)
    planner.reset(seed)
    while not env.done:
        snapshot = env.snapshot()
        env.commit(planner.select(snapshot))
    per_flow = []
    for flow in range(len(FLOW_ENDPOINTS)):
        per_flow.append(sum(
            state.completed_at is not None
            for request_id, state in env.requests.items()
            if request_id.startswith(f"f{flow}-")
        ))
    throughput = sum(per_flow) / max(env.time, 1)
    drops = sum(state.expired_at is not None for state in env.requests.values())
    mean = float(np.mean(per_flow))
    cv = float(np.std(per_flow) / mean) if mean else float("inf")
    return {
        "throughput": float(throughput),
        "drop": float(drops),
        "cv": cv,
        "completed": float(sum(per_flow)),
        "time": float(env.time),
    }


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _corr(left: list[float], right: list[float]) -> float:
    value = spearmanr(left, right).statistic
    return 0.0 if np.isnan(value) else float(value)


def run_suite(seeds: int = 3) -> dict[str, object]:
    retry = []
    for max_try in OFFICIAL["retry"]["x"]:
        retry.append(_mean([
            run_case(OFFICIAL["retry"]["window"], max_try, True, seed)
            for seed in range(seeds)
        ]))
    window = []
    for size in OFFICIAL["window"]["x"]:
        window.append(_mean([
            run_case(size, OFFICIAL["window"]["max_try"], True, seed)
            for seed in range(seeds)
        ]))
    reroute = {
        str(value).lower(): _mean([
            run_case(
                OFFICIAL["reroute"]["window"],
                OFFICIAL["reroute"]["max_try"],
                value,
                seed,
            ) for seed in range(seeds)
        ])
        for value in (False, True)
    }
    validation = {
        "retry_throughput_spearman": _corr(
            OFFICIAL["retry"]["throughput"], [row["throughput"] for row in retry]
        ),
        "retry_drop_spearman": _corr(
            OFFICIAL["retry"]["drop"], [row["drop"] for row in retry]
        ),
        "window_throughput_spearman": _corr(
            OFFICIAL["window"]["throughput"], [row["throughput"] for row in window]
        ),
        "window_drop_spearman": _corr(
            OFFICIAL["window"]["drop"], [row["drop"] for row in window]
        ),
        "window_cv_spearman": _corr(
            OFFICIAL["window"]["cv"], [row["cv"] for row in window]
        ),
        "reroute_throughput_improves": reroute["true"]["throughput"] >= reroute["false"]["throughput"],
        "reroute_drop_improves": reroute["true"]["drop"] <= reroute["false"]["drop"],
        "reroute_cv_improves": reroute["true"]["cv"] <= reroute["false"]["cv"],
    }
    window_throughput = [row["throughput"] for row in window]
    validation.update({
        "window_rises_before_congestion": window_throughput[2] > window_throughput[0],
        "window_peaks_at_moderate_load": int(np.argmax(window_throughput)) in (1, 2),
        "window_declines_under_high_load": window_throughput[-1] < max(window_throughput),
    })
    validation["overall_pass"] = bool(
        validation["retry_throughput_spearman"] >= 0.8
        and validation["retry_drop_spearman"] >= 0.8
        and validation["window_drop_spearman"] >= 0.8
        and validation["window_cv_spearman"] >= 0.8
        and validation["window_rises_before_congestion"]
        and validation["window_peaks_at_moderate_load"]
        and validation["window_declines_under_high_load"]
        and validation["reroute_throughput_improves"]
        and validation["reroute_drop_improves"]
        and validation["reroute_cv_improves"]
    )
    return {
        "official": OFFICIAL,
        "sequence": {"retry": retry, "window": window, "reroute": reroute},
        "validation": validation,
        "seeds": seeds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Q-DDCA trend reproduction on SeQUeNCE")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/qddca_sequence_trends.json"))
    args = parser.parse_args()
    result = run_suite(args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    args.output.write_text(payload, encoding="utf-8")
    print(json.dumps(result["validation"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
