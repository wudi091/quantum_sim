"""Run the Q-DDCA paper parameter sweeps on the SeQUeNCe environment.

This is a protocol-level reproduction of ``QDDCA/exp1.py`` and
``QDDCA/exp2.py``.  The original SimQN source uses 50 nodes, an Erdos-Renyi
link probability of 0.1, a 10-second run, 50-ms allocation requests, and a
window of concurrent packet streams.  The physical events are delegated to
SeQUeNCe; the output therefore reports trend agreement rather than claiming
numerical identity with the SimQN CSV files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

from algorithms.qddca import QDDCAPlanner
from qnet_core.runtime import make_sequence_env
from qnet_core.spec import EpisodeSpec, PhysicalConfig, RequestSpec


PAPER = {
    "nodes": 50,
    "link_probability": 0.1,
    "duration_s": 10.0,
    "allocation_delay_s": 0.05,
    "initial_fidelity": 0.99,
    "required_fidelity": 0.7,
    "coherence_s": 5.0,
    "rate_hz": 1000.0,
    "delay_s": 0.001,
    "exp1_windows": (10, 20, 30),
    "exp1_max_try": tuple(range(1, 11)),
    "exp2_windows": tuple(range(1, 31)),
    "exp2_max_try": (1, 5, 10),
}

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


def _connected_graph(nodes: int, probability: float, seed: int) -> nx.Graph:
    rng = random.Random(seed)
    for _ in range(256):
        graph = nx.gnp_random_graph(nodes, probability, seed=rng.randrange(2**31))
        if nx.is_connected(graph):
            return graph
    graph = nx.gnp_random_graph(nodes, probability, seed=seed)
    components = [sorted(component) for component in nx.connected_components(graph)]
    for left, right in zip(components, components[1:]):
        graph.add_edge(left[0], right[0])
    return graph


def make_spec(
    *,
    window: int,
    request_count: int,
    memory_size: int,
    max_try: int,
    seed: int,
    horizon_slots: int = 200,
) -> EpisodeSpec:
    graph = _connected_graph(PAPER["nodes"], PAPER["link_probability"], seed)
    distances = dict(nx.all_pairs_shortest_path_length(graph))
    rng = random.Random(seed ^ 0x51444443)
    endpoints: list[tuple[int, int]] = []
    for _ in range(request_count):
        source, destination = rng.sample(range(PAPER["nodes"]), 2)
        if distances[source][destination] > 8:
            destination = min(
                (node for node in distances[source] if node != source),
                key=lambda node: (abs(distances[source][node] - 6), node),
            )
        endpoints.append((source, destination))

    requests: list[RequestSpec] = []
    for flow, (source, destination) in enumerate(endpoints):
        shortest = max(1, distances[source][destination])
        concurrent = max(1, int(window) * shortest)
        for packet in range(concurrent):
            requests.append(RequestSpec(
                f"f{flow}-p{packet}",
                source,
                destination,
                ttl=horizon_slots,
                demand_pairs=10**9,
                required_fidelity=PAPER["required_fidelity"],
                max_storage_slots=10,
            ))

    slot_duration_ps = int(PAPER["allocation_delay_s"] * 1e12)
    coherence_slots = max(1, round(PAPER["coherence_s"] / PAPER["allocation_delay_s"]))
    return EpisodeSpec(
        seed=seed,
        nodes=tuple(sorted(int(node) for node in graph.nodes)),
        edges=tuple(sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges)),
        requests=tuple(requests),
        horizon=horizon_slots,
        physical=PhysicalConfig(
            generation_probability=1.0,
            swap_probability=1.0,
            memory_capacity=max(1, memory_size),
            node_memory_capacity=max(1, memory_size),
            memory_lifetime=coherence_slots,
            initial_fidelity=PAPER["initial_fidelity"],
            swap_degradation=1.0,
            slot_duration_ps=slot_duration_ps,
            detector_efficiency=1.0,
            bsm_success_probability=1.0,
        ),
    )


def run_sequence_case(
    *,
    window: int,
    max_try: int,
    request_count: int,
    memory_size: int,
    reroute: bool,
    seed: int,
    horizon_slots: int,
) -> dict[str, float]:
    spec = make_spec(
        window=window,
        request_count=request_count,
        memory_size=memory_size,
        max_try=max_try,
        seed=seed,
        horizon_slots=horizon_slots,
    )
    env = make_sequence_env(
        spec,
        candidate_count=128,
        request_driven_generation=True,
        local_candidates=True,
        best_effort_allocations=True,
    )
    planner = QDDCAPlanner(max_try=max_try, allow_reroute=reroute, seed=seed)
    planner.reset(seed)
    while not env.done:
        snapshot = env.snapshot()
        env.commit(planner.select(snapshot))

    flow_totals = [
        sum(
            state.delivered_pairs
            for request_id, state in env.requests.items()
            if request_id.startswith(f"f{flow}-")
        )
        for flow in range(request_count)
    ]
    mean = float(np.mean(flow_totals)) if flow_totals else 0.0
    return {
        "throughput_pairs_per_s": float(env.delivered_pairs / max(
            env.time * PAPER["allocation_delay_s"], 1e-12
        )),
        "delivered_pairs": float(env.delivered_pairs),
        "dropped_pairs": float(env.metrics()["dropped_pairs"]),
        "completion_rate": float(env.metrics()["completion_rate"]),
        "fairness_cv": float(np.std(flow_totals) / mean) if mean else float("inf"),
        "time_slots": float(env.time),
    }


def run_sweep(
    *,
    experiment: str,
    seeds: int,
    quick: bool,
    horizon_slots: int,
) -> dict[str, object]:
    if experiment == "exp1":
        windows = (10, 20, 30) if not quick else (10, 20)
        retries = tuple(range(1, 11)) if not quick else (1, 5, 10)
        rows = []
        for window in windows:
            for max_try in retries:
                for reroute in (False, True):
                    values = [
                        run_sequence_case(
                            window=window,
                            max_try=max_try,
                            request_count=1,
                            memory_size=20,
                            reroute=reroute,
                            seed=seed,
                            horizon_slots=horizon_slots,
                        )
                        for seed in range(seeds)
                    ]
                    row = {
                        "window": window,
                        "max_try": max_try,
                        "reroute": reroute,
                        **{
                            key: float(np.mean([value[key] for value in values]))
                            for key in values[0]
                        },
                    }
                    rows.append(row)
    elif experiment == "exp2":
        windows = tuple(range(1, 31)) if not quick else (1, 5, 10, 20, 30)
        retries = (1, 5, 10)
        rows = []
        for window in windows:
            for max_try in retries:
                for reroute in (False, True):
                    values = [
                        run_sequence_case(
                            window=window,
                            max_try=max_try,
                            request_count=5,
                            memory_size=10,
                            reroute=reroute,
                            seed=seed,
                            horizon_slots=horizon_slots,
                        )
                        for seed in range(seeds)
                    ]
                    rows.append({
                        "window": window,
                        "max_try": max_try,
                        "reroute": reroute,
                        **{
                            key: float(np.mean([value[key] for value in values]))
                            for key in values[0]
                        },
                    })
    else:
        raise ValueError("experiment must be exp1 or exp2")
    return {
        "paper": PAPER,
        "experiment": experiment,
        "seeds": seeds,
        "horizon_slots": horizon_slots,
        "rows": rows,
        "physical_backend": "SeQUeNCe",
        "reference_source": "QDDCA/exp1.py and QDDCA/exp2.py",
        "comparison_note": (
            "SimQN reference numbers are not expected to match exactly: "
            "SeQUeNCe executes the physical generation, memory, channel, and "
            "swapping events. Compare trends and report both data sets."
        ),
    }


FLOW_ENDPOINTS = ((0, 15), (3, 12), (4, 11), (7, 8), (1, 14))


def make_trend_spec(window: int, seed: int, ttl: int = 30) -> EpisodeSpec:
    """Build the small deterministic workload used by the original trend check."""
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
            generation_probability=1.0,
            swap_probability=1.0,
            memory_capacity=4,
            memory_lifetime=100,
            node_memory_capacity=20,
        ),
    )


def run_case(window: int, max_try: int, reroute: bool, seed: int) -> dict[str, float]:
    """Run one qualitative trend case, preserving the old testable contract."""
    env = make_sequence_env(
        make_trend_spec(window, seed),
        candidate_count=64,
        request_driven_generation=True,
        local_candidates=True,
        best_effort_allocations=True,
    )
    planner = QDDCAPlanner(max_try=max_try, allow_reroute=reroute, seed=seed)
    planner.reset(seed)
    while not env.done:
        snapshot = env.snapshot()
        env.commit(planner.select(snapshot))
    per_flow = [
        sum(
            state.delivered_pairs
            for request_id, state in env.requests.items()
            if request_id.startswith(f"f{flow}-")
        )
        for flow in range(len(FLOW_ENDPOINTS))
    ]
    throughput = sum(per_flow) / max(env.time, 1)
    drops = sum(state.expired_at is not None for state in env.requests.values()) + int(
        env.metrics()["dropped_pairs"]
    )
    mean = float(np.mean(per_flow))
    return {
        "throughput": float(throughput),
        "drop": float(drops),
        "cv": float(np.std(per_flow) / mean) if mean else float("inf"),
        "completed": float(sum(per_flow)),
        "time": float(env.time),
    }


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _corr(left: list[float], right: list[float]) -> float:
    value = spearmanr(left, right).statistic
    return 0.0 if np.isnan(value) else float(value)


def _shape_stats(reference: list[float], observed: list[float]) -> dict[str, float]:
    """Scale-free curve agreement for metrics with different native units."""
    left = np.asarray(reference, dtype=float)
    right = np.asarray(observed, dtype=float)
    left_scale = max(float(np.max(np.abs(left))), 1e-12)
    right_scale = max(float(np.max(np.abs(right))), 1e-12)
    left_normalized = left / left_scale
    right_normalized = right / right_scale
    left_direction = np.sign(np.diff(left))
    right_direction = np.sign(np.diff(right))
    return {
        "spearman": _corr(reference, observed),
        "direction_agreement": float(np.mean(left_direction == right_direction)),
        "normalized_bias": float(np.mean(right_normalized - left_normalized)),
        "normalized_mae": float(np.mean(np.abs(right_normalized - left_normalized))),
    }


def _relative_improvement(before: float, after: float, lower_is_better: bool) -> float:
    if before == 0:
        return 0.0 if after == 0 else float("inf")
    delta = before - after if lower_is_better else after - before
    return float(delta / abs(before))


def _meets_threshold(value: float, threshold: float, tolerance: float = 1e-12) -> bool:
    return value >= threshold - tolerance


def run_suite(seeds: int = 3) -> dict[str, object]:
    """Validate qualitative retry/window/reroute trends against paper curves."""
    if seeds < 1:
        raise ValueError("seeds must be positive")
    retry = [
        _mean([
            run_case(OFFICIAL["retry"]["window"], max_try, True, seed)
            for seed in range(seeds)
        ])
        for max_try in OFFICIAL["retry"]["x"]
    ]
    window = [
        _mean([
            run_case(size, OFFICIAL["window"]["max_try"], True, seed)
            for seed in range(seeds)
        ])
        for size in OFFICIAL["window"]["x"]
    ]
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
        "reroute_throughput_improves": reroute["true"]["throughput"] > reroute["false"]["throughput"],
        "reroute_drop_improves": reroute["true"]["drop"] < reroute["false"]["drop"],
        "reroute_cv_improves": reroute["true"]["cv"] < reroute["false"]["cv"],
    }
    window_throughput = [row["throughput"] for row in window]
    validation.update({
        "window_rises_before_congestion": window_throughput[2] > window_throughput[0],
        "window_peaks_at_moderate_load": int(np.argmax(window_throughput)) in (1, 2),
        "window_declines_under_high_load": window_throughput[-1] < max(window_throughput),
    })
    validation["overall_pass"] = bool(
        _meets_threshold(validation["retry_throughput_spearman"], 0.8)
        and _meets_threshold(validation["retry_drop_spearman"], 0.8)
        and _meets_threshold(validation["window_throughput_spearman"], 0.8)
        and _meets_threshold(validation["window_drop_spearman"], 0.8)
        and _meets_threshold(validation["window_cv_spearman"], 0.8)
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
    parser = argparse.ArgumentParser(description="Reproduce Q-DDCA on SeQUeNCe")
    parser.add_argument("--mode", choices=("sweep", "trends"), default="sweep")
    parser.add_argument("--experiment", choices=("exp1", "exp2"), default="exp1")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--horizon-slots", type=int, default=200)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/qddca_sequence_reproduction.json"))
    args = parser.parse_args()
    if args.seeds < 1 or args.horizon_slots < 1:
        raise ValueError("seeds and horizon-slots must be positive")
    if args.mode == "trends":
        result = run_suite(args.seeds)
    else:
        result = run_sweep(
            experiment=args.experiment,
            seeds=args.seeds,
            quick=args.quick,
            horizon_slots=args.horizon_slots,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "mode": args.mode,
        "experiment": result.get("experiment"),
        "rows": len(result.get("rows", [])),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
