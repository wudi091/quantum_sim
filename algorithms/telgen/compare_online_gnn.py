"""Paired online comparison on one shared generated episode suite."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from statistics import fmean
from time import perf_counter

from algorithms.baselines.online import (
    OnlineBaselineConfig,
    run_online_baseline,
)
from qnet_core.scenario import ScenarioConfig, make_episode
from qnet_core.spec import PhysicalConfig
from qnet_core.workload import resolve_periodic_arrival_workload

from .comparison_methods import (
    COMPARISON_PROFILES,
    FORMAL_METHOD_ORDER,
    ROUTING_BASELINE_METHODS,
    methods_for_profile,
    validate_profile_methods,
)
from .online import OnlineTELGENConfig, run_online_telgen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a frozen online GNN against MILP and non-learning "
            "routing baselines on paired EpisodeSpec instances."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=15000)
    workload = parser.add_mutually_exclusive_group()
    workload.add_argument(
        "--requests",
        type=int,
        help="legacy mode: fixed total request count",
    )
    workload.add_argument(
        "--arrival-rounds",
        type=int,
        help=(
            "Q-CAST-style mode: fixed traffic rounds with exactly "
            "--requests-per-batch new requests per round"
        ),
    )
    parser.add_argument("--requests-per-batch", type=int, default=5)
    parser.add_argument("--decision-interval", type=int, default=4)
    parser.add_argument("--ttl", type=int, default=16)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--min-hops", type=int, default=4)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument(
        "--endpoint-mode",
        choices=("distance_stratified", "uniform_random", "qcast_random"),
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
    parser.add_argument(
        "--fixed-swap-tree-index",
        type=int,
        help=(
            "Restrict the GNN candidate set to one swap_tree_k. Omit this "
            "option for adaptive per-request construction selection."
        ),
    )
    parser.add_argument("--gnn-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--milp-time-limit-seconds", type=float, default=300.0)
    parser.add_argument(
        "--comparison-profile",
        choices=tuple(COMPARISON_PROFILES),
        default="formal",
        help=(
            "formal=GNN/MILP/Q-PASS/Greedy; "
            "scalable=GNN/Q-PASS/Greedy; "
            "construction_ablation=GNN only"
        ),
    )
    parser.add_argument("--generation-probability", type=float, default=0.8)
    parser.add_argument("--swap-probability", type=float, default=0.9)
    parser.add_argument("--memory-capacity", type=int, default=2)
    parser.add_argument("--node-memory-capacity", type=int)
    parser.add_argument("--max-width", type=int, default=1)
    parser.add_argument("--quantum-distance-m", type=float, default=1000.0)
    parser.add_argument("--slot-duration-ps", type=int, default=50_000_000)
    return parser


def _aggregate(
    trials: list[dict[str, object]],
    *,
    comparison_profile: str,
) -> dict[str, dict[str, float]]:
    methods = validate_profile_methods(
        comparison_profile,
        {
            str(method)
            for trial in trials
            for method in trial["methods"]
        },
    )
    result = {}
    for method in methods:
        rows = [trial["methods"][method]["metrics"] for trial in trials]
        keys = sorted(set.intersection(*(set(row) for row in rows)))
        result[method] = {
            key: float(fmean(float(row[key]) for row in rows))
            for key in keys
        }
        result[method]["wall_seconds"] = float(fmean(
            float(trial["methods"][method]["wall_seconds"])
            for trial in trials
        ))
    return result


def _save(output: Path, payload: dict[str, object]) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned = output / f"online_gnn_comparison_{timestamp}.json"
    latest = output / "online_gnn_comparison.json"
    versioned.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copyfile(versioned, latest)
    return versioned, latest


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nearest_rank_percentile(
    values: list[float],
    percentile: float,
) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round(percentile / 100.0 * (len(ordered) - 1)))),
    )
    return float(ordered[index])


def _method_payload(result, wall_seconds: float) -> dict[str, object]:
    metrics = dict(result.metrics)
    decision_seconds = [
        float(decision.decision_seconds)
        for decision in result.decisions
    ]
    metrics["p95_decision_seconds"] = _nearest_rank_percentile(
        decision_seconds,
        95.0,
    )
    attempts = tuple(getattr(result, "attempts", ()))
    construction_attempt_counts = Counter(
        str(attempt.construction_kind) for attempt in attempts
    )
    successful_construction_counts = Counter(
        str(attempt.construction_kind)
        for attempt in attempts
        if attempt.success is True
    )
    return {
        "metrics": metrics,
        "wall_seconds": wall_seconds,
        "violations": [asdict(item) for item in result.violations],
        "construction_usage": {
            "attempt_counts": dict(sorted(construction_attempt_counts.items())),
            "successful_attempt_counts": dict(
                sorted(successful_construction_counts.items())
            ),
        },
    }


def _resolve_construction_space(
    construction_plans: int,
    fixed_swap_tree_index: int | None,
) -> tuple[tuple[str, ...], int | None, str]:
    if construction_plans < 1:
        raise ValueError("construction-plans must be positive")
    if fixed_swap_tree_index is None:
        return (), construction_plans, "adaptive_swap_tree_selection"
    if not 0 <= fixed_swap_tree_index < construction_plans:
        raise ValueError(
            "fixed-swap-tree-index must lie in "
            "[0, construction-plans)"
        )
    return (
        (f"swap_tree_{fixed_swap_tree_index}",),
        None,
        f"fixed_swap_tree_{fixed_swap_tree_index}",
    )


def _routing_baseline_configs(
    *,
    decision_interval: int,
    path_candidate_count: int,
) -> dict[str, OnlineBaselineConfig]:
    return {
        algorithm: OnlineBaselineConfig(
            algorithm=algorithm,
            decision_interval=decision_interval,
            path_candidate_count=path_candidate_count,
            construction_kind="left_deep",
        )
        for algorithm in ROUTING_BASELINE_METHODS
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seeds < 1:
        raise ValueError("seeds must be positive")
    profile_methods = methods_for_profile(args.comparison_profile)
    if (
        args.fixed_swap_tree_index is not None
        and args.comparison_profile != "construction_ablation"
    ):
        raise ValueError(
            "fixed-swap-tree-index is reserved for the "
            "construction_ablation profile"
        )
    construction_kinds, swap_tree_count, construction_policy = (
        _resolve_construction_space(
            args.construction_plans,
            args.fixed_swap_tree_index,
        )
    )
    workload = resolve_periodic_arrival_workload(
        request_count=args.requests,
        arrival_rounds=args.arrival_rounds,
        requests_per_round=args.requests_per_batch,
        arrival_interval_slots=args.decision_interval,
        ttl_slots=args.ttl,
        horizon_slots=args.horizon,
        default_request_count=20,
    )
    horizon = workload.horizon_slots
    physical = PhysicalConfig(
        generation_probability=args.generation_probability,
        swap_probability=args.swap_probability,
        memory_capacity=args.memory_capacity,
        node_memory_capacity=args.node_memory_capacity,
        max_width=args.max_width,
        quantum_distance_m=args.quantum_distance_m,
        slot_duration_ps=args.slot_duration_ps,
    )
    min_hops = (
        args.min_hops
        if args.endpoint_mode == "distance_stratified"
        else None
    )
    max_hops = (
        args.max_hops
        if args.endpoint_mode == "distance_stratified"
        else None
    )
    scenario = ScenarioConfig(
        request_count=workload.request_count,
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
    common = dict(
        decision_interval=args.decision_interval,
        path_candidate_count=args.paths,
        construction_kinds=construction_kinds,
        swap_tree_count=swap_tree_count,
        purification_kinds=("none",),
    )
    gnn_config = OnlineTELGENConfig(
        **common,
        decision_backend="gnn",
        gnn_checkpoint=str(args.checkpoint),
        gnn_device=args.gnn_device,
    )
    milp_config = OnlineTELGENConfig(
        **common,
        decision_backend="milp_teacher",
        milp_time_limit_seconds=args.milp_time_limit_seconds,
        milp_relative_gap=0.0,
    )
    routing_baseline_configs = _routing_baseline_configs(
        decision_interval=args.decision_interval,
        path_candidate_count=args.paths,
    )
    trials = []
    started = perf_counter()
    for index, seed in enumerate(range(args.seed_start, args.seed_start + args.seeds)):
        episode = make_episode(scenario, seed)
        methods = {}
        gnn_started = perf_counter()
        gnn = run_online_telgen(episode, gnn_config)
        methods["gnn"] = _method_payload(
            gnn,
            perf_counter() - gnn_started,
        )
        if "milp" in profile_methods:
            milp_started = perf_counter()
            milp = run_online_telgen(episode, milp_config)
            methods["milp"] = _method_payload(
                milp,
                perf_counter() - milp_started,
            )
        if "qpass" in profile_methods:
            qpass_started = perf_counter()
            qpass = run_online_baseline(
                episode,
                routing_baseline_configs["qpass"],
            )
            methods["qpass"] = _method_payload(
                qpass,
                perf_counter() - qpass_started,
            )
        if "greedy" in profile_methods:
            greedy_started = perf_counter()
            greedy = run_online_baseline(
                episode,
                routing_baseline_configs["greedy"],
            )
            methods["greedy"] = _method_payload(
                greedy,
                perf_counter() - greedy_started,
            )
        validate_profile_methods(args.comparison_profile, set(methods))
        trials.append({
            "seed": seed,
            "episode": asdict(episode),
            "methods": methods,
        })
        summary = " ".join(
            f"{name}={int(item['metrics']['completed_requests'])}"
            for name, item in methods.items()
        )
        print(
            f"episode={index + 1}/{args.seeds} seed={seed} {summary}",
            flush=True,
        )
    payload = {
        "schema_version": 1,
        "experiment": "paired_online_gnn_milp_routing_baselines",
        "comparison_contract": {
            "paired_episode_spec": True,
            "independent_persistent_executors": True,
            "future_requests_hidden": True,
            "gnn_calls_milp_online": False,
            "comparison_profile": args.comparison_profile,
            "formal_method_order": list(FORMAL_METHOD_ORDER),
            "active_method_order": list(profile_methods),
            "fixed_construction_role": "separate_paired_ablation",
            "qcast_included": False,
            "qpass_uses_gnn_or_milp": False,
            "qpass_included": "qpass" in profile_methods,
            "greedy_uses_gnn_or_milp": False,
            "greedy_included": "greedy" in profile_methods,
            "primary_metric": "completed_requests",
            "secondary_metric": "mean_censored_latency_ps",
            "gnn_construction_policy": construction_policy,
        },
        "configuration": {
            **vars(args),
            "checkpoint": str(args.checkpoint),
            "output": str(args.output),
            "workload": asdict(workload),
            "resolved_request_count": workload.request_count,
            "resolved_horizon": horizon,
        },
        "checkpoint_sha256": _checkpoint_sha256(args.checkpoint),
        "scenario": asdict(scenario),
        "gnn_config": asdict(gnn_config),
        "milp_config": (
            asdict(milp_config) if "milp" in profile_methods else None
        ),
        "qcast_config": None,
        "qpass_config": (
            asdict(routing_baseline_configs["qpass"])
            if "qpass" in profile_methods
            else None
        ),
        "greedy_config": (
            asdict(routing_baseline_configs["greedy"])
            if "greedy" in profile_methods
            else None
        ),
        "elapsed_seconds": perf_counter() - started,
        "aggregate": _aggregate(
            trials,
            comparison_profile=args.comparison_profile,
        ),
        "trials": trials,
    }
    versioned, _ = _save(args.output, payload)
    for method, metrics in payload["aggregate"].items():
        print(
            f"{method}: completed={metrics['completed_requests']:.3f} "
            f"latency_ps={metrics['mean_censored_latency_ps']:.3f} "
            f"decision_s={metrics.get('mean_decision_seconds', 0.0):.6f}"
        )
    print(f"json: {versioned}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
