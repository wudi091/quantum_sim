"""Fair rolling comparison with the Q-CAST path baseline."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
from pathlib import Path
import shutil
from statistics import fmean
from typing import Mapping

import networkx as nx

from algorithms.qcast.online import (
    OnlineQCASTConfig,
    OnlineQCASTResult,
    run_online_qcast,
)
from qnet_core.scenario import ScenarioConfig, make_episode
from qnet_core.spec import EpisodeSpec, PhysicalConfig

from .online import OnlineTELGENConfig, OnlineTELGENResult, run_online_telgen


@dataclass(frozen=True)
class OnlineComparisonTrial:
    seed: int
    episode: EpisodeSpec
    telgen: OnlineTELGENResult
    qcast: OnlineQCASTResult
    path_only: OnlineTELGENResult | None = None


@dataclass(frozen=True)
class OnlineComparisonReport:
    scenario: ScenarioConfig
    telgen_config: OnlineTELGENConfig
    qcast_config: OnlineQCASTConfig
    trials: tuple[OnlineComparisonTrial, ...]
    aggregate: Mapping[str, Mapping[str, float]]
    path_only_config: OnlineTELGENConfig | None = None


@dataclass(frozen=True)
class OnlineComparisonPaths:
    json_path: Path
    csv_path: Path
    latest_json_path: Path
    latest_csv_path: Path


@dataclass(frozen=True)
class ComparisonTemporalConfig:
    """Resolved timing for the periodic micro-batch CLI benchmark."""

    output: str
    ttl: int
    horizon: int
    decision_interval: int


def _aggregate(
    trials: tuple[OnlineComparisonTrial, ...],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    path_only_presence = [trial.path_only is not None for trial in trials]
    if any(path_only_presence) and not all(path_only_presence):
        raise ValueError("path-only ablation must be present for every trial")
    methods = ["telgen", "qcast"]
    if all(path_only_presence):
        methods.append("path_only")
    for method in methods:
        method_results = [getattr(trial, method) for trial in trials]
        rows = [dict(item.metrics) for item in method_results if item is not None]
        keys = sorted(set.intersection(*(set(row) for row in rows)))
        result[method] = {
            key: float(fmean(row[key] for row in rows))
            for key in keys
        }
    return result


def run_online_comparison(
    scenario: ScenarioConfig,
    *,
    seeds: int = 1,
    seed_start: int = 100,
    telgen_config: OnlineTELGENConfig | None = None,
    qcast_config: OnlineQCASTConfig | None = None,
    include_path_only: bool = False,
    path_only_config: OnlineTELGENConfig | None = None,
) -> OnlineComparisonReport:
    """Run both methods on identical independently recreated episodes."""

    if seeds < 1:
        raise ValueError("seeds must be positive")
    if seed_start < 0:
        raise ValueError("seed_start cannot be negative")
    default_interval = (
        scenario.arrival_interval
        if scenario.arrival_batch_size is not None
        else scenario.horizon
    )
    telgen_settings = telgen_config or OnlineTELGENConfig(
        decision_interval=default_interval,
    )
    qcast_settings = qcast_config or OnlineQCASTConfig(
        decision_interval=default_interval,
    )
    if scenario.arrival_batch_size is not None and (
        telgen_settings.decision_interval != scenario.arrival_interval
        or qcast_settings.decision_interval != scenario.arrival_interval
    ):
        raise ValueError(
            "periodic comparison requires the decision interval to match "
            "the request arrival interval"
        )
    path_only_settings = None
    if include_path_only:
        path_only_settings = path_only_config or replace(
            telgen_settings,
            construction_kinds=("left_deep",),
            swap_tree_count=None,
        )
    if (
        telgen_settings.decision_interval != qcast_settings.decision_interval
        or telgen_settings.path_candidate_count
        != qcast_settings.path_candidate_count
    ):
        raise ValueError(
            "comparison requires identical decision interval and path count"
        )
    if path_only_settings is not None and (
        path_only_settings.decision_interval
        != telgen_settings.decision_interval
        or path_only_settings.path_candidate_count
        != telgen_settings.path_candidate_count
    ):
        raise ValueError(
            "path-only ablation requires identical interval and path count"
        )
    if path_only_settings is not None and (
        path_only_settings.construction_kinds != ("left_deep",)
        or path_only_settings.swap_tree_count is not None
        or path_only_settings.purification_kinds
        != telgen_settings.purification_kinds
    ):
        raise ValueError(
            "path-only ablation must change only the construction-tree choice"
        )
    trials = []
    for seed in range(seed_start, seed_start + seeds):
        episode = make_episode(scenario, seed)
        trials.append(OnlineComparisonTrial(
            seed=seed,
            episode=episode,
            telgen=run_online_telgen(episode, telgen_settings),
            qcast=run_online_qcast(episode, qcast_settings),
            path_only=(
                None
                if path_only_settings is None
                else run_online_telgen(episode, path_only_settings)
            ),
        ))
    frozen_trials = tuple(trials)
    return OnlineComparisonReport(
        scenario=scenario,
        telgen_config=telgen_settings,
        qcast_config=qcast_settings,
        trials=frozen_trials,
        aggregate=_aggregate(frozen_trials),
        path_only_config=path_only_settings,
    )


def _episode_diagnostics(
    episode: EpisodeSpec,
    decision_interval: int,
) -> dict[str, object]:
    """Summarize whether an episode contains construction-aware choices."""

    graph = nx.Graph()
    graph.add_nodes_from(episode.nodes)
    graph.add_edges_from(episode.edges)
    distances = dict(nx.all_pairs_shortest_path_length(graph))
    hop_histogram: dict[int, int] = {}
    for request in episode.requests:
        hops = int(distances[request.source][request.destination])
        hop_histogram[hops] = hop_histogram.get(hops, 0) + 1
    arrival_windows: dict[int, int] = {}
    for request in episode.requests:
        window = request.arrival // decision_interval
        arrival_windows[window] = arrival_windows.get(window, 0) + 1
    request_count = len(episode.requests)
    depth_choice_count = sum(
        count for hops, count in hop_histogram.items() if hops >= 4
    )
    return {
        "shortest_hop_histogram": dict(sorted(hop_histogram.items())),
        "construction_depth_choice_request_count": depth_choice_count,
        "construction_depth_choice_rate": (
            0.0 if request_count == 0 else depth_choice_count / request_count
        ),
        "mean_arrivals_per_nonempty_decision_window": (
            0.0
            if not arrival_windows
            else request_count / len(arrival_windows)
        ),
        "max_arrivals_per_decision_window": max(
            arrival_windows.values(), default=0
        ),
    }


def _method_payload(result: OnlineTELGENResult | OnlineQCASTResult) -> dict[str, object]:
    return {
        "metrics": dict(result.metrics),
        "decisions": [asdict(item) for item in result.decisions],
        "attempts": [asdict(item) for item in result.attempts],
        "settlements": [asdict(item) for item in result.settlements],
        "violations": [asdict(item) for item in result.violations],
    }


def _json_payload(report: OnlineComparisonReport) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "comparison_contract": {
            "paired_episode_spec": True,
            "independent_persistent_executors": True,
            "common_runtime_metric": "mean_decision_seconds",
            "qcast_baseline": "width_one_ext_fixed_construction",
            "qcast_uses_telgen_lp_or_search_decoder": False,
            "path_only_ablation": report.path_only_config is not None,
        },
        "scenario": asdict(report.scenario),
        "telgen_config": asdict(report.telgen_config),
        "qcast_config": asdict(report.qcast_config),
        "aggregate": {
            method: dict(metrics)
            for method, metrics in report.aggregate.items()
        },
        "trials": [
            {
                "seed": trial.seed,
                "episode": asdict(trial.episode),
                "diagnostics": _episode_diagnostics(
                    trial.episode,
                    report.telgen_config.decision_interval,
                ),
                "telgen": _method_payload(trial.telgen),
                "qcast": _method_payload(trial.qcast),
                **(
                    {}
                    if trial.path_only is None
                    else {"path_only": _method_payload(trial.path_only)}
                ),
            }
            for trial in report.trials
        ],
    }
    if report.path_only_config is not None:
        payload["path_only_config"] = asdict(report.path_only_config)
    return payload


def save_online_comparison(
    report: OnlineComparisonReport,
    output_directory: str | Path,
) -> OnlineComparisonPaths:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = ""
    collision_index = 1
    while True:
        json_path = output / f"online_comparison_{timestamp}{suffix}.json"
        csv_path = output / f"online_comparison_{timestamp}{suffix}.csv"
        if not json_path.exists() and not csv_path.exists():
            break
        collision_index += 1
        suffix = f"_{collision_index}"
    latest_json = output / "online_comparison.json"
    latest_csv = output / "online_comparison.csv"
    json_path.write_text(
        json.dumps(_json_payload(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metric_keys = sorted({
        key
        for trial in report.trials
        for result in (trial.telgen, trial.qcast, trial.path_only)
        if result is not None
        for key in result.metrics
    })
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("seed", "method", *metric_keys),
        )
        writer.writeheader()
        for trial in report.trials:
            method_results = [
                ("telgen", trial.telgen),
                ("qcast", trial.qcast),
            ]
            if trial.path_only is not None:
                method_results.append(("path_only", trial.path_only))
            for method, result in method_results:
                writer.writerow({
                    "seed": trial.seed,
                    "method": method,
                    **dict(result.metrics),
                })
    shutil.copyfile(json_path, latest_json)
    shutil.copyfile(csv_path, latest_csv)
    return OnlineComparisonPaths(
        json_path=json_path,
        csv_path=csv_path,
        latest_json_path=latest_json,
        latest_csv_path=latest_csv,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare TELGEN and Q-CAST on identical rolling episodes."
    )
    parser.add_argument(
        "--output",
        help="Output directory for the periodic micro-batch benchmark.",
    )
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=3101)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument(
        "--requests-per-batch",
        type=int,
        default=10,
        help="Fixed number of new requests generated at each decision boundary.",
    )
    parser.add_argument(
        "--ttl",
        type=int,
        help="Request TTL from its own arrival (default: 16 slots).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        help="Total episode slots including final queue draining (default: automatic).",
    )
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument(
        "--min-hops",
        type=int,
        help=(
            "Override the default minimum source-destination shortest-path "
            "distance of 4; must be used together with --max-hops."
        ),
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        help=(
            "Override the default maximum source-destination shortest-path "
            "distance of 4; must be used together with --min-hops."
        ),
    )
    parser.add_argument(
        "--uniform-random-endpoints",
        action="store_true",
        help="Disable hop bounds and sample two connected endpoints uniformly.",
    )
    parser.add_argument("--paths", type=int, default=4)
    parser.add_argument("--construction-plans", type=int, default=5)
    parser.add_argument("--waxman-alpha", type=float, default=0.15)
    parser.add_argument("--waxman-beta", type=float, default=0.45)
    parser.add_argument("--topology-attempts", type=int, default=128)
    parser.add_argument("--waxman-add-mst", action="store_true")
    parser.add_argument(
        "--decision-interval",
        type=int,
        help="Slots between recurring planning decisions (default: 4).",
    )
    parser.add_argument("--generation-probability", type=float, default=0.8)
    parser.add_argument("--swap-probability", type=float, default=0.9)
    parser.add_argument("--memory-capacity", type=int, default=2)
    parser.add_argument("--memory-lifetime", type=int, default=300)
    parser.add_argument("--node-memory-capacity", type=int)
    parser.add_argument("--max-width", type=int, choices=(1,), default=1)
    parser.add_argument("--quantum-distance-m", type=float, default=1000.0)
    parser.add_argument("--slot-duration-ps", type=int, default=50_000_000)
    parser.add_argument("--without-path-only", action="store_true")
    return parser


def _endpoint_mode_for_hops(
    min_hops: int | None,
    max_hops: int | None,
) -> str:
    """Select strict hop-stratified endpoints only when both bounds are set."""

    if (min_hops is None) != (max_hops is None):
        raise ValueError("min-hops and max-hops must be provided together")
    if min_hops is None:
        return "uniform_random"
    if min_hops < 1:
        raise ValueError("min-hops must be positive")
    if max_hops < min_hops:
        raise ValueError("max-hops must be greater than or equal to min-hops")
    return "distance_stratified"


def _resolve_endpoint_configuration(
    min_hops: int | None,
    max_hops: int | None,
    uniform_random_endpoints: bool,
) -> tuple[int | None, int | None, str]:
    """Resolve the fixed-hop default or the explicit random-endpoint mode."""

    if uniform_random_endpoints:
        if min_hops is not None or max_hops is not None:
            raise ValueError(
                "uniform-random-endpoints cannot be combined with hop bounds"
            )
        return None, None, "uniform_random"
    if min_hops is None and max_hops is None:
        return 4, 4, "distance_stratified"
    return min_hops, max_hops, _endpoint_mode_for_hops(min_hops, max_hops)


def _resolve_temporal_configuration(
    *,
    output: str | None,
    ttl: int | None,
    horizon: int | None,
    decision_interval: int | None,
    request_count: int = 100,
    requests_per_batch: int = 10,
) -> ComparisonTemporalConfig:
    """Resolve periodic micro-batch arrivals and the final drain horizon."""

    if request_count < 1 or requests_per_batch < 1:
        raise ValueError("request counts must be positive")
    interval = 4 if decision_interval is None else decision_interval
    resolved_ttl = 16 if ttl is None else ttl
    arrival_rounds = (request_count + requests_per_batch - 1) // requests_per_batch
    last_arrival = (arrival_rounds - 1) * interval
    fixed_horizon = last_arrival + resolved_ttl if horizon is None else horizon
    resolved = ComparisonTemporalConfig(
        output=output or "results/telgen_qcast_waxman_fixed4_periodic",
        ttl=resolved_ttl,
        horizon=fixed_horizon,
        decision_interval=interval,
    )
    if resolved.horizon < last_arrival + resolved.ttl:
        raise ValueError(
            "episode horizon must cover the final arrival's complete TTL"
        )
    if resolved.ttl < 1 or resolved.horizon < 1:
        raise ValueError("ttl and horizon must be positive")
    if resolved.decision_interval < 1:
        raise ValueError("decision interval must be positive")
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.construction_plans < 1:
        parser.error("construction-plans must be positive")
    try:
        min_hops, max_hops, endpoint_mode = _resolve_endpoint_configuration(
            args.min_hops,
            args.max_hops,
            args.uniform_random_endpoints,
        )
        timing = _resolve_temporal_configuration(
            output=args.output,
            ttl=args.ttl,
            horizon=args.horizon,
            decision_interval=args.decision_interval,
            request_count=args.requests,
            requests_per_batch=args.requests_per_batch,
        )
    except ValueError as error:
        parser.error(str(error))
    physical = PhysicalConfig(
        generation_probability=args.generation_probability,
        swap_probability=args.swap_probability,
        memory_capacity=args.memory_capacity,
        memory_lifetime=args.memory_lifetime,
        node_memory_capacity=args.node_memory_capacity,
        max_width=args.max_width,
        quantum_distance_m=args.quantum_distance_m,
        slot_duration_ps=args.slot_duration_ps,
    )
    scenario = ScenarioConfig(
        request_count=args.requests,
        min_hops=min_hops,
        max_hops=max_hops,
        ttl=timing.ttl,
        horizon=timing.horizon,
        physical=physical,
        topology_nodes=args.nodes,
        waxman_alpha=args.waxman_alpha,
        waxman_beta=args.waxman_beta,
        topology_attempts=args.topology_attempts,
        waxman_add_mst=args.waxman_add_mst,
        endpoint_mode=endpoint_mode,
        topology_mode="waxman",
        arrival_batch_size=args.requests_per_batch,
        arrival_interval=timing.decision_interval,
    )
    report = run_online_comparison(
        scenario,
        seeds=args.seeds,
        seed_start=args.seed_start,
        telgen_config=OnlineTELGENConfig(
            decision_interval=timing.decision_interval,
            path_candidate_count=args.paths,
            construction_kinds=(),
            swap_tree_count=args.construction_plans,
            purification_kinds=("none",),
            teacher_solver_backend="highs_ipm",
        ),
        qcast_config=OnlineQCASTConfig(
            decision_interval=timing.decision_interval,
            path_candidate_count=args.paths,
            construction_kind="left_deep",
            purification_kind="none",
        ),
        include_path_only=not args.without_path_only,
    )
    paths = save_online_comparison(report, timing.output)
    for method in report.aggregate:
        metrics = report.aggregate[method]
        print(
            f"{method}: completed={metrics['completed_requests']:.3f} "
            f"rate={metrics['completion_rate']:.3f} "
            f"latency_ps={metrics['mean_censored_latency_ps']:.3f}"
        )
    print(f"json: {paths.json_path}")
    print(f"csv: {paths.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
