"""Reproducible SeQUeNCe experiment harness for construction-aware routing."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from importlib import metadata
import json
import math
from pathlib import Path
import platform
from time import perf_counter
from typing import Iterable

import networkx as nx
import numpy as np

from qnet_core.construction_catalog import (
    RouteConstructionCandidate,
    build_route_construction_catalogue,
)
from qnet_core.construction_evaluate import run_joint_plan_baseline
from qnet_core.scenario import ScenarioConfig, make_episode
from qnet_core.spec import PhysicalConfig

from .baselines import (
    BalancedConstructionPolicy,
    MemoryAwareConstructionPolicy,
    ShortestPathLeftDeepPolicy,
)
from .torch_policy import TorchCAAPPOPolicy
from .torch_trainer import TorchCAAPPORolloutTrainer
from .oracle import DeterministicJointPlanOracle, OracleLimitError


@dataclass(frozen=True)
class CAAPPOVariant:
    name: str
    candidate_count: int = 3
    construction_kinds: tuple[str, ...] = ("left_deep", "balanced")
    gamma_per_slot: float = 1.0
    risk_limit: float = 0.0
    beta: float = 1.0
    use_dag_state: bool = True
    use_capacity_context: bool = True
    potential_shaping: bool = True


@dataclass(frozen=True)
class ConstructionExperimentConfig:
    scenario: ScenarioConfig
    evaluation_seeds: tuple[int, ...] = (101, 102, 103)
    training_seeds: tuple[int, ...] = (1, 2, 3)
    training_episodes: int = 6
    candidate_count: int = 3
    confidence_level: float = 0.95
    include_nominal_oracle: bool = True
    variants: tuple[CAAPPOVariant, ...] = (
        CAAPPOVariant("caappo"),
        CAAPPOVariant("no_dual", risk_limit=1e9),
        CAAPPOVariant("time_discounted", gamma_per_slot=0.99),
        CAAPPOVariant(
            "no_construction_choice", construction_kinds=("left_deep",)
        ),
        CAAPPOVariant("no_route_choice", candidate_count=1),
        CAAPPOVariant("no_flow_reward", beta=0.0),
        CAAPPOVariant("no_dag_state", use_dag_state=False),
        CAAPPOVariant("no_potential_shaping", potential_shaping=False),
        CAAPPOVariant("no_capacity_context", use_capacity_context=False),
    )

    def __post_init__(self) -> None:
        if not self.evaluation_seeds or not self.training_seeds:
            raise ValueError("training and evaluation seed lists must be non-empty")
        if set(self.evaluation_seeds).intersection(self.training_seeds):
            raise ValueError("training and evaluation seeds must be disjoint")
        if self.training_episodes < 0 or self.candidate_count < 1:
            raise ValueError("invalid training episode or candidate count")
        if self.confidence_level != 0.95:
            raise ValueError("the current harness reports a fixed 95% normal CI")


BASELINES = {
    "shortest_left_deep": ShortestPathLeftDeepPolicy,
    "balanced": BalancedConstructionPolicy,
    "memory_aware": MemoryAwareConstructionPolicy,
}


def _numeric_metrics(values: dict[str, float]) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in values.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }


def _collapse_training_replicas(
    rows: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    """Average training replicas within each evaluation seed.

    Evaluation seeds are the independent statistical units.  Training seeds
    are replicated fits of the same policy and therefore must be averaged
    before confidence intervals are computed.
    """
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    collapsed: list[dict[str, object]] = []
    for row in rows:
        if row.get("method") != "torch_caappo" or "training_seed" not in row:
            collapsed.append(dict(row))
            continue
        key = (str(row["method"]), str(row["variant"]), int(row["seed"]))
        grouped.setdefault(key, []).append(dict(row))

    excluded = {
        "method", "variant", "seed", "training_seed",
        "wall_seconds", "training_seconds",
    }
    for (method, variant, seed), values in sorted(grouped.items()):
        representative = dict(values[0])
        representative.pop("training_seed", None)
        for key in set().union(*(value.keys() for value in values)):
            if key in excluded:
                continue
            samples = [
                float(value[key])
                for value in values
                if isinstance(value.get(key), (int, float))
                and math.isfinite(float(value[key]))
            ]
            if samples:
                representative[key] = float(np.mean(samples))
        representative["method"] = method
        representative["variant"] = variant
        representative["seed"] = seed
        collapsed.append(representative)
    return collapsed


def _aggregate(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    rows = _collapse_training_replicas(rows)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["method"]), str(row["variant"]))
        grouped.setdefault(key, []).append(row)
    result = []
    for (method, variant), values in sorted(grouped.items()):
        metric_names = sorted({
            key
            for value in values
            for key, item in value.items()
            if key not in {
                "method", "variant", "seed", "training_seed",
                "wall_seconds", "training_seconds",
            }
            and isinstance(item, (int, float))
        })
        for metric in metric_names:
            samples = np.asarray(
                [float(row[metric]) for row in values if metric in row],
                dtype=np.float64,
            )
            mean = float(samples.mean())
            std = float(samples.std(ddof=1)) if len(samples) > 1 else 0.0
            half_width = 1.96 * std / math.sqrt(len(samples)) if samples.size else 0.0
            result.append({
                "method": method,
                "variant": variant,
                "metric": metric,
                "n": len(samples),
                "mean": mean,
                "std": std,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
                "ci_supported": len(samples) >= 2,
            })
    return result


def _catalogue(spec, candidate_count: int, construction_kinds: tuple[str, ...]):
    return build_route_construction_catalogue(
        spec.planning,
        candidate_count=candidate_count,
        construction_kinds=construction_kinds,
    )


def _run_baselines(config: ConstructionExperimentConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in config.evaluation_seeds:
        spec = make_episode(config.scenario, seed)
        candidates = _catalogue(
            spec, config.candidate_count, ("left_deep", "balanced")
        )
        for name, policy_type in BASELINES.items():
            started = perf_counter()
            selected = policy_type().select(candidates)
            outcome = run_joint_plan_baseline(spec, selected)
            rows.append({
                "method": "fixed_baseline",
                "variant": name,
                "seed": seed,
                **_numeric_metrics(dict(outcome.metrics)),
                "wall_seconds": perf_counter() - started,
            })
    return rows


def _train_variant(
    config: ConstructionExperimentConfig,
    variant: CAAPPOVariant,
    training_seed: int,
) -> tuple[TorchCAAPPOPolicy, TorchCAAPPORolloutTrainer]:
    policy = TorchCAAPPOPolicy(
        seed=training_seed,
        use_dag_state=variant.use_dag_state,
        use_capacity_context=variant.use_capacity_context,
    )
    trainer = TorchCAAPPORolloutTrainer(
        policy,
        risk_limit=variant.risk_limit,
        gamma_per_slot=variant.gamma_per_slot,
        beta=variant.beta,
        potential_shaping=variant.potential_shaping,
    )
    for episode in range(config.training_episodes):
        base_seed = config.training_seeds[episode % len(config.training_seeds)]
        seed = base_seed + 10_000 * (episode // len(config.training_seeds))
        spec = make_episode(config.scenario, seed)
        candidates = _catalogue(
            spec, variant.candidate_count, variant.construction_kinds
        )
        trainer.run_episode(spec, candidates, deterministic=False, update=True)
    return policy, trainer


def _run_caappo(config: ConstructionExperimentConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant in config.variants:
        for training_seed in config.training_seeds:
            started = perf_counter()
            _, trainer = _train_variant(config, variant, training_seed)
            training_seconds = perf_counter() - started
            for seed in config.evaluation_seeds:
                spec = make_episode(config.scenario, seed)
                candidates = _catalogue(
                    spec, variant.candidate_count, variant.construction_kinds
                )
                evaluation_started = perf_counter()
                outcome = trainer.run_episode(
                    spec, candidates, deterministic=True, update=False
                )
                rows.append({
                    "method": "torch_caappo",
                    "variant": variant.name,
                    "seed": seed,
                    "training_seed": training_seed,
                    **_numeric_metrics(outcome.metrics),
                    "episode_reward": outcome.reward,
                    "discounted_return": outcome.discounted_return,
                    "training_seconds": training_seconds,
                    "wall_seconds": perf_counter() - evaluation_started,
                })
    return rows


def _run_nominal_oracle(
    config: ConstructionExperimentConfig,
) -> list[dict[str, object]]:
    """Run bounded-catalogue and, when tractable, full-path nominal oracles."""
    rows: list[dict[str, object]] = []
    oracle = DeterministicJointPlanOracle(max_states=50_000)
    for seed in config.evaluation_seeds:
        spec = make_episode(config.scenario, seed)
        candidates = _catalogue(
            spec, config.candidate_count, ("left_deep", "balanced")
        )
        reachable = _reachable_simple_paths(spec)
        route_counts = {
            request_id: len(paths)
            for request_id, (paths, _exact) in reachable.items()
        }
        coverage_exact = all(exact for _paths, exact in reachable.values())
        full_path_error: str | None = None
        full_candidates: tuple[RouteConstructionCandidate, ...] | None = None
        if coverage_exact and sum(route_counts.values()) <= 64:
            try:
                full_candidates = build_route_construction_catalogue(
                    spec.planning,
                    candidate_count=None,
                    construction_kinds=("left_deep", "balanced"),
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound, ValueError) as error:
                full_path_error = str(error)
        elif not coverage_exact:
            full_path_error = "full-path enumeration exceeds coverage limit 128"
        else:
            full_path_error = "full-path oracle skipped above 64 reachable routes"
        try:
            result = oracle.solve(spec, candidates)
        except OracleLimitError as error:
            rows.append({
                "method": "nominal_oracle",
                "variant": "skipped_limit",
                "seed": seed,
                "catalogue_route_count": float(sum(
                    len({candidate.route_nodes for candidate in candidates
                         if candidate.request_id == request.id})
                    for request in spec.requests
                )),
                "reachable_route_count": float(sum(route_counts.values())),
                "route_coverage_exact": coverage_exact,
                "oracle_error": str(error),
            })
            continue
        except ValueError as error:
            rows.append({
                "method": "nominal_oracle",
                "variant": "skipped_boundary",
                "seed": seed,
                "catalogue_route_count": float(sum(
                    len({candidate.route_nodes for candidate in candidates
                         if candidate.request_id == request.id})
                    for request in spec.requests
                )),
                "reachable_route_count": float(sum(route_counts.values())),
                "route_coverage_exact": coverage_exact,
                "oracle_error": str(error),
            })
            continue
        row: dict[str, object] = {
            "method": "nominal_oracle",
            "variant": "exact_nominal",
            "seed": seed,
            "oracle_score": result.score,
            "completed_requests": float(result.completed_requests),
            "censored_flow_time_ps": float(result.censored_flow_time_ps),
            "risk_count": float(result.risk_count),
            "makespan_ps": float(result.makespan_ps),
            "explored_states": float(result.explored_states),
            "explored_joint_plans": float(result.explored_joint_plans),
            "catalogue_route_count": float(sum(
                len({candidate.route_nodes for candidate in candidates
                     if candidate.request_id == request.id})
                for request in spec.requests
            )),
            "reachable_route_count": float(sum(route_counts.values())),
            "route_coverage_exact": coverage_exact,
        }
        if full_candidates is not None:
            try:
                full_result = oracle.solve(spec, full_candidates)
            except (OracleLimitError, ValueError) as error:
                full_path_error = str(error)
            else:
                row.update({
                    "full_path_oracle_score": float(full_result.score),
                    "full_path_oracle_gap": max(
                        0.0, float(full_result.score - result.score)
                    ),
                    "full_path_explored_states": float(full_result.explored_states),
                    "full_path_explored_joint_plans": float(
                        full_result.explored_joint_plans
                    ),
                })
        if full_path_error is not None:
            row["full_path_oracle_error"] = full_path_error
        rows.append(row)
    return rows


def _paired_differences(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    rows = list(rows)
    baseline = {
        (int(row["seed"]), metric): float(row[metric])
        for row in rows
        if row.get("method") == "fixed_baseline"
        and row.get("variant") == "shortest_left_deep"
        for metric in (
            "completed_requests",
            "completion_rate",
            "censored_flow_time_ps",
            "mean_censored_latency_ps",
            "risk_count",
        )
        if metric in row
    }
    result: list[dict[str, object]] = []
    variants = sorted({
        str(row["variant"])
        for row in rows
        if row.get("method") == "torch_caappo"
    })
    for variant in variants:
        for metric in (
            "completed_requests",
            "completion_rate",
            "censored_flow_time_ps",
            "mean_censored_latency_ps",
            "risk_count",
        ):
            by_seed: dict[int, list[float]] = {}
            for row in rows:
                if row.get("method") != "torch_caappo" or row.get("variant") != variant:
                    continue
                if metric in row:
                    by_seed.setdefault(int(row["seed"]), []).append(float(row[metric]))
            deltas = [
                float(np.mean(values)) - baseline[(seed, metric)]
                for seed, values in by_seed.items()
                if (seed, metric) in baseline
            ]
            if not deltas:
                continue
            samples = np.asarray(deltas, dtype=np.float64)
            mean = float(samples.mean())
            std = float(samples.std(ddof=1)) if len(samples) > 1 else 0.0
            half_width = 1.96 * std / math.sqrt(len(samples)) if len(samples) > 1 else 0.0
            result.append({
                "variant": variant,
                "reference": "shortest_left_deep",
                "metric": metric,
                "n": len(samples),
                "mean_delta": mean,
                "std_delta": std,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
                "ci_supported": len(samples) >= 2,
            })
    return result


def _reachable_simple_paths(
    spec,
    *,
    limit: int = 128,
) -> dict[str, tuple[tuple[tuple[int, ...], ...], bool]]:
    """Enumerate simple paths up to a cap and report whether the count is exact."""
    graph = nx.Graph()
    graph.add_nodes_from(spec.nodes)
    graph.add_edges_from(spec.edges)
    result: dict[str, tuple[tuple[tuple[int, ...], ...], bool]] = {}
    for request in sorted(spec.requests, key=lambda item: item.id):
        paths: list[tuple[int, ...]] = []
        exact = True
        try:
            iterator = nx.all_simple_paths(
                graph, request.source, request.destination
            )
            for path in iterator:
                if len(paths) >= limit:
                    exact = False
                    break
                paths.append(tuple(int(node) for node in path))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            paths = []
        result[request.id] = (tuple(paths), exact)
    return result


def _catalogue_coverage(config: ConstructionExperimentConfig) -> dict[str, object]:
    counts: list[int] = []
    route_counts: list[int] = []
    reachable_counts: list[int] = []
    ratios: list[float] = []
    kinds: set[str] = set()
    per_request: list[dict[str, object]] = []
    for seed in config.evaluation_seeds:
        spec = make_episode(config.scenario, seed)
        candidates = _catalogue(
            spec, config.candidate_count, ("left_deep", "balanced")
        )
        reachable = _reachable_simple_paths(spec)
        counts.extend(len({candidate.candidate_id for candidate in candidates
                           if candidate.request_id == request.id})
                      for request in spec.requests)
        route_counts.extend(len({candidate.route_nodes for candidate in candidates
                                 if candidate.request_id == request.id})
                            for request in spec.requests)
        kinds.update(candidate.construction_kind for candidate in candidates)
        for request in spec.requests:
            paths, exact = reachable[request.id]
            catalogue_routes = {
                candidate.route_nodes
                for candidate in candidates
                if candidate.request_id == request.id
            }
            reachable_count = len(paths)
            ratio = (
                len(catalogue_routes) / reachable_count
                if exact and reachable_count
                else None
            )
            reachable_counts.append(reachable_count)
            if ratio is not None:
                ratios.append(float(ratio))
            per_request.append({
                "seed": seed,
                "request_id": request.id,
                "catalogue_route_count": len(catalogue_routes),
                "reachable_simple_path_count": reachable_count,
                "coverage_exact": exact,
                "route_coverage": ratio,
            })
    return {
        "mean_candidates_per_request": float(np.mean(counts)) if counts else 0.0,
        "mean_route_variants_per_request": float(np.mean(route_counts)) if route_counts else 0.0,
        "mean_reachable_simple_paths_per_request": (
            float(np.mean(reachable_counts)) if reachable_counts else 0.0
        ),
        "mean_route_coverage": float(np.mean(ratios)) if ratios else None,
        "route_coverage_exact": all(item["coverage_exact"] for item in per_request),
        "per_request": per_request,
        "construction_kinds": sorted(kinds),
    }


def _manifest(config: ConstructionExperimentConfig) -> dict[str, object]:
    packages = {}
    for name in ("sequence", "numpy", "scipy", "torch", "networkx"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "physical_backend": "SeQUeNCe",
        "deterministic_oracle_role": "small-instance nominal validation only",
        "confidence_interval": (
            "95% normal interval over evaluation seeds; training replicas are "
            "averaged within seed; ci_supported=false when n<2"
        ),
        "legacy_baseline_note": (
            "QDDCA and QCAST reproductions remain available separately; their "
            "legacy action spaces are not mixed into this construction-SMDP table."
        ),
        "config": asdict(config),
    }


def run_experiment(config: ConstructionExperimentConfig) -> dict[str, object]:
    rows = _run_baselines(config) + _run_caappo(config)
    if config.include_nominal_oracle:
        rows += _run_nominal_oracle(config)
    return {
        "manifest": _manifest(config),
        "catalogue_coverage": _catalogue_coverage(config),
        "rows": rows,
        "aggregate": _aggregate(rows),
        "paired_differences": _paired_differences(rows),
    }


def write_results(result: dict[str, object], output: Path) -> tuple[Path, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    rows = list(result["rows"])
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output, csv_path


def _quick_config() -> ConstructionExperimentConfig:
    physical = PhysicalConfig(
        generation_probability=0.9,
        swap_probability=0.9,
        memory_capacity=2,
        node_memory_capacity=6,
        memory_lifetime=50,
        quantum_distance_m=1.0,
    )
    scenario = ScenarioConfig(
        request_count=2,
        min_hops=2,
        max_hops=3,
        ttl=20,
        horizon=30,
        topology_nodes=12,
        physical=physical,
    )
    return ConstructionExperimentConfig(
        scenario=scenario,
        evaluation_seeds=(101, 102),
        training_seeds=(1, 2),
        training_episodes=2,
        candidate_count=2,
        variants=(CAAPPOVariant("caappo", candidate_count=2),),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/construction_aware.json"),
    )
    args = parser.parse_args(argv)
    config = _quick_config() if args.quick else replace(
        _quick_config(),
        evaluation_seeds=(101, 102, 103, 104, 105),
        training_seeds=(1, 2, 3, 4, 5),
        training_episodes=30,
        candidate_count=3,
        variants=ConstructionExperimentConfig.__dataclass_fields__["variants"].default,
    )
    result = run_experiment(config)
    json_path, csv_path = write_results(result, args.output)
    print(json.dumps({
        "json": str(json_path),
        "csv": str(csv_path),
        "rows": len(result["rows"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAAPPOVariant",
    "ConstructionExperimentConfig",
    "run_experiment",
    "write_results",
]
