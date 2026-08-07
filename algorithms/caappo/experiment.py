"""Reproducible SeQUeNCe experiment harness for construction-aware routing."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from importlib import metadata
import json
import math
from pathlib import Path
import sys
from time import perf_counter
import tempfile
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
from .checkpoint import (
    CheckpointCompatibilityError,
    checkpoint_sha256,
    load_caappo_checkpoint,
    runtime_manifest as checkpoint_runtime_manifest,
    save_caappo_checkpoint,
)


@dataclass(frozen=True)
class CAAPPOVariant:
    name: str
    candidate_count: int = 3
    construction_kinds: tuple[str, ...] = ("left_deep", "balanced")
    dynamic_repair_paths: int = 4
    gamma_per_slot: float = 1.0
    risk_limit: float = 0.0
    beta: float = 1.0
    use_dag_state: bool = True
    use_capacity_context: bool = True
    potential_shaping: bool = True

    def __post_init__(self) -> None:
        if self.candidate_count < 1:
            raise ValueError("variant candidate_count must be positive")
        if self.dynamic_repair_paths < 0:
            raise ValueError("variant dynamic_repair_paths must be non-negative")
        if not self.construction_kinds:
            raise ValueError("variant needs at least one construction kind")


@dataclass(frozen=True)
class ConstructionExperimentConfig:
    scenario: ScenarioConfig
    evaluation_seeds: tuple[int, ...] = (101, 102, 103)
    training_seeds: tuple[int, ...] = (1, 2, 3)
    validation_seeds: tuple[int, ...] = (51, 52, 53)
    training_episodes: int = 6
    validation_interval: int = 1
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
        CAAPPOVariant(
            "no_route_choice",
            candidate_count=1,
            dynamic_repair_paths=0,
        ),
        CAAPPOVariant("no_flow_reward", beta=0.0),
        CAAPPOVariant("no_dag_state", use_dag_state=False),
        CAAPPOVariant("no_potential_shaping", potential_shaping=False),
        CAAPPOVariant("no_capacity_context", use_capacity_context=False),
    )

    def __post_init__(self) -> None:
        if (
            not self.evaluation_seeds
            or not self.training_seeds
            or not self.validation_seeds
        ):
            raise ValueError(
                "training, validation, and evaluation seed lists must be non-empty"
            )
        seed_groups = (
            ("training", self.training_seeds),
            ("validation", self.validation_seeds),
            ("evaluation", self.evaluation_seeds),
        )
        for left_index, (left_name, left_seeds) in enumerate(seed_groups):
            if any(int(seed) < 0 for seed in left_seeds):
                raise ValueError(f"{left_name} seeds must be non-negative")
            if len(set(left_seeds)) != len(left_seeds):
                raise ValueError(f"{left_name} seeds must be unique")
            for right_name, right_seeds in seed_groups[left_index + 1:]:
                if set(left_seeds).intersection(right_seeds):
                    raise ValueError(
                        f"{left_name} and {right_name} seeds must be disjoint"
                    )
        if self.training_episodes < 0 or self.candidate_count < 1:
            raise ValueError("invalid training episode or candidate count")
        if self.validation_interval < 1:
            raise ValueError("validation_interval must be positive")
        if self.confidence_level != 0.95:
            raise ValueError("the current harness reports a fixed 95% normal CI")


BASELINES = {
    "shortest_left_deep": ShortestPathLeftDeepPolicy,
    "balanced": BalancedConstructionPolicy,
    "memory_aware": MemoryAwareConstructionPolicy,
}


@dataclass(frozen=True)
class CAAPPOTrainingRun:
    checkpoint: Path
    variant: str
    training_seed: int
    completed_episodes: int
    best_validation: dict[str, object] | None
    history: tuple[dict[str, object], ...]
    sha256: str


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


def _aggregate_training_replicas(
    rows: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    """Expose variance across independently trained policy replicas."""
    by_replica: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        if row.get("method") != "torch_caappo" or "training_seed" not in row:
            continue
        key = (str(row["method"]), str(row["variant"]), int(row["training_seed"]))
        by_replica.setdefault(key, []).append(row)
    replica_rows: list[dict[str, object]] = []
    excluded = {
        "method", "variant", "seed", "training_seed", "wall_seconds",
        "training_seconds",
    }
    for (method, variant, training_seed), values in sorted(by_replica.items()):
        replica: dict[str, object] = {
            "method": method,
            "variant": variant,
            "training_seed": training_seed,
        }
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
                replica[key] = float(np.mean(samples))
        replica_rows.append(replica)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in replica_rows:
        grouped.setdefault((str(row["method"]), str(row["variant"])), []).append(row)
    result: list[dict[str, object]] = []
    for (method, variant), values in sorted(grouped.items()):
        metric_names = sorted({
            key
            for value in values
            for key, item in value.items()
            if key not in {"method", "variant", "training_seed"}
            and isinstance(item, (int, float))
        })
        for metric in metric_names:
            samples = np.asarray(
                [float(value[metric]) for value in values if metric in value],
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
                "unit": "training_replica",
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


def _scenario_from_dict(values: dict[str, object]) -> ScenarioConfig:
    data = dict(values)
    physical = data.get("physical")
    if not isinstance(physical, dict):
        raise ValueError("scenario physical configuration is missing")
    data["physical"] = PhysicalConfig(**physical)
    return ScenarioConfig(**data)


def _variant_from_dict(values: dict[str, object]) -> CAAPPOVariant:
    data = dict(values)
    kinds = data.get("construction_kinds")
    if isinstance(kinds, list):
        data["construction_kinds"] = tuple(str(kind) for kind in kinds)
    return CAAPPOVariant(**data)


def _training_contract(
    config: ConstructionExperimentConfig,
    variant: CAAPPOVariant,
    training_seed: int,
) -> dict[str, object]:
    return {
        "scenario": asdict(config.scenario),
        "variant": asdict(variant),
        "training_seed": int(training_seed),
        "training_seeds": tuple(int(seed) for seed in config.training_seeds),
        "validation_seeds": tuple(int(seed) for seed in config.validation_seeds),
        "validation_interval": int(config.validation_interval),
        "seed_protocol": "namespaced_cantor_pair_caappo_v1",
        "evaluation_seeds": tuple(int(seed) for seed in config.evaluation_seeds),
    }


def _training_metadata(
    config: ConstructionExperimentConfig,
    variant: CAAPPOVariant,
    training_seed: int,
) -> dict[str, object]:
    return _training_contract(config, variant, training_seed)


def _training_episode_seed(
    config: ConstructionExperimentConfig,
    training_seed: int,
    episode_index: int,
) -> int:
    if episode_index < 0:
        raise ValueError("episode_index cannot be negative")
    reserved = {
        *map(int, config.training_seeds),
        *map(int, config.validation_seeds),
        *map(int, config.evaluation_seeds),
    }
    total = int(training_seed) + int(episode_index)
    paired = total * (total + 1) // 2 + int(episode_index)
    seed = (1 << 63) + paired
    if seed in reserved:
        raise ValueError(
            "derived training episode seed overlaps a reserved protocol seed"
        )
    return seed


def _validation_record(
    config: ConstructionExperimentConfig,
    variant: CAAPPOVariant,
    trainer: TorchCAAPPORolloutTrainer,
    completed_episodes: int,
    *,
    selection_eligible: bool,
) -> dict[str, object]:
    rows: list[dict[str, float]] = []
    for seed in config.validation_seeds:
        spec = make_episode(config.scenario, seed)
        candidates = _catalogue(
            spec, variant.candidate_count, variant.construction_kinds
        )
        outcome = trainer.run_episode(
            spec, candidates, deterministic=True, update=False
        )
        metrics = _numeric_metrics(outcome.metrics)
        horizon_ps = max(
            float(spec.horizon * spec.physical.slot_duration_ps),
            1.0,
        )
        batch_size = max(len(spec.requests), 1)
        objective = (
            float(metrics["completion_rate"])
            - float(metrics["censored_flow_time_ps"]) / (batch_size * horizon_ps)
            - float(metrics["risk_count"]) / batch_size
        )
        rows.append({
            "seed": float(seed),
            "objective": objective,
            **metrics,
        })
    metric_names = sorted(
        set().union(*(row.keys() for row in rows)) - {"seed"}
    )
    means = {
        (name if name.startswith("mean_") else f"mean_{name}"): float(
            np.mean([row[name] for row in rows])
        )
        for name in metric_names
    }
    return {
        "event": "validation",
        "completed_episodes": int(completed_episodes),
        "selection_eligible": bool(selection_eligible),
        "seeds": tuple(int(seed) for seed in config.validation_seeds),
        **means,
    }


def _best_validation(
    current: dict[str, object] | None,
    candidate: dict[str, object],
    risk_limit: float,
) -> bool:
    if not bool(candidate.get("selection_eligible", True)):
        return False
    if current is None:
        return True
    candidate_feasible = float(candidate["mean_risk_count"]) <= risk_limit + 1e-12
    current_feasible = float(current["mean_risk_count"]) <= risk_limit + 1e-12
    if candidate_feasible != current_feasible:
        return candidate_feasible
    if not candidate_feasible:
        candidate_violation = float(candidate["mean_risk_count"]) - risk_limit
        current_violation = float(current["mean_risk_count"]) - risk_limit
        if abs(candidate_violation - current_violation) > 1e-12:
            return candidate_violation < current_violation
    return float(candidate["mean_objective"]) > float(current["mean_objective"])


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_training_report(run: CAAPPOTrainingRun) -> Path:
    report_path = run.checkpoint.with_suffix(run.checkpoint.suffix + ".json")
    _atomic_write_text(report_path, json.dumps({
        "checkpoint": str(run.checkpoint),
        "sha256": run.sha256,
        "variant": run.variant,
        "training_seed": run.training_seed,
        "completed_episodes": run.completed_episodes,
        "best_validation": run.best_validation,
        "history": run.history,
    }, indent=2, default=str))
    return report_path


def train_variant_checkpoint(
    config: ConstructionExperimentConfig,
    variant: CAAPPOVariant,
    training_seed: int,
    checkpoint: Path,
    *,
    resume: bool = False,
    strict_runtime: bool = True,
) -> CAAPPOTrainingRun:
    if training_seed not in config.training_seeds:
        raise ValueError("training_seed must belong to config.training_seeds")
    contract = _training_contract(config, variant, training_seed)
    if resume:
        loaded = load_caappo_checkpoint(
            checkpoint,
            expected_training_metadata=contract,
            strict_runtime=strict_runtime,
            restore_rng=True,
        )
        policy = loaded.policy
        trainer = loaded.trainer
        completed = loaded.completed_episodes
        history = list(loaded.history)
        best_validation = loaded.best_validation
        best_state = loaded.best_policy_state_dict
        best_optimizer_state = loaded.best_optimizer_state_dict
        best_lambda_risk = loaded.best_lambda_risk
    else:
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
            dynamic_repair_paths=variant.dynamic_repair_paths,
            dynamic_repair_construction_kinds=variant.construction_kinds,
        )
        completed = 0
        history: list[dict[str, object]] = []
        best_validation: dict[str, object] | None = None
        best_state: dict[str, object] | None = None
        best_optimizer_state: dict[str, object] | None = None
        best_lambda_risk: float | None = None
    if config.training_episodes < completed:
        raise ValueError(
            "target training_episodes is smaller than the checkpoint counter"
        )

    for episode_index in range(completed, config.training_episodes):
        episode_seed = _training_episode_seed(
            config, training_seed, episode_index
        )
        spec = make_episode(config.scenario, episode_seed)
        candidates = _catalogue(
            spec, variant.candidate_count, variant.construction_kinds
        )
        started = perf_counter()
        outcome = trainer.run_episode(
            spec, candidates, deterministic=False, update=True
        )
        completed = episode_index + 1
        history.append({
            "event": "training_episode",
            "episode_index": episode_index,
            "episode_seed": episode_seed,
            "reward": outcome.reward,
            "discounted_return": outcome.discounted_return,
            "metrics": _numeric_metrics(outcome.metrics),
            "update_stats": (
                None if outcome.update_stats is None else asdict(outcome.update_stats)
            ),
            "wall_seconds": perf_counter() - started,
        })
        if (
            completed % config.validation_interval == 0
            or completed == config.training_episodes
        ):
            validation = _validation_record(
                config,
                variant,
                trainer,
                completed,
                selection_eligible=completed % config.validation_interval == 0,
            )
            history.append(validation)
            if _best_validation(best_validation, validation, variant.risk_limit):
                best_validation = validation
                best_state = deepcopy(policy.state_dict())
                best_optimizer_state = deepcopy(policy.optimizer.state_dict())
                best_lambda_risk = float(policy.lambda_risk)
        save_caappo_checkpoint(
            checkpoint,
            policy=policy,
            trainer=trainer,
            training_metadata=_training_metadata(config, variant, training_seed),
            completed_episodes=completed,
            history=history,
            best_validation=best_validation,
            best_policy_state_dict=best_state,
            best_optimizer_state_dict=best_optimizer_state,
            best_lambda_risk=best_lambda_risk,
        )

    if (not resume and completed == 0) or not checkpoint.exists():
        save_caappo_checkpoint(
            checkpoint,
            policy=policy,
            trainer=trainer,
            training_metadata=_training_metadata(config, variant, training_seed),
            completed_episodes=completed,
            history=history,
            best_validation=best_validation,
            best_policy_state_dict=best_state,
            best_optimizer_state_dict=best_optimizer_state,
            best_lambda_risk=best_lambda_risk,
        )
    run = CAAPPOTrainingRun(
        checkpoint,
        variant.name,
        int(training_seed),
        completed,
        best_validation,
        tuple(history),
        checkpoint_sha256(checkpoint),
    )
    _write_training_report(run)
    return run


def evaluate_checkpoint(
    checkpoint: Path,
    evaluation_seeds: tuple[int, ...] | None = None,
    *,
    strict_runtime: bool = True,
    use_best: bool = True,
    expected_sha256: str | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    loaded = load_caappo_checkpoint(
        checkpoint,
        strict_runtime=strict_runtime,
        restore_rng=False,
        use_best=use_best,
        expected_sha256=expected_sha256,
    )
    metadata_values = loaded.training_metadata
    scenario_values = metadata_values.get("scenario")
    variant_values = metadata_values.get("variant")
    if not isinstance(scenario_values, dict) or not isinstance(variant_values, dict):
        raise CheckpointCompatibilityError(
            "checkpoint scenario or variant metadata is missing"
        )
    scenario = _scenario_from_dict(scenario_values)
    variant = _variant_from_dict(variant_values)
    seeds = (
        tuple(int(seed) for seed in metadata_values.get("evaluation_seeds", ()))
        if evaluation_seeds is None
        else tuple(int(seed) for seed in evaluation_seeds)
    )
    if not seeds:
        raise ValueError("evaluation seeds must be provided")
    if len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seeds must be unique")
    reserved = {
        int(metadata_values["training_seed"]),
        *(int(seed) for seed in metadata_values.get("validation_seeds", ())),
        *(
            int(row["episode_seed"])
            for row in loaded.history
            if row.get("event") == "training_episode"
        ),
    }
    overlap = reserved.intersection(seeds)
    if overlap:
        raise ValueError(
            f"evaluation seeds overlap training or validation seeds: {sorted(overlap)}"
        )

    frozen_before = {
        name: value.detach().cpu().clone()
        for name, value in loaded.policy.state_dict().items()
    }
    lambda_before = float(loaded.policy.lambda_risk)
    rows: list[dict[str, object]] = []
    for seed in seeds:
        spec = make_episode(scenario, int(seed))
        candidates = _catalogue(
            spec, variant.candidate_count, variant.construction_kinds
        )
        started = perf_counter()
        outcome = loaded.trainer.run_episode(
            spec, candidates, deterministic=True, update=False
        )
        rows.append({
            "method": "torch_caappo",
            "variant": variant.name,
            "seed": int(seed),
            "training_seed": int(metadata_values["training_seed"]),
            **_numeric_metrics(outcome.metrics),
            "episode_reward": outcome.reward,
            "discounted_return": outcome.discounted_return,
            "checkpoint_sha256": checkpoint_sha256(checkpoint),
            "checkpoint_state": "best" if use_best and loaded.best_policy_state_dict else "final",
            "wall_seconds": perf_counter() - started,
        })
    for name, value in loaded.policy.state_dict().items():
        if not np.array_equal(
            value.detach().cpu().numpy(), frozen_before[name].numpy()
        ):
            raise RuntimeError("evaluation mutated the frozen policy state")
    if loaded.policy.lambda_risk != lambda_before:
        raise RuntimeError("evaluation mutated the CMDP dual state")
    run = {
        "checkpoint": str(checkpoint),
        "sha256": checkpoint_sha256(checkpoint),
        "variant": variant.name,
        "training_seed": int(metadata_values["training_seed"]),
        "completed_episodes": loaded.completed_episodes,
        "selected_state": (
            "best" if use_best and loaded.best_policy_state_dict else "final"
        ),
        "best_validation": loaded.best_validation,
        "evaluation_seeds": tuple(int(seed) for seed in seeds),
    }
    return rows, run


def _run_caappo(
    config: ConstructionExperimentConfig,
    checkpoint_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    for variant in config.variants:
        for training_seed in config.training_seeds:
            checkpoint = checkpoint_dir / f"{variant.name}.seed-{training_seed}.pt"
            started = perf_counter()
            training_run = train_variant_checkpoint(
                config, variant, training_seed, checkpoint
            )
            training_seconds = perf_counter() - started
            evaluation_rows, evaluation_run = evaluate_checkpoint(
                checkpoint, config.evaluation_seeds
            )
            for row in evaluation_rows:
                row["training_seconds"] = training_seconds
            rows.extend(evaluation_rows)
            runs.append({
                **evaluation_run,
                "history_file": str(
                    checkpoint.with_suffix(checkpoint.suffix + ".json")
                ),
                "history_events": len(training_run.history),
                "training_seconds": training_seconds,
            })
    return rows, runs


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


def _manifest(
    config: ConstructionExperimentConfig,
    checkpoint_runs: Iterable[dict[str, object]] = (),
) -> dict[str, object]:
    runtime = checkpoint_runtime_manifest()
    packages = dict(runtime["packages"])
    try:
        packages["scipy"] = metadata.version("scipy")
    except metadata.PackageNotFoundError:
        packages["scipy"] = "not-installed"
    return {
        "python": runtime["python"],
        "platform": runtime["platform"],
        "packages": packages,
        "physical_backend": "SeQUeNCe",
        "seed_protocol": {
            "training": "injective per-replica Cantor-paired episode seeds",
            "validation": "held out for checkpoint selection",
            "evaluation": "held out for final reporting and CI",
        },
        "checkpoint_runs": list(checkpoint_runs),
        "deterministic_oracle_role": "small-instance nominal validation only",
        "confidence_interval": (
            "95% normal interval over evaluation seeds; training replicas are "
            "averaged within seed; ci_supported=false when n<2"
        ),
        "ci_estimand": (
            "expected performance over training randomness on held-out "
            "evaluation seeds"
        ),
        "training_replica_uncertainty": (
            "reported separately in training_replica_aggregate"
        ),
        "legacy_baseline_note": (
            "QDDCA and QCAST reproductions remain available separately; their "
            "legacy action spaces are not mixed into this construction-SMDP table."
        ),
        "config": asdict(config),
    }


def run_experiment(
    config: ConstructionExperimentConfig,
    checkpoint_dir: Path | None = None,
) -> dict[str, object]:
    ephemeral = checkpoint_dir is None
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if checkpoint_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="caappo-checkpoints-")
        checkpoint_dir = Path(temporary.name)
    try:
        caappo_rows, checkpoint_runs = _run_caappo(config, checkpoint_dir)
    finally:
        if temporary is not None:
            temporary.cleanup()
    if ephemeral:
        for checkpoint_run in checkpoint_runs:
            checkpoint_run.pop("checkpoint", None)
            checkpoint_run.pop("history_file", None)
            checkpoint_run["ephemeral_checkpoint"] = True
    rows = _run_baselines(config) + caappo_rows
    if config.include_nominal_oracle:
        rows += _run_nominal_oracle(config)
    return {
        "manifest": _manifest(config, checkpoint_runs),
        "catalogue_coverage": _catalogue_coverage(config),
        "rows": rows,
        "aggregate": _aggregate(rows),
        "training_replica_aggregate": _aggregate_training_replicas(rows),
        "paired_differences": _paired_differences(rows),
    }


def write_results(result: dict[str, object], output: Path) -> tuple[Path, Path]:
    _atomic_write_text(output, json.dumps(result, indent=2, default=str))
    csv_path = output.with_suffix(".csv")
    temporary_csv = csv_path.with_name(f".{csv_path.name}.tmp")
    rows = list(result["rows"])
    fieldnames = sorted({key for row in rows for key in row})
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(csv_path)
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
        validation_seeds=(51, 52),
        training_episodes=2,
        validation_interval=1,
        candidate_count=2,
        variants=(CAAPPOVariant("caappo", candidate_count=2),),
    )


def _default_config() -> ConstructionExperimentConfig:
    return replace(
        _quick_config(),
        evaluation_seeds=(101, 102, 103, 104, 105),
        training_seeds=(1, 2, 3, 4, 5),
        validation_seeds=(51, 52, 53, 54, 55),
        training_episodes=30,
        validation_interval=5,
        candidate_count=3,
        variants=ConstructionExperimentConfig.__dataclass_fields__["variants"].default,
    )


def experiment_config_from_dict(
    values: dict[str, object],
) -> ConstructionExperimentConfig:
    data = dict(values)
    scenario = data.get("scenario")
    if not isinstance(scenario, dict):
        raise ValueError("experiment config must contain a scenario mapping")
    data["scenario"] = _scenario_from_dict(scenario)
    variants = data.get("variants")
    if variants is not None:
        if not isinstance(variants, list):
            raise ValueError("experiment variants must be a list")
        data["variants"] = tuple(_variant_from_dict(value) for value in variants)
    for name in ("evaluation_seeds", "training_seeds", "validation_seeds"):
        if name in data:
            data[name] = tuple(int(seed) for seed in data[name])
    return ConstructionExperimentConfig(**data)


def load_experiment_config(path: Path) -> ConstructionExperimentConfig:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("experiment config JSON must contain an object")
    if (
        "manifest" in values
        and isinstance(values["manifest"], dict)
        and isinstance(values["manifest"].get("config"), dict)
    ):
        values = values["manifest"]["config"]
    if "config" in values and isinstance(values["config"], dict):
        values = values["config"]
    return experiment_config_from_dict(values)


def _config_from_args(args: argparse.Namespace) -> ConstructionExperimentConfig:
    config_path = getattr(args, "config", None)
    if config_path is not None:
        return load_experiment_config(config_path)
    return _quick_config() if getattr(args, "quick", False) else _default_config()


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--quick", action="store_true")
    profile.add_argument("--config", type=Path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="train checkpoints, evaluate them, and run baselines"
    )
    _add_config_arguments(run_parser)
    run_parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/construction_aware.json"),
    )
    run_parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("results/checkpoints"),
    )

    train_parser = subparsers.add_parser(
        "train", help="train or resume one variant checkpoint"
    )
    _add_config_arguments(train_parser)
    train_parser.add_argument("--variant", default="caappo")
    train_parser.add_argument("--training-seed", type=int)
    train_parser.add_argument("--episodes", type=int)
    train_parser.add_argument("--checkpoint", type=Path)
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--allow-runtime-mismatch", action="store_true")

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate a frozen checkpoint"
    )
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--evaluation-seeds", type=int, nargs="+")
    evaluate_parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/construction_aware_evaluation.json"),
    )
    evaluate_parser.add_argument("--final-state", action="store_true")
    evaluate_parser.add_argument("--expected-sha256")
    evaluate_parser.add_argument("--allow-runtime-mismatch", action="store_true")

    raw_args = list(argv) if argv is not None else list(sys.argv[1:])
    if not raw_args:
        raw_args.insert(0, "run")
    elif raw_args[0] not in {"run", "train", "evaluate", "-h", "--help"}:
        raw_args.insert(0, "run")
    args = parser.parse_args(raw_args)

    if args.command == "train":
        config = _config_from_args(args)
        if args.episodes is not None:
            config = replace(config, training_episodes=args.episodes)
        variant = next(
            (value for value in config.variants if value.name == args.variant),
            None,
        )
        if variant is None:
            raise ValueError(f"unknown CAAPPO variant: {args.variant}")
        training_seed = (
            config.training_seeds[0]
            if args.training_seed is None
            else args.training_seed
        )
        checkpoint = args.checkpoint or Path(
            f"results/checkpoints/{variant.name}.seed-{training_seed}.pt"
        )
        run = train_variant_checkpoint(
            config,
            variant,
            training_seed,
            checkpoint,
            resume=args.resume,
            strict_runtime=not args.allow_runtime_mismatch,
        )
        print(json.dumps({
            "checkpoint": str(run.checkpoint),
            "sha256": run.sha256,
            "history": str(
                run.checkpoint.with_suffix(run.checkpoint.suffix + ".json")
            ),
            "completed_episodes": run.completed_episodes,
            "best_validation": run.best_validation,
        }, indent=2, default=str))
        return 0

    if args.command == "evaluate":
        seeds = (
            None
            if args.evaluation_seeds is None
            else tuple(args.evaluation_seeds)
        )
        rows, checkpoint_run = evaluate_checkpoint(
            args.checkpoint,
            seeds,
            strict_runtime=not args.allow_runtime_mismatch,
            use_best=not args.final_state,
            expected_sha256=args.expected_sha256,
        )
        result = {
            "manifest": {
                **checkpoint_runtime_manifest(),
                "mode": "frozen_checkpoint_evaluation",
                "physical_backend": "SeQUeNCe",
                "checkpoint_run": checkpoint_run,
            },
            "rows": rows,
            "aggregate": _aggregate(rows),
            "training_replica_aggregate": _aggregate_training_replicas(rows),
            "paired_differences": [],
        }
        json_path, csv_path = write_results(result, args.output)
    else:
        config = _config_from_args(args)
        result = run_experiment(config, args.checkpoint_dir)
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
    "CAAPPOTrainingRun",
    "ConstructionExperimentConfig",
    "evaluate_checkpoint",
    "experiment_config_from_dict",
    "load_experiment_config",
    "run_experiment",
    "train_variant_checkpoint",
    "write_results",
]
