"""Paired online evaluation for ARC-Q and non-optimization baselines."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

from algorithms.baselines.online import (
    OnlineBaselineConfig,
    run_online_baseline,
)
from algorithms.qcast.online import OnlineQCASTConfig, run_online_qcast
from algorithms.routing_core.execution import OnlineExecutionConfig
from qnet_core.scenario import ScenarioConfig, make_episode

from .policy import ARCQPolicy
from .rollout import collect_episode


@dataclass(frozen=True)
class BaselineDefinition:
    name: str
    algorithm: str
    path_candidate_count: int
    construction_kind: str = "balanced"
    swap_tree_count: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("baseline name must be non-empty")
        if self.path_candidate_count < 1:
            raise ValueError("path_candidate_count must be positive")
        if self.construction_kind not in {"left_deep", "balanced"}:
            raise ValueError("unsupported construction kind")


@dataclass(frozen=True)
class EvaluationRecord:
    scenario: str
    method: str
    episode_seed: int
    topology_seed: int
    metrics: Mapping[str, float]


def default_baselines(
    environment_config: OnlineExecutionConfig,
) -> tuple[BaselineDefinition, ...]:
    """Return mechanism-level baselines without an optimization oracle."""

    construction_count = environment_config.swap_tree_count or 2
    return (
        BaselineDefinition("Greedy", "greedy", 1, "balanced"),
        BaselineDefinition(
            "Path-only",
            "qpass",
            environment_config.path_candidate_count,
            "balanced",
        ),
        BaselineDefinition(
            "Construction-only",
            "construction_only",
            1,
            "balanced",
            construction_count,
        ),
        BaselineDefinition(
            "Q-LEAP",
            "qleap",
            environment_config.path_candidate_count,
            "balanced",
        ),
        BaselineDefinition(
            "Q-CAST",
            "qcast",
            environment_config.path_candidate_count,
            "left_deep",
        ),
    )


def run_paired_evaluation(
    policy: ARCQPolicy,
    *,
    scenario_name: str,
    scenario_config: ScenarioConfig,
    environment_config: OnlineExecutionConfig,
    episode_seeds: Sequence[int],
    topology_seed: int,
    baselines: Sequence[BaselineDefinition] | None = None,
) -> tuple[EvaluationRecord, ...]:
    """Run every method on identical EpisodeSpec values and independent backends."""

    if not scenario_name:
        raise ValueError("scenario_name must be non-empty")
    seeds = tuple(int(seed) for seed in episode_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("episode seeds must be non-empty and unique")
    definitions = tuple(
        default_baselines(environment_config)
        if baselines is None
        else baselines
    )
    names = [definition.name for definition in definitions]
    if len(set(names)) != len(names) or "ARC-Q" in names:
        raise ValueError("evaluation method names must be unique")

    records: list[EvaluationRecord] = []
    for episode_seed in seeds:
        episode = make_episode(
            scenario_config,
            episode_seed,
            topology_seed=topology_seed,
        )
        arcq = collect_episode(
            policy,
            episode,
            environment_config,
            deterministic=True,
            collect_value_estimates=False,
        )
        arcq_metrics = dict(arcq.execution.metrics)
        arcq_metrics["reward_identity_error"] = float(
            arcq.reward_identity_error
        )
        records.append(EvaluationRecord(
            scenario=scenario_name,
            method="ARC-Q",
            episode_seed=episode_seed,
            topology_seed=topology_seed,
            metrics=arcq_metrics,
        ))
        for definition in definitions:
            if definition.algorithm == "qcast":
                result = run_online_qcast(
                    episode,
                    OnlineQCASTConfig(
                        decision_interval=(
                            environment_config.decision_interval
                        ),
                        path_candidate_count=(
                            definition.path_candidate_count
                        ),
                        construction_kind=definition.construction_kind,
                        purification_kind="none",
                    ),
                )
            else:
                result = run_online_baseline(
                    episode,
                    OnlineBaselineConfig(
                        algorithm=definition.algorithm,
                        decision_interval=(
                            environment_config.decision_interval
                        ),
                        path_candidate_count=(
                            definition.path_candidate_count
                        ),
                        construction_kind=definition.construction_kind,
                        swap_tree_count=definition.swap_tree_count,
                    ),
                )
            records.append(EvaluationRecord(
                scenario=scenario_name,
                method=definition.name,
                episode_seed=episode_seed,
                topology_seed=topology_seed,
                metrics=dict(result.metrics),
            ))
    return tuple(records)


def save_evaluation_records(
    records: Sequence[EvaluationRecord],
    output_stem: str | Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> tuple[Path, Path]:
    """Record raw evaluation data; plotting is intentionally separate."""

    ordered = tuple(sorted(
        records,
        key=lambda item: (item.scenario, item.episode_seed, item.method),
    ))
    if not ordered:
        raise ValueError("at least one evaluation record is required")
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = stem.with_suffix(".json")
    csv_path = stem.with_suffix(".csv")
    payload = {
        "schema_version": 1,
        "method_under_test": "ARC-Q",
        "metadata": dict(metadata or {}),
        "records": [asdict(record) for record in ordered],
    }
    temporary_json = json_path.with_suffix(".json.tmp")
    temporary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_json.replace(json_path)

    metric_names = sorted({
        metric
        for record in ordered
        for metric in record.metrics
    })
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "scenario",
                "method",
                "episode_seed",
                "topology_seed",
                *metric_names,
            ),
        )
        writer.writeheader()
        for record in ordered:
            writer.writerow({
                "scenario": record.scenario,
                "method": record.method,
                "episode_seed": record.episode_seed,
                "topology_seed": record.topology_seed,
                **record.metrics,
            })
    temporary_csv.replace(csv_path)
    return json_path, csv_path


__all__ = [
    "BaselineDefinition",
    "EvaluationRecord",
    "default_baselines",
    "run_paired_evaluation",
    "save_evaluation_records",
]
