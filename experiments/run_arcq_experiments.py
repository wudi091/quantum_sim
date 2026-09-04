"""Run resumable ARC-Q paper experiments without producing figures."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping, Sequence

import torch
import yaml

from algorithms.rl_routing.checkpoint import load_arcq_checkpoint
from algorithms.rl_routing.evaluation import (
    BaselineDefinition,
    run_paired_evaluation,
)
from algorithms.routing_core.execution import OnlineExecutionConfig
from qnet_core.scenario import ScenarioConfig
from qnet_core.spec import PhysicalConfig


RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SweepPoint:
    point_id: str
    value: int | float | str
    topology_seeds: tuple[int, ...]
    scenario_overrides: Mapping[str, object]


@dataclass(frozen=True)
class ExperimentSuite:
    suite_id: str
    x_label: str
    points: tuple[SweepPoint, ...]


@dataclass(frozen=True)
class ExperimentProtocol:
    checkpoint_path: Path
    output_path: Path
    episode_seed_start: int
    episodes_per_topology: int
    environment: OnlineExecutionConfig
    base_scenario: ScenarioConfig
    baselines: tuple[BaselineDefinition, ...]
    suites: tuple[ExperimentSuite, ...]
    raw_config: Mapping[str, object]
    fingerprint: str

    @property
    def episode_seeds(self) -> tuple[int, ...]:
        return tuple(range(
            self.episode_seed_start,
            self.episode_seed_start + self.episodes_per_topology,
        ))


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _sequence(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return list(value)


def _protocol_fingerprint(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_scenario(values: Mapping[str, object]) -> ScenarioConfig:
    scenario_values = dict(values)
    physical_values = _mapping(
        scenario_values.pop("physical", {}),
        "base_scenario.physical",
    )
    return ScenarioConfig(
        **scenario_values,
        physical=PhysicalConfig(**physical_values),
    )


def load_experiment_protocol(path: str | Path) -> ExperimentProtocol:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = _mapping(payload, "experiment protocol")
    if root.get("schema_version") != 1:
        raise ValueError("unsupported experiment protocol schema")
    required = {
        "schema_version",
        "checkpoint_path",
        "output_path",
        "replication",
        "environment",
        "base_scenario",
        "baselines",
        "suites",
    }
    unknown = set(root) - required
    missing = required - set(root)
    if unknown:
        raise ValueError(f"unknown protocol field: {sorted(unknown)[0]}")
    if missing:
        raise ValueError(f"missing protocol field: {sorted(missing)[0]}")

    replication = _mapping(root["replication"], "replication")
    if set(replication) != {
        "episode_seed_start",
        "episodes_per_topology",
    }:
        raise ValueError("replication fields do not match the schema")
    episodes_per_topology = int(replication["episodes_per_topology"])
    if episodes_per_topology < 1:
        raise ValueError("episodes_per_topology must be positive")

    environment_values = _mapping(root["environment"], "environment")
    for field in ("construction_kinds", "purification_kinds"):
        if field in environment_values:
            environment_values[field] = tuple(environment_values[field])
    environment = OnlineExecutionConfig(**environment_values)

    baselines = tuple(
        BaselineDefinition(**_mapping(item, "baseline"))
        for item in _sequence(root["baselines"], "baselines")
    )
    baseline_names = tuple(item.name for item in baselines)
    if len(baseline_names) != len(set(baseline_names)):
        raise ValueError("baseline names must be unique")
    supported_algorithms = {
        "greedy",
        "construction_only",
        "strict_fifo",
        "best_fifo",
        "qpass",
        "qpath",
        "qleap",
        "qcast",
    }
    unsupported = [
        item.algorithm
        for item in baselines
        if item.algorithm not in supported_algorithms
    ]
    if unsupported:
        raise ValueError(f"unsupported baseline algorithm: {unsupported[0]}")

    suites: list[ExperimentSuite] = []
    for raw_suite in _sequence(root["suites"], "suites"):
        suite = _mapping(raw_suite, "suite")
        if set(suite) != {"id", "x_label", "points"}:
            raise ValueError("suite fields do not match the schema")
        points: list[SweepPoint] = []
        for raw_point in _sequence(suite["points"], "suite.points"):
            point = _mapping(raw_point, "point")
            if set(point) != {"id", "value", "topology_seeds", "scenario"}:
                raise ValueError("point fields do not match the schema")
            topology_seeds = tuple(
                int(item)
                for item in _sequence(
                    point["topology_seeds"],
                    "point.topology_seeds",
                )
            )
            if not topology_seeds or len(topology_seeds) != len(
                set(topology_seeds)
            ):
                raise ValueError("topology seeds must be non-empty and unique")
            points.append(SweepPoint(
                point_id=str(point["id"]),
                value=point["value"],
                topology_seeds=topology_seeds,
                scenario_overrides=_mapping(point["scenario"], "point.scenario"),
            ))
        if len(points) < 5:
            raise ValueError("every formal suite must contain at least five points")
        suites.append(ExperimentSuite(
            suite_id=str(suite["id"]),
            x_label=str(suite["x_label"]),
            points=tuple(points),
        ))
    suite_ids = tuple(item.suite_id for item in suites)
    if not suites or len(suite_ids) != len(set(suite_ids)):
        raise ValueError("suite IDs must be non-empty and unique")

    raw_config: Mapping[str, object] = root
    return ExperimentProtocol(
        checkpoint_path=Path(str(root["checkpoint_path"])),
        output_path=Path(str(root["output_path"])),
        episode_seed_start=int(replication["episode_seed_start"]),
        episodes_per_topology=episodes_per_topology,
        environment=environment,
        base_scenario=_parse_scenario(
            _mapping(root["base_scenario"], "base_scenario")
        ),
        baselines=baselines,
        suites=tuple(suites),
        raw_config=raw_config,
        fingerprint=_protocol_fingerprint(raw_config),
    )


def scenario_for_point(
    base: ScenarioConfig,
    point: SweepPoint,
) -> ScenarioConfig:
    values = asdict(base)
    physical_values = dict(values.pop("physical"))
    overrides = dict(point.scenario_overrides)
    physical_overrides = _mapping(
        overrides.pop("physical", {}),
        f"{point.point_id}.physical",
    )
    unknown_scenario = set(overrides) - set(values)
    unknown_physical = set(physical_overrides) - set(physical_values)
    if unknown_scenario:
        raise ValueError(
            f"unknown scenario override: {sorted(unknown_scenario)[0]}"
        )
    if unknown_physical:
        raise ValueError(
            f"unknown physical override: {sorted(unknown_physical)[0]}"
        )
    values.update(overrides)
    physical_values.update(physical_overrides)
    return ScenarioConfig(
        **values,
        physical=PhysicalConfig(**physical_values),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_provenance(*, allow_dirty: bool) -> dict[str, object]:
    def git(*arguments: str) -> bytes:
        try:
            process = subprocess.run(
                ("git", *arguments),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError("formal experiments require a Git worktree") from exc
        return process.stdout

    revision = git("rev-parse", "HEAD").decode("ascii").strip()
    status = git("status", "--porcelain", "--untracked-files=no")
    clean = not bool(status.strip())
    if not clean and not allow_dirty:
        raise RuntimeError(
            "formal experiments refuse uncommitted tracked code changes"
        )
    diff = git("diff", "--binary", "HEAD")
    return {
        "git_revision": revision,
        "tracked_worktree_clean": clean,
        "tracked_diff_sha256": (
            None if clean else hashlib.sha256(diff).hexdigest()
        ),
    }


def _runtime_provenance() -> dict[str, object]:
    packages: dict[str, str] = {}
    for package in (
        "sequence",
        "numpy",
        "networkx",
        "scipy",
        "torch",
        "PyYAML",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _ranges_overlap(left: range, right: range) -> bool:
    return max(left.start, right.start) < min(left.stop, right.stop)


def _checkpoint_evaluation_provenance(
    protocol: ExperimentProtocol,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Validate model selection and train/test separation before evaluation."""

    training_state = _mapping(
        metadata.get("training_state"),
        "checkpoint training_state",
    )
    if training_state.get("selection_finalized") is not True:
        raise ValueError(
            "formal evaluation requires a best checkpoint finalized after "
            "the complete training run"
        )
    stored_config = _mapping(
        training_state.get("config"),
        "checkpoint training config",
    )
    stored_environment = _mapping(
        stored_config.get("environment"),
        "checkpoint environment config",
    )
    if stored_environment != asdict(protocol.environment):
        raise ValueError(
            "formal evaluation environment differs from the training action "
            "space"
        )
    stored_run = _mapping(
        stored_config.get("run"),
        "checkpoint run config",
    )
    training_episode_count = int(stored_run["episode_count"])
    training_seed_start = int(stored_run["random_seed"]) + 1
    validation_seed_start = int(stored_run["validation_seed"])
    validation_episode_count = int(stored_run["validation_episode_count"])
    training_seeds = range(
        training_seed_start,
        training_seed_start + training_episode_count,
    )
    validation_seeds = range(
        validation_seed_start,
        validation_seed_start + validation_episode_count,
    )
    evaluation_seeds = range(
        protocol.episode_seed_start,
        protocol.episode_seed_start + protocol.episodes_per_topology,
    )
    if _ranges_overlap(training_seeds, evaluation_seeds):
        raise ValueError("formal episode seeds overlap training seeds")
    if _ranges_overlap(validation_seeds, evaluation_seeds):
        raise ValueError("formal episode seeds overlap validation seeds")

    training_topology_seed = int(
        training_state["fixed_training_topology_seed"]
    )
    evaluation_topology_seeds = {
        seed
        for suite in protocol.suites
        for point in suite.points
        for seed in point.topology_seeds
    }
    if training_topology_seed in evaluation_topology_seeds:
        raise ValueError("formal topology seeds include the training topology")
    best_update = int(training_state["final_best_validation_update"])
    if int(training_state["update"]) != best_update:
        raise ValueError("checkpoint is not the selected best validation model")
    if training_state.get("model_selection_metric") != (
        "mean_censored_latency_slots"
    ):
        raise ValueError("checkpoint used another model-selection metric")
    completed = int(training_state["training_completed_episodes"])
    if completed != training_episode_count:
        raise ValueError("checkpoint model selection was not fully finalized")
    return {
        "checkpoint_schema_version": int(metadata["schema_version"]),
        "model": dict(_mapping(metadata["model"], "checkpoint model")),
        "training_episode_count": training_episode_count,
        "training_seed_start": training_seeds.start,
        "training_seed_stop_exclusive": training_seeds.stop,
        "validation_seed_start": validation_seeds.start,
        "validation_seed_stop_exclusive": validation_seeds.stop,
        "training_topology_seed": training_topology_seed,
        "best_validation_update": best_update,
        "best_validation_latency_slots": float(
            training_state["final_best_validation_latency_slots"]
        ),
        "model_selection_metric": training_state["model_selection_metric"],
        "selection_finalized": True,
    }


def _record_key(record: Mapping[str, object]) -> tuple[object, ...]:
    return (
        record["suite"],
        record["point_id"],
        record["topology_seed"],
        record["episode_seed"],
        record["method"],
    )


def _write_results(
    path: Path,
    payload: Mapping[str, object],
) -> tuple[Path, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)

    csv_path = path.with_suffix(".csv")
    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    records = list(payload["records"])
    metric_names = sorted({
        name
        for record in records
        for name in record["metrics"]
    })
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "suite",
            "point_id",
            "point_value",
            "topology_seed",
            "episode_seed",
            "method",
            *metric_names,
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "suite": record["suite"],
                "point_id": record["point_id"],
                "point_value": record["point_value"],
                "topology_seed": record["topology_seed"],
                "episode_seed": record["episode_seed"],
                "method": record["method"],
                **record["metrics"],
            })
    temporary_csv.replace(csv_path)
    return path, csv_path


def _initial_payload(
    protocol: ExperimentProtocol,
    checkpoint_hash: str,
    checkpoint_provenance: Mapping[str, object],
    repository_provenance: Mapping[str, object],
    runtime_provenance: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "method_under_test": "ARC-Q",
        "protocol_fingerprint": protocol.fingerprint,
        "checkpoint_path": str(protocol.checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_provenance": dict(checkpoint_provenance),
        "repository_provenance": dict(repository_provenance),
        "runtime_provenance": dict(runtime_provenance),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol.raw_config,
        "records": [],
    }


def _load_or_initialize_results(
    protocol: ExperimentProtocol,
    checkpoint_hash: str,
    checkpoint_provenance: Mapping[str, object],
    repository_provenance: Mapping[str, object],
    runtime_provenance: Mapping[str, object],
) -> dict[str, object]:
    if not protocol.output_path.exists():
        return _initial_payload(
            protocol,
            checkpoint_hash,
            checkpoint_provenance,
            repository_provenance,
            runtime_provenance,
        )
    payload = json.loads(protocol.output_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("existing result file has another schema")
    if payload.get("protocol_fingerprint") != protocol.fingerprint:
        raise ValueError("existing result file uses another protocol")
    if payload.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("existing result file uses another checkpoint")
    if payload.get("checkpoint_provenance") != dict(checkpoint_provenance):
        raise ValueError("existing result file uses other checkpoint provenance")
    if payload.get("repository_provenance") != dict(repository_provenance):
        raise ValueError("existing result file uses another code revision")
    if payload.get("runtime_provenance") != dict(runtime_provenance):
        raise ValueError("existing result file uses another runtime environment")
    if not isinstance(payload.get("records"), list):
        raise ValueError("existing result records must be a list")
    return payload


def _validate_execution_record(record: Mapping[str, object]) -> None:
    metrics = _mapping(record["metrics"], "record.metrics")
    if float(metrics.get("schedule_violation_count", -1.0)) != 0.0:
        raise RuntimeError("an experiment produced a schedule violation")
    if float(metrics.get("physical_backend_rejection_count", -1.0)) != 0.0:
        raise RuntimeError("an experiment produced a backend rejection")
    if record["method"] == "ARC-Q" and abs(
        float(metrics.get("reward_identity_error", float("inf")))
    ) > 1e-8:
        raise RuntimeError("ARC-Q reward identity check failed")


def run_experiments(
    protocol: ExperimentProtocol,
    *,
    suite_ids: Sequence[str] | None = None,
    max_instances: int | None = None,
    device: str = "cpu",
    allow_dirty: bool = False,
) -> tuple[Path, Path]:
    if max_instances is not None and max_instances < 1:
        raise ValueError("max_instances must be positive")
    selected_ids = (
        {suite.suite_id for suite in protocol.suites}
        if suite_ids is None
        else {str(item) for item in suite_ids}
    )
    known_ids = {suite.suite_id for suite in protocol.suites}
    unknown_ids = selected_ids - known_ids
    if unknown_ids:
        raise ValueError(f"unknown experiment suite: {sorted(unknown_ids)[0]}")
    if not protocol.checkpoint_path.is_file():
        raise FileNotFoundError(protocol.checkpoint_path)
    resolved_device = torch.device(device)
    policy, metadata = load_arcq_checkpoint(
        protocol.checkpoint_path,
        device=resolved_device,
    )
    policy.eval()
    checkpoint_provenance = _checkpoint_evaluation_provenance(
        protocol,
        metadata,
    )
    checkpoint_hash = _sha256(protocol.checkpoint_path)
    repository_provenance = _repository_provenance(
        allow_dirty=allow_dirty,
    )
    runtime_provenance = _runtime_provenance()
    payload = _load_or_initialize_results(
        protocol,
        checkpoint_hash,
        checkpoint_provenance,
        repository_provenance,
        runtime_provenance,
    )
    records = list(payload["records"])
    expected_methods = {
        "ARC-Q",
        *(baseline.name for baseline in protocol.baselines),
    }
    completed_keys = {_record_key(record) for record in records}
    newly_run = 0

    for suite in protocol.suites:
        if suite.suite_id not in selected_ids:
            continue
        for point in suite.points:
            scenario = scenario_for_point(protocol.base_scenario, point)
            for topology_seed in point.topology_seeds:
                for episode_seed in protocol.episode_seeds:
                    instance_prefix = (
                        suite.suite_id,
                        point.point_id,
                        topology_seed,
                        episode_seed,
                    )
                    present_methods = {
                        key[-1]
                        for key in completed_keys
                        if key[:-1] == instance_prefix
                    }
                    if present_methods == expected_methods:
                        continue
                    if max_instances is not None and newly_run >= max_instances:
                        return _write_results(protocol.output_path, payload)
                    paired = run_paired_evaluation(
                        policy,
                        scenario_name=f"{suite.suite_id}:{point.point_id}",
                        scenario_config=scenario,
                        environment_config=protocol.environment,
                        episode_seeds=(episode_seed,),
                        topology_seed=topology_seed,
                        baselines=protocol.baselines,
                    )
                    records = [
                        record
                        for record in records
                        if _record_key(record)[:-1] != instance_prefix
                    ]
                    for item in paired:
                        record = {
                            "suite": suite.suite_id,
                            "point_id": point.point_id,
                            "point_value": point.value,
                            "topology_seed": topology_seed,
                            "episode_seed": episode_seed,
                            "method": item.method,
                            "metrics": {
                                name: float(value)
                                for name, value in item.metrics.items()
                            },
                        }
                        _validate_execution_record(record)
                        records.append(record)
                    records.sort(key=_record_key)
                    payload["records"] = records
                    payload["updated_at_utc"] = datetime.now(
                        timezone.utc
                    ).isoformat()
                    _write_results(protocol.output_path, payload)
                    completed_keys = {_record_key(record) for record in records}
                    newly_run += 1
                    print(json.dumps({
                        "suite": suite.suite_id,
                        "point": point.point_id,
                        "topology_seed": topology_seed,
                        "episode_seed": episode_seed,
                        "completed_instances_this_run": newly_run,
                    }), flush=True)
    return _write_results(protocol.output_path, payload)


def protocol_summary(protocol: ExperimentProtocol) -> dict[str, object]:
    instances_by_suite = {
        suite.suite_id: sum(
            len(point.topology_seeds) * protocol.episodes_per_topology
            for point in suite.points
        )
        for suite in protocol.suites
    }
    method_count = 1 + len(protocol.baselines)
    return {
        "suite_count": len(protocol.suites),
        "methods": ["ARC-Q", *(item.name for item in protocol.baselines)],
        "instances_by_suite": instances_by_suite,
        "total_paired_instances": sum(instances_by_suite.values()),
        "total_method_episodes": sum(instances_by_suite.values()) * method_count,
        "output_path": str(protocol.output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/arcq_experiments.yaml"),
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=None,
        help="run one suite; repeat to select several",
    )
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    protocol = load_experiment_protocol(arguments.config)
    if arguments.dry_run:
        print(json.dumps(
            protocol_summary(protocol),
            ensure_ascii=False,
            indent=2,
        ))
        return
    paths = run_experiments(
        protocol,
        suite_ids=arguments.suite,
        max_instances=arguments.max_instances,
        device=arguments.device,
    )
    print(json.dumps({
        "json_path": str(paths[0]),
        "csv_path": str(paths[1]),
    }))


if __name__ == "__main__":
    main()


__all__ = [
    "ExperimentProtocol",
    "ExperimentSuite",
    "SweepPoint",
    "load_experiment_protocol",
    "main",
    "protocol_summary",
    "run_experiments",
    "scenario_for_point",
]
