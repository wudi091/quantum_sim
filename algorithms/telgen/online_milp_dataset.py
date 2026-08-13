"""Load persisted online MILP teacher graphs without simulator objects."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from qnet_core.construction_api import (
    ConstructionDAG,
    ConstructionOperation,
    ResourceDemand,
)
from qnet_core.construction_catalog import RouteConstructionCandidate

from .milp_imitation import (
    CONSTRAINT_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    VARIABLE_FEATURE_NAMES,
    MILPGraphSample,
)
from .time_expansion import (
    NominalConstructionSchedule,
    ResourceSlotUsage,
    TimeExpandedCandidate,
)


@dataclass(frozen=True)
class LoadedOnlineMILPDataset:
    manifest_path: Path
    sample_paths: tuple[Path, ...]
    samples: tuple[MILPGraphSample, ...]

    @property
    def episode_seeds(self) -> tuple[int, ...]:
        return tuple(sorted({sample.seed for sample in self.samples}))

    @property
    def samples_by_episode(self) -> dict[int, tuple[MILPGraphSample, ...]]:
        grouped: dict[int, list[MILPGraphSample]] = {}
        for sample in self.samples:
            grouped.setdefault(sample.seed, []).append(sample)
        return {
            seed: tuple(values)
            for seed, values in sorted(grouped.items())
        }


def _feature_names(payload, key: str) -> tuple[str, ...]:
    return tuple(str(value) for value in payload[key].tolist())


def _operation_from_payload(
    payload: Mapping[str, object],
    request_id: str,
) -> ConstructionOperation:
    endpoints = payload.get("output_endpoints")
    return ConstructionOperation(
        op_id=str(payload["op_id"]),
        request_id=request_id,
        kind=str(payload["kind"]),
        predecessors=tuple(str(item) for item in payload["predecessors"]),
        input_segment_ids=tuple(
            str(item) for item in payload["input_segment_ids"]
        ),
        output_segment_id=(
            None
            if payload.get("output_segment_id") is None
            else str(payload["output_segment_id"])
        ),
        output_endpoints=(
            None
            if endpoints is None
            else (int(endpoints[0]), int(endpoints[1]))
        ),
        resource_demand=ResourceDemand.from_mapping({
            str(key): int(value)
            for key, value in payload["resource_demand"].items()
        }),
        output_resource_hold=ResourceDemand.from_mapping({
            str(key): int(value)
            for key, value in payload["output_resource_hold"].items()
        }),
        duration_ps=int(payload["duration_ps"]),
        success_probability=float(payload["success_probability"]),
        required_fidelity=float(payload["required_fidelity"]),
        retry_limit=int(payload["retry_limit"]),
        retry_root_id=(
            None
            if payload.get("retry_root_id") is None
            else str(payload["retry_root_id"])
        ),
        retry_attempt=int(payload["retry_attempt"]),
        ordinal=int(payload["ordinal"]),
        dag_version=int(payload["dag_version"]),
    )


def _variable_from_payload(
    payload: Mapping[str, object],
) -> TimeExpandedCandidate:
    request_id = str(payload["request_id"])
    operations = tuple(
        _operation_from_payload(item, request_id)
        for item in payload["operations"]
    )
    terminal_segment_ids = tuple(
        str(item) for item in payload["terminal_segment_ids"]
    )
    if not terminal_segment_ids:
        raise ValueError("saved candidate has no terminal segment")
    candidate = RouteConstructionCandidate(
        candidate_id=str(payload["candidate_id"]),
        request_id=request_id,
        route_nodes=tuple(int(item) for item in payload["route_nodes"]),
        construction_kind=str(payload["construction_kind"]),
        dag=ConstructionDAG(
            request_id,
            operations,
            version=int(payload.get("dag_version", 0)),
        ),
        terminal_segment_id=terminal_segment_ids[-1],
        terminal_segment_ids=terminal_segment_ids,
        purification_kind=str(payload["purification_kind"]),
    )
    resource_usage = tuple(sorted(
        ResourceSlotUsage(
            str(item["resource_id"]),
            int(item["slot"]),
            int(item["amount"]),
        )
        for item in payload["resource_usage"]
    ))
    schedule = NominalConstructionSchedule(
        candidate_id=candidate.candidate_id,
        operation_slots=tuple(sorted(
            (str(operation_id), int(slot))
            for operation_id, slot in payload["operation_slots"]
        )),
        duration_slots=int(payload["duration_slots"]),
        resource_usage=tuple(sorted(
            ResourceSlotUsage(
                item.resource_id,
                item.slot - int(payload["start_slot"]),
                item.amount,
            )
            for item in resource_usage
        )),
    )
    fidelity = payload.get("expected_fidelity")
    return TimeExpandedCandidate(
        variable_id=str(payload["variable_id"]),
        base_candidate=candidate,
        start_slot=int(payload["start_slot"]),
        completion_slot=int(payload["completion_slot"]),
        completion_latency=int(payload["completion_latency"]),
        expected_fidelity=None if fidelity is None else float(fidelity),
        resource_usage=resource_usage,
        nominal_schedule=schedule,
        expected_success_probability=float(
            payload["expected_success_probability"]
        ),
    )


def load_online_milp_graph_sample(path: str | Path) -> MILPGraphSample:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        expected_schemas = {
            "variable_feature_names": VARIABLE_FEATURE_NAMES,
            "constraint_feature_names": CONSTRAINT_FEATURE_NAMES,
            "global_feature_names": GLOBAL_FEATURE_NAMES,
        }
        for key, expected in expected_schemas.items():
            if key not in payload.files:
                raise ValueError(f"saved graph is missing {key}")
            actual = _feature_names(payload, key)
            if actual != tuple(expected):
                raise ValueError(
                    f"saved graph {key} does not match the current schema"
                )
        context = json.loads(str(payload["context_json"].item()))
        variables = tuple(
            _variable_from_payload(item) for item in context["variables"]
        )
        variable_ids = tuple(str(item) for item in payload["variable_ids"])
        if variable_ids != tuple(item.variable_id for item in variables):
            raise ValueError("saved variable IDs do not match variable metadata")
        reservations = {
            (str(item["resource_id"]), int(item["slot"])): int(item["amount"])
            for item in context["reserved_usage"]
        }
        request_ids = tuple(
            str(item["id"]) for item in context["request_state"]
        )
        return MILPGraphSample(
            seed=int(context["episode_seed"]),
            variable_features=np.asarray(
                payload["variable_features"], dtype=np.float32
            ).copy(),
            constraint_features=np.asarray(
                payload["constraint_features"], dtype=np.float32
            ).copy(),
            global_features=np.asarray(
                payload["global_features"], dtype=np.float32
            ).copy(),
            edge_variable_indices=np.asarray(
                payload["edge_variable_indices"], dtype=np.int64
            ).copy(),
            edge_constraint_indices=np.asarray(
                payload["edge_constraint_indices"], dtype=np.int64
            ).copy(),
            edge_features=np.asarray(
                payload["edge_features"], dtype=np.float32
            ).copy(),
            constraint_rhs=np.asarray(
                payload["constraint_rhs"], dtype=np.float32
            ).copy(),
            labels=np.asarray(payload["labels"], dtype=np.float32).copy(),
            variables=variables,
            resource_capacities={
                str(key): int(value)
                for key, value in context["resource_capacities"].items()
            },
            reserved_usage=reservations,
            request_ids=request_ids,
            optimal_completed_request_count=int(
                context["optimal_completed_request_count"]
            ),
            optimal_expected_completed_request_mass=float(
                context["optimal_expected_completed_request_mass"]
            ),
            optimal_total_completion_latency=float(
                context["optimal_total_completion_latency"]
            ),
            stage_one_mip_gap=(
                None
                if context.get("stage_one_mip_gap") is None
                else float(context["stage_one_mip_gap"])
            ),
            stage_two_mip_gap=(
                None
                if context.get("stage_two_mip_gap") is None
                else float(context["stage_two_mip_gap"])
            ),
        )


def _rollout_manifest_path(path: Path) -> Path:
    payload = json.loads(path.read_text(encoding="utf-8"))
    version_directory = payload.get("version_directory")
    if version_directory is None:
        return path
    return path.parent / str(version_directory) / "manifest.json"


def _sample_paths_from_manifest(path: Path) -> tuple[Path, ...]:
    resolved = _rollout_manifest_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    kind = payload.get("dataset_kind")
    if kind == "online_milp_teacher_rollout":
        return tuple(
            resolved.parent / str(item["file"])
            for item in payload["samples"]
        )
    if kind == "online_milp_teacher_collection":
        paths: list[Path] = []
        for episode in payload["episodes"]:
            episode_manifest = resolved.parent / str(
                episode["manifest"]
            )
            paths.extend(_sample_paths_from_manifest(episode_manifest))
        return tuple(paths)
    raise ValueError(f"unsupported online MILP dataset manifest: {kind}")


def resolve_online_milp_dataset_manifest(path: str | Path) -> Path:
    source = Path(path)
    if source.is_file():
        return source
    for name in ("online_milp_dataset.json", "manifest.json"):
        candidate = source / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no online MILP dataset manifest under {source}")


def load_online_milp_dataset(
    path: str | Path,
) -> LoadedOnlineMILPDataset:
    manifest = resolve_online_milp_dataset_manifest(path)
    sample_paths = _sample_paths_from_manifest(manifest)
    if not sample_paths:
        raise ValueError("online MILP dataset contains no graph samples")
    if len(set(sample_paths)) != len(sample_paths):
        raise ValueError("online MILP dataset contains duplicate sample paths")
    missing = [sample for sample in sample_paths if not sample.exists()]
    if missing:
        raise FileNotFoundError(missing[0])
    samples = tuple(load_online_milp_graph_sample(item) for item in sample_paths)
    return LoadedOnlineMILPDataset(
        manifest_path=manifest,
        sample_paths=sample_paths,
        samples=samples,
    )


def samples_for_episode_seeds(
    samples: Iterable[MILPGraphSample],
    seeds: Iterable[int],
) -> tuple[MILPGraphSample, ...]:
    selected = {int(seed) for seed in seeds}
    return tuple(sample for sample in samples if sample.seed in selected)


__all__ = [
    "LoadedOnlineMILPDataset",
    "load_online_milp_dataset",
    "load_online_milp_graph_sample",
    "resolve_online_milp_dataset_manifest",
    "samples_for_episode_seeds",
]
