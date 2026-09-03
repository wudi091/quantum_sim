"""Run the configurable long-run online experiment protocol.

The YAML file defines experiment axes and shared settings.  Each point is
executed as one continuous online episode through the existing comparison
runner.  The runner writes machine-readable JSON and a compact CSV summary;
it does not create or inspect figures.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("online_experiments.yaml")
SUPPORTED_METHODS = frozenset({"gnn", "milp", "qcast", "qpass", "greedy"})
SUMMARY_METRIC_COLUMNS = (
    "mean_completion_delay_slots",
    "max_completion_delay_slots",
    "mean_final_fidelity_loss",
    "completion_delay_gini",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a YAML mapping: {path}")
    return payload


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _slug(value: object) -> str:
    text = str(value).strip().lower()
    normalized = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in text
    )
    return normalized.strip("_") or "point"


def _methods_flags(methods: object) -> list[str]:
    if methods is None:
        return []
    if not isinstance(methods, list) or not methods:
        raise ValueError("methods must be null or a non-empty list")
    selected = {str(item).lower() for item in methods}
    unknown = selected - SUPPORTED_METHODS
    if unknown:
        raise ValueError(
            "unsupported methods for algorithms.telgen.compare_online_gnn: "
            + ", ".join(sorted(unknown))
        )
    if "gnn" not in selected:
        raise ValueError("the current comparison runner always requires gnn")
    flags: list[str] = []
    if "milp" not in selected:
        flags.append("--skip-milp")
    if "qcast" not in selected:
        flags.append("--skip-qcast")
    if "qpass" not in selected:
        flags.append("--skip-qpass")
    if "greedy" not in selected:
        flags.append("--skip-greedy")
    return flags


def _command(
    config: Mapping[str, Any],
    overrides: Mapping[str, Any],
    output: Path,
) -> list[str]:
    runner = dict(config.get("runner", {}))
    module = str(runner.get("module", "algorithms.telgen.compare_online_gnn"))
    common = config.get("common", {})
    if not isinstance(common, Mapping):
        raise ValueError("common must be a YAML mapping")
    params = dict(common)
    params.update(overrides)
    checkpoint = config.get("checkpoint")
    if checkpoint is None:
        raise ValueError("checkpoint is required")
    params["checkpoint"] = str(_resolve_path(str(checkpoint)))
    params["output"] = str(output)
    params["time_segments"] = int(runner.get("time_segments", 5))

    command = [sys.executable, "-m", module]
    for key, value in params.items():
        if key in {"methods", "output_root", "runner", "name"} or value is None:
            continue
        if key == "topology_file":
            value = str(_resolve_path(str(value)))
        option = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(option)
        else:
            command.extend((option, str(value)))
    command.extend(_methods_flags(config.get("methods")))
    return command


def _merge_overrides(*items: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        result.update(item)
    return result


def _temporal_rows(
    payload: Mapping[str, Any],
    *,
    experiment_id: str,
    x_axis: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trials = payload.get("trials", [])
    if not isinstance(trials, list):
        raise ValueError("comparison report has invalid trials")
    by_method: dict[str, dict[int, list[dict[str, float]]]] = {}
    for trial in trials:
        if not isinstance(trial, Mapping):
            continue
        methods = trial.get("methods", {})
        if not isinstance(methods, Mapping):
            continue
        for method, method_payload in methods.items():
            if not isinstance(method_payload, Mapping):
                continue
            segments = method_payload.get("stability_segments", [])
            if not isinstance(segments, list):
                continue
            method_rows = by_method.setdefault(str(method), {})
            for segment in segments:
                if not isinstance(segment, Mapping):
                    continue
                index = int(segment["segment"])
                method_rows.setdefault(index, []).append({
                    "completed_requests": float(segment["completed_requests"]),
                    "planning_time_seconds": float(
                        segment["mean_planner_seconds"]
                    ),
                })
    for method, segments in sorted(by_method.items()):
        for segment, values in sorted(segments.items()):
            if not values:
                continue
            rows.append({
                "experiment": experiment_id,
                "x_axis": x_axis,
                "x_value": segment,
                "method": method,
                "completed_requests": sum(
                    item["completed_requests"] for item in values
                ) / len(values),
                "planning_time_seconds": sum(
                    item["planning_time_seconds"] for item in values
                ) / len(values),
                "source": "stability_segments",
            })
    return rows


def _aggregate_rows(
    payload: Mapping[str, Any],
    *,
    experiment_id: str,
    x_axis: str,
    x_value: object,
) -> list[dict[str, Any]]:
    aggregate = payload.get("aggregate", {})
    if not isinstance(aggregate, Mapping):
        raise ValueError("comparison report has invalid aggregate")
    rows: list[dict[str, Any]] = []
    slot_duration_ps = float(
        payload.get("scenario", {})
        .get("physical", {})
        .get("slot_duration_ps", 1.0)
    )
    if slot_duration_ps <= 0.0:
        raise ValueError("scenario physical slot duration must be positive")
    for method, metrics in sorted(aggregate.items()):
        if not isinstance(metrics, Mapping):
            continue
        if "completed_requests" not in metrics:
            raise ValueError(
                f"{method} report is missing completed_requests"
            )
        row: dict[str, Any] = {
            "experiment": experiment_id,
            "x_axis": x_axis,
            "x_value": x_value,
            "method": str(method),
            "completed_requests": float(metrics["completed_requests"]),
            "planning_time_seconds": float(
                metrics.get("mean_planner_seconds", 0.0)
            ),
            "source": "aggregate",
        }
        delay_ps = metrics.get(
            "mean_completion_delay_ps",
            metrics.get("mean_censored_latency_ps"),
        )
        if delay_ps is not None:
            row["mean_completion_delay_slots"] = float(delay_ps) / slot_duration_ps
        max_delay_ps = metrics.get("max_completion_delay_ps")
        if max_delay_ps is not None:
            row["max_completion_delay_slots"] = float(max_delay_ps) / slot_duration_ps
        for key in ("mean_final_fidelity_loss", "completion_delay_gini"):
            if key in metrics:
                row[key] = float(metrics[key])
        rows.append(row)
    return rows


def _run_point(
    config: Mapping[str, Any],
    *,
    experiment_id: str,
    x_axis: str,
    x_value: object,
    overrides: Mapping[str, Any],
    output: Path,
    temporal: bool,
    dry_run: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    command = _command(config, overrides, output)
    record: dict[str, Any] = {
        "experiment": experiment_id,
        "x_axis": x_axis,
        "x_value": x_value,
        "overrides": dict(overrides),
        "output": str(output),
        "command": command,
        "command_shell": shlex.join(command),
    }
    if dry_run:
        record["status"] = "planned"
        return record, []
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=ROOT, check=True)
    report_path = output / "online_gnn_comparison.json"
    if not report_path.is_file():
        raise RuntimeError(f"comparison runner did not produce {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"comparison report must be a mapping: {report_path}")
    record["status"] = "completed"
    record["report"] = str(report_path)
    rows = (
        _temporal_rows(payload, experiment_id=experiment_id, x_axis=x_axis)
        if temporal
        else _aggregate_rows(
            payload,
            experiment_id=experiment_id,
            x_axis=x_axis,
            x_value=x_value,
        )
    )
    return record, rows


def _experiment_points(
    experiment: Mapping[str, Any],
    *,
    time_segments: int,
) -> list[tuple[str, str, object, dict[str, Any], bool]]:
    experiment_id = str(experiment.get("id", "experiment"))
    kind = str(experiment.get("kind", "sweep"))
    x_axis = str(experiment.get("x_axis", "x"))
    if kind == "temporal_stability":
        values = experiment.get("values")
        expected = list(range(1, time_segments + 1))
        if values != expected:
            raise ValueError(
                f"{experiment_id}: temporal values must be {expected}"
            )
        overrides = experiment.get("overrides", {})
        if not isinstance(overrides, Mapping):
            raise ValueError(f"{experiment_id}: overrides must be a mapping")
        return [(experiment_id, x_axis, "continuous", dict(overrides), True)]
    if kind == "sweep":
        parameter = str(experiment["parameter"])
        values = experiment.get("values")
        if not isinstance(values, list) or len(values) < 5:
            raise ValueError(f"{experiment_id}: sweep must contain at least 5 values")
        return [
            (
                experiment_id,
                x_axis,
                value,
                {parameter: value},
                False,
            )
            for value in values
        ]
    if kind == "points":
        points = experiment.get("points")
        if not isinstance(points, list) or len(points) < 5:
            raise ValueError(f"{experiment_id}: points must contain at least 5 items")
        result = []
        for point in points:
            if not isinstance(point, Mapping) or "label" not in point:
                raise ValueError(f"{experiment_id}: every point needs a label")
            overrides = point.get("overrides", {})
            if not isinstance(overrides, Mapping):
                raise ValueError(f"{experiment_id}: point overrides must be a mapping")
            result.append((
                experiment_id,
                x_axis,
                point["label"],
                dict(overrides),
                False,
            ))
        return result
    if kind == "group":
        children = experiment.get("sweeps")
        if not isinstance(children, list) or not children:
            raise ValueError(f"{experiment_id}: group needs sweeps")
        result = []
        for child in children:
            if not isinstance(child, Mapping) or "id" not in child:
                raise ValueError(f"{experiment_id}: invalid child sweep")
            for _, child_axis, value, overrides, temporal in _experiment_points(
                {**child, "id": f"{experiment_id}_{child['id']}"},
                time_segments=time_segments,
            ):
                result.append((
                    f"{experiment_id}_{child['id']}",
                    child_axis,
                    value,
                    overrides,
                    temporal,
                ))
        return result
    raise ValueError(f"{experiment_id}: unsupported experiment kind {kind}")


def _write_outputs(
    run_root: Path,
    *,
    config: Mapping[str, Any],
    records: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    dry_run: bool,
) -> tuple[Path, Path]:
    run_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "protocol": config,
        "dry_run": dry_run,
        "records": records,
        "rows": rows,
    }
    json_path = run_root / "experiment_summary.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    csv_path = run_root / "experiment_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "experiment",
                "x_axis",
                "x_value",
                "method",
                "completed_requests",
                *SUMMARY_METRIC_COLUMNS,
                "planning_time_seconds",
                "source",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def run_protocol(
    config_path: Path,
    *,
    experiment_filter: str | None = None,
    dry_run: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    config = _load_yaml(config_path)
    runner = config.get("runner", {})
    if not isinstance(runner, Mapping):
        raise ValueError("runner must be a mapping")
    time_segments = int(runner.get("time_segments", 5))
    if time_segments < 5:
        raise ValueError("runner.time_segments must be at least 5")
    experiments = config.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("experiments must be a non-empty list")
    output_root_value = config.get("output_root")
    if output_root_value is None:
        raise ValueError("output_root is required")
    output_root = _resolve_path(str(output_root_value))
    output_root.mkdir(parents=True, exist_ok=True)
    prefix = "plan" if dry_run else "run"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    collision_index = 1
    while True:
        suffix = "" if collision_index == 1 else f"_{collision_index}"
        run_root = output_root / f"{prefix}_{timestamp}{suffix}"
        try:
            run_root.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            collision_index += 1
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        if not isinstance(experiment, Mapping):
            raise ValueError("every experiment must be a mapping")
        experiment_id = str(experiment.get("id", "experiment"))
        if experiment_filter and experiment_filter not in {
            experiment_id,
            "all",
        }:
            continue
        points = _experiment_points(experiment, time_segments=time_segments)
        for point_experiment_id, point_axis, x_value, overrides, temporal in points:
            point_label = _slug(x_value)
            point_output = run_root / point_experiment_id / point_label
            record, point_rows = _run_point(
                config,
                experiment_id=point_experiment_id,
                x_axis=point_axis,
                x_value=x_value,
                overrides=overrides,
                output=point_output,
                temporal=temporal,
                dry_run=dry_run,
            )
            records.append(record)
            rows.extend(point_rows)
    if not records:
        raise ValueError("experiment filter selected no experiments")
    json_path, csv_path = _write_outputs(
        run_root,
        config=config,
        records=records,
        rows=rows,
        dry_run=dry_run,
    )
    return json_path, csv_path, {
        "experiments": sorted({item["experiment"] for item in records}),
        "points": len(records),
        "rows": len(rows),
        "json": str(json_path),
        "csv": str(csv_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--experiment",
        help="run one experiment id; omit to run all configured experiments",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    json_path, csv_path, summary = run_protocol(
        args.config,
        experiment_filter=args.experiment,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_json={json_path}")
    print(f"summary_csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
