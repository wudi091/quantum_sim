"""Run the compact, extensible experiment protocol for the paper's core claim.

The protocol has three independent parts:

* ``quality`` evaluates a frozen GNN against the LP teacher on unseen samples;
* ``generalization`` changes topology family, graph scale, or candidate count;
* ``online`` delegates paired execution to the existing GNN/MILP/Q-CAST runner.

All experiment cases live in ``core_value_config.json``. Adding a case there
does not require changing this runner or the planning/physics implementation.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from algorithms.telgen.ipm_trajectory_pilot import (
    TELGENPaperGNN,
    _evaluate,
    make_samples,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("core_value_config.json")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a JSON object: {path}")
    return payload


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_model(
    checkpoint_path: Path,
    *,
    untrained_seed: int | None = None,
) -> tuple[TELGENPaperGNN, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    if checkpoint.get("model_class") != "TELGENPaperGNN":
        raise ValueError("checkpoint is not a TELGENPaperGNN checkpoint")
    config = checkpoint.get("model_config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint is missing model_config")

    if untrained_seed is not None:
        torch.manual_seed(int(untrained_seed))
    model = TELGENPaperGNN(
        hidden_dim=int(config["hidden_dim"]),
        inner_layers=int(config["inner_layers"]),
        message_mlp_layers=int(config["message_mlp_layers"]),
        prediction_layers=int(config["prediction_layers"]),
        normalization=config.get("normalization"),
        dropout=float(config.get("dropout", 0.0)),
    )
    if untrained_seed is None:
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("checkpoint is missing state_dict")
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, dict(checkpoint)


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, float | int]:
    keys = (
        "samples",
        "mean_objective_ratio",
        "mean_onoc_scaled_objective_ratio",
        "mean_normalized_constraint_violation",
        "max_normalized_constraint_violation",
        "raw_feasible_rate",
        "rounded_feasible_rate",
        "mean_final_variable_mse",
        "mean_trajectory_variable_mse",
        "mean_teacher_rounded_request_count",
        "mean_gnn_rounded_request_count",
        "mean_gnn_to_teacher_rounded_mass_ratio",
        "mean_teacher_rounded_completion_latency",
        "mean_gnn_rounded_completion_latency",
        "mean_rounded_selection_jaccard",
        "mean_rounded_request_jaccard",
    )
    return {
        key: float(metrics[key]) if key != "samples" else int(metrics[key])
        for key in keys
        if key in metrics
    }


def _offline_case(
    case: Mapping[str, Any],
    *,
    trained: torch.nn.Module,
    untrained: torch.nn.Module,
    inference_steps: int,
    output_path: Path,
) -> dict[str, Any]:
    required = (
        "name", "topology", "nodes", "samples", "seed", "requests",
        "horizon", "paths", "construction_plans", "endpoint_mode",
    )
    missing = [key for key in required if key not in case]
    if missing:
        raise ValueError(f"offline case missing keys: {missing}")

    samples = make_samples(
        topology=str(case["topology"]),
        node_count=tuple(int(item) for item in case["nodes"]),
        sample_count=int(case["samples"]),
        seed=int(case["seed"]),
        request_count=int(case["requests"]),
        horizon=int(case["horizon"]),
        path_count=int(case["paths"]),
        construction_plan_count=int(case["construction_plans"]),
        outer_steps=int(inference_steps),
        endpoint_mode=str(case["endpoint_mode"]),
    )
    trained_metrics = _compact_metrics(_evaluate(trained, samples))
    untrained_metrics = _compact_metrics(_evaluate(untrained, samples))
    payload = {
        "schema_version": 1,
        "experiment": "core_value_offline_evaluation",
        "case": dict(case),
        "contract": {
            "checkpoint_frozen": True,
            "trained_and_untrained_share_samples": True,
            "teacher": "SciPy interior-point LP trajectory",
            "physical_execution": False,
        },
        "trained": trained_metrics,
        "untrained": untrained_metrics,
        "delta": {
            "objective_ratio": (
                float(trained_metrics["mean_objective_ratio"])
                - float(untrained_metrics["mean_objective_ratio"])
            ),
            "final_mse": (
                float(trained_metrics["mean_final_variable_mse"])
                - float(untrained_metrics["mean_final_variable_mse"])
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def _online_command(
    case: Mapping[str, Any],
    *,
    common: Mapping[str, Any],
    checkpoint: Path,
    output: Path,
    smoke: bool,
) -> list[str]:
    merged = dict(common)
    merged.update(case)
    merged["checkpoint"] = str(checkpoint)
    merged["output"] = str(output)
    if smoke:
        merged["seeds"] = 1

    command = [
        sys.executable,
        "-m",
        "algorithms.telgen.compare_online_gnn",
    ]
    flag_names = {"waxman_add_mst", "skip_milp", "skip_qcast"}
    for key, value in merged.items():
        if key in {"name", "topology_mode"} or value is None:
            continue
        option = "--" + key.replace("_", "-")
        if key in flag_names:
            if value:
                command.append(option)
            continue
        command.extend((option, str(value)))
    command.extend(("--topology-mode", str(merged["topology_mode"])))
    return command


def _run_online_case(
    case: Mapping[str, Any],
    *,
    common: Mapping[str, Any],
    checkpoint: Path,
    output: Path,
    smoke: bool,
    dry_run: bool,
) -> dict[str, Any]:
    command = _online_command(
        case,
        common=common,
        checkpoint=checkpoint,
        output=output,
        smoke=smoke,
    )
    record: dict[str, Any] = {
        "name": str(case["name"]),
        "output": str(output),
        "command": command,
    }
    if dry_run:
        record["status"] = "planned"
        return record
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=ROOT, check=True)
    report = output / "online_gnn_comparison.json"
    if not report.is_file():
        raise RuntimeError(f"online case did not produce {report}")
    record["status"] = "completed"
    record["report"] = str(report)
    return record


def run_protocol(
    config_path: Path,
    *,
    case_name: str,
    checkpoint_override: Path | None = None,
    output_override: Path | None = None,
    smoke: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = _load_json(config_path)
    checkpoint = checkpoint_override or _resolve_path(str(config["checkpoint"]))
    output_root = output_override or _resolve_path(str(config["output_root"]))
    if not dry_run and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    checkpoint_meta: dict[str, Any] = {"path": str(checkpoint)}
    inference_steps = 16
    if checkpoint.is_file():
        checkpoint_meta["sha256"] = _sha256(checkpoint)
        _, checkpoint_payload = _load_model(checkpoint)
        inference_steps = int(checkpoint_payload["inference_steps"])

    trained = untrained = None
    if case_name in {"quality", "generalization", "all"} and not dry_run:
        trained, checkpoint_payload = _load_model(checkpoint)
        untrained, _ = _load_model(
            checkpoint,
            untrained_seed=int(config.get("untrained_seed", 20260902)),
        )
        inference_steps = int(checkpoint_payload["inference_steps"])

    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "core_value_experiment_protocol",
        "protocol": config,
        "checkpoint": checkpoint_meta,
        "requested_case": case_name,
        "results": {"offline": [], "online": []},
    }

    offline = config.get("offline", {})
    if case_name in {"quality", "all"}:
        quality = dict(offline.get("quality", {}))
        if smoke:
            quality["samples"] = min(2, int(quality.get("samples", 2)))
        path = output_root / "offline" / "lp_quality.json"
        if dry_run:
            report["results"]["offline"].append({"name": "lp_quality", "status": "planned", "output": str(path), "case": quality})
        else:
            report["results"]["offline"].append(_offline_case(quality, trained=trained, untrained=untrained, inference_steps=inference_steps, output_path=path))
    if case_name in {"generalization", "all"}:
        for original_case in offline.get("generalization", []):
            item = dict(original_case)
            if smoke:
                item["samples"] = min(2, int(item.get("samples", 2)))
            path = output_root / "offline" / f"{item['name']}.json"
            if dry_run:
                report["results"]["offline"].append({"name": item["name"], "status": "planned", "output": str(path), "case": item})
            else:
                report["results"]["offline"].append(_offline_case(item, trained=trained, untrained=untrained, inference_steps=inference_steps, output_path=path))

    if case_name in {"online", "all"}:
        online = config.get("online", {})
        common = dict(online.get("common", {}))
        for item in online.get("cases", []):
            case = dict(item)
            path = output_root / "online" / str(case["name"])
            report["results"]["online"].append(_run_online_case(case, common=common, checkpoint=checkpoint, output=path, smoke=smoke, dry_run=dry_run))

    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned = output_root / f"core_value_manifest_{timestamp}.json"
        fixed = output_root / "core_value_manifest.json"
        versioned.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.copyfile(versioned, fixed)
        report["manifest"] = str(versioned)
        report["latest_manifest"] = str(fixed)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--case",
        choices=("quality", "generalization", "online", "all"),
        default="all",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_protocol(
        args.config,
        case_name=args.case,
        checkpoint_override=(
            _resolve_path(args.checkpoint) if args.checkpoint else None
        ),
        output_override=(
            _resolve_path(args.output) if args.output else None
        ),
        smoke=args.smoke,
        dry_run=args.dry_run,
    )
    print(json.dumps({
        "experiment": report["experiment"],
        "requested_case": report["requested_case"],
        "checkpoint": report["checkpoint"],
        "offline_cases": [item.get("name", item.get("case", {}).get("name")) for item in report["results"]["offline"]],
        "online_cases": [item.get("name") for item in report["results"]["online"]],
        "manifest": report.get("manifest"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
