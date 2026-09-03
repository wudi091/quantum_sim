"""Export per-sample offline metrics for evidence-bound plotting.

The canonical offline reports contain aggregate means.  This companion
command recomputes the same deterministic samples and stores one record per
sample so figures can expose sample and topology-scale variation without
repeating an aggregate value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from algorithms.telgen.ipm_trajectory_pilot import _evaluate, make_samples

from .run_core_value import DEFAULT_CONFIG, _load_json, _load_model, _resolve_path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _detail_metrics(model: Any, sample: Any) -> dict[str, float | int]:
    metrics = _evaluate(model, [sample])
    keys = (
        "mean_objective_ratio",
        "mean_onoc_scaled_objective_ratio",
        "mean_normalized_constraint_violation",
        "raw_feasible_rate",
        "rounded_feasible_rate",
        "mean_final_variable_mse",
        "mean_teacher_rounded_request_count",
        "mean_gnn_rounded_request_count",
        "mean_teacher_rounded_completion_latency",
        "mean_gnn_rounded_completion_latency",
        "mean_rounded_selection_jaccard",
        "mean_rounded_request_jaccard",
    )
    return {key: float(metrics[key]) for key in keys if key in metrics}


def _export_case(
    case: Mapping[str, Any],
    *,
    checkpoint: Path,
    trained: Any,
    untrained: Any,
    inference_steps: int,
    output: Path,
) -> dict[str, Any]:
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
    records = []
    for index, sample in enumerate(samples, start=1):
        records.append(
            {
                "sample_index": index,
                "seed": int(sample.seed),
                "topology": sample.topology,
                "topology_seed": int(sample.topology_seed),
                "node_count": int(sample.node_count),
                "trained": _detail_metrics(trained, sample),
                "untrained": _detail_metrics(untrained, sample),
            }
        )
    payload = {
        "schema_version": 1,
        "experiment": "core_value_offline_sample_details",
        "case": dict(case),
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "contract": {
            "same_samples_as_aggregate_report": True,
            "per_sample_metrics_are_recomputed_from_fixed_seed": True,
            "physical_execution": False,
        },
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--case", choices=("quality", "generalization", "all"), default="all")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = _load_json(config_path)
    checkpoint = _resolve_path(str(config["checkpoint"]))
    trained, checkpoint_payload = _load_model(checkpoint)
    untrained, _ = _load_model(checkpoint, untrained_seed=int(config.get("untrained_seed", 20260902)))
    inference_steps = int(checkpoint_payload["inference_steps"])
    output_root = _resolve_path(str(config["output_root"])) / "offline"
    cases: list[Mapping[str, Any]] = []
    offline = config.get("offline", {})
    if args.case in {"quality", "all"}:
        cases.append(dict(offline["quality"]))
    if args.case in {"generalization", "all"}:
        cases.extend(dict(item) for item in offline.get("generalization", []))
    reports = []
    for case in cases:
        path = output_root / f"{case['name']}_details.json"
        payload = _export_case(
            case,
            checkpoint=checkpoint,
            trained=trained,
            untrained=untrained,
            inference_steps=inference_steps,
            output=path,
        )
        reports.append({"name": case["name"], "records": len(payload["records"]), "output": str(path)})
    print(json.dumps({"experiment": "core_value_offline_sample_details", "reports": reports}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
