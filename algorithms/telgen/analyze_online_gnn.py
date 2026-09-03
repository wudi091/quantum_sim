"""Statistically analyze paired online GNN/Q-CAST comparison reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from .analyze_online_benchmark import (
    analyze_online_payloads,
    save_analysis,
)


def _convert_payload(payload: Mapping[str, object]) -> dict[str, object]:
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported online GNN comparison schema")
    if payload.get("experiment") not in {
        "paired_online_gnn_milp_qcast",
        "paired_online_gnn_routing_baselines",
    }:
        raise ValueError("not an online GNN comparison report")
    contract = payload.get("comparison_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("comparison contract is missing")
    required_contract = {
        "paired_episode_spec": True,
        "independent_persistent_executors": True,
        "gnn_calls_milp_online": False,
        "qcast_uses_gnn_or_milp": False,
    }
    for key, expected in required_contract.items():
        if contract.get(key) != expected:
            raise ValueError(f"comparison contract mismatch for {key}")
    configuration = payload.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("comparison configuration is missing")
    if configuration.get("skip_milp") is not True:
        raise ValueError("online GNN analysis requires MILP-free evaluation")
    if payload.get("milp_config") is not None:
        raise ValueError("online report unexpectedly contains a MILP config")
    trials = payload.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError("comparison report contains no trials")

    converted_trials = []
    for trial in trials:
        if "episode" not in trial:
            raise ValueError("trial is missing the exact EpisodeSpec")
        if int(trial["episode"].get("seed", -1)) != int(trial["seed"]):
            raise ValueError("trial seed does not match its EpisodeSpec")
        methods = trial.get("methods")
        if not isinstance(methods, Mapping):
            raise ValueError("trial methods are missing")
        if set(methods) != {"gnn", "qcast"}:
            raise ValueError("trial must contain only GNN and Q-CAST")
        converted_trials.append({
            "seed": trial["seed"],
            "episode": trial["episode"],
            "telgen": methods["gnn"],
            "qcast": methods["qcast"],
        })
    return {
        "schema_version": 1,
        "comparison_contract": {
            "paired_episode_spec": True,
            "independent_persistent_executors": True,
            "common_runtime_metric": "mean_decision_seconds",
            "qcast_baseline": "width_one_ext_fixed_construction",
            "qcast_uses_telgen_lp_or_search_decoder": False,
        },
        "scenario": payload["scenario"],
        "trials": converted_trials,
    }


def _additional_hard_gates(
    payloads: Mapping[str, Mapping[str, object]],
) -> dict[str, float]:
    totals = {
        "gnn_invalid_decision_count": 0.0,
        "gnn_unsafe_schedule_violation_count": 0.0,
        "qcast_unsafe_schedule_violation_count": 0.0,
        "gnn_nominal_completion_overrun_count": 0.0,
        "qcast_nominal_completion_overrun_count": 0.0,
        "gnn_physical_backend_rejection_count": 0.0,
        "qcast_physical_backend_rejection_count": 0.0,
        "gnn_post_completion_validation_failure_count": 0.0,
        "qcast_post_completion_validation_failure_count": 0.0,
    }
    for payload in payloads.values():
        for trial in payload["trials"]:
            methods = trial["methods"]
            gnn = methods["gnn"]["metrics"]
            qcast = methods["qcast"]["metrics"]
            totals["gnn_invalid_decision_count"] += float(
                gnn.get("gnn_invalid_decision_count", 0.0)
            )
            for name, method, metrics in (
                ("gnn", methods["gnn"], gnn),
                ("qcast", methods["qcast"], qcast),
            ):
                for violation in method.get("violations", ()):
                    suffix = (
                        "nominal_completion_overrun_count"
                        if violation.get("code") == "slot_completion_overrun"
                        else "unsafe_schedule_violation_count"
                    )
                    totals[f"{name}_{suffix}"] += 1.0
                for metric in (
                    "physical_backend_rejection_count",
                    "post_completion_validation_failure_count",
                ):
                    totals[f"{name}_{metric}"] += float(
                        metrics.get(metric, 0.0)
                    )
    return totals


def analyze_gnn_reports(
    report_paths: list[Path],
    *,
    bootstrap_samples: int,
    randomization_samples: int,
    random_seed: int,
) -> dict[str, object]:
    source_payloads: dict[str, Mapping[str, object]] = {}
    converted_payloads: dict[str, Mapping[str, object]] = {}
    for path in report_paths:
        case_name = path.parent.name
        if case_name in source_payloads:
            raise ValueError(f"duplicate case directory name: {case_name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_payloads[case_name] = payload
        converted_payloads[case_name] = _convert_payload(payload)
    analysis = analyze_online_payloads(
        converted_payloads,
        bootstrap_samples=bootstrap_samples,
        randomization_samples=randomization_samples,
        random_seed=random_seed,
    )
    extra_gates = _additional_hard_gates(source_payloads)
    checkpoint_hashes = {
        str(payload.get("checkpoint_sha256", ""))
        for payload in source_payloads.values()
    }
    if "" in checkpoint_hashes or len(checkpoint_hashes) != 1:
        raise ValueError("all cases must use one recorded GNN checkpoint")
    analysis["experiment"] = "paired_online_gnn_qcast_generalization"
    analysis["method_aliases"] = {"telgen": "gnn", "qcast": "qcast"}
    analysis["inputs"] = [str(path) for path in report_paths]
    analysis["checkpoint_sha256"] = sorted(checkpoint_hashes)
    analysis["additional_hard_gates"] = extra_gates
    extra_valid = all(
        value == 0.0
        for name, value in extra_gates.items()
        if not name.endswith("nominal_completion_overrun_count")
    )
    analysis["additional_hard_gates_valid"] = extra_valid
    if not extra_valid:
        analysis["overall"]["valid"] = False
        analysis["overall"]["quality_verdict"] = "invalid_hard_gate_failure"
    return analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze paired online GNN/Q-CAST reports."
    )
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--randomization-samples", type=int, default=20_000)
    parser.add_argument("--random-seed", type=int, default=20260814)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analysis = analyze_gnn_reports(
        args.reports,
        bootstrap_samples=args.bootstrap_samples,
        randomization_samples=args.randomization_samples,
        random_seed=args.random_seed,
    )
    paths = save_analysis(analysis, args.output)
    for key in ("markdown", "latest_markdown"):
        path = paths[key]
        text = path.read_text(encoding="utf-8")
        path.write_text(
            text.replace("TELGEN", "GNN"),
            encoding="utf-8",
        )
    completed = analysis["overall"]["metrics"]["completed_requests"]
    print(
        f"completed_advantage={completed['telgen_advantage_mean']:.4f} "
        f"ci95=[{completed['ci95_low']:.4f}, {completed['ci95_high']:.4f}] "
        f"p={completed['paired_randomization_p']:.6f}"
    )
    print(f"hard_gates_valid={analysis['overall']['valid']}")
    print(f"json: {paths['json']}")
    print(f"markdown: {paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
