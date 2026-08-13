"""Small direct-MILP imitation experiment for construction-aware routing."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version as distribution_version
import json
from pathlib import Path
import random
import shutil
import sys
from time import perf_counter
from typing import Sequence

import networkx as nx
import numpy as np
import scipy
import torch

from qnet_core.scenario import ScenarioConfig
from qnet_core.spec import PhysicalConfig

from .milp_imitation import (
    CandidateConstraintGNN,
    MILPGraphSample,
    batch_graph_samples,
    generate_milp_graph_sample,
    greedy_decode_scores,
    imitation_loss,
)
from .hard_decoder import validate_decoded_selection


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _evaluate(
    model: CandidateConstraintGNN,
    samples: Sequence[MILPGraphSample],
    *,
    threshold: float,
    decode_threshold: float | None = None,
    device: torch.device,
    sample_batch_size: int | None = None,
) -> dict[str, object]:
    model.eval()
    if not samples:
        raise ValueError("evaluation requires at least one sample")
    batch_size = (
        len(samples) if sample_batch_size is None else int(sample_batch_size)
    )
    if batch_size < 1:
        raise ValueError("sample batch size must be positive")
    tp = fp = fn = tn = 0
    positive_prediction_count = 0
    positive_label_count = 0
    variable_count = 0
    evaluated_sample_count = 0
    weighted_parts: dict[str, float] = {}
    decoded = []
    resolved_decode_threshold = (
        threshold if decode_threshold is None else float(decode_threshold)
    )
    if not 0.0 <= resolved_decode_threshold <= 1.0:
        raise ValueError("decode threshold must lie in [0, 1]")
    for batch_start in range(0, len(samples), batch_size):
        batch_samples = samples[batch_start:batch_start + batch_size]
        graph = batch_graph_samples(batch_samples, device=device)
        with torch.no_grad():
            logits = model(graph)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            loss, parts = imitation_loss(logits, graph)
        true = graph.labels.cpu().numpy()
        predicted_binary = probabilities >= threshold
        true_binary = true >= 0.5
        tp += int(np.sum(predicted_binary & true_binary))
        fp += int(np.sum(predicted_binary & ~true_binary))
        fn += int(np.sum(~predicted_binary & true_binary))
        tn += int(np.sum(~predicted_binary & ~true_binary))
        positive_prediction_count += int(np.sum(predicted_binary))
        positive_label_count += int(np.sum(true_binary))
        batch_variable_count = len(true_binary)
        variable_count += batch_variable_count
        batch_sample_count = len(batch_samples)
        evaluated_sample_count += batch_sample_count
        weighted_parts["loss"] = weighted_parts.get("loss", 0.0) + (
            float(loss.detach()) * batch_sample_count
        )
        for key, value in parts.items():
            weighted_parts[key] = weighted_parts.get(key, 0.0) + (
                float(value) * batch_sample_count
            )
        for sample, (start, end) in zip(
            batch_samples, graph.variable_slices
        ):
            sample_probabilities = probabilities[start:end]
            raw_selected = tuple(
                variable
                for variable, probability in zip(
                    sample.variables, sample_probabilities
                )
                if probability >= threshold
            )
            raw_feasibility = validate_decoded_selection(
                raw_selected,
                sample.resource_capacities,
                sample.reserved_usage,
            )
            result = greedy_decode_scores(
                sample,
                sample_probabilities,
                threshold=resolved_decode_threshold,
            )
            throughput_optimal = (
                abs(
                    result.expected_completed_request_mass
                    - sample.optimal_expected_completed_request_mass
                ) <= 1e-7
            )
            latency_gap = (
                result.total_completion_latency
                - sample.optimal_total_completion_latency
                if throughput_optimal else None
            )
            latency_relative_gap = (
                latency_gap
                / max(sample.optimal_total_completion_latency, 1.0)
                if latency_gap is not None else None
            )
            decoded.append({
                "seed": sample.seed,
                "raw_threshold_selected_count": len(raw_selected),
                "raw_threshold_feasible": raw_feasibility.feasible,
                "raw_threshold_violation_count": len(
                    raw_feasibility.violations
                ),
                "post_projection_feasible": result.feasible,
                "completed_request_count": result.completed_request_count,
                "optimal_completed_request_count": (
                    sample.optimal_completed_request_count
                ),
                "throughput_ratio": (
                    result.expected_completed_request_mass
                    / sample.optimal_expected_completed_request_mass
                    if sample.optimal_expected_completed_request_mass
                    else 1.0
                ),
                "expected_completed_request_mass": (
                    result.expected_completed_request_mass
                ),
                "optimal_expected_completed_request_mass": (
                    sample.optimal_expected_completed_request_mass
                ),
                "throughput_optimal": throughput_optimal,
                "total_completion_latency": result.total_completion_latency,
                "optimal_total_completion_latency": (
                    sample.optimal_total_completion_latency
                ),
                "latency_gap_when_throughput_optimal": latency_gap,
                "latency_relative_gap_when_throughput_optimal": (
                    latency_relative_gap
                ),
                "lexicographic_objective_optimal": bool(
                    throughput_optimal
                    and latency_gap is not None
                    and abs(latency_gap) <= 1e-7
                ),
                "exact_selected_set": set(result.selected_variable_ids) == {
                    variable.variable_id
                    for variable, label in zip(
                        sample.variables, sample.labels
                    )
                    if label > 0.5
                },
            })
    comparable_latency_gaps = [
        item["latency_relative_gap_when_throughput_optimal"]
        for item in decoded
        if item["latency_relative_gap_when_throughput_optimal"] is not None
    ]
    return {
        **{
            key: value / max(evaluated_sample_count, 1)
            for key, value in weighted_parts.items()
        },
        "sample_count": len(samples),
        "variable_count": variable_count,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "positive_prediction_count": positive_prediction_count,
        "positive_label_count": positive_label_count,
        "classification_threshold": float(threshold),
        "decode_threshold": resolved_decode_threshold,
        "mean_decoded_throughput_ratio": float(np.mean([
            item["throughput_ratio"] for item in decoded
        ])),
        "minimum_decoded_throughput_ratio": float(np.min([
            item["throughput_ratio"] for item in decoded
        ])),
        "raw_threshold_feasible_rate": float(np.mean([
            item["raw_threshold_feasible"] for item in decoded
        ])),
        "mean_raw_threshold_violation_count": float(np.mean([
            item["raw_threshold_violation_count"] for item in decoded
        ])),
        "post_projection_feasible_rate": float(np.mean([
            item["post_projection_feasible"] for item in decoded
        ])),
        "throughput_optimal_rate": float(np.mean([
            item["throughput_optimal"] for item in decoded
        ])),
        "mean_latency_relative_gap_when_throughput_optimal": (
            float(np.mean(comparable_latency_gaps))
            if comparable_latency_gaps else None
        ),
        "lexicographic_objective_optimal_rate": float(np.mean([
            item["lexicographic_objective_optimal"] for item in decoded
        ])),
        "exact_selected_set_rate": float(np.mean([
            item["exact_selected_set"] for item in decoded
        ])),
        "decoded": decoded,
    }


def _build_overfit_gate(after: dict[str, object]) -> dict[str, object]:
    """Return a strict per-training-instance memorization gate."""

    tolerance = 1e-12
    passed = bool(
        float(after["minimum_decoded_throughput_ratio"])
        >= 1.0 - tolerance
        and float(after["lexicographic_objective_optimal_rate"])
        >= 1.0 - tolerance
        and float(after["raw_threshold_feasible_rate"])
        >= 1.0 - tolerance
        and float(after["post_projection_feasible_rate"])
        >= 1.0 - tolerance
    )
    return {
        "f1_is_diagnostic_only": True,
        "target_minimum_decoded_throughput_ratio": 1.0,
        "target_lexicographic_objective_optimal_rate": 1.0,
        "target_raw_threshold_feasible_rate": 1.0,
        "target_post_projection_feasible_rate": 1.0,
        "passed": passed,
    }


def _save_report(payload: dict[str, object], output_directory: Path):
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned = output_directory / f"milp_imitation_{timestamp}.json"
    latest = output_directory / "milp_imitation.json"
    versioned.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    shutil.copyfile(versioned, latest)
    return versioned, latest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Directly imitate exact construction-aware MILP labels."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-seeds", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=3101)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--training-seed", type=int, default=20260813)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--min-hops", type=int, default=4)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--paths", type=int, default=4)
    parser.add_argument("--construction-plans", type=int, default=5)
    parser.add_argument("--time-limit-seconds", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.train_seeds < 1:
        raise ValueError("train-seeds must be positive")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    _seed_everything(args.training_seed)
    device = torch.device("cpu")
    scenario = ScenarioConfig(
        request_count=args.requests,
        min_hops=args.min_hops,
        max_hops=args.max_hops,
        ttl=args.horizon,
        horizon=args.horizon,
        topology_nodes=args.nodes,
        waxman_alpha=0.15,
        waxman_beta=0.45,
        topology_attempts=128,
        waxman_add_mst=False,
        endpoint_mode="distance_stratified",
        physical=PhysicalConfig(
            generation_probability=0.8,
            swap_probability=0.9,
            memory_capacity=2,
            memory_lifetime=300,
            max_width=1,
            quantum_distance_m=1000.0,
            slot_duration_ps=50_000_000,
        ),
    )
    generation_started = perf_counter()
    samples = tuple(
        generate_milp_graph_sample(
            args.seed_start + index,
            scenario,
            path_candidate_count=args.paths,
            swap_tree_count=args.construction_plans,
            time_limit_seconds=args.time_limit_seconds,
        )
        for index in range(args.train_seeds)
    )
    generation_seconds = perf_counter() - generation_started
    if any(
        sample.stage_one_mip_gap is None
        or sample.stage_two_mip_gap is None
        or sample.stage_one_mip_gap > 1e-12
        or sample.stage_two_mip_gap > 1e-12
        for sample in samples
    ):
        raise RuntimeError("training label MILP did not reach numerical zero gap")

    model = CandidateConstraintGNN(
        hidden_dim=args.hidden_dim,
        layers=args.layers,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    graph = batch_graph_samples(samples, device=device)
    before = _evaluate(
        model, samples, threshold=args.threshold, device=device
    )
    training_started = perf_counter()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(graph)
        loss, parts = imitation_loss(logits, graph)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if epoch == 1 or epoch % max(args.epochs // 10, 1) == 0:
            history.append({
                "epoch": epoch,
                "loss": float(loss.detach()),
                **parts,
            })
            print(
                f"epoch={epoch} loss={float(loss.detach()):.6f} "
                f"bce={parts['bce']:.6f} "
                f"constraint={parts['constraint_penalty']:.6f}",
                flush=True,
            )
    training_seconds = perf_counter() - training_started
    after = _evaluate(
        model, samples, threshold=args.threshold, device=device
    )
    payload = {
        "schema_version": 2,
        "experiment": "direct_milp_binary_imitation_overfit",
        "supervision": "exact two-stage construction-aware MILP final 0/1 labels",
        "uses_lp_trajectory": False,
        "configuration": {
            **vars(args),
            "output": str(args.output),
            "device": str(device),
        },
        "runtime_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "networkx": nx.__version__,
            "sequence": distribution_version("sequence"),
        },
        "scenario": asdict(scenario),
        "sample_summary": [{
            "seed": sample.seed,
            "variable_count": len(sample.variables),
            "constraint_count": len(sample.constraint_rhs),
            "edge_count": len(sample.edge_variable_indices),
            "positive_labels": int(np.sum(sample.labels)),
            "stage_one_mip_gap": sample.stage_one_mip_gap,
            "stage_two_mip_gap": sample.stage_two_mip_gap,
        } for sample in samples],
        "generation_seconds": generation_seconds,
        "training_seconds": training_seconds,
        "before_training": before,
        "after_training": after,
        "history": history,
        "overfit_gate": _build_overfit_gate(after),
    }
    versioned, latest = _save_report(payload, args.output)
    print(
        "after: "
        f"f1={after['f1']:.4f} "
        f"decoded_ratio={after['mean_decoded_throughput_ratio']:.4f} "
        f"objective_optimal="
        f"{after['lexicographic_objective_optimal_rate']:.4f} "
        f"raw_feasible={after['raw_threshold_feasible_rate']:.4f} "
        f"projected_feasible="
        f"{after['post_projection_feasible_rate']:.4f} "
        f"gate={payload['overfit_gate']['passed']}"
    )
    print(f"json: {versioned}")
    print(f"latest: {latest}")
    return 0 if payload["overfit_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
