"""Shared evaluation helpers for the autoregressive MILP-imitation GNN."""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch

from .milp_imitation import (
    CandidateConstraintGNN,
    MILPGraphSample,
    autoregressive_rollout,
    autoregressive_set_loss,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_gnn(
    model: CandidateConstraintGNN,
    samples: Sequence[MILPGraphSample],
    *,
    device: torch.device,
    sample_batch_size: int | None = None,
    target_mode: str = "set",
) -> dict[str, object]:
    """Evaluate the model's own discrete autoregressive selections."""

    model.eval()
    if not samples:
        raise ValueError("evaluation requires at least one sample")
    batch_size = len(samples) if sample_batch_size is None else int(
        sample_batch_size
    )
    if batch_size < 1:
        raise ValueError("sample batch size must be positive")
    tp = fp = fn = tn = 0
    variable_count = 0
    evaluated_sample_count = 0
    weighted_parts: dict[str, float] = {}
    selections = []
    for batch_start in range(0, len(samples), batch_size):
        batch_samples = samples[batch_start:batch_start + batch_size]
        with torch.no_grad():
            loss, parts = autoregressive_set_loss(
                model,
                batch_samples,
                device=device,
                target_mode=target_mode,
            )
        batch_sample_count = len(batch_samples)
        evaluated_sample_count += batch_sample_count
        weighted_parts["loss"] = weighted_parts.get("loss", 0.0) + (
            float(loss.detach()) * batch_sample_count
        )
        for key, value in parts.items():
            weighted_parts[key] = weighted_parts.get(key, 0.0) + (
                float(value) * batch_sample_count
            )
        for sample in batch_samples:
            rollout = autoregressive_rollout(model, sample, device=device)
            result = rollout.selection
            predicted_binary = np.zeros(
                len(sample.variables), dtype=np.bool_
            )
            predicted_binary[list(result.selected_indices)] = True
            true_binary = sample.labels >= 0.5
            tp += int(np.sum(predicted_binary & true_binary))
            fp += int(np.sum(predicted_binary & ~true_binary))
            fn += int(np.sum(~predicted_binary & true_binary))
            tn += int(np.sum(~predicted_binary & ~true_binary))
            variable_count += len(sample.variables)
            throughput_optimal = abs(
                result.expected_completed_request_mass
                - sample.optimal_expected_completed_request_mass
            ) <= 1e-7
            latency_gap = (
                result.total_completion_latency
                - sample.optimal_total_completion_latency
                if throughput_optimal else None
            )
            latency_relative_gap = (
                latency_gap / max(sample.optimal_total_completion_latency, 1.0)
                if latency_gap is not None else None
            )
            selections.append({
                "seed": sample.seed,
                "selection_feasible": result.feasible,
                "stopped_by_model": rollout.stopped_by_model,
                "action_count": result.action_count,
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
                        sample.variables,
                        sample.labels,
                        strict=True,
                    )
                    if label > 0.5
                },
            })
    comparable_latency_gaps = [
        item["latency_relative_gap_when_throughput_optimal"]
        for item in selections
        if item["latency_relative_gap_when_throughput_optimal"] is not None
    ]
    total_predicted_mass = float(sum(
        item["expected_completed_request_mass"] for item in selections
    ))
    total_optimal_mass = float(sum(
        item["optimal_expected_completed_request_mass"] for item in selections
    ))
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
        "positive_prediction_count": tp + fp,
        "positive_label_count": tp + fn,
        "mean_throughput_ratio": float(np.mean([
            item["throughput_ratio"] for item in selections
        ])),
        "pooled_throughput_ratio": (
            total_predicted_mass / total_optimal_mass
            if total_optimal_mass else 1.0
        ),
        "minimum_throughput_ratio": float(np.min([
            item["throughput_ratio"] for item in selections
        ])),
        "selection_feasible_rate": float(np.mean([
            item["selection_feasible"] for item in selections
        ])),
        "model_stop_rate": float(np.mean([
            item["stopped_by_model"] for item in selections
        ])),
        "throughput_optimal_rate": float(np.mean([
            item["throughput_optimal"] for item in selections
        ])),
        "mean_latency_relative_gap_when_throughput_optimal": (
            float(np.mean(comparable_latency_gaps))
            if comparable_latency_gaps else None
        ),
        "lexicographic_objective_optimal_rate": float(np.mean([
            item["lexicographic_objective_optimal"] for item in selections
        ])),
        "exact_selected_set_rate": float(np.mean([
            item["exact_selected_set"] for item in selections
        ])),
        "selections": selections,
    }


def build_overfit_gate(after: dict[str, object]) -> dict[str, object]:
    """Return a strict per-training-instance memorization gate."""

    tolerance = 1e-12
    passed = bool(
        float(after["minimum_throughput_ratio"]) >= 1.0 - tolerance
        and float(after["lexicographic_objective_optimal_rate"])
        >= 1.0 - tolerance
        and float(after["selection_feasible_rate"]) >= 1.0 - tolerance
    )
    return {
        "target_minimum_throughput_ratio": 1.0,
        "target_lexicographic_objective_optimal_rate": 1.0,
        "target_selection_feasible_rate": 1.0,
        "passed": passed,
    }
