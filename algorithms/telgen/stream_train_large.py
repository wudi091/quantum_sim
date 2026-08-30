"""Memory-bounded training for the existing MILP-imitation GNN.

This experiment runner keeps the model and objective identical to
``train_online_milp_gnn`` but reads persisted graph samples in small chunks.
It is intended for the large suite, whose complete in-memory load can exceed
the available RAM.  No labels, architecture, masking, or decoding rules are
changed here.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
from time import perf_counter
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from .gnn_evaluation import evaluate_gnn, seed_everything
from .milp_imitation import (
    AUTOREGRESSIVE_ARCHITECTURE,
    AUTOREGRESSIVE_CHECKPOINT_SCHEMA_VERSION,
    CONSTRAINT_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    VARIABLE_FEATURE_NAMES,
    CandidateConstraintGNN,
    MILPGraphSample,
    autoregressive_set_loss,
)
from .milp_oracle import has_numerically_zero_mip_gap
from .online_milp_dataset import (
    _sample_paths_from_manifest,
    load_online_milp_graph_sample,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the existing GNN with bounded-memory dataset loading."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--training-seed", type=int, default=20260821)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    return parser


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def _suite_payload(path: Path) -> dict[str, object]:
    source = path if path.is_file() else path / "online_milp_dataset.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("dataset_kind") != "online_milp_teacher_collection":
        raise ValueError("dataset must be a combined online MILP collection")
    if payload.get("collection_complete", True) is not True:
        raise ValueError("dataset collection is incomplete")
    return payload


def _role_paths(
    dataset: Path,
    role: str,
    *,
    limit: int | None = None,
) -> tuple[Path, ...]:
    source = dataset if dataset.is_file() else dataset / "online_milp_dataset.json"
    payload = _suite_payload(source)
    base = source.parent
    paths: list[Path] = []
    for group in payload["episodes"]:
        if group["role"] != role:
            continue
        manifest = (base / str(group["manifest"])).resolve()
        paths.extend(_sample_paths_from_manifest(manifest))
    if limit is not None:
        if limit < 1:
            raise ValueError("sample limits must be positive")
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"dataset has no {role} samples")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return tuple(paths)


def _load_chunk(paths: Sequence[Path]) -> tuple[MILPGraphSample, ...]:
    samples = tuple(load_online_milp_graph_sample(path) for path in paths)
    if any(
        not has_numerically_zero_mip_gap(sample.stage_one_mip_gap)
        or not has_numerically_zero_mip_gap(sample.stage_two_mip_gap)
        for sample in samples
    ):
        raise RuntimeError("dataset contains a non-certified MILP label")
    return samples


def _metric_key(metrics: Mapping[str, object]) -> tuple[float, float, float, float]:
    latency = metrics.get("mean_latency_relative_gap_when_throughput_optimal")
    return (
        float(metrics["pooled_throughput_ratio"]),
        float(metrics["mean_throughput_ratio"]),
        float("-inf") if latency is None else -float(latency),
        -float(metrics["loss"]),
    )


def _stream_evaluate(
    model: CandidateConstraintGNN,
    paths: Sequence[Path],
    *,
    device: torch.device,
    chunk_size: int,
    batch_size: int,
) -> dict[str, object]:
    """Evaluate in chunks and combine exactly the metrics used for selection."""

    if not paths:
        raise ValueError("evaluation requires at least one path")
    model.eval()
    total_n = 0
    total_loss = 0.0
    total_pred_mass = 0.0
    total_opt_mass = 0.0
    total_ratio = 0.0
    min_ratio = float("inf")
    total_optimal = 0
    total_feasible = 0
    total_lexicographic = 0
    total_tp = total_fp = total_fn = total_tn = 0
    latency_sum = 0.0
    latency_count = 0
    action_count = 0
    for start in range(0, len(paths), chunk_size):
        samples = _load_chunk(paths[start:start + chunk_size])
        metrics = evaluate_gnn(
            model,
            samples,
            device=device,
            sample_batch_size=batch_size,
            target_mode="set",
        )
        n = len(samples)
        total_n += n
        total_loss += float(metrics["loss"]) * n
        total_ratio += float(metrics["mean_throughput_ratio"]) * n
        min_ratio = min(min_ratio, float(metrics["minimum_throughput_ratio"]))
        total_optimal += int(round(float(metrics["throughput_optimal_rate"]) * n))
        total_feasible += int(round(float(metrics["selection_feasible_rate"]) * n))
        total_lexicographic += int(
            round(float(metrics["lexicographic_objective_optimal_rate"]) * n)
        )
        total_pred_mass += sum(
            float(item["expected_completed_request_mass"])
            for item in metrics["selections"]
        )
        total_opt_mass += sum(
            float(item["optimal_expected_completed_request_mass"])
            for item in metrics["selections"]
        )
        total_tp += int(metrics["true_positive"])
        total_fp += int(metrics["false_positive"])
        total_fn += int(metrics["false_negative"])
        total_tn += int(metrics["true_negative"])
        for item in metrics["selections"]:
            gap = item["latency_relative_gap_when_throughput_optimal"]
            if gap is not None:
                latency_sum += float(gap)
                latency_count += 1
            action_count += int(item["action_count"])
        del samples
        gc.collect()
    if total_n == 0 or total_opt_mass <= 0.0:
        raise RuntimeError("evaluation produced no valid samples")
    return {
        "sample_count": total_n,
        "loss": total_loss / total_n,
        "pooled_throughput_ratio": total_pred_mass / total_opt_mass,
        "mean_throughput_ratio": total_ratio / total_n,
        "minimum_throughput_ratio": min_ratio,
        "throughput_optimal_rate": total_optimal / total_n,
        "selection_feasible_rate": total_feasible / total_n,
        "lexicographic_objective_optimal_rate": total_lexicographic / total_n,
        "mean_latency_relative_gap_when_throughput_optimal": (
            latency_sum / latency_count if latency_count else None
        ),
        "true_positive": total_tp,
        "false_positive": total_fp,
        "false_negative": total_fn,
        "true_negative": total_tn,
        "f1": 2.0 * total_tp / max(2 * total_tp + total_fp + total_fn, 1),
        "action_count": action_count,
    }


def _save_result(
    output: Path,
    model: CandidateConstraintGNN,
    report: dict[str, object],
    *,
    split: dict[str, object],
    dataset: Path,
    hidden_dim: int,
    layers: int,
    best_epoch: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": AUTOREGRESSIVE_CHECKPOINT_SCHEMA_VERSION,
        "model_class": "CandidateConstraintGNN",
        "architecture": AUTOREGRESSIVE_ARCHITECTURE,
        "model_config": {"hidden_dim": hidden_dim, "layers": layers},
        "state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "feature_schema": {
            "variable": list(VARIABLE_FEATURE_NAMES),
            "constraint": list(CONSTRAINT_FEATURE_NAMES),
            "global": list(GLOBAL_FEATURE_NAMES),
        },
        "dataset_manifest": str(dataset),
        "split": split,
        "best_epoch": best_epoch,
        "training_objective": {
            "target_mode": "set",
            "candidate_target": (
                "combined probability mass of all remaining teacher actions"
            ),
            "feasibility": "exact packing mask before categorical normalization",
        },
        "checkpoint_selection": {
            "primary": "validation_pooled_throughput_ratio",
            "tie_breakers": [
                "validation_mean_throughput_ratio",
                "negative_validation_latency_relative_gap",
                "negative_validation_loss",
            ],
        },
    }
    (output / "stream_training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save(checkpoint, output / "online_milp_gnn.pt")


def _json_configuration(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    """Convert CLI values to a JSON-safe provenance record."""

    values: dict[str, object] = {}
    for key, value in vars(args).items():
        values[key] = str(value) if isinstance(value, Path) else value
    values["device"] = str(device)
    return values


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    if args.batch_size < 1 or args.chunk_size < args.batch_size:
        raise ValueError("chunk-size must be at least batch-size")
    seed_everything(args.training_seed)
    device = _resolve_device(args.device)
    train_paths = _role_paths(args.dataset, "train", limit=args.max_train_samples)
    validation_paths = _role_paths(
        args.dataset, "validation", limit=args.max_validation_samples
    )
    test_paths = _role_paths(args.dataset, "test", limit=args.max_test_samples)
    payload = _suite_payload(args.dataset)
    split_payload = payload.get("split_episode_seeds", {})
    split = {
        "unit": "episode",
        "strategy": "explicit_held_out_episodes",
        "train_seeds": list(split_payload.get("train", [])),
        "validation_seeds": list(split_payload.get("validation", [])),
        "test_seeds": list(split_payload.get("test", [])),
        "train_sample_count": len(train_paths),
        "validation_sample_count": len(validation_paths),
        "test_sample_count": len(test_paths),
    }

    model = CandidateConstraintGNN(
        hidden_dim=args.hidden_dim,
        layers=args.layers,
    ).to(device)
    resumed_epoch = 0
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(
            args.resume_checkpoint, map_location=device, weights_only=True
        )
        config = checkpoint.get("model_config", {})
        if int(config.get("hidden_dim", args.hidden_dim)) != args.hidden_dim:
            raise ValueError("resume checkpoint hidden dimension differs")
        if int(config.get("layers", args.layers)) != args.layers:
            raise ValueError("resume checkpoint layer count differs")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        resumed_epoch = int(checkpoint.get("best_epoch", 0))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    best_validation = _stream_evaluate(
        model,
        validation_paths,
        device=device,
        chunk_size=args.chunk_size,
        batch_size=args.evaluation_batch_size,
    )
    best_key = _metric_key(best_validation)
    best_epoch = resumed_epoch
    stale = 0
    history: list[dict[str, object]] = []
    rng = random.Random(args.training_seed)
    started = perf_counter()
    for local_epoch in range(1, args.epochs + 1):
        model.train()
        order = list(train_paths)
        rng.shuffle(order)
        epoch_loss = 0.0
        seen = 0
        for start in range(0, len(order), args.chunk_size):
            chunk_paths = order[start:start + args.chunk_size]
            samples = _load_chunk(chunk_paths)
            for batch_start in range(0, len(samples), args.batch_size):
                batch = samples[batch_start:batch_start + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss, _ = autoregressive_set_loss(
                    model, batch, device=device, target_mode="set"
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                count = len(batch)
                epoch_loss += float(loss.detach()) * count
                seen += count
            del samples
            gc.collect()
        global_epoch = resumed_epoch + local_epoch
        if local_epoch == 1 or local_epoch % 5 == 0 or local_epoch == args.epochs:
            validation = _stream_evaluate(
                model,
                validation_paths,
                device=device,
                chunk_size=args.chunk_size,
                batch_size=args.evaluation_batch_size,
            )
            history.append({
                "epoch": global_epoch,
                "train_loss": epoch_loss / max(seen, 1),
                "validation_loss": validation["loss"],
                "validation_pooled_throughput_ratio": validation[
                    "pooled_throughput_ratio"
                ],
                "validation_mean_throughput_ratio": validation[
                    "mean_throughput_ratio"
                ],
            })
            print(
                f"epoch={global_epoch} train_loss={epoch_loss / max(seen, 1):.6f} "
                f"validation_loss={float(validation['loss']):.6f} "
                f"pooled_throughput_ratio="
                f"{float(validation['pooled_throughput_ratio']):.4f}",
                flush=True,
            )
            key = _metric_key(validation)
            if key > best_key:
                best_key = key
                best_epoch = global_epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 5
                if stale >= args.patience:
                    break
    model.load_state_dict(best_state, strict=True)
    test = _stream_evaluate(
        model,
        test_paths,
        device=device,
        chunk_size=args.chunk_size,
        batch_size=args.evaluation_batch_size,
    )
    report = {
        "schema_version": AUTOREGRESSIVE_CHECKPOINT_SCHEMA_VERSION,
        "experiment": "streamed_large_online_milp_gnn_training",
        "architecture": AUTOREGRESSIVE_ARCHITECTURE,
        "training_objective": "unchanged existing set imitation objective",
        "dataset": str(args.dataset),
        "configuration": _json_configuration(args, device),
        "split": split,
        "resumed_epoch": resumed_epoch,
        "best_epoch": best_epoch,
        "training_seconds": perf_counter() - started,
        "history": history,
        "final_test": test,
    }
    _save_result(
        args.output,
        model,
        report,
        split=split,
        dataset=args.dataset,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        best_epoch=best_epoch,
    )
    print(
        f"test: pooled_throughput_ratio={float(test['pooled_throughput_ratio']):.4f} "
        f"mean_throughput_ratio={float(test['mean_throughput_ratio']):.4f} "
        f"feasible={float(test['selection_feasible_rate']):.4f}",
        flush=True,
    )
    print(f"report: {args.output / 'stream_training_report.json'}")
    print(f"checkpoint: {args.output / 'online_milp_gnn.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
