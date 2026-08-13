"""Train the candidate--constraint GNN from persisted online MILP graphs."""

from __future__ import annotations

import argparse
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version as package_version
import json
from pathlib import Path
import random
import shutil
import sys
from time import perf_counter

import networkx as nx
import numpy as np
import scipy
import torch

from .milp_imitation import (
    CONSTRAINT_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    VARIABLE_FEATURE_NAMES,
    CandidateConstraintGNN,
    batch_graph_samples,
    greedy_decode_scores,
    imitation_loss,
)
from .online_milp_dataset import (
    load_online_milp_dataset,
    samples_for_episode_seeds,
)
from .milp_oracle import has_numerically_zero_mip_gap
from .train_milp_imitation import _evaluate, _seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a GNN from persisted online exact-MILP labels."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--training-seed", type=int, default=20260813)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--evaluation-batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def _split_episode_seeds(
    seeds: tuple[int, ...],
    *,
    validation_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if len(seeds) < 3:
        raise ValueError(
            "online dataset training requires at least three episodes"
        )
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation-fraction must lie in (0, 1)")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test-fraction must lie in (0, 1)")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation and test fractions must sum below one")
    shuffled = list(seeds)
    random.Random(random_seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    test_count = max(1, round(len(shuffled) * test_fraction))
    while validation_count + test_count >= len(shuffled):
        if validation_count >= test_count and validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise ValueError("not enough episodes for non-empty splits")
    validation = tuple(sorted(shuffled[:validation_count]))
    test = tuple(sorted(
        shuffled[validation_count:validation_count + test_count]
    ))
    train = tuple(sorted(shuffled[validation_count + test_count:]))
    if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
        raise RuntimeError("episode split leakage detected")
    return train, validation, test


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def _calibrate_decode_threshold(
    model: CandidateConstraintGNN,
    samples,
    *,
    device: torch.device,
    sample_batch_size: int,
) -> tuple[float, list[dict[str, float]]]:
    """Choose a support threshold on validation episodes only.

    Candidate probabilities are primarily ranking scores: one selected plan
    competes with many time-shift alternatives, so they need not be calibrated
    above 0.5.  The threshold only decides which candidates enter the feasible
    greedy packing; resource constraints remain enforced by the decoder.
    """

    model.eval()
    all_probabilities: list[np.ndarray] = []
    probability_slices: list[tuple[int, int]] = []
    probability_offset = 0
    for batch_start in range(0, len(samples), sample_batch_size):
        batch_samples = samples[batch_start:batch_start + sample_batch_size]
        graph = batch_graph_samples(batch_samples, device=device)
        with torch.no_grad():
            batch_probabilities = torch.sigmoid(model(graph)).cpu().numpy()
        all_probabilities.append(batch_probabilities)
        for start, end in graph.variable_slices:
            probability_slices.append((
                probability_offset + start,
                probability_offset + end,
            ))
        probability_offset += len(batch_probabilities)
    probabilities = np.concatenate(all_probabilities)
    quantiles = np.quantile(probabilities, np.linspace(0.0, 1.0, 41))
    thresholds = sorted({
        0.0,
        0.5,
        *(float(value) for value in quantiles),
    })
    rows: list[dict[str, float]] = []
    best_threshold = 0.0
    best_key: tuple[float, ...] | None = None
    for threshold in thresholds:
        ratios = []
        optimal = []
        latency_gaps = []
        for sample, (start, end) in zip(samples, probability_slices):
            decoded = greedy_decode_scores(
                sample,
                probabilities[start:end],
                threshold=threshold,
            )
            target = sample.optimal_expected_completed_request_mass
            ratio = (
                decoded.expected_completed_request_mass / target
                if target > 0.0 else 1.0
            )
            is_optimal = abs(
                decoded.expected_completed_request_mass - target
            ) <= 1e-7
            ratios.append(ratio)
            optimal.append(float(is_optimal))
            if is_optimal:
                latency_gaps.append(
                    (
                        decoded.total_completion_latency
                        - sample.optimal_total_completion_latency
                    )
                    / max(sample.optimal_total_completion_latency, 1.0)
                )
        mean_ratio = float(np.mean(ratios))
        minimum_ratio = float(np.min(ratios))
        optimal_rate = float(np.mean(optimal))
        mean_latency_gap = (
            float(np.mean(latency_gaps)) if latency_gaps else float("inf")
        )
        rows.append({
            "threshold": threshold,
            "mean_decoded_throughput_ratio": mean_ratio,
            "minimum_decoded_throughput_ratio": minimum_ratio,
            "throughput_optimal_rate": optimal_rate,
            "mean_latency_relative_gap_when_throughput_optimal": (
                mean_latency_gap
            ),
        })
        key = (
            mean_ratio,
            minimum_ratio,
            optimal_rate,
            -mean_latency_gap,
            -threshold,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
    return best_threshold, rows


def _save_outputs(
    output: Path,
    report: dict[str, object],
    checkpoint: dict[str, object],
) -> tuple[Path, Path, Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_versioned = output / f"online_milp_gnn_{timestamp}.json"
    report_latest = output / "online_milp_gnn.json"
    checkpoint_versioned = output / f"online_milp_gnn_{timestamp}.pt"
    checkpoint_latest = output / "online_milp_gnn.pt"
    report_versioned.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    shutil.copyfile(report_versioned, report_latest)
    torch.save(checkpoint, checkpoint_versioned)
    shutil.copyfile(checkpoint_versioned, checkpoint_latest)
    return (
        report_versioned,
        report_latest,
        checkpoint_versioned,
        checkpoint_latest,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("optimizer hyperparameters are invalid")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    if args.batch_size < 1 or args.evaluation_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    _seed_everything(args.training_seed)
    device = _resolve_device(args.device)
    loaded = load_online_milp_dataset(args.dataset)
    train_seeds, validation_seeds, test_seeds = _split_episode_seeds(
        loaded.episode_seeds,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        random_seed=args.training_seed,
    )
    train_samples = samples_for_episode_seeds(loaded.samples, train_seeds)
    validation_samples = samples_for_episode_seeds(
        loaded.samples, validation_seeds
    )
    test_samples = samples_for_episode_seeds(loaded.samples, test_seeds)
    if not train_samples or not validation_samples or not test_samples:
        raise RuntimeError("episode split produced an empty graph split")
    if any(
        not has_numerically_zero_mip_gap(sample.stage_one_mip_gap)
        or not has_numerically_zero_mip_gap(sample.stage_two_mip_gap)
        for sample in loaded.samples
    ):
        raise RuntimeError(
            "dataset contains a MILP label without certified numerical "
            "optimality"
        )

    model = CandidateConstraintGNN(
        hidden_dim=args.hidden_dim,
        layers=args.layers,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    before = {
        "train": _evaluate(
            model,
            train_samples,
            threshold=args.threshold,
            device=device,
            sample_batch_size=args.evaluation_batch_size,
        ),
        "validation": _evaluate(
            model,
            validation_samples,
            threshold=args.threshold,
            device=device,
            sample_batch_size=args.evaluation_batch_size,
        ),
        "test": _evaluate(
            model,
            test_samples,
            threshold=args.threshold,
            device=device,
            sample_batch_size=args.evaluation_batch_size,
        ),
    }
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    training_rng = random.Random(args.training_seed)
    started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_samples)))
        training_rng.shuffle(order)
        epoch_loss = 0.0
        epoch_parts: dict[str, float] = {}
        epoch_samples = 0
        for batch_start in range(0, len(order), args.batch_size):
            indices = order[batch_start:batch_start + args.batch_size]
            batch_samples = tuple(train_samples[index] for index in indices)
            graph = batch_graph_samples(batch_samples, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(graph)
            loss, parts = imitation_loss(logits, graph)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_sample_count = len(batch_samples)
            epoch_samples += batch_sample_count
            epoch_loss += float(loss.detach()) * batch_sample_count
            for key, value in parts.items():
                epoch_parts[key] = epoch_parts.get(key, 0.0) + (
                    float(value) * batch_sample_count
                )
        mean_train_loss = epoch_loss / max(epoch_samples, 1)
        mean_train_parts = {
            key: value / max(epoch_samples, 1)
            for key, value in epoch_parts.items()
        }

        validation_loss = None
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            validation = _evaluate(
                model,
                validation_samples,
                threshold=args.threshold,
                device=device,
                sample_batch_size=args.evaluation_batch_size,
            )
            validation_loss = float(validation["loss"])
            history.append({
                "epoch": epoch,
                "train_loss": mean_train_loss,
                "train_bce": mean_train_parts["bce"],
                "train_constraint_penalty": mean_train_parts[
                    "constraint_penalty"
                ],
                "train_expected_mass_penalty": mean_train_parts[
                    "expected_mass_penalty"
                ],
                "validation_loss": validation_loss,
                "validation_f1": float(validation["f1"]),
                "validation_decoded_throughput_ratio": float(
                    validation["mean_decoded_throughput_ratio"]
                ),
            })
            print(
                f"epoch={epoch} train_loss={mean_train_loss:.6f} "
                f"validation_loss={validation_loss:.6f} "
                f"validation_f1={float(validation['f1']):.4f} "
                f"decoded_ratio={float(validation['mean_decoded_throughput_ratio']):.4f}",
                flush=True,
            )
            if validation_loss < best_validation_loss - 1e-8:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 5
                if stale_epochs >= args.patience:
                    break
    training_seconds = perf_counter() - started
    model.load_state_dict(best_state)
    decode_threshold, threshold_calibration = _calibrate_decode_threshold(
        model,
        validation_samples,
        device=device,
        sample_batch_size=args.evaluation_batch_size,
    )
    after = {
        "train": _evaluate(
            model,
            train_samples,
            threshold=args.threshold,
            decode_threshold=decode_threshold,
            device=device,
            sample_batch_size=args.evaluation_batch_size,
        ),
        "validation": _evaluate(
            model,
            validation_samples,
            threshold=args.threshold,
            decode_threshold=decode_threshold,
            device=device,
            sample_batch_size=args.evaluation_batch_size,
        ),
        "test": _evaluate(
            model,
            test_samples,
            threshold=args.threshold,
            decode_threshold=decode_threshold,
            device=device,
            sample_batch_size=args.evaluation_batch_size,
        ),
    }
    try:
        sequence_version = package_version("sequence")
    except PackageNotFoundError:
        sequence_version = "unknown"
    report = {
        "schema_version": 1,
        "experiment": "online_exact_milp_gnn_imitation",
        "supervision": "exact two-stage MILP final binary primal",
        "dataset_manifest": str(loaded.manifest_path),
        "configuration": {
            **vars(args),
            "dataset": str(args.dataset),
            "output": str(args.output),
            "device": str(device),
        },
        "runtime_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "networkx": nx.__version__,
            "sequence": sequence_version,
        },
        "feature_schema": {
            "variable": list(VARIABLE_FEATURE_NAMES),
            "constraint": list(CONSTRAINT_FEATURE_NAMES),
            "global": list(GLOBAL_FEATURE_NAMES),
        },
        "split": {
            "unit": "episode",
            "train_seeds": list(train_seeds),
            "validation_seeds": list(validation_seeds),
            "test_seeds": list(test_seeds),
            "train_sample_count": len(train_samples),
            "validation_sample_count": len(validation_samples),
            "test_sample_count": len(test_samples),
        },
        "best_epoch": best_epoch,
        "decode_threshold": decode_threshold,
        "decode_threshold_selection": "validation_episode_lexicographic",
        "threshold_calibration": threshold_calibration,
        "training_seconds": training_seconds,
        "before_training": before,
        "after_training": after,
        "history": history,
    }
    checkpoint = {
        "schema_version": 1,
        "model_class": "CandidateConstraintGNN",
        "model_config": {
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
        },
        "state_dict": best_state,
        "classification_threshold": args.threshold,
        "decode_threshold": decode_threshold,
        "feature_schema": report["feature_schema"],
        "dataset_manifest": str(loaded.manifest_path),
        "split": report["split"],
        "best_epoch": best_epoch,
    }
    paths = _save_outputs(args.output, report, checkpoint)
    test = after["test"]
    print(
        "test: "
        f"f1={float(test['f1']):.4f} "
        f"decoded_ratio={float(test['mean_decoded_throughput_ratio']):.4f} "
        f"throughput_optimal={float(test['throughput_optimal_rate']):.4f} "
        f"feasible={float(test['post_projection_feasible_rate']):.4f}"
    )
    print(f"report: {paths[0]}")
    print(f"checkpoint: {paths[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
