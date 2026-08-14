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
from typing import Sequence

import networkx as nx
import numpy as np
import scipy
import torch

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
from .hard_decoder import greedy_feasible_projection
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
    parser.add_argument("--training-seed", type=int, default=20260813)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        help="Explicit validation episode seeds; requires --test-seeds.",
    )
    parser.add_argument(
        "--test-seeds",
        type=int,
        nargs="+",
        help="Explicit held-out test episode seeds; requires --validation-seeds.",
    )
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--evaluation-batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--random-baseline-trials", type=int, default=32)
    parser.add_argument(
        "--target-mode",
        choices=("set", "fixed_order"),
        default="set",
    )
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


def _resolve_episode_split(
    seeds: tuple[int, ...],
    *,
    validation_fraction: float,
    test_fraction: float,
    random_seed: int,
    validation_seeds: Sequence[int] | None = None,
    test_seeds: Sequence[int] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if validation_seeds is None and test_seeds is None:
        return _split_episode_seeds(
            seeds,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            random_seed=random_seed,
        )
    if validation_seeds is None or test_seeds is None:
        raise ValueError(
            "explicit episode split requires both validation and test seeds"
        )
    available = set(int(seed) for seed in seeds)
    validation = tuple(sorted(set(int(seed) for seed in validation_seeds)))
    test = tuple(sorted(set(int(seed) for seed in test_seeds)))
    if not validation or not test:
        raise ValueError("explicit validation and test splits cannot be empty")
    if len(validation) != len(tuple(validation_seeds)):
        raise ValueError("validation seeds must be unique")
    if len(test) != len(tuple(test_seeds)):
        raise ValueError("test seeds must be unique")
    unknown = (set(validation) | set(test)) - available
    if unknown:
        raise ValueError(f"unknown explicit episode seed: {min(unknown)}")
    if set(validation) & set(test):
        raise ValueError("validation and test episode seeds overlap")
    train = tuple(sorted(available - set(validation) - set(test)))
    if not train:
        raise ValueError("explicit episode split leaves no training episodes")
    return train, validation, test


def _random_feasible_baseline(
    samples: Sequence[MILPGraphSample],
    *,
    trials: int,
    random_seed: int,
) -> dict[str, float | int]:
    if not samples:
        raise ValueError("random baseline requires at least one sample")
    if trials < 1:
        raise ValueError("random baseline trials must be positive")
    trial_pooled_ratios = []
    sample_ratios = []
    feasible_count = 0
    decision_count = 0
    optimal_total_mass = float(sum(
        sample.optimal_expected_completed_request_mass
        for sample in samples
    ))
    for trial_index in range(trials):
        selected_total_mass = 0.0
        for sample_index, sample in enumerate(samples):
            rng = np.random.default_rng(np.random.SeedSequence((
                int(random_seed),
                int(trial_index),
                int(sample_index),
            )))
            projected = greedy_feasible_projection(
                sample.variables,
                sample.resource_capacities,
                rng.random(len(sample.variables)),
                request_ids=sample.request_ids,
                reserved_usage=sample.reserved_usage,
                support_tolerance=0.0,
            )
            selected_mass = projected.expected_completed_request_mass
            selected_total_mass += selected_mass
            optimal_mass = sample.optimal_expected_completed_request_mass
            sample_ratios.append(
                selected_mass / optimal_mass if optimal_mass else 1.0
            )
            feasible_count += int(projected.feasibility.feasible)
            decision_count += 1
        trial_pooled_ratios.append(
            selected_total_mass / optimal_total_mass
            if optimal_total_mass else 1.0
        )
    return {
        "trials": trials,
        "mean_sample_throughput_ratio": float(np.mean(sample_ratios)),
        "mean_pooled_throughput_ratio": float(np.mean(trial_pooled_ratios)),
        "p05_pooled_throughput_ratio": float(np.percentile(
            trial_pooled_ratios,
            5,
        )),
        "p95_pooled_throughput_ratio": float(np.percentile(
            trial_pooled_ratios,
            95,
        )),
        "feasible_rate": feasible_count / max(decision_count, 1),
    }


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def _validation_key(
    metrics: dict[str, object],
) -> tuple[float, float, float, float]:
    """Select checkpoints by pooled throughput before per-graph averages."""

    latency_gap = metrics[
        "mean_latency_relative_gap_when_throughput_optimal"
    ]
    return (
        float(metrics["pooled_throughput_ratio"]),
        float(metrics["mean_throughput_ratio"]),
        float("-inf") if latency_gap is None else -float(latency_gap),
        -float(metrics["loss"]),
    )


def _save_outputs(
    output: Path,
    report: dict[str, object],
    checkpoint: dict[str, object],
) -> tuple[Path, Path, Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = ""
    collision_index = 1
    while (output / f"online_milp_gnn_{timestamp}{suffix}.json").exists():
        collision_index += 1
        suffix = f"_{collision_index}"
    report_versioned = output / f"online_milp_gnn_{timestamp}{suffix}.json"
    report_latest = output / "online_milp_gnn.json"
    checkpoint_versioned = output / f"online_milp_gnn_{timestamp}{suffix}.pt"
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
    if args.batch_size < 1 or args.evaluation_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.random_baseline_trials < 1:
        raise ValueError("random-baseline-trials must be positive")
    _seed_everything(args.training_seed)
    device = _resolve_device(args.device)
    loaded = load_online_milp_dataset(args.dataset)
    train_seeds, validation_seeds, test_seeds = _resolve_episode_split(
        loaded.episode_seeds,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        random_seed=args.training_seed,
        validation_seeds=args.validation_seeds,
        test_seeds=args.test_seeds,
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
            device=device,
            sample_batch_size=args.evaluation_batch_size,
            target_mode=args.target_mode,
        ),
        "validation": _evaluate(
            model,
            validation_samples,
            device=device,
            sample_batch_size=args.evaluation_batch_size,
            target_mode=args.target_mode,
        ),
        "test": _evaluate(
            model,
            test_samples,
            device=device,
            sample_batch_size=args.evaluation_batch_size,
            target_mode=args.target_mode,
        ),
    }
    random_baseline = {
        "train": _random_feasible_baseline(
            train_samples,
            trials=args.random_baseline_trials,
            random_seed=args.training_seed + 101,
        ),
        "validation": _random_feasible_baseline(
            validation_samples,
            trials=args.random_baseline_trials,
            random_seed=args.training_seed + 202,
        ),
        "test": _random_feasible_baseline(
            test_samples,
            trials=args.random_baseline_trials,
            random_seed=args.training_seed + 303,
        ),
    }
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    best_validation_key = _validation_key(before["validation"])
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
            optimizer.zero_grad(set_to_none=True)
            loss, parts = autoregressive_set_loss(
                model,
                batch_samples,
                device=device,
                target_mode=args.target_mode,
            )
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
                device=device,
                sample_batch_size=args.evaluation_batch_size,
                target_mode=args.target_mode,
            )
            validation_loss = float(validation["loss"])
            history.append({
                "epoch": epoch,
                "train_loss": mean_train_loss,
                "train_candidate_set_nll": mean_train_parts[
                    "candidate_set_nll"
                ],
                "train_stop_nll": mean_train_parts["stop_nll"],
                "train_valid_candidate_fraction": mean_train_parts[
                    "valid_candidate_fraction"
                ],
                "train_masked_candidate_fraction": mean_train_parts[
                    "masked_candidate_fraction"
                ],
                "validation_loss": validation_loss,
                "validation_f1": float(validation["f1"]),
                "validation_throughput_ratio": float(
                    validation["mean_throughput_ratio"]
                ),
                "validation_pooled_throughput_ratio": float(
                    validation["pooled_throughput_ratio"]
                ),
            })
            print(
                f"epoch={epoch} train_loss={mean_train_loss:.6f} "
                f"validation_loss={validation_loss:.6f} "
                f"validation_f1={float(validation['f1']):.4f} "
                f"pooled_throughput_ratio="
                f"{float(validation['pooled_throughput_ratio']):.4f} "
                f"mean_throughput_ratio="
                f"{float(validation['mean_throughput_ratio']):.4f} "
                f"masked={mean_train_parts['masked_candidate_fraction']:.4f}",
                flush=True,
            )
            validation_key = _validation_key(validation)
            if validation_key > best_validation_key:
                best_validation_key = validation_key
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
    after = {
        "train": _evaluate(
            model,
            train_samples,
            device=device,
            sample_batch_size=args.evaluation_batch_size,
            target_mode=args.target_mode,
        ),
        "validation": _evaluate(
            model,
            validation_samples,
            device=device,
            sample_batch_size=args.evaluation_batch_size,
            target_mode=args.target_mode,
        ),
        "test": _evaluate(
            model,
            test_samples,
            device=device,
            sample_batch_size=args.evaluation_batch_size,
            target_mode=args.target_mode,
        ),
    }
    try:
        sequence_version = package_version("sequence")
    except PackageNotFoundError:
        sequence_version = "unknown"
    report = {
        "schema_version": AUTOREGRESSIVE_CHECKPOINT_SCHEMA_VERSION,
        "experiment": (
            "online_masked_autoregressive_milp_"
            f"{args.target_mode}_imitation"
        ),
        "architecture": AUTOREGRESSIVE_ARCHITECTURE,
        "supervision": (
            (
                "unordered exact-MILP selected set"
                if args.target_mode == "set"
                else "one deterministic variable-ID ordering of the MILP set"
            )
            + " plus terminal STOP over the dynamically feasible candidate "
            "action set"
        ),
        "training_objective": {
            "target_mode": args.target_mode,
            "candidate_target": (
                "combined probability mass of all remaining teacher actions"
                if args.target_mode == "set"
                else "one deterministic variable-ID order"
            ),
            "feasibility": (
                "exact packing mask before categorical normalization"
            ),
        },
        "checkpoint_selection": {
            "primary": "validation_pooled_throughput_ratio",
            "tie_breakers": [
                "validation_mean_throughput_ratio",
                "negative_validation_latency_relative_gap",
                "negative_validation_loss",
            ],
        },
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
            "strategy": (
                "explicit_held_out_episodes"
                if args.validation_seeds is not None
                else "seeded_random_episodes"
            ),
            "train_seeds": list(train_seeds),
            "validation_seeds": list(validation_seeds),
            "test_seeds": list(test_seeds),
            "train_sample_count": len(train_samples),
            "validation_sample_count": len(validation_samples),
            "test_sample_count": len(test_samples),
        },
        "best_epoch": best_epoch,
        "training_seconds": training_seconds,
        "before_training": before,
        "after_training": after,
        "random_feasible_baseline": random_baseline,
        "history": history,
    }
    checkpoint = {
        "schema_version": AUTOREGRESSIVE_CHECKPOINT_SCHEMA_VERSION,
        "model_class": "CandidateConstraintGNN",
        "architecture": AUTOREGRESSIVE_ARCHITECTURE,
        "model_config": {
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
        },
        "state_dict": best_state,
        "feature_schema": report["feature_schema"],
        "dataset_manifest": str(loaded.manifest_path),
        "split": report["split"],
        "best_epoch": best_epoch,
        "training_objective": report["training_objective"],
        "checkpoint_selection": report["checkpoint_selection"],
    }
    paths = _save_outputs(args.output, report, checkpoint)
    test = after["test"]
    print(
        "test: "
        f"f1={float(test['f1']):.4f} "
        f"pooled_throughput_ratio="
        f"{float(test['pooled_throughput_ratio']):.4f} "
        f"mean_throughput_ratio="
        f"{float(test['mean_throughput_ratio']):.4f} "
        f"throughput_optimal={float(test['throughput_optimal_rate']):.4f} "
        f"feasible={float(test['selection_feasible_rate']):.4f}"
    )
    print(f"report: {paths[0]}")
    print(f"checkpoint: {paths[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
