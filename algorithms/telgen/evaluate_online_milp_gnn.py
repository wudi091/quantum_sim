"""Evaluate one frozen autoregressive GNN on exact online MILP labels."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from typing import Mapping

from .gnn_policy import OnlineGNNPolicy, torch
from .milp_oracle import has_numerically_zero_mip_gap
from .online_milp_dataset import load_online_milp_dataset
from .gnn_evaluation import evaluate_gnn
from .train_online_milp_gnn import _random_feasible_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a frozen masked-autoregressive GNN on a disjoint "
            "exact-MILP dataset without training or checkpoint selection."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--evaluation-batch-size", type=int, default=2)
    parser.add_argument("--random-baseline-trials", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=20260814)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_seen_episode_seeds(
    checkpoint: Mapping[str, object],
) -> tuple[int, ...]:
    split = checkpoint.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("checkpoint does not record its episode split")
    seen: set[int] = set()
    for key in ("train_seeds", "validation_seeds", "test_seeds"):
        values = split.get(key)
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"checkpoint split is missing {key}")
        seen.update(int(value) for value in values)
    return tuple(sorted(seen))


def evaluate_frozen_checkpoint(
    checkpoint_path: Path,
    dataset_path: Path,
    *,
    device_name: str,
    evaluation_batch_size: int,
    random_baseline_trials: int,
    random_seed: int,
) -> dict[str, object]:
    if evaluation_batch_size < 1:
        raise ValueError("evaluation batch size must be positive")
    if random_baseline_trials < 1:
        raise ValueError("random baseline trials must be positive")
    policy = OnlineGNNPolicy.from_checkpoint(
        checkpoint_path,
        device=device_name,
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("GNN checkpoint must contain a mapping")
    training_objective = checkpoint.get("training_objective")
    if not isinstance(training_objective, Mapping):
        raise ValueError("checkpoint is missing its training objective")
    target_mode = str(training_objective.get("target_mode", ""))
    if target_mode not in {"set", "fixed_order"}:
        raise ValueError("checkpoint has an unsupported target mode")

    loaded = load_online_milp_dataset(dataset_path)
    if any(
        not has_numerically_zero_mip_gap(sample.stage_one_mip_gap)
        or not has_numerically_zero_mip_gap(sample.stage_two_mip_gap)
        for sample in loaded.samples
    ):
        raise RuntimeError(
            "evaluation dataset contains a MILP label without certified "
            "numerical optimality"
        )
    seen_seeds = set(_checkpoint_seen_episode_seeds(checkpoint))
    evaluation_seeds = set(loaded.episode_seeds)
    overlap = sorted(seen_seeds & evaluation_seeds)
    if overlap:
        raise RuntimeError(
            "evaluation episode seed overlaps checkpoint data: "
            f"{overlap[0]}"
        )

    metrics = evaluate_gnn(
        policy.model,
        loaded.samples,
        device=policy.device,
        sample_batch_size=evaluation_batch_size,
        target_mode=target_mode,
    )
    random_baseline = _random_feasible_baseline(
        loaded.samples,
        trials=random_baseline_trials,
        random_seed=random_seed,
    )
    return {
        "schema_version": 1,
        "experiment": "frozen_online_milp_gnn_evaluation",
        "evaluation_contract": {
            "checkpoint_frozen": True,
            "training_or_checkpoint_selection_performed": False,
            "evaluation_seeds_disjoint_from_checkpoint": True,
            "ground_truth": "numerically optimal two-stage 0/1 MILP",
            "online_milp_called": False,
        },
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_seen_episode_seeds": sorted(seen_seeds),
        "dataset_manifest": str(loaded.manifest_path),
        "evaluation_episode_seeds": sorted(evaluation_seeds),
        "sample_count": len(loaded.samples),
        "target_mode": target_mode,
        "device": str(policy.device),
        "metrics": metrics,
        "random_feasible_baseline": random_baseline,
        "comparison": {
            "pooled_throughput_ratio_gain_over_random": (
                float(metrics["pooled_throughput_ratio"])
                - float(random_baseline["mean_pooled_throughput_ratio"])
            ),
            "mean_throughput_ratio_gain_over_random": (
                float(metrics["mean_throughput_ratio"])
                - float(random_baseline["mean_sample_throughput_ratio"])
            ),
        },
    }


def _save_report(
    output: Path,
    payload: dict[str, object],
) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = ""
    collision_index = 1
    while (output / f"frozen_gnn_evaluation_{timestamp}{suffix}.json").exists():
        collision_index += 1
        suffix = f"_{collision_index}"
    versioned = output / f"frozen_gnn_evaluation_{timestamp}{suffix}.json"
    latest = output / "frozen_gnn_evaluation.json"
    versioned.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    shutil.copyfile(versioned, latest)
    return versioned, latest


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_frozen_checkpoint(
        args.checkpoint,
        args.dataset,
        device_name=args.device,
        evaluation_batch_size=args.evaluation_batch_size,
        random_baseline_trials=args.random_baseline_trials,
        random_seed=args.random_seed,
    )
    versioned, latest = _save_report(args.output, report)
    metrics = report["metrics"]
    random_baseline = report["random_feasible_baseline"]
    print(
        "frozen_gnn: "
        f"pooled_throughput_ratio={float(metrics['pooled_throughput_ratio']):.4f} "
        f"mean_throughput_ratio={float(metrics['mean_throughput_ratio']):.4f} "
        f"throughput_optimal={float(metrics['throughput_optimal_rate']):.4f} "
        f"feasible={float(metrics['selection_feasible_rate']):.4f}"
    )
    print(
        "random_feasible: "
        f"pooled_throughput_ratio="
        f"{float(random_baseline['mean_pooled_throughput_ratio']):.4f}"
    )
    print(f"report: {versioned}")
    print(f"latest: {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
