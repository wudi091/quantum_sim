"""Create a current-schema random-initialized GNN checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Mapping

import torch

from .milp_imitation import CandidateConstraintGNN


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the untrained initialization of a reference GNN "
            "checkpoint without loading its learned weights."
        )
    )
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialization-seed", type=int, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_untrained_checkpoint(
    reference_path: Path,
    output_path: Path,
    *,
    initialization_seed: int,
) -> dict[str, object]:
    reference = torch.load(
        reference_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(reference, Mapping):
        raise ValueError("reference checkpoint must contain a mapping")
    model_config = reference.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("reference checkpoint is missing model configuration")
    hidden_dim = int(model_config["hidden_dim"])
    layers = int(model_config["layers"])

    torch.manual_seed(initialization_seed)
    model = CandidateConstraintGNN(
        hidden_dim=hidden_dim,
        layers=layers,
    )
    checkpoint = {
        "schema_version": reference["schema_version"],
        "model_class": reference["model_class"],
        "architecture": reference["architecture"],
        "model_config": dict(model_config),
        "state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "feature_schema": reference["feature_schema"],
        "training_objective": reference.get("training_objective"),
        "checkpoint_kind": "untrained_random_initialization",
        "initialization_seed": initialization_seed,
        "gradient_update_count": 0,
        "reference_checkpoint": str(reference_path),
        "reference_checkpoint_sha256": _sha256(reference_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return checkpoint


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = create_untrained_checkpoint(
        args.reference_checkpoint,
        args.output,
        initialization_seed=args.initialization_seed,
    )
    print(f"checkpoint_kind={checkpoint['checkpoint_kind']}")
    print(f"initialization_seed={checkpoint['initialization_seed']}")
    print(f"gradient_update_count={checkpoint['gradient_update_count']}")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
