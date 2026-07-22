"""Repository-owned configuration for the formal 2--50 hop training run.

Run with ``python -m routing_rl.large_scale``.  Training parameters live here
rather than in an ad-hoc shell command, so the formal experiment is reviewable
and reproducible.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path

import torch

from .train import parse_args, run


@dataclass(frozen=True)
class LargeScaleSettings:
    output: Path = Path("results/gnn_large_scale_seed67001_u200")
    init_checkpoint: Path = Path(
        "results/gnn_small_continue_seed66001_u70/best.pt"
    )
    seed: int = 67_001
    min_hops: int = 2
    max_hops: int = 50
    updates: int = 200
    requests: int = 20
    rollout_steps: int = 512
    minibatch_size: int = 128
    ppo_epochs: int = 4
    hidden_dim: int = 128
    learning_rate: float = 1e-4
    value_coef: float = 0.5
    entropy_coef: float = 1e-3
    gamma: float = 0.99
    checkpoint_every: int = 10
    evaluate_every: int = 10
    evaluation_episodes: int = 10
    high_hop_evaluation_episodes: int = 10
    high_hop_min_hops: int = 41
    early_stopping_patience: int = 5
    allow_scratch_without_checkpoint: bool = False
    request_ttl: int = 64
    generation_probability: float = 0.5
    swap_probability: float = 0.95
    memory_capacity: int = 2
    arrival_rate: float = 1.0
    topology_nodes: int = 200
    waxman_alpha: float = 0.05
    waxman_beta: float = 0.02
    topology_attempts: int = 128
    demand_pairs: int = 1
    max_width: int = 1
    candidates_per_request: int = 6
    potential_coef: float = 0.1
    completion_bonus: float = 1.0
    makespan_coef: float = 0.005
    failure_coef: float = 0.1
    timeout_coef: float = 0.1
    torch_threads: int = 4


SETTINGS = LargeScaleSettings()


def build_args(settings: LargeScaleSettings = SETTINGS):
    """Overlay the formal preset on the normal parser defaults."""
    args = parse_args([])
    runtime_only = {"allow_scratch_without_checkpoint"}
    for field in fields(settings):
        if field.name in runtime_only:
            continue
        setattr(args, field.name, getattr(settings, field.name))
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.anneal_learning_rate = True
    args.curriculum = False
    args.select_high_hop = False
    args.reset_critic = False
    args.smoke = False
    return args


def prepare_initialization(args) -> bool:
    """Use the optional warm-start checkpoint only when it is available."""
    checkpoint = args.init_checkpoint
    if checkpoint is None or checkpoint.is_file():
        return checkpoint is not None
    args.init_checkpoint = None
    return False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate and print the formal configuration without training.",
    )
    operational = parser.parse_args(argv)
    args = build_args()
    requested_checkpoint = args.init_checkpoint
    warm_start = prepare_initialization(args)
    if requested_checkpoint is not None and not warm_start:
        message = (
            "initialization checkpoint is unavailable: "
            f"{requested_checkpoint}"
        )
        if not SETTINGS.allow_scratch_without_checkpoint:
            print(f"error: {message}")
            print(
                "formal training is configured to require warm start; "
                "place the checkpoint at the configured path or explicitly set "
                "allow_scratch_without_checkpoint=True in large_scale.py"
            )
            if operational.check:
                return
            raise FileNotFoundError(message)
        print(f"warning: {message}; training will start from scratch")
    print(
        "formal large-scale training: "
        f"device={args.device}, updates={args.updates}, "
        f"rollout_steps={args.rollout_steps}, warm_start={warm_start}, "
        f"output={args.output}"
    )
    if operational.check:
        return
    run(args)


if __name__ == "__main__":
    main()
