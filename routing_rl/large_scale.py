"""Repository-owned configuration for the formal 2--50 hop training run.

Run with ``python -m routing_rl.large_scale``.  Training parameters live here
rather than in an ad-hoc shell command, so the formal experiment is reviewable
and reproducible.
"""

from __future__ import annotations

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
    for field in fields(settings):
        setattr(args, field.name, getattr(settings, field.name))
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.anneal_learning_rate = True
    args.curriculum = False
    args.select_high_hop = False
    args.early_stopping_patience = 0
    args.reset_critic = False
    args.smoke = False
    return args


def main() -> None:
    args = build_args()
    if not args.init_checkpoint.is_file():
        raise FileNotFoundError(
            "large-scale initialization checkpoint is missing: "
            f"{args.init_checkpoint}"
        )
    print(
        "formal large-scale training: "
        f"device={args.device}, updates={args.updates}, "
        f"rollout_steps={args.rollout_steps}, output={args.output}"
    )
    run(args)


if __name__ == "__main__":
    main()
