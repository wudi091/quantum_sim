from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any, TypeVar

from .config import CurriculumStage, PPOConfig, RewardConfig
from .reliq_adapter import make_reliq_env
from .trainer import PPOTrainer


T = TypeVar("T")


def _supported_instance(cls: type[T], values: dict[str, Any]) -> T:
    parameters = inspect.signature(cls).parameters
    return cls(**{key: value for key, value in values.items() if key in parameters})


def build_environment(config: PPOConfig):
    """Construct the simulator without coupling PPO to all env config fields."""
    from .env import BatchSwapEnv, EnvConfig, RewardConfig as EnvRewardConfig

    maximum_requests = max(stage.max_requests for stage in config.curriculum)
    maximum_hops = max(stage.max_hops for stage in config.curriculum)
    env_config = _supported_instance(
        EnvConfig,
        {
            "seed": config.seed,
            "max_requests": maximum_requests,
            "max_hops": maximum_hops,
        },
    )
    reward = config.reward
    reward_config = _supported_instance(
        EnvRewardConfig,
        {
            "gamma": config.gamma,
            "potential_coef": reward.potential_coef,
            "potential_weight": reward.potential_coef,
            "completion_bonus": reward.completion_bonus,
            "makespan_coef": reward.makespan_coef,
            "makespan_cost": reward.makespan_coef,
            "elementary_epr_coef": reward.elementary_epr_coef,
            "elementary_epr_weight": reward.elementary_epr_coef,
            "epr_cost": reward.elementary_epr_coef,
            "swap_coef": reward.swap_coef,
            "swap_weight": reward.swap_coef,
            "swap_cost": reward.swap_coef,
        },
    )
    return BatchSwapEnv(env_config, reward_config)


def build_reliq_environment(config: PPOConfig, request_ttl: int | None = None):
    """Build the optional physical RELIQ backend through its lazy adapter."""
    # The backend owns its EnvConfig and uses the same fixed feature dimensions
    # across curriculum stages.  Reward fields are passed when supported by a
    # future make_env implementation; the current factory intentionally keeps
    # this call dependency-light.
    return make_reliq_env(stage=0, seed=config.seed, request_ttl=request_ttl)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pure-RL masked PPO for BatchSwap")
    parser.add_argument("--output", type=Path, default=Path("batchswap_rl/runs/default"))
    parser.add_argument(
        "--backend",
        choices=("batchswap", "reliq"),
        default="batchswap",
        help="Environment backend; reliq is provided by batchswap_reliq.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--short-updates", type=int, default=200)
    parser.add_argument("--medium-updates", type=int, default=400)
    parser.add_argument("--long-updates", type=int, default=800)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Initialize model weights from a compatible checkpoint before curriculum training.",
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--request-ttl",
        type=int,
        help="Fixed request lifetime in RELiQ physical steps; independent of hop count.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one small update per curriculum stage for integration testing",
    )
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> PPOConfig:
    updates = (args.short_updates, args.medium_updates, args.long_updates)
    rollout_steps = args.rollout_steps
    minibatch_size = args.minibatch_size
    if args.smoke:
        updates = (1, 1, 1)
        rollout_steps = min(rollout_steps, 64)
        minibatch_size = min(minibatch_size, 32)
    curriculum = (
        CurriculumStage("short", 2, 5, updates[0], 2, 4),
        CurriculumStage("medium", 5, 15, updates[1], 5, 10),
        CurriculumStage("long", 20, 50, updates[2], 100, 100),
    )
    return PPOConfig(
        hidden_dim=args.hidden_dim,
        rollout_steps=rollout_steps,
        learning_rate=args.learning_rate,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=minibatch_size,
        seed=args.seed,
        device=args.device,
        torch_threads=args.torch_threads,
        checkpoint_every=args.checkpoint_every,
        curriculum=curriculum,
        reward=RewardConfig(),
    )


def main() -> None:
    args = parse_args()
    if args.request_ttl is not None and args.request_ttl < 1:
        raise ValueError("request TTL must be positive")
    config = make_config(args)
    env = (
        build_reliq_environment(config, args.request_ttl)
        if args.backend == "reliq" else build_environment(config)
    )
    trainer = PPOTrainer(env, config, args.output)
    if args.init_checkpoint is not None:
        import torch

        payload = torch.load(args.init_checkpoint, map_location=trainer.device, weights_only=False)
        trainer.model.load_state_dict(payload["model"])
    trainer.train()


if __name__ == "__main__":
    main()
