from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import CurriculumStage, PPOConfig, RewardConfig
from .trainer import PPOTrainer


def build_evaluator(args: argparse.Namespace, config: PPOConfig):
    """Deterministic, fixed-seed validation used to select best.pt."""
    from qnet_core.gym_env import GymConfig, SequenceGymEnv
    from qnet_core.scenario import ScenarioConfig
    from qnet_core.spec import PhysicalConfig
    from .ppo import act
    from .trainer import unpack_reset, unpack_step

    keys = (
        "completion_rate", "timeout_rate", "mean_delay", "planning_slots",
        "successful_plans", "partial_plan_successes", "failed_plans",
        "progress_hops", "positive_progress_hops", "lost_progress_hops",
        "delivered_pairs", "pair_throughput", "recovery_attempts",
        "recovery_successes", "recovery_success_rate",
        "allocation_claims", "allocation_successes", "allocation_success_rate",
        "released_surplus_pairs", "epr_delivery_utilization",
    )

    def run_bucket(
        model, stage: CurriculumStage, min_hops: int, max_hops: int,
        episode_count: int, seed_offset: int,
    ) -> dict[str, float]:
        values: list[dict[str, float]] = []
        for offset in range(episode_count):
            seed = args.seed + seed_offset + offset
            scenario = ScenarioConfig(
                request_count=stage.max_requests,
                min_hops=min_hops,
                max_hops=max_hops,
                ttl=args.request_ttl or 64,
                horizon=args.request_ttl or 64,
                arrival_rate=args.arrival_rate,
                topology_nodes=args.topology_nodes,
                waxman_alpha=args.waxman_alpha,
                waxman_beta=args.waxman_beta,
                topology_attempts=args.topology_attempts,
                demand_pairs=args.demand_pairs,
                physical=PhysicalConfig(
                    generation_probability=args.generation_probability,
                    swap_probability=args.swap_probability,
                    memory_capacity=args.memory_capacity,
                    node_memory_capacity=args.node_memory_capacity,
                    max_width=args.max_width,
                ),
            )
            env = SequenceGymEnv(GymConfig(
                max_requests=stage.max_requests,
                max_candidates_per_request=args.candidates_per_request,
                max_hops=stage.max_hops,
                scenario=scenario,
                seed=seed,
                reward=config.reward,
                discount_gamma=config.gamma,
            ))
            observation, _ = unpack_reset(env.reset(seed=seed))
            while True:
                action, _, _ = act(model, observation, torch.device(config.device), deterministic=True)
                observation, _, terminated, truncated, info = unpack_step(env.step(action))
                if terminated or truncated:
                    values.append({
                        key: float(value) for key, value in info.items()
                        if isinstance(value, (int, float, bool, np.number))
                    })
                    break
        result = {
            key: float(np.mean([row.get(key, 0.0) for row in values]))
            for key in keys
        }
        rates = []
        for row in values:
            attempts = row.get("successful_plans", 0.0) + row.get("failed_plans", 0.0)
            rates.append(row.get("successful_plans", 0.0) / attempts if attempts else 0.0)
        result["plan_success_rate"] = float(np.mean(rates))
        return result

    def evaluate(model, update: int) -> dict[str, float]:
        cumulative = 0
        stage = config.curriculum[-1]
        for candidate in config.curriculum:
            cumulative += max(candidate.updates, 0)
            if update <= cumulative:
                stage = candidate
                break
        was_training = model.training
        model.eval()
        result = run_bucket(
            model, stage, stage.min_hops, stage.max_hops,
            args.evaluation_episodes, 1_000_000,
        )
        if args.high_hop_evaluation_episodes > 0:
            high_min = max(stage.min_hops, args.high_hop_min_hops)
            if high_min <= stage.max_hops:
                high = run_bucket(
                    model, stage, high_min, stage.max_hops,
                    args.high_hop_evaluation_episodes, 1_500_000,
                )
                result.update({f"high_hop_{key}": value for key, value in high.items()})
                if args.select_high_hop:
                    result["selection_score"] = (
                        high["completion_rate"]
                        + 1e-3 * high["pair_throughput"]
                    )
        if was_training:
            model.train()
        return result

    return evaluate

def build_environment(
    config: PPOConfig,
    request_ttl: int | None = None,
    generation_probability: float = 0.5,
    swap_probability: float = 0.95,
    memory_capacity: int = 2,
    arrival_rate: float = 1.0,
    topology_nodes: int | None = None,
    waxman_alpha: float = 0.05,
    waxman_beta: float = 0.02,
    topology_attempts: int = 128,
    demand_pairs: int = 1,
    node_memory_capacity: int | None = None,
    max_width: int = 1,
    candidates_per_request: int = 6,
):
    """Build the single shared SeQUeNCe environment used by every planner."""
    from qnet_core.gym_env import GymConfig, SequenceGymEnv
    from qnet_core.scenario import ScenarioConfig
    from qnet_core.spec import PhysicalConfig

    maximum_requests = max(stage.max_requests for stage in config.curriculum)
    maximum_hops = max(stage.max_hops for stage in config.curriculum)
    first = config.curriculum[0]
    scenario = ScenarioConfig(
        request_count=first.max_requests,
        min_hops=first.min_hops,
        max_hops=first.max_hops,
        ttl=request_ttl or 64,
        horizon=max(request_ttl or 64, 1),
        arrival_rate=arrival_rate,
        topology_nodes=topology_nodes,
        waxman_alpha=waxman_alpha,
        waxman_beta=waxman_beta,
        topology_attempts=topology_attempts,
        demand_pairs=demand_pairs,
        physical=PhysicalConfig(
            generation_probability=generation_probability,
            swap_probability=swap_probability,
            memory_capacity=memory_capacity,
            node_memory_capacity=node_memory_capacity,
            max_width=max_width,
        ),
    )
    return SequenceGymEnv(GymConfig(
        max_requests=maximum_requests,
        max_candidates_per_request=candidates_per_request,
        max_hops=maximum_hops,
        scenario=scenario,
        seed=config.seed,
        reward=config.reward,
        discount_gamma=config.gamma,
    ))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train masked PPO on the shared SeQUeNCe environment")
    parser.add_argument("--output", type=Path, default=Path("routing_rl/runs/default"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-hops", type=int, default=2)
    parser.add_argument("--max-hops", type=int, default=50)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Use the legacy short/medium/long curriculum instead of one full-range stage.",
    )
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--anneal-learning-rate", action="store_true",
        help="Linearly decay the PPO learning rate to zero over all updates.",
    )
    parser.add_argument("--short-updates", type=int, default=200)
    parser.add_argument("--medium-updates", type=int, default=400)
    parser.add_argument("--long-updates", type=int, default=800)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--evaluate-every", type=int, default=10)
    parser.add_argument("--evaluation-episodes", type=int, default=10)
    parser.add_argument("--high-hop-evaluation-episodes", type=int, default=0)
    parser.add_argument("--high-hop-min-hops", type=int, default=41)
    parser.add_argument(
        "--select-high-hop",
        action="store_true",
        help="Select/early-stop on fixed high-hop completion while retaining best.pt overall.",
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=0,
        help="Stop after this many consecutive evaluations without improvement; 0 disables it.",
    )
    parser.add_argument(
        "--early-stopping-min-updates", type=int, default=0,
        help="Never early-stop before this many updates in the current stage.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Initialize model weights from a compatible checkpoint before curriculum training.",
    )
    parser.add_argument(
        "--reset-critic",
        action="store_true",
        help="Reinitialize the value head after loading a checkpoint with a different reward scale.",
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--request-ttl",
        type=int,
        default=64,
        help="Fixed request lifetime in shared physical steps; independent of hop count.",
    )
    parser.add_argument("--long-requests", type=int, default=20)
    parser.add_argument("--short-requests", type=int, default=4,
                        help="Requests per short-stage episode")
    parser.add_argument("--generation-probability", type=float, default=0.5)
    parser.add_argument("--swap-probability", type=float, default=0.95)
    parser.add_argument("--memory-capacity", type=int, default=2)
    parser.add_argument("--arrival-rate", type=float, default=1.0,
                        help="Mean Poisson request arrivals per physical step")
    parser.add_argument("--topology-nodes", type=int,
                        help="Waxman node count; defaults to max(4 * max_hops, 16)")
    parser.add_argument("--waxman-alpha", type=float, default=0.05)
    parser.add_argument("--waxman-beta", type=float, default=0.02)
    parser.add_argument("--topology-attempts", type=int, default=128)
    parser.add_argument("--demand-pairs", type=int, default=1)
    parser.add_argument("--node-memory-capacity", type=int)
    parser.add_argument("--max-width", type=int, default=1)
    parser.add_argument("--candidates-per-request", type=int, default=6)
    parser.add_argument("--potential-coef", type=float, default=0.1)
    parser.add_argument("--completion-bonus", type=float, default=1.0)
    parser.add_argument("--makespan-coef", type=float, default=0.005)
    parser.add_argument("--failure-coef", type=float, default=0.1)
    parser.add_argument("--timeout-coef", type=float, default=0.1)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one small update per configured training stage for integration testing",
    )
    return parser.parse_args(argv)


def make_config(args: argparse.Namespace) -> PPOConfig:
    rollout_steps = args.rollout_steps
    minibatch_size = args.minibatch_size
    if args.curriculum:
        updates = (args.short_updates, args.medium_updates, args.long_updates)
        curriculum = (
            CurriculumStage("short", 2, 5, updates[0], args.short_requests, args.short_requests),
            CurriculumStage("medium", 5, 15, updates[1], 5, 10),
            CurriculumStage("long", 20, 50, updates[2], args.long_requests, args.long_requests),
        )
    else:
        curriculum = (
            CurriculumStage(
                "full", args.min_hops, args.max_hops,
                args.updates, args.requests, args.requests,
            ),
        )
    if args.smoke:
        curriculum = tuple(
            CurriculumStage(
                stage.name, stage.min_hops, stage.max_hops, 1,
                stage.min_requests, stage.max_requests,
            )
            for stage in curriculum
        )
        rollout_steps = min(rollout_steps, 64)
        minibatch_size = min(minibatch_size, 32)
    return PPOConfig(
        hidden_dim=args.hidden_dim,
        rollout_steps=rollout_steps,
        learning_rate=args.learning_rate,
        anneal_learning_rate=args.anneal_learning_rate,
        gamma=args.gamma,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=minibatch_size,
        seed=args.seed,
        device=args.device,
        torch_threads=args.torch_threads,
        checkpoint_every=args.checkpoint_every,
        evaluate_every=args.evaluate_every,
        evaluation_episodes=args.evaluation_episodes,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_updates=args.early_stopping_min_updates,
        curriculum=curriculum,
        reward=RewardConfig(
            potential_coef=args.potential_coef,
            completion_bonus=args.completion_bonus,
            makespan_coef=args.makespan_coef,
            failure_coef=args.failure_coef,
            timeout_coef=args.timeout_coef,
        ),
    )


def run(args: argparse.Namespace) -> None:
    if args.request_ttl is not None and args.request_ttl < 1:
        raise ValueError("request TTL must be positive")
    if args.memory_capacity < 1 or args.arrival_rate <= 0:
        raise ValueError("memory capacity and arrival rate must be positive")
    if args.topology_nodes is not None and args.topology_nodes <= args.max_hops:
        raise ValueError("topology nodes must exceed max_hops")
    if args.waxman_alpha <= 0 or not 0 < args.waxman_beta <= 1:
        raise ValueError("invalid Waxman alpha or beta")
    if args.topology_attempts < 1:
        raise ValueError("topology attempts must be positive")
    if args.early_stopping_patience < 0:
        raise ValueError("early stopping patience must be non-negative")
    if args.early_stopping_min_updates < 0:
        raise ValueError("early stopping minimum updates must be non-negative")
    if args.evaluation_episodes < 1 or args.high_hop_evaluation_episodes < 0:
        raise ValueError("evaluation episode counts are invalid")
    if args.select_high_hop and args.high_hop_evaluation_episodes < 1:
        raise ValueError("--select-high-hop requires high-hop evaluation episodes")
    if args.high_hop_min_hops < 1 or (
        args.high_hop_evaluation_episodes > 0
        and args.high_hop_min_hops > args.max_hops
    ):
        raise ValueError("invalid high-hop minimum")
    if not 0 < args.gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    if args.value_coef < 0:
        raise ValueError("value coefficient must be non-negative")
    if args.entropy_coef < 0:
        raise ValueError("entropy coefficient must be non-negative")
    if args.demand_pairs < 1 or args.max_width < 1 or args.candidates_per_request < 1:
        raise ValueError("demand, max width, and candidate count must be positive")
    if args.node_memory_capacity is not None and args.node_memory_capacity < 1:
        raise ValueError("node memory capacity must be positive")
    if args.curriculum:
        if args.short_requests < 1 or args.long_requests < 1:
            raise ValueError("curriculum request counts must be positive")
    else:
        if args.min_hops < 1 or args.max_hops < args.min_hops:
            raise ValueError("invalid hop range")
        if args.updates < 1 or args.requests < 1:
            raise ValueError("updates and requests must be positive")
    if not 0 < args.generation_probability <= 1:
        raise ValueError("generation probability must be in (0, 1]")
    if not 0 <= args.swap_probability <= 1:
        raise ValueError("swap probability must be in [0, 1]")
    if any(value < 0 for value in (
        args.potential_coef, args.completion_bonus,
        args.makespan_coef, args.failure_coef, args.timeout_coef,
    )):
        raise ValueError("reward coefficients must be non-negative")
    config = make_config(args)
    env = build_environment(
        config,
        args.request_ttl,
        args.generation_probability,
        args.swap_probability,
        args.memory_capacity,
        args.arrival_rate,
        args.topology_nodes,
        args.waxman_alpha,
        args.waxman_beta,
        args.topology_attempts,
        args.demand_pairs,
        args.node_memory_capacity,
        args.max_width,
        args.candidates_per_request,
    )
    trainer = PPOTrainer(env, config, args.output, evaluator=build_evaluator(args, config))
    if args.init_checkpoint is not None:
        import torch

        payload = torch.load(args.init_checkpoint, map_location=trainer.device, weights_only=False)
        incompatible = trainer.model.load_state_dict(payload["model"], strict=False)
        unexpected = set(incompatible.unexpected_keys)
        allowed_missing = {
            "plan_message.weight", "plan_message.bias",
            "request_update.weight", "request_update.bias",
            "plan_update.weight", "plan_update.bias",
            "request_graph_gate", "plan_graph_gate",
        }
        missing = set(incompatible.missing_keys)
        if unexpected or not missing.issubset(allowed_missing):
            raise ValueError(
                f"incompatible initialization checkpoint: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        if args.reset_critic:
            trainer.model.critic.apply(trainer.model._initialize)
            torch.nn.init.orthogonal_(trainer.model.critic[-1].weight, gain=1.0)
            torch.nn.init.zeros_(trainer.model.critic[-1].bias)
    elif args.reset_critic:
        raise ValueError("--reset-critic requires --init-checkpoint")
    trainer.train()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
