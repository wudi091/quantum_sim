"""Paired PPO/baseline evaluation on the single SeQUeNCe environment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import torch

from qnet_core.gym_env import GymConfig, SequenceGymEnv
from qnet_core.planners import GreedyPlanner, QCASTPlanner, QDDCAPlanner, RandomPlanner
from qnet_core.reward import RewardConfig
from qnet_core.scenario import ScenarioConfig
from qnet_core.spec import PhysicalConfig

from .model import DynamicPlanActorCritic
from .ppo import act
from .trainer import load_model, unpack_reset, unpack_step


class Controller(Protocol):
    def reset(self, seed: int) -> None: ...
    def act(self, env: SequenceGymEnv, observation: Mapping[str, Any]) -> int: ...


@dataclass
class LearnedController:
    model: DynamicPlanActorCritic
    device: torch.device

    def reset(self, seed: int) -> None:
        del seed

    def act(self, env: SequenceGymEnv, observation: Mapping[str, Any]) -> int:
        del env
        action, _, _ = act(self.model, observation, self.device, deterministic=True)
        return action


class PlannerController:
    def __init__(self, planner: object):
        self.planner = planner
        self.pending: list[int] = []

    def reset(self, seed: int) -> None:
        self.pending = []
        self.planner.reset(seed)

    def act(self, env: SequenceGymEnv, observation: Mapping[str, Any]) -> int:
        del observation
        while self.pending:
            action = self.pending.pop(0)
            if env.action_mask()[action]:
                return action
        selected = tuple(self.planner.select(env.snapshot))
        by_id = {
            plan.plan_id: action for action, plan in enumerate(env.slots)
            if plan is not None
        }
        self.pending = [by_id[plan_id] for plan_id in selected if plan_id in by_id]
        self.pending.append(env.stop_action)
        return self.act(env, {})


def make_env(args: argparse.Namespace, seed: int) -> SequenceGymEnv:
    scenario = ScenarioConfig(
        request_count=args.requests,
        min_hops=args.min_hops,
        max_hops=args.max_hops,
        ttl=args.request_ttl,
        horizon=args.request_ttl,
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
    return SequenceGymEnv(GymConfig(
        max_requests=args.requests,
        max_candidates_per_request=args.candidates_per_request,
        max_hops=args.max_hops,
        scenario=scenario,
        seed=seed,
        reward=RewardConfig(
            potential_coef=args.potential_coef,
            completion_bonus=args.completion_bonus,
            makespan_coef=args.makespan_coef,
            failure_coef=args.failure_coef,
            timeout_coef=args.timeout_coef,
        ),
    ))


def run_episode(env: SequenceGymEnv, controller: Controller, seed: int) -> dict[str, float]:
    observation, _ = unpack_reset(env.reset(seed=seed))
    controller.reset(seed)
    total_reward = 0.0
    decisions = 0
    final_info: dict[str, Any] = {}
    while True:
        action = controller.act(env, observation)
        observation, reward, terminated, truncated, info = unpack_step(env.step(action))
        total_reward += reward
        decisions += 1
        final_info = info
        if terminated or truncated:
            break
    result = {
        key: float(value)
        for key, value in {
            **final_info,
            "return": total_reward,
            "decisions": decisions,
        }.items()
        if isinstance(value, (int, float, bool, np.number))
    }
    attempts = result.get("successful_plans", 0.0) + result.get("failed_plans", 0.0)
    successful = result.get("successful_plans", 0.0)
    result["plan_success_rate"] = successful / attempts if attempts else 0.0
    result["net_progress_per_plan"] = (
        result.get("progress_hops", 0.0) / attempts if attempts else 0.0
    )
    result["positive_progress_per_plan"] = (
        result.get("positive_progress_hops", 0.0) / attempts if attempts else 0.0
    )
    result["lost_progress_per_plan"] = (
        result.get("lost_progress_hops", 0.0) / attempts if attempts else 0.0
    )
    result["net_progress_per_successful_plan"] = (
        result.get("progress_hops", 0.0) / successful if successful else 0.0
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair SeQUeNCe PPO/Q-DDCA evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--min-hops", type=int, default=2)
    parser.add_argument("--max-hops", type=int, default=50)
    parser.add_argument("--request-ttl", type=int, default=64)
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(value < 0 for value in (
        args.potential_coef, args.completion_bonus,
        args.makespan_coef, args.failure_coef, args.timeout_coef,
    )):
        raise ValueError("reward coefficients must be non-negative")
    if args.topology_nodes is not None and args.topology_nodes <= args.max_hops:
        raise ValueError("topology nodes must exceed max_hops")
    if args.waxman_alpha <= 0 or not 0 < args.waxman_beta <= 1:
        raise ValueError("invalid Waxman alpha or beta")
    if args.topology_attempts < 1:
        raise ValueError("topology attempts must be positive")
    if args.demand_pairs < 1 or args.max_width < 1 or args.candidates_per_request < 1:
        raise ValueError("demand, max width, and candidate count must be positive")
    if args.node_memory_capacity is not None and args.node_memory_capacity < 1:
        raise ValueError("node memory capacity must be positive")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    controllers: dict[str, Controller] = {
        "ppo": LearnedController(model, device),
        "qddca": PlannerController(QDDCAPlanner()),
        "qcast": PlannerController(QCASTPlanner()),
        "greedy": PlannerController(GreedyPlanner()),
        "random": PlannerController(RandomPlanner(0)),
    }
    rows: dict[str, list[dict[str, float]]] = {name: [] for name in controllers}
    for seed in range(args.seed, args.seed + args.episodes):
        for name, controller in controllers.items():
            rows[name].append(run_episode(make_env(args, seed), controller, seed))
    summary = {
        name: {
            key: float(np.mean([row[key] for row in values]))
            for key in values[0]
        }
        for name, values in rows.items()
    }
    result = {"summary": summary, "episodes": rows}
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
