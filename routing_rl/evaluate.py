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
from qnet_core.planners import GreedyPlanner, QDDCAPlanner, RandomPlanner
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
        if self.pending:
            return self.pending.pop(0)
        selected = tuple(self.planner.select(env.snapshot))
        by_id = {
            plan.plan_id: action for action, plan in enumerate(env.slots)
            if plan is not None
        }
        self.pending = [by_id[plan_id] for plan_id in selected if plan_id in by_id]
        self.pending.append(env.stop_action)
        return self.pending.pop(0)


def make_env(args: argparse.Namespace, seed: int) -> SequenceGymEnv:
    scenario = ScenarioConfig(
        request_count=args.requests,
        min_hops=args.min_hops,
        max_hops=args.max_hops,
        ttl=args.request_ttl,
        horizon=args.request_ttl,
        arrival_rate=args.arrival_rate,
        physical=PhysicalConfig(
            generation_probability=args.generation_probability,
            swap_probability=args.swap_probability,
            memory_capacity=args.memory_capacity,
        ),
    )
    return SequenceGymEnv(GymConfig(
        max_requests=args.requests,
        max_candidates_per_request=3,
        max_hops=args.max_hops,
        scenario=scenario,
        seed=seed,
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
    return {
        key: float(value)
        for key, value in {
            **final_info,
            "return": total_reward,
            "decisions": decisions,
        }.items()
        if isinstance(value, (int, float, bool, np.number))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair SeQUeNCe PPO/Q-DDCA evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--min-hops", type=int, default=20)
    parser.add_argument("--max-hops", type=int, default=50)
    parser.add_argument("--request-ttl", type=int, default=32)
    parser.add_argument("--generation-probability", type=float, default=0.5)
    parser.add_argument("--swap-probability", type=float, default=0.5)
    parser.add_argument("--memory-capacity", type=int, default=2)
    parser.add_argument("--arrival-rate", type=float, default=1.0,
                        help="Mean Poisson request arrivals per physical step")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    controllers: dict[str, Controller] = {
        "ppo": LearnedController(model, device),
        "qddca": PlannerController(QDDCAPlanner()),
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
