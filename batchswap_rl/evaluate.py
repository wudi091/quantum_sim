from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np
import torch

from .model import DynamicPlanActorCritic
from .ppo import act
from .trainer import Environment, load_model, unpack_reset, unpack_step


class Controller(Protocol):
    def reset(self, seed: int | None = None) -> None: ...

    def act(self, env: Environment, observation: Mapping[str, Any]) -> int: ...


@dataclass
class LearnedController:
    model: DynamicPlanActorCritic
    device: torch.device
    deterministic: bool = True

    def reset(self, seed: int | None = None) -> None:
        del seed
        return None

    def act(self, env: Environment, observation: Mapping[str, Any]) -> int:
        action, _, _ = act(self.model, observation, self.device, self.deterministic)
        return action


@dataclass
class FunctionController:
    function: Callable[[Environment, Mapping[str, Any]], int]

    def reset(self, seed: int | None = None) -> None:
        del seed
        return None

    def act(self, env: Environment, observation: Mapping[str, Any]) -> int:
        return int(self.function(env, observation))


class NamedBaselineController:
    """Resolve the repository's standard greedy or Q-DDCA baseline lazily."""

    _CANDIDATES = {
        "greedy": ("GreedyPolicy", "GreedyController", "greedy_action", "greedy_policy"),
        "qddca": ("QDDCAPolicy", "QDDCAController", "qddca_action", "qddca_policy"),
        "random": ("RandomValidPolicy", "RandomPolicy", "random_action", "random_policy"),
    }

    def __init__(self, name: str):
        normalized = name.lower().replace("-", "")
        if normalized not in self._CANDIDATES:
            raise ValueError(f"unknown baseline {name!r}")
        self.name = normalized
        self.policy: Any = None
        self.episode_seed: int | None = None

    def reset(self, seed: int | None = None) -> None:
        self.episode_seed = seed
        if self.policy is not None and hasattr(self.policy, "reset"):
            self._reset_policy()

    def _reset_policy(self) -> None:
        reset = self.policy.reset
        parameters = list(inspect.signature(reset).parameters.values())
        if parameters:
            reset(self.episode_seed)
        else:
            reset()

    def _resolve(self, env: Environment) -> Any:
        if self.policy is not None:
            return self.policy
        target = getattr(env, "environment", env)
        is_reliq = target.__class__.__module__.startswith("batchswap_reliq")
        module_order = (
            ("batchswap_reliq.baselines", "batchswap_rl.baselines")
            if is_reliq else
            ("batchswap_rl.baselines", "batchswap_reliq.baselines")
        )
        for module_name in module_order:
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                continue
            for attribute in self._CANDIDATES[self.name]:
                if hasattr(module, attribute):
                    policy = getattr(module, attribute)
                    self.policy = policy() if inspect.isclass(policy) else policy
                    if hasattr(self.policy, "reset"):
                        self._reset_policy()
                    return self.policy
        for attribute in (f"{self.name}_action", f"select_{self.name}_action"):
            if hasattr(env, attribute):
                return getattr(env, attribute)
        raise RuntimeError(
            f"could not find a {self.name} baseline in batchswap_rl.baselines or on the env"
        )

    @staticmethod
    def _call(callable_policy: Any, env: Environment, observation: Mapping[str, Any]) -> int:
        function = callable_policy.act if hasattr(callable_policy, "act") else callable_policy
        parameters = list(inspect.signature(function).parameters.values())
        positional = [
            item
            for item in parameters
            if item.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) >= 2:
            return int(function(env, observation))
        if len(positional) == 1:
            name = positional[0].name.lower()
            return int(function(observation if "obs" in name else env))
        return int(function())

    def act(self, env: Environment, observation: Mapping[str, Any]) -> int:
        return self._call(self._resolve(env), env, observation)


def run_episode(
    env: Environment,
    controller: Controller,
    seed: int,
    max_decisions: int = 100_000,
) -> dict[str, Any]:
    observation, reset_info = unpack_reset(env.reset(seed=seed))
    controller.reset(seed)
    total_reward = 0.0
    decisions = 0
    final_info: dict[str, Any] = dict(reset_info)
    while decisions < max_decisions:
        action = controller.act(env, observation)
        observation, reward, terminated, truncated, info = unpack_step(env.step(action))
        total_reward += reward
        decisions += 1
        final_info = info
        if terminated or truncated:
            break
    else:
        raise RuntimeError(f"evaluation exceeded {max_decisions} action decisions")
    requests = getattr(getattr(env, "instance", None), "requests", ())
    completed_at = getattr(env, "completed_at", {})
    expired_at = getattr(env, "expired_at", {})
    delays = [
        completed_at[request.id] - request.arrival
        for request in requests
        if request.id in completed_at
    ]
    offered = len(requests)
    completion_rate = len(delays) / max(offered, 1)
    timeout_rate = len(expired_at) / max(offered, 1)
    pending = max(0, offered - len(delays) - len(expired_at))
    request_ttl = getattr(getattr(env, "config", None), "request_ttl", None)
    capped_delays = []
    for request in requests:
        if request.id in completed_at:
            capped_delays.append(completed_at[request.id] - request.arrival)
        elif request_ttl is not None:
            capped_delays.append(int(request_ttl))
        else:
            capped_delays.append(max(0, int(final_info.get("time", 0)) - request.arrival))
    success_makespan = max(completed_at.values(), default=0)
    request_outcomes = []
    for request in requests:
        if request.id in completed_at:
            status = "completed"
            finish_time = int(completed_at[request.id])
        elif request.id in expired_at:
            status = "expired"
            finish_time = int(expired_at[request.id])
        else:
            status = "pending"
            finish_time = int(final_info.get("time", 0))
        request_outcomes.append({
            "request_id": request.id,
            "hops": int(request.hops),
            "status": status,
            "finish_time": finish_time,
            "delay": finish_time - int(request.arrival),
        })
    return {
        "return": total_reward,
        "decisions": decisions,
        **final_info,
        "completion": completion_rate,
        "completion_rate": completion_rate,
        "timeout_rate": timeout_rate,
        "offered": offered,
        "expired": len(expired_at),
        "pending": pending,
        "pending_rate": pending / max(offered, 1),
        "request_delays": [float(delay) for delay in delays],
        "request_outcomes": request_outcomes,
        "mean_success_delay": float(np.mean(delays)) if delays else 0.0,
        "p95_success_delay": float(np.percentile(delays, 95)) if delays else 0.0,
        "p95_delay": float(np.percentile(delays, 95)) if delays else 0.0,
        "mean_ttl_capped_delay": float(np.mean(capped_delays)) if capped_delays else 0.0,
        "success_makespan": float(success_makespan),
        "episode_end_time": float(final_info.get("time", 0.0)),
        "makespan": float(final_info.get("time", 0.0)),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted(
        set.intersection(
            *(
                {
                    key
                    for key, value in row.items()
                    if isinstance(value, (int, float, bool, np.number))
                }
                for row in rows
            )
        )
    )
    result = {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}
    pooled_delays = [
        float(delay) for row in rows for delay in row.get("request_delays", [])
    ]
    if pooled_delays:
        result["p95_delay"] = float(np.percentile(pooled_delays, 95))
        result["mean_delay"] = float(np.mean(pooled_delays))
        result["p95_success_delay"] = result["p95_delay"]
        result["mean_success_delay"] = result["mean_delay"]
    return result


def paired_evaluation(
    env_factory: Callable[[int], Environment],
    controllers: Mapping[str, Controller],
    seeds: list[int],
) -> dict[str, Any]:
    """Evaluate learned, Q-DDCA, and greedy controllers on identical seeds."""
    raw: dict[str, list[dict[str, Any]]] = {name: [] for name in controllers}
    for seed in seeds:
        for name, controller in controllers.items():
            raw[name].append(run_episode(env_factory(seed), controller, seed))
    summary = {name: _aggregate(rows) for name, rows in raw.items()}
    core_keys = (
        "completion", "timeout_rate", "pending_rate", "mean_ttl_capped_delay",
        "mean_success_delay", "p95_success_delay", "success_makespan",
        "episode_end_time", "return",
    )
    return {
        "seeds": seeds,
        "summary": summary,
        "core_summary": {
            name: {key: metrics[key] for key in core_keys if key in metrics}
            for name, metrics in summary.items()
        },
        "episodes": raw,
    }


def _checkpoint_reward(checkpoint: Path):
    from .env import RewardConfig

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ppo = payload.get("ppo_config", {})
    reward = ppo.get("reward", {})
    values = {
        "flow_time_weight": 1.0,
        "gamma": ppo.get("gamma", 0.99),
        "potential_coef": reward.get("potential_coef", 1.0),
        "makespan_coef": reward.get("makespan_coef", 0.0),
        "elementary_epr_coef": reward.get("elementary_epr_coef", 0.0),
        "swap_coef": reward.get("swap_coef", 0.0),
        "completion_bonus": reward.get("completion_bonus", 0.0),
    }
    supported = inspect.signature(RewardConfig).parameters
    return RewardConfig(**{key: value for key, value in values.items() if key in supported})


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired RL/Q-DDCA/greedy evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend", choices=("batchswap", "reliq"), default="batchswap")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--stage", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--request-ttl",
        type=int,
        help="Fixed request lifetime in RELiQ physical steps; independent of hop count.",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if args.request_ttl is not None and args.request_ttl < 1:
        raise ValueError("request TTL must be positive")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    reward_config = None if args.backend == "reliq" else _checkpoint_reward(args.checkpoint)
    reliq_prototype = None
    if args.backend == "reliq":
        from .reliq_adapter import make_reliq_env

        # Topology construction is the expensive part of RELiQ.  Build it once,
        # then deep-copy the pristine cached template so every paired controller
        # sees the same topology and independent episode state.
        reliq_prototype = make_reliq_env(
            stage=args.stage, seed=args.seed, request_ttl=args.request_ttl
        )
        reliq_prototype.reset(seed=args.seed)

    def env_factory(seed: int):
        if args.backend == "reliq":
            assert reliq_prototype is not None
            return copy.deepcopy(reliq_prototype)
        from .env import make_env

        return make_env(stage=args.stage, seed=seed, reward_config=reward_config)

    result = paired_evaluation(
        env_factory,
        {
            "rl": LearnedController(model, device),
            "qddca": NamedBaselineController("qddca"),
            "greedy": NamedBaselineController("greedy"),
            "random": NamedBaselineController("random"),
        },
        list(range(args.seed, args.seed + args.episodes)),
    )
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps(result["core_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
