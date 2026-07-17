from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np
import torch

from .config import CurriculumStage, PPOConfig
from .model import DynamicPlanActorCritic
from .ppo import RolloutBuffer, Transition, act, parse_observation, ppo_update


class Environment(Protocol):
    def reset(self, **kwargs: Any) -> Any: ...

    def step(self, action: int) -> Any: ...


def unpack_reset(result: Any) -> tuple[Mapping[str, Any], dict[str, Any]]:
    if isinstance(result, tuple) and len(result) == 2:
        observation, info = result
        return observation, dict(info)
    return result, {}


def unpack_step(result: Any) -> tuple[Mapping[str, Any], float, bool, bool, dict[str, Any]]:
    if not isinstance(result, tuple):
        raise TypeError("env.step(action) must return a tuple")
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, float(reward), bool(terminated), bool(truncated), dict(info)
    if len(result) == 4:
        observation, reward, done, info = result
        truncated = bool(dict(info).get("TimeLimit.truncated", False))
        return observation, float(reward), bool(done and not truncated), truncated, dict(info)
    raise ValueError("env.step(action) must return four or five values")


def configure_curriculum(env: Environment, stage: CurriculumStage) -> None:
    """Apply a curriculum stage using either of the supported env hooks."""
    if hasattr(env, "set_curriculum"):
        env.set_curriculum(stage)
        return
    if hasattr(env, "configure_curriculum"):
        env.configure_curriculum(
            min_hops=stage.min_hops,
            max_hops=stage.max_hops,
            min_requests=stage.min_requests,
            max_requests=stage.max_requests,
        )
        return
    raise TypeError("environment must expose set_curriculum(stage) or configure_curriculum(...)")


@dataclass
class CollectionResult:
    rollout: RolloutBuffer
    observation: Mapping[str, Any]
    episodes: list[dict[str, Any]]
    mean_reward: float


@torch.no_grad()
def collect_rollout(
    env: Environment,
    model: DynamicPlanActorCritic,
    observation: Mapping[str, Any],
    steps: int,
    device: torch.device,
    next_episode_seed: Callable[[], int] | None = None,
) -> CollectionResult:
    rollout = RolloutBuffer()
    episodes: list[dict[str, Any]] = []
    episode_return = 0.0
    episode_length = 0

    for _ in range(steps):
        parsed = parse_observation(observation)
        action, log_prob, value = act(model, parsed, device)
        next_observation, reward, terminated, truncated, info = unpack_step(env.step(action))
        episode_done = terminated or truncated
        if terminated:
            next_value = 0.0
        else:
            _, _, next_value = act(model, next_observation, device, deterministic=True)
        rollout.add(
            Transition(
                observation=parsed,
                action=action,
                old_log_prob=log_prob,
                old_value=value,
                reward=reward,
                next_value=next_value,
                terminated=terminated,
                episode_done=episode_done,
                duration=float(info.get("duration", 1.0)),
            )
        )
        episode_return += reward
        episode_length += 1
        observation = next_observation
        if episode_done:
            row = {"episode_return": episode_return, "episode_length": episode_length}
            row.update(
                {
                    key: float(value)
                    for key, value in info.items()
                    if isinstance(value, (int, float, bool, np.number))
                }
            )
            episodes.append(row)
            reset_seed = None if next_episode_seed is None else next_episode_seed()
            observation, _ = unpack_reset(env.reset(seed=reset_seed))
            episode_return = 0.0
            episode_length = 0

    return CollectionResult(
        rollout=rollout,
        observation=observation,
        episodes=episodes,
        mean_reward=float(np.mean([item.reward for item in rollout.transitions])),
    )


def _mean_episode_metrics(episodes: list[dict[str, Any]]) -> dict[str, float]:
    if not episodes:
        return {"episodes": 0.0}
    numeric_keys = sorted(
        set.intersection(
            *(
                {
                    key
                    for key, value in episode.items()
                    if isinstance(value, (int, float, bool, np.number))
                }
                for episode in episodes
            )
        )
    )
    result = {"episodes": float(len(episodes))}
    for key in numeric_keys:
        result[f"episode/{key}"] = float(np.mean([float(episode[key]) for episode in episodes]))
    return result


class PPOTrainer:
    def __init__(
        self,
        env: Environment,
        config: PPOConfig,
        output_dir: Path,
        evaluator: Callable[[DynamicPlanActorCritic, int], dict[str, Any]] | None = None,
    ):
        self.env = env
        self.config = config
        self.output_dir = output_dir
        self.evaluator = evaluator
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        torch.set_num_threads(max(config.torch_threads, 1))
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.observation, _ = unpack_reset(env.reset(seed=config.seed))
        parsed = parse_observation(self.observation)
        self.model = DynamicPlanActorCritic(
            parsed.plan_features.shape[1],
            parsed.global_features.shape[0],
            config.hidden_dim,
            0 if parsed.request_features is None else parsed.request_features.shape[1],
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.history: list[dict[str, Any]] = []
        self.update = 0
        self.stage_index = 0
        self.stage_episode = 0
        self.best_evaluation_score = float("-inf")

    def _next_episode_seed(self) -> int:
        seed = self.config.seed + 100_000 * self.stage_index + self.stage_episode
        self.stage_episode += 1
        return seed

    def train(self) -> list[dict[str, Any]]:
        for stage_index, stage in enumerate(self.config.curriculum):
            if stage.updates <= 0:
                continue
            self.stage_index = stage_index
            self.stage_episode = 0
            self.best_evaluation_score = float("-inf")
            evaluations_without_improvement = 0
            configure_curriculum(self.env, stage)
            self.observation, _ = unpack_reset(self.env.reset(seed=self._next_episode_seed()))
            for stage_update in range(1, stage.updates + 1):
                self.update += 1
                if self.config.anneal_learning_rate:
                    fraction = 1.0 - (self.update - 1) / max(self.config.total_updates, 1)
                    current_lr = self.config.learning_rate * max(fraction, 0.0)
                    for group in self.optimizer.param_groups:
                        group["lr"] = current_lr
                collection = collect_rollout(
                    self.env,
                    self.model,
                    self.observation,
                    self.config.rollout_steps,
                    self.device,
                    self._next_episode_seed,
                )
                self.observation = collection.observation
                update_metrics = ppo_update(
                    self.model,
                    self.optimizer,
                    collection.rollout,
                    self.config,
                    self.device,
                    self.rng,
                )
                row: dict[str, Any] = {
                    "update": self.update,
                    "stage": stage.name,
                    "stage_update": stage_update,
                    "rollout_mean_reward": collection.mean_reward,
                    "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                    **update_metrics,
                    **_mean_episode_metrics(collection.episodes),
                }
                if self.evaluator and (
                    self.update == 1 or self.update % self.config.evaluate_every == 0
                ):
                    evaluation = self.evaluator(self.model, self.update)
                    row["evaluation"] = evaluation
                    score = float(evaluation.get(
                        "pair_throughput",
                        evaluation.get("completion_rate", float("-inf")),
                    ))
                    if score > self.best_evaluation_score:
                        self.best_evaluation_score = score
                        evaluations_without_improvement = 0
                        self.save_checkpoint(self.output_dir / "best.pt")
                        self.save_checkpoint(self.output_dir / f"best_{stage.name}.pt")
                        row["best_evaluation"] = True
                    else:
                        evaluations_without_improvement += 1
                    row["evaluations_without_improvement"] = evaluations_without_improvement
                    if (
                        self.config.early_stopping_patience > 0
                        and evaluations_without_improvement >= self.config.early_stopping_patience
                    ):
                        row["early_stopping"] = True
                self.history.append(row)
                self._write_history()
                if self.update == 1 or self.update % self.config.checkpoint_every == 0:
                    self.save_checkpoint(self.output_dir / f"checkpoint_{self.update:06d}.pt")
                print(json.dumps(row, ensure_ascii=False), flush=True)
                if row.get("early_stopping"):
                    break
        self.save_checkpoint(self.output_dir / "checkpoint.pt")
        return self.history

    def save_checkpoint(self, path: Path) -> None:
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "model_config": {
                "plan_feature_dim": self.model.plan_feature_dim,
                "global_feature_dim": self.model.global_feature_dim,
                "hidden_dim": self.model.hidden_dim,
                "request_feature_dim": self.model.request_feature_dim,
            },
            "ppo_config": self.config.to_dict(),
            "update": self.update,
            "stage_index": self.stage_index,
            "stage_episode": self.stage_episode,
            "numpy_rng_state": self.rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
        }
        torch.save(payload, path)

    def _write_history(self) -> None:
        destination = self.output_dir / "history.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.history, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(destination)


def load_model(checkpoint: Path, device: torch.device) -> DynamicPlanActorCritic:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = DynamicPlanActorCritic(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model
