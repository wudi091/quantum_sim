"""Reproducible command-line training entry for ARC-Q."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from statistics import fmean
from typing import Any

import numpy as np
import torch
import yaml

from algorithms.routing_core.execution import OnlineExecutionConfig
from qnet_core.scenario import ScenarioConfig, make_episode
from qnet_core.spec import PhysicalConfig

from .checkpoint import save_arcq_checkpoint
from .policy import ARCQPolicy
from .rollout import collect_episode
from .training import PPOConfig, PPOTrainer


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 96
    message_passing_layers: int = 3

    def __post_init__(self) -> None:
        if self.hidden_dim < 8:
            raise ValueError("hidden_dim must be at least 8")
        if self.message_passing_layers < 1:
            raise ValueError("message_passing_layers must be positive")


@dataclass(frozen=True)
class TrainingRunConfig:
    episode_count: int = 1000
    episodes_per_update: int = 8
    random_seed: int = 20260905
    topology_seed: int = 3101
    checkpoint_every_updates: int = 10
    output_directory: str = "results/arcq/training"
    device: str = "auto"
    cpu_threads: int | None = None

    def __post_init__(self) -> None:
        if self.episode_count < 1 or self.episodes_per_update < 1:
            raise ValueError("training episode counts must be positive")
        if self.checkpoint_every_updates < 1:
            raise ValueError("checkpoint_every_updates must be positive")
        if not self.output_directory:
            raise ValueError("output_directory must be non-empty")
        if self.cpu_threads is not None and self.cpu_threads < 1:
            raise ValueError("cpu_threads must be positive when set")


@dataclass(frozen=True)
class ARCQTrainingConfig:
    model: ModelConfig
    ppo: PPOConfig
    environment: OnlineExecutionConfig
    scenario: ScenarioConfig
    run: TrainingRunConfig


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def load_training_config(path: str | Path) -> ARCQTrainingConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = _mapping(payload, "training config")
    expected = {"model", "ppo", "environment", "scenario", "run"}
    unknown = set(root) - expected
    missing = expected - set(root)
    if unknown:
        raise ValueError(f"unknown config section: {sorted(unknown)[0]}")
    if missing:
        raise ValueError(f"missing config section: {sorted(missing)[0]}")

    scenario_values = _mapping(root["scenario"], "scenario")
    physical_values = _mapping(
        scenario_values.pop("physical", {}),
        "scenario.physical",
    )
    environment_values = _mapping(root["environment"], "environment")
    for key in ("construction_kinds", "purification_kinds"):
        if key in environment_values:
            environment_values[key] = tuple(environment_values[key])
    return ARCQTrainingConfig(
        model=ModelConfig(**_mapping(root["model"], "model")),
        ppo=PPOConfig(**_mapping(root["ppo"], "ppo")),
        environment=OnlineExecutionConfig(**environment_values),
        scenario=ScenarioConfig(
            **scenario_values,
            physical=PhysicalConfig(**physical_values),
        ),
        run=TrainingRunConfig(**_mapping(root["run"], "run")),
    )


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_rng_state(state: object) -> None:
    values = _mapping(state, "checkpoint rng_state")
    random.setstate(values["python"])
    np.random.set_state(values["numpy"])
    torch.set_rng_state(values["torch"])
    if torch.cuda.is_available() and values.get("cuda") is not None:
        torch.cuda.set_rng_state_all(values["cuda"])


def train(
    config: ARCQTrainingConfig,
    *,
    resume_path: str | Path | None = None,
) -> tuple[ARCQPolicy, list[dict[str, object]]]:
    run = config.run
    if run.cpu_threads is not None:
        torch.set_num_threads(run.cpu_threads)
    random.seed(run.random_seed)
    np.random.seed(run.random_seed)
    torch.manual_seed(run.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run.random_seed)
    device = _resolve_device(run.device)
    policy = ARCQPolicy(
        hidden_dim=config.model.hidden_dim,
        message_passing_layers=config.model.message_passing_layers,
    ).to(device)
    trainer = PPOTrainer(policy, config.ppo)
    output = Path(run.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    episodes_completed = 0
    update_index = 0
    if resume_path is not None:
        from .checkpoint import load_arcq_checkpoint

        restored_policy, metadata = load_arcq_checkpoint(
            resume_path,
            device=device,
        )
        restored_model = _mapping(metadata["model"], "checkpoint model")
        if restored_model != asdict(config.model):
            raise ValueError("resume checkpoint model configuration differs")
        policy.load_state_dict(restored_policy.state_dict())
        optimizer_state = metadata.get("optimizer_state_dict")
        if optimizer_state is None:
            raise ValueError("resume checkpoint has no optimizer state")
        trainer.optimizer.load_state_dict(optimizer_state)
        training_state = _mapping(
            metadata["training_state"],
            "checkpoint training_state",
        )
        stored_config = _mapping(
            training_state.get("config"),
            "checkpoint training config",
        )
        for section in ("model", "ppo", "environment", "scenario"):
            if stored_config.get(section) != asdict(config).get(section):
                raise ValueError(
                    f"resume checkpoint differs in {section} configuration"
                )
        stored_run = _mapping(stored_config.get("run"), "checkpoint run config")
        if int(stored_run["random_seed"]) != run.random_seed:
            raise ValueError("resume checkpoint uses another random seed")
        if int(stored_run["episodes_per_update"]) != run.episodes_per_update:
            raise ValueError(
                "resume checkpoint uses another episodes-per-update value"
            )
        if int(training_state["fixed_training_topology_seed"]) != (
            run.topology_seed
        ):
            raise ValueError("resume checkpoint uses another training topology")
        episodes_completed = int(training_state["episodes_completed"])
        update_index = int(training_state["update"])
        raw_history = training_state.get("history", [])
        if not isinstance(raw_history, list):
            raise ValueError("checkpoint history must be a list")
        history = [dict(row) for row in raw_history]
        if episodes_completed > run.episode_count:
            raise ValueError("checkpoint exceeds configured episode count")
        if metadata.get("rng_state") is None:
            raise ValueError("resume checkpoint has no RNG state")
        _restore_rng_state(metadata["rng_state"])
    while episodes_completed < run.episode_count:
        batch_count = min(
            run.episodes_per_update,
            run.episode_count - episodes_completed,
        )
        rollouts = []
        for batch_index in range(batch_count):
            episode_index = episodes_completed + batch_index
            episode_seed = run.random_seed + episode_index + 1
            episode = make_episode(
                config.scenario,
                episode_seed,
                topology_seed=run.topology_seed,
            )
            rollouts.append(collect_episode(
                policy,
                episode,
                config.environment,
            ))
        diagnostics = trainer.update(
            rollouts,
            shuffle_seed=run.random_seed + update_index,
        )
        episodes_completed += batch_count
        update_index += 1
        history_row: dict[str, object] = {
            "update": update_index,
            "episodes_completed": episodes_completed,
            **asdict(diagnostics),
            "mean_completion_rate": fmean(
                rollout.execution.metrics["completion_rate"]
                for rollout in rollouts
            ),
            "mean_decision_seconds": fmean(
                rollout.execution.metrics["mean_decision_seconds"]
                for rollout in rollouts
            ),
        }
        history.append(history_row)
        print(json.dumps(history_row, ensure_ascii=False), flush=True)
        _write_json_atomic(output / "training_history.json", {
            "schema_version": 1,
            "method": "ARC-Q",
            "fixed_training_topology_seed": run.topology_seed,
            "config": asdict(config),
            "history": history,
        })
        if (
            update_index % run.checkpoint_every_updates == 0
            or episodes_completed == run.episode_count
        ):
            save_arcq_checkpoint(
                output / "arcq_latest.pt",
                policy,
                hidden_dim=config.model.hidden_dim,
                message_passing_layers=config.model.message_passing_layers,
                training_state={
                    "update": update_index,
                    "episodes_completed": episodes_completed,
                    "fixed_training_topology_seed": run.topology_seed,
                    "config": asdict(config),
                    "latest_metrics": history_row,
                    "history": history,
                },
                optimizer_state_dict=trainer.optimizer.state_dict(),
                rng_state=_rng_state(),
            )
    return policy, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/arcq_train.yaml"),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="resume exactly from an ARC-Q training checkpoint",
    )
    arguments = parser.parse_args()
    train(
        load_training_config(arguments.config),
        resume_path=arguments.resume,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ARCQTrainingConfig",
    "ModelConfig",
    "TrainingRunConfig",
    "load_training_config",
    "main",
    "train",
]
