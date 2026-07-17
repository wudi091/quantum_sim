from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from qnet_core.reward import RewardConfig


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    min_hops: int
    max_hops: int
    updates: int
    min_requests: int
    max_requests: int


def default_curriculum() -> tuple[CurriculumStage, ...]:
    """Return the default full-range training configuration.

    The trainer keeps a tuple of stages for checkpoint/evaluator
    compatibility, but the default run is intentionally a single stage over
    the complete 2--50 hop range.  The legacy short/medium/long curriculum is
    still available from the CLI with ``--curriculum``.
    """
    return (
        CurriculumStage("full", 2, 50, 1000, 100, 100),
    )


@dataclass
class PPOConfig:
    hidden_dim: int = 128
    rollout_steps: int = 512
    learning_rate: float = 3e-4
    anneal_learning_rate: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    ppo_epochs: int = 6
    minibatch_size: int = 128
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    normalize_advantage: bool = True
    value_clip: float | None = 0.2
    seed: int = 0
    torch_threads: int = 1
    device: str = "cpu"
    checkpoint_every: int = 25
    evaluate_every: int = 25
    evaluation_episodes: int = 20
    early_stopping_patience: int = 0
    curriculum: tuple[CurriculumStage, ...] = field(default_factory=default_curriculum)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @property
    def total_updates(self) -> int:
        return sum(stage.updates for stage in self.curriculum)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
