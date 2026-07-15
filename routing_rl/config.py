from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RewardConfig:
    """Weights for the pure-RL environment reward.

    The environment owns reward calculation.  These weights are kept with the
    checkpoint so evaluation can reproduce the exact training objective.
    """

    potential_coef: float = 1.0
    completion_bonus: float = 5.0
    makespan_coef: float = 0.05
    elementary_epr_coef: float = 0.01
    swap_coef: float = 0.005


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    min_hops: int
    max_hops: int
    updates: int
    min_requests: int
    max_requests: int


def default_curriculum() -> tuple[CurriculumStage, ...]:
    return (
        CurriculumStage("short", 2, 5, 200, 2, 4),
        CurriculumStage("medium", 5, 15, 400, 5, 10),
        CurriculumStage("long", 20, 50, 800, 100, 100),
    )


@dataclass
class PPOConfig:
    hidden_dim: int = 128
    rollout_steps: int = 512
    learning_rate: float = 3e-4
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
    curriculum: tuple[CurriculumStage, ...] = field(default_factory=default_curriculum)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @property
    def total_updates(self) -> int:
        return sum(stage.updates for stage in self.curriculum)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
