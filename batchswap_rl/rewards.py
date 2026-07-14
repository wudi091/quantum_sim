from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import RewardConfig


@dataclass(frozen=True)
class RewardState:
    """Cumulative quantities needed for a Markov transition reward."""

    potential: float
    completed: int
    elapsed_slots: int
    elementary_epr: int
    swaps: int


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    potential_shaping: float
    completion: float
    makespan: float
    elementary_epr: float
    swaps: float

    def as_dict(self) -> dict[str, float]:
        return {
            "reward": self.total,
            "reward_potential": self.potential_shaping,
            "reward_completion": self.completion,
            "reward_makespan": self.makespan,
            "reward_elementary_epr": self.elementary_epr,
            "reward_swaps": self.swaps,
        }


class RewardComposer:
    """Potential-based progress shaping plus task and resource terms.

    ``Phi`` must be a state-only quantity.  With the same discount as PPO,
    ``gamma**duration * Phi(s') - Phi(s)`` preserves the optimal policy for the
    semi-Markov transition.  Thus reservation microsteps (duration zero) do not
    create artificial discounting.  Terminal Phi is explicitly zeroed.
    """

    def __init__(self, config: RewardConfig, gamma: float = 0.99):
        self.config = config
        self.gamma = gamma

    def __call__(
        self, previous: RewardState, current: RewardState, terminal: bool = False
    ) -> RewardBreakdown:
        terminal_potential = 0.0 if terminal else current.potential
        duration = max(current.elapsed_slots - previous.elapsed_slots, 0)
        potential = self.config.potential_coef * (
            (self.gamma**duration) * terminal_potential - previous.potential
        )
        completion = self.config.completion_bonus * max(current.completed - previous.completed, 0)
        makespan = -self.config.makespan_coef * duration
        elementary = -self.config.elementary_epr_coef * max(
            current.elementary_epr - previous.elementary_epr, 0
        )
        swaps = -self.config.swap_coef * max(current.swaps - previous.swaps, 0)
        total = potential + completion + makespan + elementary + swaps
        return RewardBreakdown(total, potential, completion, makespan, elementary, swaps)


def remaining_work_potential(remaining_hops: Iterable[int], normalization: float = 1.0) -> float:
    """Default Phi: negative normalized aggregate remaining path length."""
    if normalization <= 0:
        raise ValueError("normalization must be positive")
    return -sum(max(int(hops), 0) for hops in remaining_hops) / normalization
