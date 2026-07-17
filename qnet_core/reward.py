"""Algorithm-independent reward weights for the shared routing environment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Weights for state-based, potential-shaped rewards.

    The potential difference is the graph-derived reduction in remaining
    frontier-to-destination distance. No expert action, next-hop preference,
    or hop-specific routing rule is encoded here.
    """

    # The common factor of 0.1 keeps the critic targets numerically stable;
    # relative preferences match the validated high-hop reward.
    potential_coef: float = 0.1
    completion_bonus: float = 1.0
    makespan_coef: float = 0.005
    failure_coef: float = 0.1
    timeout_coef: float = 0.1
