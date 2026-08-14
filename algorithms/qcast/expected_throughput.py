"""Expected-throughput equations used by the Q-CAST path baseline."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def _pmf(width: int, probability: float) -> list[float]:
    if width < 1:
        raise ValueError("width must be positive")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("channel probability must be in [0, 1]")
    return [
        math.comb(width, successes)
        * probability ** successes
        * (1.0 - probability) ** (width - successes)
        for successes in range(width + 1)
    ]


def propagate_distribution(
    previous: Sequence[float],
    probability: float,
    width: int | None = None,
) -> list[float]:
    """Propagate successful parallel-link counts through one channel hop."""

    resolved_width = len(previous) - 1 if width is None else int(width)
    if resolved_width < 1 or len(previous) != resolved_width + 1:
        raise ValueError("distribution length must be width + 1")
    if any(value < -1e-12 for value in previous):
        raise ValueError("distribution entries must be non-negative")
    probabilities = _pmf(resolved_width, float(probability))
    result = [0.0] * (resolved_width + 1)
    suffix_previous = [0.0] * (resolved_width + 2)
    suffix_channel = [0.0] * (resolved_width + 2)
    for index in range(resolved_width, -1, -1):
        suffix_previous[index] = (
            suffix_previous[index + 1] + float(previous[index])
        )
        suffix_channel[index] = (
            suffix_channel[index + 1] + probabilities[index]
        )
    for successes in range(resolved_width + 1):
        result[successes] = (
            float(previous[successes]) * suffix_channel[successes]
            + probabilities[successes] * suffix_previous[successes + 1]
        )
    total = sum(result)
    return result if total <= 0.0 else [value / total for value in result]


def expected_throughput(
    edge_probabilities: Iterable[float],
    width: int,
    swap_probability: float,
) -> float:
    """Return Q-CAST EXT using the official ``q ** (h - 1)`` convention."""

    probabilities = tuple(float(value) for value in edge_probabilities)
    if width < 1:
        raise ValueError("width must be positive")
    if not probabilities:
        return 0.0
    if not 0.0 <= swap_probability <= 1.0:
        raise ValueError("swap_probability must be in [0, 1]")
    distribution = _pmf(width, probabilities[0])
    for probability in probabilities[1:]:
        distribution = propagate_distribution(
            distribution,
            probability,
            width,
        )
    mean_successes = sum(
        successes * probability
        for successes, probability in enumerate(distribution)
    )
    return mean_successes * swap_probability ** (len(probabilities) - 1)


__all__ = ["expected_throughput", "propagate_distribution"]
