"""Expected-throughput (EXT) equations from the Q-CAST implementation.

For a width ``W`` path, the distribution tracks the number of successful
parallel links after each hop.  The author code uses the path's *individual*
channel probabilities and multiplies by ``q ** (h - 1)``.  Keeping the
probabilities per hop makes the implementation work for heterogeneous
Waxman link lengths as well as the homogeneous special case.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def _pmf(width: int, probability: float) -> list[float]:
    if width < 1:
        raise ValueError("width must be positive")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("channel probability must be in [0, 1]")
    return [
        math.comb(width, i)
        * probability ** i
        * (1.0 - probability) ** (width - i)
        for i in range(width + 1)
    ]


def propagate_distribution(previous: Sequence[float], probability: float, width: int | None = None) -> list[float]:
    """Propagate ``P(i)`` through one heterogeneous channel hop.

    This is the recurrence in ``Topo.e`` of commit ``2db9e71``.  ``previous``
    may include a zero state and must have length ``width + 1``.
    """

    if width is None:
        width = len(previous) - 1
    if width < 1 or len(previous) != width + 1:
        raise ValueError("distribution length must be width + 1")
    if any(value < -1e-12 for value in previous):
        raise ValueError("distribution entries must be non-negative")
    probabilities = _pmf(width, float(probability))
    result = [0.0] * (width + 1)
    # P_new[i] = P_old[i] * Pr(L >= i) + Pr(L=i) * sum_{j>i} P_old[j]
    suffix_old = [0.0] * (width + 2)
    for index in range(width, -1, -1):
        suffix_old[index] = suffix_old[index + 1] + float(previous[index])
    suffix_new_channel = [0.0] * (width + 2)
    for index in range(width, -1, -1):
        suffix_new_channel[index] = suffix_new_channel[index + 1] + probabilities[index]
    for i in range(width + 1):
        result[i] = float(previous[i]) * suffix_new_channel[i] + probabilities[i] * suffix_old[i + 1]
    # Floating point roundoff is harmless but normalization helps long paths.
    total = sum(result)
    if total > 0.0:
        result = [value / total for value in result]
    return result


def expected_throughput(
    edge_probabilities: Iterable[float],
    width: int,
    swap_probability: float,
) -> float:
    """Return Q-CAST EXT for a path's per-hop probabilities.

    The official source's ``Topo.e`` uses ``q.pow(s - 1)`` where ``s`` is the
    hop count.  Thus a one-hop path has no swap penalty, while a two-hop path
    has one swap factor.  This intentionally differs from the paper prose's
    occasionally printed ``q^h`` notation and is the compatibility behavior.
    """

    probabilities = tuple(float(value) for value in edge_probabilities)
    if width < 1:
        raise ValueError("width must be positive")
    if not probabilities:
        return 0.0
    if not 0.0 <= swap_probability <= 1.0:
        raise ValueError("swap_probability must be in [0, 1]")
    distribution = _pmf(width, probabilities[0])
    for probability in probabilities[1:]:
        distribution = propagate_distribution(distribution, probability, width)
    mean_successes = sum(index * value for index, value in enumerate(distribution))
    return mean_successes * float(swap_probability) ** (len(probabilities) - 1)


def ext(edge_probabilities: Iterable[float], width: int, swap_probability: float) -> float:
    """Short alias used by the allocation module and external callers."""

    return expected_throughput(edge_probabilities, width, swap_probability)


def expected_throughput_for_path(path: Sequence[int], edge_probabilities: Sequence[float], width: int, swap_probability: float) -> float:
    """Path-shaped convenience wrapper; validates hop/probability count."""

    if len(path) - 1 != len(edge_probabilities):
        raise ValueError("one probability is required for each path hop")
    return expected_throughput(edge_probabilities, width, swap_probability)

