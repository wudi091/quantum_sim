"""Construction-aware routing data structures.

Implements con_design.md's core thesis: a routing decision selects a full
construction plan (P, C), not just a path P. A construction plan bundles
three elements (con_design.md §6):

  1. Entanglement generation order  -> ``gen_layers`` (parallel batches)
  2. Parallelization strategy       -> expressed by the layering above
  3. Swap dependency structure       -> ``swap_tree`` (topological order)

Phase 1 of this module is a *deterministic* kernel: elementary EPR pairs
are assumed ready on demand (``generation_probability = 1``), a slot is
discretized into intra-slot phases, and the resource footprint is the
per-node memory occupation curve over those phases. This deliberately
isolates con_design.md's claim -- *same path, different C -> different footprint
-> different impact on concurrent requests* -- from SeQUeNCe's
probabilistic generation. Probabilistic generation and cross-slot
accumulation are Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# A physical edge, always stored with the smaller node id first.
Edge = tuple[int, int]
PlanKind = Literal["seq", "bal", "mid"]


def edge(u: int, v: int) -> Edge:
    """Return a canonical (min, max) edge tuple."""
    return (u, v) if u <= v else (v, u)


def path_edges(path: tuple[int, ...]) -> tuple[Edge, ...]:
    """Elementary edges along a physical path, in path order."""
    return tuple(edge(path[i], path[i + 1]) for i in range(len(path) - 1))


def elementary_ref(e: Edge) -> str:
    """Stable string id for an elementary pair over edge ``e``."""
    u, v = e
    return f"e:{u}-{v}"


@dataclass(frozen=True)
class SwapNode:
    """One swap operation in a construction plan's swap tree.

    A swap consumes two pairs (``left_ref``, ``right_ref``) that meet at
    ``middle`` and produces one longer-range pair (``output_ref``).
    ``left_ref`` / ``right_ref`` reference either an elementary pair id
    (``e:u-v``) or the ``output_ref`` of an earlier SwapNode, so the
    dependency tree is ordered topologically by position in ``swap_tree``.

    The output pair spans ``span`` -- the set of endpoints it connects --
    which is used both for footprint accounting and to recognize when a
    plan completes its request.
    """

    middle: int
    left_ref: str
    right_ref: str
    output_ref: str
    span: tuple[int, int]

    @staticmethod
    def make(
        index: int,
        middle: int,
        left_ref: str,
        left_span: tuple[int, int],
        right_ref: str,
        right_span: tuple[int, int],
    ) -> "SwapNode":
        out = f"s:{index}"
        if middle not in left_span or middle not in right_span:
            raise ValueError("swap inputs must both terminate at the middle node")
        left_outer = left_span[1] if left_span[0] == middle else left_span[0]
        right_outer = right_span[1] if right_span[0] == middle else right_span[0]
        return SwapNode(middle, left_ref, right_ref, out, (left_outer, right_outer))


@dataclass(frozen=True)
class ConstructionPlan:
    """A full (P, C) construction plan for one request over one slot.

    Attributes
    ----------
    path
        Physical path P, e.g. ``(0, 1, 2, 3, 4)`` for A-B-C-D-E.
    kind
        Construction family: ``seq`` (sequential / linear left-fold),
        ``bal`` (parallel generation + balanced swap tree), or ``mid``
        (intermediate parallelism).
    gen_layers
        Ordered parallel-generation layers. Each layer is a batch of
        elementary edges generated *simultaneously* in one intra-slot
        phase. The layering encodes con_design.md's generation order +
        parallelization strategy.
    swap_tree
        Swap operations in topological order. Each node references
        elementary pair ids or earlier swap outputs, so the tree can be
        linear (``seq``) or balanced/branching (``bal``).
    elementary_pairs
        All elementary edges used by the plan (deduplicated, sorted).
    output_ref
        Ref of the final pair produced (completes the request when the
        span equals the full path endpoints).
    is_complete
        True iff the final pair spans ``path[0]``..``path[-1]``.
    """

    path: tuple[int, ...]
    kind: PlanKind
    gen_layers: tuple[tuple[Edge, ...], ...]
    swap_tree: tuple[SwapNode, ...]
    elementary_pairs: tuple[Edge, ...] = field(default_factory=tuple)
    output_ref: str = ""
    is_complete: bool = False

    def __post_init__(self) -> None:
        # Frozen dataclass: use object.__setattr__ to fill derived fields.
        pairs = tuple(sorted(set(e for layer in self.gen_layers for e in layer)))
        if not self.elementary_pairs:
            object.__setattr__(self, "elementary_pairs", pairs)
        if not self.output_ref:
            if self.swap_tree:
                object.__setattr__(self, "output_ref", self.swap_tree[-1].output_ref)
            elif len(pairs) == 1:
                object.__setattr__(self, "output_ref", elementary_ref(pairs[0]))
        if not self.is_complete:
            if self.swap_tree:
                final_span = self.swap_tree[-1].span
                complete = final_span == (self.path[0], self.path[-1])
            else:
                complete = len(self.path) == 2 and len(pairs) == 1
            object.__setattr__(self, "is_complete", complete)

    def all_refs(self) -> set[str]:
        """Every pair ref (elementary + swap outputs) touched by the plan."""
        refs = {elementary_ref(e) for e in self.elementary_pairs}
        refs.update(node.output_ref for node in self.swap_tree)
        return refs

    def swap_index_of(self, ref: str) -> int | None:
        """Topological index of the swap producing ``ref``, or None."""
        for i, node in enumerate(self.swap_tree):
            if node.output_ref == ref:
                return i
        return None
