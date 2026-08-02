"""Enumerate construction plans for a given path.

Given a physical path P, produces the distinct construction plans C that
con_design.md §3 contrasts:

* ``seq``  -- sequential generation (one edge per phase) + linear swap
              tree (left fold). Minimal simultaneous memory, fast release.
* ``bal``  -- parallel generation (outside-in batches) + balanced swap
              tree. Shorter makespan, but intermediate nodes (e.g. node C
              in A-B-C-D-E) are held simultaneously for longer.
* ``mid``  -- parallel generation but a *linear* swap tree, isolating the
              effect of generation parallelism from swap-tree balance.

All three share the same path P; only the construction plan C differs.
"""

from __future__ import annotations

from .plan import (
    ConstructionPlan,
    Edge,
    SwapNode,
    edge,
    elementary_ref,
    path_edges,
)


def _leaves(path: tuple[int, ...]) -> list[tuple[str, tuple[int, int]]]:
    """Elementary pairs in path order, each as (ref, span)."""
    return [
        (elementary_ref(e), (path[i], path[i + 1]))
        for i, e in enumerate(path_edges(path))
    ]


def _build_linear(leaves: list[tuple[str, tuple[int, int]]]) -> list[SwapNode]:
    """Linear left-fold swap tree: AB,BC->AC, then AC,CD->AD, ..."""
    nodes: list[SwapNode] = []
    if len(leaves) <= 1:
        return nodes
    lref, lspan = leaves[0]
    counter = 0
    for i in range(1, len(leaves)):
        rref, rspan = leaves[i]
        middle = lspan[1]  # shared endpoint with the next edge
        node = SwapNode.make(counter, middle, lref, lspan, rref, rspan)
        nodes.append(node)
        counter += 1
        lref, lspan = node.output_ref, node.span
    return nodes


def _build_balanced(leaves: list[tuple[str, tuple[int, int]]]) -> list[SwapNode]:
    """Balanced binary swap tree over the ordered elementary pairs.

    Recursively splits the leaf list into two contiguous halves, builds
    each, then combines them. Because halves are contiguous, the two
    resulting spans share exactly one endpoint, which is the swap middle.
    """
    counter = [0]
    nodes: list[SwapNode] = []

    def build(seg: list[tuple[str, tuple[int, int]]]) -> tuple[str, tuple[int, int]]:
        if len(seg) == 1:
            return seg[0]
        mid = len(seg) // 2
        lref, lspan = build(seg[:mid])
        rref, rspan = build(seg[mid:])
        middle = lspan[1]  # == rspan[0] for contiguous halves
        node = SwapNode.make(counter[0], middle, lref, lspan, rref, rspan)
        counter[0] += 1
        nodes.append(node)
        return node.output_ref, node.span

    build(leaves)
    # nodes were appended in post-order (children before parents) -> already
    # topologically sorted, but reverse-includes ensure a parent never
    # precedes its children; the post-order build already guarantees this.
    return nodes


def _peel_layers(ed: tuple[Edge, ...]) -> tuple[tuple[Edge, ...], ...]:
    """Outside-in parallel generation layers: (e0,eN),(e1,eN-1),..."""
    layers: list[tuple[Edge, ...]] = []
    lo, hi = 0, len(ed) - 1
    while lo <= hi:
        layer = [ed[lo]]
        if hi > lo:
            layer.append(ed[hi])
        layers.append(tuple(layer))
        lo += 1
        hi -= 1
    return tuple(layers)


def _single_layers(ed: tuple[Edge, ...]) -> tuple[tuple[Edge, ...], ...]:
    """One edge per generation layer (fully sequential)."""
    return tuple((e,) for e in ed)


def sequential_plan(path: tuple[int, ...]) -> ConstructionPlan:
    ed = path_edges(path)
    leaves = _leaves(path)
    return ConstructionPlan(
        path=path,
        kind="seq",
        gen_layers=_single_layers(ed),
        swap_tree=tuple(_build_linear(leaves)),
    )


def balanced_plan(path: tuple[int, ...]) -> ConstructionPlan:
    ed = path_edges(path)
    leaves = _leaves(path)
    return ConstructionPlan(
        path=path,
        kind="bal",
        gen_layers=_peel_layers(ed),
        swap_tree=tuple(_build_balanced(leaves)),
    )


def intermediate_plan(path: tuple[int, ...]) -> ConstructionPlan:
    """Parallel generation (outside-in) but a linear swap tree.

    Isolates generation-order parallelism from swap-tree balance: same
    memory holdings during generation as ``bal``, but swaps resolve
    sequentially like ``seq``.
    """
    ed = path_edges(path)
    leaves = _leaves(path)
    return ConstructionPlan(
        path=path,
        kind="mid",
        gen_layers=_peel_layers(ed),
        swap_tree=tuple(_build_linear(leaves)),
    )


def enumerate_constructions(path: tuple[int, ...]) -> list[ConstructionPlan]:
    """All distinct construction plans currently supported for ``path``."""
    if len(path) < 2:
        return []
    if len(path) == 2:
        # Single edge: only generation, no swap. One trivial plan.
        ed = path_edges(path)
        return [ConstructionPlan(path=path, kind="seq", gen_layers=_single_layers(ed), swap_tree=())]
    return [sequential_plan(path), balanced_plan(path), intermediate_plan(path)]
