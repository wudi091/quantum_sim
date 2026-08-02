"""Algorithm-independent complete swap-group schedule contract.

The representation in this module is deliberately structural.  A schedule is
an ordered tuple of groups, where every group contains swaps that consume
pairwise-disjoint entangled segments and can therefore start in parallel.

For a path with at most five internal nodes (the current Waxman default), the
entire legal schedule space is small: 1, 1, 2, 7, 34, and 214 schedules for
zero through five internal swaps.  We enumerate that exact space and leave
algorithm-specific portfolio selection to packages under :mod:`algorithms`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import TypeAlias


Node: TypeAlias = int | str
Span: TypeAlias = tuple[int, int]
SwapGroup: TypeAlias = tuple[Node, ...]
ScheduleTemplate: TypeAlias = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class SwapOperation:
    """One swap compiled against path-index spans.

    ``group_index`` is one-based.  ``dependencies`` contains the internal path
    indices of the swaps that produced the two input spans, omitting
    elementary-link leaves.
    """

    middle: Node
    middle_index: int
    group_index: int
    inputs: tuple[Span, Span]
    output: Span
    dependencies: tuple[int, ...]


@dataclass(frozen=True)
class SwapDependencyTree:
    """A compact, execution-round-labelled binary swap tree."""

    operations: tuple[SwapOperation, ...]
    root_middle_index: int | None
    output_span: Span

    @cached_property
    def operation_by_middle_index(self) -> dict[int, SwapOperation]:
        return {
            operation.middle_index: operation
            for operation in self.operations
        }


def _compile_dependency_tree(
    path: tuple[Node, ...],
    groups: tuple[SwapGroup, ...],
) -> SwapDependencyTree:
    """Compile and validate one complete schedule using atomic group updates."""

    final_span = (0, len(path) - 1)
    if len(path) == 2:
        return SwapDependencyTree((), None, final_span)

    position = {node: index for index, node in enumerate(path)}
    # Each active segment records the internal swap that produced it.  ``None``
    # denotes an elementary-link leaf.
    active: dict[Span, int | None] = {
        (index, index + 1): None
        for index in range(len(path) - 1)
    }
    operations: list[SwapOperation] = []

    for group_index, group in enumerate(groups, start=1):
        prepared: list[tuple[SwapOperation, Span, Span]] = []
        consumed: set[Span] = set()
        for middle in group:
            middle_index = position[middle]
            left_inputs = tuple(
                span for span in active if span[1] == middle_index
            )
            right_inputs = tuple(
                span for span in active if span[0] == middle_index
            )
            if len(left_inputs) != 1 or len(right_inputs) != 1:
                raise ValueError(
                    "swap group violates the current segment dependencies"
                )
            left_input = left_inputs[0]
            right_input = right_inputs[0]
            if left_input in consumed or right_input in consumed:
                raise ValueError(
                    "parallel swaps cannot share or consume the same input "
                    "entangled segment"
                )
            consumed.update((left_input, right_input))
            dependencies = tuple(
                producer
                for producer in (
                    active[left_input], active[right_input]
                )
                if producer is not None
            )
            operation = SwapOperation(
                middle=middle,
                middle_index=middle_index,
                group_index=group_index,
                inputs=(left_input, right_input),
                output=(left_input[0], right_input[1]),
                dependencies=dependencies,
            )
            prepared.append((operation, left_input, right_input))

        # A group is atomic: every operation above was resolved against the
        # same pre-group active-segment state.
        for _, left_input, right_input in prepared:
            del active[left_input]
            del active[right_input]
        for operation, _, _ in prepared:
            if operation.output in active:
                raise ValueError("swap group produced a duplicate active span")
            active[operation.output] = operation.middle_index
            operations.append(operation)

    if set(active) != {final_span}:
        raise ValueError(
            "swap schedule is not complete: it must produce exactly the "
            "end-to-end path span"
        )
    return SwapDependencyTree(
        operations=tuple(operations),
        root_middle_index=active[final_span],
        output_span=final_span,
    )


@dataclass(frozen=True)
class CompleteSchedule:
    """A legal, complete ordered sequence of parallel swap groups."""

    path: tuple[Node, ...]
    groups: tuple[SwapGroup, ...]
    dependency_tree: SwapDependencyTree = field(init=False, repr=False)

    def __post_init__(self) -> None:
        path = tuple(self.path)
        groups = tuple(tuple(group) for group in self.groups)
        if len(path) < 2 or len(set(path)) != len(path):
            raise ValueError("path must be a simple path with at least two nodes")
        if any(not group for group in groups):
            raise ValueError("swap group cannot be empty")

        internal = path[1:-1]
        position = {node: index for index, node in enumerate(path)}
        flattened = tuple(node for group in groups for node in group)
        unknown = tuple(node for node in flattened if node not in position)
        if unknown:
            raise ValueError(
                "every swap node must be an internal node of the path"
            )
        if len(set(flattened)) != len(flattened):
            raise ValueError("swap schedule cannot repeat or duplicate a node")
        if set(flattened) != set(internal):
            raise ValueError(
                "swap schedule is incomplete or contains a non-internal path "
                "node"
            )

        normalized = tuple(
            tuple(sorted(group, key=position.__getitem__))
            for group in groups
        )
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "groups", normalized)
        object.__setattr__(
            self,
            "dependency_tree",
            _compile_dependency_tree(path, normalized),
        )

    @classmethod
    def from_linear_order(
        cls,
        path: tuple[Node, ...],
        swap_order: tuple[Node, ...],
    ) -> "CompleteSchedule":
        """Lift a legacy sequential order into singleton swap groups."""

        return cls(
            path=tuple(path),
            groups=tuple((node,) for node in swap_order),
        )

    @cached_property
    def swap_order(self) -> tuple[Node, ...]:
        """Stable flattened form for labels and backward-compatible features."""

        return tuple(node for group in self.groups for node in group)

    @cached_property
    def structural_key(self) -> ScheduleTemplate:
        position = {node: index for index, node in enumerate(self.path)}
        return tuple(
            tuple(position[node] for node in group)
            for group in self.groups
        )

    @cached_property
    def release_rounds(self) -> tuple[tuple[Node, int], ...]:
        return tuple(
            (node, group_index)
            for group_index, group in enumerate(self.groups, start=1)
            for node in group
        )

    @cached_property
    def release_round_by_node(self) -> dict[Node, int]:
        return dict(self.release_rounds)

    @property
    def round_count(self) -> int:
        return len(self.groups)

    @property
    def is_linear(self) -> bool:
        return all(len(group) == 1 for group in self.groups)

    @property
    def is_left_to_right_linear(self) -> bool:
        return (
            self.is_linear
            and self.swap_order == self.path[1:-1]
        )

    def release_round(self, node: Node) -> int:
        try:
            return self.release_round_by_node[node]
        except KeyError as exc:
            raise ValueError("release round is defined only for internal nodes") \
                from exc


def _nonempty_matchings(remaining: tuple[int, ...]):
    """Yield every nonempty matching of the current ordered boundaries."""

    width = len(remaining)
    for mask in range(1, 1 << width):
        if mask & (mask << 1):
            continue
        yield tuple(
            remaining[index]
            for index in range(width)
            if mask & (1 << index)
        )


@lru_cache(maxsize=None)
def enumerate_schedule_templates(internal_count: int) -> tuple[ScheduleTemplate, ...]:
    """Enumerate all legal complete group schedules for a path shape."""

    if internal_count < 0:
        raise ValueError("internal_count cannot be negative")
    schedules: list[ScheduleTemplate] = []

    def visit(
        remaining: tuple[int, ...],
        prefix: ScheduleTemplate,
    ) -> None:
        if not remaining:
            schedules.append(prefix)
            return
        for group in _nonempty_matchings(remaining):
            chosen = set(group)
            visit(
                tuple(node for node in remaining if node not in chosen),
                prefix + (group,),
            )

    visit(tuple(range(1, internal_count + 1)), ())
    schedules.sort(key=lambda groups: (len(groups), groups))
    return tuple(schedules)


def enumerate_complete_schedules(
    path: tuple[Node, ...],
) -> tuple[CompleteSchedule, ...]:
    """Return the exact legal schedule catalogue for one concrete path."""

    path = tuple(path)
    if len(path) < 2 or len(set(path)) != len(path):
        raise ValueError("path must be a simple path with at least two nodes")
    return tuple(
        CompleteSchedule(
            path=path,
            groups=tuple(
                tuple(path[index] for index in group)
                for group in template
            ),
        )
        for template in enumerate_schedule_templates(len(path) - 2)
    )


def complete_schedule_count(internal_count: int) -> int:
    """Return the exact catalogue size for a path with ``internal_count`` swaps."""

    return len(enumerate_schedule_templates(internal_count))


def is_valid_complete_schedule(
    path: tuple[Node, ...],
    groups: tuple[SwapGroup, ...],
) -> bool:
    """Boolean convenience wrapper around :class:`CompleteSchedule`."""

    try:
        CompleteSchedule(path=path, groups=groups)
    except ValueError:
        return False
    return True
