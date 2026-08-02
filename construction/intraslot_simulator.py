"""Automatic EPR generation and ordered swapping inside one control slot.

The controller selects every request plan once at the slot boundary.  The
environment then advances a fixed number of physical rounds.  In each round:

1. missing elementary EPRs are attempted automatically;
2. at most one ready swap per request is executed in the chosen order;
3. qubits released by a swap are available to generation in the next round.

Generation is therefore environment dynamics, not an RL action.  Random draws
are keyed by ``(seed, slot, round, request, resource)`` so comparisons between
swap orders use common random numbers and do not depend on iteration history.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Literal, Mapping, TypeAlias


Node: TypeAlias = int | str
Edge: TypeAlias = tuple[Node, Node]


def _node_key(node: Node) -> tuple[str, str]:
    return type(node).__name__, repr(node)


def edge(u: Node, v: Node) -> Edge:
    """Return a stable undirected-edge representation."""
    if u == v:
        raise ValueError("an EPR edge must connect distinct nodes")
    return (u, v) if _node_key(u) <= _node_key(v) else (v, u)


@dataclass(frozen=True)
class IntraSlotPlan:
    """One selected path with a complete linear-path swap order."""

    request_id: str
    path: tuple[Node, ...]
    swap_order: tuple[Node, ...]
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if len(self.path) < 2:
            raise ValueError("path must contain at least two nodes")
        if len(set(self.path)) != len(self.path):
            raise ValueError("path must be simple")
        internal = self.path[1:-1]
        if len(set(self.swap_order)) != len(self.swap_order):
            raise ValueError("swap_order cannot repeat a node")
        if set(self.swap_order) != set(internal):
            raise ValueError("swap_order must contain every internal path node once")

    @property
    def elementary_edges(self) -> tuple[Edge, ...]:
        return tuple(edge(u, v) for u, v in zip(self.path, self.path[1:]))


@dataclass(frozen=True)
class IntraSlotConfig:
    rounds_per_slot: int = 3
    generation_probability: float = 1.0
    swap_probability: float = 1.0
    edge_capacity: int = 1
    bsm_capacity_per_node: int = 1
    seed: int = 0
    slot_id: int = 0

    def __post_init__(self) -> None:
        if self.rounds_per_slot < 1:
            raise ValueError("rounds_per_slot must be positive")
        if not 0.0 <= self.generation_probability <= 1.0:
            raise ValueError("generation_probability must be in [0, 1]")
        if not 0.0 <= self.swap_probability <= 1.0:
            raise ValueError("swap_probability must be in [0, 1]")
        if self.edge_capacity < 1:
            raise ValueError("edge_capacity must be positive")
        if self.bsm_capacity_per_node < 1:
            raise ValueError("bsm_capacity_per_node must be positive")


@dataclass(frozen=True)
class PairState:
    pair_id: str
    request_id: str
    left: Node
    right: Node
    born_round: int
    elementary: bool

    @property
    def endpoints(self) -> tuple[Node, Node]:
        return self.left, self.right


GenerationStatus = Literal[
    "success", "random_failure", "blocked_memory", "blocked_edge"
]
SwapStatus = Literal["success", "random_failure", "blocked_bsm"]


@dataclass(frozen=True)
class GenerationEvent:
    round_id: int
    request_id: str
    edge: Edge
    status: GenerationStatus
    pair_id: str | None = None


@dataclass(frozen=True)
class SwapEvent:
    round_id: int
    request_id: str
    middle: Node
    status: SwapStatus
    input_pair_ids: tuple[str, str] = ()
    output_pair_id: str | None = None


@dataclass(frozen=True)
class RoundTrace:
    round_id: int
    occupancy_start: dict[Node, int]
    occupancy_after_generation: dict[Node, int]
    occupancy_after_swaps: dict[Node, int]
    generation_events: tuple[GenerationEvent, ...]
    swap_events: tuple[SwapEvent, ...]


@dataclass(frozen=True)
class SlotResult:
    completed: tuple[str, ...]
    failed: tuple[str, ...]
    missed: tuple[str, ...]
    completion_round: dict[str, int]
    initial_occupancy: dict[Node, int]
    traces: tuple[RoundTrace, ...]

    @property
    def completed_count(self) -> int:
        return len(self.completed)


@dataclass
class _PlanRuntime:
    plan: IntraSlotPlan
    pending_elementary: set[Edge]
    active_pair_ids: set[str]
    next_swap_index: int = 0
    completed_round: int | None = None
    failed_round: int | None = None

    @property
    def active(self) -> bool:
        return self.completed_round is None and self.failed_round is None


class IntraSlotSimulator:
    """Execute selected plans under automatic, capacity-limited generation."""

    def __init__(
        self,
        plans: Iterable[IntraSlotPlan],
        node_capacity: Mapping[Node, int],
        config: IntraSlotConfig = IntraSlotConfig(),
        initially_ready_requests: Iterable[str] = (),
    ) -> None:
        plan_list = tuple(plans)
        if not plan_list:
            raise ValueError("at least one plan is required")
        request_ids = [plan.request_id for plan in plan_list]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("request_id values must be unique")

        self.config = config
        self.node_capacity = dict(node_capacity)
        for node, capacity in self.node_capacity.items():
            if capacity < 1:
                raise ValueError(f"node {node!r} has non-positive capacity")
        required_nodes = {node for plan in plan_list for node in plan.path}
        missing_nodes = required_nodes - self.node_capacity.keys()
        if missing_nodes:
            raise ValueError(f"missing node capacities: {sorted(map(repr, missing_nodes))}")

        ordered_plans = sorted(
            plan_list, key=lambda item: (item.priority, item.request_id)
        )
        self._runtimes = {
            plan.request_id: _PlanRuntime(
                plan=plan,
                pending_elementary=set(plan.elementary_edges),
                active_pair_ids=set(),
            )
            for plan in ordered_plans
        }
        self._runtime_order = tuple(plan.request_id for plan in ordered_plans)
        self._pairs: dict[str, PairState] = {}
        self._counter = 0

        ready = set(initially_ready_requests)
        unknown_ready = ready - self._runtimes.keys()
        if unknown_ready:
            raise ValueError(f"unknown initially-ready requests: {sorted(unknown_ready)}")
        for request_id in self._runtime_order:
            if request_id not in ready:
                continue
            runtime = self._runtimes[request_id]
            for elementary_edge in runtime.plan.elementary_edges:
                pair_id = self._create_pair(
                    request_id,
                    elementary_edge[0],
                    elementary_edge[1],
                    born_round=0,
                    elementary=True,
                    enforce_edge_capacity=True,
                )
                runtime.active_pair_ids.add(pair_id)
                runtime.pending_elementary.remove(elementary_edge)

    @property
    def pairs(self) -> dict[str, PairState]:
        return dict(self._pairs)

    def node_occupancy(self, node: Node) -> int:
        return sum(node in pair.endpoints for pair in self._pairs.values())

    def node_free_slots(self, node: Node) -> int:
        return self.node_capacity[node] - self.node_occupancy(node)

    def occupancy(self) -> dict[Node, int]:
        return {
            node: self.node_occupancy(node)
            for node in self.node_capacity
        }

    def _uniform(self, *parts: object) -> float:
        payload = "|".join(map(str, (self.config.seed, *parts))).encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64

    def _new_pair_id(self, request_id: str, born_round: int) -> str:
        pair_id = (
            f"pair-s{self.config.slot_id}-r{born_round}-"
            f"{request_id}-{self._counter}"
        )
        self._counter += 1
        return pair_id

    def _elementary_edge_occupancy(self, elementary_edge: Edge) -> int:
        return sum(
            pair.elementary and edge(pair.left, pair.right) == elementary_edge
            for pair in self._pairs.values()
        )

    def _create_pair(
        self,
        request_id: str,
        left: Node,
        right: Node,
        born_round: int,
        elementary: bool,
        enforce_edge_capacity: bool,
    ) -> str:
        if self.node_free_slots(left) <= 0 or self.node_free_slots(right) <= 0:
            raise ValueError("pair creation would exceed node memory capacity")
        elementary_edge = edge(left, right)
        if (enforce_edge_capacity
                and self._elementary_edge_occupancy(elementary_edge)
                >= self.config.edge_capacity):
            raise ValueError("pair creation would exceed elementary-edge capacity")
        pair_id = self._new_pair_id(request_id, born_round)
        self._pairs[pair_id] = PairState(
            pair_id=pair_id,
            request_id=request_id,
            left=left,
            right=right,
            born_round=born_round,
            elementary=elementary,
        )
        return pair_id

    def _remove_pair(self, pair_id: str) -> PairState:
        return self._pairs.pop(pair_id)

    def _release_runtime_pairs(self, runtime: _PlanRuntime) -> None:
        for pair_id in tuple(runtime.active_pair_ids):
            self._pairs.pop(pair_id, None)
        runtime.active_pair_ids.clear()

    def _attempt_generation(self, round_id: int) -> tuple[GenerationEvent, ...]:
        events: list[GenerationEvent] = []
        for request_id in self._runtime_order:
            runtime = self._runtimes[request_id]
            if not runtime.active:
                continue
            for elementary_edge in runtime.plan.elementary_edges:
                if elementary_edge not in runtime.pending_elementary:
                    continue
                u, v = elementary_edge
                if self.node_free_slots(u) <= 0 or self.node_free_slots(v) <= 0:
                    events.append(GenerationEvent(
                        round_id, request_id, elementary_edge, "blocked_memory"
                    ))
                    continue
                if (self._elementary_edge_occupancy(elementary_edge)
                        >= self.config.edge_capacity):
                    events.append(GenerationEvent(
                        round_id, request_id, elementary_edge, "blocked_edge"
                    ))
                    continue
                draw = self._uniform(
                    "generation", self.config.slot_id, round_id,
                    request_id, elementary_edge,
                )
                if draw > self.config.generation_probability:
                    events.append(GenerationEvent(
                        round_id, request_id, elementary_edge, "random_failure"
                    ))
                    continue
                pair_id = self._create_pair(
                    request_id,
                    u,
                    v,
                    born_round=round_id,
                    elementary=True,
                    enforce_edge_capacity=True,
                )
                runtime.pending_elementary.remove(elementary_edge)
                runtime.active_pair_ids.add(pair_id)
                events.append(GenerationEvent(
                    round_id, request_id, elementary_edge, "success", pair_id
                ))
        return tuple(events)

    def _incident_pairs(
        self, runtime: _PlanRuntime, middle: Node
    ) -> tuple[PairState, ...]:
        return tuple(
            self._pairs[pair_id]
            for pair_id in sorted(runtime.active_pair_ids)
            if pair_id in self._pairs and middle in self._pairs[pair_id].endpoints
        )

    def _complete_runtime(self, runtime: _PlanRuntime, round_id: int) -> None:
        runtime.completed_round = round_id
        self._release_runtime_pairs(runtime)

    def _settle_direct_requests(self, round_id: int) -> None:
        for request_id in self._runtime_order:
            runtime = self._runtimes[request_id]
            if not runtime.active or runtime.plan.swap_order:
                continue
            if runtime.pending_elementary:
                continue
            endpoint_set = {runtime.plan.path[0], runtime.plan.path[-1]}
            if any(
                set(self._pairs[pair_id].endpoints) == endpoint_set
                for pair_id in runtime.active_pair_ids
                if pair_id in self._pairs
            ):
                self._complete_runtime(runtime, round_id)

    def _execute_ready_swaps(self, round_id: int) -> tuple[SwapEvent, ...]:
        events: list[SwapEvent] = []
        bsm_used: dict[Node, int] = {}
        for request_id in self._runtime_order:
            runtime = self._runtimes[request_id]
            if not runtime.active:
                continue
            if runtime.next_swap_index >= len(runtime.plan.swap_order):
                continue
            middle = runtime.plan.swap_order[runtime.next_swap_index]
            inputs = self._incident_pairs(runtime, middle)
            if len(inputs) != 2:
                continue
            if bsm_used.get(middle, 0) >= self.config.bsm_capacity_per_node:
                events.append(SwapEvent(
                    round_id, request_id, middle, "blocked_bsm"
                ))
                continue

            bsm_used[middle] = bsm_used.get(middle, 0) + 1
            left_input, right_input = inputs
            input_ids = (left_input.pair_id, right_input.pair_id)
            left_outer = (
                left_input.right if left_input.left == middle else left_input.left
            )
            right_outer = (
                right_input.right if right_input.left == middle else right_input.left
            )
            self._remove_pair(left_input.pair_id)
            self._remove_pair(right_input.pair_id)
            runtime.active_pair_ids.difference_update(input_ids)

            draw = self._uniform(
                "swap", self.config.slot_id, round_id, request_id,
                runtime.next_swap_index, middle,
            )
            if draw > self.config.swap_probability:
                runtime.failed_round = round_id
                self._release_runtime_pairs(runtime)
                events.append(SwapEvent(
                    round_id, request_id, middle, "random_failure", input_ids
                ))
                continue

            output_id = self._create_pair(
                request_id,
                left_outer,
                right_outer,
                born_round=round_id,
                elementary=False,
                enforce_edge_capacity=False,
            )
            runtime.active_pair_ids.add(output_id)
            runtime.next_swap_index += 1
            events.append(SwapEvent(
                round_id,
                request_id,
                middle,
                "success",
                input_ids,
                output_id,
            ))

            if runtime.next_swap_index == len(runtime.plan.swap_order):
                output = self._pairs[output_id]
                expected = {runtime.plan.path[0], runtime.plan.path[-1]}
                if set(output.endpoints) != expected:
                    raise RuntimeError(
                        f"swap order for {request_id} produced {output.endpoints}, "
                        f"expected path endpoints {tuple(expected)}"
                    )
                self._complete_runtime(runtime, round_id)
        return tuple(events)

    def run(self) -> SlotResult:
        initial_occupancy = self.occupancy()
        traces: list[RoundTrace] = []
        for round_id in range(1, self.config.rounds_per_slot + 1):
            occupancy_start = self.occupancy()
            generation_events = self._attempt_generation(round_id)
            self._settle_direct_requests(round_id)
            occupancy_after_generation = self.occupancy()
            swap_events = self._execute_ready_swaps(round_id)
            occupancy_after_swaps = self.occupancy()
            traces.append(RoundTrace(
                round_id=round_id,
                occupancy_start=occupancy_start,
                occupancy_after_generation=occupancy_after_generation,
                occupancy_after_swaps=occupancy_after_swaps,
                generation_events=generation_events,
                swap_events=swap_events,
            ))

        completed = tuple(
            request_id for request_id in self._runtime_order
            if self._runtimes[request_id].completed_round is not None
        )
        failed = tuple(
            request_id for request_id in self._runtime_order
            if self._runtimes[request_id].failed_round is not None
        )
        missed = tuple(
            request_id for request_id in self._runtime_order
            if self._runtimes[request_id].active
        )
        completion_round = {
            request_id: runtime.completed_round
            for request_id, runtime in self._runtimes.items()
            if runtime.completed_round is not None
        }
        for request_id in missed:
            self._release_runtime_pairs(self._runtimes[request_id])
        return SlotResult(
            completed=completed,
            failed=failed,
            missed=missed,
            completion_round=completion_round,
            initial_occupancy=initial_occupancy,
            traces=tuple(traces),
        )


def focus_trace(result: SlotResult, node: Node) -> tuple[tuple[int, int, int, int], ...]:
    """Return ``(round, start, after_generation, after_swaps)`` for one node."""
    return tuple(
        (
            trace.round_id,
            trace.occupancy_start.get(node, 0),
            trace.occupancy_after_generation.get(node, 0),
            trace.occupancy_after_swaps.get(node, 0),
        )
        for trace in result.traces
    )
