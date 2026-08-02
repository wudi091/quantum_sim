"""Event-driven shared core for path-plus-complete-schedule batch planning.

The centralized controller acts exactly once: it selects at most one complete
``(path, swap_groups)`` candidate for every request at the control-slot
boundary.  Afterwards a fixed lower-layer executor owns physical time,
automatic elementary-link generation, BSM service, memory reset, and request
settlement.

There is deliberately no ``rounds_per_slot`` parameter.  Generation attempts
and swap opportunities arise from physical-time events determined by the slot
duration and protocol intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
import hashlib
import heapq
from typing import Iterable, Literal, Mapping, TypeAlias

from .contracts.complete_schedule import CompleteSchedule


Node: TypeAlias = int | str
Edge: TypeAlias = tuple[Node, Node]


def _node_key(node: Node) -> tuple[str, str]:
    return type(node).__name__, repr(node)


def edge(u: Node, v: Node) -> Edge:
    if u == v:
        raise ValueError("an elementary link must connect distinct nodes")
    return (u, v) if _node_key(u) <= _node_key(v) else (v, u)


@dataclass(frozen=True)
class OrderPlan:
    """One path and one complete, dependency-valid swap-group schedule.

    ``swap_order`` remains as a stable flattened label for older planner and
    observation code.  Execution semantics come exclusively from
    ``schedule.groups``; a parallel group is never silently serialized.
    """

    plan_id: str
    request_id: str
    path: tuple[Node, ...]
    swap_order: tuple[Node, ...]
    priority: int = 0
    arrival_slot: int = 0
    deadline_slot: int | None = None
    decision_slot: int = 0
    swap_groups: tuple[tuple[Node, ...], ...] | None = None
    fixed_path_baseline: bool | None = None
    schedule: CompleteSchedule = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.plan_id or not self.request_id:
            raise ValueError("plan_id and request_id must be non-empty")
        path = tuple(self.path)
        swap_order = tuple(self.swap_order)
        schedule = (
            CompleteSchedule.from_linear_order(path, swap_order)
            if self.swap_groups is None
            else CompleteSchedule(path, tuple(self.swap_groups))
        )
        if schedule.swap_order != swap_order:
            raise ValueError(
                "swap_order must equal the stable group-flattened schedule label"
            )
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "swap_order", swap_order)
        object.__setattr__(self, "swap_groups", schedule.groups)
        object.__setattr__(self, "schedule", schedule)
        if self.fixed_path_baseline is None:
            object.__setattr__(
                self,
                "fixed_path_baseline",
                schedule.is_left_to_right_linear,
            )
        if self.arrival_slot < 0 or self.decision_slot < self.arrival_slot:
            raise ValueError("plan timing must satisfy 0 <= arrival <= decision")
        if (self.deadline_slot is not None
                and self.deadline_slot < self.arrival_slot):
            raise ValueError("deadline cannot precede arrival")

    @cached_property
    def elementary_edges(self) -> tuple[Edge, ...]:
        return tuple(edge(u, v) for u, v in zip(self.path, self.path[1:]))

    @property
    def is_fixed_order(self) -> bool:
        """Whether path-only baselines project this path onto this schedule."""

        return bool(self.fixed_path_baseline)

    @property
    def swap_round_count(self) -> int:
        return self.schedule.round_count

    @cached_property
    def swap_round_by_node(self) -> dict[Node, int]:
        """Zero-based group index of every internal swap node."""

        return {
            node: group_index
            for group_index, group in enumerate(self.schedule.groups)
            for node in group
        }

    @property
    def schedule_key(self):
        return self.schedule.structural_key


@dataclass(frozen=True)
class OrderLinkSpec:
    """One physical link with an explicit buffer and HEG probability."""

    left: Node
    right: Node
    capacity: int = 1
    generation_probability: float = 1.0

    def __post_init__(self) -> None:
        canonical = edge(self.left, self.right)
        object.__setattr__(self, "left", canonical[0])
        object.__setattr__(self, "right", canonical[1])
        if self.capacity < 1:
            raise ValueError("link capacity must be positive")
        if not 0.0 <= self.generation_probability <= 1.0:
            raise ValueError("link generation probability must lie in [0, 1]")

    @property
    def elementary_edge(self) -> Edge:
        return self.left, self.right


@dataclass(frozen=True)
class OrderStoredPair:
    """One unconsumed elementary EPR carried across control-slot boundaries."""

    pair_id: str
    left: Node
    right: Node
    born_slot: int
    expires_slot: int

    def __post_init__(self) -> None:
        if not self.pair_id:
            raise ValueError("stored pair ID must be non-empty")
        canonical = edge(self.left, self.right)
        object.__setattr__(self, "left", canonical[0])
        object.__setattr__(self, "right", canonical[1])
        if self.born_slot < 0:
            raise ValueError("stored pair birth slot cannot be negative")
        if self.expires_slot <= self.born_slot:
            raise ValueError("stored pair expiry must follow its birth slot")

    @property
    def elementary_edge(self) -> Edge:
        return self.left, self.right


@dataclass(frozen=True)
class OrderCoreConfig:
    """Fixed lower-layer protocol timing for one centralized control slot."""

    slot_duration_ps: int = 3_000
    generation_interval_ps: int = 1_000
    swap_service_ps: int = 1_000
    memory_reset_ps: int = 100
    generation_probability: float = 1.0
    swap_probability: float = 1.0
    edge_capacity: int = 1
    bsm_capacity_per_node: int = 1
    epr_ttl_slots: int = 3
    seed: int = 0
    slot_id: int = 0

    def __post_init__(self) -> None:
        if self.slot_duration_ps < 1:
            raise ValueError("slot_duration_ps must be positive")
        if self.generation_interval_ps < 1:
            raise ValueError("generation_interval_ps must be positive")
        if self.swap_service_ps < 1:
            raise ValueError("swap_service_ps must be positive")
        if self.memory_reset_ps < 0:
            raise ValueError("memory_reset_ps cannot be negative")
        if not 0.0 <= self.generation_probability <= 1.0:
            raise ValueError("generation_probability must lie in [0, 1]")
        if not 0.0 <= self.swap_probability <= 1.0:
            raise ValueError("swap_probability must lie in [0, 1]")
        if self.edge_capacity < 1:
            raise ValueError("edge_capacity must be positive")
        if self.bsm_capacity_per_node < 1:
            raise ValueError("bsm_capacity_per_node must be positive")
        if self.epr_ttl_slots < 1:
            raise ValueError("epr_ttl_slots must be positive")
        if self.slot_id < 0:
            raise ValueError("slot_id cannot be negative")


@dataclass(frozen=True)
class OrderBatchProblem:
    """Immutable public input shared by every order-aware planner."""

    candidates: tuple[OrderPlan, ...]
    node_capacities: tuple[tuple[Node, int], ...]
    links: tuple[OrderLinkSpec, ...] = ()
    initial_inventory: tuple[OrderStoredPair, ...] = ()
    config: OrderCoreConfig = OrderCoreConfig()
    required_requests: frozenset[str] = frozenset()
    preloaded_requests: frozenset[str] = frozenset()
    name: str = "order-batch"

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("an order batch needs at least one candidate")
        plan_ids = [plan.plan_id for plan in self.candidates]
        if len(set(plan_ids)) != len(plan_ids):
            raise ValueError("candidate plan IDs must be unique")
        capacity = dict(self.node_capacities)
        if len(capacity) != len(self.node_capacities):
            raise ValueError("node_capacities cannot repeat a node")
        if any(value < 1 for value in capacity.values()):
            raise ValueError("all node capacities must be positive")
        required_nodes = {
            node for plan in self.candidates for node in plan.path
        }
        missing_nodes = required_nodes - capacity.keys()
        if missing_nodes:
            raise ValueError(
                f"missing node capacities: {sorted(map(repr, missing_nodes))}"
            )
        request_ids = {plan.request_id for plan in self.candidates}
        unknown_required = self.required_requests - request_ids
        unknown_preloaded = self.preloaded_requests - request_ids
        if unknown_required or unknown_preloaded:
            raise ValueError("required/preloaded request is absent from candidates")
        if not self.preloaded_requests <= self.required_requests:
            raise ValueError("a preloaded request must also be required")
        link_edges = [link.elementary_edge for link in self.links]
        if len(set(link_edges)) != len(link_edges):
            raise ValueError("physical links cannot repeat an edge")
        if any(
            link.left not in capacity or link.right not in capacity
            for link in self.links
        ):
            raise ValueError("physical link endpoint is absent from node capacities")
        candidate_edges = {
            elementary_edge
            for plan in self.candidates
            for elementary_edge in plan.elementary_edges
        }
        if self.links:
            missing_links = candidate_edges - set(link_edges)
            if missing_links:
                raise ValueError(
                    "candidate uses non-physical links: "
                    f"{sorted(map(repr, missing_links))}"
                )
        pair_ids = [pair.pair_id for pair in self.initial_inventory]
        if len(set(pair_ids)) != len(pair_ids):
            raise ValueError("stored pair IDs must be unique")
        physical_edges = set(link_edges) if self.links else candidate_edges
        link_capacity = (
            {link.elementary_edge: link.capacity for link in self.links}
            if self.links else {
                elementary_edge: self.config.edge_capacity
                for elementary_edge in candidate_edges
            }
        )
        node_use = {node: 0 for node in capacity}
        edge_use = {elementary_edge: 0 for elementary_edge in physical_edges}
        for pair in self.initial_inventory:
            if pair.left not in capacity or pair.right not in capacity:
                raise ValueError(
                    "stored pair endpoint is absent from node capacities"
                )
            if pair.elementary_edge not in physical_edges:
                raise ValueError("stored pair is absent from physical links")
            if not (
                pair.born_slot <= self.config.slot_id < pair.expires_slot
            ):
                raise ValueError(
                    "stored pair must be alive in the problem's control slot"
                )
            node_use[pair.left] += 1
            node_use[pair.right] += 1
            edge_use[pair.elementary_edge] += 1
        if any(node_use[node] > capacity[node] for node in capacity):
            raise ValueError("stored inventory exceeds node memory capacity")
        if any(
            edge_use[elementary_edge] > link_capacity[elementary_edge]
            for elementary_edge in physical_edges
        ):
            raise ValueError("stored inventory exceeds physical link capacity")
        for request_id in self.preloaded_requests:
            paths = {
                plan.path for plan in self.candidates
                if plan.request_id == request_id
            }
            if len(paths) != 1:
                raise ValueError(
                    "all order candidates of a preloaded request must share one path"
                )

    @classmethod
    def create(
        cls,
        *,
        candidates: Iterable[OrderPlan],
        node_capacity: Mapping[Node, int],
        links: Iterable[OrderLinkSpec] = (),
        initial_inventory: Iterable[OrderStoredPair] = (),
        config: OrderCoreConfig = OrderCoreConfig(),
        required_requests: Iterable[str] = (),
        preloaded_requests: Iterable[str] = (),
        name: str = "order-batch",
    ) -> "OrderBatchProblem":
        capacities = tuple(sorted(
            node_capacity.items(), key=lambda item: _node_key(item[0])
        ))
        return cls(
            candidates=tuple(candidates),
            node_capacities=capacities,
            links=tuple(sorted(
                links,
                key=lambda item: (
                    _node_key(item.left), _node_key(item.right)
                ),
            )),
            initial_inventory=tuple(sorted(
                initial_inventory,
                key=lambda item: item.pair_id,
            )),
            config=config,
            required_requests=frozenset(required_requests),
            preloaded_requests=frozenset(preloaded_requests),
            name=name,
        )

    @cached_property
    def _capacity_cache(self) -> dict[Node, int]:
        return dict(self.node_capacities)

    @property
    def capacity(self) -> dict[Node, int]:
        # Preserve the original copy-on-access public contract.  The immutable
        # problem and its shared planner snapshot must not become mutable just
        # because their derived representation is cached internally.
        return dict(self._capacity_cache)

    @cached_property
    def _link_by_edge_cache(self) -> dict[Edge, OrderLinkSpec]:
        if self.links:
            return {link.elementary_edge: link for link in self.links}
        return {
            elementary_edge: OrderLinkSpec(
                *elementary_edge,
                capacity=self.config.edge_capacity,
                generation_probability=self.config.generation_probability,
            )
            for elementary_edge in {
                value
                for plan in self.candidates
                for value in plan.elementary_edges
            }
        }

    @property
    def link_by_edge(self) -> dict[Edge, OrderLinkSpec]:
        return dict(self._link_by_edge_cache)

    @cached_property
    def physical_edges(self) -> tuple[Edge, ...]:
        return tuple(sorted(
            self._link_by_edge_cache,
            key=lambda value: (_node_key(value[0]), _node_key(value[1])),
        ))

    def link_capacity(self, elementary_edge: Edge) -> int:
        return self._link_by_edge_cache[edge(*elementary_edge)].capacity

    def link_generation_probability(self, elementary_edge: Edge) -> float:
        return self._link_by_edge_cache[
            edge(*elementary_edge)
        ].generation_probability

    def with_physics_seed(self, seed: int) -> "OrderBatchProblem":
        """Return the same public planning instance with another RNG stream."""
        return replace(
            self,
            config=replace(self.config, seed=int(seed)),
        )

    def public_view(self) -> "OrderBatchProblem":
        """Hide the realized physical RNG stream from every planner."""
        return self.with_physics_seed(0)


@dataclass(frozen=True)
class OrderBatchSnapshot:
    """Read-only planner view; no backend object or mutation method is exposed."""

    problem: OrderBatchProblem

    @property
    def candidates(self) -> tuple[OrderPlan, ...]:
        return self.problem.candidates


@dataclass(frozen=True)
class _Pair:
    pair_id: str
    request_id: str | None
    left: Node
    right: Node
    elementary: bool
    born_slot: int
    expires_slot: int

    @property
    def endpoints(self) -> tuple[Node, Node]:
        return self.left, self.right


GenerationStatus = Literal[
    "success", "random_failure", "blocked_memory", "blocked_edge"
]
SwapStatus = Literal[
    "success", "random_failure", "blocked_bsm", "insufficient_time"
]


@dataclass(frozen=True)
class OrderGenerationEvent:
    time_ps: int
    request_id: str
    elementary_edge: Edge
    status: GenerationStatus


@dataclass(frozen=True)
class OrderSwapEvent:
    time_ps: int
    request_id: str
    middle: Node
    status: SwapStatus


@dataclass(frozen=True)
class OrderEventTrace:
    time_ps: int
    occupancy_before: dict[Node, int]
    occupancy_after_generation: dict[Node, int]
    occupancy_after_swaps: dict[Node, int]
    generation_events: tuple[OrderGenerationEvent, ...]
    swap_events: tuple[OrderSwapEvent, ...]


@dataclass(frozen=True)
class OrderSlotResult:
    selected_plan_ids: tuple[str, ...]
    completed: tuple[str, ...]
    failed: tuple[str, ...]
    missed: tuple[str, ...]
    completion_time_ps: dict[str, int]
    traces: tuple[OrderEventTrace, ...]
    remaining_inventory: tuple[OrderStoredPair, ...]

    @property
    def completed_count(self) -> int:
        return len(self.completed)

    @property
    def completion_rate(self) -> float:
        total = len(self.completed) + len(self.failed) + len(self.missed)
        return self.completed_count / max(total, 1)


@dataclass
class _Runtime:
    plan: OrderPlan
    pending_elementary: set[Edge]
    active_pair_ids: set[str]
    generation_attempts: dict[Edge, int]
    next_group_index: int = 0
    next_swap_time_ps: int = 0
    finishing_time_ps: int | None = None
    completed_time_ps: int | None = None
    failed_time_ps: int | None = None

    @property
    def active(self) -> bool:
        return (
            self.finishing_time_ps is None
            and self.completed_time_ps is None
            and self.failed_time_ps is None
        )


class _OrderExecution:
    def __init__(
        self,
        problem: OrderBatchProblem,
        plans: tuple[OrderPlan, ...],
        *,
        record_traces: bool,
    ) -> None:
        self.problem = problem
        self.config = problem.config
        self.capacity = problem._capacity_cache
        self.links = problem._link_by_edge_cache
        self.record_traces = record_traces
        ordered = tuple(sorted(
            plans, key=lambda plan: (plan.priority, plan.request_id, plan.plan_id)
        ))
        self.runtime_order = tuple(plan.request_id for plan in ordered)
        self.runtimes = {
            plan.request_id: _Runtime(
                plan=plan,
                pending_elementary=set(plan.elementary_edges),
                active_pair_ids=set(),
                generation_attempts={},
            )
            for plan in ordered
        }
        self.pairs: dict[str, _Pair] = {}
        self._pair_use_by_node: dict[Node, int] = {
            node: 0 for node in self.capacity
        }
        self._elementary_pair_use_by_edge: dict[Edge, int] = {
            elementary_edge: 0 for elementary_edge in self.links
        }
        self.resetting: dict[Node, list[int]] = {
            node: [] for node in self.capacity
        }
        self.bsm_busy_until: dict[Node, list[int]] = {
            node: [] for node in self.capacity
        }
        self.counter = 0
        self.event_heap: list[int] = []
        self.queued_times: set[int] = set()
        self.generation_times = set(range(
            0,
            self.config.slot_duration_ps,
            self.config.generation_interval_ps,
        ))
        for time_ps in self.generation_times:
            self._queue(time_ps)
        self._load_inventory()
        self._preload()
        self._assign_inventory()

    def _queue(self, time_ps: int) -> None:
        if time_ps > self.config.slot_duration_ps or time_ps in self.queued_times:
            return
        heapq.heappush(self.event_heap, time_ps)
        self.queued_times.add(time_ps)

    def _uniform(self, *parts: object) -> float:
        payload = "|".join(map(str, (self.config.seed, *parts))).encode()
        digest = hashlib.sha256(payload).digest()[:8]
        return int.from_bytes(digest, "big") / 2**64

    def _new_pair_id(self, request_id: str, time_ps: int) -> str:
        while True:
            pair_id = (
                f"ord-s{self.config.slot_id}-t{time_ps}-"
                f"{request_id}-{self.counter}"
            )
            self.counter += 1
            if pair_id not in self.pairs:
                return pair_id

    def _release_due(self, time_ps: int) -> None:
        for values in self.resetting.values():
            values[:] = [until for until in values if until > time_ps]
        for values in self.bsm_busy_until.values():
            values[:] = [until for until in values if until > time_ps]
        for runtime in self.runtimes.values():
            if runtime.finishing_time_ps != time_ps:
                continue
            runtime.completed_time_ps = time_ps
            runtime.finishing_time_ps = None
            self._release_runtime_pairs(runtime)

    def node_occupancy(self, node: Node) -> int:
        return self._pair_use_by_node[node] + len(self.resetting[node])

    def occupancy(self) -> dict[Node, int]:
        return {node: self.node_occupancy(node) for node in self.capacity}

    def node_free_slots(self, node: Node) -> int:
        return self.capacity[node] - self.node_occupancy(node)

    def _elementary_edge_occupancy(self, elementary_edge: Edge) -> int:
        return self._elementary_pair_use_by_edge[elementary_edge]

    def _update_pair_occupancy(self, pair: _Pair, delta: int) -> None:
        """Apply one pair insertion/removal to the maintained counters."""

        self._pair_use_by_node[pair.left] += delta
        self._pair_use_by_node[pair.right] += delta
        if pair.elementary:
            elementary_edge = edge(pair.left, pair.right)
            self._elementary_pair_use_by_edge[elementary_edge] += delta

    def _store_pair(self, pair: _Pair) -> None:
        """Insert or replace a pair while keeping occupancy exact."""

        previous = self.pairs.get(pair.pair_id)
        if previous is not None:
            self._update_pair_occupancy(previous, -1)
        self.pairs[pair.pair_id] = pair
        self._update_pair_occupancy(pair, 1)

    def _remove_pair(self, pair_id: str) -> _Pair | None:
        """Remove a pair, returning ``None`` when it is already absent."""

        pair = self.pairs.pop(pair_id, None)
        if pair is not None:
            self._update_pair_occupancy(pair, -1)
        return pair

    def _replace_pair(self, pair_id: str, **changes: object) -> _Pair:
        """Replace pair metadata through the same counter-aware path."""

        pair = replace(self.pairs[pair_id], **changes)
        self._store_pair(pair)
        return pair

    def _create_pair(
        self,
        request_id: str | None,
        left: Node,
        right: Node,
        time_ps: int,
        *,
        elementary: bool,
        enforce_edge_capacity: bool,
    ) -> str:
        if self.node_free_slots(left) <= 0 or self.node_free_slots(right) <= 0:
            raise ValueError("pair creation would exceed node memory capacity")
        elementary_edge = edge(left, right)
        if (enforce_edge_capacity
                and self._elementary_edge_occupancy(elementary_edge)
                >= self.links[elementary_edge].capacity):
            raise ValueError("pair creation would exceed elementary edge capacity")
        pair_id = self._new_pair_id(request_id, time_ps)
        self._store_pair(_Pair(
            pair_id=pair_id,
            request_id=request_id,
            left=left,
            right=right,
            elementary=elementary,
            born_slot=self.config.slot_id,
            expires_slot=(
                self.config.slot_id + self.config.epr_ttl_slots
            ),
        ))
        return pair_id

    def _load_inventory(self) -> None:
        for stored in self.problem.initial_inventory:
            self._store_pair(_Pair(
                pair_id=stored.pair_id,
                request_id=None,
                left=stored.left,
                right=stored.right,
                elementary=True,
                born_slot=stored.born_slot,
                expires_slot=stored.expires_slot,
            ))

    def _preload(self) -> None:
        for request_id in self.runtime_order:
            if request_id not in self.problem.preloaded_requests:
                continue
            runtime = self.runtimes[request_id]
            for elementary_edge in runtime.plan.elementary_edges:
                pair_id = self._create_pair(
                    request_id,
                    elementary_edge[0],
                    elementary_edge[1],
                    0,
                    elementary=True,
                    enforce_edge_capacity=True,
                )
                runtime.active_pair_ids.add(pair_id)
                runtime.pending_elementary.remove(elementary_edge)

    def _assign_inventory(self) -> None:
        for request_id in self.runtime_order:
            runtime = self.runtimes[request_id]
            for elementary_edge in runtime.plan.elementary_edges:
                if elementary_edge not in runtime.pending_elementary:
                    continue
                available = sorted(
                    pair.pair_id for pair in self.pairs.values()
                    if pair.request_id is None
                    and pair.elementary
                    and edge(pair.left, pair.right) == elementary_edge
                )
                if not available:
                    continue
                pair_id = available[0]
                self._replace_pair(pair_id, request_id=request_id)
                runtime.active_pair_ids.add(pair_id)
                runtime.pending_elementary.remove(elementary_edge)

    def _release_runtime_pairs(
        self,
        runtime: _Runtime,
        *,
        preserve_elementary: bool = False,
    ) -> None:
        for pair_id in tuple(runtime.active_pair_ids):
            pair = self.pairs.get(pair_id)
            if pair is None:
                continue
            if preserve_elementary and pair.elementary:
                self._replace_pair(pair_id, request_id=None)
            else:
                self._remove_pair(pair_id)
        runtime.active_pair_ids.clear()

    def _attempt_generation(
        self, time_ps: int
    ) -> tuple[OrderGenerationEvent, ...]:
        events: list[OrderGenerationEvent] = []
        for request_id in self.runtime_order:
            runtime = self.runtimes[request_id]
            if not runtime.active:
                continue
            for elementary_edge in runtime.plan.elementary_edges:
                if elementary_edge not in runtime.pending_elementary:
                    continue
                u, v = elementary_edge
                if self.node_free_slots(u) <= 0 or self.node_free_slots(v) <= 0:
                    events.append(OrderGenerationEvent(
                        time_ps, request_id, elementary_edge, "blocked_memory"
                    ))
                    continue
                if (self._elementary_edge_occupancy(elementary_edge)
                        >= self.links[elementary_edge].capacity):
                    events.append(OrderGenerationEvent(
                        time_ps, request_id, elementary_edge, "blocked_edge"
                    ))
                    continue
                attempt_index = runtime.generation_attempts.get(
                    elementary_edge, 0
                )
                runtime.generation_attempts[elementary_edge] = attempt_index + 1
                draw = self._uniform(
                    "generation", self.config.slot_id,
                    request_id, elementary_edge, attempt_index,
                )
                if draw > self.links[elementary_edge].generation_probability:
                    events.append(OrderGenerationEvent(
                        time_ps, request_id, elementary_edge, "random_failure"
                    ))
                    continue
                pair_id = self._create_pair(
                    request_id,
                    u,
                    v,
                    time_ps,
                    elementary=True,
                    enforce_edge_capacity=True,
                )
                runtime.pending_elementary.remove(elementary_edge)
                runtime.active_pair_ids.add(pair_id)
                events.append(OrderGenerationEvent(
                    time_ps, request_id, elementary_edge, "success"
                ))
        return tuple(events)

    def _incident_pairs(
        self, runtime: _Runtime, middle: Node
    ) -> tuple[_Pair, ...]:
        return tuple(
            self.pairs[pair_id]
            for pair_id in sorted(runtime.active_pair_ids)
            if pair_id in self.pairs and middle in self.pairs[pair_id].endpoints
        )

    def _settle_direct_requests(self, time_ps: int) -> None:
        for request_id in self.runtime_order:
            runtime = self.runtimes[request_id]
            if not runtime.active or runtime.plan.schedule.groups:
                continue
            if runtime.pending_elementary:
                continue
            endpoints = {runtime.plan.path[0], runtime.plan.path[-1]}
            if any(
                set(self.pairs[pair_id].endpoints) == endpoints
                for pair_id in runtime.active_pair_ids
                if pair_id in self.pairs
            ):
                runtime.completed_time_ps = time_ps
                self._release_runtime_pairs(runtime)

    def _reserve_reset(self, middle: Node, time_ps: int, count: int = 2) -> None:
        if self.config.memory_reset_ps == 0:
            return
        until = time_ps + self.config.memory_reset_ps
        self.resetting[middle].extend([until] * count)
        self._queue(until)

    def _bsm_available(self, middle: Node, time_ps: int) -> bool:
        active = sum(until > time_ps for until in self.bsm_busy_until[middle])
        return active < self.config.bsm_capacity_per_node

    def _execute_swaps(self, time_ps: int) -> tuple[OrderSwapEvent, ...]:
        events: list[OrderSwapEvent] = []
        for request_id in self.runtime_order:
            runtime = self.runtimes[request_id]
            if not runtime.active or runtime.next_swap_time_ps > time_ps:
                continue
            groups = runtime.plan.schedule.groups
            if runtime.next_group_index >= len(groups):
                continue
            group = groups[runtime.next_group_index]
            prepared: list[tuple[Node, _Pair, _Pair, Node, Node]] = []
            consumed_ids: set[str] = set()
            ready = True
            for middle in group:
                inputs = self._incident_pairs(runtime, middle)
                if len(inputs) != 2:
                    ready = False
                    break
                left_input, right_input = inputs
                input_ids = {left_input.pair_id, right_input.pair_id}
                if consumed_ids & input_ids:
                    raise RuntimeError(
                        "validated parallel swap group shares an input pair"
                    )
                consumed_ids.update(input_ids)
                left_outer = (
                    left_input.right
                    if left_input.left == middle else left_input.left
                )
                right_outer = (
                    right_input.right
                    if right_input.left == middle else right_input.left
                )
                prepared.append((
                    middle,
                    left_input,
                    right_input,
                    left_outer,
                    right_outer,
                ))
            if not ready:
                continue
            finish = time_ps + self.config.swap_service_ps
            if finish > self.config.slot_duration_ps:
                events.extend(
                    OrderSwapEvent(
                        time_ps, request_id, middle, "insufficient_time"
                    )
                    for middle in group
                )
                continue
            blocked = tuple(
                middle for middle in group
                if not self._bsm_available(middle, time_ps)
            )
            if blocked:
                retry = min(
                    min(self.bsm_busy_until[middle])
                    for middle in blocked
                )
                runtime.next_swap_time_ps = max(runtime.next_swap_time_ps, retry)
                self._queue(runtime.next_swap_time_ps)
                events.extend(
                    OrderSwapEvent(
                        time_ps, request_id, middle, "blocked_bsm"
                    )
                    for middle in blocked
                )
                continue

            self._queue(finish)
            for middle in group:
                self.bsm_busy_until[middle].append(finish)
            for pair_id in consumed_ids:
                self._remove_pair(pair_id)
            runtime.active_pair_ids.difference_update(consumed_ids)

            failed = False
            output_ids: list[str] = []
            for (
                middle,
                _left_input,
                _right_input,
                left_outer,
                right_outer,
            ) in prepared:
                self._reserve_reset(middle, time_ps)
                draw = self._uniform(
                    "swap", self.config.slot_id, request_id, middle,
                )
                if draw > self.config.swap_probability:
                    failed = True
                    events.append(OrderSwapEvent(
                        time_ps, request_id, middle, "random_failure"
                    ))
                    continue
                output_id = self._create_pair(
                    request_id,
                    left_outer,
                    right_outer,
                    time_ps,
                    elementary=False,
                    enforce_edge_capacity=False,
                )
                runtime.active_pair_ids.add(output_id)
                output_ids.append(output_id)
                events.append(OrderSwapEvent(
                    time_ps, request_id, middle, "success"
                ))

            if failed:
                runtime.failed_time_ps = finish
                self._release_runtime_pairs(
                    runtime, preserve_elementary=True
                )
                continue

            runtime.next_group_index += 1
            runtime.next_swap_time_ps = finish

            if runtime.next_group_index == len(groups):
                if len(output_ids) != 1:
                    raise RuntimeError(
                        "final swap group must produce one end-to-end pair"
                    )
                output = self.pairs[output_ids[0]]
                expected = {runtime.plan.path[0], runtime.plan.path[-1]}
                if set(output.endpoints) != expected:
                    raise RuntimeError(
                        f"schedule {runtime.plan.schedule.groups} produced "
                        f"{output.endpoints}, expected path endpoints"
                    )
                runtime.finishing_time_ps = finish
            else:
                self._queue(finish)
        return tuple(events)

    def run(self) -> OrderSlotResult:
        traces: list[OrderEventTrace] = []
        while self.event_heap:
            time_ps = heapq.heappop(self.event_heap)
            self.queued_times.remove(time_ps)
            self._release_due(time_ps)
            occupancy_before = self.occupancy() if self.record_traces else None
            generation_events = (
                self._attempt_generation(time_ps)
                if time_ps in self.generation_times else ()
            )
            self._settle_direct_requests(time_ps)
            occupancy_after_generation = (
                self.occupancy() if self.record_traces else None
            )
            swap_events = self._execute_swaps(time_ps)
            occupancy_after_swaps = (
                self.occupancy() if self.record_traces else None
            )
            if self.record_traces and (generation_events or swap_events):
                assert occupancy_before is not None
                assert occupancy_after_generation is not None
                assert occupancy_after_swaps is not None
                traces.append(OrderEventTrace(
                    time_ps=time_ps,
                    occupancy_before=occupancy_before,
                    occupancy_after_generation=occupancy_after_generation,
                    occupancy_after_swaps=occupancy_after_swaps,
                    generation_events=generation_events,
                    swap_events=swap_events,
                ))

        completed = tuple(
            request_id for request_id in self.runtime_order
            if self.runtimes[request_id].completed_time_ps is not None
        )
        failed = tuple(
            request_id for request_id in self.runtime_order
            if self.runtimes[request_id].failed_time_ps is not None
        )
        missed = tuple(
            request_id for request_id in self.runtime_order
            if request_id not in completed and request_id not in failed
        )
        for request_id in missed:
            self._release_runtime_pairs(
                self.runtimes[request_id], preserve_elementary=True
            )
        completion_time = {
            request_id: runtime.completed_time_ps
            for request_id, runtime in self.runtimes.items()
            if runtime.completed_time_ps is not None
        }
        remaining_inventory = tuple(
            OrderStoredPair(
                pair_id=pair.pair_id,
                left=pair.left,
                right=pair.right,
                born_slot=pair.born_slot,
                expires_slot=pair.expires_slot,
            )
            for pair in sorted(
                self.pairs.values(), key=lambda item: item.pair_id
            )
            if pair.request_id is None and pair.elementary
        )
        return OrderSlotResult(
            selected_plan_ids=tuple(
                self.runtimes[request_id].plan.plan_id
                for request_id in self.runtime_order
            ),
            completed=completed,
            failed=failed,
            missed=missed,
            completion_time_ps=completion_time,
            traces=tuple(traces),
            remaining_inventory=remaining_inventory,
        )


def simulate_order_batch(
    problem: OrderBatchProblem,
    plan_ids: Iterable[str],
    *,
    record_traces: bool = True,
) -> OrderSlotResult:
    lookup = {plan.plan_id: plan for plan in problem.candidates}
    selected_ids = tuple(plan_ids)
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("a plan cannot be selected twice")
    unknown = set(selected_ids) - lookup.keys()
    if unknown:
        raise ValueError(f"unknown plan IDs: {sorted(unknown)}")
    plans = tuple(lookup[plan_id] for plan_id in selected_ids)
    request_ids = [plan.request_id for plan in plans]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("a request may select at most one complete plan")
    missing_required = problem.required_requests - set(request_ids)
    if missing_required:
        raise ValueError(
            f"required requests have no selected plan: {sorted(missing_required)}"
        )
    return _OrderExecution(
        problem, plans, record_traces=record_traces
    ).run()


class OrderAwareBatchEnv:
    """One-shot environment shared unchanged by all benchmark planners."""

    def __init__(self, problem: OrderBatchProblem):
        self.problem = problem
        self._committed = False
        self._result: OrderSlotResult | None = None

    def snapshot(self) -> OrderBatchSnapshot:
        return OrderBatchSnapshot(self.problem.public_view())

    def commit(self, plan_ids: Iterable[str]) -> OrderSlotResult:
        if self._committed:
            raise RuntimeError("an order-aware batch can be committed only once")
        self._result = simulate_order_batch(
            self.problem, tuple(plan_ids), record_traces=True
        )
        self._committed = True
        return self._result

    @property
    def result(self) -> OrderSlotResult | None:
        return self._result
