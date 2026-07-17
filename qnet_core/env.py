"""Algorithm-independent episode control on the SeQUeNCe resource kernel."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Iterable

import networkx as nx

from .planner_api import (
    PlanDescriptor, PlanningSnapshot, ResourceClaim, SwapAction, SwapLane,
)
from .sequence_backend import SequenceBackend
from .spec import EpisodeSpec, RequestSpec


@dataclass
class RequestState:
    spec: RequestSpec
    frontier: int
    carried_pair_id: str | None = None
    carried_pair_ids: tuple[str, ...] = ()
    delivered_pairs: int = 0
    completed_at: int | None = None
    expired_at: int | None = None

    @property
    def active(self) -> bool:
        return self.completed_at is None and self.expired_at is None


class SharedRoutingEnv:
    """One request/physics/settlement implementation for every planner."""

    def __init__(self, spec: EpisodeSpec, candidate_count: int = 3):
        self.spec = spec
        self.candidate_count = max(1, int(candidate_count))
        self.graph = nx.Graph()
        self.graph.add_nodes_from(spec.nodes)
        self.graph.add_edges_from(spec.edges)
        self._distances = dict(nx.all_pairs_shortest_path_length(self.graph))
        self.backend = SequenceBackend(spec)
        self.requests = {
            request.id: RequestState(request, request.source)
            for request in spec.requests
        }
        self._initial_hops = {
            request.id: float(self._distances.get(request.source, {}).get(
                request.destination, spec.horizon
            ))
            for request in spec.requests
        }
        self.generated_eprs = 0
        self.swaps = 0
        self.failed_plans = 0
        self.successful_plans = 0
        self.partial_plan_successes = 0
        self.progress_hops = 0.0
        self.positive_progress_hops = 0.0
        self.lost_progress_hops = 0.0
        self.delivered_pairs = 0
        self.recovery_attempts = 0
        self.recovery_successes = 0
        self.allocation_claims = 0
        self.allocation_successes = 0
        self.released_surplus_pairs = 0
        self.phase = "allocate" if spec.physical.max_width > 1 else "primary"
        self._active_allocation_ids: set[str] = set()
        self._allocated_pair_ids: set[str] = set()
        self._prepared_time: int | None = None
        self._candidates: dict[str, PlanDescriptor] = {}

    @property
    def time(self) -> int:
        return self.backend.time

    @staticmethod
    def _carried_ids(state: RequestState) -> tuple[str, ...]:
        if state.carried_pair_ids:
            return state.carried_pair_ids
        return () if state.carried_pair_id is None else (state.carried_pair_id,)

    @staticmethod
    def _set_carried(state: RequestState, pair_ids: Iterable[str]) -> None:
        values = tuple(pair_ids)
        state.carried_pair_ids = values
        state.carried_pair_id = values[0] if values else None

    def _prepare_slot(self) -> None:
        if self._prepared_time == self.time:
            return
        for state in self.requests.values():
            carried = tuple(
                pair_id for pair_id in self._carried_ids(state)
                if pair_id in self.backend.pairs
            )
            if len(carried) != len(self._carried_ids(state)):
                self._set_carried(state, carried)
            if not carried and state.frontier != state.spec.source:
                state.frontier = state.spec.source
        if self.phase == "allocate":
            self._candidates = self._build_allocation_candidates()
        elif self.phase == "recover":
            self._candidates = self._build_recovery_candidates()
        else:
            self.generated_eprs += len(self.backend.generate_elementary_pairs())
            self._candidates = self._build_candidates()
        self._prepared_time = self.time

    def _available_pair(self, u: int, v: int) -> str | None:
        matches = [
            pair for pair in self.backend.pairs.values()
            if pair.owner_request is None and set(pair.endpoints) == {u, v}
        ]
        if not matches:
            return None
        return max(matches, key=lambda pair: (pair.fidelity, pair.pair_id)).pair_id

    @staticmethod
    def _compile_actions(
        request_id: str,
        route_nodes: tuple[int, ...],
        pair_ids: list[str],
        carried: bool,
    ) -> tuple[SwapAction, ...]:
        if len(pair_ids) <= 1:
            return ()
        actions: list[SwapAction] = []
        left_ref = pair_ids[0]
        middle_offset = 0 if carried else 1
        for index, right_id in enumerate(pair_ids[1:]):
            middle = route_nodes[index + middle_offset]
            actions.append(SwapAction(request_id, middle, left_ref, right_id))
            left_ref = f"@{index}"
        return tuple(actions)

    def _expected_throughput(self, hops: int, width: int) -> float:
        """QCAST EXT for homogeneous generation and swapping probabilities."""
        p = self.spec.physical.generation_probability
        q = self.spec.physical.swap_probability
        expected_bottleneck = 0.0
        for minimum in range(1, width + 1):
            tail = sum(
                math.comb(width, successes)
                * p**successes * (1.0 - p) ** (width - successes)
                for successes in range(minimum, width + 1)
            )
            expected_bottleneck += tail**max(hops, 1)
        return float(expected_bottleneck * q ** max(hops - 1, 0))

    def _build_allocation_candidates(self) -> dict[str, PlanDescriptor]:
        candidates: dict[str, PlanDescriptor] = {}
        max_width = self.spec.physical.max_width
        for state in self.requests.values():
            request = state.spec
            if not state.active or request.arrival > self.time:
                continue
            try:
                paths = list(itertools.islice(
                    nx.shortest_simple_paths(self.graph, state.frontier, request.destination),
                    self.candidate_count,
                ))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            proposals: list[tuple[int, ...]] = []
            for path in paths:
                route = tuple(int(node) for node in path)
                proposals.append(route)
                if len(route) > 2:
                    proposals.append(route[:2])
            seen: set[tuple[int, ...]] = set()
            slot = 0
            for route in proposals:
                if route in seen:
                    continue
                seen.add(route)
                for width in range(1, max_width + 1):
                    claims = tuple(
                        ResourceClaim(int(u), int(v), lane)
                        for lane in range(width)
                        for u, v in zip(route, route[1:])
                    )
                    plan_id = f"a{self.time}:{request.id}:{slot}"
                    hops = len(route) - 1
                    candidates[plan_id] = PlanDescriptor(
                        plan_id=plan_id,
                        request_id=request.id,
                        route_nodes=route,
                        reached_node=route[-1],
                        elementary_pair_ids=(),
                        swap_actions=(),
                        duration=0,
                        remaining_hops=int(self._distances[route[-1]][request.destination]),
                        completes_request=route[-1] == request.destination,
                        kind="allocation",
                        width=width,
                        claims=claims,
                        allocation_id=plan_id,
                        expected_throughput=self._expected_throughput(hops, width),
                        memory_cost=2 * len(claims),
                    )
                    slot += 1
                    if slot >= self.candidate_count:
                        break
                if slot >= self.candidate_count:
                    break
        return candidates

    def _build_recovery_candidates(self) -> dict[str, PlanDescriptor]:
        candidates: dict[str, PlanDescriptor] = {}
        edge_pairs: dict[tuple[int, int], list[str]] = {}
        resource_graph = nx.Graph()
        resource_graph.add_nodes_from(self.graph.nodes)
        for pair_id in sorted(self._allocated_pair_ids):
            pair = self.backend.pairs.get(pair_id)
            if pair is None or pair.owner_request is not None:
                continue
            edge = (min(pair.left, pair.right), max(pair.left, pair.right))
            edge_pairs.setdefault(edge, []).append(pair_id)
            resource_graph.add_edge(*edge)
        for values in edge_pairs.values():
            values.sort()

        for state in self.requests.values():
            request = state.spec
            if not state.active or request.arrival > self.time:
                continue
            frontier = state.frontier
            if frontier not in resource_graph:
                continue
            routes: list[tuple[int, ...]] = []
            try:
                if nx.has_path(resource_graph, frontier, request.destination):
                    routes = [tuple(map(int, path)) for path in itertools.islice(
                        nx.shortest_simple_paths(resource_graph, frontier, request.destination),
                        self.candidate_count,
                    )]
                else:
                    current_remaining = self._distances[frontier][request.destination]
                    reachable = nx.node_connected_component(resource_graph, frontier)
                    targets = sorted(
                        ((
                            self._distances[node][request.destination], int(node)
                        )
                            for node in reachable
                            if self._distances[node][request.destination] < current_remaining
                        ),
                        key=lambda item: (item[0], item[1]),
                    )
                    routes = [
                        tuple(map(int, nx.shortest_path(resource_graph, frontier, target)))
                        for _, target in targets[:self.candidate_count]
                    ]
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            carried = self._carried_ids(state)
            slot = 0
            for route in routes:
                if len(route) < 2:
                    continue
                available_width = min(
                    len(edge_pairs[(min(u, v), max(u, v))])
                    for u, v in zip(route, route[1:])
                )
                if carried:
                    available_width = min(available_width, len(carried))
                available_width = min(available_width, self.spec.physical.max_width)
                for width in range(1, available_width + 1):
                    lanes: list[SwapLane] = []
                    flat_ids: list[str] = []
                    for lane_index in range(width):
                        base_ids = [
                            edge_pairs[(min(u, v), max(u, v))][lane_index]
                            for u, v in zip(route, route[1:])
                        ]
                        pair_ids = ([carried[lane_index]] if carried else []) + base_ids
                        actions = self._compile_actions(
                            request.id, route, pair_ids, bool(carried)
                        )
                        lanes.append(SwapLane(lane_index, tuple(pair_ids), actions))
                        flat_ids.extend(pair_ids)
                    plan_id = f"r{self.time}:{request.id}:{slot}"
                    realized_ext = sum(
                        self.spec.physical.swap_probability ** len(lane.swap_actions)
                        for lane in lanes
                    )
                    candidates[plan_id] = PlanDescriptor(
                        plan_id=plan_id,
                        request_id=request.id,
                        route_nodes=route,
                        reached_node=route[-1],
                        elementary_pair_ids=tuple(dict.fromkeys(flat_ids)),
                        swap_actions=lanes[0].swap_actions,
                        duration=max(1, max(len(lane.swap_actions) for lane in lanes)),
                        remaining_hops=int(self._distances[route[-1]][request.destination]),
                        completes_request=route[-1] == request.destination,
                        kind="recovery",
                        width=width,
                        lanes=tuple(lanes),
                        expected_throughput=float(realized_ext),
                        memory_cost=len(set(flat_ids)),
                    )
                    slot += 1
                    if slot >= self.candidate_count:
                        break
                if slot >= self.candidate_count:
                    break
        return candidates

    def _build_candidates(self) -> dict[str, PlanDescriptor]:
        candidates: dict[str, PlanDescriptor] = {}
        for state in self.requests.values():
            request = state.spec
            if not state.active or request.arrival > self.time:
                continue
            try:
                paths = itertools.islice(
                    nx.shortest_simple_paths(self.graph, state.frontier, request.destination),
                    self.candidate_count,
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            resource_paths: list[tuple[list[int], list[str], int]] = []
            for path in paths:
                route = [int(path[0])]
                base_ids: list[str] = []
                for u, v in zip(path, path[1:]):
                    pair_id = self._available_pair(int(u), int(v))
                    if pair_id is None:
                        break
                    base_ids.append(pair_id)
                    route.append(int(v))
                if not base_ids:
                    continue
                resource_paths.append((route, base_ids, len(path) - 1))
            if not resource_paths:
                continue

            proposals: list[tuple[list[int], list[str], int]] = []
            # Always expose the farthest executable prefix on the best route.
            proposals.append(resource_paths[0])
            # Local algorithms receive one-hop alternatives through the same
            # immutable candidate catalogue; they never rewrite a PPO plan.
            for route, base_ids, full_hops in resource_paths:
                proposals.append((route[:2], base_ids[:1], full_hops))
            # Keep a medium-granularity option for long paths when space allows.
            best_route, best_ids, best_full_hops = resource_paths[0]
            half = max(1, len(best_ids) // 2)
            proposals.append((best_route[:half + 1], best_ids[:half], best_full_hops))
            proposals.extend(resource_paths[1:])

            seen: set[tuple[str, ...]] = set()
            slot = 0
            for route, base_ids, full_hops in proposals:
                signature = tuple(base_ids)
                if signature in seen:
                    continue
                seen.add(signature)
                input_ids = ([state.carried_pair_id] if state.carried_pair_id else []) + base_ids
                actions = self._compile_actions(
                    request.id, tuple(route), input_ids, state.carried_pair_id is not None
                )
                plan_id = f"t{self.time}:{request.id}:{slot}"
                candidates[plan_id] = PlanDescriptor(
                    plan_id=plan_id,
                    request_id=request.id,
                    route_nodes=tuple(route),
                    reached_node=route[-1],
                    elementary_pair_ids=tuple(base_ids),
                    swap_actions=actions,
                    duration=max(1, len(actions)),
                    remaining_hops=max(0, full_hops - (len(route) - 1)),
                    completes_request=route[-1] == request.destination,
                )
                slot += 1
                if slot >= self.candidate_count:
                    break
        return candidates

    def snapshot(self) -> PlanningSnapshot:
        self._prepare_slot()
        request_rows = []
        for state in self.requests.values():
            try:
                shortest = nx.shortest_path(self.graph, state.frontier, state.spec.destination)
                shortest_next_hop = shortest[1] if len(shortest) > 1 else state.frontier
                shortest_hops = len(shortest) - 1
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                shortest_next_hop = None
                shortest_hops = self.spec.horizon
            request_rows.append({
                "id": state.spec.id,
                "source": state.spec.source,
                "destination": state.spec.destination,
                "arrival": state.spec.arrival,
                "deadline": state.spec.deadline,
                "frontier": state.frontier,
                "initial_hops": self._initial_hops[state.spec.id],
                "shortest_next_hop": shortest_next_hop,
                "shortest_hops": shortest_hops,
                "carried_pair_id": state.carried_pair_id,
                "carried_pair_ids": self._carried_ids(state),
                "delivered_pairs": state.delivered_pairs,
                "demand_pairs": state.spec.demand_pairs,
                "completed_at": state.completed_at,
                "expired_at": state.expired_at,
            })
        request_rows = tuple(request_rows)
        resources = tuple({
            "pair_id": pair.pair_id,
            "left": pair.left,
            "right": pair.right,
            "fidelity": pair.fidelity,
            "born": pair.born,
            "owner_request": pair.owner_request,
        } for pair in sorted(self.backend.pairs.values(), key=lambda item: item.pair_id))
        candidates = tuple(self._candidates.values())
        link_capacities = tuple({
            "left": min(u, v),
            "right": max(u, v),
            "max_width": self.spec.physical.max_width,
            "generation_probability": self.spec.physical.generation_probability,
        } for u, v in self.spec.edges)
        return PlanningSnapshot(
            time=self.time,
            requests=request_rows,
            resources=resources,
            candidates=candidates,
            action_mask=tuple(True for _ in candidates),
            metrics=self.metrics(),
            phase=self.phase,
            link_capacities=link_capacities,
        )

    def _request_remaining_hops(self, state: RequestState) -> float:
        if not state.active:
            return 0.0
        return float(self._distances.get(state.frontier, {}).get(
            state.spec.destination, self.spec.horizon
        ))

    def remaining_hops(self) -> float:
        """Total shortest-path distance left for all active requests."""
        return float(sum(
            self._request_remaining_hops(state)
            for state in self.requests.values()
        ))

    def progress_potential(self) -> float:
        """Graph-derived progress, with expired requests settled back to zero."""
        potential = 0.0
        for request_id, state in self.requests.items():
            initial = self._initial_hops[request_id]
            if state.completed_at is not None:
                potential += initial
            elif state.expired_at is None:
                potential += initial - self._request_remaining_hops(state)
        return float(potential)

    @staticmethod
    def _input_ids(plan: PlanDescriptor, carried_pair_id: str | None) -> set[str]:
        values = set(plan.elementary_pair_ids)
        if carried_pair_id is not None:
            values.add(carried_pair_id)
        return values

    def commit(self, plan_ids: Iterable[str]) -> dict[str, object]:
        if self.phase == "allocate":
            return self._commit_allocations(plan_ids)
        return self._commit_execution(plan_ids)

    def _commit_allocations(self, plan_ids: Iterable[str]) -> dict[str, object]:
        self._prepare_slot()
        plans = [self._candidates[plan_id] for plan_id in plan_ids]
        request_ids: set[str] = set()
        claims: set[ResourceClaim] = set()
        node_claims: dict[int, int] = {}
        edge_claims: dict[tuple[int, int], int] = {}
        for plan in plans:
            if plan.kind != "allocation":
                raise ValueError("allocation phase accepts allocation plans only")
            if plan.request_id in request_ids:
                raise ValueError("a batch may allocate at most one plan per request")
            request_ids.add(plan.request_id)
            overlap = claims & set(plan.claims)
            if overlap:
                raise ValueError(f"allocation claims overlap: {overlap}")
            claims.update(plan.claims)
            for claim in plan.claims:
                edge_claims[claim.endpoints] = edge_claims.get(claim.endpoints, 0) + 1
                for node in claim.endpoints:
                    node_claims[node] = node_claims.get(node, 0) + 1
        capacity = self.spec.physical.node_memory_capacity
        for edge, count in edge_claims.items():
            occupied = sum(
                set(pair.endpoints) == set(edge)
                for pair in self.backend.pairs.values()
            )
            if occupied + count > self.spec.physical.memory_capacity:
                raise ValueError(f"edge {edge} memory capacity exceeded")
        if capacity is not None:
            for node, count in node_claims.items():
                if self.backend.node_occupancy(node) + count > capacity:
                    raise ValueError(f"node {node} memory capacity exceeded")

        generated = 0
        self.allocation_claims += sum(len(plan.claims) for plan in plans)
        self._active_allocation_ids = set()
        self._allocated_pair_ids = set()
        for plan in sorted(plans, key=lambda item: item.plan_id):
            allocation_id = plan.allocation_id or plan.plan_id
            self._active_allocation_ids.add(allocation_id)
            outcomes = self.backend.generate_claimed_pairs(plan.claims, allocation_id)
            for pair_id in outcomes.values():
                if pair_id is not None:
                    self._allocated_pair_ids.add(pair_id)
                    generated += 1
        self.generated_eprs += generated
        self.allocation_successes += generated
        self.phase = "recover"
        self._prepared_time = None
        self._candidates = {}
        self._prepare_slot()
        return {
            "time": self.time,
            "duration": 0.0,
            "completed_now": 0,
            "failed_now": 0,
            "expired_now": 0,
            "generated_now": generated,
            "phase": "allocate",
            "phase_after": "recover",
            "metrics": self.metrics(),
        }

    def _commit_execution(self, plan_ids: Iterable[str]) -> dict[str, object]:
        """Validate, execute, settle, and advance exactly one shared slot."""
        self._prepare_slot()
        remaining_before = self.remaining_hops()
        potential_before = self.progress_potential()
        plans = [self._candidates[plan_id] for plan_id in plan_ids]
        request_ids: set[str] = set()
        pair_ids: set[str] = set()
        for plan in plans:
            if self.phase == "recover" and plan.kind != "recovery":
                raise ValueError("recovery phase accepts recovery plans only")
            if plan.request_id in request_ids:
                raise ValueError("a batch may select at most one plan per request")
            request_ids.add(plan.request_id)
            inputs = set(plan.elementary_pair_ids)
            if not plan.lanes:
                inputs.update(self._carried_ids(self.requests[plan.request_id]))
            if inputs & pair_ids:
                raise ValueError("two selected plans consume the same EPR pair")
            if not inputs <= self.backend.pairs.keys():
                raise ValueError("selected plan references a missing EPR pair")
            pair_ids.update(inputs)

        completed_now = 0
        failed_now = 0
        successful_now = 0
        partial_successes_now = 0
        progress_hops = 0.0
        positive_progress_hops = 0.0
        lost_progress_hops = 0.0
        batch_duration = max((plan.duration for plan in plans), default=1)
        batch_start = self.time
        delivered_now = 0
        for plan in sorted(plans, key=lambda item: item.plan_id):
            state = self.requests[plan.request_id]
            request_remaining_before = self._request_remaining_hops(state)
            old_carried = self._carried_ids(state)
            if plan.lanes:
                lane_results = self.backend.execute_lanes(plan.lanes)
                self.swaps += sum(result.attempted_swaps for result in lane_results)
                output_ids = tuple(
                    result.output_pair_id for result in lane_results
                    if result.output_pair_id is not None
                )
            else:
                inputs = list(self._input_ids(plan, state.carried_pair_id))
                if plan.swap_actions:
                    output_id = self.backend.execute_actions(plan.swap_actions)
                    self.swaps += len(plan.swap_actions)
                else:
                    output_id = inputs[0] if inputs else None
                output_ids = () if output_id is None else (output_id,)
            if not output_ids:
                state.frontier = state.spec.source
                self._set_carried(state, ())
                self.failed_plans += 1
                failed_now += 1
                delta = request_remaining_before - self._request_remaining_hops(state)
                progress_hops += delta
                positive_progress_hops += max(delta, 0.0)
                lost_progress_hops += max(-delta, 0.0)
                continue
            successful_now += 1
            if not plan.completes_request:
                partial_successes_now += 1
            for output_id in output_ids:
                output = self.backend.pairs[output_id]
                output.owner_request = state.spec.id
                output.reserved_by = None
            state.frontier = plan.reached_node
            self._set_carried(state, output_ids)
            finish_time = batch_start + plan.duration
            deadline = state.spec.deadline
            if plan.completes_request and (deadline is None or finish_time <= deadline):
                delivered = len(output_ids)
                state.delivered_pairs += delivered
                self.delivered_pairs += delivered
                delivered_now += delivered
                for output_id in output_ids:
                    self.backend.discard_pair(output_id)
                self._set_carried(state, ())
                state.frontier = state.spec.source
                if state.delivered_pairs >= state.spec.demand_pairs:
                    state.completed_at = finish_time
                    completed_now += 1
            for pair_id in old_carried:
                if pair_id in self.backend.pairs and pair_id not in output_ids:
                    self.backend.discard_pair(pair_id)
            delta = request_remaining_before - self._request_remaining_hops(state)
            progress_hops += delta
            positive_progress_hops += max(delta, 0.0)
            lost_progress_hops += max(-delta, 0.0)

        remaining_after_plans = self.remaining_hops()
        self.successful_plans += successful_now
        self.partial_plan_successes += partial_successes_now
        self.progress_hops += progress_hops
        self.positive_progress_hops += positive_progress_hops
        self.lost_progress_hops += lost_progress_hops

        if self.phase == "recover":
            self.recovery_attempts += len(plans)
            self.recovery_successes += successful_now
            for pair_id in tuple(self._allocated_pair_ids):
                pair = self.backend.pairs.get(pair_id)
                if pair is not None and pair.owner_request is None:
                    self.backend.discard_pair(pair_id)
                    self.released_surplus_pairs += 1
            self._allocated_pair_ids.clear()
            self._active_allocation_ids.clear()

        for subslot in range(batch_duration):
            self.backend.advance_slot()
            if self.phase != "recover" and subslot + 1 < batch_duration:
                self.generated_eprs += len(self.backend.generate_elementary_pairs())
        expired_now = 0
        for state in self.requests.values():
            if not state.active:
                continue
            deadline = state.spec.deadline
            if deadline is not None and self.time >= deadline:
                state.expired_at = self.time
                for pair_id in self._carried_ids(state):
                    self.backend.discard_pair(pair_id)
                self._set_carried(state, ())
                expired_now += 1
        phase_before = self.phase
        if self.phase == "recover":
            self.phase = "allocate"
        self._prepared_time = None
        self._candidates = {}
        potential_after = self.progress_potential()
        return {
            "time": self.time,
            "duration": batch_duration,
            "completed_now": completed_now,
            "failed_now": failed_now,
            "expired_now": expired_now,
            "delivered_pairs_now": delivered_now,
            "core_phase": phase_before,
            "phase_after": self.phase,
            "successful_plans_now": successful_now,
            "partial_plan_successes_now": partial_successes_now,
            "progress_hops_now": progress_hops,
            "positive_progress_hops_now": positive_progress_hops,
            "lost_progress_hops_now": lost_progress_hops,
            "remaining_hops_before": remaining_before,
            "remaining_hops_after_plans": remaining_after_plans,
            "remaining_hops_after": self.remaining_hops(),
            "progress_potential_before": potential_before,
            "progress_potential_after": potential_after,
            "progress_potential_delta": potential_after - potential_before,
            "metrics": self.metrics(),
        }

    def metrics(self) -> dict[str, float]:
        total = max(len(self.requests), 1)
        completed = sum(state.completed_at is not None for state in self.requests.values())
        expired = sum(state.expired_at is not None for state in self.requests.values())
        delays = [
            state.completed_at - state.spec.arrival
            for state in self.requests.values() if state.completed_at is not None
        ]
        return {
            "completion_rate": completed / total,
            "timeout_rate": expired / total,
            "mean_delay": sum(delays) / len(delays) if delays else 0.0,
            "generated_eprs": float(self.generated_eprs),
            "swaps": float(self.swaps),
            "failed_plans": float(self.failed_plans),
            "successful_plans": float(self.successful_plans),
            "partial_plan_successes": float(self.partial_plan_successes),
            "progress_hops": float(self.progress_hops),
            "positive_progress_hops": float(self.positive_progress_hops),
            "lost_progress_hops": float(self.lost_progress_hops),
            "remaining_hops": self.remaining_hops(),
            "progress_potential": self.progress_potential(),
            "delivered_pairs": float(self.delivered_pairs),
            "pair_throughput": self.delivered_pairs / max(self.time, 1),
            "recovery_attempts": float(self.recovery_attempts),
            "recovery_successes": float(self.recovery_successes),
            "recovery_success_rate": (
                self.recovery_successes / self.recovery_attempts
                if self.recovery_attempts else 0.0
            ),
            "allocation_claims": float(self.allocation_claims),
            "allocation_successes": float(self.allocation_successes),
            "allocation_success_rate": (
                self.allocation_successes / self.allocation_claims
                if self.allocation_claims else 0.0
            ),
            "released_surplus_pairs": float(self.released_surplus_pairs),
            "epr_delivery_utilization": (
                self.delivered_pairs / self.generated_eprs
                if self.generated_eprs else 0.0
            ),
            "makespan": float(self.time),
        }

    @property
    def done(self) -> bool:
        settled = sum(not state.active for state in self.requests.values())
        return settled == len(self.requests) or self.time >= self.spec.horizon
