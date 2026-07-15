"""Algorithm-independent episode control on the SeQUeNCe resource kernel."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Iterable

import networkx as nx

from .planner_api import PlanDescriptor, PlanningSnapshot, SwapAction
from .sequence_backend import SequenceBackend
from .spec import EpisodeSpec, RequestSpec


@dataclass
class RequestState:
    spec: RequestSpec
    frontier: int
    carried_pair_id: str | None = None
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
        self.backend = SequenceBackend(spec)
        self.requests = {
            request.id: RequestState(request, request.source)
            for request in spec.requests
        }
        self.generated_eprs = 0
        self.swaps = 0
        self.failed_plans = 0
        self._prepared_time: int | None = None
        self._candidates: dict[str, PlanDescriptor] = {}

    @property
    def time(self) -> int:
        return self.backend.time

    def _prepare_slot(self) -> None:
        if self._prepared_time == self.time:
            return
        for state in self.requests.values():
            if (state.carried_pair_id is not None
                    and state.carried_pair_id not in self.backend.pairs):
                state.carried_pair_id = None
                state.frontier = state.spec.source
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
                "shortest_next_hop": shortest_next_hop,
                "shortest_hops": shortest_hops,
                "carried_pair_id": state.carried_pair_id,
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
        return PlanningSnapshot(
            time=self.time,
            requests=request_rows,
            resources=resources,
            candidates=candidates,
            action_mask=tuple(True for _ in candidates),
            metrics=self.metrics(),
        )

    @staticmethod
    def _input_ids(plan: PlanDescriptor, carried_pair_id: str | None) -> set[str]:
        values = set(plan.elementary_pair_ids)
        if carried_pair_id is not None:
            values.add(carried_pair_id)
        return values

    def commit(self, plan_ids: Iterable[str]) -> dict[str, object]:
        """Validate, execute, settle, and advance exactly one shared slot."""
        self._prepare_slot()
        plans = [self._candidates[plan_id] for plan_id in plan_ids]
        request_ids: set[str] = set()
        pair_ids: set[str] = set()
        for plan in plans:
            if plan.request_id in request_ids:
                raise ValueError("a batch may select at most one plan per request")
            request_ids.add(plan.request_id)
            inputs = self._input_ids(plan, self.requests[plan.request_id].carried_pair_id)
            if inputs & pair_ids:
                raise ValueError("two selected plans consume the same EPR pair")
            if not inputs <= self.backend.pairs.keys():
                raise ValueError("selected plan references a missing EPR pair")
            pair_ids.update(inputs)

        completed_now = 0
        failed_now = 0
        batch_duration = max((plan.duration for plan in plans), default=1)
        batch_start = self.time
        for plan in sorted(plans, key=lambda item: item.plan_id):
            state = self.requests[plan.request_id]
            inputs = list(self._input_ids(plan, state.carried_pair_id))
            if plan.swap_actions:
                output_id = self.backend.execute_actions(plan.swap_actions)
                self.swaps += len(plan.swap_actions)
            else:
                output_id = inputs[0] if inputs else None
            if output_id is None:
                state.frontier = state.spec.source
                state.carried_pair_id = None
                self.failed_plans += 1
                failed_now += 1
                continue
            output = self.backend.pairs[output_id]
            output.owner_request = state.spec.id
            state.frontier = plan.reached_node
            state.carried_pair_id = output_id
            finish_time = batch_start + plan.duration
            deadline = state.spec.deadline
            if plan.completes_request and (deadline is None or finish_time <= deadline):
                state.completed_at = finish_time
                state.carried_pair_id = None
                self.backend.pairs.pop(output_id, None)
                completed_now += 1

        for subslot in range(batch_duration):
            self.backend.advance_slot()
            if subslot + 1 < batch_duration:
                self.generated_eprs += len(self.backend.generate_elementary_pairs())
        expired_now = 0
        for state in self.requests.values():
            if not state.active:
                continue
            deadline = state.spec.deadline
            if deadline is not None and self.time >= deadline:
                state.expired_at = self.time
                if state.carried_pair_id is not None:
                    self.backend.pairs.pop(state.carried_pair_id, None)
                    state.carried_pair_id = None
                expired_now += 1
        self._prepared_time = None
        self._candidates = {}
        return {
            "time": self.time,
            "duration": batch_duration,
            "completed_now": completed_now,
            "failed_now": failed_now,
            "expired_now": expired_now,
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
            "makespan": float(self.time),
        }

    @property
    def done(self) -> bool:
        settled = sum(not state.active for state in self.requests.values())
        return settled == len(self.requests) or self.time >= self.spec.horizon
