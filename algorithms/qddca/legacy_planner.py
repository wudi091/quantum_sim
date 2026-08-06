"""Q-DDCA local-scoring adapter for the legacy shared environment."""

from __future__ import annotations

from collections import defaultdict, deque
import random

from qnet_core.planner_api import PlanDescriptor, PlanningSnapshot

from algorithms._legacy_shared import pack


class QDDCAPlanner:
    def __init__(
        self,
        max_try: int = 5,
        history_length: int = 10,
        allow_reroute: bool = True,
        seed: int = 0,
    ):
        self.max_try = max(1, int(max_try))
        self.history_length = max(1, int(history_length))
        self.allow_reroute = bool(allow_reroute)
        self.seed = seed
        self.rng = random.Random(seed)
        self.history: dict[tuple[str, int], deque[bool]] = defaultdict(
            lambda: deque(maxlen=self.history_length)
        )
        self.try_count: dict[str, int] = defaultdict(int)
        self.route_history: dict[str, list[int]] = {}
        self.pending: dict[str, int] = {}

    def reset(self, episode_seed: int) -> None:
        self.rng = random.Random(self.seed ^ episode_seed)
        self.history.clear()
        self.try_count.clear()
        self.route_history.clear()
        self.pending.clear()

    def _settle(self, snapshot: PlanningSnapshot) -> None:
        requests = {str(row["id"]): row for row in snapshot.requests}
        for request_id, expected in list(self.pending.items()):
            row = requests.get(request_id)
            success = row is not None and int(row["frontier"]) == expected
            self.history[request_id, expected].append(success)
            if success:
                self.try_count[request_id] = 0
                self.route_history.setdefault(request_id, []).append(expected)
            else:
                self.try_count[request_id] += 1
            self.pending.pop(request_id, None)

    def _score(self, plan: PlanDescriptor, row: dict[str, object]) -> float:
        request_id = plan.request_id
        neighbor = plan.reached_node
        original_distance = max(1, int(row.get("shortest_hops", 1)))
        remaining = 1 + plan.remaining_hops
        history = self.history[request_id, neighbor]
        successes = sum(bool(value) for value in history)
        probability = (successes + 0.5) / (len(history) + 0.5)
        attempts = min(self.try_count[request_id], self.max_try)
        failure = (1.0 - probability) ** max(0, self.max_try - attempts)
        metric_drop = original_distance * 2
        return (1.0 - failure) * remaining + failure * metric_drop

    def select(self, snapshot: PlanningSnapshot) -> tuple[str, ...]:
        if snapshot.phase == "allocate":
            self._settle(snapshot)
            local = [
                plan for plan in snapshot.candidates
                if len(plan.route_nodes) == 2 and plan.width == 1
            ]
            local.sort(key=lambda plan: (
                plan.remaining_hops,
                plan.request_id,
                plan.reached_node,
                plan.plan_id,
            ))
            return pack(local)

        attempt_interval = max(1, round(5 / self.max_try))
        if snapshot.time % attempt_interval:
            return ()
        rows = {str(row["id"]): row for row in snapshot.requests}
        local = [
            plan for plan in snapshot.candidates if len(plan.route_nodes) == 2
        ]
        for plan in local:
            self.history[plan.request_id, plan.reached_node].append(True)
        if self.allow_reroute:
            local.sort(key=lambda plan: (
                self._score(plan, rows[plan.request_id]),
                plan.request_id,
                plan.reached_node,
                plan.plan_id,
            ))
        else:
            local = [
                plan for plan in local
                if plan.reached_node
                == rows[plan.request_id].get("shortest_next_hop")
            ]
            local.sort(key=lambda plan: (
                plan.remaining_hops,
                plan.request_id,
                plan.reached_node,
                plan.plan_id,
            ))
        selected = pack(local)
        by_id = {plan.plan_id: plan for plan in local}
        for plan_id in selected:
            plan = by_id[plan_id]
            self.pending[plan.request_id] = plan.reached_node
            row = rows[plan.request_id]
            self.route_history.setdefault(
                plan.request_id,
                [int(row["source"])],
            )
        return selected
