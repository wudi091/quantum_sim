"""Planning-only implementation of the Q-DDCA routing decision.

The upstream ``QDDCA`` directory contains a SimQN prototype.  This adapter
keeps its routing state machine and equations, while the actual resource
request, entanglement generation, swapping, memory lifetime, and time advance
remain in :class:`qnet_core.env.SharedRoutingEnv` and SeQUeNCe.
"""

from __future__ import annotations

from collections import defaultdict, deque
import math
import random

from qnet_core.planner_api import PlanDescriptor, PlanningSnapshot


class QDDCAPlanner:
    """Q-DDCA Algorithm 2/3 over an immutable planning snapshot."""

    def __init__(
        self,
        max_try: int = 10,
        history_length: int = 10,
        epsilon: float = 0.5,
        allow_reroute: bool = True,
        seed: int = 0,
    ):
        self.max_try = max(1, int(max_try))
        self.history_length = max(1, int(history_length))
        self.epsilon = float(epsilon)
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.allow_reroute = bool(allow_reroute)
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.history: dict[tuple[int, int], deque[bool]] = defaultdict(
            lambda: deque(maxlen=self.history_length)
        )
        self.attempts: dict[str, int] = defaultdict(int)
        self.route_history: dict[str, list[int]] = {}
        self.pending: dict[str, tuple[int, int, int]] = {}
        self._last_feedback_id = 0

    def reset(self, episode_seed: int) -> None:
        self.rng = random.Random(self.seed ^ int(episode_seed))
        self.history.clear()
        self.attempts.clear()
        self.route_history.clear()
        self.pending.clear()
        self._last_feedback_id = 0

    def _acceptance_rate(self, current: int, neighbor: int) -> float:
        values = self.history[current, neighbor]
        accepted = sum(bool(value) for value in values)
        return (accepted + self.epsilon) / (len(values) + self.epsilon)

    def _consume_feedback(self, snapshot: PlanningSnapshot) -> None:
        """Update only from simulator-neutral outcomes of our own requests."""
        for feedback in sorted(snapshot.feedback, key=lambda item: item.feedback_id):
            if feedback.feedback_id <= self._last_feedback_id:
                continue
            self._last_feedback_id = feedback.feedback_id
            current, neighbor, _ = self.pending.pop(
                feedback.plan_id, (None, feedback.reached_node, 0)
            )
            if current is None:
                current = feedback.reached_node
            if feedback.phase == "allocate":
                if feedback.reason == "drop":
                    self.attempts[feedback.request_id] = 0
                    source = next(
                        (node for node in self.route_history.get(feedback.request_id, [])),
                        feedback.reached_node,
                    )
                    self.route_history[feedback.request_id] = [source]
                    continue
                self.history[current, neighbor].append(bool(feedback.accepted))
                if feedback.accepted:
                    # A successful resource response creates one physical
                    # pair; the attempt counter resets after this hop.
                    if feedback.succeeded:
                        self.attempts[feedback.request_id] = 0
                    else:
                        self.attempts[feedback.request_id] = 0
                else:
                    self.attempts[feedback.request_id] += 1
            elif feedback.phase == "recover":
                if feedback.succeeded:
                    self.attempts[feedback.request_id] = 0
                    route = self.route_history.setdefault(
                        feedback.request_id, [current]
                    )
                    if not route or route[-1] != feedback.reached_node:
                        route.append(feedback.reached_node)
                else:
                    self.attempts[feedback.request_id] = 0
                    source = self.route_history.get(
                        feedback.request_id, [current]
                    )[0]
                    self.route_history[feedback.request_id] = [source]
            else:
                # The non-request-driven compatibility path executes a
                # selected plan directly in the physical slot.
                if feedback.succeeded:
                    route = self.route_history.setdefault(
                        feedback.request_id, [current]
                    )
                    if not route or route[-1] != feedback.reached_node:
                        route.append(feedback.reached_node)
                else:
                    source = self.route_history.get(
                        feedback.request_id, [current]
                    )[0]
                    self.route_history[feedback.request_id] = [source]

    @staticmethod
    def _rows(snapshot: PlanningSnapshot) -> dict[str, dict[str, object]]:
        return {str(row["id"]): row for row in snapshot.requests}

    def _route(self, row: dict[str, object]) -> list[int]:
        request_id = str(row["id"])
        source = int(row["source"])
        frontier = int(row["frontier"])
        route = self.route_history.setdefault(request_id, [source])
        # The environment is authoritative for a reset after drop, memory
        # expiry, or a failed swap.  Keep the planner's route prefix local.
        if not route or route[-1] != frontier:
            route = [source] if frontier == source else [source, frontier]
            self.route_history[request_id] = route
        return route

    def _candidate_set(
        self,
        row: dict[str, object],
        candidates: list[PlanDescriptor],
    ) -> list[PlanDescriptor]:
        """Algorithm 2: hard fidelity bound plus probabilistic soft bound."""
        if not candidates:
            return []
        route = self._route(row)
        current = int(row["frontier"])
        remain = int(row.get("shortest_hops", 0))
        initial = max(1, int(round(float(row.get("initial_hops", remain)))))
        fidelity_bound = max(1, int(row.get("fidelity_hop_bound", initial)))
        preceding = max(0, len(route) - 1)
        hard_limit = max(0, fidelity_bound - preceding)
        # L'_k is the shortest source-to-short-term-destination segment.  When
        # the final destination is already within the bound, it is simply the
        # source-to-destination distance used by the paper's p expression.
        short_term_length = max(1, min(initial, fidelity_bound))
        reroute_probability = max(
            0.0, min(1.0, fidelity_bound / short_term_length - 1.0)
        )
        shortest_next = row.get("shortest_next_hop")
        result: list[PlanDescriptor] = []
        for plan in candidates:
            if plan.kind == "drop" or len(plan.route_nodes) != 2:
                continue
            neighbor = int(plan.reached_node)
            if neighbor in route:
                continue
            remaining = int(plan.remaining_hops)
            if remaining > hard_limit:
                continue
            if not self.allow_reroute:
                if shortest_next is None or neighbor != int(shortest_next):
                    continue
            else:
                normal = remaining == max(0, remain - 1)
                extended = (
                    remaining == remain
                    and reroute_probability > 0.0
                    and self.rng.random() < reroute_probability
                )
                if not (normal or extended):
                    continue
            result.append(plan)
        return result

    def _utility(
        self,
        row: dict[str, object],
        plan: PlanDescriptor,
        attempt: int,
    ) -> float:
        current = int(row["frontier"])
        neighbor = int(plan.reached_node)
        distance_source = max(1, int(round(float(row.get("initial_hops", 1)))))
        remaining_attempts = max(0, self.max_try - attempt)
        q = self._acceptance_rate(current, neighbor)
        tail = (1.0 - q) ** remaining_attempts
        return (1.0 - tail) * float(plan.remaining_hops) + tail * float(
            2 * distance_source
        )

    def _select_local(self, snapshot: PlanningSnapshot) -> tuple[str, ...]:
        rows = self._rows(snapshot)
        by_request: dict[str, list[PlanDescriptor]] = defaultdict(list)
        for plan in snapshot.candidates:
            by_request[plan.request_id].append(plan)
        selected: list[str] = []
        for request_id, row in sorted(rows.items()):
            if row.get("completed_at") is not None or row.get("expired_at") is not None:
                continue
            if int(row.get("arrival", 0)) > snapshot.time:
                continue
            current = int(row["frontier"])
            route = self._route(row)
            attempt = self.attempts[request_id] + 1
            options = self._candidate_set(row, by_request[request_id])
            drop = next(
                (plan for plan in by_request[request_id] if plan.kind == "drop"),
                None,
            )
            if attempt > self.max_try:
                if drop is not None:
                    selected.append(drop.plan_id)
                    self.pending[drop.plan_id] = (current, current, attempt)
                continue
            scored = [
                (self._utility(row, plan, attempt), plan.reached_node, plan.plan_id)
                for plan in options
            ]
            if drop is not None:
                scored.append((2.0 * max(1, int(row.get("initial_hops", 1))), current, drop.plan_id))
            if not scored:
                continue
            _, _, plan_id = min(scored)
            chosen = next(plan for plan in by_request[request_id] if plan.plan_id == plan_id)
            if chosen.kind == "drop" or (
                drop is not None
                and self._utility(row, chosen, attempt)
                >= 2.0 * max(1, int(row.get("initial_hops", 1)))
            ):
                chosen = drop
            if chosen is None:
                continue
            selected.append(chosen.plan_id)
            self.pending[chosen.plan_id] = (current, int(chosen.reached_node), attempt)
            if chosen.kind != "drop":
                self.attempts[request_id] = attempt
        return tuple(selected)

    def _select_compatibility(self, snapshot: PlanningSnapshot) -> tuple[str, ...]:
        """Use the same one-hop decision when the environment pre-generates EPRs."""
        rows = self._rows(snapshot)
        local = [
            plan for plan in snapshot.candidates
            if len(plan.route_nodes) == 2 and plan.kind == "primary"
        ]
        by_request: dict[str, list[PlanDescriptor]] = defaultdict(list)
        for plan in local:
            by_request[plan.request_id].append(plan)
        selected: list[str] = []
        for request_id, plans in sorted(by_request.items()):
            row = rows[request_id]
            options = self._candidate_set(row, plans)
            if not options:
                continue
            attempt = self.attempts[request_id] + 1
            chosen = min(
                options,
                key=lambda plan: (self._utility(row, plan, attempt), plan.plan_id),
            )
            selected.append(chosen.plan_id)
            self.pending[chosen.plan_id] = (
                int(row["frontier"]), int(chosen.reached_node), attempt
            )
            self.attempts[request_id] = attempt
        return tuple(selected)

    def select(self, snapshot: PlanningSnapshot) -> tuple[str, ...]:
        self._consume_feedback(snapshot)
        if snapshot.phase in {"allocate", "recover"}:
            if snapshot.phase == "recover":
                rows = self._rows(snapshot)
                selected: list[str] = []
                seen: set[str] = set()
                for plan in sorted(snapshot.candidates, key=lambda item: item.plan_id):
                    if plan.request_id in seen:
                        continue
                    row = rows.get(plan.request_id)
                    if row is None or int(row.get("arrival", 0)) > snapshot.time:
                        continue
                    selected.append(plan.plan_id)
                    seen.add(plan.request_id)
                    self.pending[plan.plan_id] = (
                        int(row["frontier"]), int(plan.reached_node), 0
                    )
                return tuple(selected)
            return self._select_local(snapshot)
        return self._select_compatibility(snapshot)
