"""Planning-only baselines for the shared environment."""

from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .planner_api import PlanDescriptor, PlanningSnapshot


def _pair_ids(plan: PlanDescriptor) -> set[str]:
    values = set(plan.elementary_pair_ids)
    for action in plan.swap_actions:
        if not action.left_pair_id.startswith("@"):
            values.add(action.left_pair_id)
        if not action.right_pair_id.startswith("@"):
            values.add(action.right_pair_id)
    return values


def _pack(plans: list[PlanDescriptor]) -> tuple[str, ...]:
    selected: list[str] = []
    requests: set[str] = set()
    pairs: set[str] = set()
    claims: set[tuple[tuple[int, int], int]] = set()
    for plan in plans:
        inputs = _pair_ids(plan)
        plan_claims = {(claim.endpoints, claim.lane) for claim in plan.claims}
        if plan.request_id in requests or inputs & pairs or plan_claims & claims:
            continue
        selected.append(plan.plan_id)
        requests.add(plan.request_id)
        pairs.update(inputs)
        claims.update(plan_claims)
    return tuple(selected)


def _require_exact_milp_optimum(result) -> None:
    """Reject a tolerance-optimal result before calling it optimal."""

    gap = getattr(result, "mip_gap", None)
    if gap is not None:
        gap_value = float(gap)
        if not np.isfinite(gap_value) or gap_value != 0.0:
            raise RuntimeError(
                "one-slot optimal MILP is not proven optimal: "
                f"mip_gap={gap_value}"
            )

    primal = getattr(result, "fun", None)
    dual = getattr(result, "mip_dual_bound", None)
    if primal is not None and dual is not None:
        primal_value = float(primal)
        dual_value = float(dual)
        if (
            not np.isfinite(primal_value)
            or not np.isfinite(dual_value)
            or not np.isclose(
                primal_value, dual_value, rtol=0.0, atol=1e-8
            )
        ):
            raise RuntimeError(
                "one-slot optimal MILP lacks a closed objective bound: "
                f"primal={primal_value}, dual={dual_value}"
            )


class GreedyPlanner:
    def reset(self, episode_seed: int) -> None:
        del episode_seed

    def select(self, snapshot: PlanningSnapshot) -> tuple[str, ...]:
        plans = sorted(
            snapshot.candidates,
            key=lambda plan: (
                not plan.completes_request,
                -len(plan.route_nodes),
                plan.duration,
                plan.plan_id,
            ),
        )
        return _pack(plans)


class RandomPlanner:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = random.Random(seed)

    def reset(self, episode_seed: int) -> None:
        self.rng = random.Random(self.seed ^ episode_seed)

    def select(self, snapshot: PlanningSnapshot) -> tuple[str, ...]:
        plans = list(snapshot.candidates)
        self.rng.shuffle(plans)
        return _pack(plans)


class OptimalPlanner:
    """Exact one-slot batch oracle over the public candidate catalogue.

    The planner has exactly the same authority as every other planner: it may
    only return candidate plan IDs from an immutable ``PlanningSnapshot``.
    It does not generate EPRs, execute swaps, advance time, or inspect future
    physical outcomes.

    For the ordinary execution phase the MILP optimizes, lexicographically:

    1. requests that can finish in the current committed slot;
    2. aggregate shortest-path progress;
    3. lower physical resource/work cost.

    With deterministic swapping this is an exact realized one-slot optimum.
    With stochastic swapping it remains an exact optimum of the visible
    candidate abstraction, not a clairvoyant episode-level optimum.
    """

    def reset(self, episode_seed: int) -> None:
        del episode_seed

    @staticmethod
    def _completion_credit(
        plan: PlanDescriptor,
        request: dict[str, object],
        snapshot: PlanningSnapshot,
    ) -> int:
        if snapshot.phase == "allocate" or not plan.completes_request:
            return 0
        deadline = request.get("deadline")
        if deadline is not None and snapshot.time + max(plan.duration, 1) > int(deadline):
            return 0
        delivered = int(request.get("delivered_pairs", 0))
        demand = int(request.get("demand_pairs", 1))
        produced = max(1, int(plan.width))
        return int(delivered + produced >= demand)

    @staticmethod
    def _progress_credit(
        plan: PlanDescriptor,
        request: dict[str, object],
        snapshot: PlanningSnapshot,
    ) -> int:
        if snapshot.phase == "allocate":
            # Allocation itself has zero physical duration in the shared core.
            # Quantized EXT supplies a deterministic secondary objective so the
            # oracle can also be used when the public width/recovery phases are
            # enabled, without claiming an immediate request completion.
            return max(0, int(round(plan.expected_throughput * 1_000_000)))
        current = int(request.get("shortest_hops", plan.remaining_hops))
        return max(0, current - int(plan.remaining_hops))

    @staticmethod
    def _work_cost(plan: PlanDescriptor) -> int:
        return max(
            1,
            len(_pair_ids(plan))
            + len(plan.swap_actions)
            + len(plan.claims)
            + int(plan.duration)
            + int(plan.memory_cost),
        )

    def select(self, snapshot: PlanningSnapshot) -> tuple[str, ...]:
        candidates = tuple(
            plan
            for index, plan in enumerate(snapshot.candidates)
            if index >= len(snapshot.action_mask) or snapshot.action_mask[index]
        )
        if not candidates:
            return ()

        request_rows = {str(row["id"]): row for row in snapshot.requests}
        missing = {
            plan.request_id for plan in candidates
            if plan.request_id not in request_rows
        }
        if missing:
            raise ValueError(f"candidate requests missing from snapshot: {sorted(missing)}")

        completion = np.asarray([
            self._completion_credit(plan, request_rows[plan.request_id], snapshot)
            for plan in candidates
        ], dtype=np.int64)
        progress = np.asarray([
            self._progress_credit(plan, request_rows[plan.request_id], snapshot)
            for plan in candidates
        ], dtype=np.int64)
        work = np.asarray([
            self._work_cost(plan) for plan in candidates
        ], dtype=np.int64)

        request_ids = tuple(dict.fromkeys(plan.request_id for plan in candidates))
        max_progress = sum(
            max(
                (int(progress[index]) for index, plan in enumerate(candidates)
                 if plan.request_id == request_id),
                default=0,
            )
            for request_id in request_ids
        )
        max_work = sum(
            max(
                (int(work[index]) for index, plan in enumerate(candidates)
                 if plan.request_id == request_id),
                default=0,
            )
            for request_id in request_ids
        )
        progress_weight = max_work + 1
        completion_weight = max_progress * progress_weight + max_work + 1
        objective = (
            -completion.astype(float) * completion_weight
            -progress.astype(float) * progress_weight
            +work.astype(float)
        )

        rows: list[np.ndarray] = []
        upper: list[float] = []

        # A request may execute at most one complete candidate plan.
        for request_id in request_ids:
            row = np.zeros(len(candidates), dtype=float)
            for index, plan in enumerate(candidates):
                if plan.request_id == request_id:
                    row[index] = 1.0
            rows.append(row)
            upper.append(1.0)

        # Existing EPRs are exclusive consumable resources.
        pair_to_indices: dict[str, list[int]] = defaultdict(list)
        for index, plan in enumerate(candidates):
            for pair_id in _pair_ids(plan):
                pair_to_indices[pair_id].append(index)
        for indices in pair_to_indices.values():
            if len(indices) < 2:
                continue
            row = np.zeros(len(candidates), dtype=float)
            row[indices] = 1.0
            rows.append(row)
            upper.append(1.0)

        # Width-allocation lanes are likewise exclusive in the public API.
        claim_to_indices: dict[tuple[tuple[int, int], int], list[int]] = defaultdict(list)
        for index, plan in enumerate(candidates):
            for claim in plan.claims:
                claim_to_indices[(claim.endpoints, claim.lane)].append(index)
        for indices in claim_to_indices.values():
            if len(indices) < 2:
                continue
            row = np.zeros(len(candidates), dtype=float)
            row[indices] = 1.0
            rows.append(row)
            upper.append(1.0)

        constraints = LinearConstraint(
            np.vstack(rows),
            np.full(len(rows), -np.inf, dtype=float),
            np.asarray(upper, dtype=float),
        )
        result = milp(
            c=objective,
            integrality=np.ones(len(candidates), dtype=int),
            bounds=Bounds(
                np.zeros(len(candidates), dtype=float),
                np.ones(len(candidates), dtype=float),
            ),
            constraints=constraints,
            options={"disp": False, "mip_rel_gap": 0.0},
        )
        if not result.success or result.x is None:
            raise RuntimeError(f"one-slot optimal MILP failed: {result.message}")
        _require_exact_milp_optimum(result)
        return tuple(
            plan.plan_id
            for plan, selected in zip(candidates, result.x)
            if selected > 0.5
        )


class QCASTPlanner:
    """Planning-only QCAST port over the public width/recovery catalogue."""

    def reset(self, episode_seed: int) -> None:
        del episode_seed

    def select(self, snapshot: PlanningSnapshot) -> tuple[str, ...]:
        if snapshot.phase == "allocate":
            complete = [plan for plan in snapshot.candidates if plan.completes_request]
            catalogue = complete or list(snapshot.candidates)
            plans = sorted(
                catalogue,
                key=lambda plan: (
                    -plan.expected_throughput,
                    plan.memory_cost,
                    plan.remaining_hops,
                    plan.plan_id,
                ),
            )
        else:
            plans = sorted(
                snapshot.candidates,
                key=lambda plan: (
                    not plan.completes_request,
                    -plan.expected_throughput,
                    -plan.width,
                    plan.remaining_hops,
                    plan.plan_id,
                ),
            )
        return _pack(plans)


class QDDCAPlanner:
    """Q-DDCA local scoring port using only the immutable shared snapshot."""

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
        p = (successes + 0.5) / (len(history) + 0.5)
        attempts = min(self.try_count[request_id], self.max_try)
        failure = (1.0 - p) ** max(0, self.max_try - attempts)
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
            return _pack(local)
        # The official prototype sets queryTime=0.5/M. In the slotted common
        # environment this becomes a deterministic attempt interval: larger M
        # permits more routing attempts within the same physical horizon.
        attempt_interval = max(1, round(5 / self.max_try))
        if snapshot.time % attempt_interval:
            return ()
        rows = {str(row["id"]): row for row in snapshot.requests}
        local = [plan for plan in snapshot.candidates if len(plan.route_nodes) == 2]
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
                if plan.reached_node == rows[plan.request_id].get("shortest_next_hop")
            ]
            local.sort(key=lambda plan: (
                plan.remaining_hops,
                plan.request_id,
                plan.reached_node,
                plan.plan_id,
            ))
        selected = _pack(local)
        by_id = {plan.plan_id: plan for plan in local}
        for plan_id in selected:
            plan = by_id[plan_id]
            self.pending[plan.request_id] = plan.reached_node
            row = rows[plan.request_id]
            self.route_history.setdefault(plan.request_id, [int(row["source"])])
        return selected
