"""Sequential masked PPO wrapper around the common SeQUeNCe environment."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Mapping

import numpy as np

from .env import SharedRoutingEnv
from .planner_api import PlanDescriptor, PlanningSnapshot
from .scenario import ScenarioConfig, make_episode


@dataclass
class GymConfig:
    max_requests: int = 30
    max_candidates_per_request: int = 3
    max_hops: int = 50
    scenario: ScenarioConfig = ScenarioConfig()
    seed: int = 0


class SequenceGymEnv:
    global_feature_dim = 12
    request_feature_dim = 11
    candidate_feature_dim = 20

    def __init__(self, config: GymConfig | None = None):
        self.config = config or GymConfig()
        self.stop_action = self.config.max_requests * self.config.max_candidates_per_request
        self.action_size = self.stop_action + 1
        self._next_seed = self.config.seed

    def set_curriculum(self, stage: object) -> None:
        request_count = int(getattr(stage, "max_requests"))
        min_hops = int(getattr(stage, "min_hops"))
        max_hops = int(getattr(stage, "max_hops"))
        if request_count > self.config.max_requests or max_hops > self.config.max_hops:
            raise ValueError("curriculum exceeds fixed SequenceGymEnv tensor bounds")
        self.config.scenario = replace(
            self.config.scenario,
            request_count=request_count,
            min_hops=min_hops,
            max_hops=max_hops,
        )
        self._next_seed = self.config.seed

    def reset(self, seed: int | None = None, options: Mapping[str, object] | None = None):
        del options
        episode_seed = self._next_seed if seed is None else int(seed)
        self._next_seed = episode_seed + 1
        self.core = SharedRoutingEnv(
            make_episode(self.config.scenario, episode_seed),
            candidate_count=self.config.max_candidates_per_request,
        )
        self.selected: list[str] = []
        self.selected_requests: set[str] = set()
        self.selected_pairs: set[str] = set()
        self.snapshot = self.core.snapshot()
        self.slots = self._slot_candidates(self.snapshot)
        return self.observe(), self._info(phase="reset")

    def _slot_candidates(self, snapshot: PlanningSnapshot) -> list[PlanDescriptor | None]:
        slots: list[PlanDescriptor | None] = [None] * self.stop_action
        request_index = {
            str(row["id"]): index for index, row in enumerate(snapshot.requests)
        }
        counts: dict[str, int] = {}
        for plan in snapshot.candidates:
            index = request_index[plan.request_id]
            offset = counts.get(plan.request_id, 0)
            if index >= self.config.max_requests or offset >= self.config.max_candidates_per_request:
                continue
            slots[index * self.config.max_candidates_per_request + offset] = plan
            counts[plan.request_id] = offset + 1
        return slots

    @staticmethod
    def _pairs(plan: PlanDescriptor) -> set[str]:
        values = set(plan.elementary_pair_ids)
        for action in plan.swap_actions:
            if not action.left_pair_id.startswith("@"):
                values.add(action.left_pair_id)
            if not action.right_pair_id.startswith("@"):
                values.add(action.right_pair_id)
        return values

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_size, dtype=bool)
        for action, plan in enumerate(self.slots):
            if plan is None or plan.request_id in self.selected_requests:
                continue
            if self._pairs(plan) & self.selected_pairs:
                continue
            mask[action] = True
        mask[self.stop_action] = bool(self.selected) or not bool(mask[:-1].any())
        return mask

    def step(self, action: int):
        action = int(action)
        mask = self.action_mask()
        if not 0 <= action < self.action_size or not mask[action]:
            raise ValueError(f"invalid or masked action {action}")
        if action != self.stop_action:
            plan = self.slots[action]
            assert plan is not None
            self.selected.append(plan.plan_id)
            self.selected_requests.add(plan.request_id)
            self.selected_pairs.update(self._pairs(plan))
            return self.observe(), 0.0, False, False, self._info(phase="select")

        before = self.core.metrics()
        planning_slots = len(self.selected)
        outcome = self.core.commit(self.selected)
        after = self.core.metrics()
        # Use request counts rather than rate deltas.  Rate-based rewards shrink
        # as batch size grows (one completion is only 1/N), which can make the
        # time cost dominate and teach the policy to stop acting at large N.
        request_count = max(len(self.core.requests), 1)
        completed_delta = (
            after["completion_rate"] - before["completion_rate"]
        ) * request_count
        timeout_delta = (
            after["timeout_rate"] - before["timeout_rate"]
        ) * request_count
        time_delta = after["makespan"] - before["makespan"]
        reward = completed_delta - timeout_delta - time_delta / max(self.core.spec.horizon, 1)
        settled = all(not state.active for state in self.core.requests.values())
        terminated = settled
        truncated = self.core.time >= self.core.spec.horizon and not settled
        self.selected = []
        self.selected_requests = set()
        self.selected_pairs = set()
        if not (terminated or truncated):
            self.snapshot = self.core.snapshot()
            self.slots = self._slot_candidates(self.snapshot)
        else:
            self.snapshot = PlanningSnapshot(self.core.time, (), (), (), (), after)
            self.slots = [None] * self.stop_action
        info = self._info(phase="execute")
        info.update(outcome)
        info["planning_slots"] = planning_slots
        return self.observe(), float(reward), terminated, truncated, info

    def observe(self) -> dict[str, np.ndarray]:
        global_features = np.zeros(self.global_feature_dim, dtype=np.float32)
        metrics = self.core.metrics()
        active = sum(state.active for state in self.core.requests.values())
        global_features[:8] = (
            self.core.time / max(self.core.spec.horizon, 1),
            active / max(len(self.core.requests), 1),
            metrics["completion_rate"], metrics["timeout_rate"],
            len(self.core.backend.pairs) / max(len(self.core.spec.edges), 1),
            len(self.snapshot.candidates) / max(self.stop_action, 1),
            metrics["generated_eprs"] / max(len(self.core.spec.edges), 1),
            metrics["swaps"] / max(self.config.max_hops, 1),
        )
        request_features = np.zeros(
            (self.config.max_requests, self.request_feature_dim), dtype=np.float32
        )
        request_mask = np.zeros(self.config.max_requests, dtype=bool)
        request_index: dict[str, int] = {}
        for index, row in enumerate(self.snapshot.requests[:self.config.max_requests]):
            request_index[str(row["id"])] = index
            if row["completed_at"] is not None or row["expired_at"] is not None:
                continue
            request_mask[index] = True
            source, destination, frontier = int(row["source"]), int(row["destination"]), int(row["frontier"])
            hops = max(1, destination - source)
            remaining = max(0, destination - frontier)
            deadline = row["deadline"]
            ttl_remaining = 1.0 if deadline is None else max(0.0, (int(deadline) - self.core.time) / max(hops, 1))
            request_features[index, :8] = (
                1.0, hops / max(self.config.max_hops, 1),
                (hops - remaining) / hops, remaining / max(self.config.max_hops, 1),
                min(ttl_remaining, 1.0), float(row["carried_pair_id"] is not None),
                float(str(row["id"]) in self.selected_requests),
                self.core.time / max(self.core.spec.horizon, 1),
            )
        candidate_features = np.zeros(
            (self.stop_action, self.candidate_feature_dim), dtype=np.float32
        )
        for action, plan in enumerate(self.slots):
            if plan is None:
                continue
            candidate_features[action, :9] = (
                1.0, request_index.get(plan.request_id, 0) / max(self.config.max_requests - 1, 1),
                (len(plan.route_nodes) - 1) / max(self.config.max_hops, 1),
                plan.duration / max(self.config.max_hops, 1),
                float(plan.completes_request), len(plan.elementary_pair_ids) / max(self.config.max_hops, 1),
                len(plan.swap_actions) / max(self.config.max_hops, 1),
                float(plan.request_id in self.selected_requests),
                float(bool(self._pairs(plan) & self.selected_pairs)),
            )
        return {
            "global_features": global_features,
            "request_features": request_features,
            "request_mask": request_mask,
            "candidate_features": candidate_features,
            "action_mask": self.action_mask(),
        }

    def _info(self, phase: str) -> dict[str, object]:
        return {"phase": phase, **self.core.metrics(), "stop_action": self.stop_action}
