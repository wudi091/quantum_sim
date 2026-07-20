"""Sequential masked PPO wrapper around the common SeQUeNCe environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace
from typing import Mapping

import numpy as np

from .env import SharedRoutingEnv
from .planner_api import PlanDescriptor, PlanningSnapshot
from .reward import RewardConfig
from .scenario import ScenarioConfig, make_episode


@dataclass
class GymConfig:
    max_requests: int = 30
    max_candidates_per_request: int = 3
    max_hops: int = 50
    scenario: ScenarioConfig = ScenarioConfig()
    seed: int = 0
    reward: RewardConfig = field(default_factory=RewardConfig)
    discount_gamma: float = 0.99


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
        self.selected_claims: set[tuple[tuple[int, int], int]] = set()
        self.selected_node_claims: dict[int, int] = {}
        self.selected_edge_claims: dict[tuple[int, int], int] = {}
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
                raise ValueError("candidate catalogue exceeds configured action slots")
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
        for lane in plan.lanes:
            values.update(lane.elementary_pair_ids)
            for action in lane.swap_actions:
                if not action.left_pair_id.startswith("@"):
                    values.add(action.left_pair_id)
                if not action.right_pair_id.startswith("@"):
                    values.add(action.right_pair_id)
        return values

    @staticmethod
    def _claims(plan: PlanDescriptor) -> set[tuple[tuple[int, int], int]]:
        return {(claim.endpoints, claim.lane) for claim in plan.claims}

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_size, dtype=bool)
        for action, plan in enumerate(self.slots):
            if plan is None or plan.request_id in self.selected_requests:
                continue
            if self._pairs(plan) & self.selected_pairs:
                continue
            if self._claims(plan) & self.selected_claims:
                continue
            node_capacity = self.core.spec.physical.node_memory_capacity
            edge_counts: dict[tuple[int, int], int] = {}
            node_counts: dict[int, int] = {}
            for claim in plan.claims:
                edge_counts[claim.endpoints] = edge_counts.get(claim.endpoints, 0) + 1
                for node in claim.endpoints:
                    node_counts[node] = node_counts.get(node, 0) + 1
            if node_capacity is not None and any(
                self.core.backend.node_occupancy(node)
                + self.selected_node_claims.get(node, 0) + count > node_capacity
                for node, count in node_counts.items()
            ):
                continue
            if any(
                sum(
                    set(pair.endpoints) == set(edge)
                    for pair in self.core.backend.pairs.values()
                ) + self.selected_edge_claims.get(edge, 0) + count
                > self.core.spec.physical.memory_capacity
                for edge, count in edge_counts.items()
            ):
                continue
            mask[action] = True
        # STOP is also a legal empty commit (one physical wait slot).  This
        # lets the policy preserve a carried frontier while waiting for a
        # better resource snapshot; forcing a plan whenever any candidate is
        # visible turns resource availability into an avoidable failure.
        mask[self.stop_action] = True
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
            self.selected_claims.update(self._claims(plan))
            for claim in plan.claims:
                self.selected_edge_claims[claim.endpoints] = (
                    self.selected_edge_claims.get(claim.endpoints, 0) + 1
                )
                for node in claim.endpoints:
                    self.selected_node_claims[node] = self.selected_node_claims.get(node, 0) + 1
            info = self._info(phase="select")
            info["duration"] = 0.0
            return self.observe(), 0.0, False, False, info

        planning_slots = len(self.selected)
        outcome = self.core.commit(self.selected)
        duration = float(outcome["duration"])
        if outcome.get("phase") == "allocate":
            self.selected = []
            self.selected_requests = set()
            self.selected_pairs = set()
            self.selected_claims = set()
            self.selected_node_claims = {}
            self.selected_edge_claims = {}
            self.snapshot = self.core.snapshot()
            self.slots = self._slot_candidates(self.snapshot)
            info = self._info(phase="allocate")
            info.update(outcome)
            info["planning_slots"] = planning_slots
            return self.observe(), 0.0, False, False, info
        potential_before = float(outcome["progress_potential_before"])
        potential_after = float(outcome["progress_potential_after"])
        reward_progress = (
            self.config.discount_gamma ** duration * potential_after
            - potential_before
        )
        reward = self.config.reward.potential_coef * reward_progress
        reward += self.config.reward.completion_bonus * float(
            outcome.get("delivered_pairs_now", outcome["completed_now"])
        )
        reward -= self.config.reward.failure_coef * float(outcome["failed_now"])
        reward -= self.config.reward.timeout_coef * float(outcome["expired_now"])
        reward -= self.config.reward.makespan_coef * duration
        settled = all(not state.active for state in self.core.requests.values())
        terminated = settled
        truncated = self.core.time >= self.core.spec.horizon and not settled
        self.selected = []
        self.selected_requests = set()
        self.selected_pairs = set()
        self.selected_claims = set()
        self.selected_node_claims = {}
        self.selected_edge_claims = {}
        if not (terminated or truncated):
            self.snapshot = self.core.snapshot()
            self.slots = self._slot_candidates(self.snapshot)
        else:
            self.snapshot = PlanningSnapshot(
                self.core.time, (), (), (), (), dict(outcome["metrics"])
            )
            self.slots = [None] * self.stop_action
        info = self._info(phase="execute")
        info.update(outcome)
        info["reward_progress"] = reward_progress
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
        global_features[8] = float(self.snapshot.phase == "allocate")
        global_features[9] = float(self.snapshot.phase == "recover")
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
            hops = max(1, int(row["initial_hops"]))
            remaining = max(0, int(row["shortest_hops"]))
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
            # Bind each candidate to its post-plan frontier state.  The request
            # table is pooled by the policy, so without this feature the actor
            # cannot tell whether a particular action leaves a long or nearly
            # complete request.  This is graph-derived state, not a routing
            # preference or expert rule.
            candidate_features[action, 9] = plan.remaining_hops / max(self.config.max_hops, 1)
            candidate_features[action, 10:14] = (
                float(plan.kind == "recovery"),
                plan.width / max(self.core.spec.physical.max_width, 1),
                min(plan.expected_throughput, 1.0),
                plan.memory_cost / max(
                    2 * self.config.max_hops * self.core.spec.physical.max_width, 1
                ),
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
