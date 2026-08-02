"""Single-slot graph encoder and legacy autoregressive batch helper.

The authoritative multi-step training environment is
:class:`qnet_core.order_episode_env.OrderEpisodeEnv`.  This helper remains for
the deterministic one-slot mechanism tests and for encoding one slot problem.
Its candidate-selection microsteps have zero physical duration; STOP commits
the selected plans to the lower-layer executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .order_core import (
    Edge,
    Node,
    OrderAwareBatchEnv,
    OrderBatchProblem,
    OrderPlan,
    edge,
)
from .order_scenarios import make_seeded_hotspot_problem


def _node_key(node: Node) -> tuple[str, str]:
    return type(node).__name__, repr(node)


@dataclass(frozen=True)
class OrderGymConfig:
    max_nodes: int = 32
    max_edges: int = 64
    max_requests: int = 8
    max_candidates: int = 256
    max_hops: int = 8
    hotspot_capacity: int = 2
    generation_probability: float = 1.0
    swap_probability: float = 1.0
    seed: int = 0
    completion_bonus: float = 1.0
    missed_penalty: float = 0.0
    completion_time_coef: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "max_nodes", "max_edges", "max_requests",
            "max_candidates", "max_hops",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_hops < 2:
            raise ValueError("max_hops must be at least two")


class OrderGymEnv:
    """Legacy one-slot decoder plus reusable padded observation encoder."""

    global_feature_dim = 10
    node_feature_dim = 8
    edge_feature_dim = 6
    request_feature_dim = 10
    candidate_feature_dim = 10

    def __init__(self, config: OrderGymConfig | None = None):
        self.config = config or OrderGymConfig()
        self.stop_action = self.config.max_candidates
        self.action_size = self.stop_action + 1
        self._next_seed = self.config.seed
        self._terminated = True

    def reset(
        self,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        options = options or {}
        episode_seed = self._next_seed if seed is None else int(seed)
        self._next_seed = episode_seed + 1
        supplied = options.get("problem")
        if supplied is not None and not isinstance(supplied, OrderBatchProblem):
            raise TypeError("options['problem'] must be an OrderBatchProblem")
        problem = supplied or make_seeded_hotspot_problem(
            episode_seed,
            hotspot_capacity=self.config.hotspot_capacity,
            generation_probability=self.config.generation_probability,
            swap_probability=self.config.swap_probability,
            physics_seed=episode_seed,
        )
        self.core = OrderAwareBatchEnv(problem)
        self.planning_snapshot = self.core.snapshot()
        self.problem = problem
        self.candidates = tuple(sorted(
            self.planning_snapshot.candidates,
            key=lambda plan: (
                plan.priority,
                plan.request_id,
                len(plan.path),
                tuple(map(repr, plan.path)),
                plan.schedule_key,
                plan.plan_id,
            ),
        ))
        if len(self.candidates) > self.config.max_candidates:
            raise ValueError("candidate catalogue exceeds max_candidates")
        if any(len(plan.path) - 1 > self.config.max_hops for plan in self.candidates):
            raise ValueError("candidate path exceeds max_hops")
        self.candidate_by_id = {
            plan.plan_id: index for index, plan in enumerate(self.candidates)
        }
        self.request_ids = tuple(dict.fromkeys(
            plan.request_id for plan in self.candidates
        ))
        if len(self.request_ids) > self.config.max_requests:
            raise ValueError("request catalogue exceeds max_requests")
        self.request_index = {
            request_id: index for index, request_id in enumerate(self.request_ids)
        }
        self.nodes = tuple(sorted(
            self.problem.capacity, key=_node_key
        ))
        if len(self.nodes) > self.config.max_nodes:
            raise ValueError("topology exceeds max_nodes")
        self.node_index = {node: index for index, node in enumerate(self.nodes)}
        self.edges = self.problem.physical_edges
        if len(self.edges) > self.config.max_edges:
            raise ValueError("topology exceeds max_edges")
        self.selected_plan_ids: list[str] = []
        self.selected_requests: set[str] = set()
        self._terminated = False
        self._last_result = None
        return self.observe(), self._info("reset")

    def action_for_plan(self, plan_id: str) -> int:
        if plan_id not in self.candidate_by_id:
            raise ValueError(f"unknown plan ID {plan_id}")
        return self.candidate_by_id[plan_id]

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_size, dtype=bool)
        if self._terminated:
            return mask
        for action, plan in enumerate(self.candidates):
            if plan.request_id not in self.selected_requests:
                mask[action] = True
        selected_required = self.problem.required_requests <= self.selected_requests
        mask[self.stop_action] = selected_required
        return mask

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        if self._terminated:
            raise RuntimeError("step called after terminal STOP")
        action = int(action)
        mask = self.action_mask()
        if not 0 <= action < self.action_size or not mask[action]:
            raise ValueError(f"invalid or masked action {action}")
        if action != self.stop_action:
            plan = self.candidates[action]
            self.selected_plan_ids.append(plan.plan_id)
            self.selected_requests.add(plan.request_id)
            info = self._info("select")
            info["duration_ps"] = 0
            return self.observe(), 0.0, False, False, info

        result = self.core.commit(self.selected_plan_ids)
        self._last_result = result
        self._terminated = True
        total_requests = len(self.request_ids)
        normalized_time = (
            sum(result.completion_time_ps.values())
            / max(total_requests * self.problem.config.slot_duration_ps, 1)
        )
        reward = self.config.completion_bonus * result.completed_count
        reward -= self.config.missed_penalty * len(result.missed)
        reward -= self.config.completion_time_coef * normalized_time
        info = self._info("execute")
        info.update({
            "completed_count": result.completed_count,
            "completion_rate": result.completed_count / max(total_requests, 1),
            "completed": result.completed,
            "failed": result.failed,
            "missed": result.missed,
            "completion_time_ps": result.completion_time_ps,
            "duration_ps": self.problem.config.slot_duration_ps,
            "selected_plan_ids": tuple(self.selected_plan_ids),
            "remaining_inventory": result.remaining_inventory,
        })
        return self.observe(), float(reward), True, False, info

    def _initial_edge_pair_counts(self) -> dict[Edge, int]:
        counts = {elementary_edge: 0 for elementary_edge in self.edges}
        for pair in self.problem.initial_inventory:
            counts[pair.elementary_edge] += 1
        for request_id in self.problem.preloaded_requests:
            plan = next(
                candidate for candidate in self.candidates
                if candidate.request_id == request_id
            )
            for elementary_edge in plan.elementary_edges:
                counts[elementary_edge] += 1
        return counts

    def _initial_occupancy(self) -> dict[Node, int]:
        occupancy = {node: 0 for node in self.nodes}
        for (left, right), pair_count in self._initial_edge_pair_counts().items():
            occupancy[left] += pair_count
            occupancy[right] += pair_count
        return occupancy

    def _unique_request_paths(self) -> tuple[tuple[str, tuple[Node, ...]], ...]:
        values = {
            (plan.request_id, plan.path) for plan in self.candidates
        }
        return tuple(sorted(
            values,
            key=lambda item: (item[0], tuple(map(repr, item[1]))),
        ))

    def observe(self) -> dict[str, np.ndarray]:
        cfg = self.config
        capacity = self.problem.capacity
        max_capacity = max(capacity.values())
        link_specs = self.problem.link_by_edge
        max_link_capacity = max(
            link.capacity for link in link_specs.values()
        )
        total_requests = max(len(self.request_ids), 1)
        initial_occupancy = self._initial_occupancy()
        unique_paths = self._unique_request_paths()

        node_features = np.zeros(
            (cfg.max_nodes, self.node_feature_dim), dtype=np.float32
        )
        node_mask = np.zeros(cfg.max_nodes, dtype=bool)
        degree = {node: 0 for node in self.nodes}
        for left, right in self.edges:
            degree[left] += 1
            degree[right] += 1
        request_incidence = {node: set() for node in self.nodes}
        endpoint_nodes: set[Node] = set()
        internal_nodes: set[Node] = set()
        for request_id, path in unique_paths:
            endpoint_nodes.update((path[0], path[-1]))
            internal_nodes.update(path[1:-1])
            for node in path:
                request_incidence[node].add(request_id)
        for node, index in self.node_index.items():
            node_mask[index] = True
            cap = capacity[node]
            occupied = initial_occupancy[node]
            node_features[index] = (
                1.0,
                cap / max_capacity,
                occupied / cap,
                (cap - occupied) / cap,
                degree[node] / max(cfg.max_hops, 1),
                len(request_incidence[node]) / total_requests,
                float(node in endpoint_nodes),
                float(node in internal_nodes),
            )

        edge_index = np.full((2, cfg.max_edges), -1, dtype=np.int64)
        edge_features = np.zeros(
            (cfg.max_edges, self.edge_feature_dim), dtype=np.float32
        )
        edge_mask = np.zeros(cfg.max_edges, dtype=bool)
        initial_edge_pairs = self._initial_edge_pair_counts()
        edge_requests: dict[Edge, set[str]] = {value: set() for value in self.edges}
        for request_id, path in unique_paths:
            for left, right in zip(path, path[1:]):
                edge_requests[edge(left, right)].add(request_id)
        for index, (left, right) in enumerate(self.edges):
            edge_mask[index] = True
            edge_index[:, index] = self.node_index[left], self.node_index[right]
            edge_features[index] = (
                1.0,
                initial_edge_pairs[(left, right)]
                / link_specs[(left, right)].capacity,
                len(edge_requests[(left, right)]) / total_requests,
                min(capacity[left], capacity[right]) / max_capacity,
                link_specs[(left, right)].capacity / max_link_capacity,
                link_specs[(left, right)].generation_probability,
            )

        request_features = np.zeros(
            (cfg.max_requests, self.request_feature_dim), dtype=np.float32
        )
        request_mask = np.zeros(cfg.max_requests, dtype=bool)
        plans_by_request = {
            request_id: tuple(
                plan for plan in self.candidates
                if plan.request_id == request_id
            )
            for request_id in self.request_ids
        }
        max_paths_per_request = max((
            len({plan.path for plan in plans})
            for plans in plans_by_request.values()
        ), default=1)
        max_plans_per_request = max(
            (len(plans) for plans in plans_by_request.values()),
            default=1,
        )
        for request_id, index in self.request_index.items():
            plans = plans_by_request[request_id]
            paths = {plan.path for plan in plans}
            priority = min(plan.priority for plan in plans)
            representative = min(
                plans,
                key=lambda plan: (plan.priority, plan.plan_id),
            )
            ttl = (
                None if representative.deadline_slot is None
                else max(
                    representative.deadline_slot
                    - representative.arrival_slot,
                    1,
                )
            )
            age = max(
                0,
                representative.decision_slot - representative.arrival_slot,
            )
            slack = (
                0 if representative.deadline_slot is None
                else max(
                    representative.deadline_slot
                    - representative.decision_slot,
                    0,
                )
            )
            request_mask[index] = True
            request_features[index] = (
                1.0,
                priority / max(total_requests - 1, 1),
                float(request_id in self.problem.required_requests),
                float(request_id in self.problem.preloaded_requests),
                float(request_id in self.selected_requests),
                min(len(path) - 1 for path in paths) / cfg.max_hops,
                len(paths) / max(max_paths_per_request, 1),
                len(plans) / max(max_plans_per_request, 1),
                0.0 if ttl is None else min(age / ttl, 1.0),
                1.0 if ttl is None else min(slack / ttl, 1.0),
            )

        candidate_features = np.zeros(
            (cfg.max_candidates, self.candidate_feature_dim), dtype=np.float32
        )
        candidate_mask = np.zeros(cfg.max_candidates, dtype=bool)
        candidate_request_index = np.full(
            cfg.max_candidates, -1, dtype=np.int64
        )
        candidate_path_nodes = np.full(
            (cfg.max_candidates, cfg.max_hops + 1), -1, dtype=np.int64
        )
        candidate_path_mask = np.zeros(
            (cfg.max_candidates, cfg.max_hops + 1), dtype=bool
        )
        candidate_order_nodes = np.full(
            (cfg.max_candidates, cfg.max_hops - 1), -1, dtype=np.int64
        )
        candidate_order_mask = np.zeros(
            (cfg.max_candidates, cfg.max_hops - 1), dtype=bool
        )
        candidate_order_position = np.full(
            (cfg.max_candidates, cfg.max_nodes), -1.0, dtype=np.float32
        )
        candidate_node_incidence = np.zeros(
            (cfg.max_candidates, cfg.max_nodes), dtype=bool
        )
        node_pressure = {
            node: len(request_incidence[node]) / total_requests
            for node in self.nodes
        }
        for action, plan in enumerate(self.candidates):
            candidate_mask[action] = True
            request_index = self.request_index[plan.request_id]
            candidate_request_index[action] = request_index
            internal = plan.path[1:-1]
            first_pressure = (
                max(node_pressure[node] for node in plan.schedule.groups[0])
                if plan.schedule.groups else 0.0
            )
            mean_pressure = (
                sum(node_pressure[node] for node in internal) / len(internal)
                if internal else 0.0
            )
            candidate_features[action] = (
                1.0,
                request_index / max(total_requests - 1, 1),
                (len(plan.path) - 1) / cfg.max_hops,
                plan.swap_round_count / max(cfg.max_hops - 1, 1),
                float(plan.is_fixed_order),
                float(plan.request_id in self.selected_requests),
                float(plan.request_id in self.problem.required_requests),
                float(plan.request_id in self.problem.preloaded_requests),
                mean_pressure,
                first_pressure,
            )
            for offset, node in enumerate(plan.path):
                candidate_path_nodes[action, offset] = self.node_index[node]
                candidate_path_mask[action, offset] = True
                candidate_node_incidence[action, self.node_index[node]] = True
            denominator = max(plan.swap_round_count - 1, 1)
            offset = 0
            for group_index, group in enumerate(plan.schedule.groups):
                for node in group:
                    candidate_order_nodes[action, offset] = self.node_index[node]
                    candidate_order_mask[action, offset] = True
                    candidate_order_position[action, self.node_index[node]] = (
                        group_index / denominator
                    )
                    offset += 1

        global_features = np.asarray((
            1.0,
            self.problem.config.generation_interval_ps
            / self.problem.config.slot_duration_ps,
            self.problem.config.swap_service_ps
            / self.problem.config.slot_duration_ps,
            self.problem.config.memory_reset_ps
            / self.problem.config.slot_duration_ps,
            sum(
                link.generation_probability for link in link_specs.values()
            ) / max(len(link_specs), 1),
            self.problem.config.swap_probability,
            len(self.selected_requests) / total_requests,
            len(self.candidates)
            / max(total_requests * max_plans_per_request, 1),
            min(capacity.values()) / max_capacity,
            len(self.problem.required_requests) / total_requests,
        ), dtype=np.float32)

        selected_candidate_mask = np.zeros(cfg.max_candidates, dtype=bool)
        for plan_id in self.selected_plan_ids:
            selected_candidate_mask[self.candidate_by_id[plan_id]] = True
        return {
            "global_features": global_features,
            "node_features": node_features,
            "node_mask": node_mask,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "edge_mask": edge_mask,
            "request_features": request_features,
            "request_mask": request_mask,
            "candidate_features": candidate_features,
            "candidate_mask": candidate_mask,
            "candidate_request_index": candidate_request_index,
            "candidate_path_nodes": candidate_path_nodes,
            "candidate_path_mask": candidate_path_mask,
            "candidate_order_nodes": candidate_order_nodes,
            "candidate_order_mask": candidate_order_mask,
            "candidate_order_position": candidate_order_position,
            "candidate_node_incidence": candidate_node_incidence,
            "selected_candidate_mask": selected_candidate_mask,
            "action_mask": self.action_mask(),
        }

    def _info(self, phase: str) -> dict[str, object]:
        return {
            "phase": phase,
            "problem": self.problem.name,
            "stop_action": self.stop_action,
            "selected_count": len(self.selected_plan_ids),
        }
