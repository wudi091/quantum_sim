"""Pure-RL environment for sequential BatchSwap plan selection.

The environment keeps the long-hop benchmark's aggregate per-edge EPR
inventory, but replaces the MILP selector with a masked sequential action
interface.  In one planning epoch an agent repeatedly selects one bounded
prefix plan or STOP.  STOP atomically reserves the selected elementary EPRs,
then advances physical time by the maximum balanced swap-tree depth of the
batch.  Consequently a deep prefix cannot complete in one physical subslot.

No optimizer labels or demonstrations are used anywhere in this module.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np

Edge = tuple[str, str]


def _uniform(seed: int, *key: object) -> float:
    """Stable keyed randomness shared semantically with the long-hop trace."""
    digest = hashlib.sha256((str(seed) + "|" + "|".join(map(str, key))).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def edge_key(u: str, v: str) -> Edge:
    return (u, v) if u < v else (v, u)


@dataclass(frozen=True)
class RequestSpec:
    id: str
    path: tuple[str, ...]
    arrival: int = 0

    @property
    def hops(self) -> int:
        return len(self.path) - 1

    @property
    def edges(self) -> tuple[Edge, ...]:
        return tuple(edge_key(u, v) for u, v in zip(self.path, self.path[1:]))


@dataclass(frozen=True)
class BatchSwapInstance:
    seed: int
    requests: tuple[RequestSpec, ...]
    edges: tuple[Edge, ...]
    arrivals: Mapping[int, tuple[Edge, ...]]


@dataclass(frozen=True)
class CandidatePlan:
    request_id: str
    request_index: int
    plan_slot: int
    kind: str
    start_index: int
    reach_index: int
    edges: tuple[Edge, ...]
    swap_nodes: tuple[str, ...]
    completed: bool

    @property
    def progress(self) -> int:
        return self.reach_index - self.start_index

    @property
    def input_segments(self) -> int:
        # A progressed request owns one source--frontier virtual segment.
        return self.progress + int(self.start_index > 0)

    @property
    def swap_count(self) -> int:
        return max(0, self.input_segments - 1)

    @property
    def swap_depth(self) -> int:
        """Balanced binary swap-tree depth in physical subslots."""
        if self.input_segments <= 1:
            return 0
        return math.ceil(math.log2(self.input_segments))


@dataclass
class EnvConfig:
    # Fixed tensor bounds. Curriculum changes never alter these dimensions.
    max_requests: int = 30
    max_candidates_per_request: int = 3
    max_hops: int = 50
    max_subslots: int = 5000

    # Workload used on the next reset.
    request_count: int = 10
    min_hops: int = 20
    curriculum_max_hops: int = 50
    generation_probability: float = 0.30
    node_capacity: int = 2
    trace_cap: int = 5000
    seed: int = 0


@dataclass
class RewardConfig:
    """Reward weights; defaults optimize normalized total flow time."""

    flow_time_weight: float = 1.0
    gamma: float = 0.99
    potential_coef: float = 1.0
    elementary_epr_weight: float = 0.0
    swap_weight: float = 0.0
    # Alias names used by the trainer configuration.
    makespan_coef: float = 0.0
    elementary_epr_coef: float | None = None
    swap_coef: float | None = None
    completion_bonus: float = 0.0
    truncation_remaining_weight: float = 1.0


CURRICULUM: tuple[dict[str, int | float], ...] = (
    {"request_count": 4, "min_hops": 2, "curriculum_max_hops": 5,
     "generation_probability": 0.80, "node_capacity": 3},
    {"request_count": 10, "min_hops": 5, "curriculum_max_hops": 15,
     "generation_probability": 0.50, "node_capacity": 2},
    {"request_count": 30, "min_hops": 20, "curriculum_max_hops": 50,
     "generation_probability": 0.30, "node_capacity": 2},
)


def _shared_path(request_index: int, hops: int) -> tuple[str, ...]:
    """Simple path of exactly ``hops`` with one shared center edge."""
    if hops < 1:
        raise ValueError("hops must be positive")
    if hops == 1:
        return (f"r{request_index}_s", f"r{request_index}_d")
    if hops == 2:
        return (f"r{request_index}_s", "core0", f"r{request_index}_d")

    side_edges = hops - 1  # everything except core0--core1
    left_edges = side_edges // 2
    right_edges = side_edges - left_edges
    nodes = [f"r{request_index}_s"]
    nodes.extend(f"r{request_index}_l{k}" for k in range(left_edges - 1))
    nodes.extend(("core0", "core1"))
    nodes.extend(f"r{request_index}_r{k}" for k in range(right_edges - 1))
    nodes.append(f"r{request_index}_d")
    assert len(nodes) - 1 == hops
    return tuple(nodes)


def make_instance(config: EnvConfig, seed: int | None = None) -> BatchSwapInstance:
    """Create a keyed, policy-independent, demand-capped EPR trace."""
    seed = config.seed if seed is None else seed
    if not 1 <= config.request_count <= config.max_requests:
        raise ValueError("request_count must be in [1, max_requests]")
    if not 1 <= config.min_hops <= config.curriculum_max_hops <= config.max_hops:
        raise ValueError("invalid curriculum hop range")
    if not 0.0 < config.generation_probability <= 1.0:
        raise ValueError("generation_probability must be in (0, 1]")
    if config.max_candidates_per_request < 1:
        raise ValueError("max_candidates_per_request must be positive")

    if config.request_count == 1:
        hops = [config.curriculum_max_hops]
    else:
        span = config.curriculum_max_hops - config.min_hops
        hops = [config.min_hops + round(span * i / (config.request_count - 1))
                for i in range(config.request_count)]
    rng = np.random.default_rng(seed)
    rng.shuffle(hops)
    requests = tuple(RequestSpec(f"r{i}", _shared_path(i, int(hops[i])))
                     for i in range(config.request_count))

    demand: Counter[Edge] = Counter()
    for request in requests:
        demand.update(set(request.edges))
    edges = tuple(sorted(demand))
    generated: Counter[Edge] = Counter()
    arrivals: dict[int, list[Edge]] = defaultdict(list)
    for subslot in range(config.trace_cap):
        for edge_index, edge in enumerate(edges):
            if generated[edge] >= demand[edge]:
                continue
            if _uniform(seed, "batchswap_rl_generation", subslot, edge_index) < config.generation_probability:
                generated[edge] += 1
                arrivals[subslot].append(edge)
        if all(generated[edge] == demand[edge] for edge in edges):
            break
    else:
        missing = sum(demand[e] - generated[e] for e in edges)
        raise RuntimeError(f"EPR trace did not fill by trace_cap; {missing} pairs missing")
    return BatchSwapInstance(seed, requests, edges,
                             {t: tuple(values) for t, values in arrivals.items()})


class BatchSwapEnv:
    """Fixed-shape masked environment with sequential batch construction."""

    global_feature_dim = 10
    request_feature_dim = 10
    candidate_feature_dim = 18

    def __init__(self, config: EnvConfig | None = None,
                 reward_config: RewardConfig | None = None,
                 instance: BatchSwapInstance | None = None) -> None:
        self.config = config or EnvConfig()
        self.reward_config = reward_config or RewardConfig()
        self._fixed_instance = instance
        self.curriculum_stage: int | None = None
        # Explicit resets anchor a deterministic episode-seed stream.  Training
        # can then call reset() after each episode without replaying one fixed
        # instance forever.
        self._next_episode_seed = self.config.seed
        self.stop_action = (self.config.max_requests
                            * self.config.max_candidates_per_request)
        self.action_size = self.stop_action + 1

    def set_curriculum(self, stage: int | object) -> None:
        # PPOTrainer passes its CurriculumStage object; the standalone API
        # also accepts integer indices for convenient smoke runs.
        if not isinstance(stage, (int, np.integer)):
            max_requests = int(getattr(stage, "max_requests"))
            min_hops = int(getattr(stage, "min_hops"))
            max_hops = int(getattr(stage, "max_hops"))
            if max_requests > self.config.max_requests:
                raise ValueError("fixed max_requests is too small for this curriculum stage")
            if max_hops > self.config.max_hops:
                raise ValueError("fixed max_hops is too small for this curriculum stage")
            stage_name = str(getattr(stage, "name", "")).lower()
            stage_index = {"short": 0, "medium": 1, "long": 2}.get(stage_name)
            physical = CURRICULUM[stage_index] if stage_index is not None else {}
            self.config = replace(
                self.config,
                request_count=max_requests,
                min_hops=min_hops,
                curriculum_max_hops=max_hops,
                generation_probability=float(
                    physical.get("generation_probability", self.config.generation_probability)
                ),
                node_capacity=int(physical.get("node_capacity", self.config.node_capacity)),
            )
            self.curriculum_stage = None
            self._fixed_instance = None
            self._next_episode_seed = self.config.seed
            return
        stage = int(stage)
        if not 0 <= stage < len(CURRICULUM):
            raise ValueError(f"curriculum stage must be in [0, {len(CURRICULUM) - 1}]")
        updates = CURRICULUM[stage]
        if int(updates["request_count"]) > self.config.max_requests:
            raise ValueError("fixed max_requests is too small for this curriculum stage")
        if int(updates["curriculum_max_hops"]) > self.config.max_hops:
            raise ValueError("fixed max_hops is too small for this curriculum stage")
        self.config = replace(self.config, **updates)
        self.curriculum_stage = stage
        self._fixed_instance = None
        self._next_episode_seed = self.config.seed

    def reset(self, seed: int | None = None,
              options: Mapping[str, object] | None = None) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        del options
        if seed is None:
            episode_seed = self._next_episode_seed
        else:
            episode_seed = int(seed)
        self._next_episode_seed = episode_seed + 1
        self.instance = self._fixed_instance or make_instance(self.config, episode_seed)
        if len(self.instance.requests) > self.config.max_requests:
            raise ValueError("instance has more requests than fixed observation capacity")
        if any(r.hops > self.config.max_hops for r in self.instance.requests):
            raise ValueError("instance hop count exceeds fixed observation capacity")
        self.time = 0
        self.planning_slots = 0
        self.inventory: Counter[Edge] = Counter(self.instance.arrivals.get(0, ()))
        self.frontier = {request.id: 0 for request in self.instance.requests}
        self.completed_at: dict[str, int] = {}
        self.last_service = {request.id: request.arrival for request in self.instance.requests}
        self.elementary_eprs = 0
        self.virtual_inputs = 0
        self.swaps = 0
        self._begin_selection()
        return self.observe(), self._info(duration=0, completed_now=0)

    def _begin_selection(self) -> None:
        self.selected_plans: list[CandidatePlan] = []
        self.selected_requests: set[str] = set()
        self.reserved_edges: Counter[Edge] = Counter()
        self.node_load: Counter[str] = Counter()
        self.current_plans: list[CandidatePlan | None] = [None] * self.stop_action
        for request_index, request in enumerate(self.instance.requests):
            if request.id in self.completed_at or request.arrival > self.time:
                continue
            plans = self._plans_for_request(request, request_index)
            for plan in plans:
                action = request_index * self.config.max_candidates_per_request + plan.plan_slot
                self.current_plans[action] = plan

    def _plans_for_request(self, request: RequestSpec,
                           request_index: int) -> tuple[CandidatePlan, ...]:
        start = self.frontier[request.id]
        reach = start
        needed: Counter[Edge] = Counter()
        while reach < request.hops:
            edge = request.edges[reach]
            needed[edge] += 1
            if needed[edge] > self.inventory[edge]:
                break
            reach += 1
        progress = reach - start
        if progress <= 0:
            return ()

        proposals = (
            ("max", reach),
            ("half", start + max(1, math.ceil(progress / 2))),
            ("short", start + 1),
        )
        unique: list[tuple[str, int]] = []
        seen: set[int] = set()
        for kind, target in proposals:
            if target not in seen:
                unique.append((kind, target))
                seen.add(target)
            if len(unique) >= self.config.max_candidates_per_request:
                break

        plans = []
        for plan_slot, (kind, target) in enumerate(unique):
            swap_nodes = (request.path[1:target] if start == 0
                          else request.path[start:target])
            plans.append(CandidatePlan(
                request.id, request_index, plan_slot, kind, start, target,
                request.edges[start:target], tuple(swap_nodes),
                target == request.hops,
            ))
        return tuple(plans)

    def decode_action(self, action: int) -> CandidatePlan | None:
        if action == self.stop_action:
            return None
        if not 0 <= action < self.stop_action:
            raise ValueError(f"action {action} outside [0, {self.stop_action}]")
        return self.current_plans[action]

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_size, dtype=bool)
        for action, plan in enumerate(self.current_plans):
            if plan is None or plan.request_id in self.selected_requests:
                continue
            edge_need = Counter(plan.edges)
            if any(self.reserved_edges[e] + count > self.inventory[e]
                   for e, count in edge_need.items()):
                continue
            node_need = Counter(plan.swap_nodes)
            if any(self.node_load[node] + count > self.config.node_capacity
                   for node, count in node_need.items()):
                continue
            mask[action] = True
        mask[self.stop_action] = True
        return mask

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        action = int(action)
        mask = self.action_mask()
        if not 0 <= action < self.action_size or not mask[action]:
            raise ValueError(f"invalid or masked action {action}")
        if action != self.stop_action:
            plan = self.current_plans[action]
            assert plan is not None
            self.selected_plans.append(plan)
            self.selected_requests.add(plan.request_id)
            self.reserved_edges.update(plan.edges)
            self.node_load.update(plan.swap_nodes)
            info = self._info(duration=0, completed_now=0)
            info["phase"] = "select"
            info["selected_action"] = action
            return self.observe(), 0.0, False, False, info
        return self._execute_batch()

    def _execute_batch(self) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        active_before = self._active_requests()
        previous_potential = self._potential()
        duration = max(1, max((plan.swap_depth for plan in self.selected_plans), default=0))
        elementary_now = sum(len(plan.edges) for plan in self.selected_plans)
        swaps_now = sum(plan.swap_count for plan in self.selected_plans)
        for edge, count in self.reserved_edges.items():
            if self.inventory[edge] < count:
                raise RuntimeError("reserved EPR inventory invariant violated")
            self.inventory[edge] -= count

        old_time = self.time
        self.time += duration
        for subslot in range(old_time + 1, self.time + 1):
            self.inventory.update(self.instance.arrivals.get(subslot, ()))

        completed_now = 0
        for plan in self.selected_plans:
            self.frontier[plan.request_id] = plan.reach_index
            self.last_service[plan.request_id] = self.time
            if plan.completed:
                self.completed_at[plan.request_id] = self.time
                completed_now += 1
        self.elementary_eprs += elementary_now
        self.virtual_inputs += sum(plan.start_index > 0 for plan in self.selected_plans)
        self.swaps += swaps_now
        self.planning_slots += 1

        flow_reward = -(self.reward_config.flow_time_weight * len(active_before) * duration
                        / max(self.config.max_requests, 1))
        makespan_reward = -self.reward_config.makespan_coef * duration
        epr_weight = (self.reward_config.elementary_epr_weight
                      if self.reward_config.elementary_epr_coef is None
                      else self.reward_config.elementary_epr_coef)
        swap_weight = (self.reward_config.swap_weight
                       if self.reward_config.swap_coef is None
                       else self.reward_config.swap_coef)
        epr_reward = -epr_weight * elementary_now / max(self.config.max_hops, 1)
        swap_reward = -swap_weight * swaps_now / max(self.config.max_hops, 1)
        completion_reward = self.reward_config.completion_bonus * completed_now
        reward = flow_reward + makespan_reward + epr_reward + swap_reward + completion_reward

        terminated = len(self.completed_at) == len(self.instance.requests)
        truncated = not terminated and self.time >= self.config.max_subslots
        next_potential = 0.0 if terminated else self._potential()
        potential_reward = self.reward_config.potential_coef * (
            self.reward_config.gamma ** duration * next_potential - previous_potential
        )
        reward += potential_reward
        if truncated:
            remaining = sum(request.hops - self.frontier[request.id]
                            for request in self.instance.requests
                            if request.id not in self.completed_at)
            reward -= (self.reward_config.truncation_remaining_weight * remaining
                       / max(self.config.max_hops, 1))

        selected_count = len(self.selected_plans)
        if not terminated and not truncated:
            self._begin_selection()
        else:
            # Keep terminal observations well-formed with STOP as the sole action.
            self.selected_plans = []
            self.selected_requests = set()
            self.reserved_edges = Counter()
            self.node_load = Counter()
            self.current_plans = [None] * self.stop_action
        obs = self.observe()
        info = self._info(duration=duration, completed_now=completed_now)
        info.update({"phase": "execute", "selected_count": selected_count,
                     "elementary_now": elementary_now, "swaps_now": swaps_now,
                     "reward_potential": potential_reward,
                     "reward_flow_time": flow_reward,
                     "reward_makespan": makespan_reward,
                     "reward_elementary_epr": epr_reward,
                     "reward_swaps": swap_reward,
                     "reward_completion": completion_reward})
        return obs, float(reward), terminated, truncated, info

    def _potential(self) -> float:
        """State-only Phi: negative normalized aggregate remaining hops."""
        remaining = sum(request.hops - self.frontier[request.id]
                        for request in self.instance.requests
                        if request.arrival <= self.time and request.id not in self.completed_at)
        return -remaining / max(self.config.max_hops, 1)

    def _active_requests(self) -> list[RequestSpec]:
        return [request for request in self.instance.requests
                if request.arrival <= self.time and request.id not in self.completed_at]

    def observe(self) -> dict[str, np.ndarray]:
        request_features = np.zeros((self.config.max_requests, self.request_feature_dim),
                                    dtype=np.float32)
        request_mask = np.zeros(self.config.max_requests, dtype=bool)
        for i, request in enumerate(self.instance.requests):
            if request.id in self.completed_at or request.arrival > self.time:
                continue
            request_mask[i] = True
            frontier = self.frontier[request.id]
            remaining = request.hops - frontier
            contiguous = 0
            need: Counter[Edge] = Counter()
            for edge in request.edges[frontier:]:
                need[edge] += 1
                if need[edge] > self.inventory[edge]:
                    break
                contiguous += 1
            age = self.time - request.arrival
            request_features[i] = np.asarray([
                1.0,
                request.hops / max(self.config.max_hops, 1),
                frontier / max(request.hops, 1),
                remaining / max(self.config.max_hops, 1),
                contiguous / max(self.config.max_hops, 1),
                age / max(self.config.max_subslots, 1),
                (self.time - self.last_service[request.id]) / max(self.config.max_subslots, 1),
                float(request.id in self.selected_requests),
                self.inventory.total() / max(len(self.instance.edges) * self.config.max_requests, 1),
                1.0,
            ], dtype=np.float32)

        candidate_features = np.zeros((self.stop_action, self.candidate_feature_dim),
                                      dtype=np.float32)
        legal = self.action_mask()
        for action, plan in enumerate(self.current_plans):
            if plan is None:
                continue
            request = self.instance.requests[plan.request_index]
            before = request.hops - plan.start_index
            after = request.hops - plan.reach_index
            conflicts = sum(
                other is not None and other.request_id != plan.request_id
                and (bool(set(plan.edges) & set(other.edges))
                     or bool(set(plan.swap_nodes) & set(other.swap_nodes)))
                for other in self.current_plans
            )
            kind = (float(plan.kind == "max"), float(plan.kind == "half"),
                    float(plan.kind == "short"))
            candidate_features[action] = np.asarray([
                1.0,
                plan.request_index / max(self.config.max_requests - 1, 1),
                plan.progress / max(self.config.max_hops, 1),
                before / max(self.config.max_hops, 1),
                after / max(self.config.max_hops, 1),
                float(plan.completed),
                len(plan.edges) / max(self.config.max_hops, 1),
                plan.swap_count / max(self.config.max_hops, 1),
                plan.swap_depth / max(math.ceil(math.log2(self.config.max_hops + 1)), 1),
                (self.time - request.arrival) / max(self.config.max_subslots, 1),
                float(plan.start_index > 0),
                plan.progress / max(plan.input_segments, 1),
                conflicts / max(self.stop_action - 1, 1),
                float(legal[action]),
                *kind,
                1.0,
            ], dtype=np.float32)

        active = self._active_requests()
        remaining = [request.hops - self.frontier[request.id] for request in active]
        global_features = np.asarray([
            self.time / max(self.config.max_subslots, 1),
            self.planning_slots / max(self.config.max_subslots, 1),
            len(active) / max(self.config.max_requests, 1),
            len(self.completed_at) / max(len(self.instance.requests), 1),
            self.inventory.total() / max(len(self.instance.edges) * self.config.max_requests, 1),
            len(self.selected_plans) / max(self.config.max_requests, 1),
            max(self.node_load.values(), default=0) / max(self.config.node_capacity, 1),
            (sum(remaining) / max(len(remaining), 1)) / max(self.config.max_hops, 1),
            self.config.generation_probability,
            1.0,
        ], dtype=np.float32)
        return {
            "global_features": global_features,
            "request_features": request_features,
            "request_mask": request_mask,
            "candidate_features": candidate_features,
            "action_mask": legal,
        }

    def _info(self, *, duration: int, completed_now: int) -> dict[str, object]:
        delays = [self.completed_at[r.id] - r.arrival for r in self.instance.requests
                  if r.id in self.completed_at]
        return {
            "time": self.time,
            "planning_slots": self.planning_slots,
            "duration": duration,
            "completed": len(self.completed_at),
            "completed_now": completed_now,
            "active": len(self._active_requests()),
            "selected_count": len(self.selected_plans),
            "sum_delay": sum(delays),
            "mean_delay": float(np.mean(delays)) if delays else 0.0,
            "elementary_eprs": self.elementary_eprs,
            "virtual_inputs": self.virtual_inputs,
            "swaps": self.swaps,
            "stop_action": self.stop_action,
        }


def make_env(stage: int = 0, seed: int = 0,
             reward_config: RewardConfig | None = None) -> BatchSwapEnv:
    env = BatchSwapEnv(EnvConfig(seed=seed), reward_config)
    env.set_curriculum(stage)
    return env
