"""Multi-step training environment for online path/swap-order control.

One episode fixes a topology and a Poisson request trace.  One call to
``step`` atomically submits a complete batch of ``(request, path, swap_order)``
plans and advances exactly one control slot.  Autoregressive candidate
decoding belongs inside the policy; it never advances environment time.

The default request set is the complete arrived, unexpired pending set.  A
workload may explicitly configure ``candidate_request_cap`` to expose only an
EDF prefix as a candidate-pruning approximation; eligible, considered, and
pruned requests remain separately observable and measurable.

The existing :mod:`qnet_core.order_core` remains the single-slot physical
kernel.  This module owns the cross-slot request, inventory, deadline, and
episode state that was previously duplicated in the Waxman benchmark loop.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import secrets
from typing import Any

import numpy as np

try:  # Gymnasium is optional at source-test time, but listed for training.
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised when dependency is absent.
    gym = None
    spaces = None

from .order_core import (
    Edge,
    Node,
    OrderAwareBatchEnv,
    OrderBatchProblem,
    OrderBatchSnapshot,
    OrderSlotResult,
    OrderStoredPair,
)
from .order_gym_env import OrderGymConfig, OrderGymEnv as OrderSlotEncodingEnv
from .order_waxman import (
    WaxmanOrderConfig,
    WaxmanOrderEpisode,
    make_waxman_order_episode,
)


def _node_key(node: Node) -> tuple[str, str]:
    return type(node).__name__, repr(node)


@dataclass(frozen=True)
class _FallbackMultiBinary:
    """Small compatibility surface used when Gymnasium is not installed."""

    n: int

    @property
    def shape(self) -> tuple[int]:
        return (self.n,)

    @property
    def dtype(self):
        return np.int8

    def contains(self, value: object) -> bool:
        array = np.asarray(value)
        return (
            array.shape == (self.n,)
            and np.all((array == 0) | (array == 1))
        )

    def sample(self) -> np.ndarray:
        return np.zeros(self.n, dtype=np.int8)


if spaces is not None:
    class _CandidateBatchSpace(spaces.MultiBinary):
        """MultiBinary space with a state-independent valid default sample.

        Gymnasium's checker may reuse one sampled action across different
        resets.  The empty batch is the only action guaranteed legal in every
        active or idle slot, so unmasked ``sample()`` returns that no-op.
        Policies construct non-empty batches from the observation masks.
        """

        def __init__(self, env: "OrderEpisodeEnv", n: int) -> None:
            super().__init__(n)
            self._env = env

        def sample(self, mask=None, probability=None) -> np.ndarray:
            if mask is not None or probability is not None:
                return super().sample(mask=mask, probability=probability)
            return np.zeros(self.n, dtype=self.dtype)


_BaseEnv = gym.Env if gym is not None else object


class OrderEpisodeEnv(_BaseEnv):
    """Fixed-topology, multi-slot online training and evaluation environment."""

    metadata = {"render_modes": []}
    episode_feature_dim = 10

    def __init__(
        self,
        gym_config: OrderGymConfig | None = None,
        workload_config: WaxmanOrderConfig | None = None,
        *,
        physics_seed_root: int | None = None,
    ) -> None:
        self.workload_config = workload_config or WaxmanOrderConfig()
        self.gym_config = gym_config or self._default_gym_config(
            self.workload_config
        )
        # Never derive future physical draws from the workload/episode seed.
        # Reproducible callers pass an explicit independent root; otherwise a
        # private root is sampled once for this environment instance.
        self._configured_physics_seed_root = (
            int(physics_seed_root)
            if physics_seed_root is not None
            else secrets.randbits(32)
        )
        self._next_seed = self.gym_config.seed
        self._terminated = True
        self.candidates = ()
        self.current_problem = None

        if spaces is not None:
            self.action_space = _CandidateBatchSpace(
                self, self.gym_config.max_candidates
            )
            self.observation_space = self._gym_observation_space()
        else:
            self.action_space = _FallbackMultiBinary(
                self.gym_config.max_candidates
            )
            self.observation_space = None

    @staticmethod
    def _default_gym_config(config: WaxmanOrderConfig) -> OrderGymConfig:
        return OrderGymConfig(
            max_nodes=max(128, config.node_count),
            max_edges=max(512, config.node_count * config.average_degree),
            max_requests=config.request_count,
            max_candidates=(
                config.request_count
                * config.candidate_paths
                * config.max_swap_orders_per_path
            ),
            max_hops=config.max_hops,
            completion_time_coef=0.0,
        )

    def _gym_observation_space(self):
        cfg = self.gym_config
        box = spaces.Box
        binary = spaces.MultiBinary
        return spaces.Dict({
            "global_features": box(
                -np.inf, np.inf, (10,), dtype=np.float32
            ),
            "episode_features": box(
                -np.inf, np.inf,
                (self.episode_feature_dim,), dtype=np.float32,
            ),
            "node_features": box(
                -np.inf, np.inf, (cfg.max_nodes, 8), dtype=np.float32
            ),
            "node_mask": binary(cfg.max_nodes),
            "edge_index": box(
                -1, cfg.max_nodes - 1,
                (2, cfg.max_edges), dtype=np.int64,
            ),
            "edge_features": box(
                -np.inf, np.inf, (cfg.max_edges, 6), dtype=np.float32
            ),
            "edge_mask": binary(cfg.max_edges),
            "request_features": box(
                -np.inf, np.inf,
                (cfg.max_requests, 10), dtype=np.float32,
            ),
            "request_mask": binary(cfg.max_requests),
            "candidate_features": box(
                -np.inf, np.inf,
                (cfg.max_candidates, 10), dtype=np.float32,
            ),
            "candidate_mask": binary(cfg.max_candidates),
            "candidate_request_index": box(
                -1, cfg.max_requests - 1,
                (cfg.max_candidates,), dtype=np.int64,
            ),
            "candidate_path_nodes": box(
                -1, cfg.max_nodes - 1,
                (cfg.max_candidates, cfg.max_hops + 1), dtype=np.int64,
            ),
            "candidate_path_mask": binary(
                (cfg.max_candidates, cfg.max_hops + 1)
            ),
            "candidate_order_nodes": box(
                -1, cfg.max_nodes - 1,
                (cfg.max_candidates, cfg.max_hops - 1), dtype=np.int64,
            ),
            "candidate_order_mask": binary(
                (cfg.max_candidates, cfg.max_hops - 1)
            ),
            "candidate_order_position": box(
                -1.0, 1.0,
                (cfg.max_candidates, cfg.max_nodes), dtype=np.float32,
            ),
            "candidate_node_incidence": binary(
                (cfg.max_candidates, cfg.max_nodes)
            ),
            "selected_candidate_mask": binary(cfg.max_candidates),
            "action_mask": binary(cfg.max_candidates),
        })

    def reset(
        self,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        if gym is not None:
            super().reset(seed=seed)
        options = options or {}
        episode_seed = self._next_seed if seed is None else int(seed)
        self._next_seed = episode_seed + 1
        supplied = options.get("episode")
        if supplied is not None and not isinstance(
            supplied, WaxmanOrderEpisode
        ):
            raise TypeError("options['episode'] must be a WaxmanOrderEpisode")
        self.episode = supplied or make_waxman_order_episode(
            self.workload_config, episode_seed
        )
        self._validate_dimensions()

        initial_inventory = tuple(options.get("initial_inventory", ()))
        if any(
            not isinstance(pair, OrderStoredPair)
            for pair in initial_inventory
        ):
            raise TypeError("initial_inventory must contain OrderStoredPair")

        self._physics_seed_root = int(self._configured_physics_seed_root)
        self.current_slot = 0
        self.pending_request_ids = {
            request.request_id for request in self.episode.requests
        }
        self.completed_at: dict[str, float] = {}
        self.expired_request_ids: set[str] = set()
        self.deadline_expired_request_ids: set[str] = set()
        self.horizon_expired_request_ids: set[str] = set()
        self.inventory = initial_inventory
        self.last_result: OrderSlotResult | None = None
        self.current_problem: OrderBatchProblem | None = None
        self.planning_snapshot: OrderBatchSnapshot | None = None
        self.current_eligible_request_ids: tuple[str, ...] = ()
        self.current_considered_request_ids: tuple[str, ...] = ()
        self.current_pruned_request_ids: tuple[str, ...] = ()
        # Backward-compatible alias: a "batch" is the considered candidate
        # set, not the exogenous arrival count or complete eligible backlog.
        self.current_batch_request_ids: tuple[str, ...] = ()
        self.candidates = ()
        self.candidate_by_id: dict[str, int] = {}
        self._slot_encoder: OrderSlotEncodingEnv | None = None
        self._terminated = False

        self._steps = 0
        self._selected_plans = 0
        self._decision_batch_sum = 0
        self._decision_slots = 0
        self._active_pending_sum = 0
        self._active_pending_slots = 0
        self._max_active_pending = 0
        self._considered_request_sum = 0
        self._max_considered_requests = 0
        self._pruned_request_sum = 0
        self._slots_with_pruning = 0
        self._blocked_memory_events = 0
        self._blocked_edge_events = 0
        self._successful_swaps = 0
        self._failed_swaps = 0
        self._inventory_start_sum = 0
        self._inventory_end_sum = 0
        self._max_inventory_pairs = len(initial_inventory)
        self._expired_inventory_pairs = 0
        self._slots_with_arrivals = 0

        self._arrivals_by_slot: dict[int, int] = {}
        for request in self.episode.requests:
            self._arrivals_by_slot[request.arrival_slot] = (
                self._arrivals_by_slot.get(request.arrival_slot, 0) + 1
            )

        self._prepare_current_slot()
        return self._observation, self._info("reset")

    def _validate_dimensions(self) -> None:
        cfg = self.gym_config
        episode = self.episode
        if len(episode.nodes) > cfg.max_nodes:
            raise ValueError("episode topology exceeds max_nodes")
        if len(episode.links) > cfg.max_edges:
            raise ValueError("episode topology exceeds max_edges")
        all_request_ids = tuple(
            request.request_id for request in episode.requests
        )
        max_eligible_requests = max((
            len(episode.eligible_request_ids(all_request_ids, slot))
            for slot in range(episode.horizon_slots)
        ), default=0)
        cap = episode.config.candidate_request_cap
        # The default encoder is sized for the full workload independently of
        # pruning.  A caller that deliberately supplies a compact custom Gym
        # config alongside an explicit cap only needs to encode the considered
        # EDF prefix; eligibility is still tracked separately in state/info.
        max_considered_requests = (
            max_eligible_requests
            if cap is None
            else min(max_eligible_requests, cap)
        )
        if max_considered_requests > cfg.max_requests:
            raise ValueError(
                "episode considered request set exceeds max_requests"
            )
        max_candidates = (
            max_considered_requests
            * episode.config.candidate_paths
            * episode.config.max_swap_orders_per_path
        )
        if max_candidates > cfg.max_candidates:
            raise ValueError("episode candidate catalogue exceeds max_candidates")
        if episode.config.max_hops > cfg.max_hops:
            raise ValueError("episode path cap exceeds max_hops")

    def _physics_seed(self, slot: int) -> int:
        return int(np.random.SeedSequence([
            self._physics_seed_root,
            int(slot),
            0x534C4F54,
        ]).generate_state(1, dtype=np.uint32)[0])

    def _prepare_current_slot(self) -> None:
        slot = self.current_slot
        alive_inventory = tuple(
            pair for pair in self.inventory if pair.expires_slot > slot
        )
        self._expired_inventory_pairs += (
            len(self.inventory) - len(alive_inventory)
        )
        self.inventory = alive_inventory
        self._inventory_start = tuple(self.inventory)
        self._inventory_start_sum += len(self.inventory)
        self._max_inventory_pairs = max(
            self._max_inventory_pairs, len(self.inventory)
        )
        if self._arrivals_by_slot.get(slot, 0):
            self._slots_with_arrivals += 1

        eligible_request_ids = self.episode.eligible_request_ids(
            self.pending_request_ids, slot
        )
        considered_request_ids = self.episode.considered_request_ids(
            self.pending_request_ids, slot
        )
        considered_set = set(considered_request_ids)
        pruned_request_ids = tuple(
            request_id for request_id in eligible_request_ids
            if request_id not in considered_set
        )
        self.current_eligible_request_ids = eligible_request_ids
        self.current_considered_request_ids = considered_request_ids
        self.current_pruned_request_ids = pruned_request_ids
        self.current_batch_request_ids = considered_request_ids

        self._active_pending_slots += 1
        self._active_pending_sum += len(eligible_request_ids)
        self._max_active_pending = max(
            self._max_active_pending, len(eligible_request_ids)
        )
        self._pruned_request_sum += len(pruned_request_ids)
        self._slots_with_pruning += int(bool(pruned_request_ids))
        if considered_request_ids:
            self._decision_slots += 1
            self._decision_batch_sum += len(considered_request_ids)
            self._considered_request_sum += len(considered_request_ids)
            self._max_considered_requests = max(
                self._max_considered_requests,
                len(considered_request_ids),
            )
            problem = self.episode.problem_for_slot(
                considered_request_ids,
                slot,
                physics_seed=self._physics_seed(slot),
                initial_inventory=self.inventory,
            )
            self.current_problem = problem
            self._slot_encoder = OrderSlotEncodingEnv(self.gym_config)
            base, _ = self._slot_encoder.reset(options={"problem": problem})
            self.planning_snapshot = self._slot_encoder.planning_snapshot
            self.candidates = self._slot_encoder.candidates
            self.candidate_by_id = dict(self._slot_encoder.candidate_by_id)
            self._observation = self._episode_observation(base)
        else:
            self.current_problem = None
            self.planning_snapshot = None
            self.candidates = ()
            self.candidate_by_id = {}
            self._slot_encoder = None
            self._observation = self._empty_observation()

    def _episode_features(self) -> np.ndarray:
        total = max(len(self.episode.requests), 1)
        horizon = max(self.episode.horizon_slots, 1)
        arrived_pending = len(self.current_eligible_request_ids)
        future_pending = len(self.pending_request_ids) - arrived_pending
        inventory_scale = max(
            sum(link.capacity for link in self.episode.links), 1
        )
        return np.asarray((
            1.0,
            self.current_slot / horizon,
            max(horizon - self.current_slot, 0) / horizon,
            self._arrivals_by_slot.get(self.current_slot, 0) / total,
            len(self.current_considered_request_ids)
            / max(arrived_pending, 1),
            arrived_pending / total,
            future_pending / total,
            len(self.completed_at) / total,
            len(self.expired_request_ids) / total,
            len(self.inventory) / inventory_scale,
        ), dtype=np.float32)

    def _episode_observation(
        self, base: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        observation = dict(base)
        observation["episode_features"] = self._episode_features()
        observation["selected_candidate_mask"] = np.zeros(
            self.gym_config.max_candidates, dtype=bool
        )
        observation["action_mask"] = observation["candidate_mask"].copy()
        return observation

    def _empty_observation(self) -> dict[str, np.ndarray]:
        cfg = self.gym_config
        nodes = tuple(sorted(self.episode.nodes, key=_node_key))
        node_index = {node: index for index, node in enumerate(nodes)}
        capacity = self.episode.capacity
        max_capacity = max(capacity.values())
        edges = tuple(link.elementary_edge for link in self.episode.links)
        max_link_capacity = max(link.capacity for link in self.episode.links)

        inventory_by_edge: dict[Edge, int] = {value: 0 for value in edges}
        occupancy = {node: 0 for node in nodes}
        for pair in self.inventory:
            inventory_by_edge[pair.elementary_edge] += 1
            occupancy[pair.left] += 1
            occupancy[pair.right] += 1

        degree = {node: 0 for node in nodes}
        for left, right in edges:
            degree[left] += 1
            degree[right] += 1

        node_features = np.zeros((cfg.max_nodes, 8), dtype=np.float32)
        node_mask = np.zeros(cfg.max_nodes, dtype=bool)
        for node, index in node_index.items():
            node_mask[index] = True
            cap = capacity[node]
            used = occupancy[node]
            node_features[index] = (
                1.0,
                cap / max_capacity,
                used / cap,
                (cap - used) / cap,
                degree[node] / max(cfg.max_hops, 1),
                0.0,
                0.0,
                0.0,
            )

        edge_index = np.full((2, cfg.max_edges), -1, dtype=np.int64)
        edge_features = np.zeros((cfg.max_edges, 6), dtype=np.float32)
        edge_mask = np.zeros(cfg.max_edges, dtype=bool)
        for index, link in enumerate(self.episode.links):
            left, right = link.elementary_edge
            edge_mask[index] = True
            edge_index[:, index] = node_index[left], node_index[right]
            edge_features[index] = (
                1.0,
                inventory_by_edge[link.elementary_edge] / link.capacity,
                0.0,
                min(capacity[left], capacity[right]) / max_capacity,
                link.capacity / max_link_capacity,
                link.generation_probability,
            )

        mean_probability = sum(
            link.generation_probability for link in self.episode.links
        ) / len(self.episode.links)
        global_features = np.asarray((
            1.0,
            self.episode.config.generation_interval_ps
            / self.episode.config.slot_duration_ps,
            self.episode.config.swap_service_ps
            / self.episode.config.slot_duration_ps,
            self.episode.config.memory_reset_ps
            / self.episode.config.slot_duration_ps,
            mean_probability,
            self.episode.config.swap_probability,
            0.0,
            0.0,
            min(capacity.values()) / max_capacity,
            0.0,
        ), dtype=np.float32)

        candidate_request_index = np.full(
            cfg.max_candidates, -1, dtype=np.int64
        )
        candidate_path_nodes = np.full(
            (cfg.max_candidates, cfg.max_hops + 1), -1, dtype=np.int64
        )
        candidate_order_nodes = np.full(
            (cfg.max_candidates, cfg.max_hops - 1), -1, dtype=np.int64
        )
        candidate_order_position = np.full(
            (cfg.max_candidates, cfg.max_nodes), -1.0, dtype=np.float32
        )
        return {
            "global_features": global_features,
            "episode_features": self._episode_features(),
            "node_features": node_features,
            "node_mask": node_mask,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "edge_mask": edge_mask,
            "request_features": np.zeros(
                (cfg.max_requests, 10), dtype=np.float32
            ),
            "request_mask": np.zeros(cfg.max_requests, dtype=bool),
            "candidate_features": np.zeros(
                (cfg.max_candidates, 10), dtype=np.float32
            ),
            "candidate_mask": np.zeros(cfg.max_candidates, dtype=bool),
            "candidate_request_index": candidate_request_index,
            "candidate_path_nodes": candidate_path_nodes,
            "candidate_path_mask": np.zeros(
                (cfg.max_candidates, cfg.max_hops + 1), dtype=bool
            ),
            "candidate_order_nodes": candidate_order_nodes,
            "candidate_order_mask": np.zeros(
                (cfg.max_candidates, cfg.max_hops - 1), dtype=bool
            ),
            "candidate_order_position": candidate_order_position,
            "candidate_node_incidence": np.zeros(
                (cfg.max_candidates, cfg.max_nodes), dtype=bool
            ),
            "selected_candidate_mask": np.zeros(
                cfg.max_candidates, dtype=bool
            ),
            "action_mask": np.zeros(cfg.max_candidates, dtype=bool),
        }

    def action_for_plan_ids(
        self, plan_ids: Iterable[str]
    ) -> np.ndarray:
        values = tuple(plan_ids)
        if len(set(values)) != len(values):
            raise ValueError("a plan ID cannot appear twice in one batch")
        unknown = set(values) - self.candidate_by_id.keys()
        if unknown:
            raise ValueError(f"unknown current-slot plan IDs: {sorted(unknown)}")
        action = np.zeros(self.gym_config.max_candidates, dtype=np.int8)
        for plan_id in values:
            action[self.candidate_by_id[plan_id]] = 1
        self._validate_selected_plan_ids(values)
        return action

    def _decode_action(self, action: Any) -> tuple[str, ...]:
        self._action_violation_count = 0
        if isinstance(action, Mapping):
            if "selected" not in action:
                raise ValueError("batch action mapping needs a 'selected' field")
            action = action["selected"]
        if action is None:
            values: tuple[Any, ...] = ()
        elif isinstance(action, str):
            values = (action,)
        elif isinstance(action, np.ndarray):
            array = np.asarray(action)
            if array.shape != (self.gym_config.max_candidates,):
                raise ValueError(
                    "multi-hot action must have shape "
                    f"({self.gym_config.max_candidates},)"
                )
            if not np.all((array == 0) | (array == 1)):
                raise ValueError("multi-hot action must contain only 0/1 values")
            indices = tuple(np.flatnonzero(array).tolist())
            if self.current_problem is None:
                if indices:
                    raise ValueError("an idle slot accepts only an empty batch")
                return ()
            if any(index >= len(self.candidates) for index in indices):
                raise ValueError("batch action selects a padded candidate")
            plan_ids = tuple(
                self.candidates[index].plan_id for index in indices
            )
            self._validate_selected_plan_ids(plan_ids)
            return plan_ids
        else:
            values = tuple(action)

        if not values:
            plan_ids: tuple[str, ...] = ()
        elif all(isinstance(value, str) for value in values):
            plan_ids = tuple(values)
        elif all(isinstance(value, (int, np.integer)) for value in values):
            indices = tuple(map(int, values))
            if len(set(indices)) != len(indices):
                raise ValueError("a candidate index cannot appear twice")
            if any(
                index < 0 or index >= self.gym_config.max_candidates
                for index in indices
            ):
                raise ValueError("candidate index is outside the action vector")
            if any(index >= len(self.candidates) for index in indices):
                raise ValueError("batch action selects a padded candidate")
            plan_ids = tuple(self.candidates[index].plan_id for index in indices)
        else:
            raise TypeError("batch action must use plan IDs, indices, or multi-hot")

        self._validate_selected_plan_ids(plan_ids)
        return plan_ids

    def _validate_selected_plan_ids(self, plan_ids: Sequence[str]) -> None:
        if not self.current_problem:
            if plan_ids:
                raise ValueError("an idle slot accepts only an empty batch")
            return
        lookup = {plan.plan_id: plan for plan in self.candidates}
        unknown = set(plan_ids) - lookup.keys()
        if unknown:
            raise ValueError(f"unknown current-slot plan IDs: {sorted(unknown)}")
        request_ids = [lookup[plan_id].request_id for plan_id in plan_ids]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("a request may select at most one complete candidate")
        missing = self.current_problem.required_requests - set(request_ids)
        if missing:
            raise ValueError(f"required requests are missing: {sorted(missing)}")

    def step(
        self, action: Any
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        if self._terminated:
            raise RuntimeError("step called after episode termination")
        executed_slot = self.current_slot
        eligible_request_ids = self.current_eligible_request_ids
        considered_request_ids = self.current_considered_request_ids
        pruned_request_ids = self.current_pruned_request_ids
        batch_request_ids = self.current_batch_request_ids
        inventory_start = tuple(self.inventory)
        selected_plan_ids = self._decode_action(action)

        if self.current_problem is None:
            result = OrderSlotResult(
                selected_plan_ids=(),
                completed=(),
                failed=(),
                missed=(),
                completion_time_ps={},
                traces=(),
                remaining_inventory=inventory_start,
            )
        else:
            result = OrderAwareBatchEnv(self.current_problem).commit(
                selected_plan_ids
            )
        self.last_result = result
        self._steps += 1
        self._selected_plans += len(selected_plan_ids)

        request_lookup = self.episode.request_by_id
        for request_id in result.completed:
            local_time = result.completion_time_ps[request_id]
            self.completed_at[request_id] = (
                executed_slot
                + local_time / self.episode.config.slot_duration_ps
            )
            self.pending_request_ids.discard(request_id)

        for trace in result.traces:
            self._blocked_memory_events += sum(
                event.status == "blocked_memory"
                for event in trace.generation_events
            )
            self._blocked_edge_events += sum(
                event.status == "blocked_edge"
                for event in trace.generation_events
            )
            self._successful_swaps += sum(
                event.status == "success" for event in trace.swap_events
            )
            self._failed_swaps += sum(
                event.status == "random_failure"
                for event in trace.swap_events
            )

        self.inventory = result.remaining_inventory
        self._inventory_end_sum += len(self.inventory)
        self._max_inventory_pairs = max(
            self._max_inventory_pairs, len(self.inventory)
        )

        self.current_slot += 1
        deadline_expired_now: list[str] = []
        for request_id in tuple(self.pending_request_ids):
            if request_lookup[request_id].deadline_slot <= self.current_slot:
                self.pending_request_ids.remove(request_id)
                self.expired_request_ids.add(request_id)
                self.deadline_expired_request_ids.add(request_id)
                deadline_expired_now.append(request_id)
        deadline_expired_now.sort()

        self._terminated = self.current_slot >= self.episode.horizon_slots
        horizon_expired_now: list[str] = []
        if self._terminated and self.pending_request_ids:
            horizon_expired_now = sorted(self.pending_request_ids)
            self.pending_request_ids.clear()
            self.expired_request_ids.update(horizon_expired_now)
            self.horizon_expired_request_ids.update(horizon_expired_now)
        expired_now = tuple(deadline_expired_now + horizon_expired_now)

        normalized_time = (
            sum(result.completion_time_ps.values())
            / max(
                len(result.completed) * self.episode.config.slot_duration_ps,
                1,
            )
        )
        reward = (
            self.gym_config.completion_bonus * result.completed_count
            - self.gym_config.missed_penalty * len(expired_now)
            - self.gym_config.completion_time_coef * normalized_time
        )

        if self._terminated:
            self.current_problem = None
            self.planning_snapshot = None
            self.current_eligible_request_ids = ()
            self.current_considered_request_ids = ()
            self.current_pruned_request_ids = ()
            self.current_batch_request_ids = ()
            self.candidates = ()
            self.candidate_by_id = {}
            self._slot_encoder = None
            self._observation = self._empty_observation()
        else:
            self._prepare_current_slot()

        info = self._info("execute" if batch_request_ids else "idle")
        info.update({
            "slot": executed_slot,
            "next_slot": self.current_slot,
            "duration_ps": self.episode.config.slot_duration_ps,
            "eligible_request_ids": eligible_request_ids,
            "considered_request_ids": considered_request_ids,
            "pruned_request_ids": pruned_request_ids,
            "eligible_request_count": len(eligible_request_ids),
            "considered_request_count": len(considered_request_ids),
            "pruned_request_count": len(pruned_request_ids),
            "batch_request_ids": batch_request_ids,
            "selected_plan_ids": selected_plan_ids,
            "completed": result.completed,
            "failed": result.failed,
            "missed": result.missed,
            "expired": expired_now,
            "deadline_expired": tuple(deadline_expired_now),
            "horizon_expired": tuple(horizon_expired_now),
            "completion_time_ps": result.completion_time_ps,
            "inventory_start": inventory_start,
            "inventory_end": tuple(self.inventory),
            "completed_count": result.completed_count,
            "physics_seed_visible": False,
            "invalid_action_count": self._action_violation_count,
        })
        return (
            self._observation,
            float(reward),
            self._terminated,
            False,
            info,
        )

    def metrics(self) -> dict[str, float | int]:
        request_lookup = self.episode.request_by_id
        delays = [
            completion - request_lookup[request_id].arrival_slot
            for request_id, completion in self.completed_at.items()
        ]
        total = max(len(self.episode.requests), 1)
        return {
            "completed_count": len(self.completed_at),
            "completion_rate": len(self.completed_at) / total,
            "timeout_count": len(self.expired_request_ids),
            "timeout_rate": len(self.expired_request_ids) / total,
            "deadline_timeout_count": len(
                self.deadline_expired_request_ids
            ),
            "horizon_timeout_count": len(
                self.horizon_expired_request_ids
            ),
            "mean_delay_slots": (
                sum(delays) / len(delays) if delays else 0.0
            ),
            "episode_steps": self._steps,
            "selected_plans": self._selected_plans,
            "mean_selected_per_decision": (
                self._selected_plans / max(self._decision_slots, 1)
            ),
            "mean_decision_batch": (
                self._decision_batch_sum / max(self._decision_slots, 1)
            ),
            "mean_active_pending": (
                self._active_pending_sum
                / max(self._active_pending_slots, 1)
            ),
            "max_active_pending": self._max_active_pending,
            "mean_considered_requests": (
                self._considered_request_sum
                / max(self._decision_slots, 1)
            ),
            "max_considered_requests": self._max_considered_requests,
            "mean_pruned_requests": (
                self._pruned_request_sum
                / max(self._active_pending_slots, 1)
            ),
            "total_pruned_request_occurrences": self._pruned_request_sum,
            "slots_with_pruning": self._slots_with_pruning,
            "blocked_memory_events": self._blocked_memory_events,
            "blocked_edge_events": self._blocked_edge_events,
            "successful_swaps": self._successful_swaps,
            "failed_swaps": self._failed_swaps,
            "mean_inventory_pairs_at_slot_start": (
                self._inventory_start_sum
                / max(self.episode.horizon_slots, 1)
            ),
            "mean_inventory_pairs_at_slot_end": (
                self._inventory_end_sum
                / max(self.episode.horizon_slots, 1)
            ),
            "max_inventory_pairs": self._max_inventory_pairs,
            "carried_inventory_pair_slots": self._inventory_start_sum,
            "expired_inventory_pairs": self._expired_inventory_pairs,
            "remaining_inventory_pairs": len(self.inventory),
            "horizon_slots": self.episode.horizon_slots,
            "slots_with_arrivals": self._slots_with_arrivals,
            "last_arrival_slot": max(
                request.arrival_slot for request in self.episode.requests
            ),
            "mean_shortest_hops": sum(
                request.shortest_hops for request in self.episode.requests
            ) / total,
            "topology_nodes": len(self.episode.nodes),
            "topology_edges": len(self.episode.links),
            "topology_average_degree": (
                2.0 * len(self.episode.links) / len(self.episode.nodes)
            ),
        }

    def _info(self, phase: str) -> dict[str, object]:
        unsettled_total_count = len(self.pending_request_ids)
        arrived_pending_count = len(self.current_eligible_request_ids)
        cap = self.episode.config.candidate_request_cap
        return {
            "phase": phase,
            "slot": self.current_slot,
            "horizon_slots": self.episode.horizon_slots,
            "eligible_request_ids": self.current_eligible_request_ids,
            "considered_request_ids": self.current_considered_request_ids,
            "pruned_request_ids": self.current_pruned_request_ids,
            "eligible_request_count": len(
                self.current_eligible_request_ids
            ),
            "considered_request_count": len(
                self.current_considered_request_ids
            ),
            "pruned_request_count": len(self.current_pruned_request_ids),
            "batch_request_ids": self.current_batch_request_ids,
            "candidate_plan_ids": tuple(
                plan.plan_id for plan in self.candidates
            ),
            "unsettled_total_count": unsettled_total_count,
            "arrived_pending_count": arrived_pending_count,
            "candidate_request_cap": cap,
            "request_cap_applied": bool(self.current_pruned_request_ids),
            "pruning_rule": (
                "none"
                if cap is None
                else "earliest_deadline_then_arrival_then_request_id"
            ),
            # Deprecated compatibility alias.  Historically ``pending_count``
            # included requests that had not arrived yet, so preserve that
            # meaning while exposing the two unambiguous counts above.
            "pending_count": unsettled_total_count,
            "completed_count_total": len(self.completed_at),
            "expired_count_total": len(self.expired_request_ids),
            "deadline_expired_count_total": len(
                self.deadline_expired_request_ids
            ),
            "horizon_expired_count_total": len(
                self.horizon_expired_request_ids
            ),
            "inventory_pairs": len(self.inventory),
            "physics_seed_visible": False,
        }
