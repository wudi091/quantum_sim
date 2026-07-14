from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
import unittest

import numpy as np

from batchswap_reliq.env import (
    BatchSwapReliqEnv,
    EnvConfig,
    ReliqInstance,
    RequestSpec,
    _sample_requests,
)


def edge_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


@dataclass(eq=False)
class FakeLink:
    """Small QuantumLink substitute with identity-preserving token semantics."""

    start: int
    end: int
    fidelity: float = 0.99
    creation: int = 0
    token_id: str | None = None
    owner_request_id: str | None = None
    available_time: int = 0

    def __post_init__(self) -> None:
        self.initial_fidelity = self.fidelity
        self.cost = 0

    @staticmethod
    def swap(link1, link2, swap_probability, source, rng_generator,
             ignore_drop=False):
        del source, rng_generator, ignore_drop
        if swap_probability <= 0:
            fidelity = 0.0
        else:
            fidelity = min(link1.fidelity, link2.fidelity)
        endpoints = [link1.start, link1.end, link2.start, link2.end]
        shared = set((link1.start, link1.end)) & set((link2.start, link2.end))
        if not shared:
            raise ValueError("links do not share a swap node")
        shared_node = next(iter(shared))
        outer = [node for node in endpoints if node != shared_node]
        if len(outer) != 2:
            raise ValueError("fake swap only supports non-loop links")
        return FakeLink(min(outer), max(outer), fidelity)


@dataclass
class FakeEdge:
    start: int
    end: int
    links: list[FakeLink]

    def __post_init__(self) -> None:
        self.reserved_links = {}
        self.dead = False


@dataclass
class FakeNode:
    id: int
    swap_prob: float = 1.0


class FakeGraph:
    def __init__(self, nodes: list[int], edges: list[tuple[int, int]]) -> None:
        self.nodes = tuple(nodes)
        self.edges = tuple(edges)

    def has_edge(self, u: int, v: int) -> bool:
        return edge_key(u, v) in self.edges


class FakeNetwork:
    """Deterministic subset of the RELiQ QuantumNetwork API used by the env."""

    def __init__(self, edge_inventory: dict[tuple[int, int], int],
                 arrivals: dict[int, tuple[tuple[int, int], ...]] | None = None) -> None:
        self._edge_inventory = {
            edge_key(*edge): int(count) for edge, count in edge_inventory.items()
        }
        self._arrivals = {
            int(time): tuple(edge_key(*edge) for edge in edges)
            for time, edges in (arrivals or {}).items()
        }
        self.quantum_generator = np.random.default_rng(0)
        self.reset()

    def reset(self) -> None:
        self.env_steps = 0
        self.pre_step_calls = 0
        self.step_calls = 0
        self._next_token = 0
        node_ids = sorted({node for edge in self._edge_inventory for node in edge})
        self.nodes = [FakeNode(node) for node in range(max(node_ids, default=-1) + 1)]
        self.edges = []
        self.edge_node_association = {}
        for (u, v), count in sorted(self._edge_inventory.items()):
            edge = FakeEdge(u, v, [self._make_link(u, v, creation=0)
                                   for _ in range(count)])
            self.edges.append(edge)
            self.edge_node_association[(u, v)] = edge
        self.G = FakeGraph(node_ids, list(self.edge_node_association))
        size = max(node_ids, default=-1) + 1
        self.adj = np.zeros((size, size), dtype=np.float32)
        for u, v in self.edge_node_association:
            self.adj[u, v] = self.adj[v, u] = 1.0
        self.total_active_quantum_links = sum(len(edge.links) for edge in self.edges)

    def _make_link(self, u: int, v: int, *, creation: int) -> FakeLink:
        link = FakeLink(u, v, creation=creation,
                        token_id=f"base-{self._next_token}",
                        available_time=creation)
        self._next_token += 1
        return link

    def pre_step(self) -> None:
        self.pre_step_calls += 1
        # BatchSwap calls pre_step() and then step() for each physical subslot.
        arrival_time = self.env_steps + 1
        for u, v in self._arrivals.get(arrival_time, ()):
            edge = self.edge_node_association[(u, v)]
            edge.links.append(self._make_link(u, v, creation=arrival_time))
            self.total_active_quantum_links += 1

    def step(self) -> None:
        self.step_calls += 1
        self.env_steps += 1


def make_config(requests: tuple[RequestSpec, ...], *, max_subslots: int = 20,
                node_capacity: int = 2,
                request_ttl: int | None = None) -> EnvConfig:
    """Pass only fields supported by the concrete EnvConfig under test."""

    max_hops = max(len(request.path) - 1 for request in requests)
    values = {
        "max_requests": max(len(requests), 1),
        "max_candidates_per_request": 3,
        "max_hops": max(max_hops, 1),
        "max_subslots": max_subslots,
        "request_count": len(requests),
        "min_hops": 1,
        "curriculum_max_hops": max(max_hops, 1),
        "node_capacity": node_capacity,
        "request_ttl": request_ttl,
    }
    parameters = inspect.signature(EnvConfig).parameters
    return EnvConfig(**{key: value for key, value in values.items()
                        if key in parameters})


def make_env(requests: tuple[RequestSpec, ...],
             edge_inventory: dict[tuple[int, int], int], *,
             arrivals: dict[int, tuple[tuple[int, int], ...]] | None = None,
             max_subslots: int = 20,
             node_capacity: int = 2,
             request_ttl: int | None = None) -> tuple[BatchSwapReliqEnv, FakeNetwork]:
    network = FakeNetwork(edge_inventory, arrivals)
    instance = ReliqInstance(network, requests)
    env = BatchSwapReliqEnv(
        make_config(requests, max_subslots=max_subslots,
                    node_capacity=node_capacity, request_ttl=request_ttl),
        instance=instance,
    )
    env.reset()
    return env, network


def action_for(env: BatchSwapReliqEnv, request_id: str,
               kind: str = "max") -> int:
    for action, plan in enumerate(env.current_plans):
        if plan is not None and plan.request_id == request_id and plan.kind == kind:
            return action
    raise AssertionError(f"no {kind!r} candidate for request {request_id!r}")


def actions_for(env: BatchSwapReliqEnv, request_id: str) -> list[int]:
    return [action for action, plan in enumerate(env.current_plans)
            if plan is not None and plan.request_id == request_id]


class BatchSwapReliqP0Tests(unittest.TestCase):
    def test_reward_scale_is_invariant_to_request_count(self):
        single, _ = make_env((RequestSpec("r0", (0, 1)),), {(0, 1): 1})
        single.step(action_for(single, "r0"))
        _, single_reward, _, _, _ = single.step(single.stop_action)

        requests = (RequestSpec("r0", (0, 1)), RequestSpec("r1", (2, 3)))
        doubled, _ = make_env(requests, {(0, 1): 1, (2, 3): 1})
        doubled.step(action_for(doubled, "r0"))
        doubled.step(action_for(doubled, "r1"))
        _, doubled_reward, _, _, _ = doubled.step(doubled.stop_action)

        self.assertAlmostEqual(single_reward, doubled_reward, places=7)

    def test_shaping_potential_normalizes_each_request_by_its_hops(self):
        requests = (
            RequestSpec("two", (0, 1, 2)),
            RequestSpec("four", (3, 4, 5, 6, 7)),
        )
        env, _ = make_env(
            requests,
            {(0, 1): 0, (1, 2): 0, (3, 4): 0, (4, 5): 0,
             (5, 6): 0, (6, 7): 0},
        )
        env.frontier["two"] = 1
        env.frontier["four"] = 5

        self.assertAlmostEqual(env._shaping_potential(), 0.5)

    def test_timeout_revokes_prior_progress_shaping(self):
        request = RequestSpec("r0", (0, 1, 2))
        env, _ = make_env(
            (request,), {(0, 1): 1, (1, 2): 0}, request_ttl=2
        )

        env.step(action_for(env, "r0"))
        _, _, terminated, truncated, first_info = env.step(env.stop_action)
        self.assertFalse(terminated or truncated)
        _, _, terminated, truncated, second_info = env.step(env.stop_action)

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        discounted_shaping = (
            first_info["reward_progress"]
            + env.reward_config.gamma * second_info["reward_progress"]
        )
        self.assertAlmostEqual(discounted_shaping, 0.0, places=7)

    def test_topology_path_cache_is_bounded_to_one_episode(self):
        request = RequestSpec("r0", (0, 1))
        env, _ = make_env((request,), {(0, 1): 1})
        sentinel = (999, 1000, ())
        env._topology_path_cache[sentinel] = ((999, 1000),)

        env.reset(seed=1)

        self.assertNotIn(sentinel, env._topology_path_cache)

    def test_high_hop_stage_expands_to_one_hundred_requests(self):
        env = BatchSwapReliqEnv()
        env.set_curriculum(2)

        self.assertEqual(env.config.request_count, 100)
        self.assertEqual(env.config.max_requests, 100)
        self.assertEqual(env.stop_action, 300)
        self.assertEqual(env.action_size, 301)

    def test_high_hop_request_sampling_is_balanced_across_three_buckets(self):
        import networkx as nx
        from types import SimpleNamespace

        graph = nx.path_graph(110)
        config = EnvConfig(
            request_count=100,
            max_requests=100,
            min_hops=20,
            curriculum_max_hops=50,
            balanced_hop_buckets=True,
        )
        requests = _sample_requests(SimpleNamespace(G=graph), config, seed=7)
        counts = (
            sum(20 <= request.hops <= 29 for request in requests),
            sum(30 <= request.hops <= 39 for request in requests),
            sum(40 <= request.hops <= 50 for request in requests),
        )

        self.assertEqual(len(requests), 100)
        self.assertEqual(counts, (34, 33, 33))
        self.assertEqual(len({request.id for request in requests}), 100)

    def test_fixed_ttl_expires_pending_request_after_survival_steps(self):
        request = RequestSpec("r0", (0, 1))
        env, network = make_env(
            (request,), {(0, 1): 0}, max_subslots=20, request_ttl=2
        )

        _, _, terminated, truncated, info = env.step(env.stop_action)
        self.assertFalse(terminated or truncated)
        self.assertEqual(info["expired"], 0)
        _, reward, terminated, truncated, info = env.step(env.stop_action)

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertLess(reward, 0.0)
        self.assertEqual(info["time"], 2)
        self.assertEqual(info["completed"], 0)
        self.assertEqual(info["expired"], 1)
        self.assertEqual(info["timeout_rate"], 1.0)
        self.assertEqual(info["active"], 0)
        self.assertEqual(env.expired_at, {"r0": 2})
        self.assertEqual(env.instance.requests, (request,))
        self.assertEqual(network.env_steps, 2)

    def test_fixed_ttl_is_independent_of_request_hops(self):
        requests = (
            RequestSpec("short", (0, 1)),
            RequestSpec("long", (0, 1, 2, 3, 4)),
        )
        env, _ = make_env(
            requests,
            {(0, 1): 0, (1, 2): 0, (2, 3): 0, (3, 4): 0},
            request_ttl=3,
        )

        self.assertEqual(env._deadline(requests[0]), 3)
        self.assertEqual(env._deadline(requests[1]), 3)

    def test_completion_at_exact_ttl_boundary_counts_as_success(self):
        request = RequestSpec("r0", (0, 1))
        env, _ = make_env((request,), {(0, 1): 1}, request_ttl=1)

        env.step(action_for(env, "r0"))
        _, _, terminated, truncated, info = env.step(env.stop_action)

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["completed"], 1)
        self.assertEqual(info["expired"], 0)
        self.assertEqual(env.completed_at["r0"], 1)

    def test_each_plan_uses_its_own_finish_step_inside_a_batch(self):
        short = RequestSpec("short", (9, 10))
        long = RequestSpec("long", tuple(range(9)))
        inventory = {(index, index + 1): 1 for index in range(8)}
        inventory[(9, 10)] = 1
        env, _ = make_env((short, long), inventory, request_ttl=1)

        env.step(action_for(env, "short"))
        env.step(action_for(env, "long"))
        _, _, terminated, truncated, info = env.step(env.stop_action)

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["time"], 3)
        self.assertEqual(info["alive_subslots"], 2.0)
        self.assertEqual(env.completed_at, {"short": 1})
        self.assertEqual(env.expired_at, {"long": 3})

    def test_empty_stop_is_masked_while_a_request_can_advance(self):
        request = RequestSpec("r0", (0, 1))
        env, _ = make_env((request,), {(0, 1): 1})
        self.assertFalse(env.action_mask()[env.stop_action])
        env.step(action_for(env, "r0"))
        self.assertTrue(env.action_mask()[env.stop_action])

    def test_shortest_routes_are_validated_to_farthest_resource_prefix(self):
        request = RequestSpec("r0", (0, 2))
        env, _ = make_env(
            (request,), {(0, 2): 0, (0, 1): 1, (1, 2): 1}
        )

        action = action_for(env, "r0", "max")
        plan = env.decode_action(action)
        self.assertEqual(plan.start_index, 0)
        self.assertEqual(plan.reach_index, 2)
        self.assertEqual(
            {edge_key(link.start, link.end) for link in plan.base_links},
            {(0, 1), (1, 2)},
        )
        env.step(action)
        _, _, terminated, truncated, _ = env.step(env.stop_action)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_concrete_base_token_cannot_be_double_spent(self):
        requests = (
            RequestSpec("r0", (0, 1)),
            RequestSpec("r1", (0, 1)),
        )
        env, network = make_env(requests, {(0, 1): 1})
        token = network.edge_node_association[(0, 1)].links[0]
        self.assertIsNone(token.owner_request_id)

        env.step(action_for(env, "r0"))
        # Sequential planning is instantaneous; only STOP advances physics.
        self.assertEqual(network.env_steps, 0)
        self.assertEqual(network.pre_step_calls, 0)
        self.assertIn(token, network.edge_node_association[(0, 1)].links)
        mask = env.action_mask()
        self.assertTrue(actions_for(env, "r1"))
        self.assertTrue(all(not mask[action] for action in actions_for(env, "r1")))

        _, _, terminated, truncated, info = env.step(env.stop_action)
        self.assertFalse(terminated or truncated)
        self.assertEqual(info["completed_now"], 1)
        self.assertNotIn(token, network.edge_node_association[(0, 1)].links)
        self.assertEqual(len(network.edge_node_association[(0, 1)].links), 0)

    def test_partial_swap_creates_request_owned_token_used_by_next_batch(self):
        request = RequestSpec("r0", (0, 1, 2, 3))
        env, network = make_env(
            (request,),
            {(0, 1): 1, (1, 2): 1, (2, 3): 0},
            arrivals={1: ((2, 3),)},
        )

        env.step(action_for(env, "r0"))
        _, _, terminated, truncated, info = env.step(env.stop_action)
        self.assertFalse(terminated or truncated)
        self.assertEqual(info["duration"], 1)
        self.assertEqual(env.frontier["r0"], 2)
        carried = env.carried_links["r0"]
        self.assertEqual({carried.start, carried.end}, {0, 2})
        self.assertEqual(carried.owner_request_id, "r0")
        self.assertEqual(carried.available_time, 1)
        self.assertEqual(len(network.edge_node_association[(2, 3)].links), 1)

        env.step(action_for(env, "r0"))
        _, _, terminated, truncated, info = env.step(env.stop_action)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["elementary_now"], 1)
        self.assertEqual(info["swaps_now"], 1)
        self.assertEqual(env.completed_at["r0"], 2)

    def test_stop_commit_validates_whole_batch_before_consuming_any_token(self):
        requests = (
            RequestSpec("r0", (0, 1)),
            RequestSpec("r1", (2, 3)),
        )
        env, network = make_env(requests, {(0, 1): 1, (2, 3): 1})
        first_edge = network.edge_node_association[(0, 1)]
        second_edge = network.edge_node_association[(2, 3)]
        first_token = first_edge.links[0]

        env.step(action_for(env, "r0"))
        env.step(action_for(env, "r1"))
        second_edge.links.clear()  # Simulate an external invariant violation.

        with self.assertRaises(RuntimeError):
            env.step(env.stop_action)
        self.assertIn(first_token, first_edge.links)
        self.assertEqual(env.frontier, {"r0": 0, "r1": 2})
        self.assertEqual(env.completed_at, {})
        self.assertEqual(env.time, 0)
        self.assertEqual(network.env_steps, 0)

    def test_partial_output_is_not_reused_by_another_request(self):
        producer = RequestSpec("producer", (0, 1, 2, 3))
        consumer = RequestSpec("consumer", (0, 2))
        env, network = make_env(
            (producer, consumer),
            {(0, 1): 1, (1, 2): 1, (2, 3): 0, (0, 2): 0},
        )
        # Dynamic resource-graph routing may offer consumer the physical
        # 0--1--2 path even though its canonical shortest path is 0--2.
        self.assertTrue(actions_for(env, "consumer"))

        env.step(action_for(env, "producer"))
        self.assertTrue(all(
            not env.action_mask()[action]
            for action in actions_for(env, "consumer")
        ))
        _, _, terminated, truncated, _ = env.step(env.stop_action)

        self.assertFalse(terminated or truncated)
        self.assertEqual(env.frontier["consumer"], 0)
        self.assertEqual(actions_for(env, "consumer"), [])
        self.assertEqual(len(network.edge_node_association[(0, 2)].links), 0)
        self.assertEqual(env.carried_links["producer"].owner_request_id, "producer")

    def test_node_capacity_counts_each_selected_plan_not_unique_nodes(self):
        requests = (
            RequestSpec("r0", (0, 1, 2)),
            RequestSpec("r1", (3, 1, 4)),
            RequestSpec("r2", (5, 1, 6)),
        )
        inventory = {
            (0, 1): 1, (1, 2): 1,
            (1, 3): 1, (1, 4): 1,
            (1, 5): 1, (1, 6): 1,
        }
        env, _ = make_env(requests, inventory, node_capacity=2)

        env.step(action_for(env, "r0"))
        self.assertTrue(env.action_mask()[action_for(env, "r1")])
        env.step(action_for(env, "r1"))
        self.assertFalse(env.action_mask()[action_for(env, "r2")])
        self.assertEqual(env.node_load[1], 2)

    def test_one_hop_completion_takes_one_physical_subslot(self):
        request = RequestSpec("r0", (0, 1))
        env, network = make_env((request,), {(0, 1): 1})

        env.step(action_for(env, "r0"))
        _, _, terminated, truncated, info = env.step(env.stop_action)

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["duration"], 1)
        self.assertEqual(info["swaps_now"], 0)
        self.assertEqual(info["elementary_now"], 1)
        self.assertEqual(info["time"], 1)
        self.assertEqual(env.completed_at["r0"], 1)
        self.assertEqual(network.env_steps, 1)

    def test_noop_stop_advances_network_and_makes_new_epr_visible(self):
        request = RequestSpec("r0", (0, 1))
        env, network = make_env(
            (request,), {(0, 1): 0}, arrivals={1: ((0, 1),)}
        )
        self.assertEqual(actions_for(env, "r0"), [])

        _, _, terminated, truncated, info = env.step(env.stop_action)

        self.assertFalse(terminated or truncated)
        self.assertEqual(info["duration"], 1)
        self.assertEqual(info["time"], 1)
        self.assertEqual(network.pre_step_calls, 1)
        self.assertEqual(network.step_calls, 1)
        self.assertEqual(len(network.edge_node_association[(0, 1)].links), 1)
        self.assertTrue(env.action_mask()[action_for(env, "r0")])

    def test_eprs_generated_during_swap_depth_are_retained(self):
        request = RequestSpec("r0", tuple(range(9)))
        inventory = {(i, i + 1): 1 for i in range(8)}
        inventory[(9, 10)] = 0
        arrivals = {time: ((9, 10),) for time in (1, 2, 3)}
        env, network = make_env((request,), inventory, arrivals=arrivals)

        env.step(action_for(env, "r0"))
        _, _, terminated, truncated, info = env.step(env.stop_action)

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["duration"], math.ceil(math.log2(8)))
        self.assertEqual(network.pre_step_calls, 3)
        self.assertEqual(network.step_calls, 3)
        generated = network.edge_node_association[(9, 10)].links
        self.assertEqual(len(generated), 3)
        self.assertEqual({link.creation for link in generated}, {1, 2, 3})
        self.assertEqual(len({link.token_id for link in generated}), 3)

    def test_truncation_keeps_the_pending_request_and_owned_state(self):
        request = RequestSpec("r0", (0, 1))
        env, network = make_env(
            (request,), {(0, 1): 0}, max_subslots=2
        )

        _, _, terminated, truncated, _ = env.step(env.stop_action)
        self.assertFalse(terminated or truncated)
        _, _, terminated, truncated, info = env.step(env.stop_action)

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["active"], 1)
        self.assertEqual(info["completed"], 0)
        self.assertEqual(env.frontier["r0"], 0)
        self.assertIsNone(env.carried_links["r0"])
        self.assertEqual(env.instance.requests, (request,))
        self.assertEqual(network.env_steps, 2)


if __name__ == "__main__":
    unittest.main()
