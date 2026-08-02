from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import random
import textwrap
import unittest

import qnet_core.order_core as order_core

from qnet_core.order_core import OrderStoredPair, edge
from qnet_core.order_waxman import (
    WaxmanOrderConfig,
    make_waxman_order_episode,
)


class _SlowScanExecution(order_core._OrderExecution):
    """Reference the pre-counter occupancy semantics directly from pairs."""

    def node_occupancy(self, node):
        pair_use = sum(
            node in pair.endpoints for pair in self.pairs.values()
        )
        return pair_use + len(self.resetting[node])

    def _elementary_edge_occupancy(self, elementary_edge):
        return sum(
            pair.elementary
            and edge(pair.left, pair.right) == elementary_edge
            for pair in self.pairs.values()
        )


class _InvariantExecution(order_core._OrderExecution):
    """Check the maintained counters after every pair-store mutation."""

    def _assert_counter_invariants(self) -> None:
        node_use = {node: 0 for node in self.capacity}
        edge_use = {
            elementary_edge: 0 for elementary_edge in self.links
        }
        for pair in self.pairs.values():
            node_use[pair.left] += 1
            node_use[pair.right] += 1
            if pair.elementary:
                edge_use[edge(pair.left, pair.right)] += 1
        if node_use != self._pair_use_by_node:
            raise AssertionError(
                f"node occupancy counter mismatch: {node_use} != "
                f"{self._pair_use_by_node}"
            )
        if edge_use != self._elementary_pair_use_by_edge:
            raise AssertionError(
                f"edge occupancy counter mismatch: {edge_use} != "
                f"{self._elementary_pair_use_by_edge}"
            )

    def _store_pair(self, pair):
        super()._store_pair(pair)
        self._assert_counter_invariants()

    def _remove_pair(self, pair_id):
        pair = super()._remove_pair(pair_id)
        self._assert_counter_invariants()
        return pair


class _PairMutationVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.functions: list[str] = []
        self.subscript_writes: list[str] = []
        self.mutating_calls: list[tuple[str, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    @staticmethod
    def _is_pairs_attribute(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "pairs"
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        )

    def _record_target(self, target: ast.AST) -> None:
        if (
            isinstance(target, ast.Subscript)
            and self._is_pairs_attribute(target.value)
        ):
            self.subscript_writes.append(self.functions[-1])

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._record_target(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._record_target(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._record_target(node.target)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self._record_target(target)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and self._is_pairs_attribute(function.value)
            and function.attr in {
                "clear", "pop", "popitem", "setdefault", "update"
            }
        ):
            self.mutating_calls.append(
                (self.functions[-1], function.attr)
            )
        self.generic_visit(node)


class OrderCoreOccupancyCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = WaxmanOrderConfig(
            node_count=10,
            average_degree=4,
            request_count=24,
            arrival_rate=4.0,
            episode_steps=6,
            request_ttl_slots=4,
            min_hops=2,
            max_hops=5,
            candidate_paths=2,
            order_variants_per_path=2,
            candidate_request_cap=None,
            node_memory_cap=3,
            slot_duration_ps=4_000,
            generation_interval_ps=1_000,
            swap_service_ps=1_000,
            memory_reset_ps=100,
            swap_probability=0.73,
            bsm_capacity_per_node=1,
            epr_ttl_slots=3,
        )
        episode = make_waxman_order_episode(config, seed=19)
        all_request_ids = tuple(
            request.request_id for request in episode.requests
        )
        slot = max(
            range(config.episode_steps),
            key=lambda value: len(
                episode.eligible_request_ids(all_request_ids, value)
            ),
        )
        request_ids = episode.eligible_request_ids(
            all_request_ids, slot
        )
        problem = episode.problem_for_slot(
            request_ids, slot=slot, physics_seed=0
        )

        inventory_edges = []
        for plan in problem.candidates:
            for elementary_edge in plan.elementary_edges:
                if elementary_edge not in inventory_edges:
                    inventory_edges.append(elementary_edge)
                if len(inventory_edges) == 2:
                    break
            if len(inventory_edges) == 2:
                break
        inventory = tuple(
            OrderStoredPair(
                pair_id=f"cached-inventory-{index}",
                left=elementary_edge[0],
                right=elementary_edge[1],
                born_slot=max(0, slot - 1),
                expires_slot=slot + 2,
            )
            for index, elementary_edge in enumerate(inventory_edges)
        )
        cls.problem = replace(problem, initial_inventory=inventory)
        cls.plan_lookup = {
            plan.plan_id: plan for plan in cls.problem.candidates
        }
        by_request = {}
        for plan in cls.problem.candidates:
            by_request.setdefault(plan.request_id, []).append(plan)
        cls.plans_by_request = tuple(
            (request_id, tuple(plans))
            for request_id, plans in by_request.items()
        )

        rng = random.Random(0xC0FFEE)
        assignments = []
        for _ in range(2_048):
            assignments.append(tuple(
                rng.choice(plans).plan_id
                for _, plans in cls.plans_by_request
                if rng.random() < 0.68
            ))
        cls.assignments = tuple(assignments)

    @classmethod
    def _run(cls, execution_type, problem, plan_ids, record_traces):
        plans = tuple(cls.plan_lookup[plan_id] for plan_id in plan_ids)
        return execution_type(
            problem, plans, record_traces=record_traces
        ).run()

    def test_immutable_derived_values_are_cached(self):
        plan = self.problem.candidates[0]
        capacity = self.problem.capacity
        links = self.problem.link_by_edge

        self.assertIs(plan.elementary_edges, plan.elementary_edges)
        self.assertIs(
            self.problem._capacity_cache, self.problem._capacity_cache
        )
        self.assertIs(
            self.problem._link_by_edge_cache,
            self.problem._link_by_edge_cache,
        )
        self.assertIs(self.problem.physical_edges, self.problem.physical_edges)

        # Public mapping access remains copy-on-read, so one planner cannot
        # corrupt the immutable snapshot observed by another planner.
        capacity[next(iter(capacity))] = -1
        links.clear()
        self.assertNotIn(-1, self.problem.capacity.values())
        self.assertTrue(self.problem.link_by_edge)

    def test_all_pair_mutations_are_routed_through_helpers(self):
        source = textwrap.dedent(
            inspect.getsource(order_core._OrderExecution)
        )
        visitor = _PairMutationVisitor()
        visitor.visit(ast.parse(source))

        self.assertEqual(visitor.subscript_writes, ["_store_pair"])
        self.assertEqual(visitor.mutating_calls, [("_remove_pair", "pop")])

    def test_thousands_of_random_assignments_match_slow_reference(self):
        for index, plan_ids in enumerate(self.assignments):
            problem = self.problem.with_physics_seed(index % 17)
            fast = self._run(
                order_core._OrderExecution,
                problem,
                plan_ids,
                record_traces=False,
            )
            slow = self._run(
                _SlowScanExecution,
                problem,
                plan_ids,
                record_traces=False,
            )
            with self.subTest(index=index, plan_ids=plan_ids):
                self.assertEqual(fast, slow)

    def test_traces_match_slow_reference(self):
        for index, plan_ids in enumerate(self.assignments[:64]):
            problem = self.problem.with_physics_seed(100 + index)
            fast = self._run(
                order_core._OrderExecution,
                problem,
                plan_ids,
                record_traces=True,
            )
            slow = self._run(
                _SlowScanExecution,
                problem,
                plan_ids,
                record_traces=True,
            )
            with self.subTest(index=index, plan_ids=plan_ids):
                self.assertEqual(fast, slow)

    def test_counters_match_pair_store_after_every_mutation(self):
        for index, plan_ids in enumerate(self.assignments[:256]):
            problem = self.problem.with_physics_seed(1_000 + index)
            execution = _InvariantExecution(
                problem,
                tuple(
                    self.plan_lookup[plan_id] for plan_id in plan_ids
                ),
                record_traces=(index < 16),
            )
            execution.run()
            with self.subTest(index=index, plan_ids=plan_ids):
                execution._assert_counter_invariants()


if __name__ == "__main__":
    unittest.main()
