from __future__ import annotations

from itertools import product
import unittest

from qnet_core.order_core import (
    OrderAwareBatchEnv,
    OrderBatchProblem,
    OrderCoreConfig,
    OrderLinkSpec,
    OrderPlan,
    OrderStoredPair,
)
from qnet_core.order_milp import (
    MilpReliableMemoryPathOrderPlanner,
    MilpReliableMemoryPathPlanner,
    compile_reliable_memory_candidate,
    reliable_binomial_capacity,
)


def _release_order_problem() -> OrderBatchProblem:
    candidates = (
        OrderPlan(
            "a:early",
            "a",
            (0, 1, 2, 3),
            (1, 2),
            swap_groups=((1,), (2,)),
            fixed_path_baseline=False,
        ),
        OrderPlan(
            "a:late",
            "a",
            (0, 1, 2, 3),
            (2, 1),
            swap_groups=((2,), (1,)),
            fixed_path_baseline=True,
        ),
        OrderPlan(
            "b:fixed",
            "b",
            (4, 1, 5),
            (1,),
            swap_groups=((1,),),
            fixed_path_baseline=True,
        ),
    )
    edges = tuple(sorted({
        elementary_edge
        for plan in candidates
        for elementary_edge in plan.elementary_edges
    }))
    return OrderBatchProblem.create(
        candidates=candidates,
        node_capacity={0: 1, 1: 2, 2: 2, 3: 1, 4: 1, 5: 1},
        links=tuple(
            OrderLinkSpec(
                *elementary_edge,
                capacity=1,
                generation_probability=1.0,
            )
            for elementary_edge in edges
        ),
        config=OrderCoreConfig(
            slot_duration_ps=2_000,
            generation_interval_ps=1_000,
            swap_service_ps=1_000,
            memory_reset_ps=0,
            generation_probability=1.0,
            swap_probability=1.0,
            bsm_capacity_per_node=2,
        ),
    )


def _inventory_direct_problem() -> OrderBatchProblem:
    candidates = (
        OrderPlan("r1", "r1", (0, 1), (), priority=0),
        OrderPlan("r2", "r2", (0, 1), (), priority=1),
        OrderPlan("r3", "r3", (1, 2), (), priority=2),
    )
    return OrderBatchProblem.create(
        candidates=candidates,
        node_capacity={0: 1, 1: 1, 2: 1},
        links=(
            OrderLinkSpec(0, 1, capacity=1, generation_probability=1.0),
            OrderLinkSpec(1, 2, capacity=1, generation_probability=1.0),
        ),
        initial_inventory=(
            OrderStoredPair("stored-01", 0, 1, 0, 3),
        ),
        config=OrderCoreConfig(
            slot_duration_ps=2_000,
            generation_interval_ps=1_000,
            swap_service_ps=1_000,
            memory_reset_ps=0,
            generation_probability=1.0,
            swap_probability=1.0,
            edge_capacity=1,
            epr_ttl_slots=3,
            slot_id=0,
        ),
    )


def _bruteforce_direct_inventory(problem: OrderBatchProblem):
    """Independent exhaustive checker for the two-tick direct motif."""

    horizon = 2
    inventory = problem.initial_inventory[0]
    plans = problem.candidates
    best_key = None
    best_metrics = None
    options = tuple((None, (plan, 0), (plan, 1)) for plan in plans)
    for selected in product(*options):
        chosen = tuple(value for value in selected if value is not None)
        if len({plan.request_id for plan, _ in chosen}) != len(chosen):
            continue
        sources = []
        for plan, start in chosen:
            elementary_edge = plan.elementary_edges[0]
            values = [("generated", tick) for tick in range(start + 1)]
            if elementary_edge == inventory.elementary_edge:
                values.append(("inventory", 0))
            sources.append(tuple(values))
        for assignment in product(*sources) if sources else ((),):
            inventory_uses = sum(
                source == "inventory" for source, _ in assignment
            )
            if inventory_uses > 1:
                continue
            generated_uses = {}
            for (plan, _), (source, birth) in zip(chosen, assignment):
                if source != "generated":
                    continue
                key = plan.elementary_edges[0], birth
                generated_uses[key] = generated_uses.get(key, 0) + 1
            if any(value > 1 for value in generated_uses.values()):
                continue

            memory_time = 0
            completion_time = 0
            feasible = True
            for tick in range(horizon):
                node_use = {0: 1, 1: 1, 2: 0}
                edge_use = {(0, 1): 1, (1, 2): 0}
                for (plan, start), (source, birth) in zip(
                    chosen, assignment
                ):
                    elementary_edge = plan.elementary_edges[0]
                    if source == "inventory" and tick >= start:
                        for node in elementary_edge:
                            node_use[node] -= 1
                        edge_use[elementary_edge] -= 1
                    if source == "generated" and birth <= tick < start:
                        for node in elementary_edge:
                            node_use[node] += 1
                        edge_use[elementary_edge] += 1
                    if tick == start:
                        for node in elementary_edge:
                            node_use[node] += 1
                        edge_use[elementary_edge] += 1
                if any(
                    node_use[node] > problem.capacity[node]
                    for node in node_use
                ) or any(
                    edge_use[edge] > problem.link_capacity(edge)
                    for edge in edge_use
                ):
                    feasible = False
                    break
                memory_time += sum(node_use.values())
            if not feasible:
                continue
            completion_time = sum(start + 1 for _, start in chosen)
            count = len(chosen)
            tertiary = memory_time * (horizon * len(plans) + 1) \
                + completion_time
            key = (-count, 0, tertiary)
            if best_key is None or key < best_key:
                best_key = key
                best_metrics = count, memory_time, completion_time
    assert best_metrics is not None
    return best_metrics


class ReliableMemoryMilpTests(unittest.TestCase):
    def test_binomial_reliable_capacity_matches_requested_example(self):
        self.assertEqual(reliable_binomial_capacity(4, 0.6, 0.9), 1)
        self.assertEqual(reliable_binomial_capacity(4, 0.6, 0.8), 2)

    def test_early_hotspot_release_admits_the_second_request(self):
        problem = _release_order_problem()
        early = compile_reliable_memory_candidate(
            problem, problem.candidates[0]
        )
        late = compile_reliable_memory_candidate(
            problem, problem.candidates[1]
        )

        self.assertEqual(early.memory_profile[1], (2, 0))
        self.assertEqual(late.memory_profile[1], (2, 2))

        snapshot = OrderAwareBatchEnv(problem).snapshot()
        fixed = MilpReliableMemoryPathPlanner()
        joint = MilpReliableMemoryPathOrderPlanner()
        fixed_ids = fixed.select(snapshot)
        joint_ids = joint.select(snapshot)

        self.assertEqual(fixed.last_objective, 1)
        self.assertEqual(joint.last_objective, 2)
        self.assertEqual(joint_ids, ("a:early", "b:fixed"))
        self.assertNotIn("a:early", fixed_ids)
        self.assertEqual(
            dict(joint.last_solution.scheduled_start_ticks),
            {"a:early": 0, "b:fixed": 1},
        )
        self.assertTrue(joint.last_solution.proven_optimal)
        self.assertFalse(joint.last_solution.certified_optimal)

    def test_inventory_and_reliable_births_match_bruteforce(self):
        problem = _inventory_direct_problem()
        expected_count, expected_memory, expected_completion = (
            _bruteforce_direct_inventory(problem)
        )
        planner = MilpReliableMemoryPathOrderPlanner()
        selected = planner.select(OrderAwareBatchEnv(problem).snapshot())
        solution = planner.last_solution

        self.assertEqual(len(selected), expected_count)
        self.assertEqual(solution.completed_count, expected_count)
        self.assertEqual(solution.memory_time_qubit_ticks, expected_memory)
        self.assertEqual(solution.completion_time_ticks, expected_completion)
        self.assertEqual(len(solution.inventory_assignments), 1)
        self.assertEqual(
            len({item[0] for item in solution.inventory_assignments}), 1
        )


if __name__ == "__main__":
    unittest.main()
