import unittest
from dataclasses import replace
from construction.cpsat import solve_cpsat
from construction.enumerator import (
    balanced_plan,
    enumerate_constructions,
    sequential_plan,
)
from construction.plan import ConstructionPlan
from construction.planners import greedy_select
from construction.simulator import SlotSimulator, plan_footprint, simulate_plan


class ConstructionKernelTests(unittest.TestCase):
    def setUp(self):
        self.r1_path = (0, 1, 2, 3, 4)
        self.r2_path = (5, 2, 6)
        self.capacity = {node: 2 for node in range(7)}

    def test_same_path_has_seq_bal_and_mid_constructions(self):
        plans = enumerate_constructions(self.r1_path)
        self.assertEqual({plan.kind for plan in plans}, {"seq", "bal", "mid"})
        self.assertTrue(all(plan.is_complete for plan in plans))
        self.assertTrue(all(plan.path == self.r1_path for plan in plans))

    def test_seq_and_bal_have_different_c_footprints(self):
        seq = plan_footprint(simulate_plan(sequential_plan(self.r1_path)))
        bal = plan_footprint(simulate_plan(balanced_plan(self.r1_path)))
        self.assertEqual(seq["series"][2], [0, 1, 1, 0])
        self.assertEqual(bal["series"][2], [0, 2])
        self.assertEqual(seq["peak"][2], 1)
        self.assertEqual(bal["peak"][2], 2)
        self.assertGreater(seq["makespan"], bal["makespan"])

    def test_non_monotonic_node_ids_preserve_path_endpoints(self):
        for plan in enumerate_constructions(self.r2_path):
            execution = simulate_plan(plan)
            self.assertEqual(execution.output_span, (5, 6))
            self.assertTrue(plan.is_complete)

    def test_con_md_counterexample(self):
        seq_slot = SlotSimulator(self.capacity)
        seq = sequential_plan(self.r1_path)
        self.assertTrue(seq_slot.admit(seq, simulate_plan(seq)))
        r2 = greedy_select(seq_slot, [self.r2_path])
        self.assertIsNotNone(r2)
        self.assertEqual(r2.kind, "seq")

        bal_slot = SlotSimulator(self.capacity)
        bal = balanced_plan(self.r1_path)
        self.assertTrue(bal_slot.admit(bal, simulate_plan(bal)))
        self.assertIsNone(greedy_select(bal_slot, [self.r2_path]))

    def test_cpsat_maximizes_batch_completions(self):
        candidates = {
            "r1": enumerate_constructions(self.r1_path),
            "r2": enumerate_constructions(self.r2_path),
        }
        result = solve_cpsat(candidates, self.capacity)
        self.assertEqual(result.completed_requests, 2)
        self.assertEqual(set(result.plans), {"r1", "r2"})
        self.assertEqual(result.plans["r1"].kind, "seq")
        self.assertEqual(result.plans["r2"].kind, "seq")

    def test_single_edge_reverse_path_and_invalid_pair_reuse(self):
        single = sequential_plan((5, 2))
        execution = simulate_plan(single)
        self.assertTrue(single.is_complete)
        self.assertEqual(execution.output_span, (5, 2))
        self.assertEqual(execution.swap_depth, 0)

        plan = sequential_plan((0, 1, 2, 3))
        bad_last = replace(plan.swap_tree[-1], right_ref=plan.swap_tree[0].right_ref)
        invalid = replace(plan, swap_tree=plan.swap_tree[:-1] + (bad_last,))
        with self.assertRaisesRegex(ValueError, "reuses a consumed input"):
            simulate_plan(invalid)

    def test_rejects_off_path_generation_and_forged_swap_span(self):
        off_path = ConstructionPlan(
            path=(0, 2),
            kind="seq",
            gen_layers=(((0, 3),),),
            swap_tree=(),
        )
        with self.assertRaisesRegex(ValueError, "plan edges do not match path"):
            simulate_plan(off_path)

        valid = sequential_plan((0, 1, 2))
        duplicated = replace(
            valid, elementary_pairs=valid.elementary_pairs + (valid.elementary_pairs[0],)
        )
        with self.assertRaisesRegex(ValueError, "duplicate pair refs"):
            simulate_plan(duplicated)

        plan = sequential_plan((0, 1, 2, 3))
        forged_first = replace(plan.swap_tree[0], span=(0, 3))
        forged = replace(plan, swap_tree=(forged_first,) + plan.swap_tree[1:])
        with self.assertRaisesRegex(ValueError, "declares span"):
            simulate_plan(forged)

    def test_cpsat_requires_an_optimality_proof(self):
        candidates = {"r1": enumerate_constructions(self.r1_path)}
        with self.assertRaisesRegex(TimeoutError, "did not prove optimality"):
            solve_cpsat(candidates, self.capacity, time_limit_seconds=0.0)


if __name__ == "__main__":
    unittest.main()
