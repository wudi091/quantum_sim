from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from qnet_core.order_paired_benchmark import (
    _parser,
    _print_slot_progress,
    _solution_diagnostics,
    run_paired_episode,
    run_suite,
)
from qnet_core.order_milp import (
    MilpNominalPathOrderPlanner,
    MilpNominalPathPlanner,
)
from qnet_core.order_waxman import (
    WaxmanOrderConfig,
    make_waxman_order_episode,
)


def _small_config() -> WaxmanOrderConfig:
    return WaxmanOrderConfig(
        node_count=10,
        average_degree=4,
        request_count=4,
        arrival_rate=4 / 3,
        episode_steps=3,
        request_ttl_slots=3,
        min_hops=2,
        max_hops=3,
        candidate_paths=2,
        order_variants_per_path=2,
        candidate_request_cap=2,
        node_memory_cap=2,
        slot_duration_ps=4_000,
        generation_interval_ps=1_000,
        swap_service_ps=1_000,
        memory_reset_ps=100,
        swap_probability=0.9,
        epr_ttl_slots=2,
    )


class PairedOrderBenchmarkTests(unittest.TestCase):
    def test_solution_diagnostics_are_safe_for_legacy_planners(self):
        planner = SimpleNamespace(
            last_evaluations=7,
            last_solution=SimpleNamespace(certified_optimal=True),
        )

        diagnostics = _solution_diagnostics(planner)

        self.assertEqual(diagnostics["evaluations"], 7)
        self.assertIsNone(diagnostics["backend"])
        self.assertIsNone(diagnostics["eligible_candidates"])
        self.assertIsNone(diagnostics["milp_solves"])

    def test_cli_defaults_to_formal_four_by_four_scope(self):
        args = _parser().parse_args([])

        self.assertEqual(args.candidate_paths, 4)
        self.assertEqual(args.order_variants, 4)
        self.assertIsNone(args.candidate_request_cap)
        self.assertEqual(args.oracle_workers, 1)
        self.assertFalse(args.progress)

    def test_cli_accepts_formal_parallel_oracle_worker_count(self):
        args = _parser().parse_args(["--oracle-workers", "4"])

        self.assertEqual(args.oracle_workers, 4)

    def test_progress_line_is_flushed_and_contains_slot_diagnostics(self):
        with patch("builtins.print") as mocked_print:
            _print_slot_progress(
                episode_seed=11,
                slot_id=3,
                eligible_request_count=5,
                fixed_objective=2,
                joint_objective=3,
                fixed_enumerated_assignments=7,
                joint_enumerated_assignments=13,
                fixed_ms=1.25,
                joint_ms=2.5,
            )

        line = mocked_print.call_args.args[0]
        self.assertIn("seed=11 slot=3", line)
        self.assertIn("eligible=5", line)
        self.assertIn("fixed_obj=2 joint_obj=3", line)
        self.assertIn("fixed_enum=7 joint_enum=13", line)
        self.assertIn("fixed_ms=1.250 joint_ms=2.500", line)
        self.assertTrue(mocked_print.call_args.kwargs["flush"])

    def test_uncapped_snapshot_solves_all_six_arrived_requests(self):
        config = WaxmanOrderConfig(
            node_count=10,
            average_degree=4,
            request_count=6,
            arrival_rate=6.0,
            episode_steps=1,
            request_ttl_slots=1,
            min_hops=2,
            max_hops=2,
            candidate_paths=1,
            order_variants_per_path=1,
            candidate_request_cap=None,
            node_memory_cap=2,
            slot_duration_ps=3_000,
            generation_interval_ps=1_000,
            swap_service_ps=1_000,
            memory_reset_ps=100,
            swap_probability=0.9,
            bsm_capacity_per_node=2,
            epr_ttl_slots=2,
        )

        result = run_suite(
            episodes=1,
            base_seed=3,
            physics_seed_base=910_000,
            planner_seed_base=920_000,
            config=config,
        )

        slot = result["rows"][0]["slots"][0]
        self.assertEqual(result["model"]["request_scope"], "all_active_pending")
        self.assertEqual(result["model"]["oracle_workers"], 1)
        self.assertEqual(slot["eligible_request_count"], 6)
        self.assertEqual(slot["considered_request_count"], 6)
        self.assertEqual(slot["pruned_request_count"], 0)
        self.assertEqual(slot["request_count"], 6)
        self.assertEqual(slot["candidate_count"], 6)
        self.assertTrue(slot["fixed_proven_optimal"])
        self.assertTrue(slot["joint_proven_optimal"])
        for prefix in ("fixed", "joint"):
            self.assertIn("cp-sat", slot[f"{prefix}_backend"])
            self.assertGreaterEqual(
                slot[f"{prefix}_eligible_candidates"],
                slot[f"{prefix}_static_upper_bound"],
            )
            self.assertGreaterEqual(
                slot[f"{prefix}_enumerated_assignments"], 1
            )
            self.assertEqual(slot[f"{prefix}_milp_solves"], 1)
            self.assertEqual(
                slot[f"{prefix}_evaluations"],
                slot[f"{prefix}_enumerated_assignments"],
            )

    def test_fixed_driver_compares_both_planners_on_every_snapshot(self):
        result = run_suite(
            episodes=1,
            base_seed=0,
            physics_seed_base=810_000,
            planner_seed_base=820_000,
            config=_small_config(),
        )

        self.assertEqual(result["model"]["driver"], "fixed")
        self.assertEqual(result["episode_count"], 1)
        row = result["rows"][0]
        self.assertEqual(row["slot_count"], 3)
        self.assertEqual(len(row["slots"]), 3)
        self.assertEqual(
            row["delta_sum"],
            row["joint_objective_sum"] - row["fixed_objective_sum"],
        )
        self.assertEqual(
            row["positive_gap_slots"],
            sum(slot["delta"] > 0 for slot in row["slots"]),
        )

        for slot in row["slots"]:
            self.assertGreaterEqual(
                slot["joint_objective"], slot["fixed_objective"]
            )
            self.assertEqual(slot["fixed_noncanonical_selected"], 0)
            self.assertGreaterEqual(
                slot["request_count"], slot["order_relevant_requests"]
            )
            if slot["decision_slot"]:
                self.assertEqual(len(slot["snapshot_hash"]), 64)
            else:
                self.assertIsNone(slot["snapshot_hash"])
            if slot["delta"] > 0:
                self.assertGreater(slot["joint_noncanonical_selected"], 0)
            self.assertEqual(
                tuple(slot["driver_selected_plan_ids"]),
                tuple(slot["fixed_selected_plan_ids"]),
            )
            if slot["decision_slot"]:
                self.assertTrue(slot["fixed_proven_optimal"])
                self.assertTrue(slot["joint_proven_optimal"])

        aggregate = result["aggregate"]
        self.assertEqual(aggregate["delta_sum"], row["delta_sum"])
        self.assertEqual(
            aggregate["positive_gap_slots"], row["positive_gap_slots"]
        )

    def test_paired_passes_fixed_solution_as_verified_joint_incumbent(self):
        episode = make_waxman_order_episode(_small_config(), seed=0)
        fixed = MilpNominalPathPlanner()
        joint = MilpNominalPathOrderPlanner()

        with patch.object(
            joint,
            "select_with_incumbent",
            wraps=joint.select_with_incumbent,
        ) as incumbent_select:
            result = run_paired_episode(
                episode,
                physics_seed_root=930_000,
                planner_seed=940_000,
                fixed_planner=fixed,
                joint_planner=joint,
            )

        decision_slots = tuple(
            slot for slot in result["slots"] if slot["decision_slot"]
        )
        self.assertEqual(incumbent_select.call_count, len(decision_slots))
        for call, slot in zip(
            incumbent_select.call_args_list, decision_slots, strict=True
        ):
            self.assertEqual(
                tuple(call.args[1]),
                tuple(slot["fixed_selected_plan_ids"]),
            )

    def test_paired_keeps_legacy_custom_joint_select_protocol(self):
        episode = make_waxman_order_episode(_small_config(), seed=0)
        inner = MilpNominalPathOrderPlanner()
        calls = 0
        legacy = SimpleNamespace(
            last_objective=0,
            last_solution=None,
            last_evaluations=0,
        )

        def reset(seed: int) -> None:
            inner.reset(seed)

        def select(snapshot):
            nonlocal calls
            calls += 1
            selected = inner.select(snapshot)
            legacy.last_objective = inner.last_objective
            legacy.last_solution = inner.last_solution
            legacy.last_evaluations = inner.last_evaluations
            return selected

        legacy.reset = reset
        legacy.select = select
        result = run_paired_episode(
            episode,
            physics_seed_root=950_000,
            planner_seed=960_000,
            joint_planner=legacy,
        )

        self.assertEqual(calls, result["decision_slot_count"])
        self.assertTrue(all(
            slot["joint_proven_optimal"]
            for slot in result["slots"] if slot["decision_slot"]
        ))

    def test_joint_driver_executes_joint_selection_without_changing_pairing(self):
        result = run_suite(
            episodes=1,
            base_seed=1,
            physics_seed_base=830_000,
            planner_seed_base=840_000,
            config=_small_config(),
            driver="joint",
        )

        row = result["rows"][0]
        self.assertEqual(row["driver"], "joint")
        for slot in row["slots"]:
            self.assertGreaterEqual(
                slot["joint_objective"], slot["fixed_objective"]
            )
            self.assertEqual(
                tuple(slot["driver_selected_plan_ids"]),
                tuple(slot["joint_selected_plan_ids"]),
            )

    def test_rejects_unknown_driver(self):
        with self.assertRaisesRegex(ValueError, "driver"):
            run_suite(
                episodes=1,
                config=_small_config(),
                driver="other",  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
