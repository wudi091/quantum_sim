import unittest

from tempfile import TemporaryDirectory

import torch

from algorithms.telgen import (
    OnlineTELGENConfig,
    OnlineTELGENController,
    run_online_telgen,
)
from algorithms.telgen.ipm_trajectory_pilot import TELGENPaperGNN
from qnet_core.planning_spec import RequestSpec
from qnet_core.spec import EpisodeSpec, PhysicalConfig


def deterministic_physical(**overrides):
    values = dict(
        generation_probability=1.0,
        swap_probability=1.0,
        detector_efficiency=1.0,
        bsm_success_probability=1.0,
        quantum_distance_m=1.0,
    )
    values.update(overrides)
    return PhysicalConfig(**values)


class OnlineTELGENTests(unittest.TestCase):
    @staticmethod
    def milp_config(**overrides):
        values = dict(
            decision_interval=1,
            path_candidate_count=1,
            construction_kinds=("balanced",),
            purification_kinds=("none",),
            decision_backend="milp_teacher",
            milp_time_limit_seconds=10.0,
        )
        values.update(overrides)
        return OnlineTELGENConfig(**values)

    @staticmethod
    def _save_ipm_gnn_checkpoint(path):
        torch.manual_seed(11)
        model = TELGENPaperGNN(
            hidden_dim=8,
            inner_layers=1,
            message_mlp_layers=1,
            prediction_layers=1,
        )
        torch.save({
            "schema_version": 3,
            "model_class": "TELGENPaperGNN",
            "method": "ipm_trajectory_with_shared_rounding",
            "model_config": {
                "hidden_dim": 8,
                "inner_layers": 1,
                "message_mlp_layers": 1,
                "prediction_layers": 1,
                "normalization": "layer",
                "dropout": 0.0,
            },
            "inference_steps": 2,
            "state_dict": model.state_dict(),
        }, path)

    def test_default_controller_uses_periodic_decisions(self):
        spec = EpisodeSpec(
            seed=19,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r0", 0, 1, arrival=0, ttl=8),),
            horizon=8,
            physical=deterministic_physical(),
        )
        result = run_online_telgen(spec)
        self.assertEqual(result.config.decision_interval, 4)
        self.assertEqual(
            [item.decision_slot for item in result.decisions],
            [0, 4],
        )
        self.assertEqual(
            [item.window_end_slot for item in result.decisions],
            [4, 8],
        )
        self.assertEqual(
            [item.completion_end_slot for item in result.decisions],
            [8, 8],
        )

    def test_ipm_gnn_checkpoint_runs_as_online_backend(self):
        spec = EpisodeSpec(
            seed=119,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r0", 0, 1, arrival=0, ttl=4),),
            horizon=4,
            physical=deterministic_physical(),
        )
        with TemporaryDirectory() as directory:
            checkpoint = f"{directory}/ipm.pt"
            self._save_ipm_gnn_checkpoint(checkpoint)
            result = run_online_telgen(
                spec,
                OnlineTELGENConfig(
                    decision_interval=2,
                    path_candidate_count=1,
                    construction_kinds=("balanced",),
                    purification_kinds=("none",),
                    decision_backend="ipm_gnn",
                    gnn_checkpoint=checkpoint,
                    gnn_device="cpu",
                ),
            )
        self.assertTrue(result.decisions)
        self.assertTrue(all(
            item.decision_backend == "ipm_gnn"
            for item in result.decisions
        ))
        self.assertEqual(result.metrics["gnn_invalid_decision_count"], 0.0)

    def test_decisions_are_periodic_and_future_requests_are_invisible(self):
        spec = EpisodeSpec(
            seed=20,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(
                RequestSpec("r0", 0, 1, arrival=0, ttl=8),
                RequestSpec("r1", 0, 1, arrival=3, ttl=5),
            ),
            horizon=8,
            physical=deterministic_physical(),
        )
        result = run_online_telgen(
            spec,
            OnlineTELGENConfig(
                decision_interval=4,
                path_candidate_count=1,
                construction_kinds=("balanced",),
                purification_kinds=("none",),
            ),
        )
        self.assertEqual(
            [item.decision_slot for item in result.decisions],
            [0, 4],
        )
        self.assertEqual(result.decisions[0].visible_request_ids, ("r0",))
        self.assertNotIn("r1", result.decisions[0].eligible_request_ids)
        self.assertEqual(result.metrics["completed_requests"], 2.0)

    def test_running_plan_is_fixed_without_double_counting_its_reservation(self):
        spec = EpisodeSpec(
            seed=21,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(
                RequestSpec("r0", 0, 2, arrival=0, ttl=6),
                RequestSpec("r1", 0, 2, arrival=1, ttl=5),
            ),
            horizon=6,
            physical=deterministic_physical(node_memory_capacity=4),
        )
        result = run_online_telgen(
            spec,
            OnlineTELGENConfig(
                decision_interval=1,
                path_candidate_count=1,
                construction_kinds=("balanced",),
                purification_kinds=("none",),
            ),
        )
        first = result.decisions[0]
        second = result.decisions[1]
        self.assertEqual(first.selected_request_count, 1)
        self.assertAlmostEqual(first.selected_expected_completed_mass, 1.0)
        self.assertEqual(second.running_request_ids, ("r0",))
        self.assertGreater(second.reserved_resource_slot_count, 0)
        r1_attempt = next(item for item in result.attempts if item.request_id == "r1")
        self.assertAlmostEqual(r1_attempt.expected_success_probability, 1.0)
        self.assertEqual(r1_attempt.planned_start_slot, 1)

    def test_physical_failure_releases_plan_and_retries_before_ttl(self):
        spec = EpisodeSpec(
            seed=2,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r0", 0, 1, arrival=0, ttl=12),),
            horizon=12,
            physical=deterministic_physical(generation_probability=0.5),
        )
        result = run_online_telgen(
            spec,
            OnlineTELGENConfig(
                decision_interval=2,
                path_candidate_count=1,
                construction_kinds=("balanced",),
                purification_kinds=("none",),
            ),
        )
        self.assertEqual(len(result.attempts), 2)
        self.assertFalse(result.attempts[0].success)
        self.assertEqual(result.attempts[0].failure_cause, "physical_failure")
        self.assertTrue(result.attempts[1].success)
        self.assertTrue(result.settlements[0].success)
        self.assertEqual(result.metrics["retry_count"], 1.0)

    def test_request_that_expires_between_decisions_is_never_planned(self):
        spec = EpisodeSpec(
            seed=22,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("late", 0, 1, arrival=1, ttl=1),),
            horizon=6,
            physical=deterministic_physical(),
        )
        result = run_online_telgen(
            spec,
            OnlineTELGENConfig(
                decision_interval=4,
                path_candidate_count=1,
                construction_kinds=("balanced",),
                purification_kinds=("none",),
            ),
        )
        self.assertEqual(result.attempts, ())
        self.assertFalse(result.settlements[0].success)
        self.assertEqual(result.settlements[0].settlement_time, 2_000_000)

    def test_overdue_fixed_swap_blocks_a_new_plan_on_the_same_bsm(self):
        spec = EpisodeSpec(
            seed=44,
            nodes=(0, 1, 2, 3, 4),
            edges=((0, 1), (1, 2), (3, 1), (1, 4)),
            requests=(
                RequestSpec(
                    "z_old",
                    0,
                    2,
                    arrival=0,
                    ttl=60,
                    required_fidelity=0.5,
                ),
                RequestSpec(
                    "a_new",
                    3,
                    4,
                    arrival=2,
                    ttl=58,
                    required_fidelity=0.5,
                ),
            ),
            horizon=60,
            physical=deterministic_physical(
                classical_delay_ps=5_000,
                slot_duration_ps=1_000,
                memory_capacity=1,
                node_memory_capacity=4,
                initial_fidelity=1.0,
                swap_degradation=1.0,
            ),
        )
        config = OnlineTELGENConfig(
            decision_interval=1,
            path_candidate_count=1,
            construction_kinds=("balanced",),
            purification_kinds=("none",),
        )
        controller = OnlineTELGENController(spec, config)
        controller._decision(0)
        controller._process_update(controller.scheduler.advance_to_slot(1))
        controller._decision(1)
        controller._process_update(controller.scheduler.advance_to_slot(2))

        overdue = tuple(controller.scheduler._due.values())
        self.assertEqual(len(overdue), 1)
        self.assertEqual(
            dict(overdue[0].resource_demand.items()),
            {
                "bsm:1": 1,
                "swapnode:0": 1,
                "swapnode:1": 1,
                "swapnode:2": 1,
            },
        )

        reserved = controller._reserved_usage(14)
        for resource_id in (
            "bsm:1",
            "swapnode:0",
            "swapnode:1",
            "swapnode:2",
        ):
            self.assertTrue(all(
                reserved.get((resource_id, slot)) == 1
                for slot in range(2, 14)
            ))
        problem = controller._build_decision_problem(
            2,
            14,
            ("a_new",),
            reserved,
        )
        solution, _ = controller._solve_milp_decision(problem)
        self.assertIsNone(solution)

    def test_plan_may_complete_after_the_next_decision_boundary(self):
        spec = EpisodeSpec(
            seed=45,
            nodes=(0, 1, 2, 3, 4),
            edges=((0, 1), (1, 2), (2, 3), (3, 4)),
            requests=(RequestSpec("r0", 0, 4, arrival=0, ttl=12),),
            horizon=12,
            physical=deterministic_physical(
                memory_capacity=2,
                node_memory_capacity=8,
            ),
        )
        result = run_online_telgen(
            spec,
            OnlineTELGENConfig(
                decision_interval=2,
                path_candidate_count=1,
                construction_kinds=("balanced",),
                purification_kinds=("none",),
            ),
        )

        attempt = result.attempts[0]
        self.assertLess(attempt.planned_start_slot, 2)
        self.assertGreater(attempt.planned_completion_slot, 2)
        self.assertEqual(result.decisions[0].completion_end_slot, 12)
        self.assertEqual(result.decisions[1].running_request_ids, ("r0",))
        self.assertGreater(
            result.decisions[1].reserved_resource_slot_count,
            0,
        )

    def test_deferred_request_is_reconsidered_at_the_next_boundary(self):
        spec = EpisodeSpec(
            seed=46,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(
                RequestSpec("a", 0, 1, arrival=0, ttl=8),
                RequestSpec("b", 0, 1, arrival=0, ttl=8),
            ),
            horizon=8,
            physical=deterministic_physical(
                memory_capacity=1,
                node_memory_capacity=2,
                max_width=1,
            ),
        )
        result = run_online_telgen(
            spec,
            OnlineTELGENConfig(
                decision_interval=1,
                path_candidate_count=1,
                construction_kinds=("balanced",),
                purification_kinds=("none",),
            ),
        )

        # The two requests are exactly symmetric, so different SciPy versions
        # may return either equivalent optimum.  The online contract is that
        # exactly one request is deferred and reconsidered at the next
        # boundary, not that a particular request ID wins the tie.
        self.assertEqual(len(result.decisions[0].deferred_request_ids), 1)
        deferred = result.decisions[0].deferred_request_ids[0]
        self.assertIn(deferred, result.decisions[1].eligible_request_ids)
        deferred_attempt = next(
            attempt for attempt in result.attempts
            if attempt.request_id == deferred
        )
        self.assertEqual(deferred_attempt.decision_slot, 1)
        self.assertEqual(
            {attempt.request_id for attempt in result.attempts},
            {"a", "b"},
        )

    def test_every_new_plan_starts_inside_its_decision_period(self):
        spec = EpisodeSpec(
            seed=47,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(
                RequestSpec("r0", 0, 2, arrival=0, ttl=10),
                RequestSpec("r1", 0, 2, arrival=2, ttl=8),
            ),
            horizon=10,
            physical=deterministic_physical(node_memory_capacity=4),
        )
        interval = 2
        result = run_online_telgen(
            spec,
            OnlineTELGENConfig(
                decision_interval=interval,
                path_candidate_count=1,
                construction_kinds=("balanced",),
                purification_kinds=("none",),
            ),
        )

        for attempt in result.attempts:
            self.assertLessEqual(attempt.decision_slot, attempt.planned_start_slot)
            self.assertLess(
                attempt.planned_start_slot,
                min(spec.horizon, attempt.decision_slot + interval),
            )

    def test_failed_request_reappears_as_retry_state(self):
        spec = EpisodeSpec(
            seed=2,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r0", 0, 1, arrival=0, ttl=12),),
            horizon=12,
            physical=deterministic_physical(generation_probability=0.5),
        )
        result = run_online_telgen(
            spec,
            self.milp_config(decision_interval=2),
        )
        self.assertGreaterEqual(len(result.attempts), 2)
        self.assertEqual(
            [attempt.attempt for attempt in result.attempts],
            [1, 2],
        )
        self.assertEqual(result.metrics["retry_count"], 1.0)
    def test_all_infeasible_boundary_records_no_attempt(self):
        spec = EpisodeSpec(
            seed=84,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec(
                "tight",
                0,
                2,
                arrival=0,
                ttl=1,
            ),),
            horizon=2,
            physical=deterministic_physical(node_memory_capacity=4),
        )
        result = run_online_telgen(spec, self.milp_config())
        self.assertTrue(result.decisions)
        self.assertEqual(result.metrics["construction_attempt_count"], 0.0)


if __name__ == "__main__":
    unittest.main()
