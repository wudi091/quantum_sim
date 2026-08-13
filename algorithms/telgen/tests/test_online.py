import unittest

import json
from tempfile import TemporaryDirectory

import numpy as np
import torch

from algorithms.telgen import (
    OnlineTELGENConfig,
    OnlineTELGENController,
    generate_online_milp_dataset,
    load_online_milp_dataset,
    load_online_milp_graph_sample,
    run_online_telgen,
    save_online_milp_dataset,
)
from algorithms.telgen.hard_decoder import validate_decoded_selection
from algorithms.telgen.milp_imitation import CONSTRAINT_FEATURE_NAMES
from algorithms.telgen.milp_imitation import (
    GLOBAL_FEATURE_NAMES,
    VARIABLE_FEATURE_NAMES,
    CandidateConstraintGNN,
)
from algorithms.telgen.gnn_policy import OnlineGNNPolicy
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
    def _save_gnn_checkpoint(path, *, decode_threshold=0.0):
        torch.manual_seed(7)
        model = CandidateConstraintGNN(hidden_dim=8, layers=1)
        torch.save({
            "schema_version": 1,
            "model_class": "CandidateConstraintGNN",
            "model_config": {"hidden_dim": 8, "layers": 1},
            "state_dict": model.state_dict(),
            "decode_threshold": decode_threshold,
            "feature_schema": {
                "variable": list(VARIABLE_FEATURE_NAMES),
                "constraint": list(CONSTRAINT_FEATURE_NAMES),
                "global": list(GLOBAL_FEATURE_NAMES),
            },
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
        self.assertEqual(first.decoded_request_count, 1)
        self.assertAlmostEqual(first.decoded_expected_completed_mass, 1.0)
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
        _, decoded = controller._solve_decision(
            2,
            14,
            ("a_new",),
            reserved,
        )
        self.assertEqual(decoded.completed_request_count, 0)
        self.assertEqual(decoded.selected_variables, ())

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

    def test_milp_rollout_hides_future_requests_and_executes_its_label(self):
        spec = EpisodeSpec(
            seed=80,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(
                RequestSpec("now", 0, 1, arrival=0, ttl=6),
                RequestSpec("future", 0, 1, arrival=2, ttl=4),
            ),
            horizon=6,
            physical=deterministic_physical(),
        )
        result = run_online_telgen(spec, self.milp_config())

        first = result.milp_samples[0]
        self.assertEqual(first.eligible_request_ids, ("now",))
        self.assertEqual(first.graph.request_ids, ("now",))
        self.assertTrue(all(
            variable.request_id == "now"
            for variable in first.graph.variables
        ))
        self.assertNotIn("future", first.visible_request_ids)
        selected = {
            variable.variable_id
            for variable, label in zip(
                first.graph.variables, first.graph.labels
            )
            if label > 0.5
        }
        self.assertEqual(selected, set(first.selected_variable_ids))
        attempts_at_first_boundary = {
            attempt.variable_id
            for attempt in result.attempts
            if attempt.decision_slot == first.decision_slot
        }
        self.assertEqual(attempts_at_first_boundary, selected)

    def test_milp_graph_encodes_running_reservations_as_residual_capacity(self):
        spec = EpisodeSpec(
            seed=81,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(
                RequestSpec("old", 0, 2, arrival=0, ttl=8),
                RequestSpec("new", 0, 2, arrival=1, ttl=7),
            ),
            horizon=8,
            physical=deterministic_physical(
                memory_capacity=2,
                node_memory_capacity=4,
            ),
        )
        result = run_online_telgen(spec, self.milp_config())
        sample = next(
            item for item in result.milp_samples
            if item.decision_slot == 1
        )
        self.assertEqual(sample.running_request_ids, ("old",))
        self.assertTrue(sample.reserved_usage)
        self.assertEqual(sample.graph.reserved_usage, {
            (resource_id, slot): amount
            for resource_id, slot, amount in sample.reserved_usage
        })
        names = {name: index for index, name in enumerate(
            CONSTRAINT_FEATURE_NAMES
        )}
        resource_rows = sample.graph.constraint_features[
            :, names["is_resource_time"]
        ] > 0.5
        self.assertTrue(np.any(
            sample.graph.constraint_features[
                resource_rows, names["reserved_amount"]
            ] > 0.0
        ))
        selected = tuple(
            variable
            for variable, label in zip(
                sample.graph.variables, sample.graph.labels
            )
            if label > 0.5
        )
        self.assertTrue(validate_decoded_selection(
            selected,
            sample.graph.resource_capacities,
            sample.graph.reserved_usage,
        ).feasible)

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
        retry_sample = next(
            sample for sample in result.milp_samples
            if dict(sample.attempt_counts).get("r0") == 1
        )
        self.assertIn("r0", retry_sample.eligible_request_ids)
        self.assertEqual(result.metrics["retry_count"], 1.0)

    def test_saved_milp_dataset_contains_only_neutral_pre_action_data(self):
        spec = EpisodeSpec(
            seed=82,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r0", 0, 1, arrival=0, ttl=4),),
            horizon=4,
            physical=deterministic_physical(),
        )
        result = run_online_telgen(spec, self.milp_config())
        with TemporaryDirectory() as directory:
            paths = save_online_milp_dataset(result, directory)
            manifest = json.loads(paths.manifest_path.read_text(
                encoding="utf-8"
            ))
            self.assertEqual(manifest["sample_count"], len(paths.sample_paths))
            self.assertIn("runtime_versions", manifest)
            self.assertIn("feature_schema", manifest)
            latest_manifest = json.loads(
                (paths.manifest_path.parent.parent / "manifest.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(
                latest_manifest["version_directory"],
                paths.manifest_path.parent.name,
            )
            self.assertNotIn("events", manifest)
            with np.load(paths.sample_paths[0]) as payload:
                context = json.loads(str(payload["context_json"].item()))
                self.assertNotIn("events", context)
                self.assertNotIn("settlements", context)
                self.assertNotIn("sequence", json.dumps(context).lower())
                self.assertTrue(np.array_equal(
                    payload["labels"], result.milp_samples[0].graph.labels
                ))
            loaded_sample = load_online_milp_graph_sample(
                paths.sample_paths[0]
            )
            expected_sample = result.milp_samples[0].graph
            self.assertEqual(
                tuple(item.variable_id for item in loaded_sample.variables),
                tuple(item.variable_id for item in expected_sample.variables),
            )
            self.assertTrue(np.array_equal(
                loaded_sample.variable_features,
                expected_sample.variable_features,
            ))
            loaded_dataset = load_online_milp_dataset(directory)
            self.assertEqual(
                len(loaded_dataset.samples),
                len(result.milp_samples),
            )
            self.assertEqual(loaded_dataset.episode_seeds, (82,))

    def test_dataset_helper_runs_the_teacher_policy_rollout(self):
        spec = EpisodeSpec(
            seed=83,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r0", 0, 1, arrival=0, ttl=3),),
            horizon=3,
            physical=deterministic_physical(),
        )
        with TemporaryDirectory() as directory:
            result, paths = generate_online_milp_dataset(
                spec,
                directory,
                self.milp_config(),
            )
            self.assertEqual(result.config.decision_backend, "milp_teacher")
            self.assertTrue(paths.manifest_path.exists())
            self.assertTrue(paths.manifest_path.parent.name.startswith(
                "rollout_"
            ))
            self.assertEqual(len(paths.sample_paths), len(result.milp_samples))

    def test_all_infeasible_boundary_is_recorded_as_skipped(self):
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
        self.assertEqual(result.milp_samples, ())
        self.assertEqual(
            result.skipped_milp_boundaries[0].reason,
            "no_feasible_time_expanded_variables",
        )

    def test_gnn_checkpoint_rollout_submits_only_feasible_plans(self):
        spec = EpisodeSpec(
            seed=85,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(
                RequestSpec("r0", 0, 2, arrival=0, ttl=6),
                RequestSpec("r1", 0, 2, arrival=0, ttl=6),
            ),
            horizon=6,
            physical=deterministic_physical(
                memory_capacity=2,
                node_memory_capacity=4,
            ),
        )
        with TemporaryDirectory() as directory:
            checkpoint = f"{directory}/policy.pt"
            self._save_gnn_checkpoint(checkpoint)
            result = run_online_telgen(
                spec,
                OnlineTELGENConfig(
                    decision_interval=2,
                    path_candidate_count=1,
                    construction_kinds=("balanced",),
                    purification_kinds=("none",),
                    decision_backend="gnn",
                    gnn_checkpoint=checkpoint,
                    gnn_device="cpu",
                ),
            )
        self.assertEqual(result.config.decision_backend, "gnn")
        self.assertTrue(result.decisions)
        self.assertTrue(all(
            item.decision_backend == "gnn" for item in result.decisions
        ))
        self.assertTrue(any(
            item.decoder_search_strategy == "gnn_greedy_projection"
            for item in result.decisions
        ))
        self.assertEqual(result.metrics["schedule_violation_count"], 0.0)

    def test_gnn_policy_rejects_checkpoint_schema_mismatch(self):
        with TemporaryDirectory() as directory:
            checkpoint = f"{directory}/policy.pt"
            self._save_gnn_checkpoint(checkpoint)
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
            payload["feature_schema"]["global"] = ["wrong"]
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "feature schema"):
                OnlineGNNPolicy.from_checkpoint(
                    checkpoint, device="cpu"
                )


if __name__ == "__main__":
    unittest.main()
