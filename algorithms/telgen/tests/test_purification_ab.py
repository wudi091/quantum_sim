import csv
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from algorithms.telgen import (
    run_purification_ab,
    save_purification_ab_report,
    solve_teacher_episode,
)
from algorithms.telgen.evaluate_purification_ab import (
    _resolve_required_fidelities,
    _source_snapshot,
)
from qnet_core.spec import EpisodeSpec, PhysicalConfig, RequestSpec


class PurificationABTests(unittest.TestCase):
    def test_mixed_fidelity_pattern_cycles_over_requests(self):
        self.assertEqual(
            _resolve_required_fidelities(4, 0.7, "0.60,0.69"),
            (0.60, 0.69, 0.60, 0.69),
        )

    def test_invalid_fidelity_pattern_is_rejected(self):
        with self.assertRaises(ValueError):
            _resolve_required_fidelities(4, 0.7, "0.60,,0.69")

    def test_source_snapshot_is_deterministic_and_covers_runtime_code(self):
        first = _source_snapshot()
        second = _source_snapshot()

        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 64)
        paths = {path for path, _ in first[1]}
        self.assertIn("algorithms/__init__.py", paths)
        self.assertIn("algorithms/telgen/purification_ab.py", paths)
        self.assertIn("qnet_core/sequence_backend.py", paths)
        self.assertIn("environment.yml", paths)

    @staticmethod
    def _episode() -> EpisodeSpec:
        return EpisodeSpec(
            seed=2,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec(
                "r", 0, 1, required_fidelity=0.82
            ),),
            horizon=8,
            physical=PhysicalConfig(
                initial_fidelity=0.8,
                swap_degradation=1.0,
                generation_probability=1.0,
                swap_probability=1.0,
                memory_capacity=2,
                node_memory_capacity=2,
                memory_lifetime=1000,
                quantum_distance_m=1.0,
            ),
        )

    def test_ab_report_compares_the_same_physical_seed(self):
        report = run_purification_ab(
            self._episode(),
            physical_seeds=(2,),
            path_candidate_count=1,
            construction_kinds=("balanced",),
            decoder_random_restarts=0,
            code_revision="test-revision",
            working_directory="test-directory",
            run_command=("python", "test"),
            source_tree_sha256="abc123",
            source_file_hashes=(("file.py", "hash"),),
        )

        self.assertEqual(report.physical_seeds, (2,))
        self.assertEqual(
            tuple(item.seed for item in report.baseline.trials),
            tuple(item.seed for item in report.on_demand.trials),
        )
        self.assertEqual(report.baseline.planned_selected_requests, 0)
        self.assertIsNone(report.baseline.mean_completion_retention)
        self.assertEqual(report.baseline.mean_censored_latency_slots, 8.0)
        self.assertEqual(report.on_demand.planned_selected_requests, 1)
        self.assertEqual(report.on_demand.planned_purified_requests, 1)
        self.assertEqual(report.on_demand.planned_purified_request_ids, ("r",))
        self.assertEqual(report.on_demand.mean_completed_requests, 1.0)
        self.assertEqual(report.on_demand.purification_success_rate, 1.0)
        self.assertEqual(report.provenance.episode, self._episode())
        self.assertEqual(report.provenance.code_revision, "test-revision")
        self.assertEqual(
            report.provenance.working_directory,
            "test-directory",
        )
        self.assertEqual(report.provenance.run_command, ("python", "test"))
        self.assertEqual(report.provenance.source_tree_sha256, "abc123")
        self.assertEqual(
            report.provenance.source_file_hashes,
            (("file.py", "hash"),),
        )
        self.assertIn("not strict common", report.provenance.pairing_semantics)
        self.assertIn("single-episode sanity", report.provenance.evidence_scope)
        self.assertTrue(report.provenance.on_demand_scope.startswith(
            "candidate-level"
        ))

        repeated = run_purification_ab(
            self._episode(),
            physical_seeds=(2,),
            path_candidate_count=1,
            construction_kinds=("balanced",),
            decoder_random_restarts=0,
            code_revision="test-revision",
            working_directory="test-directory",
            run_command=("python", "test"),
            source_tree_sha256="abc123",
            source_file_hashes=(("file.py", "hash"),),
        )
        self.assertEqual(asdict(repeated), asdict(report))

    @staticmethod
    def _mixed_episode() -> EpisodeSpec:
        return EpisodeSpec(
            seed=7,
            nodes=tuple(range(12)),
            edges=(
                (0, 1), (1, 2),
                (3, 4), (4, 5),
                (6, 7), (7, 8),
                (9, 10), (10, 11),
            ),
            requests=(
                RequestSpec("r0", 0, 2, required_fidelity=0.60),
                RequestSpec("r1", 3, 5, required_fidelity=0.69),
                RequestSpec("r2", 6, 8, required_fidelity=0.60),
                RequestSpec("r3", 9, 11, required_fidelity=0.69),
            ),
            horizon=8,
            physical=PhysicalConfig(
                initial_fidelity=0.8,
                swap_degradation=1.0,
                generation_probability=1.0,
                swap_probability=1.0,
                memory_capacity=2,
                node_memory_capacity=4,
                memory_lifetime=1000,
                quantum_distance_m=1.0,
            ),
        )

    def test_mixed_thresholds_apply_candidate_level_purification(self):
        report = run_purification_ab(
            self._mixed_episode(),
            physical_seeds=(7,),
            path_candidate_count=1,
            construction_kinds=("balanced",),
            decoder_random_restarts=0,
        )

        self.assertEqual(report.baseline.planned_selected_requests, 2)
        self.assertEqual(report.on_demand.planned_selected_requests, 4)
        self.assertEqual(report.on_demand.planned_purified_requests, 2)
        self.assertEqual(
            report.on_demand.planned_purified_request_ids,
            ("r1", "r3"),
        )
        self.assertIn(
            ("purification_unnecessary", 2),
            report.on_demand.rejection_counts,
        )
        self.assertEqual(
            report.request_required_fidelities,
            (("r0", 0.60), ("r1", 0.69), ("r2", 0.60), ("r3", 0.69)),
        )
        delta_names = dict(report.deltas)
        self.assertIn("mean_successful_latency_slots", delta_names)
        self.assertIn("mean_completion_retention", delta_names)
        self.assertIn("mean_physical_failures", delta_names)
        self.assertIn("mean_memory_time_unit_slots", delta_names)
        self.assertIn(
            "pooled_memory_time_per_completed_request_slots",
            delta_names,
        )
        expected_memory_per_completion = (
            sum(
                trial.memory_time_unit_slots
                for trial in report.on_demand.trials
            )
            / sum(
                trial.completed_requests
                for trial in report.on_demand.trials
            )
        )
        self.assertAlmostEqual(
            report.on_demand.pooled_memory_time_per_completed_request_slots,
            expected_memory_per_completion,
        )
        self.assertAlmostEqual(
            delta_names["pooled_memory_time_per_completed_request_slots"],
            report.on_demand.pooled_memory_time_per_completed_request_slots
            - report.baseline.pooled_memory_time_per_completed_request_slots,
        )

    def test_candidate_level_gate_keeps_purification_for_a_harder_path(self):
        episode = EpisodeSpec(
            seed=11,
            nodes=(0, 1, 2),
            edges=((0, 2), (0, 1), (1, 2)),
            requests=(RequestSpec(
                "r", 0, 2, required_fidelity=0.69
            ),),
            horizon=8,
            physical=PhysicalConfig(
                initial_fidelity=0.8,
                swap_degradation=1.0,
                generation_probability=1.0,
                swap_probability=1.0,
                memory_capacity=2,
                node_memory_capacity=4,
                memory_lifetime=1000,
                quantum_distance_m=1.0,
            ),
        )
        record = solve_teacher_episode(
            episode,
            path_candidate_count=2,
            construction_kinds=("balanced",),
            purification_kinds=("none", "elementary_once"),
        )

        feasible_route_modes = {
            (variable.route_nodes, variable.purification_kind)
            for variable in record.expansion.variables
        }
        self.assertIn(((0, 2), "none"), feasible_route_modes)
        self.assertNotIn(((0, 2), "elementary_once"), feasible_route_modes)
        self.assertNotIn(((0, 1, 2), "none"), feasible_route_modes)
        self.assertIn(
            ((0, 1, 2), "elementary_once"),
            feasible_route_modes,
        )

    def test_report_saves_timestamped_and_latest_json_csv(self):
        report = run_purification_ab(
            self._episode(),
            physical_seeds=(2,),
            path_candidate_count=1,
            construction_kinds=("balanced",),
            decoder_random_restarts=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = save_purification_ab_report(report, directory)

            self.assertTrue(paths.timestamped_json.exists())
            self.assertTrue(paths.latest_json.exists())
            self.assertTrue(paths.timestamped_csv.exists())
            self.assertTrue(paths.latest_csv.exists())
            payload = json.loads(
                Path(paths.latest_json).read_text(encoding="utf-8")
            )
            self.assertEqual(payload["episode_seed"], 2)
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(
                payload["request_required_fidelities"],
                [["r", 0.82]],
            )
            self.assertEqual(
                payload["on_demand"]["planned_purified_requests"],
                1,
            )
            with paths.latest_csv.open(
                newline="",
                encoding="utf-8-sig",
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[0]["row_type"], "trial")
            self.assertEqual(rows[1]["row_type"], "trial")
            self.assertEqual(
                rows[0]["request_required_fidelities"],
                '[["r", 0.82]]',
            )
            self.assertEqual(
                rows[1]["planned_purified_request_ids"],
                '["r"]',
            )
            provenance = json.loads(rows[0]["provenance"])
            self.assertEqual(provenance["episode"]["seed"], 2)
            self.assertEqual(provenance["path_candidate_count"], 1)
            self.assertEqual(
                rows[0]["fidelity_model"],
                payload["baseline"]["fidelity_model"],
            )
            aggregate = next(
                row for row in rows
                if row["row_type"] == "aggregate"
                and row["variant"] == "on_demand_elementary_once"
            )
            self.assertEqual(
                float(aggregate["aggregate_mean_completed_requests"]),
                report.on_demand.mean_completed_requests,
            )
            delta = next(row for row in rows if row["row_type"] == "delta")
            self.assertEqual(
                json.loads(delta["delta_metrics"]),
                dict(report.deltas),
            )

    def test_report_versioning_avoids_same_second_collisions(self):
        report = run_purification_ab(
            self._episode(),
            physical_seeds=(2,),
            path_candidate_count=1,
            construction_kinds=("balanced",),
            decoder_random_restarts=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch("algorithms.telgen.purification_ab.datetime") as clock:
                clock.now.return_value.strftime.return_value = "20260811_120000"
                first = save_purification_ab_report(report, directory)
                second = save_purification_ab_report(report, directory)

            self.assertNotEqual(first.timestamped_json, second.timestamped_json)
            self.assertTrue(second.timestamped_json.name.endswith("_2.json"))
            self.assertTrue(second.timestamped_csv.name.endswith("_2.csv"))


if __name__ == "__main__":
    unittest.main()
