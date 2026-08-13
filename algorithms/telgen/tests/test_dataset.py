import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from algorithms.telgen import (
    FIDELITY_MODEL_NAME,
    SUCCESS_PROBABILITY_MODEL_NAME,
    build_teacher_batch_record,
    build_planning_batch_problem,
    generate_teacher_dataset,
    save_teacher_batch_record,
    solve_teacher_episode,
    solve_teacher_window,
)
from algorithms.telgen.dataset import (
    _canonicalize_planning_equivalent_candidates,
)
from algorithms.telgen.fidelity import candidate_fidelity_estimate_map
from algorithms.telgen.success_probability import (
    candidate_success_probability_map,
)
from qnet_core.construction_api import ConstructionDAG
from qnet_core.construction_catalog import (
    build_route_construction_catalogue,
)
from qnet_core.resource_catalog import build_resource_capacities
from qnet_core.scenario import ScenarioConfig
from qnet_core.spec import EpisodeSpec, PhysicalConfig, RequestSpec


def small_batch_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        request_count=3,
        min_hops=2,
        max_hops=3,
        ttl=8,
        horizon=8,
        topology_nodes=16,
    )


class TeacherDatasetTests(unittest.TestCase):
    def test_planning_problem_deduplicates_only_behavioral_aliases(self):
        short = EpisodeSpec(
            seed=1,
            nodes=(0, 1, 2, 3),
            edges=((0, 1), (1, 2), (2, 3)),
            requests=(RequestSpec("r", 0, 3, ttl=8),),
            horizon=8,
            physical=PhysicalConfig(
                memory_capacity=2,
                node_memory_capacity=8,
                quantum_distance_m=1.0,
            ),
        )
        deduplicated = build_planning_batch_problem(
            short,
            path_candidate_count=1,
            construction_kinds=("left_deep", "balanced"),
            purification_kinds=("none",),
        )
        self.assertEqual(len(deduplicated.candidates), 1)
        self.assertEqual(len(deduplicated.equivalent_candidate_aliases), 1)

        four_hop = EpisodeSpec(
            seed=2,
            nodes=(0, 1, 2, 3, 4),
            edges=((0, 1), (1, 2), (2, 3), (3, 4)),
            requests=(RequestSpec("r", 0, 4, ttl=8),),
            horizon=8,
            physical=PhysicalConfig(
                memory_capacity=2,
                node_memory_capacity=8,
                quantum_distance_m=1.0,
            ),
        )
        distinct = build_planning_batch_problem(
            four_hop,
            path_candidate_count=1,
            construction_kinds=("left_deep", "balanced"),
            purification_kinds=("none",),
        )
        self.assertEqual(
            {candidate.construction_kind for candidate in distinct.candidates},
            {"left_deep", "balanced"},
        )
        self.assertEqual(distinct.equivalent_candidate_aliases, ())

    def test_candidate_dedup_preserves_operation_level_physical_semantics(self):
        episode = EpisodeSpec(
            seed=3,
            nodes=(0, 1, 2, 3),
            edges=((0, 1), (1, 2), (2, 3)),
            requests=(RequestSpec("r", 0, 3, ttl=8),),
            horizon=8,
            physical=PhysicalConfig(
                memory_capacity=2,
                node_memory_capacity=8,
                quantum_distance_m=1.0,
            ),
        )
        base = build_route_construction_catalogue(
            episode.planning,
            candidate_count=1,
            construction_kinds=("balanced",),
            purification_kinds=("none",),
        )[0]
        first = base.dag.operations[0]
        variants = (
            replace(first, duration_ps=first.duration_ps + 1),
            replace(first, success_probability=0.5),
            replace(first, required_fidelity=1.0),
            replace(first, retry_limit=first.retry_limit + 1),
        )
        candidates = [base]
        for index, operation in enumerate(variants):
            candidates.append(replace(
                base,
                candidate_id=f"variant:{index}",
                dag=ConstructionDAG(
                    base.request_id,
                    (operation, *base.dag.operations[1:]),
                    version=base.dag.version,
                ),
            ))
        frozen = tuple(candidates)
        canonical, aliases = _canonicalize_planning_equivalent_candidates(
            frozen,
            candidate_fidelity_estimate_map(episode, frozen),
            candidate_success_probability_map(episode, frozen),
        )
        self.assertEqual(len(canonical), len(frozen))
        self.assertEqual(aliases, ())

        swap = base.dag.operations[-1]
        first_swap = base.dag.operations[-2]
        dataflow_variant = replace(
            base,
            candidate_id="dataflow-variant",
            dag=ConstructionDAG(
                base.request_id,
                (
                    base.dag.operations[0],
                    base.dag.operations[1],
                    base.dag.operations[2],
                    replace(
                        first_swap,
                        input_segment_ids=(
                            base.dag.operations[0].output_segment_id,
                            base.dag.operations[2].output_segment_id,
                        ),
                        predecessors=(
                            base.dag.operations[0].op_id,
                            base.dag.operations[2].op_id,
                        ),
                    ),
                    replace(
                        swap,
                        input_segment_ids=(
                            first_swap.output_segment_id,
                            base.dag.operations[1].output_segment_id,
                        ),
                        predecessors=(
                            first_swap.op_id,
                            base.dag.operations[1].op_id,
                        ),
                    ),
                ),
                version=base.dag.version,
            ),
        )
        retry_lineage_variant = replace(
            base,
            candidate_id="retry-lineage-variant",
            dag=ConstructionDAG(
                base.request_id,
                (*base.dag.operations[:-1], replace(
                    swap,
                    retry_root_id=base.dag.operations[0].op_id,
                )),
                version=base.dag.version,
            ),
        )
        terminal_variant = replace(
            base,
            candidate_id="terminal-variant",
            terminal_segment_ids=(
                base.terminal_segment_id,
                base.dag.operations[-2].output_segment_id,
            ),
        )
        structural = (
            base,
            dataflow_variant,
            retry_lineage_variant,
            terminal_variant,
        )
        canonical, aliases = _canonicalize_planning_equivalent_candidates(
            structural,
            {item.candidate_id: 1.0 for item in structural},
            {item.candidate_id: 1.0 for item in structural},
        )
        self.assertEqual(len(canonical), len(structural))
        self.assertEqual(aliases, ())

    def test_teacher_uses_purification_only_when_unpurified_fidelity_is_too_low(self):
        episode = EpisodeSpec(
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
                memory_capacity=2,
                node_memory_capacity=2,
                memory_lifetime=1000,
                quantum_distance_m=1.0,
            ),
        )

        record = solve_teacher_episode(
            episode,
            path_candidate_count=1,
            construction_kinds=("balanced",),
        )
        selected = [
            variable for variable, value in zip(
                record.solution.variables,
                record.solution.stage_two.primal,
            )
            if value > 1e-6
        ]

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].purification_kind, "elementary_once")
        self.assertIn(
            "fidelity",
            {item.reason for item in record.expansion.rejections},
        )

    def test_builds_and_solves_distributed_simultaneous_batch(self):
        record = build_teacher_batch_record(
            small_batch_scenario(),
            seed=42,
            path_candidate_count=2,
        )
        self.assertEqual(
            {request.arrival for request in record.episode.requests},
            {0},
        )
        self.assertGreater(
            len({
                (request.source, request.destination)
                for request in record.episode.requests
            }),
            1,
        )
        self.assertEqual(
            record.capacities,
            build_resource_capacities(record.episode),
        )
        self.assertTrue(record.candidates)
        self.assertTrue(record.expansion.variables)
        self.assertEqual(record.fidelity_model, FIDELITY_MODEL_NAME)
        self.assertEqual(
            record.success_probability_model,
            SUCCESS_PROBABILITY_MODEL_NAME,
        )
        self.assertTrue(all(
            variable.expected_fidelity is not None
            for variable in record.expansion.variables
        ))
        self.assertTrue(all(
            0.0 <= variable.expected_success_probability <= 1.0
            for variable in record.expansion.variables
        ))
        self.assertTrue(record.solution.stage_one.success)
        self.assertTrue(record.solution.stage_two.success)
        self.assertLess(
            record.solution.stage_two.max_violation_trajectory[-1],
            1e-7,
        )
        self.assertLessEqual(
            record.solution.completed_request_mass,
            len(record.episode.requests) + 1e-7,
        )

    def test_record_and_dataset_manifest_include_reproducible_context(self):
        scenario = small_batch_scenario()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = build_teacher_batch_record(
                scenario,
                seed=43,
                path_candidate_count=1,
            )
            single_path = save_teacher_batch_record(
                record,
                root / "single.npz",
            )
            with np.load(single_path) as payload:
                metadata = json.loads(str(payload["metadata"]))
                context = metadata["context"]
                self.assertEqual(context["episode"]["seed"], 43)
                self.assertEqual(
                    {item["arrival"] for item in context["episode"]["requests"]},
                    {0},
                )
                self.assertEqual(
                    context["time_expansion"]["variable_count"],
                    len(record.expansion.variables),
                )
                self.assertEqual(
                    context["catalogue"]["fidelity_model"],
                    FIDELITY_MODEL_NAME,
                )
                self.assertEqual(
                    context["catalogue"]["success_probability_model"],
                    SUCCESS_PROBABILITY_MODEL_NAME,
                )

            result = generate_teacher_dataset(
                scenario,
                seeds=(44,),
                output_directory=root / "dataset",
                path_candidate_count=1,
            )
            self.assertTrue(result.manifest_path.exists())
            manifest = json.loads(
                result.manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["records"]), 1)
            entry = manifest["records"][0]
            self.assertEqual(entry["seed"], 44)
            self.assertTrue((result.manifest_path.parent / entry["file"]).exists())

    def test_batch_builder_uses_simultaneous_arrivals_by_default(self):
        record = build_teacher_batch_record(
            ScenarioConfig(
                request_count=1,
                min_hops=2,
                max_hops=2,
                topology_mode="parallel_corridors",
                ttl=6,
                horizon=6,
            ),
            seed=1,
            path_candidate_count=1,
        )
        self.assertEqual({request.arrival for request in record.episode.requests}, {0})

    def test_online_window_accepts_arrived_nonzero_requests(self):
        episode = EpisodeSpec(
            seed=9,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r", 0, 1, arrival=3, ttl=6),),
            horizon=12,
            physical=PhysicalConfig(quantum_distance_m=1.0),
        )
        record = solve_teacher_window(
            episode,
            window_start_slot=4,
            window_end_slot=10,
            path_candidate_count=1,
            construction_kinds=("balanced",),
            purification_kinds=("none",),
        )
        self.assertEqual(record.planning_window, (4, 10))
        self.assertTrue(record.expansion.variables)
        self.assertTrue(all(
            variable.start_slot >= 4
            for variable in record.expansion.variables
        ))

    def test_online_window_rejects_future_request_leakage(self):
        episode = EpisodeSpec(
            seed=10,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("future", 0, 1, arrival=5, ttl=4),),
            horizon=12,
        )
        with self.assertRaisesRegex(ValueError, "future request"):
            solve_teacher_window(
                episode,
                window_start_slot=4,
                window_end_slot=10,
                path_candidate_count=1,
            )


if __name__ == "__main__":
    unittest.main()
