import unittest
from dataclasses import replace

from algorithms.routing_core import build_planning_batch_problem
from algorithms.routing_core.candidates import (
    _canonicalize_planning_equivalent_candidates,
)
from algorithms.routing_core.fidelity import candidate_fidelity_estimate_map
from algorithms.routing_core.success_probability import (
    candidate_success_probability_map,
)
from qnet_core.construction_api import ConstructionDAG
from qnet_core.construction_catalog import (
    build_route_construction_catalogue,
)
from qnet_core.spec import EpisodeSpec, PhysicalConfig, RequestSpec


class PlanningProblemTests(unittest.TestCase):
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

    def test_candidate_dedup_preserves_operation_level_physics(self):
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

    def test_fidelity_filter_keeps_purification_only_when_required(self):
        episode = EpisodeSpec(
            seed=4,
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
        problem = build_planning_batch_problem(
            episode,
            path_candidate_count=1,
            construction_kinds=("balanced",),
        )
        self.assertTrue(problem.expansion.variables)
        self.assertEqual(
            {item.purification_kind for item in problem.expansion.variables},
            {"elementary_once"},
        )
        self.assertIn(
            "fidelity",
            {item.reason for item in problem.expansion.rejections},
        )

    def test_online_window_accepts_arrived_requests(self):
        episode = EpisodeSpec(
            seed=9,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r", 0, 1, arrival=3, ttl=6),),
            horizon=12,
            physical=PhysicalConfig(quantum_distance_m=1.0),
        )
        problem = build_planning_batch_problem(
            episode,
            window_start_slot=4,
            window_end_slot=10,
            path_candidate_count=1,
            construction_kinds=("balanced",),
            purification_kinds=("none",),
        )
        self.assertEqual(problem.planning_window, (4, 10))
        self.assertTrue(problem.expansion.variables)
        self.assertTrue(all(
            variable.start_slot >= 4
            for variable in problem.expansion.variables
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
            build_planning_batch_problem(
                episode,
                window_start_slot=4,
                window_end_slot=10,
                path_candidate_count=1,
            )


if __name__ == "__main__":
    unittest.main()
