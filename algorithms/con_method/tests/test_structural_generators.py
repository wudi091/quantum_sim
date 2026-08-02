from dataclasses import replace
import unittest

from algorithms.con_method.offline_library import (
    GENERATOR_PRESETS,
    build_topology_template_pool,
    build_waxman_selection_context,
    build_waxman_topology_pool,
    compile_structural_topology_library,
    compile_waxman_topology_library,
    select_structural_library,
    waxman_topology_fingerprint,
)
from qnet_core.order_core import OrderLinkSpec
from qnet_core.order_waxman import WaxmanOrderConfig, WaxmanOrderEpisode


class StructuralGeneratorTests(unittest.TestCase):
    @staticmethod
    def _parallel_routes_episode():
        nodes = tuple(range(7))
        edges = tuple(
            edge
            for middle in range(1, 6)
            for edge in ((0, middle), (middle, 6))
        )
        links = tuple(
            OrderLinkSpec(
                left,
                right,
                generation_probability=(
                    0.1 if 1 in (left, right) else 0.9
                ),
            )
            for left, right in edges
        )
        config = WaxmanOrderConfig(
            node_count=7,
            average_degree=2,
            request_count=1,
            arrival_rate=1.0,
            episode_steps=1,
            min_hops=1,
            max_hops=2,
            candidate_paths=4,
            order_variants_per_path=4,
            node_memory_cap=4,
        )
        return WaxmanOrderEpisode(
            seed=0,
            config=config,
            nodes=nodes,
            links=links,
            node_capacities=tuple((node, 4) for node in nodes),
            positions=tuple((node, (float(node), 0.0)) for node in nodes),
            requests=(),
            request_paths=(),
            topology_beta=1.0,
            link_alpha=1.0,
            horizon_slots=1,
        )

    def test_quality_and_exact_portfolio_avoid_low_reliability_route(self):
        episode = self._parallel_routes_episode()
        pool = build_waxman_topology_pool(
            episode,
            path_pool_per_pair=5,
            max_hops=2,
        )
        context = build_waxman_selection_context(episode)
        pair_id = pool.pair_id_by_endpoints[(0, 6)]

        canonical = select_structural_library(
            pool, preset="canonical", context=context
        )
        quality = select_structural_library(
            pool, preset="quality", context=context
        )
        exact = select_structural_library(
            pool, preset="pareto", context=context
        )
        canonical_paths = tuple(
            pool.path_by_id[path_id].path
            for path_id in canonical.path_ids_by_pair[pair_id]
        )
        quality_paths = tuple(
            pool.path_by_id[path_id].path
            for path_id in quality.path_ids_by_pair[pair_id]
        )
        exact_paths = tuple(
            pool.path_by_id[path_id].path
            for path_id in exact.path_ids_by_pair[pair_id]
        )

        self.assertIn((0, 1, 6), canonical_paths)
        self.assertNotIn((0, 1, 6), quality_paths)
        self.assertNotIn((0, 1, 6), exact_paths)
        self.assertEqual(len(quality_paths), 4)
        self.assertEqual(len(exact_paths), 4)

    def test_pareto_schedule_generator_keeps_only_useful_release_front(self):
        pool = build_topology_template_pool(
            nodes=range(5),
            edges=((0, 1), (1, 2), (2, 3), (3, 4)),
            topology_fingerprint="line-five",
            path_pool_per_pair=1,
            max_hops=4,
        )
        pair_id = pool.pair_id_by_endpoints[(0, 4)]

        canonical = select_structural_library(pool, preset="canonical")
        pareto = select_structural_library(pool, preset="pareto")
        exact = select_structural_library(pool, preset="exact_kcenter")
        path_id = pareto.path_ids_by_pair[pair_id][0]
        pareto_schedules = tuple(
            pool.template_by_id[template_id].schedule
            for template_id in pareto.template_ids_by_path[path_id]
        )

        self.assertEqual(
            len(canonical.template_ids_by_path[path_id]), 4
        )
        self.assertEqual(len(pareto_schedules), 3)
        self.assertEqual(
            set(pareto.template_ids_by_path[path_id]),
            set(exact.template_ids_by_path[path_id]),
        )
        self.assertEqual(
            {schedule.groups for schedule in pareto_schedules},
            {
                ((1, 3), (2,)),
                ((2,), (1,), (3,)),
                ((2,), (3,), (1,)),
            },
        )
        self.assertTrue(all(schedule.round_count <= 3 for schedule in pareto_schedules))
        compiled = compile_structural_topology_library(pool, pareto)
        grid = compiled.library.lookup_grid(0, 4)
        self.assertEqual(
            grid.schedule_valid_mask[0],
            (True, True, True, False),
        )

    def test_all_presets_are_deterministic_and_request_independent(self):
        episode = self._parallel_routes_episode()
        pool = build_waxman_topology_pool(
            episode,
            path_pool_per_pair=5,
            max_hops=2,
        )
        context = build_waxman_selection_context(episode)
        changed_request_metadata = replace(
            episode,
            request_paths=(("ignored", ((0, 1, 6),)),),
        )
        changed_context = build_waxman_selection_context(
            changed_request_metadata
        )

        for preset in GENERATOR_PRESETS:
            with self.subTest(preset=preset):
                first = select_structural_library(
                    pool, preset=preset, context=context
                )
                second = select_structural_library(
                    pool, preset=preset, context=changed_context
                )
                self.assertEqual(first, second)
                artifact = compile_structural_topology_library(
                    pool, first
                ).library
                self.assertEqual(
                    artifact.selection_mode,
                    f"topology-only:{preset}",
                )
                self.assertEqual(
                    artifact.topology_fingerprint,
                    waxman_topology_fingerprint(episode),
                )

    def test_context_rejects_wrong_fingerprint(self):
        episode = self._parallel_routes_episode()
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_waxman_selection_context(
                episode,
                topology_fingerprint="wrong",
            )

    def test_waxman_default_compiler_uses_algorithm_not_empty_milp(self):
        episode = self._parallel_routes_episode()
        compiled = compile_waxman_topology_library(
            episode,
            path_pool_per_pair=5,
            max_hops=2,
        )

        self.assertEqual(compiled.selection.generator_name, "hybrid")
        self.assertEqual(
            compiled.library.selection_mode,
            "topology-only:hybrid",
        )
        self.assertEqual(
            compiled.library.solver_certificate.solver,
            "algorithm:hybrid",
        )


if __name__ == "__main__":
    unittest.main()
