import unittest

from qnet_core.construction_catalog import (
    build_dynamic_repair_catalogue,
    build_route_construction_catalogue,
    candidates_by_request,
)
from qnet_core.construction_api import OperationKind
from qnet_core.construction_plans import (
    ordered_swap_trees,
    swap_tree_kinds,
    swap_tree_path_dag,
)
from qnet_core.spec import EpisodeSpec, PhysicalConfig
from qnet_core.planning_spec import RequestSpec


def _select_fixed(candidates, construction_kind):
    return {
        request_id: min(
            values,
            key=lambda candidate: (
                candidate.hop_count,
                candidate.construction_kind != construction_kind,
                candidate.candidate_id,
            ),
        )
        for request_id, values in candidates_by_request(candidates).items()
    }


class ConstructionCatalogueTests(unittest.TestCase):
    @staticmethod
    def _tree_height(tree):
        if isinstance(tree, int):
            return 0
        return 1 + max(
            ConstructionCatalogueTests._tree_height(tree[0]),
            ConstructionCatalogueTests._tree_height(tree[1]),
        )

    def _spec(self):
        return EpisodeSpec(
            seed=211,
            nodes=(0, 1, 2, 3),
            edges=((0, 1), (1, 2), (2, 3), (0, 2)),
            requests=(RequestSpec("r0", 0, 3, ttl=100),),
            horizon=100,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                memory_capacity=2,
                node_memory_capacity=5,
                quantum_distance_m=1.0,
            ),
        )

    def test_catalogue_contains_route_and_construction_choices(self):
        candidates = build_route_construction_catalogue(self._spec().planning, candidate_count=2)
        self.assertEqual({candidate.construction_kind for candidate in candidates}, {"left_deep", "balanced"})
        self.assertGreaterEqual(len({candidate.route_nodes for candidate in candidates}), 2)
        left = _select_fixed(candidates, "left_deep")["r0"]
        balanced = _select_fixed(candidates, "balanced")["r0"]
        self.assertEqual(left.hop_count, 2)
        self.assertEqual(balanced.hop_count, 2)
        self.assertEqual(left.construction_kind, "left_deep")
        self.assertEqual(balanced.construction_kind, "balanced")

    def test_four_hop_route_has_all_five_distinct_swap_trees(self):
        route = (0, 1, 2, 3, 4)
        kinds = swap_tree_kinds(4)
        self.assertEqual(
            kinds,
            tuple(f"swap_tree_{index}" for index in range(5)),
        )
        signatures = set()
        for index, kind in enumerate(kinds):
            dag = swap_tree_path_dag(
                "r0",
                route,
                index,
                required_fidelity=0.8,
            )
            generations = tuple(
                operation for operation in dag.operations
                if operation.kind == OperationKind.GEN
            )
            swaps = tuple(
                operation for operation in dag.operations
                if operation.kind == OperationKind.SWAP
            )
            self.assertEqual(len(generations), 4, kind)
            self.assertEqual(len(swaps), 3, kind)
            terminal = next(
                operation for operation in swaps
                if operation.output_endpoints == (0, 4)
            )
            self.assertEqual(terminal.required_fidelity, 0.8, kind)
            self.assertEqual(
                sum(operation.required_fidelity > 0 for operation in swaps),
                1,
                kind,
            )
            signatures.add(tuple(sorted(
                operation.output_endpoints for operation in swaps
            )))
        self.assertEqual(len(signatures), 5)

    def test_limited_swap_trees_are_top_k_by_parallel_depth(self):
        five_link_trees = ordered_swap_trees(5, limit=5)
        six_link_trees = ordered_swap_trees(6, limit=5)

        self.assertEqual(len(five_link_trees), 5)
        self.assertEqual(len(six_link_trees), 5)
        self.assertEqual(
            [self._tree_height(tree) for tree in five_link_trees],
            [3, 3, 3, 3, 3],
        )
        self.assertEqual(
            [self._tree_height(tree) for tree in six_link_trees],
            [3, 3, 3, 3, 3],
        )

    def test_top_k_swap_tree_prefix_is_stable(self):
        self.assertEqual(
            ordered_swap_trees(6, limit=5),
            ordered_swap_trees(6, limit=6)[:5],
        )

    def test_four_paths_times_five_trees_produce_twenty_candidates(self):
        spec = EpisodeSpec(
            seed=214,
            nodes=tuple(range(14)),
            edges=(
                (0, 2), (2, 3), (3, 4), (4, 1),
                (0, 5), (5, 6), (6, 7), (7, 1),
                (0, 8), (8, 9), (9, 10), (10, 1),
                (0, 11), (11, 12), (12, 13), (13, 1),
            ),
            requests=(RequestSpec("r0", 0, 1, ttl=20),),
            horizon=20,
            physical=PhysicalConfig(),
        )
        candidates = build_route_construction_catalogue(
            spec.planning,
            candidate_count=4,
            construction_kinds=swap_tree_kinds(4),
            purification_kinds=("none",),
        )
        self.assertEqual(len(candidates), 20)
        self.assertEqual(len({item.route_nodes for item in candidates}), 4)
        self.assertEqual(
            {item.construction_kind for item in candidates},
            set(swap_tree_kinds(4)),
        )

    def test_variable_length_routes_use_up_to_requested_swap_tree_count(self):
        spec = EpisodeSpec(
            seed=215,
            nodes=(0, 1, 2, 3),
            edges=((0, 1), (0, 2), (2, 3), (3, 1)),
            requests=(RequestSpec("r0", 0, 1, ttl=20),),
            horizon=20,
            physical=PhysicalConfig(),
        )
        candidates = build_route_construction_catalogue(
            spec.planning,
            candidate_count=2,
            construction_kinds=(),
            swap_tree_count=5,
            purification_kinds=("none",),
        )
        counts_by_hops = {}
        for candidate in candidates:
            counts_by_hops[candidate.hop_count] = (
                counts_by_hops.get(candidate.hop_count, 0) + 1
            )
        self.assertEqual(counts_by_hops, {1: 1, 3: 2})

    def test_dynamic_repair_catalogue_excludes_seen_routes(self):
        spec = EpisodeSpec(
            seed=212,
            nodes=(0, 1, 2, 3),
            edges=((0, 1), (1, 3), (0, 2), (2, 3), (0, 3)),
            requests=(RequestSpec("r0", 0, 3, ttl=100),),
            horizon=100,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                node_memory_capacity=5,
                quantum_distance_m=1.0,
            ),
        )
        initial = build_route_construction_catalogue(
            spec.planning, candidate_count=1, construction_kinds=("balanced",)
        )
        dynamic = build_dynamic_repair_catalogue(
            spec.planning,
            "r0",
            excluded_routes=(initial[0].route_nodes,),
            max_paths=2,
            construction_kinds=("balanced",),
        )
        self.assertTrue(dynamic)
        self.assertTrue(all(
            candidate.route_nodes != initial[0].route_nodes
            for candidate in dynamic
        ))
        self.assertTrue(all(
            candidate.candidate_id.startswith("r0:dynamic:path:")
            for candidate in dynamic
        ))

    def test_elementary_once_candidate_inserts_two_generations_and_bbpssw(self):
        spec = EpisodeSpec(
            seed=213,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r0", 0, 1, required_fidelity=0.8),),
            horizon=8,
            physical=PhysicalConfig(),
        )
        candidates = build_route_construction_catalogue(
            spec.planning,
            candidate_count=1,
            construction_kinds=("balanced",),
            purification_kinds=("none", "elementary_once"),
        )
        purified = next(
            item for item in candidates
            if item.purification_kind == "elementary_once"
        )
        generations = tuple(
            operation for operation in purified.dag.operations
            if operation.kind == OperationKind.GEN
        )
        purification = next(
            operation for operation in purified.dag.operations
            if operation.kind == OperationKind.PURIFY
        )

        self.assertEqual(len(generations), 2)
        self.assertEqual(len(purification.input_segment_ids), 2)
        self.assertEqual(
            purification.output_segment_id,
            purified.terminal_segment_id,
        )
        self.assertEqual(
            purification.resource_demand.get("purify:0-1"),
            1,
        )

if __name__ == "__main__":
    unittest.main()
