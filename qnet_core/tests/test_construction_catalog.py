import unittest

from algorithms.caappo import BalancedConstructionPolicy, ShortestPathLeftDeepPolicy
from qnet_core.construction_catalog import (
    build_dynamic_repair_catalogue,
    build_route_construction_catalogue,
)
from qnet_core.construction_evaluate import run_joint_plan_baseline
from qnet_core.spec import EpisodeSpec, PhysicalConfig
from qnet_core.planning_spec import RequestSpec


class ConstructionCatalogueTests(unittest.TestCase):
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
        left = ShortestPathLeftDeepPolicy().select(candidates)["r0"]
        balanced = BalancedConstructionPolicy().select(candidates)["r0"]
        self.assertEqual(left.hop_count, 2)
        self.assertEqual(balanced.hop_count, 2)
        self.assertEqual(left.construction_kind, "left_deep")
        self.assertEqual(balanced.construction_kind, "balanced")

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

    def test_fixed_joint_plan_runs_through_sequence_and_reports_risk(self):
        spec = self._spec()
        candidates = build_route_construction_catalogue(spec.planning, candidate_count=1)
        selected = ShortestPathLeftDeepPolicy().select(candidates)
        result = run_joint_plan_baseline(spec, selected)
        self.assertEqual(result.metrics["completed_requests"], 1.0)
        self.assertEqual(result.metrics["risk_count"], 0.0)
        self.assertGreater(len(result.event_trace), 0)

    def test_baseline_packs_bsm_contention_without_corrupting_in_flight_pairs(self):
        spec = EpisodeSpec(
            seed=307,
            nodes=(0, 1, 2, 3, 4),
            edges=((0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (2, 4)),
            requests=(
                RequestSpec("r0", 0, 4, ttl=100),
                RequestSpec("r1", 1, 3, ttl=100),
            ),
            horizon=100,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                memory_capacity=2,
                node_memory_capacity=6,
                quantum_distance_m=1.0,
            ),
        )
        candidates = build_route_construction_catalogue(spec.planning, candidate_count=3)
        result = run_joint_plan_baseline(spec, ShortestPathLeftDeepPolicy().select(candidates))
        self.assertEqual(result.metrics["completed_requests"], 2.0)
        self.assertEqual(result.metrics["risk_count"], 0.0)


if __name__ == "__main__":
    unittest.main()
