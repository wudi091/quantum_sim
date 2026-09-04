import unittest

from algorithms.routing_core import (
    effective_generation_probability,
    estimate_candidate_success_probability,
)
from qnet_core.construction_catalog import build_route_construction_catalogue
from qnet_core.planning_spec import PlanningSpec, RequestSpec
from qnet_core.spec import EpisodeSpec, PhysicalConfig


class SuccessProbabilityTests(unittest.TestCase):
    def test_four_hop_candidate_matches_generation_and_swap_product(self):
        planning = PlanningSpec(
            seed=1,
            nodes=(0, 1, 2, 3, 4),
            edges=((0, 1), (1, 2), (2, 3), (3, 4)),
            requests=(RequestSpec("r", 0, 4, ttl=8),),
            horizon=8,
        )
        episode = EpisodeSpec(
            seed=planning.seed,
            nodes=planning.nodes,
            edges=planning.edges,
            requests=planning.requests,
            horizon=planning.horizon,
            physical=PhysicalConfig(
                generation_probability=0.8,
                swap_probability=0.9,
            ),
        )
        candidate = build_route_construction_catalogue(
            planning,
            candidate_count=1,
            construction_kinds=("balanced",),
            purification_kinds=("none",),
        )[0]

        estimate = estimate_candidate_success_probability(episode, candidate)

        self.assertAlmostEqual(estimate.probability, 0.8 ** 4 * 0.9 ** 3)

    def test_effective_generation_probability_includes_physical_losses(self):
        physical = PhysicalConfig(
            generation_probability=0.8,
            quantum_distance_m=1000.0,
            quantum_attenuation_db_per_m=0.0002,
            detector_efficiency=0.9,
            bsm_success_probability=0.5,
        )

        expected = 0.8 * 10.0 ** (-0.0002 * 1000.0 / 10.0) * 0.9 ** 2 * 0.5
        self.assertAlmostEqual(
            effective_generation_probability(physical),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
