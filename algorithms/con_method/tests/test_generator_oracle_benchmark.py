import unittest

from algorithms.con_method.benchmarks import run_generator_oracle_benchmark
from qnet_core.order_waxman import WaxmanOrderConfig


class GeneratorOracleBenchmarkTests(unittest.TestCase):
    def test_generators_share_trace_physics_and_online_oracle(self):
        config = WaxmanOrderConfig(
            node_count=6,
            average_degree=2,
            request_count=4,
            arrival_rate=2.0,
            episode_steps=2,
            request_ttl_slots=2,
            min_hops=1,
            max_hops=4,
            candidate_paths=4,
            order_variants_per_path=4,
            candidate_request_cap=None,
            node_memory_cap=2,
            swap_probability=0.9,
        )
        result = run_generator_oracle_benchmark(
            config=config,
            episode_seeds=(44,),
            generator_names=("canonical", "pareto"),
            baseline_names=("qddca_fixed", "qcast_fixed"),
            path_pool_per_pair=5,
            max_hops=4,
            planning_seeds=(0,),
        )

        self.assertFalse(
            result["protocol"]["offline_generator_observes_requests"]
        )
        self.assertTrue(
            result["protocol"][
                "same_topology_trace_and_physics_per_generator"
            ]
        )
        self.assertEqual(
            result["protocol"]["online_selector"],
            "MilpReliableMemoryPathOrderPlanner",
        )
        self.assertEqual(result["protocol"]["reliability_confidence"], 0.9)
        self.assertEqual(set(result["aggregate"]), {"canonical", "pareto"})
        self.assertEqual(
            set(result["baseline_aggregate"]),
            {"qddca_fixed", "qcast_fixed"},
        )
        self.assertEqual(len(result["rows"]["canonical"]), 1)
        self.assertEqual(len(result["rows"]["pareto"]), 1)
        self.assertEqual(
            result["rows"]["canonical"][0]["episode_seed"],
            result["rows"]["pareto"][0]["episode_seed"],
        )
        self.assertIn(result["recommended_generator"], {"canonical", "pareto"})
        self.assertIn(
            "pareto_minus_qddca_fixed", result["method_deltas"]
        )


if __name__ == "__main__":
    unittest.main()
