import unittest

from qnet_core.construction_plans import balanced_path_dag
from qnet_core.fidelity_estimation import (
    FIDELITY_MODEL_NAME,
    estimate_sequence_bds_fidelity_lower_bound,
    werner_bbpssw_result,
    werner_storage_fidelity_lower_bound,
)
from qnet_core.spec import PhysicalConfig


class FidelityEstimationTests(unittest.TestCase):
    def test_bbpssw_improves_a_successful_point_eight_werner_pair(self):
        success_probability, fidelity = werner_bbpssw_result(0.8, 0.8, 1.0)

        self.assertGreater(success_probability, 0.5)
        self.assertLess(success_probability, 1.0)
        self.assertGreater(fidelity, 0.8)

    def test_storage_bound_decreases_without_using_random_outcomes(self):
        physical = PhysicalConfig(initial_fidelity=0.99, memory_lifetime=100)

        immediate = werner_storage_fidelity_lower_bound(
            physical.initial_fidelity, 0, physical.memory_lifetime
        )
        delayed = werner_storage_fidelity_lower_bound(
            physical.initial_fidelity, 1, physical.memory_lifetime
        )

        self.assertEqual(immediate, physical.initial_fidelity)
        self.assertLess(delayed, immediate)

    def test_bound_accounts_for_generation_storage_and_swap_degradation(self):
        dag = balanced_path_dag("r", (0, 1, 2))
        slots = {
            operation.op_id: (0 if operation.kind == "GEN" else 1)
            for operation in dag.operations
        }
        terminal = next(
            operation.output_segment_id
            for operation in dag.operations
            if operation.kind == "SWAP"
        )
        assert terminal is not None

        bound = estimate_sequence_bds_fidelity_lower_bound(
            PhysicalConfig(
                initial_fidelity=0.99,
                swap_degradation=0.95,
                memory_lifetime=100,
            ),
            dag,
            (terminal,),
            slots,
        )

        self.assertEqual(bound.model_name, FIDELITY_MODEL_NAME)
        self.assertGreater(bound.lower_bound, 0.7)
        self.assertLess(bound.lower_bound, 0.99)


if __name__ == "__main__":
    unittest.main()
