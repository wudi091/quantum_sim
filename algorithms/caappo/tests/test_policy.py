import unittest

from algorithms.caappo import CAAPPOPolicy, PPOTransition
from qnet_core.construction_catalog import build_route_construction_catalogue
from qnet_core.construction_decoder import CapacityFeasibilityOracle
from qnet_core.construction_executor import ConstructionDAGExecutor
from qnet_core.construction_plans import left_deep_path_dag
from qnet_core.construction_api import ConstructionOperation
from qnet_core.planning_spec import PlanningSpec, RequestSpec


class CAAPPOPolicyTests(unittest.TestCase):
    def test_joint_sample_respects_capacity_mask_and_is_seeded(self):
        spec = PlanningSpec(
            seed=1,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r", 0, 2),),
            horizon=10,
        )
        candidates = build_route_construction_catalogue(spec, candidate_count=1)
        dag = candidates[0].dag
        capacities = {
            "link:0-1": 1,
            "link:1-2": 1,
            "genlane:0-1": 1,
            "genlane:1-2": 1,
            "memory:0": 1,
            "memory:1": 2,
            "memory:2": 1,
            "bsm:1": 1,
        }
        executor = ConstructionDAGExecutor((dag,), capacities)
        policy = CAAPPOPolicy(seed=5)
        first = policy.joint_sample(
            executor.snapshot(), candidates, {},
            CapacityFeasibilityOracle(capacities),
            stop_legal=True,
            deterministic=True,
        )
        self.assertIn(first.action.candidate_id, {candidate.candidate_id for candidate in candidates})
        chosen = next(candidate for candidate in candidates if candidate.candidate_id == first.action.candidate_id)
        self.assertTrue(set(first.action.operation_ids).issubset({operation.op_id for operation in chosen.dag.operations}))
        self.assertFalse(any("swap" in operation_id for operation_id in first.action.operation_ids))

    def test_dual_update_increases_risk_multiplier(self):
        policy = CAAPPOPolicy(seed=7)
        feature = policy.encoder.encode(
            ConstructionDAGExecutor(
                (left_deep_path_dag("r", (0, 1)),),
                {
                    "link:0-1": 1,
                    "genlane:0-1": 1,
                    "memory:0": 1,
                    "memory:1": 1,
                },
            ).snapshot(),
            left_deep_path_dag("r", (0, 1)).operations,
        )
        policy._operation_weights = feature * 0.0
        result = policy.update((PPOTransition(feature, 0, 0.0, 1.0, 1.0, 1.0),), risk_limit=0.0)
        self.assertGreater(result["lambda_risk"], 0.0)


if __name__ == "__main__":
    unittest.main()
