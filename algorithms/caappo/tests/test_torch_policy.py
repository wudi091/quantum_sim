import math
import unittest
from dataclasses import replace

import numpy as np
import torch

from algorithms.caappo import (
    TorchCAAPPOPolicy,
    TorchCAAPPORolloutTrainer,
    compute_gae,
)
from qnet_core.joint_construction_gym import JointPhase, JointStep
from qnet_core.construction_api import (
    ConstructionDAG,
    ConstructionOperation,
    OperationKind,
    ResourceDemand,
)
from qnet_core.construction_catalog import build_route_construction_catalogue
from qnet_core.construction_decoder import CapacityFeasibilityOracle
from qnet_core.construction_executor import ConstructionDAGExecutor
from qnet_core.planning_spec import RequestSpec
from qnet_core.spec import EpisodeSpec, PhysicalConfig


class TorchCAAPPOTests(unittest.TestCase):
    def test_gae_accepts_event_duration_discounts(self):
        advantages, targets = compute_gae(
            (1.0, 2.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (False, True),
            gae_lambda=1.0,
            discounts=(0.5, 1.0),
        )
        np.testing.assert_allclose(advantages, (2.0, 2.0))
        np.testing.assert_allclose(targets, (2.0, 2.0))

    def test_operation_log_probability_uses_dynamic_prefix_mask(self):
        demand = ResourceDemand.from_mapping({"shared": 1})
        operations = tuple(
            ConstructionOperation(
                f"r:g{index}",
                "r",
                OperationKind.GEN,
                output_segment_id=f"r:s{index}",
                output_endpoints=(index, index + 1),
                resource_demand=demand,
                ordinal=index,
            )
            for index in range(2)
        )
        executor = ConstructionDAGExecutor(
            (ConstructionDAG("r", operations),), {"shared": 1}
        )
        policy = TorchCAAPPOPolicy(seed=31)
        with torch.no_grad():
            policy.operation_actor[-1].weight.zero_()
            policy.operation_actor[-1].bias.fill_(10.0)
        sample = policy.sample_operation(
            executor.snapshot(),
            operations,
            CapacityFeasibilityOracle({"shared": 1}),
            stop_legal=True,
            deterministic=True,
        ).sample
        self.assertEqual(sample.selected_indices, (0,))
        self.assertEqual(sample.legal_indices, (0,))
        evaluated = policy.evaluate_operation_log_probability(sample)
        self.assertAlmostEqual(
            sample.log_probability, float(evaluated.detach().item()), places=6
        )

    def test_encoder_consumes_full_snapshot_dag_not_only_ready_frontier(self):
        first = ConstructionOperation(
            "r:g0",
            "r",
            OperationKind.GEN,
            output_segment_id="r:s0",
            output_endpoints=(0, 1),
            ordinal=0,
        )
        second = ConstructionOperation(
            "r:g1",
            "r",
            OperationKind.GEN,
            predecessors=("r:g0",),
            output_segment_id="r:s1",
            output_endpoints=(1, 2),
            ordinal=1,
        )
        executor = ConstructionDAGExecutor(
            (ConstructionDAG("r", (first, second)),), {}
        )
        policy = TorchCAAPPOPolicy(seed=32)
        snapshot = executor.snapshot()
        full = policy.encoder.encode(snapshot, (first,))
        frontier_only = policy.encoder.encode(
            replace(snapshot, operations=()), (first,)
        )
        self.assertFalse(np.allclose(full, frontier_only))

    def test_potential_uses_remaining_critical_path(self):
        chain_first = ConstructionOperation(
            "chain:g0", "chain", OperationKind.GEN,
            output_segment_id="chain:s0", output_endpoints=(0, 1),
            duration_ps=2,
        )
        chain_second = ConstructionOperation(
            "chain:g1", "chain", OperationKind.GEN,
            predecessors=("chain:g0",),
            output_segment_id="chain:s1", output_endpoints=(1, 2),
            duration_ps=2,
        )
        parallel_first = ConstructionOperation(
            "parallel:g0", "parallel", OperationKind.GEN,
            output_segment_id="parallel:s0", output_endpoints=(0, 1),
            duration_ps=2,
        )
        parallel_second = ConstructionOperation(
            "parallel:g1", "parallel", OperationKind.GEN,
            output_segment_id="parallel:s1", output_endpoints=(1, 2),
            duration_ps=2,
        )
        chain_snapshot = ConstructionDAGExecutor(
            (ConstructionDAG("chain", (chain_first, chain_second)),), {}
        ).snapshot()
        parallel_snapshot = ConstructionDAGExecutor(
            (ConstructionDAG("parallel", (parallel_first, parallel_second)),), {}
        ).snapshot()
        chain_state = JointStep(
            JointPhase.EXECUTION, chain_snapshot, (), {}, 0.0, False, {}
        )
        parallel_state = JointStep(
            JointPhase.EXECUTION, parallel_snapshot, (), {}, 0.0, False, {}
        )
        self.assertLess(
            TorchCAAPPORolloutTrainer._potential(chain_state),
            TorchCAAPPORolloutTrainer._potential(parallel_state),
        )
        settled_chain = replace(chain_state, info={"settled_request_ids": ("chain",)})
        self.assertEqual(TorchCAAPPORolloutTrainer._potential(settled_chain), 0.0)
        settled_state = replace(
            chain_state,
            observation=replace(
                chain_snapshot, settled_request_ids=("chain",)
            ),
        )
        self.assertEqual(TorchCAAPPORolloutTrainer._potential(settled_state), 0.0)

    def test_repair_head_assigns_probability_to_each_retry_option(self):
        operation_a = ConstructionOperation(
            "r:a", "r", OperationKind.GEN,
            output_segment_id="r:a-segment", output_endpoints=(0, 1),
            ordinal=0, retry_limit=1,
        )
        operation_b = ConstructionOperation(
            "r:b", "r", OperationKind.GEN,
            output_segment_id="r:b-segment", output_endpoints=(1, 2),
            ordinal=1, retry_limit=1,
        )
        executor = ConstructionDAGExecutor(
            (ConstructionDAG("r", (operation_a, operation_b)),), {}
        )
        policy = TorchCAAPPOPolicy(seed=33)
        options = ((operation_a,), (operation_b,))
        sample = policy.sample_repair(
            executor.snapshot(), options, deterministic=True
        ).sample
        self.assertIn(sample.repair_action, (0, 1, 2))
        evaluated = policy.evaluate_repair_log_probability(sample)
        self.assertAlmostEqual(
            sample.log_probability, float(evaluated.detach().item()), places=6
        )

    def test_tiny_sequence_episode_updates_torch_policy(self):
        spec = EpisodeSpec(
            seed=733,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r", 0, 1, required_fidelity=0.5),),
            horizon=20,
            physical=PhysicalConfig(
                generation_probability=1.0,
                detector_efficiency=1.0,
                initial_fidelity=0.9,
                node_memory_capacity=1,
                memory_capacity=1,
                quantum_distance_m=1.0,
            ),
        )
        catalogue = build_route_construction_catalogue(
            spec.planning,
            candidate_count=1,
            construction_kinds=("left_deep",),
        )
        policy = TorchCAAPPOPolicy(seed=37)
        encoder_before = policy.encoder.self_projection.weight.detach().clone()
        result = TorchCAAPPORolloutTrainer(policy).run_episode(
            spec, catalogue, deterministic=False, update=True
        )
        self.assertEqual(result.metrics["completed_requests"], 1.0)
        self.assertIsNotNone(result.update_stats)
        self.assertTrue(math.isfinite(result.update_stats.policy_loss))
        self.assertFalse(torch.equal(
            encoder_before, policy.encoder.self_projection.weight.detach()
        ))


if __name__ == "__main__":
    unittest.main()
