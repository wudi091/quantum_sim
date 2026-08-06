import unittest
from dataclasses import replace

from algorithms.caappo import DeterministicJointPlanOracle
from qnet_core.construction_api import ConstructionDAG
from qnet_core.construction_api import ConstructionOperation, OperationKind, ResourceDemand
from qnet_core.construction_catalog import (
    RouteConstructionCandidate,
    build_route_construction_catalogue,
)
from qnet_core.planning_spec import RequestSpec
from qnet_core.spec import EpisodeSpec, PhysicalConfig


class DeterministicOracleTests(unittest.TestCase):
    @staticmethod
    def _generation(request_id: str, operation_id: str, segment_id: str, duration_ps: int = 1):
        return ConstructionOperation(
            operation_id,
            request_id,
            OperationKind.GEN,
            output_segment_id=segment_id,
            output_endpoints=(0, 1),
            resource_demand=ResourceDemand.from_mapping({
                "link:0-1": 1,
                "genlane:0-1": 1,
                "memory:0": 1,
                "memory:1": 1,
            }),
            output_resource_hold=ResourceDemand.from_mapping({
                "link:0-1": 1,
                "memory:0": 1,
                "memory:1": 1,
            }),
            duration_ps=duration_ps,
            success_probability=1.0,
        )

    def test_oracle_selects_lower_flow_time_candidate(self):
        spec = EpisodeSpec(
            seed=811,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r", 0, 1, required_fidelity=0.5),),
            horizon=10,
            physical=PhysicalConfig(
                memory_capacity=1,
                node_memory_capacity=1,
                max_width=1,
                slot_duration_ps=1,
            ),
        )
        base = build_route_construction_catalogue(
            spec.planning,
            candidate_count=1,
            construction_kinds=("left_deep",),
        )[0]
        fast = RouteConstructionCandidate(
            "r:fast",
            "r",
            base.route_nodes,
            "custom",
            ConstructionDAG("r", tuple(
                replace(operation, duration_ps=1)
                for operation in base.dag.operations
            )),
            base.terminal_segment_id,
            base.terminal_segment_ids,
        )
        slow = RouteConstructionCandidate(
            "r:slow",
            "r",
            base.route_nodes,
            "custom",
            ConstructionDAG("r", tuple(
                replace(operation, duration_ps=4)
                for operation in base.dag.operations
            )),
            base.terminal_segment_id,
            base.terminal_segment_ids,
        )
        result = DeterministicJointPlanOracle().solve(spec, (slow, fast))
        self.assertEqual(result.selected_candidate_ids, (("r", "r:fast"),))
        self.assertEqual(result.completed_requests, 1)
        self.assertEqual(result.censored_flow_time_ps, 1)
        self.assertEqual(result.optimality_gap(result.score), 0.0)

    def test_oracle_finds_concurrent_disjoint_generations(self):
        spec = EpisodeSpec(
            seed=812,
            nodes=(0, 1, 2, 3),
            edges=((0, 1), (2, 3)),
            requests=(
                RequestSpec("r0", 0, 1, required_fidelity=0.5),
                RequestSpec("r1", 2, 3, required_fidelity=0.5),
            ),
            horizon=10,
            physical=PhysicalConfig(
                memory_capacity=1,
                node_memory_capacity=1,
                max_width=1,
                slot_duration_ps=1,
            ),
        )
        catalogue = build_route_construction_catalogue(
            spec.planning,
            candidate_count=1,
            construction_kinds=("left_deep",),
        )
        result = DeterministicJointPlanOracle().solve(spec, catalogue)
        self.assertEqual(result.completed_requests, 2)
        self.assertTrue(any(len(action) == 2 for action in result.action_trace))

    def test_oracle_releases_late_output_after_deadline_settlement(self):
        spec = EpisodeSpec(
            seed=813,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(
                RequestSpec("r0", 0, 1, ttl=1),
                RequestSpec("r1", 0, 1, arrival=3),
            ),
            horizon=6,
            physical=PhysicalConfig(
                memory_capacity=1,
                node_memory_capacity=1,
                max_width=1,
                slot_duration_ps=1,
            ),
        )
        candidates = (
            RouteConstructionCandidate(
                "r0:late",
                "r0",
                (0, 1),
                "custom",
                ConstructionDAG("r0", (self._generation("r0", "g0", "s0", 2),)),
                "s0",
            ),
            RouteConstructionCandidate(
                "r1:on-time",
                "r1",
                (0, 1),
                "custom",
                ConstructionDAG("r1", (self._generation("r1", "g1", "s1"),)),
                "s1",
            ),
        )
        result = DeterministicJointPlanOracle().solve(spec, candidates)
        self.assertEqual(result.completed_requests, 1)
        self.assertEqual(result.risk_count, 1)

    def test_oracle_rejects_unmodeled_memory_expiration_horizon(self):
        spec = EpisodeSpec(
            seed=814,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r", 0, 1),),
            horizon=10,
            physical=PhysicalConfig(memory_lifetime=1),
        )
        catalogue = build_route_construction_catalogue(
            spec.planning, candidate_count=1, construction_kinds=("left_deep",)
        )
        with self.assertRaisesRegex(ValueError, "memory expiration"):
            DeterministicJointPlanOracle().solve(spec, catalogue)


if __name__ == "__main__":
    unittest.main()
