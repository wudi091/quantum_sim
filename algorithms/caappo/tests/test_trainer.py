import unittest

from algorithms.caappo import (
    CAAPPOPolicy,
    CAAPPORolloutTrainer,
    PolicyAction,
    PolicySample,
)
from qnet_core.construction_catalog import build_route_construction_catalogue
from qnet_core.planning_spec import RequestSpec
from qnet_core.spec import EpisodeSpec, PhysicalConfig


class _CapturePolicy(CAAPPOPolicy):
    def __init__(self):
        super().__init__(seed=17)
        self.captured = ()

    def select_candidate(self, candidates, *, deterministic=False):
        return candidates[0], 0.0

    def operation_sample(
        self,
        snapshot,
        candidates,
        oracle,
        *,
        stop_legal,
        deterministic=False,
    ):
        ordered = tuple(sorted(candidates, key=lambda operation: operation.canonical_key))
        chosen = next(
            operation for operation in ordered
            if operation.request_id == "r0"
        ) if any(operation.request_id == "r0" for operation in ordered) else ordered[0]
        index = ordered.index(chosen)
        feature = self.encoder.encode(snapshot, ordered)
        return PolicySample(
            PolicyAction(None, (chosen.op_id,), False),
            0.0,
            0.0,
            feature,
            tuple(operation.op_id for operation in ordered),
            tuple(operation.ordinal for operation in ordered),
            tuple(range(len(ordered))),
            (index,),
        )

    def update(self, transitions, *args, **kwargs):
        self.captured = tuple(transitions)
        return {}

    def update_routes(self, *args, **kwargs):
        return {}


class CAAPPORolloutTrainerTests(unittest.TestCase):
    def test_drop_reward_does_not_shift_later_operation_return(self):
        spec = EpisodeSpec(
            seed=701,
            nodes=(0, 1, 2, 3),
            edges=((0, 1), (2, 3)),
            requests=(
                RequestSpec("r0", 0, 1, required_fidelity=1.0),
                RequestSpec("r1", 2, 3, required_fidelity=0.5),
            ),
            horizon=100,
            physical=PhysicalConfig(
                generation_probability=1.0,
                detector_efficiency=1.0,
                quantum_distance_m=1.0,
            ),
        )
        catalogue = build_route_construction_catalogue(
            spec.planning,
            candidate_count=1,
            construction_kinds=("left_deep",),
        )
        policy = _CapturePolicy()
        result = CAAPPORolloutTrainer(policy).run_episode(
            spec,
            catalogue,
            deterministic=False,
            update=True,
        )
        self.assertEqual(len(policy.captured), 2)
        self.assertAlmostEqual(policy.captured[0].return_value, result.reward)
        self.assertGreater(policy.captured[1].return_value, 0.0)
        self.assertEqual(policy.captured[0].risk_cost, 1.0)
        self.assertEqual(policy.captured[1].risk_cost, 0.0)
        self.assertEqual(policy.captured[0].risk_advantage, 1.0)
        self.assertEqual(policy.captured[1].risk_advantage, 0.0)
        self.assertTrue(all(
            transition.episode_risk_cost == 1.0
            for transition in policy.captured
        ))


if __name__ == "__main__":
    unittest.main()
