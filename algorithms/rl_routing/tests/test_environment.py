import unittest

from algorithms.rl_routing.environment import (
    STOP_ACTION,
    ConstructionAwareRoutingEnvironment,
    FeasiblePlanBuilder,
    RoutingAction,
)
from algorithms.routing_core.execution import OnlineExecutionConfig
from qnet_core.planning_spec import RequestSpec
from qnet_core.spec import EpisodeSpec, PhysicalConfig


def deterministic_physical(**overrides):
    values = dict(
        generation_probability=1.0,
        swap_probability=1.0,
        detector_efficiency=1.0,
        bsm_success_probability=1.0,
        quantum_distance_m=1.0,
        slot_duration_ps=1_000_000,
    )
    values.update(overrides)
    return PhysicalConfig(**values)


class FeasiblePlanBuilderTests(unittest.TestCase):
    def make_environment(self, request_count=2):
        spec = EpisodeSpec(
            seed=8100,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=tuple(
                RequestSpec(f"r{index}", 0, 1, ttl=4)
                for index in range(request_count)
            ),
            horizon=4,
            physical=deterministic_physical(
                memory_capacity=1,
                node_memory_capacity=1,
                max_width=1,
            ),
        )
        return ConstructionAwareRoutingEnvironment(
            spec,
            OnlineExecutionConfig(
                decision_interval=2,
                path_candidate_count=1,
                construction_kinds=("balanced",),
                purification_kinds=("none",),
            ),
        )

    def test_mask_enforces_request_uniqueness_and_residual_capacity(self):
        environment = self.make_environment()
        observation = environment.observe()
        builder = FeasiblePlanBuilder(observation)
        first = next(
            variable
            for variable in observation.variables
            if variable.request_id == "r0" and variable.start_slot == 0
        )
        same_request = next(
            variable
            for variable in observation.variables
            if variable.request_id == "r0" and variable.start_slot == 1
        )
        conflicting = next(
            variable
            for variable in observation.variables
            if variable.request_id == "r1" and variable.start_slot == 0
        )

        builder.select(first.variable_id)
        self.assertFalse(builder.can_select(same_request.variable_id))
        self.assertFalse(builder.can_select(conflicting.variable_id))
        self.assertIn(STOP_ACTION, builder.legal_action_ids())

    def test_builder_never_adds_an_action_for_the_policy(self):
        environment = self.make_environment()
        observation = environment.observe()
        builder = FeasiblePlanBuilder(observation)
        action = builder.finish()
        self.assertEqual(action.variable_ids, ())
        self.assertEqual(action.action_ids, (STOP_ACTION,))


class ConstructionAwareRoutingEnvironmentTests(unittest.TestCase):
    def test_selected_joint_plan_executes_through_sequence(self):
        spec = EpisodeSpec(
            seed=8200,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r0", 0, 1, ttl=4),),
            horizon=4,
            physical=deterministic_physical(),
        )
        environment = ConstructionAwareRoutingEnvironment(
            spec,
            OnlineExecutionConfig(
                decision_interval=1,
                path_candidate_count=1,
                construction_kinds=("balanced",),
                purification_kinds=("none",),
            ),
        )
        observation = environment.observe()
        variable = next(
            item for item in observation.variables if item.start_slot == 0
        )
        environment.step(RoutingAction(
            decision_slot=0,
            action_ids=(variable.variable_id, STOP_ACTION),
        ))
        while not environment.done:
            current = environment.observe()
            environment.step(RoutingAction(
                decision_slot=current.slot,
                action_ids=(STOP_ACTION,),
            ))

        result = environment.result()
        self.assertEqual(result.metrics["completed_requests"], 1.0)
        self.assertEqual(len(result.attempts), 1)
        self.assertTrue(result.attempts[0].success)
        self.assertTrue(result.event_trace)
        self.assertAlmostEqual(environment.reward_identity_error(), 0.0)

    def test_noop_return_matches_censored_completion_latency(self):
        spec = EpisodeSpec(
            seed=8201,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r0", 0, 1, ttl=2),),
            horizon=5,
            physical=deterministic_physical(),
        )
        environment = ConstructionAwareRoutingEnvironment(
            spec,
            OnlineExecutionConfig(decision_interval=1),
        )
        while not environment.done:
            observation = environment.observe()
            environment.step(RoutingAction(
                decision_slot=observation.slot,
                action_ids=(STOP_ACTION,),
            ))

        self.assertEqual(environment.result().metrics["completed_requests"], 0.0)
        self.assertEqual(sum(
            transition.terminal_censoring_slots
            for transition in environment.transitions
        ), 3.0)
        self.assertAlmostEqual(environment.reward_identity_error(), 0.0)

    def test_future_requests_are_not_exposed_as_actions(self):
        spec = EpisodeSpec(
            seed=8202,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(
                RequestSpec("r0", 0, 1, arrival=0, ttl=5),
                RequestSpec("r1", 0, 1, arrival=2, ttl=3),
            ),
            horizon=5,
            physical=deterministic_physical(node_memory_capacity=4),
        )
        environment = ConstructionAwareRoutingEnvironment(
            spec,
            OnlineExecutionConfig(decision_interval=1),
        )
        observation = environment.observe()
        self.assertEqual(observation.eligible_request_ids, ("r0",))
        self.assertEqual(
            {variable.request_id for variable in observation.variables},
            {"r0"},
        )


if __name__ == "__main__":
    unittest.main()
