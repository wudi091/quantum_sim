import unittest

import torch

from algorithms.rl_routing.environment import (
    ConstructionAwareRoutingEnvironment,
    FeasiblePlanBuilder,
)
from algorithms.rl_routing.graph import (
    CANDIDATE_NODE,
    PHYSICAL_NODE,
    REQUEST_NODE,
    RESOURCE_SLOT_NODE,
    build_routing_graph,
)
from algorithms.rl_routing.policy import ARCQPolicy
from algorithms.routing_core.execution import OnlineExecutionConfig
from qnet_core.planning_spec import RequestSpec
from qnet_core.spec import EpisodeSpec, PhysicalConfig


def make_observation():
    spec = EpisodeSpec(
        seed=8300,
        nodes=(0, 1, 2),
        edges=((0, 1), (1, 2), (0, 2)),
        requests=(
            RequestSpec("r0", 0, 2, ttl=5),
            RequestSpec("r1", 0, 2, ttl=5),
        ),
        horizon=5,
        physical=PhysicalConfig(
            generation_probability=1.0,
            swap_probability=1.0,
            detector_efficiency=1.0,
            bsm_success_probability=1.0,
            quantum_distance_m=1.0,
            node_memory_capacity=2,
            memory_capacity=1,
            max_width=1,
        ),
    )
    environment = ConstructionAwareRoutingEnvironment(
        spec,
        OnlineExecutionConfig(
            decision_interval=2,
            path_candidate_count=2,
            construction_kinds=("left_deep", "balanced"),
            purification_kinds=("none",),
        ),
    )
    return environment, environment.observe()


class RoutingGraphTests(unittest.TestCase):
    def test_graph_contains_all_four_semantic_node_types(self):
        _environment, observation = make_observation()
        graph = build_routing_graph(observation)
        self.assertEqual(graph.node_features.ndim, 2)
        self.assertEqual(graph.edge_index.shape[0], 2)
        self.assertEqual(
            set(graph.node_types.tolist()),
            {PHYSICAL_NODE, REQUEST_NODE, CANDIDATE_NODE, RESOURCE_SLOT_NODE},
        )
        self.assertEqual(
            graph.candidate_node_indices.numel(),
            len(observation.variables),
        )
        self.assertEqual(
            graph.candidate_legal_mask.shape[0],
            len(observation.variables),
        )

    def test_graph_tracks_autoregressive_residual_capacity(self):
        _environment, observation = make_observation()
        builder = FeasiblePlanBuilder(observation)
        chosen = observation.variables[0]
        builder.select(chosen.variable_id)
        graph = build_routing_graph(observation, builder)
        request_mask = torch.tensor([
            variable.request_id == chosen.request_id
            for variable in observation.variables
        ])
        self.assertFalse(bool(graph.candidate_legal_mask[request_mask].any()))


class ARCQPolicyTests(unittest.TestCase):
    def test_sampled_action_is_directly_feasible_without_decoder(self):
        environment, observation = make_observation()
        torch.manual_seed(8301)
        policy = ARCQPolicy(hidden_dim=24, message_passing_layers=2)
        policy.eval()
        sample = policy.sample_action(observation)
        builder = FeasiblePlanBuilder(observation)
        builder.apply(sample.action)
        transition = environment.step(sample.action)
        self.assertEqual(
            transition.observation.slot,
            sample.action.decision_slot,
        )

    def test_action_log_probability_can_be_recomputed_and_differentiated(self):
        _environment, observation = make_observation()
        torch.manual_seed(8302)
        policy = ARCQPolicy(hidden_dim=24, message_passing_layers=2)
        policy.eval()
        with torch.no_grad():
            sample = policy.sample_action(observation)
        self.assertEqual(len(sample.tokens), sample.token_count)
        self.assertEqual(
            tuple(token.action_id for token in sample.tokens),
            sample.action.action_ids,
        )
        for index, token in enumerate(sample.tokens):
            self.assertEqual(
                token.prefix_action_ids,
                sample.action.action_ids[:index],
            )
            recomputed = policy.evaluate_token(
                observation,
                token.prefix_action_ids,
                token.action_id,
            )
            self.assertTrue(torch.allclose(
                recomputed.log_probability,
                token.log_probability,
            ))
        evaluated = policy.evaluate_action(observation, sample.action)
        self.assertTrue(torch.isfinite(evaluated.log_probability))
        self.assertTrue(torch.isfinite(evaluated.value))
        loss = -evaluated.log_probability + evaluated.value.square()
        loss.backward()
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in policy.parameters()
        ))

    def test_deployment_action_does_not_require_critic_values(self):
        _environment, observation = make_observation()
        torch.manual_seed(8303)
        policy = ARCQPolicy(hidden_dim=24, message_passing_layers=2)
        policy.eval()
        with torch.no_grad():
            sample = policy.sample_action(
                observation,
                deterministic=True,
                include_value=False,
            )
        self.assertEqual(float(sample.value.item()), 0.0)
        self.assertTrue(all(
            float(token.value.item()) == 0.0 for token in sample.tokens
        ))


if __name__ == "__main__":
    unittest.main()
