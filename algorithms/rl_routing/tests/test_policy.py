import unittest
from dataclasses import replace

import torch

from algorithms.rl_routing.environment import (
    STOP_ACTION,
    ConstructionAwareRoutingEnvironment,
    FeasiblePlanBuilder,
)
from algorithms.rl_routing.graph import (
    CANDIDATE_NODE,
    PHYSICAL_NODE,
    REQUEST_NODE,
    RESOURCE_SLOT_NODE,
    RoutingGraph,
    _hierarchical_candidate_masses,
    build_routing_graph,
)
from algorithms.rl_routing.policy import ARCQPolicy, RelationalMessageLayer
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


class CapacityAwareAggregationTests(unittest.TestCase):
    def test_resource_demand_is_capacity_weighted_and_additive(self):
        messages = torch.tensor(((2.0, 4.0), (2.0, 4.0), (9.0, 9.0)))
        destinations = torch.tensor((0, 0, 0))
        edge_types = torch.tensor((10, 10, 10))
        edge_features = torch.tensor((
            (0.5, 0.0, 0.0, 0.0, 1.0),
            (0.5, 0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 0.0, 0.0),
        ))

        aggregate = RelationalMessageLayer._aggregate_messages(
            messages,
            destinations,
            edge_types,
            edge_features,
            node_count=1,
        )

        self.assertTrue(torch.equal(aggregate, torch.tensor(((2.0, 4.0),))))

    def test_non_resource_relations_remain_degree_normalized(self):
        messages = torch.tensor(((2.0, 4.0), (2.0, 4.0)))
        aggregate = RelationalMessageLayer._aggregate_messages(
            messages,
            torch.tensor((0, 0)),
            torch.tensor((0, 0)),
            torch.zeros((2, 5)),
            node_count=1,
        )

        self.assertTrue(torch.equal(aggregate, torch.tensor(((2.0, 4.0),))))


class RoutingGraphTests(unittest.TestCase):
    def test_hierarchical_prior_assigns_one_mass_unit_per_request(self):
        _environment, observation = make_observation()
        legal_ids = set(FeasiblePlanBuilder(observation).legal_action_ids())
        masses = _hierarchical_candidate_masses(
            observation.variables,
            legal_ids,
        )
        for request_id in observation.eligible_request_ids:
            request_mass = sum(
                masses.get(variable.variable_id, 0.0)
                for variable in observation.variables
                if variable.request_id == request_id
            )
            self.assertAlmostEqual(request_mass, 1.0)

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
        self.assertEqual(
            graph.request_node_indices.numel(),
            len(observation.visible_request_ids),
        )

    def test_start_offset_uses_the_local_decision_window(self):
        _environment, observation = make_observation()
        graph = build_routing_graph(observation)
        start_features = graph.node_features[
            graph.candidate_node_indices, 6
        ]
        self.assertAlmostEqual(float(start_features.min().item()), 0.0)
        self.assertAlmostEqual(float(start_features.max().item()), 1.0)

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

    def test_resource_pressure_tracks_only_currently_legal_options(self):
        _environment, observation = make_observation()
        initial = build_routing_graph(observation)
        builder = FeasiblePlanBuilder(observation)
        builder.select(observation.variables[0].variable_id)
        residual = build_routing_graph(observation, builder)
        initial_pressure = initial.node_features[
            initial.node_types == RESOURCE_SLOT_NODE, 17
        ].sum()
        residual_pressure = residual.node_features[
            residual.node_types == RESOURCE_SLOT_NODE, 17
        ].sum()

        self.assertLess(float(residual_pressure), float(initial_pressure))

    def test_actor_critic_is_equivariant_to_tensor_node_order(self):
        _environment, observation = make_observation()
        graph = build_routing_graph(observation)
        torch.manual_seed(8304)
        policy = ARCQPolicy(hidden_dim=24, message_passing_layers=2)
        policy.eval()
        permutation = torch.randperm(graph.node_features.shape[0])
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(permutation.numel())
        permuted = RoutingGraph(
            node_features=graph.node_features[permutation],
            node_types=graph.node_types[permutation],
            edge_index=inverse[graph.edge_index],
            edge_types=graph.edge_types,
            edge_features=graph.edge_features,
            global_features=graph.global_features,
            request_node_indices=inverse[graph.request_node_indices],
            request_ids=graph.request_ids,
            candidate_node_indices=inverse[graph.candidate_node_indices],
            candidate_variable_ids=graph.candidate_variable_ids,
            candidate_request_ids=graph.candidate_request_ids,
            candidate_route_nodes=graph.candidate_route_nodes,
            candidate_construction_ids=graph.candidate_construction_ids,
            candidate_legal_mask=graph.candidate_legal_mask,
        )
        with torch.no_grad():
            original_embeddings, original_context, original_stop = (
                policy.actor_critic.actor_forward(graph)
            )
            permuted_embeddings, permuted_context, permuted_stop = (
                policy.actor_critic.actor_forward(permuted)
            )
            original_value = policy.actor_critic.critic_forward(graph)
            permuted_value = policy.actor_critic.critic_forward(permuted)
        self.assertTrue(torch.allclose(
            original_embeddings,
            permuted_embeddings[inverse],
            atol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            original_context,
            permuted_context,
            atol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            original_stop,
            permuted_stop,
            atol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            original_value,
            permuted_value,
            atol=1e-6,
        ))


class ARCQPolicyTests(unittest.TestCase):
    def test_hierarchical_action_probabilities_sum_to_one(self):
        _environment, observation = make_observation()
        torch.manual_seed(8305)
        policy = ARCQPolicy(hidden_dim=24, message_passing_layers=2)
        graph = build_routing_graph(observation)
        with torch.no_grad():
            action_ids = (
                *(
                    action_id
                    for action_id, legal in zip(
                        graph.candidate_variable_ids,
                        graph.candidate_legal_mask.tolist(),
                        strict=True,
                    )
                    if legal
                ),
                STOP_ACTION,
            )
            probabilities = tuple(
                policy._token_choice(
                    graph,
                    action_id=action_id,
                    include_value=False,
                    include_entropy=False,
                )[1].exp()
                for action_id in action_ids
            )
        self.assertTrue(torch.allclose(
            torch.stack(probabilities).sum(),
            torch.ones(()),
            atol=1e-5,
        ))

    def test_reported_entropy_matches_the_joint_action_distribution(self):
        _environment, observation = make_observation()
        torch.manual_seed(8307)
        policy = ARCQPolicy(hidden_dim=24, message_passing_layers=2)
        graph = build_routing_graph(observation)
        with torch.no_grad():
            action_ids = (
                *(
                    action_id
                    for action_id, legal in zip(
                        graph.candidate_variable_ids,
                        graph.candidate_legal_mask.tolist(),
                        strict=True,
                    )
                    if legal
                ),
                STOP_ACTION,
            )
            evaluations = tuple(
                policy._token_choice(
                    graph,
                    action_id=action_id,
                    include_value=False,
                )
                for action_id in action_ids
            )
            log_probabilities = torch.stack(tuple(
                evaluation[1] for evaluation in evaluations
            ))
            enumerated_entropy = -(
                log_probabilities.exp() * log_probabilities
            ).sum()
        self.assertTrue(torch.allclose(
            evaluations[0][2],
            enumerated_entropy,
            atol=1e-5,
        ))

    def test_request_probability_is_not_biased_by_candidate_count(self):
        _environment, observation = make_observation()
        graph = build_routing_graph(observation)
        legal_mask = torch.zeros_like(graph.candidate_legal_mask)
        first_r1 = None
        for index, request_id in enumerate(graph.candidate_request_ids):
            if request_id == "r0":
                legal_mask[index] = True
            elif request_id == "r1" and first_r1 is None:
                legal_mask[index] = True
                first_r1 = index
        graph = replace(graph, candidate_legal_mask=legal_mask)
        torch.manual_seed(8306)
        policy = ARCQPolicy(hidden_dim=24, message_passing_layers=2)
        with torch.no_grad():
            for parameter in policy.actor_parameters():
                parameter.zero_()
            probability_by_request = {"r0": 0.0, "r1": 0.0}
            for index, legal in enumerate(legal_mask.tolist()):
                if not legal:
                    continue
                action_id = graph.candidate_variable_ids[index]
                log_probability = policy._token_choice(
                    graph,
                    action_id=action_id,
                    include_value=False,
                    include_entropy=False,
                )[1]
                request_id = graph.candidate_request_ids[index]
                probability_by_request[request_id] += float(
                    log_probability.exp().item()
                )
            stop_probability = float(policy._token_choice(
                graph,
                action_id=STOP_ACTION,
                include_value=False,
                include_entropy=False,
            )[1].exp().item())
        self.assertGreater(
            sum(request_id == "r0" for request_id in graph.candidate_request_ids),
            1,
        )
        self.assertAlmostEqual(probability_by_request["r0"], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(probability_by_request["r1"], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(stop_probability, 1.0 / 3.0, places=5)

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
