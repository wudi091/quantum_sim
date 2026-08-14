from dataclasses import replace
import unittest

import numpy as np
import torch

from algorithms.telgen.milp_imitation import (
    CONSTRAINT_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    VARIABLE_FEATURE_NAMES,
    CandidateConstraintGraph,
    CandidateConstraintGNN,
    apply_candidate_action,
    autoregressive_rollout,
    autoregressive_set_loss,
    batch_graph_samples,
    build_candidate_constraint_graph,
    build_sparse_packing_incidence,
    candidate_action_violation,
    graph_sample_from_solution,
    initial_autoregressive_state,
    selection_from_state,
)
from algorithms.telgen.milp_oracle import ConstructionAwareMILPOracle
from algorithms.telgen.time_expansion import expand_construction_candidates
from algorithms.telgen.train_milp_imitation import (
    _build_overfit_gate,
    _evaluate,
)
from algorithms.telgen.train_online_milp_gnn import (
    _random_feasible_baseline,
    _resolve_episode_split,
    _split_episode_seeds,
    _validation_key,
)
from qnet_core.construction_catalog import build_route_construction_catalogue
from qnet_core.planning_spec import RequestSpec
from qnet_core.resource_catalog import build_resource_capacities
from qnet_core.spec import EpisodeSpec, PhysicalConfig


def _sample():
    episode = EpisodeSpec(
        seed=71,
        nodes=(0, 1, 2),
        # Edge (0, 2) is deliberately absent from the selected route.  It
        # verifies that topology features come from the episode, not from the
        # candidate-path subgraph.
        edges=((0, 1), (0, 2)),
        requests=(RequestSpec("r0", 0, 1, ttl=2),),
        horizon=2,
        physical=PhysicalConfig(
            memory_capacity=1,
            node_memory_capacity=2,
            max_width=1,
        ),
    )
    capacities = build_resource_capacities(episode)
    candidates = build_route_construction_catalogue(
        episode.planning,
        candidate_count=1,
        construction_kinds=("balanced",),
        purification_kinds=("none",),
    )
    expansion = expand_construction_candidates(
        episode.planning,
        candidates,
        capacities,
        success_probability_estimates={
            candidate.candidate_id: 1.0 for candidate in candidates
        },
    )
    solution = ConstructionAwareMILPOracle().solve(
        expansion,
        capacities,
    )
    return graph_sample_from_solution(
        episode.seed,
        episode,
        solution,
        capacities,
    )


def _second_sample():
    episode = EpisodeSpec(
        seed=72,
        nodes=(0, 1, 2, 3),
        edges=((0, 1), (0, 2), (1, 2), (2, 3)),
        requests=(
            RequestSpec("r0", 0, 3, ttl=4),
            RequestSpec("r1", 1, 3, ttl=4),
        ),
        horizon=4,
        physical=PhysicalConfig(
            memory_capacity=1,
            node_memory_capacity=3,
            max_width=1,
        ),
    )
    capacities = build_resource_capacities(episode)
    candidates = build_route_construction_catalogue(
        episode.planning,
        candidate_count=2,
        construction_kinds=("left_deep", "balanced"),
        purification_kinds=("none",),
    )
    expansion = expand_construction_candidates(
        episode.planning,
        candidates,
        capacities,
        success_probability_estimates={
            candidate.candidate_id: 1.0 for candidate in candidates
        },
    )
    solution = ConstructionAwareMILPOracle().solve(
        expansion,
        capacities,
    )
    return graph_sample_from_solution(
        episode.seed,
        episode,
        solution,
        capacities,
    )


def _unlabelled_graph(sample, *, variables=None, reserved_usage=None):
    return CandidateConstraintGraph(
        seed=sample.seed,
        variable_features=sample.variable_features,
        constraint_features=sample.constraint_features,
        global_features=sample.global_features,
        edge_variable_indices=sample.edge_variable_indices,
        edge_constraint_indices=sample.edge_constraint_indices,
        edge_features=sample.edge_features,
        constraint_rhs=sample.constraint_rhs,
        variables=sample.variables if variables is None else variables,
        resource_capacities=sample.resource_capacities,
        reserved_usage=(
            sample.reserved_usage
            if reserved_usage is None
            else reserved_usage
        ),
        request_ids=sample.request_ids,
    )


class MILPImitationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = _sample()
        cls.second_sample = _second_sample()

    def test_validation_checkpoint_selection_prioritizes_pooled_throughput(self):
        high_pooled = {
            "pooled_throughput_ratio": 0.95,
            "mean_throughput_ratio": 0.90,
            "mean_latency_relative_gap_when_throughput_optimal": 0.2,
            "loss": 2.0,
        }
        high_unweighted_mean = {
            "pooled_throughput_ratio": 0.94,
            "mean_throughput_ratio": 1.0,
            "mean_latency_relative_gap_when_throughput_optimal": 0.0,
            "loss": 0.1,
        }
        self.assertGreater(
            _validation_key(high_pooled),
            _validation_key(high_unweighted_mean),
        )

    def test_graph_dimensions_labels_and_real_topology_features(self):
        sample = self.sample
        self.assertEqual(sample.request_ids, ("r0",))
        self.assertEqual(
            sample.variable_features.shape,
            (len(sample.variables), len(VARIABLE_FEATURE_NAMES)),
        )
        self.assertEqual(
            sample.constraint_features.shape,
            (len(sample.constraint_rhs), len(CONSTRAINT_FEATURE_NAMES)),
        )
        self.assertEqual(
            sample.global_features.shape,
            (len(GLOBAL_FEATURE_NAMES),),
        )
        self.assertEqual(
            sample.edge_features.shape,
            (len(sample.edge_variable_indices), 2),
        )
        self.assertTrue(np.all(np.isfinite(sample.variable_features)))
        self.assertTrue(np.all(np.isfinite(sample.constraint_features)))
        self.assertTrue(np.all(np.isin(sample.labels, (0.0, 1.0))))
        self.assertEqual(
            int(np.sum(sample.labels)),
            sample.optimal_completed_request_count,
        )

        source_index = VARIABLE_FEATURE_NAMES.index("source_degree")
        destination_index = VARIABLE_FEATURE_NAMES.index(
            "destination_degree"
        )
        # The true topology has degrees deg(0)=2 and deg(1)=1.  Reconstructing
        # topology from the sole route 0--1 would incorrectly return 1 and 1.
        self.assertTrue(np.allclose(
            sample.variable_features[:, source_index], 1.0
        ))
        self.assertTrue(np.allclose(
            sample.variable_features[:, destination_index], 0.5
        ))

    def test_unlabelled_online_graph_matches_teacher_graph_features(self):
        sample = self.second_sample
        episode = EpisodeSpec(
            seed=72,
            nodes=(0, 1, 2, 3),
            edges=((0, 1), (0, 2), (1, 2), (2, 3)),
            requests=(
                RequestSpec("r0", 0, 3, ttl=4),
                RequestSpec("r1", 1, 3, ttl=4),
            ),
            horizon=4,
            physical=PhysicalConfig(
                memory_capacity=1,
                node_memory_capacity=3,
                max_width=1,
            ),
        )
        graph = build_candidate_constraint_graph(
            sample.seed,
            episode,
            sample.variables,
            sample.resource_capacities,
        )
        self.assertTrue(np.array_equal(
            graph.variable_features, sample.variable_features
        ))
        self.assertTrue(np.array_equal(
            graph.constraint_features, sample.constraint_features
        ))
        self.assertTrue(np.array_equal(
            graph.global_features, sample.global_features
        ))
        batch = batch_graph_samples((graph,))
        self.assertEqual(batch.labels.numel(), 0)

    def test_milp_label_set_is_valid_under_masked_state_transitions(self):
        sample = self.sample
        incidence = build_sparse_packing_incidence(sample)
        state = initial_autoregressive_state(incidence)
        expected_indices = tuple(int(index) for index in np.flatnonzero(
            sample.labels > 0.5
        ))
        for index in expected_indices:
            self.assertIsNone(
                candidate_action_violation(incidence, state, index)
            )
            state = apply_candidate_action(incidence, state, index)
        selection = selection_from_state(sample, incidence, state)
        self.assertTrue(selection.feasible)
        self.assertEqual(selection.selected_indices, expected_indices)
        self.assertEqual(
            selection.completed_request_count,
            sample.optimal_completed_request_count,
        )
        self.assertAlmostEqual(
            selection.total_completion_latency,
            sample.optimal_total_completion_latency,
        )

    def test_gnn_forward_loss_and_backward_are_finite(self):
        graph = batch_graph_samples((self.sample,))
        model = CandidateConstraintGNN(hidden_dim=16, layers=2)
        actions = model(graph)
        loss, parts = autoregressive_set_loss(model, (self.sample,))

        self.assertEqual(actions.candidate_logits.shape, graph.labels.shape)
        self.assertEqual(actions.stop_logits.shape, (1,))
        self.assertTrue(bool(torch.all(torch.isfinite(
            actions.candidate_logits
        ))))
        self.assertTrue(bool(torch.all(torch.isfinite(actions.stop_logits))))
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(all(np.isfinite(value) for value in parts.values()))

        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(
            bool(torch.all(torch.isfinite(gradient)))
            for gradient in gradients
        ))

    def test_fixed_order_ablation_uses_the_same_feasible_action_space(self):
        model = CandidateConstraintGNN(hidden_dim=16, layers=2)
        loss, parts = autoregressive_set_loss(
            model,
            (self.second_sample,),
            target_mode="fixed_order",
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreater(parts["masked_candidate_fraction"], 0.0)
        with self.assertRaisesRegex(ValueError, "target mode"):
            autoregressive_set_loss(
                model,
                (self.second_sample,),
                target_mode="unknown",
            )

    def test_graph_encodes_construction_family_and_expected_mass_weights(self):
        sample = self.second_sample
        left_index = VARIABLE_FEATURE_NAMES.index("construction_left_deep")
        balanced_index = VARIABLE_FEATURE_NAMES.index(
            "construction_balanced"
        )
        for variable, features in zip(
            sample.variables, sample.variable_features
        ):
            self.assertEqual(
                features[left_index],
                float(variable.construction_kind == "left_deep"),
            )
            self.assertEqual(
                features[balanced_index],
                float(variable.construction_kind == "balanced"),
            )
        graph = batch_graph_samples((sample,))
        self.assertTrue(torch.allclose(
            graph.success_probabilities,
            torch.tensor([
                variable.expected_success_probability
                for variable in sample.variables
            ]),
        ))

    def test_batched_graphs_match_independent_forward_passes(self):
        torch.manual_seed(17)
        model = CandidateConstraintGNN(hidden_dim=16, layers=2).eval()
        first = batch_graph_samples((self.sample,))
        second = batch_graph_samples((self.second_sample,))
        combined = batch_graph_samples((self.sample, self.second_sample))
        with torch.no_grad():
            first_actions = model(first)
            second_actions = model(second)
            actual = model(combined)
        self.assertTrue(torch.allclose(
            actual.candidate_logits,
            torch.cat((
                first_actions.candidate_logits,
                second_actions.candidate_logits,
            )),
            atol=1e-6,
            rtol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            actual.stop_logits,
            torch.cat((
                first_actions.stop_logits,
                second_actions.stop_logits,
            )),
            atol=1e-6,
            rtol=1e-6,
        ))

    def test_zero_success_candidate_is_rejected_not_masked(self):
        variables = (
            replace(
                self.sample.variables[0],
                expected_success_probability=0.0,
            ),
            *self.sample.variables[1:],
        )
        graph = _unlabelled_graph(self.sample, variables=variables)
        incidence = build_sparse_packing_incidence(graph)
        state = initial_autoregressive_state(incidence)
        self.assertEqual(
            candidate_action_violation(incidence, state, 0),
            "nonpositive_success_probability",
        )
        with self.assertRaisesRegex(
            ValueError, "nonpositive_success_probability"
        ):
            apply_candidate_action(incidence, state, 0)

    def test_request_uniqueness_is_enforced_by_the_matrix_state(self):
        incidence = build_sparse_packing_incidence(self.sample)
        state = initial_autoregressive_state(incidence)
        state = apply_candidate_action(incidence, state, 0)
        self.assertTrue(all(
            candidate_action_violation(incidence, state, index) is not None
            for index in range(incidence.variable_count)
        ))

    def test_rollout_encodes_the_static_graph_once(self):
        class CountingGNN(CandidateConstraintGNN):
            def __init__(self):
                super().__init__(hidden_dim=16, layers=2)
                self.encode_count = 0

            def encode(self, graph):
                self.encode_count += 1
                return super().encode(graph)

        model = CountingGNN().eval()
        rollout = autoregressive_rollout(model, self.second_sample)
        self.assertEqual(model.encode_count, 1)

    def test_stop_is_a_learned_model_action(self):
        model = CandidateConstraintGNN(hidden_dim=16, layers=1).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.candidate_action_head.layers[-1].bias.fill_(-10.0)
            model.stop_action_head.layers[-1].bias.fill_(10.0)
        rollout = autoregressive_rollout(model, self.second_sample)
        self.assertTrue(rollout.stopped_by_model)
        self.assertEqual(rollout.selection.selected_variables, ())

    def test_non_unit_milp_weights_are_supported_when_targets_match(self):
        weighted = replace(
            self.sample.variables[0],
            expected_success_probability=0.5,
        )
        variables = (weighted, *self.sample.variables[1:])
        selected = tuple(
            variable
            for variable, label in zip(variables, self.sample.labels)
            if label > 0.5
        )
        updated = replace(
            self.sample,
            variables=variables,
            optimal_expected_completed_request_mass=sum(
                variable.expected_success_probability
                for variable in selected
            ),
            optimal_total_completion_latency=sum(
                variable.expected_success_probability
                * variable.completion_latency
                for variable in selected
            ),
        )
        self.assertLess(
            updated.optimal_expected_completed_request_mass,
            self.sample.optimal_expected_completed_request_mass,
        )

    def test_evaluation_reports_masked_autoregressive_feasibility(self):
        model = CandidateConstraintGNN(hidden_dim=16, layers=1)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.candidate_action_head.layers[-1].bias.fill_(10.0)
            model.stop_action_head.layers[-1].bias.fill_(-10.0)
        metrics = _evaluate(
            model,
            (self.sample,),
            device=torch.device("cpu"),
        )
        self.assertEqual(metrics["selection_feasible_rate"], 1.0)
        self.assertNotIn("classification_threshold", metrics)
        self.assertNotIn("post_projection_feasible_rate", metrics)

    def test_overfit_gate_requires_every_instance_to_be_optimal(self):
        passing = {
            "minimum_throughput_ratio": 1.0,
            "lexicographic_objective_optimal_rate": 1.0,
            "selection_feasible_rate": 1.0,
        }
        self.assertTrue(_build_overfit_gate(passing)["passed"])
        for key in passing:
            with self.subTest(key=key):
                failing = dict(passing)
                failing[key] = 0.99
                self.assertFalse(_build_overfit_gate(failing)["passed"])

    def test_rollout_masks_duplicate_and_capacity_violating_actions(self):
        model = CandidateConstraintGNN(hidden_dim=16, layers=1).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.candidate_action_head.layers[-1].bias.fill_(10.0)
            model.stop_action_head.layers[-1].bias.fill_(-10.0)
        rollout = autoregressive_rollout(model, self.second_sample)
        self.assertTrue(rollout.selection.feasible)
        self.assertEqual(
            len(set(rollout.action_indices)),
            len(rollout.action_indices),
        )
        self.assertIsNone(rollout.invalid_action_index)
        self.assertIsNone(rollout.invalid_action_reason)
        incidence = build_sparse_packing_incidence(self.second_sample)
        state = initial_autoregressive_state(incidence)
        for action_index in rollout.action_indices:
            self.assertIsNone(candidate_action_violation(
                incidence,
                state,
                action_index,
            ))
            state = apply_candidate_action(
                incidence,
                state,
                action_index,
            )

    def test_training_masks_invalid_actions_from_later_states(self):
        model = CandidateConstraintGNN(hidden_dim=16, layers=1)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.candidate_action_head.layers[-1].bias.fill_(10.0)
            model.stop_action_head.layers[-1].bias.fill_(-10.0)
        loss, parts = autoregressive_set_loss(
            model, (self.second_sample,)
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreater(parts["masked_candidate_fraction"], 0.0)
        self.assertLess(parts["valid_candidate_fraction"], 1.0)
        self.assertNotIn("validity_auxiliary_nll", parts)

    def test_training_penalizes_stop_while_teacher_actions_remain(self):
        model = CandidateConstraintGNN(hidden_dim=16, layers=1)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.candidate_action_head.layers[-1].bias.fill_(-10.0)
            model.stop_action_head.layers[-1].bias.fill_(10.0)
        loss, parts = autoregressive_set_loss(
            model, (self.second_sample,)
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreater(parts["candidate_set_nll"], 10.0)
        self.assertGreater(parts["masked_candidate_fraction"], 0.0)

    def test_episode_split_is_disjoint_and_deterministic(self):
        seeds = tuple(range(10))
        first = _split_episode_seeds(
            seeds,
            validation_fraction=0.2,
            test_fraction=0.2,
            random_seed=7,
        )
        second = _split_episode_seeds(
            seeds,
            validation_fraction=0.2,
            test_fraction=0.2,
            random_seed=7,
        )
        self.assertEqual(first, second)
        train, validation, test = first
        self.assertEqual(
            set(train) | set(validation) | set(test), set(seeds)
        )
        self.assertFalse(set(train) & set(validation))
        self.assertFalse(set(train) & set(test))
        self.assertFalse(set(validation) & set(test))

    def test_explicit_episode_split_holds_out_requested_topologies(self):
        train, validation, test = _resolve_episode_split(
            tuple(range(8)),
            validation_fraction=0.2,
            test_fraction=0.2,
            random_seed=7,
            validation_seeds=(5,),
            test_seeds=(6, 7),
        )
        self.assertEqual(validation, (5,))
        self.assertEqual(test, (6, 7))
        self.assertEqual(train, (0, 1, 2, 3, 4))
        with self.assertRaisesRegex(ValueError, "unknown explicit"):
            _resolve_episode_split(
                tuple(range(8)),
                validation_fraction=0.2,
                test_fraction=0.2,
                random_seed=7,
                validation_seeds=(5,),
                test_seeds=(8,),
            )

    def test_random_feasible_baseline_uses_the_same_capacity_contract(self):
        metrics = _random_feasible_baseline(
            (self.sample, self.second_sample),
            trials=3,
            random_seed=11,
        )
        self.assertEqual(metrics["trials"], 3)
        self.assertEqual(metrics["feasible_rate"], 1.0)
        self.assertGreaterEqual(
            metrics["mean_pooled_throughput_ratio"],
            0.0,
        )
        self.assertLessEqual(
            metrics["mean_pooled_throughput_ratio"],
            1.0 + 1e-7,
        )

    def test_duplicate_matrix_edges_are_aggregated_before_validation(self):
        graph = _unlabelled_graph(self.sample)
        edge_index = next(
            index
            for index, variable_index in enumerate(
                graph.edge_variable_indices
            )
            if int(variable_index) == 0
        )
        duplicated = replace(
            graph,
            edge_variable_indices=np.append(
                graph.edge_variable_indices,
                graph.edge_variable_indices[edge_index],
            ),
            edge_constraint_indices=np.append(
                graph.edge_constraint_indices,
                graph.edge_constraint_indices[edge_index],
            ),
            edge_features=np.vstack((
                graph.edge_features,
                graph.edge_features[edge_index],
            )),
        )
        incidence = build_sparse_packing_incidence(duplicated)
        self.assertEqual(
            candidate_action_violation(
                incidence, initial_autoregressive_state(incidence), 0
            ),
            "packing_constraint_violation",
        )

    def test_zero_residual_capacity_rejects_every_incident_candidate(self):
        graph = _unlabelled_graph(self.second_sample)
        resource_flag = CONSTRAINT_FEATURE_NAMES.index("is_resource_time")
        row = int(np.flatnonzero(
            graph.constraint_features[:, resource_flag] > 0.5
        )[0])
        rhs = graph.constraint_rhs.copy()
        rhs[row] = 0.0
        constrained = replace(graph, constraint_rhs=rhs)
        incidence = build_sparse_packing_incidence(constrained)
        state = initial_autoregressive_state(incidence)
        incident = set(int(index) for index in graph.edge_variable_indices[
            graph.edge_constraint_indices == row
        ])
        self.assertTrue(incident)
        self.assertTrue(all(
            candidate_action_violation(incidence, state, index)
            == "packing_constraint_violation"
            for index in incident
        ))

    def test_unordered_set_loss_is_permutation_equivariant(self):
        sample = self.second_sample
        order = np.arange(len(sample.variables))[::-1]
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        permuted = replace(
            sample,
            variable_features=sample.variable_features[order],
            labels=sample.labels[order],
            variables=tuple(sample.variables[index] for index in order),
            edge_variable_indices=inverse[sample.edge_variable_indices],
        )
        torch.manual_seed(31)
        model = CandidateConstraintGNN(hidden_dim=16, layers=2).eval()
        first, _ = autoregressive_set_loss(model, (sample,))
        second, _ = autoregressive_set_loss(model, (permuted,))
        self.assertAlmostEqual(
            float(first.detach()), float(second.detach()), places=6
        )

    def test_evaluation_loss_is_independent_of_graph_batching(self):
        torch.manual_seed(19)
        model = CandidateConstraintGNN(hidden_dim=16, layers=2).eval()
        samples = (self.sample, self.second_sample)
        individual = _evaluate(
            model,
            samples,
            device=torch.device("cpu"),
            sample_batch_size=1,
        )
        combined = _evaluate(
            model,
            samples,
            device=torch.device("cpu"),
            sample_batch_size=2,
        )
        for key in (
            "loss",
            "autoregressive_nll",
            "candidate_set_nll",
            "stop_nll",
        ):
            self.assertAlmostEqual(
                float(individual[key]), float(combined[key]), places=6
            )


if __name__ == "__main__":
    unittest.main()
