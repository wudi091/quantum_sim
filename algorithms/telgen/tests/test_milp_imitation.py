from dataclasses import replace
import unittest

import numpy as np
import torch

from algorithms.telgen.hard_decoder import validate_decoded_selection
from algorithms.telgen.milp_imitation import (
    CONSTRAINT_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    VARIABLE_FEATURE_NAMES,
    CandidateConstraintGNN,
    batch_graph_samples,
    graph_sample_from_solution,
    greedy_decode_scores,
    imitation_loss,
)
from algorithms.telgen.milp_oracle import ConstructionAwareMILPOracle
from algorithms.telgen.time_expansion import expand_construction_candidates
from algorithms.telgen.train_milp_imitation import (
    _build_overfit_gate,
    _evaluate,
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


class MILPImitationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = _sample()
        cls.second_sample = _second_sample()

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

    def test_milp_labels_round_trip_through_feasible_projection(self):
        sample = self.sample
        decoded = greedy_decode_scores(
            sample,
            sample.labels,
            threshold=0.5,
        )
        expected_ids = tuple(sorted(
            variable.variable_id
            for variable, label in zip(sample.variables, sample.labels)
            if label > 0.5
        ))
        self.assertTrue(decoded.feasible)
        self.assertEqual(decoded.selected_variable_ids, expected_ids)
        self.assertEqual(
            decoded.completed_request_count,
            sample.optimal_completed_request_count,
        )
        self.assertAlmostEqual(
            decoded.total_completion_latency,
            sample.optimal_total_completion_latency,
        )

    def test_gnn_forward_loss_and_backward_are_finite(self):
        graph = batch_graph_samples((self.sample,))
        model = CandidateConstraintGNN(hidden_dim=16, layers=2)
        logits = model(graph)
        loss, parts = imitation_loss(logits, graph)

        self.assertEqual(logits.shape, graph.labels.shape)
        self.assertTrue(bool(torch.all(torch.isfinite(logits))))
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

    def test_batched_graphs_match_independent_forward_passes(self):
        torch.manual_seed(17)
        model = CandidateConstraintGNN(hidden_dim=16, layers=2).eval()
        first = batch_graph_samples((self.sample,))
        second = batch_graph_samples((self.second_sample,))
        combined = batch_graph_samples((self.sample, self.second_sample))
        with torch.no_grad():
            expected = torch.cat((model(first), model(second)))
            actual = model(combined)
        self.assertTrue(torch.allclose(
            actual, expected, atol=1e-6, rtol=1e-6
        ))

    def test_loss_rejects_invalid_auxiliary_weights_and_logits(self):
        graph = batch_graph_samples((self.sample,))
        logits = torch.zeros_like(graph.labels)
        for name, kwargs in (
            ("negative constraint", {"constraint_weight": -1.0}),
            ("nan constraint", {"constraint_weight": float("nan")}),
            ("negative count", {"count_weight": -1.0}),
            ("infinite count", {"count_weight": float("inf")}),
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    imitation_loss(logits, graph, **kwargs)
        invalid_logits = logits.clone()
        invalid_logits[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "logits must be finite"):
            imitation_loss(invalid_logits, graph)

    def test_non_unit_milp_weights_are_rejected(self):
        weighted = replace(
            self.sample.variables[0],
            expected_success_probability=0.5,
        )
        with self.assertRaisesRegex(ValueError, "unit weights"):
            replace(
                self.sample,
                variables=(weighted, *self.sample.variables[1:]),
            )

    def test_raw_and_projected_feasibility_are_reported_separately(self):
        class AllPositiveModel(torch.nn.Module):
            def forward(self, graph):
                return torch.full_like(graph.labels, 10.0)

        metrics = _evaluate(
            AllPositiveModel(),
            (self.sample,),
            threshold=0.5,
            device=torch.device("cpu"),
        )
        self.assertEqual(metrics["raw_threshold_feasible_rate"], 0.0)
        self.assertGreater(
            metrics["mean_raw_threshold_violation_count"], 0.0
        )
        self.assertEqual(metrics["post_projection_feasible_rate"], 1.0)

    def test_overfit_gate_requires_every_instance_to_be_optimal(self):
        passing = {
            "minimum_decoded_throughput_ratio": 1.0,
            "lexicographic_objective_optimal_rate": 1.0,
            "raw_threshold_feasible_rate": 1.0,
            "post_projection_feasible_rate": 1.0,
        }
        self.assertTrue(_build_overfit_gate(passing)["passed"])
        for key in passing:
            with self.subTest(key=key):
                failing = dict(passing)
                failing[key] = 0.99
                self.assertFalse(_build_overfit_gate(failing)["passed"])

    def test_projection_is_feasible_for_arbitrary_finite_scores(self):
        sample = self.sample
        rng = np.random.default_rng(99)
        score_vectors = (
            np.ones(len(sample.variables)),
            np.linspace(0.0, 1.0, len(sample.variables)),
            *(rng.random(len(sample.variables)) for _ in range(20)),
        )
        for scores in score_vectors:
            with self.subTest(scores=scores.tolist()):
                decoded = greedy_decode_scores(
                    sample, scores, threshold=0.0
                )
                report = validate_decoded_selection(
                    decoded.selected_variables,
                    sample.resource_capacities,
                )
                self.assertTrue(decoded.feasible)
                self.assertTrue(report.feasible)


if __name__ == "__main__":
    unittest.main()
