import unittest

import numpy as np
import torch

from algorithms.telgen.ipm_trajectory_pilot import (
    TELGENPaperGNN,
    TELGENQuantumAdapterGNN,
    _resolve_device,
    _sample_loss,
    make_samples,
    round_continuous_plan,
    solve_scipy_ipm_trajectory,
)


class IPMTrajectoryPilotTests(unittest.TestCase):
    def test_explicit_unavailable_cuda_does_not_silently_fall_back(self):
        if not torch.cuda.is_available():
            with self.assertRaises(RuntimeError):
                _resolve_device("cuda")
        self.assertEqual(_resolve_device("cpu"), torch.device("cpu"))

    def test_scipy_teacher_records_and_resamples_callback_trajectory(self):
        trajectory = solve_scipy_ipm_trajectory(
            np.asarray([[1.0, 1.0]]),
            np.asarray([1.0]),
            np.asarray([-1.0, -1.0]),
            outer_steps=4,
        )
        self.assertEqual(trajectory.points.shape, (4, 2))
        self.assertGreaterEqual(len(trajectory.raw_points), 2)
        self.assertTrue(np.isfinite(trajectory.points).all())
        self.assertTrue(np.isfinite(trajectory.raw_points).all())
        self.assertAlmostEqual(trajectory.lp_optimum, -1.0, places=5)

    def test_graph_uses_paper_matrix_statistics_and_six_relations(self):
        sample = make_samples(
            topology="waxman",
            node_count=[10, 12],
            sample_count=1,
            seed=20260826,
            request_count=2,
            horizon=6,
            path_count=2,
            construction_plan_count=3,
            outer_steps=4,
        )[0]
        graph = sample.graph
        self.assertEqual(graph.variable_features.shape[1], 2)
        self.assertEqual(graph.constraint_features.shape[1], 2)
        self.assertEqual(graph.objective_features.shape, (1, 2))
        self.assertEqual(
            len(graph.variable_edge_indices),
            len(graph.constraint_edge_indices),
        )
        self.assertEqual(graph.edge_features.shape[1], 1)
        self.assertEqual(graph.objective_coefficients.shape[0], graph.matrix.shape[1])

    def test_paper_model_supports_variable_graph_sizes_and_backprop(self):
        samples = make_samples(
            topology="waxman",
            node_count=[10, 12],
            sample_count=2,
            seed=100,
            request_count=2,
            horizon=6,
            path_count=2,
            construction_plan_count=3,
            outer_steps=4,
        )
        model = TELGENPaperGNN(
            hidden_dim=12,
            inner_layers=2,
            message_mlp_layers=2,
            prediction_layers=2,
        )
        for sample in samples:
            model.zero_grad(set_to_none=True)
            prediction = model(sample.graph, steps=4)
            self.assertEqual(
                tuple(prediction.shape),
                (4, sample.graph.matrix.shape[1]),
            )
            self.assertTrue(bool(torch.isfinite(prediction).all()))
            self.assertTrue(bool(torch.all(prediction >= 0.0)))
            loss, parts = _sample_loss(prediction, sample)
            self.assertTrue(bool(torch.isfinite(loss)))
            self.assertTrue(np.isfinite(list(parts.values())).all())
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(all(bool(torch.isfinite(item).all()) for item in gradients))

    def test_paper_model_enforces_request_mass_at_every_iteration(self):
        sample = make_samples(
            topology="waxman",
            node_count=16,
            sample_count=1,
            seed=101,
            request_count=5,
            horizon=8,
            path_count=3,
            construction_plan_count=3,
            outer_steps=5,
            endpoint_mode="cut_hotspot",
        )[0]
        model = TELGENPaperGNN(
            hidden_dim=12,
            inner_layers=2,
            message_mlp_layers=2,
            prediction_layers=2,
        )
        with torch.no_grad():
            trace = model(sample.graph, steps=5)
        groups, _, _ = model._request_partition(
            sample.graph, next(model.parameters()).device
        )
        for point in trace:
            for indices in groups:
                self.assertLessEqual(
                    float(point.index_select(0, indices).sum()),
                    1.0 + 1e-6,
                )

    def test_default_training_samples_use_multiple_graphs(self):
        samples = make_samples(
            topology="waxman",
            node_count=[10, 12],
            sample_count=5,
            seed=300,
            request_count=2,
            horizon=6,
            path_count=2,
            construction_plan_count=3,
            outer_steps=2,
        )
        signatures = {
            (sample.node_count, sample.topology_signature)
            for sample in samples
        }
        self.assertGreaterEqual(len(signatures), 2)

    def test_fixed_topology_is_explicit_stress_protocol(self):
        samples = make_samples(
            topology="waxman",
            node_count=10,
            sample_count=3,
            seed=400,
            topology_seed=700,
            fixed_topology=True,
            request_count=2,
            horizon=6,
            path_count=2,
            construction_plan_count=3,
            outer_steps=2,
        )
        self.assertEqual(
            len({sample.topology_signature for sample in samples}), 1
        )

    def test_cut_hotspot_samples_create_shared_active_constraints(self):
        sample = make_samples(
            topology="waxman",
            node_count=24,
            sample_count=1,
            seed=920,
            request_count=50,
            horizon=12,
            path_count=3,
            construction_plan_count=3,
            outer_steps=4,
            endpoint_mode="cut_hotspot",
        )[0]
        final = sample.trajectory.points[-1]
        request_mass = {}
        for value, variable in zip(final, sample.variables):
            request_mass[variable.request_id] = (
                request_mass.get(variable.request_id, 0.0) + float(value)
            )
        self.assertGreaterEqual(len(request_mass), 5)
        self.assertTrue(all(value <= 1.0 + 1e-6 for value in request_mass.values()))

    def test_quantum_semantic_extension_is_not_mixed_into_paper_model(self):
        with self.assertRaises(NotImplementedError):
            TELGENQuantumAdapterGNN()

    def test_shared_rounding_produces_capacity_safe_unique_requests(self):
        sample = make_samples(
            topology="waxman",
            node_count=10,
            sample_count=1,
            seed=500,
            request_count=3,
            horizon=6,
            path_count=2,
            construction_plan_count=3,
            outer_steps=4,
        )[0]
        teacher = round_continuous_plan(
            sample.trajectory.points[-1], sample
        )
        dense = round_continuous_plan(
            np.ones(len(sample.variables), dtype=float), sample
        )
        empty = round_continuous_plan(
            np.zeros(len(sample.variables), dtype=float), sample
        )
        self.assertTrue(teacher.feasible)
        self.assertTrue(dense.feasible)
        self.assertEqual(empty.completed_request_count, 0)
        dense_requests = [
            sample.variables[index].request_id
            for index in dense.selected_indices
        ]
        self.assertEqual(len(dense_requests), len(set(dense_requests)))


if __name__ == "__main__":
    unittest.main()
