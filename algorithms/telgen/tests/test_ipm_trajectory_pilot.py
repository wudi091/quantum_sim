import unittest

import numpy as np
import torch

from algorithms.telgen.ipm_trajectory_pilot import (
    TELGENPaperGNN,
    TELGENQuantumAdapterGNN,
    _resolve_device,
    _sample_loss,
    make_samples,
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

    def test_quantum_semantic_extension_is_not_mixed_into_paper_model(self):
        with self.assertRaises(NotImplementedError):
            TELGENQuantumAdapterGNN()


if __name__ == "__main__":
    unittest.main()
