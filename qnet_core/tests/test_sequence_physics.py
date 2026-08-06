import unittest

from qnet_core.sequence_backend import SequenceBackend
from qnet_core.spec import EpisodeSpec, PhysicalConfig


class SequencePhysicsTests(unittest.TestCase):
    def test_sequence_entities_and_timeline_expiration(self):
        from sequence.topology.node import BSMNode, QuantumRouter

        backend = SequenceBackend(EpisodeSpec(
            seed=17,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(),
            horizon=2,
            physical=PhysicalConfig(
                generation_probability=1.0,
                memory_lifetime=1,
                quantum_distance_m=1.0,
                slot_duration_ps=1_000_000,
            ),
        ))
        self.assertIsInstance(backend.nodes[0], QuantumRouter)
        self.assertIsInstance(backend._edge_bsm[(0, 1)], BSMNode)
        self.assertIn("bsm-0-1", backend.nodes[0].qchannels)

        generated = backend.generate_elementary_pairs()
        self.assertEqual(len(generated), 1)
        pair = backend.pairs[generated[0]]
        expire_time = min(
            pair.left_memory.get_expire_time(),
            pair.right_memory.get_expire_time(),
        )
        self.assertGreater(expire_time, backend.timeline.now())

        backend.advance_slot()
        self.assertNotIn(generated[0], backend.pairs)
        self.assertGreaterEqual(backend.timeline.now(), expire_time)

    def test_detector_efficiency_controls_generation(self):
        backend = SequenceBackend(EpisodeSpec(
            seed=17,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(),
            horizon=2,
            physical=PhysicalConfig(
                generation_probability=1.0,
                detector_efficiency=0.0,
                quantum_distance_m=1.0,
            ),
        ))
        self.assertEqual(backend.generate_elementary_pairs(), ())
        self.assertEqual(backend.estimate_route_throughput((0, 1), 1), 0.0)
        self.assertEqual(
            backend.link_capacities()[0]["generation_probability"], 0.0
        )

    def test_classical_delay_uses_topology_distance(self):
        backend = SequenceBackend(EpisodeSpec(
            seed=19,
            nodes=(0, 1, 2, 3),
            edges=((0, 1), (1, 2), (2, 3)),
            requests=(),
            horizon=2,
            physical=PhysicalConfig(quantum_distance_m=1000.0),
        ))
        self.assertEqual(backend.nodes[0].cchannels["3"].delay, 15_000_000)

    def test_resource_view_refreshes_bds_fidelity(self):
        backend = SequenceBackend(EpisodeSpec(
            seed=23,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(),
            horizon=2,
            physical=PhysicalConfig(
                generation_probability=1.0,
                memory_lifetime=2,
                quantum_distance_m=1.0,
                slot_duration_ps=1_000_000,
            ),
        ))
        pair_id = backend.generate_elementary_pairs()[0]
        initial = backend.resource(pair_id).fidelity
        backend.advance_slot()
        current = backend.resource(pair_id).fidelity
        self.assertLess(current, initial)


if __name__ == "__main__":
    unittest.main()
