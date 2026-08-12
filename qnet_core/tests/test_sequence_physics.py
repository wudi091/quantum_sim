import unittest

from qnet_core.command_api import ResourceClaim, SwapAction
from qnet_core.sequence_backend import SequenceBackend
from qnet_core.spec import EpisodeSpec, PhysicalConfig


class SequencePhysicsTests(unittest.TestCase):
    def test_memory_exposure_tracks_staggered_physical_transitions_exactly(self):
        backend = SequenceBackend(EpisodeSpec(
            seed=15,
            nodes=(0, 1, 2, 3),
            edges=((0, 1), (2, 3)),
            requests=(),
            horizon=40,
            physical=PhysicalConfig(
                generation_probability=1.0,
                detector_efficiency=0.0,
                memory_capacity=1,
                node_memory_capacity=1,
                quantum_distance_m=1000.0,
            ),
        ))
        first = backend.begin_generation(
            (ResourceClaim(0, 1, 0),), "first"
        )
        backend.advance_physical_to(1_000_000, synchronize=False)
        second = backend.begin_generation(
            (ResourceClaim(2, 3, 0),), "second"
        )

        backend.run_prepared_protocols(first + second)
        backend.finish_generation(first + second)
        state = dict(backend.construction_state())

        self.assertEqual(state["physical_memory_usage"], 0)
        self.assertEqual(state["peak_physical_memory_usage"], 4)
        self.assertEqual(
            state["physical_memory_time_unit_ps"], 60_000_040
        )

    def test_generation_preparation_rejection_has_explicit_cause(self):
        backend = SequenceBackend(EpisodeSpec(
            seed=16,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(),
            horizon=2,
            physical=PhysicalConfig(
                generation_probability=1.0,
                memory_capacity=2,
                node_memory_capacity=1,
                quantum_distance_m=1.0,
            ),
        ))

        prepared = backend.begin_generation(
            (ResourceClaim(0, 1, 0), ResourceClaim(1, 2, 0)),
            "capacity-test",
        )

        self.assertEqual(len(prepared), 2)
        self.assertEqual(
            sum(item.failure_cause == "physical_backend_rejection"
                for item in prepared),
            1,
        )
        self.assertEqual(
            sum(item.context is not None for item in prepared),
            1,
        )
        backend.cancel_generation(prepared)

    def test_swapped_adjacent_pair_does_not_consume_elementary_link_capacity(self):
        backend = SequenceBackend(EpisodeSpec(
            seed=18,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2), (0, 2)),
            requests=(),
            horizon=4,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                memory_capacity=1,
                node_memory_capacity=3,
                quantum_distance_m=1.0,
            ),
        ))
        generated = backend.generate_claimed_pairs(
            (ResourceClaim(0, 1, 0), ResourceClaim(1, 2, 0)),
            "path",
        )
        left = generated[ResourceClaim(0, 1, 0)]
        right = generated[ResourceClaim(1, 2, 0)]
        assert left is not None and right is not None
        backend.release_allocation("path")

        prepared_swap = backend.begin_swap(
            SwapAction("r0", 1, left, right),
            "swap",
        )
        assert prepared_swap is not None
        backend.run_prepared_protocols(swaps=(prepared_swap,))
        swapped = backend.finish_swap(prepared_swap)
        self.assertIsNotNone(swapped)
        self.assertEqual(backend.resource(swapped).endpoints, (0, 2))
        self.assertEqual(backend.edge_occupancy(0, 2), 0)

        direct = backend.begin_generation(
            (ResourceClaim(0, 2, 0),),
            "direct",
        )
        self.assertEqual(direct[0].failure_cause, "")
        self.assertIsNotNone(direct[0].context)
        backend.cancel_generation(direct)

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

    def test_seeded_physics_isolated_from_other_backend_construction(self):
        spec = EpisodeSpec(
            seed=(1 << 63) + 173,
            nodes=tuple(range(9)),
            edges=tuple((node, node + 1) for node in range(8)),
            requests=(),
            horizon=2,
            physical=PhysicalConfig(
                generation_probability=0.5,
                memory_capacity=2,
                node_memory_capacity=4,
                quantum_distance_m=1.0,
            ),
        )
        first = SequenceBackend(spec)
        # Constructing a second physical world must not perturb the first
        # world's seeded protocol outcomes.
        SequenceBackend(EpisodeSpec(
            seed=(1 << 63) + 174,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(),
            horizon=2,
            physical=PhysicalConfig(generation_probability=0.5),
        ))
        first_generated = first.generate_elementary_pairs()
        second_generated = SequenceBackend(spec).generate_elementary_pairs()
        self.assertEqual(first_generated, second_generated)


if __name__ == "__main__":
    unittest.main()
