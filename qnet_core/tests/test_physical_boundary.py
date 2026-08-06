import unittest

from qnet_core.env import SharedRoutingEnv
from qnet_core.physical_api import PhysicalCapabilities
from qnet_core.spec import EpisodeSpec, PhysicalConfig


class StubPhysicalBackend:
    """Minimal backend proving that the planning layer has no simulator access."""

    time = 0
    capabilities = PhysicalCapabilities(
        max_width=2,
        memory_capacity=2,
        node_memory_capacity=None,
    )

    def synchronize(self):
        pass

    def resources(self):
        return ()

    def estimate_route_throughput(self, route_nodes, width):
        return float(width)

    def link_capacities(self):
        return ({
            "left": 0,
            "right": 1,
            "max_width": 2,
            "generation_probability": 1.0,
        },)


class PhysicalBoundaryTests(unittest.TestCase):
    def test_planning_layer_accepts_an_opaque_physical_backend(self):
        spec = EpisodeSpec(
            seed=1,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(),
            horizon=2,
            physical=PhysicalConfig(max_width=1),
        )
        backend = StubPhysicalBackend()
        env = SharedRoutingEnv(spec.planning, backend=backend)

        self.assertFalse(hasattr(backend, "pairs"))
        self.assertFalse(hasattr(env.spec, "physical"))
        self.assertEqual(env.phase, "allocate")
        self.assertEqual(env.snapshot().link_capacities[0]["max_width"], 2)


if __name__ == "__main__":
    unittest.main()
