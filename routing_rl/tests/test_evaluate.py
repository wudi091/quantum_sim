import unittest

from qnet_core.planners import GreedyPlanner
from routing_rl.evaluate import PlannerController


class PlannerControllerTests(unittest.TestCase):
    def test_controller_starts_without_pending_actions(self):
        controller = PlannerController(GreedyPlanner())
        controller.reset(3)
        self.assertEqual(controller.pending, [])


if __name__ == "__main__":
    unittest.main()
