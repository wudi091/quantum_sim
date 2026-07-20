import argparse
import unittest

from qnet_core.planners import GreedyPlanner
from routing_rl.evaluate import PlannerController, parse_controller_names


class PlannerControllerTests(unittest.TestCase):
    def test_controller_starts_without_pending_actions(self):
        controller = PlannerController(GreedyPlanner())
        controller.reset(3)
        self.assertEqual(controller.pending, [])


class ControllerSelectionTests(unittest.TestCase):
    def test_parses_ordered_controller_subset(self):
        self.assertEqual(
            parse_controller_names("ppo, qddca,ppo"),
            ("ppo", "qddca"),
        )

    def test_rejects_unknown_controller(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_controller_names("ppo,unknown")


if __name__ == "__main__":
    unittest.main()
