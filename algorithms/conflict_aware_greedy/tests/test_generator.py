import unittest

from algorithms.conflict_aware_greedy import generate_batch_schedule_portfolios
from qnet_core.contracts.complete_schedule import (
    CompleteSchedule,
    complete_schedule_count,
    enumerate_complete_schedules,
    is_valid_complete_schedule,
)


class CompleteScheduleGeneratorTests(unittest.TestCase):
    def test_enumerates_exact_legal_schedules_for_five_node_path(self):
        path = tuple("ABCDE")
        schedules = enumerate_complete_schedules(path)

        self.assertEqual(complete_schedule_count(3), 7)
        self.assertEqual(len(schedules), 7)
        self.assertEqual(
            {schedule.groups for schedule in schedules},
            {
                (("B", "D"), ("C",)),
                (("B",), ("C",), ("D",)),
                (("B",), ("D",), ("C",)),
                (("C",), ("B",), ("D",)),
                (("C",), ("D",), ("B",)),
                (("D",), ("B",), ("C",)),
                (("D",), ("C",), ("B",)),
            },
        )
        for schedule in schedules:
            self.assertEqual(set(schedule.swap_order), set("BCD"))
            self.assertEqual(
                schedule.dependency_tree.output_span,
                (0, 4),
            )
            self.assertTrue(is_valid_complete_schedule(path, schedule.groups))

    def test_rejects_invalid_group_schedules(self):
        path = tuple("ABCDE")
        cases = (
            ((("B", "C"), ("D",)), "parallel|share|consume"),
            ((("B", "D"),), "incomplete"),
            ((("B",), ("B",), ("C",), ("D",)), "repeat|duplicate"),
            ((("X",), ("B",), ("C",), ("D",)), "internal|path"),
            (((), ("B", "D"), ("C",)), "empty"),
            ((("C",), ("B", "D")), "parallel|share|consume"),
        )
        for groups, message in cases:
            with self.subTest(groups=groups):
                with self.assertRaisesRegex(ValueError, message):
                    CompleteSchedule(path, groups)

    def test_balanced_and_center_first_release_different_nodes(self):
        request_paths = {
            "R1": (tuple("ABCDE"),),
            "R2": (tuple("XCY"),),
            "R3": (tuple("UCV"),),
        }
        capacities = {
            node: 2
            for node in set("ABCDEXYUV")
        }
        portfolio = generate_batch_schedule_portfolios(
            request_paths,
            capacities,
            limit_per_path=4,
        )[("R1", 0)]
        by_groups = {candidate.groups: candidate for candidate in portfolio}

        balanced = by_groups[(("B", "D"), ("C",))]
        center_first = next(
            candidate for candidate in portfolio
            if candidate.groups[0] == ("C",)
        )
        self.assertEqual(
            [boundary.at("C") for boundary in balanced.estimate.memory_profile],
            [2, 2, 0],
        )
        self.assertEqual(
            [boundary.at("C") for boundary in center_first.estimate.memory_profile],
            [2, 0, 0, 0],
        )
        self.assertEqual(
            center_first.estimate.conflict_signature.coverage_request_ids,
            ("R2", "R3"),
        )

    def test_portfolio_is_capped_unique_and_deterministic(self):
        request_paths = {"R": (tuple("ABCDE"),)}
        capacities = {node: 2 for node in "ABCDE"}
        first = generate_batch_schedule_portfolios(
            request_paths, capacities
        )[("R", 0)]
        replay = generate_batch_schedule_portfolios(
            request_paths, capacities
        )[("R", 0)]

        self.assertEqual(len(first), 4)
        self.assertEqual(first, replay)
        self.assertEqual(
            len({candidate.schedule.structural_key for candidate in first}),
            4,
        )
        short = generate_batch_schedule_portfolios(
            {"short": (("A", "B", "C"),)},
            {"A": 2, "B": 2, "C": 2},
        )[("short", 0)]
        self.assertEqual(len(short), 1)
        self.assertEqual(short[0].groups, (("B",),))

    def test_hotspot_pressure_counts_distinct_requests_not_paths(self):
        request_paths = {
            "R1": (tuple("ABCDE"),),
            "RC": (tuple("XCY"), tuple("UCV")),
            "RC2": (tuple("PCQ"),),
            "RB": (tuple("MBN"),),
        }
        capacities = {
            node: 2
            for node in set("ABCDEXYUVPQMN")
        }
        portfolio = generate_batch_schedule_portfolios(
            request_paths, capacities
        )[("R1", 0)]

        self.assertTrue(any(
            candidate.groups[0] == ("C",)
            and candidate.estimate.target_node == "C"
            for candidate in portfolio
        ))
        scores = dict(portfolio[0].estimate.hotspot_scores)
        # RC contributes once at C despite having two alternative C paths.
        self.assertEqual(scores["C"], 3 / 2)
        self.assertEqual(scores["B"], 2 / 2)


if __name__ == "__main__":
    unittest.main()
