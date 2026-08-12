import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from algorithms.telgen import (
    DEFAULT_STATIC_LOAD_PROFILES,
    StaticLoadProfile,
    generate_static_load_calibration,
)
from qnet_core.scenario import ScenarioConfig


class StaticLoadCalibrationTests(unittest.TestCase):
    def test_default_profiles_increase_requests_and_reduce_time(self):
        self.assertEqual(
            [profile.name for profile in DEFAULT_STATIC_LOAD_PROFILES],
            ["light", "medium", "heavy"],
        )
        self.assertEqual(
            [profile.request_count for profile in DEFAULT_STATIC_LOAD_PROFILES],
            sorted(profile.request_count for profile in DEFAULT_STATIC_LOAD_PROFILES),
        )
        self.assertEqual(
            [profile.horizon for profile in DEFAULT_STATIC_LOAD_PROFILES],
            sorted(
                (profile.horizon for profile in DEFAULT_STATIC_LOAD_PROFILES),
                reverse=True,
            ),
        )

    def test_profiles_share_topology_and_use_nested_request_prefixes(self):
        scenario = ScenarioConfig(
            request_count=4,
            min_hops=2,
            max_hops=2,
            ttl=5,
            horizon=5,
            topology_nodes=8,
        )
        profiles = (
            StaticLoadProfile("light", 2, 5),
            StaticLoadProfile("medium", 3, 4),
            StaticLoadProfile("heavy", 4, 3),
        )
        with tempfile.TemporaryDirectory() as directory:
            result = generate_static_load_calibration(
                scenario,
                seeds=(7,),
                output_directory=directory,
                profiles=profiles,
                path_candidate_count=1,
            )
            self.assertTrue(result.manifest_path.exists())
            self.assertTrue(result.csv_path.exists())
            self.assertEqual(len(result.entries), 3)
            self.assertEqual(len(result.aggregates), 3)

            contexts = {}
            for entry in result.entries:
                target = Path(directory) / entry.file
                self.assertTrue(target.exists())
                with np.load(target) as payload:
                    metadata = json.loads(str(payload["metadata"]))
                contexts[entry.load_profile] = metadata["context"]
                stats = entry.statistics
                self.assertGreaterEqual(stats.completion_ratio, 0.0)
                self.assertLessEqual(stats.completion_ratio, 1.0)
                self.assertGreaterEqual(stats.peak_resource_utilization, 0.0)
                self.assertLessEqual(stats.peak_resource_utilization, 1.0)
                self.assertLess(stats.max_constraint_violation, 1e-7)

            self.assertEqual(
                contexts["light"]["episode"]["edges"],
                contexts["medium"]["episode"]["edges"],
            )
            self.assertEqual(
                contexts["medium"]["episode"]["edges"],
                contexts["heavy"]["episode"]["edges"],
            )
            endpoints = {
                name: [
                    (item["source"], item["destination"])
                    for item in context["episode"]["requests"]
                ]
                for name, context in contexts.items()
            }
            self.assertEqual(endpoints["light"], endpoints["medium"][:2])
            self.assertEqual(endpoints["medium"], endpoints["heavy"][:3])
            self.assertEqual(
                [
                    contexts[name]["episode"]["horizon"]
                    for name in ("light", "medium", "heavy")
                ],
                [5, 4, 3],
            )

            manifest = json.loads(
                result.manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["records"]), 3)
            self.assertEqual(len(manifest["aggregates"]), 3)
            with result.csv_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertIn("fractional_request_ratio", rows[0])

    def test_profile_name_must_be_safe_for_output_path(self):
        with self.assertRaisesRegex(ValueError, "path-safe"):
            StaticLoadProfile("../heavy", 4, 3)


if __name__ == "__main__":
    unittest.main()
