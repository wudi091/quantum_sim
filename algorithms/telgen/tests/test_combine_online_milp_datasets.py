import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from algorithms.telgen.combine_online_milp_datasets import combine_collections


class CombinedOnlineMILPDatasetTests(unittest.TestCase):
    def test_combines_disjoint_split_collections(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = []
            loaded = []
            for index, seed in enumerate((101, 201, 301)):
                manifest = root / f"group_{index}" / "online_milp_dataset.json"
                manifest.parent.mkdir()
                manifest.write_text(
                    json.dumps({
                        "dataset_kind": "online_milp_teacher_collection",
                        "collection_complete": True,
                        "scenario": {"topology_nodes": 64 + index},
                    }),
                    encoding="utf-8",
                )
                manifests.append(manifest)
                loaded.append(SimpleNamespace(
                    episode_seeds=(seed,),
                    sample_paths=(root / f"sample_{index}.npz",),
                    samples=(object(),),
                ))
            with patch(
                "algorithms.telgen.combine_online_milp_datasets."
                "resolve_online_milp_dataset_manifest",
                side_effect=manifests,
            ), patch(
                "algorithms.telgen.combine_online_milp_datasets."
                "load_online_milp_dataset",
                side_effect=loaded,
            ):
                payload = combine_collections(
                    root / "suite",
                    "unit_test",
                    [
                        ("train_group", "train", manifests[0]),
                        ("validation_group", "validation", manifests[1]),
                        ("test_group", "test", manifests[2]),
                    ],
                )
        self.assertEqual(payload["episode_count"], 3)
        self.assertEqual(payload["sample_count"], 3)
        self.assertEqual(payload["split_episode_seeds"]["train"], [101])
        self.assertEqual(
            payload["split_episode_seeds"]["validation"], [201]
        )
        self.assertEqual(payload["split_episode_seeds"]["test"], [301])

    def test_rejects_duplicate_episode_seeds(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = []
            for index in range(2):
                manifest = root / f"group_{index}.json"
                manifest.write_text(json.dumps({
                    "dataset_kind": "online_milp_teacher_collection",
                    "collection_complete": True,
                }), encoding="utf-8")
                manifests.append(manifest)
            repeated = SimpleNamespace(
                episode_seeds=(101,),
                sample_paths=(root / "sample.npz",),
                samples=(object(),),
            )
            with patch(
                "algorithms.telgen.combine_online_milp_datasets."
                "resolve_online_milp_dataset_manifest",
                side_effect=manifests,
            ), patch(
                "algorithms.telgen.combine_online_milp_datasets."
                "load_online_milp_dataset",
                side_effect=(repeated, repeated),
            ):
                with self.assertRaisesRegex(ValueError, "duplicate episode seed"):
                    combine_collections(
                        root / "suite",
                        "unit_test",
                        [
                            ("train", "train", manifests[0]),
                            ("test", "test", manifests[1]),
                        ],
                    )


if __name__ == "__main__":
    unittest.main()
