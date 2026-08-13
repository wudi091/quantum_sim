import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from algorithms.telgen.generate_online_milp_data import main
from algorithms.telgen.online_milp_dataset import load_online_milp_dataset


class OnlineMILPCollectionTests(unittest.TestCase):
    def test_resume_reuses_a_valid_completed_episode(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "teacher_data"
            arguments = [
                "--output", str(output),
                "--episodes", "1",
                "--seed-start", "14000",
                "--requests", "1",
                "--requests-per-batch", "1",
                "--decision-interval", "1",
                "--ttl", "4",
                "--nodes", "64",
                "--min-hops", "4",
                "--max-hops", "4",
                "--paths", "1",
                "--construction-plans", "1",
                "--time-limit-seconds", "10",
                "--generation-probability", "1",
                "--swap-probability", "1",
                "--memory-capacity", "2",
                "--quantum-distance-m", "1",
                "--slot-duration-ps", "1000000",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)
            episode_dataset = output / "episode_00014000" / "dataset"
            rollout_directories = tuple(episode_dataset.glob("rollout_*"))
            self.assertEqual(len(rollout_directories), 1)

            resumed_output = io.StringIO()
            with contextlib.redirect_stdout(resumed_output):
                self.assertEqual(main([*arguments, "--resume"]), 0)
            self.assertIn("resumed=1", resumed_output.getvalue())
            self.assertEqual(
                tuple(episode_dataset.glob("rollout_*")),
                rollout_directories,
            )
            manifest = json.loads(
                (output / "online_milp_dataset.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(manifest["collection_complete"])
            self.assertEqual(manifest["episode_count"], 1)
            self.assertTrue(manifest["episodes"][0]["resumed"])
            self.assertGreater(len(load_online_milp_dataset(output).samples), 0)


if __name__ == "__main__":
    unittest.main()
