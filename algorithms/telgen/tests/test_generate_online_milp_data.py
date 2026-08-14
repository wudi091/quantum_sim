import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import algorithms.telgen.generate_online_milp_data as generator_module
from algorithms.telgen.generate_online_milp_data import main
from algorithms.telgen.milp_oracle import DiscreteOracleSolveError
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

            changed_budget = list(arguments)
            budget_index = changed_budget.index("--time-limit-seconds") + 1
            changed_budget[budget_index] = "20"
            resumed_with_new_budget = io.StringIO()
            with contextlib.redirect_stdout(resumed_with_new_budget):
                self.assertEqual(main([*changed_budget, "--resume"]), 0)
            self.assertIn("resumed=1", resumed_with_new_budget.getvalue())
            self.assertEqual(
                tuple(episode_dataset.glob("rollout_*")),
                rollout_directories,
            )

    def test_time_limit_failure_retries_with_a_larger_exact_budget(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "teacher_data"
            arguments = [
                "--output", str(output),
                "--episodes", "1",
                "--seed-start", "14001",
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
                "--time-limit-retries", "1",
                "--time-limit-multiplier", "3",
                "--generation-probability", "1",
                "--swap-probability", "1",
                "--memory-capacity", "2",
                "--quantum-distance-m", "1",
                "--slot-duration-ps", "1000000",
            ]
            original_generate = (
                generator_module.generate_online_milp_dataset
            )
            observed_limits = []

            def fail_once_then_solve(episode, destination, config):
                observed_limits.append(config.milp_time_limit_seconds)
                if len(observed_limits) == 1:
                    raise DiscreteOracleSolveError("Time limit reached")
                return original_generate(episode, destination, config)

            captured = io.StringIO()
            with patch.object(
                generator_module,
                "generate_online_milp_dataset",
                side_effect=fail_once_then_solve,
            ), contextlib.redirect_stdout(captured):
                self.assertEqual(main(arguments), 0)

            self.assertEqual(observed_limits, [10.0, 30.0])
            self.assertIn("next_limit=30.000s", captured.getvalue())
            manifest = json.loads(
                (output / "online_milp_dataset.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["episodes"][0]["solver_attempts"], 2)
            self.assertEqual(
                manifest["episodes"][0]["milp_time_limit_seconds"],
                30.0,
            )


if __name__ == "__main__":
    unittest.main()
