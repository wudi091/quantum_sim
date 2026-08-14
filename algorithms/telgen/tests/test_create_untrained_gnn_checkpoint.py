from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from algorithms.telgen.create_untrained_gnn_checkpoint import (
    create_untrained_checkpoint,
)
from algorithms.telgen.milp_imitation import (
    AUTOREGRESSIVE_ARCHITECTURE,
    AUTOREGRESSIVE_CHECKPOINT_SCHEMA_VERSION,
    CONSTRAINT_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    VARIABLE_FEATURE_NAMES,
    CandidateConstraintGNN,
)


class UntrainedCheckpointTests(unittest.TestCase):
    def test_reconstructs_deterministic_current_schema_initialization(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "reference.pt"
            reference_model = CandidateConstraintGNN(hidden_dim=16, layers=1)
            torch.save({
                "schema_version": AUTOREGRESSIVE_CHECKPOINT_SCHEMA_VERSION,
                "model_class": "CandidateConstraintGNN",
                "architecture": AUTOREGRESSIVE_ARCHITECTURE,
                "model_config": {"hidden_dim": 16, "layers": 1},
                "state_dict": reference_model.state_dict(),
                "feature_schema": {
                    "variable": list(VARIABLE_FEATURE_NAMES),
                    "constraint": list(CONSTRAINT_FEATURE_NAMES),
                    "global": list(GLOBAL_FEATURE_NAMES),
                },
                "training_objective": {"target_mode": "set"},
            }, reference_path)
            first_path = root / "first.pt"
            second_path = root / "second.pt"
            first = create_untrained_checkpoint(
                reference_path,
                first_path,
                initialization_seed=123,
            )
            second = create_untrained_checkpoint(
                reference_path,
                second_path,
                initialization_seed=123,
            )
            self.assertEqual(first["gradient_update_count"], 0)
            self.assertEqual(
                first["reference_checkpoint_sha256"],
                second["reference_checkpoint_sha256"],
            )
            for key in first["state_dict"]:
                self.assertTrue(torch.equal(
                    first["state_dict"][key],
                    second["state_dict"][key],
                ))


if __name__ == "__main__":
    unittest.main()
