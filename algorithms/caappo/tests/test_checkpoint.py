import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from algorithms.caappo.checkpoint import (
    CheckpointCompatibilityError,
    load_caappo_checkpoint,
)
from algorithms.caappo.experiment import (
    CAAPPOVariant,
    ConstructionExperimentConfig,
    evaluate_checkpoint,
    train_variant_checkpoint,
)
from qnet_core.construction_api import (
    ConstructionDAG,
    ConstructionOperation,
    OperationKind,
    ResourceDemand,
)
from qnet_core.construction_catalog import RouteConstructionCandidate
from qnet_core.scenario import ScenarioConfig
from qnet_core.spec import PhysicalConfig


class CAAPPOCheckpointTests(unittest.TestCase):
    @staticmethod
    def _config(episodes: int) -> ConstructionExperimentConfig:
        variant = CAAPPOVariant(
            "caappo",
            candidate_count=1,
            construction_kinds=("left_deep",),
        )
        return ConstructionExperimentConfig(
            scenario=ScenarioConfig(
                request_count=1,
                min_hops=1,
                max_hops=1,
                ttl=10,
                horizon=10,
                topology_nodes=4,
                physical=PhysicalConfig(
                    generation_probability=1.0,
                    swap_probability=1.0,
                    memory_capacity=1,
                    node_memory_capacity=2,
                    quantum_distance_m=1.0,
                ),
            ),
            evaluation_seeds=(41,),
            training_seeds=(1,),
            validation_seeds=(31,),
            training_episodes=episodes,
            validation_interval=1,
            candidate_count=1,
            include_nominal_oracle=False,
            variants=(variant,),
        )

    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(2)
            config = replace(config, validation_interval=2)
            variant = config.variants[0]
            uninterrupted_path = root / "uninterrupted.pt"
            resumed_path = root / "resumed.pt"

            train_variant_checkpoint(
                config, variant, 1, uninterrupted_path
            )
            train_variant_checkpoint(
                replace(config, training_episodes=1),
                variant,
                1,
                resumed_path,
            )
            train_variant_checkpoint(
                config,
                variant,
                1,
                resumed_path,
                resume=True,
            )

            uninterrupted = load_caappo_checkpoint(uninterrupted_path)
            resumed = load_caappo_checkpoint(resumed_path)
            self.assertEqual(uninterrupted.completed_episodes, 2)
            self.assertEqual(resumed.completed_episodes, 2)
            for name, expected in uninterrupted.policy.state_dict().items():
                self.assertTrue(torch.equal(
                    expected,
                    resumed.policy.state_dict()[name],
                ), name)
            self.assertEqual(
                uninterrupted.policy.lambda_risk,
                resumed.policy.lambda_risk,
            )
            self.assertEqual(
                uninterrupted.best_validation,
                resumed.best_validation,
            )
            self.assertIsNotNone(uninterrupted.best_policy_state_dict)
            self.assertIsNotNone(resumed.best_policy_state_dict)
            for name, expected in uninterrupted.best_policy_state_dict.items():
                self.assertTrue(torch.equal(
                    expected,
                    resumed.best_policy_state_dict[name],
                ), name)
            self.assertEqual(
                [
                    row["episode_seed"]
                    for row in uninterrupted.history
                    if row["event"] == "training_episode"
                ],
                [
                    row["episode_seed"]
                    for row in resumed.history
                    if row["event"] == "training_episode"
                ],
            )

    def test_frozen_evaluation_and_strict_config_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "caappo.pt"
            config = self._config(1)
            train_variant_checkpoint(
                config, config.variants[0], 1, checkpoint
            )
            torch.manual_seed(9876)
            rng_before_load = torch.get_rng_state().clone()
            before = load_caappo_checkpoint(checkpoint)
            self.assertTrue(torch.equal(rng_before_load, torch.get_rng_state()))
            best_loaded = load_caappo_checkpoint(checkpoint, use_best=True)
            self.assertIsNotNone(best_loaded.best_optimizer_state_dict)
            self.assertEqual(
                best_loaded.policy.lambda_risk,
                best_loaded.best_lambda_risk,
            )
            rows, run = evaluate_checkpoint(checkpoint, (41,))
            after = load_caappo_checkpoint(checkpoint)

            self.assertEqual(len(rows), 1)
            self.assertEqual(run["selected_state"], "best")
            self.assertEqual(run["evaluation_seeds"], (41,))
            self.assertTrue(rows[0]["selected_candidates"])
            self.assertTrue(rows[0]["event_trace"])
            self.assertGreater(rows[0]["peak_memory_usage"], 0.0)
            self.assertIn("admission_mask_check_count", rows[0])
            self.assertIn("execution_mask_check_count", rows[0])
            self.assertIn("executor_rejection_count", rows[0])
            self.assertEqual(
                {
                    "request_id",
                    "candidate_id",
                    "route_nodes",
                    "construction_kind",
                },
                set(rows[0]["selected_candidates"][0]),
            )
            for name, expected in before.policy.state_dict().items():
                self.assertTrue(torch.equal(
                    expected,
                    after.policy.state_dict()[name],
                ), name)
            with self.assertRaises(CheckpointCompatibilityError):
                load_caappo_checkpoint(
                    checkpoint,
                    expected_training_metadata={"training_seed": 999},
                )

    def test_sequence_failure_telemetry_reaches_frozen_evaluation_rows(self):
        cases = (
            (
                "physical_failure_count",
                replace(
                    self._config(0).scenario,
                    physical=replace(
                        self._config(0).scenario.physical,
                        detector_efficiency=0.0,
                    ),
                ),
                "physical_failure",
            ),
            (
                "fidelity_violation_count",
                replace(
                    self._config(0).scenario,
                    physical=replace(
                        self._config(0).scenario.physical,
                        initial_fidelity=0.5,
                    ),
                ),
                "fidelity_reject",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (metric, scenario, cause) in enumerate(cases):
                with self.subTest(metric=metric):
                    config = replace(self._config(0), scenario=scenario)
                    checkpoint = root / f"telemetry-{index}.pt"
                    train_variant_checkpoint(
                        config, config.variants[0], 1, checkpoint
                    )
                    rows, _run = evaluate_checkpoint(checkpoint, (41,))
                    self.assertGreaterEqual(rows[0][metric], 1.0)
                    self.assertTrue(any(
                        event["failure_cause"] == cause
                        for event in rows[0]["event_trace"]
                    ))

            expiration_config = replace(
                self._config(0),
                scenario=replace(
                    self._config(0).scenario,
                    physical=replace(
                        self._config(0).scenario.physical,
                        memory_lifetime=1,
                    ),
                ),
            )
            expiration_checkpoint = root / "telemetry-expiration.pt"
            train_variant_checkpoint(
                expiration_config,
                expiration_config.variants[0],
                1,
                expiration_checkpoint,
            )

            def expiration_catalogue(spec, _count, _kinds):
                request = spec.requests[0]
                left, right = request.source, request.destination
                edge = f"{min(left, right)}-{max(left, right)}"
                generation = ConstructionOperation(
                    "r0:g0",
                    "r0",
                    OperationKind.GEN,
                    output_segment_id="r0:s0",
                    output_endpoints=(left, right),
                    resource_demand=ResourceDemand.from_mapping({
                        f"link:{edge}": 1,
                        f"genlane:{edge}": 1,
                        f"memory:{left}": 1,
                        f"memory:{right}": 1,
                    }),
                    output_resource_hold=ResourceDemand.from_mapping({
                        f"link:{edge}": 1,
                        f"memory:{left}": 1,
                        f"memory:{right}": 1,
                    }),
                )
                delay = ConstructionOperation(
                    "r0:delay",
                    "r0",
                    OperationKind.RELEASE,
                    predecessors=(generation.op_id,),
                    duration_ps=2_000_000,
                )
                return (RouteConstructionCandidate(
                    "r0:expiration",
                    "r0",
                    (left, right),
                    "left_deep",
                    ConstructionDAG("r0", (generation, delay)),
                    "r0:missing-terminal",
                ),)

            with patch(
                "algorithms.caappo.experiment._catalogue",
                side_effect=expiration_catalogue,
            ):
                rows, _run = evaluate_checkpoint(
                    expiration_checkpoint, (41,)
                )
            self.assertGreaterEqual(rows[0]["expiration_count"], 1.0)
            self.assertTrue(any(
                event["failure_cause"] == "expiration"
                for event in rows[0]["event_trace"]
            ))


if __name__ == "__main__":
    unittest.main()
