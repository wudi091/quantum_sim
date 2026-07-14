from __future__ import annotations

import unittest

import numpy as np

from batchswap_rl.reliq_adapter import ReliqEnvironmentAdapter, canonicalize_observation


class _StubReliq:
    def __init__(self):
        self.stages = []

    def set_curriculum(self, stage: int):
        if not isinstance(stage, int):
            raise TypeError("stub expects an integer stage")
        self.stages.append(stage)

    def reset(self, **kwargs):
        return {
            "plans": np.zeros((2, 3), dtype=np.float32),
            "state": np.ones(4, dtype=np.float32),
            "mask": np.asarray([True, False]),
        }, {"seed": kwargs.get("seed")}

    def step(self, action):
        self.last_action = action
        return self.reset(seed=action)[0], 1.25, False, False, {"duration": 2}


class _Stage:
    name = "long"


class ReliqAdapterTest(unittest.TestCase):
    def test_aliases_and_stop_are_canonicalized(self):
        result = canonicalize_observation(
            {
                "plans": np.zeros((2, 3)),
                "state": np.ones(4),
                "mask": [True, False],
                "requests": np.zeros((1, 2)),
                "active_request_mask": [True],
            }
        )
        self.assertEqual(result["candidate_features"].shape, (2, 3))
        self.assertEqual(result["global_features"].shape, (4,))
        self.assertEqual(result["action_mask"].tolist(), [True, False, True])
        self.assertIn("request_features", result)

    def test_stage_object_falls_back_to_integer_and_step_preserves_duration(self):
        raw = _StubReliq()
        env = ReliqEnvironmentAdapter(raw)
        env.set_curriculum(_Stage())
        self.assertEqual(raw.stages, [2])
        observation, info = env.reset(seed=7)
        self.assertEqual(info["seed"], 7)
        self.assertIn("candidate_features", observation)
        _, reward, terminated, truncated, step_info = env.step(4)
        self.assertEqual(reward, 1.25)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(step_info["duration"], 2)

    def test_adapter_can_recreate_backend_when_only_factory_supports_curriculum(self):
        created = []

        def factory(**kwargs):
            created.append(kwargs)
            return _StubReliq()

        raw = object()
        env = ReliqEnvironmentAdapter(raw, factory=factory, seed=11)
        # Model a backend exposing only make_env(stage, seed).
        env.set_curriculum(_Stage())
        self.assertEqual(created, [{"stage": 2, "seed": 11}])


if __name__ == "__main__":
    unittest.main()
