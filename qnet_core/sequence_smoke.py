"""Optional SeQUeNCe smoke test runnable in the Python 3.12 environment."""

from __future__ import annotations

from .planner_api import SwapAction
from .sequence_backend import SequenceBackend
from .spec import EpisodeSpec, PhysicalConfig


def run_three_node_smoke() -> dict[str, object]:
    spec = EpisodeSpec(
        seed=7,
        nodes=(0, 1, 2),
        edges=((0, 1), (1, 2)),
        requests=(),
        horizon=10,
        physical=PhysicalConfig(generation_probability=1.0, swap_probability=1.0),
    )
    backend = SequenceBackend(spec)
    generated = backend.generate_elementary_pairs()
    if len(generated) != 2:
        raise AssertionError(f"expected two elementary pairs, got {generated}")
    success = backend.execute_swap(
        SwapAction("smoke", 1, generated[0], generated[1])
    )
    return {"generated": generated, "success": success, "pairs": tuple(backend.pairs)}


if __name__ == "__main__":
    print(run_three_node_smoke())
