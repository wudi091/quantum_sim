from __future__ import annotations

import unittest

from construction.intraslot_simulator import (
    IntraSlotConfig,
    IntraSlotPlan,
    IntraSlotSimulator,
    focus_trace,
)


def _counterexample_plans(order: tuple[str, ...]) -> tuple[IntraSlotPlan, ...]:
    return (
        IntraSlotPlan("R1", ("A", "B", "C", "D", "E"), order, priority=0),
        IntraSlotPlan("R2", ("X", "C", "Y"), ("C",), priority=1),
        IntraSlotPlan("R3", ("U", "C", "V"), ("C",), priority=2),
    )


def _capacity(c_capacity: int) -> dict[str, int]:
    nodes = ("A", "B", "C", "D", "E", "X", "Y", "U", "V")
    values = {node: 2 for node in nodes}
    values["C"] = c_capacity
    return values


def _run(
    order: tuple[str, ...],
    c_capacity: int = 2,
    generation_probability: float = 1.0,
    seed: int = 11,
):
    return IntraSlotSimulator(
        plans=_counterexample_plans(order),
        node_capacity=_capacity(c_capacity),
        config=IntraSlotConfig(
            rounds_per_slot=3,
            generation_probability=generation_probability,
            swap_probability=1.0,
            edge_capacity=1,
            bsm_capacity_per_node=1,
            seed=seed,
        ),
        initially_ready_requests=("R1",),
    ).run()


class IntraSlotSimulatorTests(unittest.TestCase):
    def test_late_release_completes_only_two_requests(self) -> None:
        result = _run(("B", "C", "D"))

        self.assertEqual(result.completed, ("R1", "R2"))
        self.assertEqual(result.missed, ("R3",))
        self.assertEqual(
            focus_trace(result, "C"),
            (
                (1, 2, 2, 2),
                (2, 2, 2, 0),
                (3, 0, 2, 0),
            ),
        )

    def test_early_release_completes_all_three_requests(self) -> None:
        result = _run(("C", "B", "D"))

        self.assertEqual(result.completed, ("R1", "R2", "R3"))
        self.assertEqual(result.missed, ())
        self.assertEqual(
            focus_trace(result, "C"),
            (
                (1, 2, 2, 0),
                (2, 0, 2, 0),
                (3, 0, 2, 0),
            ),
        )

    def test_extra_memory_removes_swap_order_completion_gap(self) -> None:
        late = _run(("B", "C", "D"), c_capacity=4)
        early = _run(("C", "B", "D"), c_capacity=4)

        self.assertEqual(late.completed_count, 3)
        self.assertEqual(early.completed_count, 3)

    def test_generation_never_exceeds_node_capacity(self) -> None:
        plan = IntraSlotPlan("R", ("X", "C", "Y"), ("C",))
        result = IntraSlotSimulator(
            plans=(plan,),
            node_capacity={"X": 1, "C": 1, "Y": 1},
            config=IntraSlotConfig(
                rounds_per_slot=1,
                generation_probability=1.0,
                swap_probability=1.0,
            ),
        ).run()

        trace = result.traces[0]
        self.assertEqual(trace.occupancy_after_generation["C"], 1)
        self.assertLessEqual(trace.occupancy_after_generation["C"], 1)
        self.assertIn("R", result.missed)
        statuses = [event.status for event in trace.generation_events]
        self.assertEqual(statuses, ["success", "blocked_memory"])

    def test_stochastic_generation_is_reproducible(self) -> None:
        first = _run(
            ("C", "B", "D"), generation_probability=0.55, seed=97
        )
        second = _run(
            ("C", "B", "D"), generation_probability=0.55, seed=97
        )

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
