"""Reproduce the automatic-generation swap-order memory counterexample."""

from __future__ import annotations

from .intraslot_simulator import (
    IntraSlotConfig,
    IntraSlotPlan,
    IntraSlotSimulator,
    SlotResult,
    focus_trace,
)


def _plans(r1_order: tuple[str, ...]) -> tuple[IntraSlotPlan, ...]:
    return (
        IntraSlotPlan("R1", ("A", "B", "C", "D", "E"), r1_order, priority=0),
        IntraSlotPlan("R2", ("X", "C", "Y"), ("C",), priority=1),
        IntraSlotPlan("R3", ("U", "C", "V"), ("C",), priority=2),
    )


def run_case(r1_order: tuple[str, ...], c_capacity: int = 2) -> SlotResult:
    nodes = ("A", "B", "C", "D", "E", "X", "Y", "U", "V")
    capacity = {node: 2 for node in nodes}
    capacity["C"] = c_capacity
    simulator = IntraSlotSimulator(
        plans=_plans(r1_order),
        node_capacity=capacity,
        config=IntraSlotConfig(
            rounds_per_slot=3,
            generation_probability=1.0,
            swap_probability=1.0,
            edge_capacity=1,
            bsm_capacity_per_node=1,
            seed=7,
        ),
        initially_ready_requests=("R1",),
    )
    return simulator.run()


def _print_case(name: str, result: SlotResult) -> None:
    print(f"{name}: completed={len(result.completed)}/3 {result.completed}")
    print("  C trace: round | start -> after generation -> after swaps")
    for round_id, start, generated, swapped in focus_trace(result, "C"):
        print(f"           {round_id} | {start} -> {generated} -> {swapped}")
    for trace in result.traces:
        generated = [
            f"{event.request_id}:{event.edge}:{event.status}"
            for event in trace.generation_events
            if "C" in event.edge
        ]
        swaps = [
            f"{event.request_id}@{event.middle}:{event.status}"
            for event in trace.swap_events
        ]
        print(f"  round {trace.round_id}: generation={generated} swaps={swaps}")


def main() -> None:
    late = run_case(("B", "C", "D"), c_capacity=2)
    early = run_case(("C", "B", "D"), c_capacity=2)
    late_roomy = run_case(("B", "C", "D"), c_capacity=4)
    early_roomy = run_case(("C", "B", "D"), c_capacity=4)

    _print_case("late release, M_C=2", late)
    _print_case("early release, M_C=2", early)
    _print_case("late release, M_C=4", late_roomy)
    _print_case("early release, M_C=4", early_roomy)

    assert len(late.completed) == 2
    assert len(early.completed) == 3
    assert len(late_roomy.completed) == 3
    assert len(early_roomy.completed) == 3


if __name__ == "__main__":
    main()
