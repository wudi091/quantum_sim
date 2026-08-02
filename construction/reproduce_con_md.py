"""Reproduce con_design.md §3-4 with the deterministic Phase 1 kernel.

R1 uses path A-B-C-D-E. R2 uses X-C-Y in the same centralized slot.
Every node has memory capacity 2.

Expected result:
* R1=C_seq reserves 1 memory at C; greedy R2=C_seq reserves 1 -> accepted.
* R1=C_bal reserves 2 memories at C; any R2 needs >=1 -> rejected.

Run:
    python -m construction.reproduce_con_md
"""

from __future__ import annotations

from .enumerator import balanced_plan, sequential_plan
from .planners import greedy_select
from .simulator import SlotSimulator, plan_footprint, simulate_plan

PATH_R1 = (0, 1, 2, 3, 4)  # A-B-C-D-E
PATH_R2 = (5, 2, 6)        # X-C-Y
NODE_C = 2
CAPACITY = {node: 2 for node in range(7)}


def _show(label: str, plan) -> None:
    execution = simulate_plan(plan)
    fp = plan_footprint(execution)
    print(f"{label}: kind={plan.kind}, makespan={execution.makespan}")
    print(f"  C generation curve: {fp['series'][NODE_C]}")
    print(
        f"  C peak={fp['peak'][NODE_C]}, "
        f"duration_at_peak={fp['duration_at_peak'][NODE_C]}"
    )


def _run(r1_plan) -> tuple[bool, str | None]:
    slot = SlotSimulator(CAPACITY)
    assert slot.admit(r1_plan, simulate_plan(r1_plan))
    r2_plan = greedy_select(slot, [PATH_R2])
    return r2_plan is not None, None if r2_plan is None else r2_plan.kind


def main() -> None:
    seq = sequential_plan(PATH_R1)
    bal = balanced_plan(PATH_R1)

    print("con_design.md §3: same path, different construction footprint")
    _show("R1 C_seq", seq)
    _show("R1 C_bal", bal)

    seq_ok, seq_r2_kind = _run(seq)
    bal_ok, bal_r2_kind = _run(bal)

    print("\ncon_design.md §4: same-slot concurrent R2")
    print(f"  R1=C_seq -> R2 {'ACCEPTED (' + seq_r2_kind + ')' if seq_ok else 'REJECTED'}")
    print(f"  R1=C_bal -> R2 {'ACCEPTED (' + bal_r2_kind + ')' if bal_ok else 'REJECTED'}")

    assert seq_ok, "C_seq must leave one C memory for R2"
    assert not bal_ok, "C_bal must occupy both C memories and reject R2"
    print("\nPASS: same P, different C -> different concurrent-request outcome")


if __name__ == "__main__":
    main()
