"""Solve and cross-check the deterministic intra-slot MILP oracle."""

from __future__ import annotations

from .intraslot_order_milp import solve_counterexample_milp
from .reproduce_intraslot_generation import run_case


def _format_order(order: tuple[str, ...]) -> str:
    return " -> ".join(order)


def main() -> None:
    result = solve_counterexample_milp()

    print("order | C occupancy at generation | C BSM use | MILP | simulator")
    for order, outcome in result.order_outcomes.items():
        profile = result.profiles[order]
        simulated = run_case(order, c_capacity=2)
        print(
            f"{_format_order(order):11} | "
            f"{profile.hotspot_occupancy!s:25} | "
            f"{profile.hotspot_bsm!s:9} | "
            f"{outcome.completed_requests}/3   | "
            f"{simulated.completed_count}/3"
        )
        if outcome.completed_requests != simulated.completed_count:
            raise RuntimeError(
                f"MILP/simulator disagreement for order {order}: "
                f"{outcome.completed_requests} != {simulated.completed_count}"
            )

    print()
    print(f"global optimum: {result.completed_requests}/3 requests")
    print(
        "all optimal orders: "
        + ", ".join(_format_order(order) for order in result.optimal_orders)
    )
    print(
        "one optimal automatic schedule: "
        + ", ".join(
            f"{request_id}@round{round_id}"
            for request_id, round_id
            in result.waiting_completion_round.items()
        )
    )


if __name__ == "__main__":
    main()
