"""Reproduce joint admission/path/swap-order MILP selection."""

from __future__ import annotations

from .batch_plan_milp import solve_joint_counterexample_milp


def main() -> None:
    solution = solve_joint_counterexample_milp()

    print(
        "joint optimum: "
        f"{solution.completed_requests}/3 requests, "
        f"reward={solution.total_reward}"
    )
    for request_id, scheduled in solution.selected.items():
        candidate = scheduled.candidate
        print(
            f"{request_id}: path={'-'.join(candidate.path)}, "
            f"swap={' -> '.join(candidate.swap_order)}, "
            f"start=round{scheduled.start + 1}, "
            f"finish=round{scheduled.finish}"
        )


if __name__ == "__main__":
    main()
