"""Planning-only Q-CAST adapter for the legacy shared environment."""

from qnet_core.planner_api import PlanningSnapshot

from algorithms._legacy_shared import pack


class QCASTPlanner:
    def reset(self, episode_seed: int) -> None:
        del episode_seed

    def select(self, snapshot: PlanningSnapshot) -> tuple[str, ...]:
        if snapshot.phase == "allocate":
            complete = [
                plan for plan in snapshot.candidates if plan.completes_request
            ]
            catalogue = complete or list(snapshot.candidates)
            plans = sorted(
                catalogue,
                key=lambda plan: (
                    -plan.expected_throughput,
                    plan.memory_cost,
                    plan.remaining_hops,
                    plan.plan_id,
                ),
            )
        else:
            plans = sorted(
                snapshot.candidates,
                key=lambda plan: (
                    not plan.completes_request,
                    -plan.expected_throughput,
                    -plan.width,
                    plan.remaining_hops,
                    plan.plan_id,
                ),
            )
        return pack(plans)
