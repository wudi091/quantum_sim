"""Shared packing helpers for legacy PlanningSnapshot baselines."""

from qnet_core.planner_api import PlanDescriptor


def pair_ids(plan: PlanDescriptor) -> set[str]:
    values = set(plan.elementary_pair_ids)
    for action in plan.swap_actions:
        if not action.left_pair_id.startswith("@"):
            values.add(action.left_pair_id)
        if not action.right_pair_id.startswith("@"):
            values.add(action.right_pair_id)
    return values


def pack(plans: list[PlanDescriptor]) -> tuple[str, ...]:
    selected: list[str] = []
    requests: set[str] = set()
    pairs: set[str] = set()
    claims: set[tuple[tuple[int, int], int]] = set()
    for plan in plans:
        inputs = pair_ids(plan)
        plan_claims = {(claim.endpoints, claim.lane) for claim in plan.claims}
        if plan.request_id in requests or inputs & pairs or plan_claims & claims:
            continue
        selected.append(plan.plan_id)
        requests.add(plan.request_id)
        pairs.update(inputs)
        claims.update(plan_claims)
    return tuple(selected)
