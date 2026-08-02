"""Build exact CON offline-scenario columns with the shared executor."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
from math import prod
from typing import Mapping

from qnet_core.order_core import OrderBatchProblem, simulate_order_batch
from .models import ScenarioConfiguration


@dataclass(frozen=True)
class ConfigurationBuildResult:
    configurations: tuple[ScenarioConfiguration, ...]
    enumerated_assignments: int
    feasible_assignments: int


def build_deterministic_scenario_configurations(
    problem: OrderBatchProblem,
    template_id_by_plan_id: Mapping[str, str],
    *,
    max_assignments: int | None = None,
) -> ConfigurationBuildResult:
    """Enumerate and validate every reject/one-plan-per-request assignment.

    This correctness-first bridge is intended for small offline scenarios.  It
    refuses stochastic planning probabilities so the offline library cannot
    observe hidden future RNG outcomes.  Large instances should replace full
    enumeration with column generation while keeping the same configuration
    contract.
    """

    if problem.required_requests or problem.preloaded_requests:
        raise ValueError(
            "offline admission scenarios cannot contain required/preloaded requests"
        )
    if problem.config.swap_probability != 1.0:
        raise ValueError(
            "offline configuration building requires deterministic swap success"
        )
    if any(
        link.generation_probability != 1.0
        for link in problem.link_by_edge.values()
    ):
        raise ValueError(
            "offline configuration building requires deterministic generation"
        )

    plan_ids = {plan.plan_id for plan in problem.candidates}
    if set(template_id_by_plan_id) != plan_ids:
        missing = plan_ids - set(template_id_by_plan_id)
        extra = set(template_id_by_plan_id) - plan_ids
        raise ValueError(
            "template mapping must cover exactly the scenario candidates; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if any(not value for value in template_id_by_plan_id.values()):
        raise ValueError("template IDs must be non-empty")

    by_request: dict[str, list[str]] = {}
    for plan in sorted(problem.candidates, key=lambda value: (
        value.priority,
        value.request_id,
        value.schedule_key,
        value.plan_id,
    )):
        by_request.setdefault(plan.request_id, []).append(plan.plan_id)
    request_ids = tuple(by_request)
    option_sets = tuple(
        (None, *by_request[request_id]) for request_id in request_ids
    )
    assignment_count = prod(len(options) for options in option_sets)
    if max_assignments is not None and assignment_count > max_assignments:
        raise ValueError(
            f"scenario needs {assignment_count} assignments, exceeding "
            f"max_assignments={max_assignments}"
        )

    unique: dict[
        tuple[frozenset[str], frozenset[str]],
        ScenarioConfiguration,
    ] = {}
    feasible_assignments = 0
    for choices in product(*option_sets):
        selected_plan_ids = tuple(
            plan_id for plan_id in choices if plan_id is not None
        )
        if not selected_plan_ids:
            configuration = ScenarioConfiguration("empty")
        else:
            result = simulate_order_batch(
                problem,
                selected_plan_ids,
                record_traces=False,
            )
            selected_requests = frozenset(
                request_id
                for request_id, plan_id in zip(request_ids, choices)
                if plan_id is not None
            )
            if frozenset(result.completed) != selected_requests:
                continue
            used_templates = frozenset(
                template_id_by_plan_id[plan_id]
                for plan_id in selected_plan_ids
            )
            payload = tuple(
                (request_id, template_id_by_plan_id[plan_id])
                for request_id, plan_id in zip(request_ids, choices)
                if plan_id is not None
            )
            digest = hashlib.sha256(
                repr(payload).encode("utf-8")
            ).hexdigest()[:16]
            configuration = ScenarioConfiguration(
                configuration_id=f"cfg:{digest}",
                used_template_ids=used_templates,
                completed_request_ids=selected_requests,
            )
        feasible_assignments += 1
        key = (
            configuration.used_template_ids,
            configuration.completed_request_ids,
        )
        incumbent = unique.get(key)
        if (
            incumbent is None
            or configuration.configuration_id < incumbent.configuration_id
        ):
            unique[key] = configuration

    configurations = tuple(sorted(
        unique.values(),
        key=lambda configuration: (
            configuration.completed_count,
            len(configuration.used_template_ids),
            configuration.configuration_id,
        ),
    ))
    return ConfigurationBuildResult(
        configurations=configurations,
        enumerated_assignments=assignment_count,
        feasible_assignments=feasible_assignments,
    )
