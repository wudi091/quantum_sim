"""Exact scenario MILPs for CON's offline path-schedule library."""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import lil_matrix

from construction.exact_milp import solve_exact_milp

from .models import (
    FixedPathLibraryProblem,
    LibraryPathCandidate,
    LibraryScheduleTemplate,
    OfflineLibraryMilpResult,
    OfflineScheduleLibrary,
    TopologyLibraryProblem,
    common_fingerprints,
    library_digest,
    library_layout_digest,
    scenario_configurations,
)


def solve_topology_schedule_library(
    problem: TopologyLibraryProblem,
) -> OfflineLibraryMilpResult:
    """Jointly select paths and complete schedules across all scenarios.

    ``u[path]`` chooses a path, ``z[template]`` chooses one of that path's
    complete schedules, and each scenario chooses one already-validated joint
    configuration column.  The shared ``u``/``z`` variables are the immutable
    offline library; scenario columns are recourse decisions used only while
    fitting it.
    """

    paths = tuple(sorted(problem.paths, key=lambda path: (
        path.pair_id,
        path.pool_rank,
        tuple(map(repr, path.path)),
        path.path_id,
    )))
    templates = tuple(sorted(problem.templates, key=lambda template: (
        template.pair_id,
        template.path_id,
        template.structural_key,
        template.template_id,
    )))
    scenarios = tuple(sorted(
        problem.scenarios, key=lambda scenario: scenario.scenario_id
    ))
    configurations = {
        scenario.scenario_id: tuple(sorted(
            scenario_configurations(scenario),
            key=lambda configuration: configuration.configuration_id,
        ))
        for scenario in scenarios
    }

    path_by_id = {path.path_id: path for path in paths}
    templates_by_path: dict[str, list[LibraryScheduleTemplate]] = {}
    template_path_id: dict[str, str] = {}
    for template in templates:
        templates_by_path.setdefault(template.path_id, []).append(template)
        template_path_id[template.template_id] = template.path_id
    paths_by_pair: dict[str, list[LibraryPathCandidate]] = {}
    for path in paths:
        paths_by_pair.setdefault(path.pair_id, []).append(path)

    variable_keys: list[tuple[str, ...]] = [
        ("u", path.path_id) for path in paths
    ] + [
        ("z", template.template_id) for template in templates
    ]
    for scenario in scenarios:
        variable_keys.extend(
            ("x", scenario.scenario_id, configuration.configuration_id)
            for configuration in configurations[scenario.scenario_id]
        )
    variable_index = {key: index for index, key in enumerate(variable_keys)}
    variable_count = len(variable_keys)

    rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []

    # At most four real paths are retained for each unordered endpoint pair.
    # The equality is safe because adding a candidate cannot reduce recourse
    # feasibility; short pools naturally use min(4, |P_pair|).
    for pair_id in sorted(paths_by_pair):
        rows.append({
            variable_index[("u", path.path_id)]: 1.0
            for path in paths_by_pair[pair_id]
        })
        budget = problem.effective_path_budget_by_pair[pair_id]
        lower.append(float(budget))
        upper.append(float(budget))

    # A selected path receives its effective number of unique schedules; an
    # unselected path receives none.  z <= u also strengthens the LP relaxation.
    for path in paths:
        schedule_budget = problem.effective_schedule_budget_by_path[path.path_id]
        row = {
            variable_index[("z", template.template_id)]: 1.0
            for template in templates_by_path[path.path_id]
        }
        row[variable_index[("u", path.path_id)]] = -float(schedule_budget)
        rows.append(row)
        lower.append(0.0)
        upper.append(0.0)
        for template in templates_by_path[path.path_id]:
            rows.append({
                variable_index[("z", template.template_id)]: 1.0,
                variable_index[("u", path.path_id)]: -1.0,
            })
            lower.append(-np.inf)
            upper.append(0.0)

    for scenario in scenarios:
        scenario_id = scenario.scenario_id
        rows.append({
            variable_index[(
                "x", scenario_id, configuration.configuration_id
            )]: 1.0
            for configuration in configurations[scenario_id]
        })
        lower.append(1.0)
        upper.append(1.0)

        # Exactly one configuration is selected, so these aggregated links are
        # equivalent to x[s,k] <= z[c] for each template used by column k.
        for template in templates:
            using = tuple(
                configuration
                for configuration in configurations[scenario_id]
                if template.template_id in configuration.used_template_ids
            )
            if not using:
                continue
            row = {
                variable_index[(
                    "x", scenario_id, configuration.configuration_id
                )]: 1.0
                for configuration in using
            }
            row[variable_index[("z", template.template_id)]] = -1.0
            rows.append(row)
            lower.append(-np.inf)
            upper.append(0.0)

        # Redundant for integer solutions but useful for the relaxation when
        # fractional columns use different schedules of one path.
        for path in paths:
            using = tuple(
                configuration
                for configuration in configurations[scenario_id]
                if any(
                    template_path_id[template_id] == path.path_id
                    for template_id in configuration.used_template_ids
                )
            )
            if not using:
                continue
            row = {
                variable_index[(
                    "x", scenario_id, configuration.configuration_id
                )]: 1.0
                for configuration in using
            }
            row[variable_index[("u", path.path_id)]] = -1.0
            rows.append(row)
            lower.append(-np.inf)
            upper.append(0.0)

    matrix = lil_matrix((len(rows), variable_count), dtype=float)
    for row_index, coefficients in enumerate(rows):
        for column, coefficient in coefficients.items():
            matrix[row_index, column] = coefficient
    base_constraints = LinearConstraint(
        matrix.tocsr(),
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )
    bounds = Bounds(
        np.zeros(variable_count, dtype=float),
        np.ones(variable_count, dtype=float),
    )
    integrality = np.ones(variable_count, dtype=int)

    # Stage 1: maximize the weighted number of completed requests.
    primary = np.zeros(variable_count, dtype=float)
    for scenario in scenarios:
        for configuration in configurations[scenario.scenario_id]:
            primary[variable_index[(
                "x", scenario.scenario_id, configuration.configuration_id
            )]] = -scenario.weight * configuration.completed_count
    primary_result = solve_exact_milp(
        c=primary,
        integrality=integrality,
        bounds=bounds,
        constraints=base_constraints,
        options={"disp": False},
    )
    optimal_weighted_completed = int(round(-float(primary_result.fun)))

    # Stage 2 fixes the paper objective and provides a stable structural
    # preference for shorter/earlier pool paths and earlier complete schedules.
    primary_row = lil_matrix((1, variable_count), dtype=float)
    for index, coefficient in enumerate(-primary):
        if coefficient:
            primary_row[0, index] = coefficient
    fixed_primary = LinearConstraint(
        primary_row.tocsr(),
        np.asarray([optimal_weighted_completed], dtype=float),
        np.asarray([optimal_weighted_completed], dtype=float),
    )
    secondary = np.zeros(variable_count, dtype=float)
    template_scale = len(templates) + 1
    for rank, path in enumerate(paths, start=1):
        secondary[variable_index[("u", path.path_id)]] = float(
            rank * template_scale
        )
    for rank, template in enumerate(templates, start=1):
        secondary[variable_index[("z", template.template_id)]] = float(rank)
    secondary_result = solve_exact_milp(
        c=secondary,
        integrality=integrality,
        bounds=bounds,
        constraints=(base_constraints, fixed_primary),
        options={"disp": False},
    )

    selected_paths = tuple(
        path for path in paths
        if secondary_result.x[variable_index[("u", path.path_id)]] > 0.5
    )
    selected_templates = tuple(
        template for template in templates
        if secondary_result.x[
            variable_index[("z", template.template_id)]
        ] > 0.5
    )
    selected_path_ids = tuple(path.path_id for path in selected_paths)
    selected_ids = tuple(
        template.template_id for template in selected_templates
    )
    selected_by_pair = {
        pair_id: tuple(
            path.path_id for path in selected_paths if path.pair_id == pair_id
        )
        for pair_id in sorted(paths_by_pair)
    }
    selected_by_path = {
        path.path_id: tuple(
            template.template_id
            for template in selected_templates
            if template.path_id == path.path_id
        )
        for path in paths
    }

    choice: dict[str, str] = {}
    completed: dict[str, int] = {}
    for scenario in scenarios:
        selected_configuration = next(
            configuration
            for configuration in configurations[scenario.scenario_id]
            if secondary_result.x[variable_index[(
                "x", scenario.scenario_id, configuration.configuration_id
            )]] > 0.5
        )
        choice[scenario.scenario_id] = selected_configuration.configuration_id
        completed[scenario.scenario_id] = selected_configuration.completed_count

    topology, distribution, physics = common_fingerprints(scenarios)
    library = OfflineScheduleLibrary(
        selected_template_ids=selected_ids,
        selected_by_path=selected_by_path,
        effective_budget_by_path=(
            problem.effective_schedule_budget_by_path
        ),
        topology_fingerprint=topology,
        request_distribution_fingerprint=distribution,
        physics_fingerprint=physics,
        training_scenario_ids=tuple(
            scenario.scenario_id for scenario in scenarios
        ),
        training_trace_digests=tuple(
            scenario.trace_digest for scenario in scenarios
        ),
        structural_digest=library_digest(selected_templates),
        selected_path_ids=selected_path_ids,
        selected_by_pair=selected_by_pair,
        effective_path_budget_by_pair=(
            problem.effective_path_budget_by_pair
        ),
        layout_digest=library_layout_digest(
            selected_paths, selected_templates
        ),
    )
    return OfflineLibraryMilpResult(
        library=library,
        training_configuration_by_scenario=choice,
        training_completed_by_scenario=completed,
        training_total_completed=sum(completed.values()),
        training_weighted_completed=optimal_weighted_completed,
        solver_objective=float(primary_result.fun),
        solver_mip_gap=float(getattr(primary_result, "mip_gap", 0.0)),
    )


def solve_fixed_path_schedule_library(
    problem: FixedPathLibraryProblem,
) -> OfflineLibraryMilpResult:
    """Compatibility wrapper that fixes every supplied path into the library."""

    path_values: dict[str, tuple[object, ...]] = {}
    for template in problem.templates:
        path_values.setdefault(template.path_id, template.schedule.path)
    paths = tuple(
        LibraryPathCandidate(
            pair_id=path_id,
            path_id=path_id,
            path=path,
            pool_rank=0,
        )
        for path_id, path in sorted(path_values.items())
    )
    templates = tuple(
        LibraryScheduleTemplate(
            template_id=template.template_id,
            path_id=template.path_id,
            schedule=template.schedule,
            pair_id=template.path_id,
        )
        for template in problem.templates
    )
    return solve_topology_schedule_library(TopologyLibraryProblem(
        paths=paths,
        templates=templates,
        scenarios=problem.scenarios,
        paths_per_pair=1,
        schedules_per_path=problem.schedules_per_path,
    ))
