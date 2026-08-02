"""Held-out evaluation and leakage checks for fitted CON libraries."""

from __future__ import annotations

from typing import Iterable

from .models import (
    OfflineLibraryEvaluation,
    OfflineLibraryMilpResult,
    OfflineLibraryScenario,
    ScenarioConfiguration,
    common_fingerprints,
    scenario_configurations,
)


def validate_scenario_split(
    training: Iterable[OfflineLibraryScenario],
    evaluation: Iterable[OfflineLibraryScenario],
) -> None:
    training = tuple(training)
    evaluation = tuple(evaluation)
    if not training or not evaluation:
        raise ValueError("training and evaluation scenario sets must be non-empty")
    overlapping_ids = (
        {scenario.scenario_id for scenario in training}
        & {scenario.scenario_id for scenario in evaluation}
    )
    if overlapping_ids:
        raise ValueError(
            f"training/evaluation scenario ID overlap: {sorted(overlapping_ids)}"
        )
    overlapping_traces = (
        {scenario.trace_digest for scenario in training}
        & {scenario.trace_digest for scenario in evaluation}
    )
    if overlapping_traces:
        raise ValueError(
            "training/evaluation request-trace overlap: "
            f"{sorted(overlapping_traces)}"
        )
    if common_fingerprints(training) != common_fingerprints(evaluation):
        raise ValueError(
            "training and evaluation must share topology, request distribution, "
            "and planning physics fingerprints"
        )


def _best_configuration(
    scenario: OfflineLibraryScenario,
    selected_template_ids: frozenset[str],
) -> ScenarioConfiguration:
    feasible = tuple(
        configuration
        for configuration in scenario_configurations(scenario)
        if configuration.used_template_ids <= selected_template_ids
    )
    if not feasible:
        raise RuntimeError("the automatically supplied empty configuration vanished")
    best_count = max(configuration.completed_count for configuration in feasible)
    return min(
        (
            configuration for configuration in feasible
            if configuration.completed_count == best_count
        ),
        key=lambda configuration: (
            len(configuration.used_template_ids),
            configuration.configuration_id,
        ),
    )


def evaluate_offline_library(
    result: OfflineLibraryMilpResult,
    evaluation_scenarios: Iterable[OfflineLibraryScenario],
) -> OfflineLibraryEvaluation:
    evaluation = tuple(evaluation_scenarios)
    training_stub = tuple(
        OfflineLibraryScenario(
            scenario_id=scenario_id,
            trace_digest=trace_digest,
            topology_fingerprint=result.library.topology_fingerprint,
            request_distribution_fingerprint=(
                result.library.request_distribution_fingerprint
            ),
            physics_fingerprint=result.library.physics_fingerprint,
            configurations=(ScenarioConfiguration("metadata-only-empty"),),
        )
        for scenario_id, trace_digest in zip(
            result.library.training_scenario_ids,
            result.library.training_trace_digests,
        )
    )
    validate_scenario_split(training_stub, evaluation)

    selected = frozenset(result.library.selected_template_ids)
    choice: dict[str, str] = {}
    completed: dict[str, int] = {}
    for scenario in sorted(evaluation, key=lambda value: value.scenario_id):
        configuration = _best_configuration(scenario, selected)
        choice[scenario.scenario_id] = configuration.configuration_id
        completed[scenario.scenario_id] = configuration.completed_count
    weight_by_id = {
        scenario.scenario_id: scenario.weight for scenario in evaluation
    }
    return OfflineLibraryEvaluation(
        library_structural_digest=result.library.structural_digest,
        evaluation_scenario_ids=tuple(sorted(choice)),
        evaluation_trace_digests=tuple(
            scenario.trace_digest
            for scenario in sorted(evaluation, key=lambda value: value.scenario_id)
        ),
        configuration_by_scenario=choice,
        completed_by_scenario=completed,
        total_completed=sum(completed.values()),
        weighted_completed=sum(
            weight_by_id[scenario_id] * count
            for scenario_id, count in completed.items()
        ),
    )
