"""Immutable data contracts for CON's offline schedule library."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Iterable

from qnet_core.contracts.complete_schedule import CompleteSchedule


@dataclass(frozen=True)
class LibraryScheduleTemplate:
    template_id: str
    path_id: str
    schedule: CompleteSchedule
    pair_id: str | None = None

    def __post_init__(self) -> None:
        if not self.template_id or not self.path_id:
            raise ValueError("template_id and path_id must be non-empty")
        if self.pair_id is not None and not self.pair_id:
            raise ValueError("pair_id must be non-empty when supplied")

    @property
    def source(self):
        return self.schedule.path[0]

    @property
    def destination(self):
        return self.schedule.path[-1]

    @property
    def structural_key(self):
        return self.schedule.structural_key


@dataclass(frozen=True)
class LibraryPathCandidate:
    """One topology-bound path eligible for the offline portfolio."""

    pair_id: str
    path_id: str
    path: tuple[object, ...]
    pool_rank: int

    def __post_init__(self) -> None:
        path = tuple(self.path)
        if not self.pair_id or not self.path_id:
            raise ValueError("pair_id and path_id must be non-empty")
        if len(path) < 2 or len(set(path)) != len(path):
            raise ValueError("library path candidates must be simple paths")
        if self.pool_rank < 0:
            raise ValueError("path pool_rank cannot be negative")
        object.__setattr__(self, "path", path)


@dataclass(frozen=True)
class ScenarioConfiguration:
    """One fully validated joint decision available in an offline scenario."""

    configuration_id: str
    used_template_ids: frozenset[str] = frozenset()
    completed_request_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.configuration_id:
            raise ValueError("configuration_id must be non-empty")
        if any(not value for value in self.used_template_ids):
            raise ValueError("used template IDs must be non-empty")
        if any(not value for value in self.completed_request_ids):
            raise ValueError("completed request IDs must be non-empty")

    @property
    def completed_count(self) -> int:
        return len(self.completed_request_ids)

    @property
    def is_empty(self) -> bool:
        return not self.used_template_ids and not self.completed_request_ids


@dataclass(frozen=True)
class OfflineLibraryScenario:
    scenario_id: str
    trace_digest: str
    topology_fingerprint: str
    request_distribution_fingerprint: str
    physics_fingerprint: str
    configurations: tuple[ScenarioConfiguration, ...]
    weight: int = 1

    def __post_init__(self) -> None:
        for name in (
            "scenario_id",
            "trace_digest",
            "topology_fingerprint",
            "request_distribution_fingerprint",
            "physics_fingerprint",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.weight < 1:
            raise ValueError("scenario weight must be a positive integer")
        if not self.configurations:
            raise ValueError("an offline scenario needs feasible configurations")
        ids = tuple(
            configuration.configuration_id
            for configuration in self.configurations
        )
        if len(set(ids)) != len(ids):
            raise ValueError("configuration IDs must be unique within a scenario")


@dataclass(frozen=True)
class FixedPathLibraryProblem:
    templates: tuple[LibraryScheduleTemplate, ...]
    scenarios: tuple[OfflineLibraryScenario, ...]
    schedules_per_path: int = 4

    def __post_init__(self) -> None:
        if not self.templates:
            raise ValueError("the offline library needs schedule templates")
        if not self.scenarios:
            raise ValueError("the offline library needs training scenarios")
        if self.schedules_per_path < 1:
            raise ValueError("schedules_per_path must be positive")
        template_ids = tuple(template.template_id for template in self.templates)
        if len(set(template_ids)) != len(template_ids):
            raise ValueError("template IDs must be globally unique")

        template_set = set(template_ids)
        path_by_id: dict[str, tuple[object, ...]] = {}
        structural_keys_by_path: dict[str, set[object]] = {}
        for template in self.templates:
            previous = path_by_id.setdefault(
                template.path_id, template.schedule.path
            )
            if previous != template.schedule.path:
                raise ValueError("one path_id cannot refer to different paths")
            keys = structural_keys_by_path.setdefault(template.path_id, set())
            if template.structural_key in keys:
                raise ValueError(
                    "one path pool cannot contain duplicate schedule structures"
                )
            keys.add(template.structural_key)

        scenario_ids = tuple(scenario.scenario_id for scenario in self.scenarios)
        trace_digests = tuple(
            scenario.trace_digest for scenario in self.scenarios
        )
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("training scenario IDs must be unique")
        if len(set(trace_digests)) != len(trace_digests):
            raise ValueError("training request traces must be unique")
        for scenario in self.scenarios:
            for configuration in scenario.configurations:
                unknown = configuration.used_template_ids - template_set
                if unknown:
                    raise ValueError(
                        f"scenario uses unknown templates: {sorted(unknown)}"
                    )
        common_fingerprints(self.scenarios)

    @property
    def effective_budget_by_path(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for template in self.templates:
            counts[template.path_id] = counts.get(template.path_id, 0) + 1
        return {
            path_id: min(self.schedules_per_path, count)
            for path_id, count in counts.items()
        }


@dataclass(frozen=True)
class TopologyLibraryProblem:
    """Joint path-and-schedule portfolio problem for every topology pair."""

    paths: tuple[LibraryPathCandidate, ...]
    templates: tuple[LibraryScheduleTemplate, ...]
    scenarios: tuple[OfflineLibraryScenario, ...]
    paths_per_pair: int = 4
    schedules_per_path: int = 4

    def __post_init__(self) -> None:
        if not self.paths:
            raise ValueError("the topology library needs path candidates")
        if not self.templates:
            raise ValueError("the topology library needs schedule templates")
        if not self.scenarios:
            raise ValueError("the topology library needs training scenarios")
        if self.paths_per_pair < 1 or self.schedules_per_path < 1:
            raise ValueError("path and schedule budgets must be positive")

        path_ids = tuple(path.path_id for path in self.paths)
        if len(set(path_ids)) != len(path_ids):
            raise ValueError("path IDs must be globally unique")
        ranks_by_pair: dict[str, set[int]] = {}
        path_by_id = {path.path_id: path for path in self.paths}
        for path in self.paths:
            ranks = ranks_by_pair.setdefault(path.pair_id, set())
            if path.pool_rank in ranks:
                raise ValueError("path pool ranks must be unique within a pair")
            ranks.add(path.pool_rank)

        template_ids = tuple(
            template.template_id for template in self.templates
        )
        if len(set(template_ids)) != len(template_ids):
            raise ValueError("template IDs must be globally unique")
        structural_keys_by_path: dict[str, set[object]] = {}
        template_count_by_path: dict[str, int] = {}
        for template in self.templates:
            try:
                path = path_by_id[template.path_id]
            except KeyError as exc:
                raise ValueError(
                    "schedule template refers to an unknown path"
                ) from exc
            if template.pair_id != path.pair_id:
                raise ValueError(
                    "topology schedule templates need the path's pair_id"
                )
            if template.schedule.path != path.path:
                raise ValueError("schedule template is attached to the wrong path")
            keys = structural_keys_by_path.setdefault(template.path_id, set())
            if template.structural_key in keys:
                raise ValueError(
                    "one path pool cannot contain duplicate schedule structures"
                )
            keys.add(template.structural_key)
            template_count_by_path[template.path_id] = (
                template_count_by_path.get(template.path_id, 0) + 1
            )
        missing_templates = set(path_ids) - set(template_count_by_path)
        if missing_templates:
            raise ValueError(
                "every selectable path needs at least one complete schedule"
            )

        scenario_ids = tuple(scenario.scenario_id for scenario in self.scenarios)
        trace_digests = tuple(
            scenario.trace_digest for scenario in self.scenarios
        )
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("training scenario IDs must be unique")
        if len(set(trace_digests)) != len(trace_digests):
            raise ValueError("training request traces must be unique")
        template_set = set(template_ids)
        for scenario in self.scenarios:
            for configuration in scenario.configurations:
                unknown = configuration.used_template_ids - template_set
                if unknown:
                    raise ValueError(
                        f"scenario uses unknown templates: {sorted(unknown)}"
                    )
        common_fingerprints(self.scenarios)

    @property
    def effective_path_budget_by_pair(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for path in self.paths:
            counts[path.pair_id] = counts.get(path.pair_id, 0) + 1
        return {
            pair_id: min(self.paths_per_pair, count)
            for pair_id, count in counts.items()
        }

    @property
    def effective_schedule_budget_by_path(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for template in self.templates:
            counts[template.path_id] = counts.get(template.path_id, 0) + 1
        return {
            path_id: min(self.schedules_per_path, count)
            for path_id, count in counts.items()
        }


@dataclass(frozen=True)
class OfflineScheduleLibrary:
    selected_template_ids: tuple[str, ...]
    selected_by_path: dict[str, tuple[str, ...]]
    effective_budget_by_path: dict[str, int]
    topology_fingerprint: str
    request_distribution_fingerprint: str
    physics_fingerprint: str
    training_scenario_ids: tuple[str, ...]
    training_trace_digests: tuple[str, ...]
    structural_digest: str
    selected_path_ids: tuple[str, ...] = ()
    selected_by_pair: dict[str, tuple[str, ...]] = field(default_factory=dict)
    effective_path_budget_by_pair: dict[str, int] = field(default_factory=dict)
    layout_digest: str = ""


@dataclass(frozen=True)
class OfflineLibraryMilpResult:
    library: OfflineScheduleLibrary
    training_configuration_by_scenario: dict[str, str]
    training_completed_by_scenario: dict[str, int]
    training_total_completed: int
    training_weighted_completed: int
    solver_objective: float
    solver_mip_gap: float


@dataclass(frozen=True)
class OfflineLibraryEvaluation:
    library_structural_digest: str
    evaluation_scenario_ids: tuple[str, ...]
    evaluation_trace_digests: tuple[str, ...]
    configuration_by_scenario: dict[str, str]
    completed_by_scenario: dict[str, int]
    total_completed: int
    weighted_completed: int


def common_fingerprints(
    scenarios: Iterable[OfflineLibraryScenario],
) -> tuple[str, str, str]:
    scenarios = tuple(scenarios)
    topology = {scenario.topology_fingerprint for scenario in scenarios}
    distribution = {
        scenario.request_distribution_fingerprint for scenario in scenarios
    }
    physics = {scenario.physics_fingerprint for scenario in scenarios}
    if len(topology) != 1:
        raise ValueError("all scenarios must use the same fixed topology")
    if len(distribution) != 1:
        raise ValueError("all scenarios must use the same request distribution")
    if len(physics) != 1:
        raise ValueError("all scenarios must use the same planning physics model")
    return next(iter(topology)), next(iter(distribution)), next(iter(physics))


def scenario_configurations(
    scenario: OfflineLibraryScenario,
) -> tuple[ScenarioConfiguration, ...]:
    configurations = tuple(scenario.configurations)
    if any(configuration.is_empty for configuration in configurations):
        return configurations
    return configurations + (ScenarioConfiguration(
        configuration_id=f"{scenario.scenario_id}:empty",
    ),)


def library_digest(
    templates: Iterable[LibraryScheduleTemplate],
) -> str:
    payload = tuple(sorted(
        (
            template.template_id,
            template.pair_id or "",
            template.path_id,
            tuple(map(repr, template.schedule.path)),
            template.schedule.structural_key,
        )
        for template in templates
    ))
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def library_layout_digest(
    paths: Iterable[LibraryPathCandidate],
    templates: Iterable[LibraryScheduleTemplate],
) -> str:
    """Hash ordered pair/path/schedule slots, not only the selected set."""

    path_payload = tuple(
        (
            path.pair_id,
            path.path_id,
            path.pool_rank,
            tuple(map(repr, path.path)),
        )
        for path in paths
    )
    template_payload = tuple(
        (
            template.pair_id,
            template.path_id,
            template.template_id,
            template.structural_key,
        )
        for template in templates
    )
    return hashlib.sha256(
        repr((path_payload, template_payload)).encode("utf-8")
    ).hexdigest()
