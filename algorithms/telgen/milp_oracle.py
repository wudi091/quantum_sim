"""Small-instance binary oracle for auditing the continuous LP teacher."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .teacher import (
    LinearProgramStage,
    TeacherSolution,
    build_stage_one_lp,
    build_stage_two_lp,
)
from .time_expansion import TimeExpandedCandidate, TimeExpansionResult


@dataclass(frozen=True)
class DiscreteStageResult:
    stage_name: str
    success: bool
    status: int
    message: str
    primal: np.ndarray
    objective_value: float
    mip_gap: float | None
    mip_node_count: int | None
    mip_dual_bound: float | None


@dataclass(frozen=True)
class DiscreteOracleSolution:
    variables: tuple[TimeExpandedCandidate, ...]
    stage_one_lp: LinearProgramStage
    stage_two_lp: LinearProgramStage
    stage_one: DiscreteStageResult
    stage_two: DiscreteStageResult
    completed_request_count: int
    expected_completed_request_mass: float
    total_completion_latency: float

    @property
    def final_values(self) -> dict[str, int]:
        return {
            variable.variable_id: int(round(value))
            for variable, value in zip(self.variables, self.stage_two.primal)
        }

    @property
    def selected_variables(self) -> tuple[TimeExpandedCandidate, ...]:
        return tuple(
            variable
            for variable, value in zip(self.variables, self.stage_two.primal)
            if value > 0.5
        )


@dataclass(frozen=True)
class LPDiscreteGapReport:
    variable_count: int
    lp_completed_request_mass: float
    discrete_completed_request_count: int
    discrete_expected_completed_request_mass: float
    throughput_absolute_gap: float
    throughput_relative_gap: float | None
    latency_is_comparable: bool
    lp_total_completion_latency: float
    discrete_total_completion_latency: float
    latency_absolute_gap: float | None
    latency_relative_gap: float | None
    fractional_variable_count: int
    fractional_request_count: int
    lp_max_constraint_violation: float
    selected_variable_ids: tuple[str, ...]


class DiscreteOracleSolveError(RuntimeError):
    """Raised when the exact small-instance MILP does not reach optimality."""


def _constraints_for(lp: LinearProgramStage) -> tuple[LinearConstraint, ...]:
    constraints: list[LinearConstraint] = []
    if len(lp.b_ub):
        constraints.append(LinearConstraint(
            lp.a_ub,
            np.full(len(lp.b_ub), -np.inf, dtype=float),
            lp.b_ub,
        ))
    if len(lp.b_eq):
        constraints.append(LinearConstraint(lp.a_eq, lp.b_eq, lp.b_eq))
    return tuple(constraints)


def _max_violation(lp: LinearProgramStage, primal: np.ndarray) -> float:
    violations = [0.0]
    if len(lp.b_ub):
        violations.append(float(np.max(lp.a_ub @ primal - lp.b_ub)))
    if len(lp.b_eq):
        violations.append(float(np.max(np.abs(lp.a_eq @ primal - lp.b_eq))))
    if len(primal):
        violations.append(float(np.max(-primal)))
        violations.append(float(np.max(primal - 1.0)))
        violations.append(float(np.max(np.abs(primal - np.rint(primal)))))
    return max(0.0, *violations)


class ConstructionAwareMILPOracle:
    """Two-stage binary oracle sharing the teacher's exact LP matrices."""

    def __init__(
        self,
        *,
        time_limit_seconds: float = 60.0,
        mip_relative_gap: float = 0.0,
        feasibility_tolerance: float = 1e-7,
    ):
        if time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")
        if mip_relative_gap < 0:
            raise ValueError("mip_relative_gap cannot be negative")
        if feasibility_tolerance <= 0:
            raise ValueError("feasibility_tolerance must be positive")
        self.time_limit_seconds = float(time_limit_seconds)
        self.mip_relative_gap = float(mip_relative_gap)
        self.feasibility_tolerance = float(feasibility_tolerance)

    @staticmethod
    def _trivial_result(lp: LinearProgramStage) -> DiscreteStageResult:
        primal = np.zeros(len(lp.variable_ids), dtype=float)
        return DiscreteStageResult(
            stage_name=lp.name,
            success=True,
            status=0,
            message="trivial feasible MILP",
            primal=primal,
            objective_value=float(lp.objective @ primal),
            mip_gap=0.0,
            mip_node_count=0,
            mip_dual_bound=float(lp.objective @ primal),
        )

    def _solve_stage(self, lp: LinearProgramStage) -> DiscreteStageResult:
        variable_count = len(lp.variable_ids)
        if variable_count == 0:
            return self._trivial_result(lp)
        result = milp(
            c=lp.objective,
            integrality=np.ones(variable_count, dtype=np.int32),
            bounds=Bounds(
                np.zeros(variable_count, dtype=float),
                np.ones(variable_count, dtype=float),
            ),
            constraints=_constraints_for(lp),
            options={
                "time_limit": self.time_limit_seconds,
                "mip_rel_gap": self.mip_relative_gap,
                "presolve": True,
                "disp": False,
            },
        )
        if not result.success or result.x is None:
            raise DiscreteOracleSolveError(
                f"{lp.name} failed: status={result.status}, {result.message}"
            )
        raw_primal = np.asarray(result.x, dtype=float)
        primal = np.rint(raw_primal)
        if np.max(np.abs(raw_primal - primal)) > self.feasibility_tolerance:
            raise DiscreteOracleSolveError(
                f"{lp.name} returned a non-integral incumbent"
            )
        violation = _max_violation(lp, primal)
        if violation > self.feasibility_tolerance:
            raise DiscreteOracleSolveError(
                f"{lp.name} returned an infeasible incumbent: violation={violation}"
            )
        return DiscreteStageResult(
            stage_name=lp.name,
            success=True,
            status=int(result.status),
            message=str(result.message),
            primal=primal,
            objective_value=float(lp.objective @ primal),
            mip_gap=(
                None if getattr(result, "mip_gap", None) is None
                else float(result.mip_gap)
            ),
            mip_node_count=(
                None if getattr(result, "mip_node_count", None) is None
                else int(result.mip_node_count)
            ),
            mip_dual_bound=(
                None if getattr(result, "mip_dual_bound", None) is None
                else float(result.mip_dual_bound)
            ),
        )

    def solve(
        self,
        expanded: TimeExpansionResult | Sequence[TimeExpandedCandidate],
        resource_capacities: Mapping[str, int],
        *,
        reserved_usage: Mapping[tuple[str, int], int] | None = None,
    ) -> DiscreteOracleSolution:
        raw_variables = (
            expanded.variables
            if isinstance(expanded, TimeExpansionResult)
            else expanded
        )
        variables = tuple(sorted(
            raw_variables,
            key=lambda item: item.variable_id,
        ))
        stage_one_lp = build_stage_one_lp(
            variables,
            resource_capacities,
            reserved_usage,
        )
        stage_one = self._solve_stage(stage_one_lp)
        success_probabilities = np.asarray(
            [item.expected_success_probability for item in variables],
            dtype=float,
        )
        expected_completed_mass = float(
            success_probabilities @ stage_one.primal
        )
        stage_two_lp = build_stage_two_lp(
            variables,
            resource_capacities,
            expected_completed_mass,
            reserved_usage,
        )
        stage_two = self._solve_stage(stage_two_lp)
        final_expected_mass = float(success_probabilities @ stage_two.primal)
        if (
            abs(final_expected_mass - expected_completed_mass)
            > self.feasibility_tolerance
        ):
            raise DiscreteOracleSolveError(
                "second-stage MILP did not preserve optimal expected throughput"
            )
        final_count = int(round(float(np.sum(stage_two.primal))))
        return DiscreteOracleSolution(
            variables=variables,
            stage_one_lp=stage_one_lp,
            stage_two_lp=stage_two_lp,
            stage_one=stage_one,
            stage_two=stage_two,
            completed_request_count=final_count,
            expected_completed_request_mass=final_expected_mass,
            total_completion_latency=float(
                stage_two_lp.objective @ stage_two.primal
            ),
        )


def compare_lp_and_milp(
    lp_solution: TeacherSolution,
    discrete_solution: DiscreteOracleSolution,
    *,
    tolerance: float = 1e-7,
) -> LPDiscreteGapReport:
    """Compare the relaxation upper bound with the exact binary optimum."""

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    lp_ids = tuple(variable.variable_id for variable in lp_solution.variables)
    discrete_ids = tuple(
        variable.variable_id for variable in discrete_solution.variables
    )
    if lp_ids != discrete_ids:
        raise ValueError("LP and MILP solutions use different variables")

    lp_completed = float(lp_solution.stage_one_completed_mass)
    discrete_completed = discrete_solution.expected_completed_request_mass
    raw_throughput_gap = lp_completed - discrete_completed
    numerical_tolerance = 10.0 * tolerance
    if raw_throughput_gap < -numerical_tolerance:
        raise ValueError("MILP throughput exceeds the LP relaxation bound")
    throughput_gap = max(0.0, raw_throughput_gap)
    throughput_relative_gap = (
        throughput_gap / discrete_completed
        if discrete_completed > 0 else None
    )

    latency_is_comparable = throughput_gap <= numerical_tolerance
    latency_absolute_gap: float | None = None
    latency_relative_gap: float | None = None
    if latency_is_comparable:
        raw_latency_gap = (
            discrete_solution.total_completion_latency
            - lp_solution.total_completion_latency
        )
        if raw_latency_gap < -numerical_tolerance:
            raise ValueError("MILP latency is below the LP relaxation bound")
        latency_absolute_gap = max(0.0, raw_latency_gap)
        latency_relative_gap = (
            latency_absolute_gap / discrete_solution.total_completion_latency
            if discrete_solution.total_completion_latency > tolerance else 0.0
        )

    values = np.asarray(lp_solution.stage_two.primal, dtype=float)
    fractional_variable_count = int(np.sum(
        (values > tolerance) & (values < 1.0 - tolerance)
    ))
    values_by_request: dict[str, list[float]] = {}
    for variable, value in zip(lp_solution.variables, values):
        values_by_request.setdefault(variable.request_id, []).append(float(value))
    fractional_request_count = 0
    for request_values in values_by_request.values():
        active = [value for value in request_values if value > tolerance]
        if active and not (
            len(active) == 1 and abs(active[0] - 1.0) <= tolerance
        ):
            fractional_request_count += 1

    return LPDiscreteGapReport(
        variable_count=len(lp_ids),
        lp_completed_request_mass=lp_completed,
        discrete_completed_request_count=(
            discrete_solution.completed_request_count
        ),
        discrete_expected_completed_request_mass=discrete_completed,
        throughput_absolute_gap=throughput_gap,
        throughput_relative_gap=throughput_relative_gap,
        latency_is_comparable=latency_is_comparable,
        lp_total_completion_latency=lp_solution.total_completion_latency,
        discrete_total_completion_latency=(
            discrete_solution.total_completion_latency
        ),
        latency_absolute_gap=latency_absolute_gap,
        latency_relative_gap=latency_relative_gap,
        fractional_variable_count=fractional_variable_count,
        fractional_request_count=fractional_request_count,
        lp_max_constraint_violation=float(
            lp_solution.stage_two.max_violation_trajectory[-1]
        ),
        selected_variable_ids=tuple(
            variable.variable_id
            for variable in discrete_solution.selected_variables
        ),
    )


def save_gap_report(
    report: LPDiscreteGapReport,
    path: str | Path,
    *,
    context: Mapping[str, object] | None = None,
) -> Path:
    target = Path(path)
    if target.suffix.lower() != ".json":
        raise ValueError("gap report path must end with .json")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    if context is not None:
        payload["context"] = dict(context)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target
