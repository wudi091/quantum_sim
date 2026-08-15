"""Exact two-stage binary teacher for construction-aware planning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .optimization_model import (
    PackingModelStage,
    build_stage_one_model,
    build_stage_two_model,
)
from .time_expansion import TimeExpandedCandidate, TimeExpansionResult


# HiGHS can report an "Optimal" MILP with a small residual relative gap because
# its primal and dual objectives are floating-point values.  A 1e-7 threshold
# remains stricter than the solver's usual feasibility scale while accepting
# solutions that HiGHS has certified optimal up to numerical roundoff.
NUMERICAL_ZERO_MIP_GAP_TOLERANCE = 1e-7
DEFAULT_MILP_INTEGRALITY_TOLERANCE = 1e-6
# HiGHS may round a certified optimal primal/dual pair slightly apart.  Require
# agreement at the same relative scale as the accepted numerical MIP gap and
# retain a small absolute floor for objectives close to zero.
NUMERICAL_OBJECTIVE_ABSOLUTE_TOLERANCE = 1e-6
NUMERICAL_OBJECTIVE_RELATIVE_TOLERANCE = 1e-7


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
    stage_one_model: PackingModelStage
    stage_two_model: PackingModelStage
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


class DiscreteOracleSolveError(RuntimeError):
    """Raised when the exact MILP teacher does not return a valid solution."""


def has_numerically_zero_mip_gap(
    mip_gap: float | None,
    *,
    tolerance: float = NUMERICAL_ZERO_MIP_GAP_TOLERANCE,
) -> bool:
    """Return whether a solver-reported MILP gap is numerical zero."""

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if mip_gap is None:
        return False
    value = float(mip_gap)
    return math.isfinite(value) and abs(value) <= tolerance


def is_numerically_optimal_stage(stage: DiscreteStageResult) -> bool:
    """Certify an optimal HiGHS stage up to floating-point gap tolerance."""

    return (
        stage.success
        and stage.status == 0
        and has_numerically_zero_mip_gap(stage.mip_gap)
        and stage.mip_dual_bound is not None
        and math.isfinite(float(stage.mip_dual_bound))
        and math.isclose(
            stage.objective_value,
            float(stage.mip_dual_bound),
            rel_tol=NUMERICAL_OBJECTIVE_RELATIVE_TOLERANCE,
            abs_tol=NUMERICAL_OBJECTIVE_ABSOLUTE_TOLERANCE,
        )
    )


def _constraints_for(
    model: PackingModelStage,
) -> tuple[LinearConstraint, ...]:
    constraints: list[LinearConstraint] = []
    if len(model.b_ub):
        constraints.append(LinearConstraint(
            model.a_ub,
            np.full(len(model.b_ub), -np.inf, dtype=float),
            model.b_ub,
        ))
    if len(model.b_eq):
        constraints.append(LinearConstraint(
            model.a_eq,
            model.b_eq,
            model.b_eq,
        ))
    return tuple(constraints)


def _max_violation(model: PackingModelStage, primal: np.ndarray) -> float:
    violations = [0.0]
    if len(model.b_ub):
        violations.append(float(np.max(
            model.a_ub @ primal - model.b_ub
        )))
    if len(model.b_eq):
        violations.append(float(np.max(np.abs(
            model.a_eq @ primal - model.b_eq
        ))))
    if len(primal):
        violations.append(float(np.max(-primal)))
        violations.append(float(np.max(primal - 1.0)))
        violations.append(float(np.max(np.abs(primal - np.rint(primal)))))
    return max(0.0, *violations)


class ConstructionAwareMILPOracle:
    """Solve the lexicographic construction-aware packing MILP exactly."""

    def __init__(
        self,
        *,
        time_limit_seconds: float = 60.0,
        mip_relative_gap: float = 0.0,
        integrality_tolerance: float = DEFAULT_MILP_INTEGRALITY_TOLERANCE,
        feasibility_tolerance: float = 1e-7,
    ):
        if time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")
        if mip_relative_gap < 0:
            raise ValueError("mip_relative_gap cannot be negative")
        if integrality_tolerance <= 0:
            raise ValueError("integrality_tolerance must be positive")
        if feasibility_tolerance <= 0:
            raise ValueError("feasibility_tolerance must be positive")
        self.time_limit_seconds = float(time_limit_seconds)
        self.mip_relative_gap = float(mip_relative_gap)
        self.integrality_tolerance = float(integrality_tolerance)
        self.feasibility_tolerance = float(feasibility_tolerance)

    @staticmethod
    def _trivial_result(model: PackingModelStage) -> DiscreteStageResult:
        primal = np.zeros(len(model.variable_ids), dtype=float)
        return DiscreteStageResult(
            stage_name=model.name,
            success=True,
            status=0,
            message="trivial feasible MILP",
            primal=primal,
            objective_value=float(model.objective @ primal),
            mip_gap=0.0,
            mip_node_count=0,
            mip_dual_bound=float(model.objective @ primal),
        )

    def _solve_stage(self, model: PackingModelStage) -> DiscreteStageResult:
        variable_count = len(model.variable_ids)
        if variable_count == 0:
            return self._trivial_result(model)
        result = milp(
            c=model.objective,
            integrality=np.ones(variable_count, dtype=np.int32),
            bounds=Bounds(
                np.zeros(variable_count, dtype=float),
                np.ones(variable_count, dtype=float),
            ),
            constraints=_constraints_for(model),
            options={
                "time_limit": self.time_limit_seconds,
                "mip_rel_gap": self.mip_relative_gap,
                "presolve": True,
                "disp": False,
            },
        )
        if not result.success or result.x is None:
            raise DiscreteOracleSolveError(
                f"{model.name} failed: status={result.status}, {result.message}"
            )
        raw_primal = np.asarray(result.x, dtype=float)
        primal = np.rint(raw_primal)
        max_integrality_deviation = float(
            np.max(np.abs(raw_primal - primal))
        )
        if max_integrality_deviation > self.integrality_tolerance:
            raise DiscreteOracleSolveError(
                f"{model.name} returned a non-integral incumbent: "
                f"max_deviation={max_integrality_deviation}, "
                f"tolerance={self.integrality_tolerance}"
            )
        violation = _max_violation(model, primal)
        if violation > self.feasibility_tolerance:
            raise DiscreteOracleSolveError(
                f"{model.name} returned an infeasible incumbent: "
                f"violation={violation}"
            )
        return DiscreteStageResult(
            stage_name=model.name,
            success=True,
            status=int(result.status),
            message=str(result.message),
            primal=primal,
            objective_value=float(model.objective @ primal),
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
        stage_one_model = build_stage_one_model(
            variables,
            resource_capacities,
            reserved_usage,
        )
        stage_one = self._solve_stage(stage_one_model)
        success_probabilities = np.asarray(
            [item.expected_success_probability for item in variables],
            dtype=float,
        )
        expected_completed_mass = float(
            success_probabilities @ stage_one.primal
        )
        stage_two_model = build_stage_two_model(
            variables,
            resource_capacities,
            expected_completed_mass,
            reserved_usage,
        )
        stage_two = self._solve_stage(stage_two_model)
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
            stage_one_model=stage_one_model,
            stage_two_model=stage_two_model,
            stage_one=stage_one,
            stage_two=stage_two,
            completed_request_count=final_count,
            expected_completed_request_mass=final_expected_mass,
            total_completion_latency=float(
                stage_two_model.objective @ stage_two.primal
            ),
        )
