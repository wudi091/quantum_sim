"""Exact binary teacher for the single-stage construction-aware LP."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .optimization_model import (
    PackingModel,
    build_delay_model,
    evaluate_expected_censored_delay,
)
from .time_expansion import TimeExpandedCandidate, TimeExpansionResult


# HiGHS can report an "Optimal" MILP with a small residual relative gap because
# its primal and dual objectives are floating-point values.  A 1e-7 threshold
# remains stricter than the solver's usual feasibility scale while accepting
# solutions that HiGHS has certified optimal up to numerical roundoff.
NUMERICAL_ZERO_MIP_GAP_TOLERANCE = 1e-7
DEFAULT_MILP_INTEGRALITY_TOLERANCE = 1e-6
NUMERICAL_OBJECTIVE_ABSOLUTE_TOLERANCE = 1e-6
NUMERICAL_OBJECTIVE_RELATIVE_TOLERANCE = 1e-7


@dataclass(frozen=True)
class DiscreteSolveResult:
    """One certified HiGHS binary solve result."""

    solve_name: str
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
    """Result of one exact binary solve using the delay objective."""

    variables: tuple[TimeExpandedCandidate, ...]
    model: PackingModel
    result: DiscreteSolveResult
    completed_request_count: int
    expected_completed_request_mass: float
    total_completion_latency: float

    @property
    def final_values(self) -> dict[str, int]:
        return {
            variable.variable_id: int(round(value))
            for variable, value in zip(self.variables, self.result.primal)
        }

    @property
    def selected_variables(self) -> tuple[TimeExpandedCandidate, ...]:
        return tuple(
            variable
            for variable, value in zip(self.variables, self.result.primal)
            if value > 0.5
        )

    @property
    def reduced_objective_value(self) -> float:
        """The solver objective without the request-censoring constant."""

        return float(self.result.objective_value)

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


def is_numerically_optimal_result(result: DiscreteSolveResult) -> bool:
    """Certify one optimal HiGHS result up to floating-point gap tolerance."""

    return (
        result.success
        and result.status == 0
        and has_numerically_zero_mip_gap(result.mip_gap)
        and result.mip_dual_bound is not None
        and math.isfinite(float(result.mip_dual_bound))
        and math.isclose(
            result.objective_value,
            float(result.mip_dual_bound),
            rel_tol=NUMERICAL_OBJECTIVE_RELATIVE_TOLERANCE,
            abs_tol=NUMERICAL_OBJECTIVE_ABSOLUTE_TOLERANCE,
        )
    )


def _constraints_for(
    model: PackingModel,
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


def _max_violation(model: PackingModel, primal: np.ndarray) -> float:
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
    """Solve the single construction-aware binary delay model exactly."""

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
    def _trivial_result(model: PackingModel) -> DiscreteSolveResult:
        primal = np.zeros(len(model.variable_ids), dtype=float)
        objective = float(model.objective @ primal)
        return DiscreteSolveResult(
            solve_name=model.name,
            success=True,
            status=0,
            message="trivial feasible MILP",
            primal=primal,
            objective_value=objective,
            mip_gap=0.0,
            mip_node_count=0,
            mip_dual_bound=objective,
        )

    def _solve_model(self, model: PackingModel) -> DiscreteSolveResult:
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
        return DiscreteSolveResult(
            solve_name=model.name,
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
        request_censoring_latencies: Mapping[str, float] | None = None,
    ) -> DiscreteOracleSolution:
        """Solve one binary model with the shared expected-delay objective."""

        raw_variables = (
            expanded.variables
            if isinstance(expanded, TimeExpansionResult)
            else expanded
        )
        variables = tuple(sorted(
            raw_variables,
            key=lambda item: item.variable_id,
        ))
        model = build_delay_model(
            variables,
            resource_capacities,
            reserved_usage,
            request_censoring_latencies=request_censoring_latencies,
        )
        result = self._solve_model(model)
        success_probabilities = np.asarray(
            [item.expected_success_probability for item in variables],
            dtype=float,
        )
        selected = result.primal > 0.5
        expected_completed_mass = float(
            success_probabilities @ selected.astype(float)
        )
        final_count = int(np.sum(selected))
        total_delay = evaluate_expected_censored_delay(model, result.primal)
        # Tiny negative values can only be floating-point roundoff when a
        # candidate reaches its censoring boundary exactly.
        if total_delay < 0.0 and total_delay > -self.feasibility_tolerance:
            total_delay = 0.0
        if total_delay < 0.0:
            raise DiscreteOracleSolveError(
                f"{model.name} produced a negative expected delay: {total_delay}"
            )
        return DiscreteOracleSolution(
            variables=variables,
            model=model,
            result=result,
            completed_request_count=final_count,
            expected_completed_request_mass=expected_completed_mass,
            total_completion_latency=total_delay,
        )


__all__ = [
    "ConstructionAwareMILPOracle",
    "DEFAULT_MILP_INTEGRALITY_TOLERANCE",
    "DiscreteOracleSolution",
    "DiscreteOracleSolveError",
    "DiscreteSolveResult",
    "NUMERICAL_ZERO_MIP_GAP_TOLERANCE",
    "has_numerically_zero_mip_gap",
    "is_numerically_optimal_result",
]
