"""Two-stage continuous LP teacher with interior-point trajectories.

The teacher solves the time-expanded candidate packing problem
lexicographically:

1. maximize expected completed-request mass;
2. hold that expected mass fixed and minimize expected completion latency.

The exact discrete problem is intentionally not claimed here.  Every LP
variable remains continuous in ``[0, 1]`` so that the complete primal IPM
trajectory can supervise a later TELGEN-style GNN.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence
import warnings

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, spmatrix

from .time_expansion import (
    TimeExpandedCandidate,
    TimeExpansionResult,
    normalize_reserved_usage,
)


@dataclass(frozen=True)
class ConstraintDescriptor:
    """Semantic identity of one LP row for later graph construction."""

    constraint_id: str
    kind: str
    rhs: float
    sense: str = "<="
    request_id: str | None = None
    resource_id: str | None = None
    slot: int | None = None


@dataclass(frozen=True)
class LinearProgramStage:
    """Sparse LP representation plus semantic row metadata."""

    name: str
    variable_ids: tuple[str, ...]
    objective: np.ndarray
    a_ub: spmatrix
    b_ub: np.ndarray
    ub_constraints: tuple[ConstraintDescriptor, ...]
    a_eq: spmatrix
    b_eq: np.ndarray
    eq_constraints: tuple[ConstraintDescriptor, ...]

    def __post_init__(self) -> None:
        variable_count = len(self.variable_ids)
        if self.objective.shape != (variable_count,):
            raise ValueError("objective has the wrong shape")
        if self.a_ub.shape != (len(self.b_ub), variable_count):
            raise ValueError("A_ub has the wrong shape")
        if self.a_eq.shape != (len(self.b_eq), variable_count):
            raise ValueError("A_eq has the wrong shape")
        if len(self.ub_constraints) != len(self.b_ub):
            raise ValueError("inequality metadata does not match A_ub")
        if len(self.eq_constraints) != len(self.b_eq):
            raise ValueError("equality metadata does not match A_eq")


@dataclass(frozen=True)
class TeacherStageResult:
    """One LP solve and the available primal trajectory."""

    stage_name: str
    solver_backend: str
    trajectory_complete: bool
    success: bool
    message: str
    iterations: int
    primal: np.ndarray
    primal_trajectory: np.ndarray
    objective_value: float
    objective_trajectory: np.ndarray
    max_violation_trajectory: np.ndarray


@dataclass(frozen=True)
class TeacherSolution:
    """Teacher output consumed by future graph-dataset generation."""

    variables: tuple[TimeExpandedCandidate, ...]
    stage_one_lp: LinearProgramStage
    stage_two_lp: LinearProgramStage
    stage_one: TeacherStageResult
    stage_two: TeacherStageResult
    stage_one_completed_mass: float
    completed_request_mass: float
    total_completion_latency: float

    @property
    def final_values(self) -> dict[str, float]:
        return {
            variable.variable_id: float(value)
            for variable, value in zip(self.variables, self.stage_two.primal)
        }


class TeacherSolveError(RuntimeError):
    """Raised when the required trajectory-producing IPM does not converge."""


def _empty_matrix(rows: int, columns: int) -> csr_matrix:
    return csr_matrix((rows, columns), dtype=float)


def _base_constraints(
    variables: Sequence[TimeExpandedCandidate],
    resource_capacities: Mapping[str, int],
    reserved_usage: Mapping[tuple[str, int], int] | None = None,
) -> tuple[csr_matrix, np.ndarray, tuple[ConstraintDescriptor, ...]]:
    variable_count = len(variables)
    request_rows: dict[str, list[int]] = {}
    resource_rows: dict[tuple[str, int], dict[int, int]] = {}
    for index, variable in enumerate(variables):
        request_rows.setdefault(variable.request_id, []).append(index)
        for usage in variable.resource_usage:
            resource_rows.setdefault(
                (usage.resource_id, usage.slot), {}
            )[index] = usage.amount

    row_indices: list[int] = []
    column_indices: list[int] = []
    data: list[float] = []
    rhs: list[float] = []
    descriptors: list[ConstraintDescriptor] = []
    for request_id in sorted(request_rows):
        row_index = len(rhs)
        indices = request_rows[request_id]
        row_indices.extend([row_index] * len(indices))
        column_indices.extend(indices)
        data.extend([1.0] * len(indices))
        rhs.append(1.0)
        descriptors.append(ConstraintDescriptor(
            constraint_id=f"request:{request_id}",
            kind="request",
            rhs=1.0,
            request_id=request_id,
        ))

    capacities = {str(key): int(value) for key, value in resource_capacities.items()}
    reservations = normalize_reserved_usage(reserved_usage, capacities)
    for (resource_id, slot), coefficients in sorted(resource_rows.items()):
        if resource_id not in capacities:
            raise ValueError(f"missing capacity for resource: {resource_id}")
        capacity = capacities[resource_id]
        if capacity < 1:
            raise ValueError("resource capacities must be positive")
        row_index = len(rhs)
        for index, amount in coefficients.items():
            row_indices.append(row_index)
            column_indices.append(index)
            data.append(float(amount))
        residual_capacity = capacity - reservations.get((resource_id, slot), 0)
        rhs.append(float(residual_capacity))
        descriptors.append(ConstraintDescriptor(
            constraint_id=f"resource:{resource_id}:slot:{slot}",
            kind="resource_time",
            rhs=float(residual_capacity),
            resource_id=resource_id,
            slot=slot,
        ))

    matrix = coo_matrix(
        (data, (row_indices, column_indices)),
        shape=(len(rhs), variable_count),
        dtype=float,
    ).tocsr()
    return matrix, np.asarray(rhs, dtype=float), tuple(descriptors)


def build_stage_one_lp(
    variables: Sequence[TimeExpandedCandidate],
    resource_capacities: Mapping[str, int],
    reserved_usage: Mapping[tuple[str, int], int] | None = None,
) -> LinearProgramStage:
    """Build expected-completion maximization as a minimization LP."""

    ordered = tuple(sorted(variables, key=lambda item: item.variable_id))
    a_ub, b_ub, descriptors = _base_constraints(
        ordered,
        resource_capacities,
        reserved_usage,
    )
    variable_count = len(ordered)
    return LinearProgramStage(
        name="maximize_expected_completed_requests",
        variable_ids=tuple(item.variable_id for item in ordered),
        objective=-np.asarray(
            [item.expected_success_probability for item in ordered],
            dtype=float,
        ),
        a_ub=a_ub,
        b_ub=b_ub,
        ub_constraints=descriptors,
        a_eq=_empty_matrix(0, variable_count),
        b_eq=np.zeros(0, dtype=float),
        eq_constraints=(),
    )


def build_stage_two_lp(
    variables: Sequence[TimeExpandedCandidate],
    resource_capacities: Mapping[str, int],
    completed_mass: float,
    reserved_usage: Mapping[tuple[str, int], int] | None = None,
) -> LinearProgramStage:
    """Fix expected throughput and minimize expected completion latency."""

    ordered = tuple(sorted(variables, key=lambda item: item.variable_id))
    a_ub, b_ub, descriptors = _base_constraints(
        ordered,
        resource_capacities,
        reserved_usage,
    )
    variable_count = len(ordered)
    success_probabilities = np.asarray(
        [item.expected_success_probability for item in ordered],
        dtype=float,
    )
    equality = csr_matrix(success_probabilities.reshape(1, -1))
    return LinearProgramStage(
        name="minimize_expected_completion_latency",
        variable_ids=tuple(item.variable_id for item in ordered),
        objective=np.asarray(
            [
                item.expected_success_probability * item.completion_latency
                for item in ordered
            ],
            dtype=float,
        ),
        a_ub=a_ub,
        b_ub=b_ub,
        ub_constraints=descriptors,
        a_eq=equality,
        b_eq=np.asarray([float(completed_mass)], dtype=float),
        eq_constraints=(ConstraintDescriptor(
            constraint_id="throughput:stage_one_optimum",
            kind="throughput_equality",
            rhs=float(completed_mass),
            sense="=",
        ),),
    )


def _scale_rows(
    matrix: spmatrix, rhs: np.ndarray
) -> tuple[csr_matrix, np.ndarray]:
    sparse_matrix = csr_matrix(matrix, dtype=float)
    if len(rhs) == 0:
        return sparse_matrix, rhs
    row_max = (
        np.asarray(abs(sparse_matrix).max(axis=1).toarray()).ravel()
        if sparse_matrix.shape[1]
        else np.zeros(len(rhs))
    )
    scale = np.maximum(row_max, np.abs(rhs))
    scale[scale <= np.finfo(float).eps] = 1.0
    return sparse_matrix.multiply((1.0 / scale)[:, None]).tocsr(), rhs / scale


def _max_violation(lp: LinearProgramStage, primal: np.ndarray) -> float:
    violations = [0.0]
    if len(lp.b_ub):
        residual = np.asarray(lp.a_ub @ primal).ravel() - lp.b_ub
        violations.append(float(np.max(residual)))
    if len(lp.b_eq):
        residual = np.asarray(lp.a_eq @ primal).ravel() - lp.b_eq
        violations.append(float(np.max(np.abs(residual))))
    if len(primal):
        violations.append(float(np.max(-primal)))
        violations.append(float(np.max(primal - 1.0)))
    return max(0.0, *violations)


class ConstructionAwareLPTeacher:
    """Two-stage LP teacher with trajectory and scalable IPM backends."""

    def __init__(
        self,
        *,
        tolerance: float = 1e-7,
        max_iterations: int = 200,
        solver_backend: str = "trajectory_ipm",
    ):
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if solver_backend not in {"trajectory_ipm", "highs_ipm"}:
            raise ValueError(f"unknown solver backend: {solver_backend}")
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.solver_backend = solver_backend

    def _trivial_result(
        self, lp: LinearProgramStage, primal: np.ndarray | None = None
    ) -> TeacherStageResult:
        if primal is None:
            primal = np.zeros(len(lp.variable_ids), dtype=float)
        trajectory = primal.reshape(1, -1)
        return TeacherStageResult(
            stage_name=lp.name,
            solver_backend="trivial",
            trajectory_complete=True,
            success=True,
            message="trivial feasible LP",
            iterations=0,
            primal=primal,
            primal_trajectory=trajectory,
            objective_value=float(lp.objective @ primal),
            objective_trajectory=np.asarray(
                [float(lp.objective @ primal)], dtype=float
            ),
            max_violation_trajectory=np.asarray(
                [_max_violation(lp, primal)], dtype=float
            ),
        )

    def _normalize_primal(self, raw_primal: np.ndarray) -> np.ndarray:
        primal = np.clip(np.asarray(raw_primal, dtype=float), 0.0, 1.0)
        near_zero = np.abs(primal) <= 10 * self.tolerance
        near_one = np.abs(primal - 1.0) <= 10 * self.tolerance
        primal[near_zero] = 0.0
        primal[near_one] = 1.0
        return primal

    def _solve_stage_trajectory_ipm(
        self,
        lp: LinearProgramStage,
    ) -> TeacherStageResult:
        variable_count = len(lp.variable_ids)
        if variable_count == 0:
            return self._trivial_result(lp)

        objective_scale = max(1.0, float(np.max(np.abs(lp.objective))))
        solver_objective = lp.objective / objective_scale
        solver_a_ub, solver_b_ub = _scale_rows(lp.a_ub, lp.b_ub)
        solver_a_eq, solver_b_eq = _scale_rows(lp.a_eq, lp.b_eq)
        sparse_a_ub = solver_a_ub if len(solver_b_ub) else None
        sparse_a_eq = solver_a_eq if len(solver_b_eq) else None
        trajectory: list[np.ndarray] = []

        def callback(result) -> None:
            trajectory.append(np.asarray(result.x, dtype=float).copy())

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*method='interior-point'.*deprecated.*",
                    category=DeprecationWarning,
                )
                result = linprog(
                    solver_objective,
                    A_ub=sparse_a_ub,
                    b_ub=solver_b_ub if len(solver_b_ub) else None,
                    A_eq=sparse_a_eq,
                    b_eq=solver_b_eq if len(solver_b_eq) else None,
                    bounds=[(0.0, 1.0)] * variable_count,
                    method="interior-point",
                    callback=callback,
                    options={
                        "presolve": False,
                        "tol": self.tolerance,
                        "maxiter": self.max_iterations,
                        "sparse": True,
                        "cholesky": False,
                        "sym_pos": True,
                        "lstsq": False,
                        "permc_spec": "MMD_AT_PLUS_A",
                    },
                )
        except (TypeError, ValueError, NotImplementedError) as exc:
            raise TeacherSolveError(
                "SciPy's trajectory-producing interior-point solver is unavailable"
            ) from exc
        if not result.success:
            raise TeacherSolveError(
                f"{lp.name} failed: status={result.status}, {result.message}"
            )

        primal = self._normalize_primal(result.x)
        if not trajectory or not np.allclose(
            trajectory[-1], primal, atol=10 * self.tolerance, rtol=0.0
        ):
            trajectory.append(primal.copy())
        else:
            trajectory[-1] = primal.copy()
        primal_trajectory = np.vstack(trajectory)
        objective_trajectory = primal_trajectory @ lp.objective
        violation_trajectory = np.asarray(
            [_max_violation(lp, item) for item in primal_trajectory],
            dtype=float,
        )
        return TeacherStageResult(
            stage_name=lp.name,
            solver_backend="trajectory_ipm",
            trajectory_complete=True,
            success=True,
            message=str(result.message),
            iterations=int(result.nit),
            primal=primal,
            primal_trajectory=primal_trajectory,
            objective_value=float(lp.objective @ primal),
            objective_trajectory=objective_trajectory,
            max_violation_trajectory=violation_trajectory,
        )

    def _solve_stage_highs_ipm(
        self,
        lp: LinearProgramStage,
    ) -> TeacherStageResult:
        """Solve a large sparse LP with HiGHS-IPM and retain its optimum.

        SciPy does not expose HiGHS intermediate primal iterates.  This backend
        therefore returns a one-row final trajectory and marks it incomplete.
        It is intended for large benchmark execution, while ``trajectory_ipm``
        remains the data-generation backend for iteration-supervised GNNs.
        """

        variable_count = len(lp.variable_ids)
        if variable_count == 0:
            return self._trivial_result(lp)
        objective_scale = max(1.0, float(np.max(np.abs(lp.objective))))
        solver_objective = lp.objective / objective_scale
        solver_a_ub, solver_b_ub = _scale_rows(lp.a_ub, lp.b_ub)
        solver_a_eq, solver_b_eq = _scale_rows(lp.a_eq, lp.b_eq)
        try:
            result = linprog(
                solver_objective,
                A_ub=solver_a_ub if len(solver_b_ub) else None,
                b_ub=solver_b_ub if len(solver_b_ub) else None,
                A_eq=solver_a_eq if len(solver_b_eq) else None,
                b_eq=solver_b_eq if len(solver_b_eq) else None,
                bounds=(0.0, 1.0),
                method="highs-ipm",
                options={"presolve": True},
            )
        except (TypeError, ValueError, NotImplementedError) as exc:
            raise TeacherSolveError("HiGHS-IPM is unavailable") from exc
        if not result.success or result.x is None:
            raise TeacherSolveError(
                f"{lp.name} failed: status={result.status}, {result.message}"
            )
        primal = self._normalize_primal(result.x)
        primal_trajectory = primal.reshape(1, -1)
        return TeacherStageResult(
            stage_name=lp.name,
            solver_backend="highs_ipm",
            trajectory_complete=False,
            success=True,
            message=str(result.message),
            iterations=int(result.nit),
            primal=primal,
            primal_trajectory=primal_trajectory,
            objective_value=float(lp.objective @ primal),
            objective_trajectory=np.asarray(
                [float(lp.objective @ primal)], dtype=float
            ),
            max_violation_trajectory=np.asarray(
                [_max_violation(lp, primal)], dtype=float
            ),
        )

    def _solve_stage(self, lp: LinearProgramStage) -> TeacherStageResult:
        if self.solver_backend == "trajectory_ipm":
            return self._solve_stage_trajectory_ipm(lp)
        return self._solve_stage_highs_ipm(lp)

    def solve(
        self,
        expanded: TimeExpansionResult | Sequence[TimeExpandedCandidate],
        resource_capacities: Mapping[str, int],
        *,
        reserved_usage: Mapping[tuple[str, int], int] | None = None,
    ) -> TeacherSolution:
        """Solve both lexicographic stages and return GNN-ready trajectories."""

        raw_variables = expanded.variables if isinstance(expanded, TimeExpansionResult) else expanded
        variables = tuple(sorted(raw_variables, key=lambda item: item.variable_id))
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
        completed_mass = float(success_probabilities @ stage_one.primal)
        if (
            len(success_probabilities)
            and np.all(success_probabilities == 1.0)
        ):
            nearest_integer = round(completed_mass)
            if abs(completed_mass - nearest_integer) <= 10 * self.tolerance:
                completed_mass = float(nearest_integer)

        stage_two_lp = build_stage_two_lp(
            variables,
            resource_capacities,
            completed_mass,
            reserved_usage,
        )
        if completed_mass <= 10 * self.tolerance:
            stage_two = self._trivial_result(stage_two_lp)
        else:
            try:
                stage_two = self._solve_stage(stage_two_lp)
            except TeacherSolveError as exc:
                warnings.warn(
                    f"stage-two degrade: {exc}; reusing stage-one primal "
                    "(throughput preserved, latency unoptimized)",
                    RuntimeWarning,
                    stacklevel=2,
                )
                stage_two = self._trivial_result(
                    stage_two_lp,
                    np.clip(stage_one.primal, 0.0, 1.0),
                )
        return TeacherSolution(
            variables=variables,
            stage_one_lp=stage_one_lp,
            stage_two_lp=stage_two_lp,
            stage_one=stage_one,
            stage_two=stage_two,
            stage_one_completed_mass=completed_mass,
            completed_request_mass=completed_mass,
            total_completion_latency=float(
                stage_two_lp.objective @ stage_two.primal
            ),
        )


def save_teacher_solution(
    solution: TeacherSolution,
    path: str | Path,
    *,
    context: Mapping[str, object] | None = None,
) -> Path:
    """Save one complete LP graph/trajectory training record as compressed NPZ.

    ``context`` is optional dataset provenance such as topology, requests, and
    the capacity catalogue.  It does not participate in optimization.
    """

    target = Path(path)
    if target.suffix.lower() != ".npz":
        raise ValueError("teacher trace path must end with .npz")
    target.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "matrix_storage": "csr_v1",
        "variables": [
            {
                "variable_id": item.variable_id,
                "candidate_id": item.candidate_id,
                "request_id": item.request_id,
                "route_nodes": list(item.route_nodes),
                "construction_kind": item.construction_kind,
                "purification_kind": item.purification_kind,
                "start_slot": item.start_slot,
                "completion_slot": item.completion_slot,
                "completion_latency": item.completion_latency,
                "expected_fidelity": item.expected_fidelity,
                "expected_success_probability": (
                    item.expected_success_probability
                ),
                "resource_usage": [asdict(usage) for usage in item.resource_usage],
            }
            for item in solution.variables
        ],
        "stage_one_constraints": [
            asdict(item) for item in solution.stage_one_lp.ub_constraints
        ],
        "stage_two_equalities": [
            asdict(item) for item in solution.stage_two_lp.eq_constraints
        ],
        "stage_one_completed_mass": solution.stage_one_completed_mass,
        "completed_request_mass": solution.completed_request_mass,
        "total_completion_latency": solution.total_completion_latency,
        "stage_one_solver_backend": solution.stage_one.solver_backend,
        "stage_one_trajectory_complete": solution.stage_one.trajectory_complete,
        "stage_two_solver_backend": solution.stage_two.solver_backend,
        "stage_two_trajectory_complete": solution.stage_two.trajectory_complete,
    }
    if context is not None:
        metadata["context"] = dict(context)
    stage_one_a_ub = csr_matrix(solution.stage_one_lp.a_ub)
    stage_two_a_ub = csr_matrix(solution.stage_two_lp.a_ub)
    stage_two_a_eq = csr_matrix(solution.stage_two_lp.a_eq)
    np.savez_compressed(
        target,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        stage_one_objective=solution.stage_one_lp.objective,
        stage_one_a_ub_data=stage_one_a_ub.data,
        stage_one_a_ub_indices=stage_one_a_ub.indices,
        stage_one_a_ub_indptr=stage_one_a_ub.indptr,
        stage_one_a_ub_shape=np.asarray(stage_one_a_ub.shape, dtype=np.int64),
        stage_one_b_ub=solution.stage_one_lp.b_ub,
        stage_one_trajectory=solution.stage_one.primal_trajectory,
        stage_one_objective_trajectory=solution.stage_one.objective_trajectory,
        stage_one_violation_trajectory=solution.stage_one.max_violation_trajectory,
        stage_two_objective=solution.stage_two_lp.objective,
        stage_two_a_ub_data=stage_two_a_ub.data,
        stage_two_a_ub_indices=stage_two_a_ub.indices,
        stage_two_a_ub_indptr=stage_two_a_ub.indptr,
        stage_two_a_ub_shape=np.asarray(stage_two_a_ub.shape, dtype=np.int64),
        stage_two_b_ub=solution.stage_two_lp.b_ub,
        stage_two_a_eq_data=stage_two_a_eq.data,
        stage_two_a_eq_indices=stage_two_a_eq.indices,
        stage_two_a_eq_indptr=stage_two_a_eq.indptr,
        stage_two_a_eq_shape=np.asarray(stage_two_a_eq.shape, dtype=np.int64),
        stage_two_b_eq=solution.stage_two_lp.b_eq,
        stage_two_trajectory=solution.stage_two.primal_trajectory,
        stage_two_objective_trajectory=solution.stage_two.objective_trajectory,
        stage_two_violation_trajectory=solution.stage_two.max_violation_trajectory,
    )
    return target
