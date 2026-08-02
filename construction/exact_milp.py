"""Shared SciPy/HiGHS wrapper for certified construction MILP solves."""

from __future__ import annotations

from math import isclose, isfinite
from typing import Any, Mapping

from scipy.optimize import OptimizeResult, milp


_BOUND_ABS_TOLERANCE = 1e-9


def solve_exact_milp(
    c: Any,
    *,
    integrality: Any = None,
    bounds: Any = None,
    constraints: Any = None,
    options: Mapping[str, Any] | None = None,
) -> OptimizeResult:
    """Solve a MILP and require HiGHS' available optimality certificate.

    SciPy otherwise defaults to a positive relative MIP gap.  A successful
    solve at that tolerance is useful for heuristics, but it is not an exact
    oracle.  Force a zero gap and reject a nominally successful result when
    HiGHS reports either a nonzero gap or non-closing primal/dual bounds.
    """

    exact_options = dict(options or {})
    exact_options["mip_rel_gap"] = 0.0
    result = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options=exact_options,
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"HiGHS MILP failed: {result.message}")

    mip_gap = getattr(result, "mip_gap", None)
    if mip_gap is not None and float(mip_gap) != 0.0:
        raise RuntimeError(
            "HiGHS MILP did not provide an exact certificate: "
            f"mip_gap={mip_gap!r}"
        )

    primal_bound = getattr(result, "fun", None)
    dual_bound = getattr(result, "mip_dual_bound", None)
    if primal_bound is not None and dual_bound is not None:
        primal = float(primal_bound)
        dual = float(dual_bound)
        if (
            not isfinite(primal)
            or not isfinite(dual)
            or not isclose(
                primal,
                dual,
                rel_tol=0.0,
                abs_tol=_BOUND_ABS_TOLERANCE,
            )
        ):
            raise RuntimeError(
                "HiGHS MILP did not close its primal/dual bounds: "
                f"primal={primal_bound!r}, dual={dual_bound!r}"
            )

    return result
