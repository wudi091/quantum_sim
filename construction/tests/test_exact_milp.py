from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import OptimizeResult

from construction.exact_milp import solve_exact_milp


def _successful_result(
    *,
    mip_gap: float = 0.0,
    primal: float = -1.0,
    dual: float = -1.0,
) -> OptimizeResult:
    return OptimizeResult(
        success=True,
        message="optimal",
        x=np.asarray([1.0]),
        fun=primal,
        mip_gap=mip_gap,
        mip_dual_bound=dual,
    )


class ExactMilpTests(unittest.TestCase):
    def test_forces_zero_relative_gap_option(self) -> None:
        with patch(
            "construction.exact_milp.milp",
            return_value=_successful_result(),
        ) as scipy_milp:
            solve_exact_milp(
                np.asarray([-1.0]),
                options={"disp": False, "mip_rel_gap": 0.25},
            )

        self.assertEqual(
            scipy_milp.call_args.kwargs["options"],
            {"disp": False, "mip_rel_gap": 0.0},
        )

    def test_rejects_success_with_nonzero_mip_gap(self) -> None:
        with patch(
            "construction.exact_milp.milp",
            return_value=_successful_result(mip_gap=1e-6),
        ):
            with self.assertRaisesRegex(RuntimeError, "mip_gap"):
                solve_exact_milp(np.asarray([-1.0]))

    def test_rejects_success_with_open_primal_dual_bounds(self) -> None:
        with patch(
            "construction.exact_milp.milp",
            return_value=_successful_result(primal=-1.0, dual=-1.5),
        ):
            with self.assertRaisesRegex(RuntimeError, "primal/dual"):
                solve_exact_milp(np.asarray([-1.0]))


if __name__ == "__main__":
    unittest.main()
