"""One-slot MILP planners for complete path/order candidates.

The controller observes one immutable batch snapshot, solves a time-indexed
0-1 MILP, and returns only complete candidate IDs to the same event-driven
executor used by every other planner.  The time-indexed profile model is a
scheduling relaxation of the fixed executor.  This first implementation is
deliberately strict: it accepts only the deterministic single-hotspot motif
for which the relaxation is an upper bound, then replays the selected batch in
:func:`simulate_order_batch`.  When execution attains the MILP upper bound,
the returned plan IDs are certified optimal for that snapshot and catalogue.

If the MILP objective and executor completion count disagree, planning fails
instead of reporting a selected-plan count as an optimum.  MILP start windows
are internal certificate variables; they are not controller actions and are
not sent to the executor.

The general Waxman planners at the end use a separate, non-clairvoyant
definition: HiGHS proves a static resource upper bound, deterministic CP-SAT
enumerates complete assignments by cardinality, and fixed planner-owned
physical scenarios are evaluated by the shared executor.  They are exact only
for that finite scenario catalogue, not for the environment's hidden physical
realization.

The scalable formal CON oracle is the reliable-memory CP-SAT model.  It turns
link probabilities into marginal per-link reliable EPR arrivals, allocates
inventory once, and enforces timed memory release, link buffers, and BSM
capacity.  Its optimum is exact for that deterministic abstraction and is
reported separately from stochastic executor completion.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
from collections import defaultdict
from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_CEILING
from math import ceil, gcd, log
from typing import Iterable, Sequence

import numpy as np
from ortools.sat.python import cp_model
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from scipy.stats import binom

from construction.batch_plan_milp import (
    BatchMilpSolution,
    TimedPlanCandidate,
    solve_time_indexed_batch_milp,
)

from .order_core import (
    Edge,
    Node,
    OrderBatchProblem,
    OrderBatchSnapshot,
    OrderPlan,
    simulate_order_batch,
)


class UnsupportedOrderMilpProblem(ValueError):
    """Raised when the certified hotspot-MILP assumptions do not hold."""


class OrderMilpEnvironmentMismatch(RuntimeError):
    """Raised when the MILP prediction disagrees with the shared executor."""


@dataclass(frozen=True)
class OrderMilpPlanResult:
    selected_plan_ids: tuple[str, ...]
    completed_count: int
    scheduled_start_windows: dict[str, int]
    raw_solution: BatchMilpSolution
    certified_optimal: bool = True


def _validate_single_hotspot_motif(problem: OrderBatchProblem) -> None:
    """Restrict the profile relaxation to its proven mechanism-test motif."""

    if problem.required_requests != problem.preloaded_requests:
        raise UnsupportedOrderMilpProblem(
            "certified hotspot MILP requires required == preloaded requests"
        )
    if len(problem.preloaded_requests) != 1:
        raise UnsupportedOrderMilpProblem(
            "certified hotspot MILP requires exactly one preloaded main request"
        )
    if problem.config.edge_capacity != 1 or any(
        link.capacity != 1 for link in problem.links
    ):
        raise UnsupportedOrderMilpProblem(
            "certified hotspot MILP currently requires unit edge buffers"
        )
    if problem.config.bsm_capacity_per_node != 1:
        raise UnsupportedOrderMilpProblem(
            "certified hotspot MILP currently requires one BSM per node"
        )

    by_request: dict[str, list[OrderPlan]] = {}
    for plan in problem.candidates:
        by_request.setdefault(plan.request_id, []).append(plan)

    main_request = next(iter(problem.preloaded_requests))
    main_plans = by_request[main_request]
    main_paths = {plan.path for plan in main_plans}
    if len(main_paths) != 1:
        raise UnsupportedOrderMilpProblem(
            "all main-request order candidates must use one common path"
        )
    main_path = next(iter(main_paths))
    if len(main_path) < 4:
        raise UnsupportedOrderMilpProblem(
            "the certified hotspot motif requires a multi-swap main path"
        )
    main_priorities = {plan.priority for plan in main_plans}
    if len(main_priorities) != 1:
        raise UnsupportedOrderMilpProblem(
            "all main-request candidates must have the same priority"
        )
    main_priority = next(iter(main_priorities))

    waiting_plans: list[OrderPlan] = []
    for request_id, plans in by_request.items():
        if request_id == main_request:
            continue
        if len(plans) != 1:
            raise UnsupportedOrderMilpProblem(
                "each waiting request must have exactly one two-hop candidate"
            )
        plan = plans[0]
        if len(plan.path) != 3 or len(plan.swap_order) != 1:
            raise UnsupportedOrderMilpProblem(
                "each waiting request must contain exactly one swap"
            )
        if plan.priority <= main_priority:
            raise UnsupportedOrderMilpProblem(
                "the preloaded main request must precede every waiting request"
            )
        waiting_plans.append(plan)

    if not waiting_plans:
        raise UnsupportedOrderMilpProblem(
            "the certified hotspot motif requires at least one waiting request"
        )
    hotspots = {plan.path[1] for plan in waiting_plans}
    if len(hotspots) != 1:
        raise UnsupportedOrderMilpProblem(
            "all waiting requests must share one hotspot middle node"
        )
    hotspot = next(iter(hotspots))
    if hotspot not in main_path[1:-1]:
        raise UnsupportedOrderMilpProblem(
            "the waiting-request hotspot must be internal to the main path"
        )
    if problem.capacity[hotspot] % 2:
        raise UnsupportedOrderMilpProblem(
            "hotspot memory must be even to exclude partial one-edge admission"
        )

    main_nodes = set(main_path)
    for plan in waiting_plans:
        if set(plan.path) & main_nodes != {hotspot}:
            raise UnsupportedOrderMilpProblem(
                "waiting paths may intersect the main path only at the hotspot"
            )
    for index, left in enumerate(waiting_plans):
        for right in waiting_plans[index + 1:]:
            if set(left.path) & set(right.path) != {hotspot}:
                raise UnsupportedOrderMilpProblem(
                    "waiting paths may intersect each other only at the hotspot"
                )


def _validate_problem(problem: OrderBatchProblem) -> int:
    config = problem.config
    if config.swap_probability != 1.0:
        raise UnsupportedOrderMilpProblem(
            "deterministic MILP requires swap_probability == 1"
        )
    if any(
        link.generation_probability != 1.0
        for link in problem.link_by_edge.values()
    ):
        raise UnsupportedOrderMilpProblem(
            "deterministic MILP requires every link probability == 1"
        )
    if problem.initial_inventory:
        raise UnsupportedOrderMilpProblem(
            "the certified hotspot MILP does not yet support initial inventory"
        )
    if config.slot_duration_ps % config.generation_interval_ps:
        raise UnsupportedOrderMilpProblem(
            "slot duration must be an integer number of generation windows"
        )
    if config.swap_service_ps != config.generation_interval_ps:
        raise UnsupportedOrderMilpProblem(
            "certified hotspot MILP requires swap service == generation interval"
        )
    if config.memory_reset_ps > config.generation_interval_ps:
        raise UnsupportedOrderMilpProblem(
            "certified hotspot MILP requires reset <= generation interval"
        )

    # A shifted isolated trace is an upper-bound relaxation for the certified
    # hotspot motif when a non-preloaded request has at most one swap.  Longer
    # waiting paths need an event-level formulation rather than one shifted
    # profile.
    unsupported = {
        plan.request_id
        for plan in problem.candidates
        if plan.request_id not in problem.preloaded_requests
        and len(plan.path) > 3
    }
    if unsupported:
        raise UnsupportedOrderMilpProblem(
            "non-preloaded paths with multiple swaps need the event-level MILP: "
            f"{sorted(unsupported)}"
        )

    # The current profile model has node-memory and BSM rows.  Reject shared
    # physical edges across requests until time-indexed link-buffer rows are
    # added, rather than silently claiming an inexact optimum.
    edge_requests: dict[tuple[object, object], set[str]] = {}
    for plan in problem.candidates:
        for elementary_edge in plan.elementary_edges:
            edge_requests.setdefault(elementary_edge, set()).add(
                plan.request_id
            )
    shared = {
        elementary_edge: requests
        for elementary_edge, requests in edge_requests.items()
        if len(requests) > 1
    }
    if shared:
        raise UnsupportedOrderMilpProblem(
            "shared elementary edges need time-indexed link-buffer rows: "
            f"{sorted(map(repr, shared))}"
        )
    _validate_single_hotspot_motif(problem)
    return config.slot_duration_ps // config.generation_interval_ps


def _single_plan_problem(
    problem: OrderBatchProblem,
    plan: OrderPlan,
) -> OrderBatchProblem:
    required = (
        frozenset((plan.request_id,))
        if plan.request_id in problem.required_requests
        else frozenset()
    )
    preloaded = (
        frozenset((plan.request_id,))
        if plan.request_id in problem.preloaded_requests
        else frozenset()
    )
    return replace(
        problem,
        required_requests=required,
        preloaded_requests=preloaded,
        config=replace(problem.config, seed=0),
    )


def _window_memory_profile(
    problem: OrderBatchProblem,
    plan: OrderPlan,
    duration: int,
    traces,
) -> dict[Node, tuple[int, ...]]:
    interval = problem.config.generation_interval_ps
    profile: dict[Node, tuple[int, ...]] = {}
    for node in plan.path:
        values: list[int] = []
        for window in range(duration):
            start = window * interval
            finish = start + interval
            relevant = tuple(
                trace for trace in traces
                if start <= trace.time_ps < finish
            )
            value = max(
                (0, *(
                    occupancy[node]
                    for trace in relevant
                    for occupancy in (
                        trace.occupancy_before,
                        trace.occupancy_after_generation,
                        trace.occupancy_after_swaps,
                    )
                )),
            )
            values.append(int(value))

        # A direct request is generated and delivered within the same event;
        # the public trace is sampled after direct settlement.  Preserve its
        # instantaneous endpoint claim in the MILP window.
        if not plan.swap_order and values:
            values[0] = max(values[0], 1)
        profile[node] = tuple(values)
    return profile


def _window_bsm_profile(
    problem: OrderBatchProblem,
    plan: OrderPlan,
    duration: int,
    traces,
) -> dict[Node, tuple[int, ...]]:
    interval = problem.config.generation_interval_ps
    profile: dict[Node, tuple[int, ...]] = {}
    for node in plan.path[1:-1]:
        values = []
        for window in range(duration):
            start = window * interval
            finish = start + interval
            values.append(sum(
                event.middle == node and event.status == "success"
                for trace in traces
                if start <= trace.time_ps < finish
                for event in trace.swap_events
            ))
        profile[node] = tuple(map(int, values))
    return profile


def compile_timed_candidate(
    problem: OrderBatchProblem,
    plan: OrderPlan,
) -> TimedPlanCandidate:
    """Compile one complete path/order into a deterministic resource trace."""

    result = simulate_order_batch(
        _single_plan_problem(problem, plan),
        (plan.plan_id,),
        record_traces=True,
    )
    completion = result.completion_time_ps.get(plan.request_id)
    if completion is None:
        raise UnsupportedOrderMilpProblem(
            f"candidate does not complete in isolation: {plan.plan_id}"
        )
    interval = problem.config.generation_interval_ps
    duration = max(1, ceil(completion / interval))
    return TimedPlanCandidate(
        request_id=plan.request_id,
        candidate_id=plan.plan_id,
        path=tuple(plan.path),
        swap_order=tuple(plan.swap_order),
        duration=duration,
        memory_profile=_window_memory_profile(
            problem, plan, duration, result.traces
        ),
        bsm_profile=_window_bsm_profile(
            problem, plan, duration, result.traces
        ),
        allowed_starts=(0,)
        if plan.request_id in problem.preloaded_requests else None,
        reward=1,
        priority=plan.priority,
    )


def solve_order_milp(
    snapshot: OrderBatchSnapshot,
    *,
    allow_swap_orders: bool,
) -> OrderMilpPlanResult:
    """Solve an upper-bound MILP and certify it in the shared executor."""

    problem = snapshot.problem
    horizon = _validate_problem(problem)
    eligible = tuple(
        plan for plan in snapshot.candidates
        if allow_swap_orders or plan.is_fixed_order
    )
    missing = problem.required_requests - {
        plan.request_id for plan in eligible
    }
    if missing:
        raise UnsupportedOrderMilpProblem(
            f"required request has no eligible candidate: {sorted(missing)}"
        )
    timed = tuple(
        compile_timed_candidate(problem, plan) for plan in eligible
    )
    solution = solve_time_indexed_batch_milp(
        timed,
        memory_capacity=problem.capacity,
        bsm_capacity={
            node: problem.config.bsm_capacity_per_node
            for node in problem.capacity
        },
        horizon=horizon,
        required_requests=problem.required_requests,
    )
    plan_lookup = {plan.plan_id: plan for plan in eligible}
    selected = tuple(sorted(
        (
            scheduled.candidate.candidate_id
            for scheduled in solution.selected.values()
        ),
        key=lambda plan_id: (
            plan_lookup[plan_id].priority,
            plan_lookup[plan_id].request_id,
            plan_id,
        ),
    ))
    verified = simulate_order_batch(
        problem, selected, record_traces=False
    )
    selected_requests = {
        plan_lookup[plan_id].request_id for plan_id in selected
    }
    if (
        verified.completed_count != solution.completed_requests
        or set(verified.completed) != selected_requests
    ):
        raise OrderMilpEnvironmentMismatch(
            "MILP/executor mismatch: "
            f"objective={solution.completed_requests}, "
            f"selected={sorted(selected_requests)}, "
            f"completed={sorted(verified.completed)}"
        )
    return OrderMilpPlanResult(
        selected_plan_ids=selected,
        completed_count=solution.completed_requests,
        scheduled_start_windows={
            request_id: scheduled.start
            for request_id, scheduled in solution.selected.items()
        },
        raw_solution=solution,
        certified_optimal=True,
    )


class _DeterministicOrderMilpPlanner:
    allow_swap_orders = False
    name = "milp_path"

    def __init__(self) -> None:
        self.last_objective = 0
        self.last_solution: OrderMilpPlanResult | None = None
        self.last_evaluations = 0

    def reset(self, episode_seed: int) -> None:
        del episode_seed
        self.last_objective = 0
        self.last_solution = None
        self.last_evaluations = 0

    def select(self, snapshot: OrderBatchSnapshot) -> tuple[str, ...]:
        solution = solve_order_milp(
            snapshot,
            allow_swap_orders=self.allow_swap_orders,
        )
        self.last_solution = solution
        self.last_objective = solution.completed_count
        return solution.selected_plan_ids


class DeterministicPathMilpPlanner(_DeterministicOrderMilpPlanner):
    """Certified deterministic MILP restricted to canonical fixed orders."""

    allow_swap_orders = False
    name = "milp_path"


class DeterministicPathOrderMilpPlanner(_DeterministicOrderMilpPlanner):
    """Certified deterministic MILP over complete path/order candidates."""

    allow_swap_orders = True
    name = "milp_path_order"


@dataclass(frozen=True)
class MilpNominalPlanResult:
    """Exact hybrid optimum for fixed planner-owned physical scenarios.

    This certificate is deliberately narrower than
    :class:`OrderMilpPlanResult`.  It proves optimality only for the supplied
    finite planning scenarios and chance threshold.  It says nothing about
    the hidden physical realization subsequently used by the environment.

    HiGHS solves a static 0-1 resource relaxation once.  OR-Tools CP-SAT then
    enumerates every assignment at each still-possible cardinality, and the
    unchanged event executor validates every enumerated complete assignment.
    No failed partial assignment is used as a cut.
    """

    selected_plan_ids: tuple[str, ...]
    completed_count: int
    scenario_completion_counts: tuple[tuple[str, int], ...]
    planning_seeds: tuple[int, ...]
    required_scenarios: int
    evaluations: int
    milp_solves: int
    cuts: int
    backend: str
    eligible_candidates: int
    filtered_candidates: int
    enumerated_assignments: int
    static_upper_bound: int
    proven_optimal: bool = True
    certified_optimal: bool = False


@dataclass(frozen=True)
class MilpStaticPlanResult:
    """Proven optimum of a static necessary-condition MILP relaxation.

    The resource rows are optimistic aggregate conditions, so the objective
    is an upper bound for a stricter deterministic event model, not a proof
    that every selected request can complete.  The environment subsequently
    executes the selected plans and records realized completions separately.
    """

    selected_plan_ids: tuple[str, ...]
    completed_count: int
    planning_seeds: tuple[int, ...]
    required_scenarios: int
    backend: str
    eligible_candidates: int
    filtered_candidates: int
    resource_constraint_count: int
    secondary_objective: int
    proven_optimal: bool = True
    certified_optimal: bool = False


def _require_proven_mip_optimum(result) -> None:
    """Reject a tolerance-optimal master before claiming exactness.

    SciPy/HiGHS normally reports both the relative MIP gap and the dual
    objective bound.  They are checked when present; older SciPy versions
    that omit either field still have to return the solver's optimal status,
    with ``mip_rel_gap=0`` requested by the caller.
    """

    gap = getattr(result, "mip_gap", None)
    if gap is not None:
        gap_value = float(gap)
        if not np.isfinite(gap_value) or abs(gap_value) > 1e-10:
            raise RuntimeError(
                "HiGHS nominal static master is not proven optimal: "
                f"mip_gap={gap_value}"
            )

    primal = getattr(result, "fun", None)
    dual = getattr(result, "mip_dual_bound", None)
    if primal is not None and dual is not None:
        primal_value = float(primal)
        dual_value = float(dual)
        if (
            not np.isfinite(primal_value)
            or not np.isfinite(dual_value)
            or not np.isclose(
                primal_value, dual_value, rtol=0.0, atol=1e-8
            )
        ):
            raise RuntimeError(
                "HiGHS nominal static master lacks a closed objective bound: "
                f"primal={primal_value}, dual={dual_value}"
            )


def _nominal_candidate_key(plan: OrderPlan) -> tuple[object, ...]:
    return (
        plan.priority,
        plan.request_id,
        len(plan.path),
        tuple(map(repr, plan.path)),
        plan.schedule_key,
        plan.plan_id,
    )


_NOMINAL_BACKEND = "highs-static-master+ortools-cp-sat-enumeration"
_DEFAULT_NOMINAL_ORACLE_BATCH_SIZE = 128


# A spawned worker receives the immutable snapshot and planning scenarios only
# once.  Per-task payloads then contain plan IDs only.  Every simulator call
# still constructs its own _OrderExecution, so these process globals contain
# no mutable physical state shared between assignments.
_NOMINAL_WORKER_PROBLEMS: tuple[OrderBatchProblem, ...] = ()
_NOMINAL_WORKER_PLAN_TO_REQUEST: dict[str, str] = {}
_NOMINAL_WORKER_REQUIRED_SCENARIOS = 1


@dataclass(frozen=True)
class _NominalOracleResult:
    completion_counts: tuple[tuple[str, int], ...]
    feasible: bool


def _init_nominal_oracle_worker(
    problem: OrderBatchProblem,
    planning_seeds: tuple[int, ...],
    required_scenarios: int,
) -> None:
    """Initialize one spawned executor worker for a single snapshot."""

    global _NOMINAL_WORKER_PROBLEMS
    global _NOMINAL_WORKER_PLAN_TO_REQUEST
    global _NOMINAL_WORKER_REQUIRED_SCENARIOS
    _NOMINAL_WORKER_PROBLEMS = tuple(
        problem.with_physics_seed(seed) for seed in planning_seeds
    )
    _NOMINAL_WORKER_PLAN_TO_REQUEST = {
        plan.plan_id: plan.request_id for plan in problem.candidates
    }
    _NOMINAL_WORKER_REQUIRED_SCENARIOS = int(required_scenarios)


def _evaluate_nominal_oracle_worker_batch(
    assignments: tuple[tuple[str, ...], ...],
) -> tuple[_NominalOracleResult, ...]:
    """Evaluate an ordered plan-ID sub-batch in one spawned process."""

    results: list[_NominalOracleResult] = []
    for plan_ids in assignments:
        request_ids = tuple(
            _NOMINAL_WORKER_PLAN_TO_REQUEST[plan_id]
            for plan_id in plan_ids
        )
        counts = {request_id: 0 for request_id in request_ids}
        if plan_ids:
            for problem in _NOMINAL_WORKER_PROBLEMS:
                result = simulate_order_batch(
                    problem, plan_ids, record_traces=False
                )
                completed = set(result.completed)
                for request_id in request_ids:
                    counts[request_id] += request_id in completed
        ordered_counts = tuple(sorted(counts.items()))
        results.append(_NominalOracleResult(
            completion_counts=ordered_counts,
            feasible=all(
                count >= _NOMINAL_WORKER_REQUIRED_SCENARIOS
                for _, count in ordered_counts
            ),
        ))
    return tuple(results)


def _nominal_spawn_context():
    """Return an explicit spawn context on every platform.

    Windows always uses spawn; selecting it explicitly on other platforms
    keeps tests and production behavior aligned with the target platform.
    """

    return mp.get_context("spawn")


class _NominalOracleBatchEvaluator:
    """Ordered batched facade over the unchanged physical executor."""

    def __init__(
        self,
        planner,
        problem: OrderBatchProblem,
    ) -> None:
        self.planner = planner
        self.problem = problem
        self.pool = None

    def __enter__(self) -> "_NominalOracleBatchEvaluator":
        return self

    def _start_pool(self) -> None:
        if self.pool is not None:
            return
        self.pool = _nominal_spawn_context().Pool(
            processes=self.planner.oracle_workers,
            initializer=_init_nominal_oracle_worker,
            initargs=(
                self.problem,
                self.planner.planning_seeds,
                self.planner.required_scenarios,
            ),
        )

    def evaluate(
        self,
        assignments: Sequence[tuple[str, ...]],
        *,
        allow_pool_start: bool,
    ) -> tuple[_NominalOracleResult, ...]:
        """Evaluate one ordered callback batch and preserve its order."""

        ordered_assignments = tuple(assignments)
        if not ordered_assignments:
            return ()
        should_start = (
            self.planner.oracle_workers > 1
            and allow_pool_start
            and len(ordered_assignments) >= 2
        )
        if self.pool is None and should_start:
            self._start_pool()
        if self.pool is None:
            serial: list[_NominalOracleResult] = []
            for plan_ids in ordered_assignments:
                counts = self.planner._completion_counts(
                    self.problem, plan_ids
                )
                ordered_counts = tuple(sorted(counts.items()))
                serial.append(_NominalOracleResult(
                    completion_counts=ordered_counts,
                    feasible=all(
                        count >= self.planner.required_scenarios
                        for _, count in ordered_counts
                    ),
                ))
            return tuple(serial)

        # Submit several ordered sub-batches per process.  Pool.map preserves
        # their order, and flattening therefore reproduces the exact CP-SAT
        # callback order independently of worker completion timing.
        target_sub_batches = max(1, self.planner.oracle_workers * 4)
        sub_batch_size = max(
            1, ceil(len(ordered_assignments) / target_sub_batches)
        )
        sub_batches = tuple(
            ordered_assignments[start : start + sub_batch_size]
            for start in range(0, len(ordered_assignments), sub_batch_size)
        )
        nested = self.pool.map(
            _evaluate_nominal_oracle_worker_batch,
            sub_batches,
            chunksize=1,
        )
        flattened = tuple(
            result for sub_batch in nested for result in sub_batch
        )
        if len(flattened) != len(ordered_assignments):
            raise RuntimeError(
                "parallel nominal oracle returned the wrong result count: "
                f"{len(flattened)} != {len(ordered_assignments)}"
            )
        self.planner.last_evaluations += (
            sum(bool(plan_ids) for plan_ids in ordered_assignments)
            * len(self.planner.planning_seeds)
        )
        return flattened

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        del exc_value, traceback
        if self.pool is None:
            return False
        pool = self.pool
        self.pool = None
        if exc_type is not None:
            pool.terminate()
            pool.join()
            return False
        try:
            pool.close()
            pool.join()
        except BaseException:
            pool.terminate()
            pool.join()
            raise
        return False


def _planner_uniform(seed: int, *parts: object) -> float:
    """Reproduce the executor's request-local deterministic physical draw."""

    payload = "|".join(map(str, (int(seed), *parts))).encode()
    digest = hashlib.sha256(payload).digest()[:8]
    return int.from_bytes(digest, "big") / 2**64


def _edge_generation_deadline_ps(
    problem: OrderBatchProblem,
    plan: OrderPlan,
    elementary_edge,
) -> int:
    """Latest physical time at which the selected order can acquire an edge.

    A path need not have every elementary pair before its first swap.  An edge
    is consumed by the earlier swap, in the selected order, at either of its
    internal endpoints.  If that swap has rank ``j`` among ``s`` sequential
    swaps, it must start no later than ``T - (s-j) * service``.  Competition
    can delay a request-local attempt index but can never make it occur before
    the corresponding no-contention generation epoch.
    """

    config = problem.config
    if not plan.swap_order:
        # Generation events occur strictly before the slot boundary, whereas a
        # direct request settles in the same event in which its pair appears.
        return config.slot_duration_ps - 1

    rank = plan.swap_round_by_node
    adjacent_internal = tuple(
        node for node in elementary_edge if node in rank
    )
    if not adjacent_internal:
        raise RuntimeError(
            "a non-direct elementary edge has no internal endpoint"
        )
    consuming_rank = min(rank[node] for node in adjacent_internal)
    latest_start = (
        config.slot_duration_ps
        - (plan.swap_round_count - consuming_rank)
        * config.swap_service_ps
    )
    return latest_start


def _edge_generation_attempt_limit(
    problem: OrderBatchProblem,
    plan: OrderPlan,
    elementary_edge,
) -> int:
    """Optimistic number of attempts available before this edge is needed."""

    latest_start = _edge_generation_deadline_ps(
        problem, plan, elementary_edge
    )
    if latest_start < 0:
        return 0
    generation_times = range(
        0,
        problem.config.slot_duration_ps,
        problem.config.generation_interval_ps,
    )
    return sum(time_ps <= latest_start for time_ps in generation_times)


def _plan_hard_possible_in_scenario(
    problem: OrderBatchProblem,
    plan: OrderPlan,
    planning_seed: int,
) -> bool:
    """Return a conservative upper test for one plan/scenario.

    ``False`` is a proof of impossibility, not an isolation heuristic.  Every
    completed request must execute all of its swaps, whose draws are fixed by
    request and middle node.  For an edge not already available in inventory,
    it must also consume a successful request-local generation draw before the
    first selected-order swap that needs the edge.  Other requests can block
    and therefore delay those attempt indices, but cannot skip a failed draw
    or make a later index happen earlier.

    Inventory is treated optimistically: the existence of one matching pair
    is enough for this per-plan test.  Contention for that pair is left to the
    exact executor, so this filter cannot reject a feasible shared batch.
    """

    config = problem.config
    if (
        plan.swap_round_count * config.swap_service_ps
        > config.slot_duration_ps
    ):
        return False
    if any(
        _planner_uniform(
            planning_seed,
            "swap",
            config.slot_id,
            plan.request_id,
            middle,
        ) > config.swap_probability
        for middle in plan.swap_order
    ):
        return False

    # The executor materializes every elementary pair of a preloaded request
    # at t=0 before inventory assignment or physical generation begins.  Such
    # a request is still subject to swap draws and the slot horizon, but its
    # own link-generation draws are irrelevant.
    if plan.request_id in problem.preloaded_requests:
        return True

    inventory_edges = {
        stored.elementary_edge for stored in problem.initial_inventory
    }
    for elementary_edge in plan.elementary_edges:
        if elementary_edge in inventory_edges:
            continue
        attempts = _edge_generation_attempt_limit(
            problem, plan, elementary_edge
        )
        probability = problem.link_generation_probability(elementary_edge)
        if not any(
            _planner_uniform(
                planning_seed,
                "generation",
                config.slot_id,
                plan.request_id,
                elementary_edge,
                attempt_index,
            ) <= probability
            for attempt_index in range(attempts)
        ):
            return False
    return True


def _static_resource_rows(
    problem: OrderBatchProblem,
    plans: tuple[OrderPlan, ...],
    *,
    scenario_count: int,
    required_scenarios: int,
) -> tuple[tuple[dict[int, int], int], ...]:
    """Build safe aggregate resource upper bounds for the hybrid master."""

    config = problem.config
    inventory_by_edge: dict[object, int] = defaultdict(int)
    inventory_by_node: dict[object, int] = defaultdict(int)
    for stored in problem.initial_inventory:
        inventory_by_edge[stored.elementary_edge] += 1
        inventory_by_node[stored.left] += 1
        inventory_by_node[stored.right] += 1

    rows: list[tuple[dict[int, int], int]] = []

    generation_times = tuple(range(
        0, config.slot_duration_ps, config.generation_interval_ps
    ))

    # Every completion executes each selected internal-node BSM once.  A swap
    # at rank j has the optimistic job window
    # [j*service, T-(s-j-1)*service].  Standard interval-energy rows are safe:
    # every job whose entire window lies inside [a,b] must consume its service
    # there.  The full-slot row is included as one such interval.
    for node in problem.capacity:
        jobs: list[tuple[int, int, int]] = []
        for index, plan in enumerate(plans):
            if node not in plan.swap_order:
                continue
            swap_rank = plan.swap_round_by_node[node]
            release = swap_rank * config.swap_service_ps
            deadline = (
                config.slot_duration_ps
                - (plan.swap_round_count - swap_rank - 1)
                * config.swap_service_ps
            )
            jobs.append((index, release, deadline))
        if not jobs:
            continue
        boundaries = tuple(sorted({
            0,
            config.slot_duration_ps,
            *(release for _, release, _ in jobs),
            *(deadline for _, _, deadline in jobs),
        }))
        seen_bsm_rows: set[tuple[tuple[tuple[int, int], ...], int]] = set()
        for left_index, interval_start in enumerate(boundaries):
            for interval_end in boundaries[left_index + 1:]:
                coefficients = {
                    index: required_scenarios * config.swap_service_ps
                    for index, release, deadline in jobs
                    if release >= interval_start and deadline <= interval_end
                }
                if not coefficients:
                    continue
                upper = (
                    scenario_count
                    * (interval_end - interval_start)
                    * config.bsm_capacity_per_node
                )
                signature = tuple(sorted(coefficients.items())), upper
                if signature in seen_bsm_rows:
                    continue
                seen_bsm_rows.add(signature)
                rows.append((coefficients, upper))

    # Cumulative edge supply: by a selected-order consumption deadline d, an
    # elementary edge can have supplied at most its initial pairs plus one full
    # edge buffer at each generation epoch no later than d.
    for elementary_edge, link in problem.link_by_edge.items():
        uses = tuple(
            (
                index,
                _edge_generation_deadline_ps(
                    problem, plan, elementary_edge
                ),
            )
            for index, plan in enumerate(plans)
            if elementary_edge in plan.elementary_edges
        )
        for deadline in sorted({deadline for _, deadline in uses}):
            coefficients = {
                index: required_scenarios
                for index, needed_by in uses
                if needed_by <= deadline
            }
            rows.append((
                coefficients,
                scenario_count * (
                    inventory_by_edge[elementary_edge]
                    + sum(
                        time_ps <= deadline
                        for time_ps in generation_times
                    ) * link.capacity
                ),
            ))

    # Cumulative node supply is analogous.  Each incident elementary edge uses
    # one memory endpoint.  Resets and long pairs only consume more memory, so
    # omitting them keeps these rows optimistic and therefore necessary only.
    for node, memory_capacity in problem.capacity.items():
        uses: list[tuple[int, int]] = []
        for index, plan in enumerate(plans):
            for elementary_edge in plan.elementary_edges:
                if node in elementary_edge:
                    uses.append((
                        index,
                        _edge_generation_deadline_ps(
                            problem, plan, elementary_edge
                        ),
                    ))
        for deadline in sorted({deadline for _, deadline in uses}):
            coefficients_by_plan: dict[int, int] = defaultdict(int)
            for index, needed_by in uses:
                if needed_by <= deadline:
                    coefficients_by_plan[index] += required_scenarios
            rows.append((
                dict(coefficients_by_plan),
                scenario_count * (
                    inventory_by_node[node]
                    + sum(
                        time_ps <= deadline
                        for time_ps in generation_times
                    ) * memory_capacity
                ),
            ))
    return tuple(rows)


def _solve_static_snapshot_milp(
    snapshot: OrderBatchSnapshot,
    *,
    allow_swap_orders: bool,
    planning_seeds: tuple[int, ...],
    required_scenarios: int,
) -> MilpStaticPlanResult:
    """Solve the static necessary-condition online relaxation directly.

    Binary candidate variables choose at most one complete path/schedule per
    request.  The model includes the snapshot's current inventory, cumulative
    link and node supply, BSM interval capacity, schedule-dependent generation
    deadlines, required carried requests, and a fixed set of planner-owned
    probability scenarios used only for hard-possibility screening.

    No physical executor call is made while optimizing.  Consequently the
    returned optimum is exact for this relaxation only.  It must not be called
    a deterministic completion optimum; realized physical completion remains
    a separate environment metric.
    """

    inactive = tuple(
        plan for plan in snapshot.candidates
        if (
            plan.deadline_slot is not None
            and plan.decision_slot >= plan.deadline_slot
        )
    )
    if inactive:
        details = ", ".join(
            f"{plan.plan_id}(decision_slot={plan.decision_slot}, "
            f"deadline_slot={plan.deadline_slot})"
            for plan in inactive
        )
        raise ValueError(
            "static MILP snapshot contains inactive candidates; active "
            "requests require decision_slot < deadline_slot: " + details
        )

    eligible = tuple(sorted((
        plan for plan in snapshot.candidates
        if allow_swap_orders or plan.is_fixed_order
    ), key=_nominal_candidate_key))
    missing = snapshot.problem.required_requests - {
        plan.request_id for plan in eligible
    }
    if missing:
        raise RuntimeError(
            "static MILP catalogue misses required requests: "
            f"{sorted(missing)}"
        )

    raw_by_request: dict[str, list[OrderPlan]] = defaultdict(list)
    for plan in eligible:
        raw_by_request[plan.request_id].append(plan)
    request_ids = tuple(sorted(
        raw_by_request,
        key=lambda request_id: (
            min(plan.priority for plan in raw_by_request[request_id]),
            request_id,
        ),
    ))
    if not request_ids:
        return MilpStaticPlanResult(
            selected_plan_ids=(),
            completed_count=0,
            planning_seeds=planning_seeds,
            required_scenarios=required_scenarios,
            backend="ortools-cp-sat-deterministic-static-snapshot-milp",
            eligible_candidates=0,
            filtered_candidates=0,
            resource_constraint_count=0,
            secondary_objective=0,
        )

    hard_possible_counts = {
        plan.plan_id: sum(
            _plan_hard_possible_in_scenario(
                snapshot.problem, plan, planning_seed
            )
            for planning_seed in planning_seeds
        )
        for plan in eligible
    }
    plans = tuple(
        plan for plan in eligible
        if hard_possible_counts[plan.plan_id] >= required_scenarios
    )
    filtered_candidates = len(eligible) - len(plans)
    plans_by_request = {
        request_id: tuple(
            plan for plan in plans if plan.request_id == request_id
        )
        for request_id in request_ids
    }
    unavailable_required = {
        request_id for request_id in snapshot.problem.required_requests
        if not plans_by_request[request_id]
    }
    if unavailable_required:
        raise RuntimeError(
            "no statically model-completable candidate remains for required "
            f"requests: {sorted(unavailable_required)}"
        )
    searchable_request_ids = tuple(
        request_id for request_id in request_ids
        if plans_by_request[request_id]
    )
    if not searchable_request_ids:
        return MilpStaticPlanResult(
            selected_plan_ids=(),
            completed_count=0,
            planning_seeds=planning_seeds,
            required_scenarios=required_scenarios,
            backend="ortools-cp-sat-deterministic-static-snapshot-milp",
            eligible_candidates=len(eligible),
            filtered_candidates=filtered_candidates,
            resource_constraint_count=0,
            secondary_objective=0,
        )

    plan_index = {
        plan.plan_id: index for index, plan in enumerate(plans)
    }
    resource_rows = _static_resource_rows(
        snapshot.problem,
        plans,
        scenario_count=len(planning_seeds),
        required_scenarios=required_scenarios,
    )

    def build_model(
        *,
        fixed_cardinality: int | None,
    ):
        model = cp_model.CpModel()
        x_vars = tuple(
            model.NewBoolVar(f"x_{index}")
            for index in range(len(plans))
        )
        y_vars = {
            request_id: model.NewBoolVar(f"y_{index}")
            for index, request_id in enumerate(searchable_request_ids)
        }
        for request_id in searchable_request_ids:
            model.Add(
                sum(
                    x_vars[plan_index[plan.plan_id]]
                    for plan in plans_by_request[request_id]
                ) == y_vars[request_id]
            )
        for request_id in snapshot.problem.required_requests:
            model.Add(y_vars[request_id] == 1)
        for coefficients, upper in resource_rows:
            model.Add(
                sum(
                    int(value) * x_vars[index]
                    for index, value in coefficients.items()
                ) <= int(upper)
            )
        cardinality = sum(y_vars.values())
        if fixed_cardinality is None:
            model.Maximize(cardinality)
        else:
            model.Add(cardinality == int(fixed_cardinality))
            model.Minimize(sum(
                (index + 1) * x_vars[index]
                for index in range(len(plans))
            ))
        model.AddDecisionStrategy(
            x_vars,
            cp_model.CHOOSE_FIRST,
            cp_model.SELECT_MAX_VALUE,
        )
        return model, x_vars, y_vars

    def solve_model(model: cp_model.CpModel):
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.search_branching = cp_model.FIXED_SEARCH
        status = solver.Solve(model)
        return solver, status

    primary_model, _, _ = build_model(fixed_cardinality=None)
    primary_solver, primary_status = solve_model(primary_model)
    if primary_status == cp_model.INFEASIBLE:
        raise RuntimeError(
            "no deterministic static assignment satisfies all required "
            "requests"
        )
    if primary_status != cp_model.OPTIMAL:
        raise RuntimeError(
            "CP-SAT deterministic static MILP did not prove the primary "
            f"optimum: status={primary_solver.StatusName()}"
        )
    completed_count = int(round(primary_solver.ObjectiveValue()))

    # A second exact solve keeps the primary cardinality fixed and chooses a
    # stable low-rank candidate set.  This affects only ties between primary-
    # optimal actions and makes repeated physical evaluation reproducible.
    secondary_model, secondary_x, secondary_y = build_model(
        fixed_cardinality=completed_count
    )
    secondary_solver, secondary_status = solve_model(secondary_model)
    if secondary_status != cp_model.OPTIMAL:
        raise RuntimeError(
            "CP-SAT deterministic static MILP did not prove the tie-break "
            f"optimum: status={secondary_solver.StatusName()}"
        )

    selected_ids: list[str] = []
    for request_id in searchable_request_ids:
        if not secondary_solver.Value(secondary_y[request_id]):
            continue
        chosen = tuple(
            plan.plan_id for plan in plans_by_request[request_id]
            if secondary_solver.Value(
                secondary_x[plan_index[plan.plan_id]]
            )
        )
        if len(chosen) != 1:
            raise RuntimeError(
                "deterministic static MILP did not select exactly one "
                f"candidate for request {request_id!r}: {chosen}"
            )
        selected_ids.append(chosen[0])
    if len(selected_ids) != completed_count:
        raise RuntimeError(
            "deterministic static MILP cardinality disagrees with selected "
            f"plans: {completed_count} != {len(selected_ids)}"
        )
    selected_requests = {
        plans[plan_index[plan_id]].request_id for plan_id in selected_ids
    }
    missing_required_after_solve = (
        snapshot.problem.required_requests - selected_requests
    )
    if missing_required_after_solve:
        raise RuntimeError(
            "deterministic static MILP omitted required requests: "
            f"{sorted(missing_required_after_solve)}"
        )

    secondary_integer = int(round(secondary_solver.ObjectiveValue()))
    return MilpStaticPlanResult(
        selected_plan_ids=tuple(selected_ids),
        completed_count=completed_count,
        planning_seeds=planning_seeds,
        required_scenarios=required_scenarios,
        backend="ortools-cp-sat-deterministic-static-snapshot-milp",
        eligible_candidates=len(eligible),
        filtered_candidates=filtered_candidates,
        resource_constraint_count=len(resource_rows),
        secondary_objective=secondary_integer,
    )


class _MilpStaticPlanner:
    """Direct static resource-relaxation MILP for one online snapshot."""

    allow_swap_orders = False
    name = "milp_static_path"

    def __init__(
        self,
        planning_seeds: Iterable[int] = (0,),
        *,
        chance_threshold: float = 1.0,
    ) -> None:
        seeds = tuple(map(int, planning_seeds))
        if not seeds:
            raise ValueError("planning_seeds cannot be empty")
        if len(set(seeds)) != len(seeds):
            raise ValueError("planning_seeds cannot contain duplicates")
        if not 0.0 < chance_threshold <= 1.0:
            raise ValueError("chance_threshold must lie in (0, 1]")
        self.planning_seeds = seeds
        self.chance_threshold = float(chance_threshold)
        self.required_scenarios = int(
            (
                Decimal(str(chance_threshold)) * len(seeds)
            ).to_integral_value(rounding=ROUND_CEILING)
        )
        self.last_objective = 0
        self.last_solution: MilpStaticPlanResult | None = None
        self.last_evaluations = 0
        self.last_proven_optimal = False

    def reset(self, episode_seed: int) -> None:
        del episode_seed
        self.last_objective = 0
        self.last_solution = None
        self.last_evaluations = 0
        self.last_proven_optimal = False

    def select(self, snapshot: OrderBatchSnapshot) -> tuple[str, ...]:
        solution = _solve_static_snapshot_milp(
            snapshot,
            allow_swap_orders=self.allow_swap_orders,
            planning_seeds=self.planning_seeds,
            required_scenarios=self.required_scenarios,
        )
        self.last_solution = solution
        self.last_objective = solution.completed_count
        self.last_evaluations = 0
        self.last_proven_optimal = True
        return solution.selected_plan_ids


class MilpStaticPathPlanner(_MilpStaticPlanner):
    """Static resource relaxation using one schedule per path."""

    allow_swap_orders = False
    name = "milp_static_path"


class MilpStaticPathOrderPlanner(_MilpStaticPlanner):
    """Static resource relaxation over complete cached schedules."""

    allow_swap_orders = True
    name = "milp_static_path_order"


def reliable_binomial_capacity(
    attempt_count: int,
    probability: float,
    confidence: float,
) -> int:
    """Return a one-sided reliable lower bound on successful attempts.

    The result is the largest integer ``k`` for which a
    ``Binomial(attempt_count, probability)`` random variable satisfies
    ``P[X >= k] >= confidence``.  It is a deterministic planning quantity,
    not a sampled physical realization.
    """

    if int(attempt_count) != attempt_count or attempt_count < 0:
        raise ValueError("attempt_count must be a non-negative integer")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    if not 0.0 < confidence <= 1.0:
        raise ValueError("confidence must lie in (0, 1]")
    attempts = int(attempt_count)
    if attempts == 0 or probability == 0.0:
        return 0
    if probability == 1.0:
        return attempts

    # The survival function is monotone decreasing in k.  Binary search keeps
    # this helper cheap even when a simulator exposes many physical attempts.
    low = 0
    high = attempts
    tolerance = 1e-12
    while low < high:
        middle = (low + high + 1) // 2
        tail = float(binom.sf(middle - 1, attempts, probability))
        if tail + tolerance >= confidence:
            low = middle
        else:
            high = middle - 1
    return low


def _reliable_time_quantum_ps(problem: OrderBatchProblem) -> int:
    """Common exact grid for generation epochs and BSM service."""

    config = problem.config
    quantum = gcd(
        int(config.generation_interval_ps),
        int(config.swap_service_ps),
    )
    if config.slot_duration_ps % quantum:
        raise ValueError(
            "reliable-memory MILP requires slot_duration_ps to be an "
            "integer number of the generation/service time quantum"
        )
    return quantum


def reliable_link_arrivals(
    problem: OrderBatchProblem,
    confidence: float,
    *,
    time_quantum_ps: int | None = None,
) -> dict[Edge, tuple[int, ...]]:
    """Compile per-link reliable EPR arrivals on the deterministic grid.

    At every physical generation epoch a link contributes ``capacity``
    Bernoulli attempts.  If the reliable lower bound after all attempts up to
    tick ``t`` rises from ``q`` to ``q+d``, this abstraction exposes ``d`` new
    EPR pairs at that tick.  Assigning those pairs to candidates is still
    constrained by link buffers and endpoint memory at every tick.

    Confidence is marginal per physical link.  The function deliberately does
    not claim a joint all-links chance guarantee.
    """

    if not 0.0 < confidence <= 1.0:
        raise ValueError("confidence must lie in (0, 1]")
    quantum = (
        _reliable_time_quantum_ps(problem)
        if time_quantum_ps is None else int(time_quantum_ps)
    )
    if quantum < 1:
        raise ValueError("time_quantum_ps must be positive")
    config = problem.config
    if (
        config.generation_interval_ps % quantum
        or config.swap_service_ps % quantum
        or config.slot_duration_ps % quantum
    ):
        raise ValueError(
            "time_quantum_ps must divide generation, service, and slot times"
        )
    horizon = config.slot_duration_ps // quantum
    total_epochs = len(range(
        0,
        config.slot_duration_ps,
        config.generation_interval_ps,
    ))
    arrivals: dict[Edge, tuple[int, ...]] = {}
    for elementary_edge, link in problem.link_by_edge.items():
        previous = 0
        values: list[int] = []
        for tick in range(horizon):
            time_ps = tick * quantum
            epochs = min(
                total_epochs,
                time_ps // config.generation_interval_ps + 1,
            )
            reliable = reliable_binomial_capacity(
                epochs * link.capacity,
                link.generation_probability,
                confidence,
            )
            values.append(reliable - previous)
            previous = reliable
        arrivals[elementary_edge] = tuple(values)
    return arrivals


@dataclass(frozen=True)
class ReliableMemoryCandidateProfile:
    """Deterministic time profile of one complete path/schedule candidate."""

    plan: OrderPlan
    time_quantum_ps: int
    completion_ticks: int
    resource_ticks: int
    memory_profile: dict[Node, tuple[int, ...]]
    edge_profile: dict[Edge, tuple[int, ...]]
    bsm_profile: dict[Node, tuple[int, ...]]


def compile_reliable_memory_candidate(
    problem: OrderBatchProblem,
    plan: OrderPlan,
    *,
    time_quantum_ps: int | None = None,
) -> ReliableMemoryCandidateProfile:
    """Compile release, link-buffer, and BSM profiles for one schedule.

    A candidate starts only after every elementary EPR assigned to it is
    present.  Its swap groups then execute consecutively.  The two memories at
    an internal node are released after that node's group has completed (or
    after reset if reset is longer); endpoint memories are released when the
    request completes.  Start ticks are internal feasibility variables and do
    not become extra controller actions.
    """

    quantum = (
        _reliable_time_quantum_ps(problem)
        if time_quantum_ps is None else int(time_quantum_ps)
    )
    config = problem.config
    if (
        quantum < 1
        or config.generation_interval_ps % quantum
        or config.swap_service_ps % quantum
        or config.slot_duration_ps % quantum
    ):
        raise ValueError("invalid reliable-memory MILP time quantum")
    service_ticks = config.swap_service_ps // quantum
    reset_ticks = ceil(config.memory_reset_ps / quantum)
    groups = plan.schedule.groups
    completion_ticks = max(1, len(groups) * service_ticks)

    group_by_node = {
        node: group_index
        for group_index, group in enumerate(groups, start=1)
        for node in group
    }
    internal_release = {
        node: (
            (group_index - 1) * service_ticks
            + max(service_ticks, reset_ticks)
        )
        for node, group_index in group_by_node.items()
    }
    resource_ticks = max(
        completion_ticks,
        *(internal_release.values() or (0,)),
    )

    memory_profile: dict[Node, tuple[int, ...]] = {}
    for path_index, node in enumerate(plan.path):
        if path_index in {0, len(plan.path) - 1}:
            release = completion_ticks
            amount = 1
        else:
            release = internal_release[node]
            amount = 2
        memory_profile[node] = tuple(
            amount if tick < release else 0
            for tick in range(resource_ticks)
        )

    edge_profile: dict[Edge, tuple[int, ...]] = {}
    internal_nodes = set(plan.path[1:-1])
    for elementary_edge in plan.elementary_edges:
        adjacent = tuple(
            node for node in elementary_edge if node in internal_nodes
        )
        release = (
            min(
                group_by_node[node] * service_ticks
                for node in adjacent
            )
            if adjacent else completion_ticks
        )
        edge_profile[elementary_edge] = tuple(
            1 if tick < release else 0
            for tick in range(resource_ticks)
        )

    bsm_profile: dict[Node, tuple[int, ...]] = {
        node: tuple(0 for _ in range(resource_ticks))
        for node in plan.path[1:-1]
    }
    mutable_bsm = {
        node: list(values) for node, values in bsm_profile.items()
    }
    for group_index, group in enumerate(groups):
        start = group_index * service_ticks
        finish = start + service_ticks
        for node in group:
            for tick in range(start, finish):
                mutable_bsm[node][tick] = 1
    bsm_profile = {
        node: tuple(values) for node, values in mutable_bsm.items()
    }
    return ReliableMemoryCandidateProfile(
        plan=plan,
        time_quantum_ps=quantum,
        completion_ticks=completion_ticks,
        resource_ticks=resource_ticks,
        memory_profile=memory_profile,
        edge_profile=edge_profile,
        bsm_profile=bsm_profile,
    )


@dataclass(frozen=True)
class MilpReliableMemoryPlanResult:
    """Proven optimum of the deterministic reliable-memory abstraction."""

    selected_plan_ids: tuple[str, ...]
    completed_count: int
    scheduled_start_ticks: tuple[tuple[str, int], ...]
    scheduled_start_ps: tuple[tuple[str, int], ...]
    inventory_assignments: tuple[tuple[str, str, Edge], ...]
    reliable_generation_assignments: tuple[
        tuple[str, Edge, int], ...
    ]
    reliability_confidence: float
    time_quantum_ps: int
    reliable_supply_by_edge: tuple[tuple[Edge, int], ...]
    nominal_risk_micros: int
    memory_time_qubit_ticks: int
    completion_time_ticks: int
    backend: str
    eligible_candidates: int
    feasible_candidate_starts: int
    filtered_candidates: int
    resource_constraint_count: int
    proven_optimal: bool = True
    certified_optimal: bool = False


@dataclass(frozen=True)
class _ReliableChoice:
    profile: ReliableMemoryCandidateProfile
    start_tick: int


def _probability_risk_micros(probability: float) -> int:
    if probability <= 0.0:
        return 1_000_000_000
    if probability >= 1.0:
        return 0
    return int(round(-log(probability) * 1_000_000))


def _solve_reliable_memory_snapshot_milp(
    snapshot: OrderBatchSnapshot,
    *,
    allow_swap_orders: bool,
    reliability_confidence: float,
) -> MilpReliableMemoryPlanResult:
    """Optimize one online batch under reliable EPR and timed memory supply."""

    if not 0.0 < reliability_confidence <= 1.0:
        raise ValueError("reliability_confidence must lie in (0, 1]")
    problem = snapshot.problem
    inactive = tuple(
        plan for plan in snapshot.candidates
        if (
            plan.deadline_slot is not None
            and plan.decision_slot >= plan.deadline_slot
        )
    )
    if inactive:
        raise ValueError(
            "reliable-memory MILP snapshot contains inactive candidates"
        )

    eligible = tuple(sorted((
        plan for plan in snapshot.candidates
        if allow_swap_orders or plan.is_fixed_order
    ), key=_nominal_candidate_key))
    missing = problem.required_requests - {
        plan.request_id for plan in eligible
    }
    if missing:
        raise RuntimeError(
            "reliable-memory MILP catalogue misses required requests: "
            f"{sorted(missing)}"
        )

    quantum = _reliable_time_quantum_ps(problem)
    horizon = problem.config.slot_duration_ps // quantum
    arrivals = reliable_link_arrivals(
        problem,
        reliability_confidence,
        time_quantum_ps=quantum,
    )
    inventory = problem.initial_inventory
    inventory_by_edge: dict[Edge, tuple[int, ...]] = {}
    for elementary_edge in problem.physical_edges:
        inventory_by_edge[elementary_edge] = tuple(
            index for index, stored in enumerate(inventory)
            if stored.elementary_edge == elementary_edge
        )

    choices: list[_ReliableChoice] = []
    feasible_plan_ids: set[str] = set()
    for plan in eligible:
        if plan.swap_order and problem.config.swap_probability == 0.0:
            continue
        profile = compile_reliable_memory_candidate(
            problem, plan, time_quantum_ps=quantum
        )
        if profile.completion_ticks > horizon:
            continue
        for start_tick in range(
            0, horizon - profile.completion_ticks + 1
        ):
            if plan.request_id not in problem.preloaded_requests:
                sources_exist = all(
                    bool(inventory_by_edge[elementary_edge])
                    or any(
                        amount > 0 and birth_tick <= start_tick
                        for birth_tick, amount in enumerate(
                            arrivals[elementary_edge]
                        )
                    )
                    for elementary_edge in plan.elementary_edges
                )
                if not sources_exist:
                    continue
            choices.append(_ReliableChoice(profile, start_tick))
            feasible_plan_ids.add(plan.plan_id)

    request_ids = tuple(sorted(
        {plan.request_id for plan in eligible},
        key=lambda request_id: (
            min(
                plan.priority for plan in eligible
                if plan.request_id == request_id
            ),
            request_id,
        ),
    ))
    choices_by_request = {
        request_id: tuple(
            index for index, choice in enumerate(choices)
            if choice.profile.plan.request_id == request_id
        )
        for request_id in request_ids
    }
    unavailable_required = {
        request_id for request_id in problem.required_requests
        if not choices_by_request[request_id]
    }
    if unavailable_required:
        raise RuntimeError(
            "no reliable-memory candidate remains for required requests: "
            f"{sorted(unavailable_required)}"
        )

    baseline_memory = defaultdict(int)
    baseline_edges = defaultdict(int)
    for stored in inventory:
        baseline_memory[stored.left] += 1
        baseline_memory[stored.right] += 1
        baseline_edges[stored.elementary_edge] += 1

    total_generation_attempts = {
        elementary_edge: (
            len(range(
                0,
                problem.config.slot_duration_ps,
                problem.config.generation_interval_ps,
            )) * link.capacity
        )
        for elementary_edge, link in problem.link_by_edge.items()
    }
    link_risk = {}
    for elementary_edge, link in problem.link_by_edge.items():
        attempts = total_generation_attempts[elementary_edge]
        readiness = (
            0.0 if attempts == 0 else
            1.0 - (1.0 - link.generation_probability) ** attempts
        )
        link_risk[elementary_edge] = _probability_risk_micros(
            readiness
        )
    swap_risk = _probability_risk_micros(
        problem.config.swap_probability
    )
    completion_bound = horizon * max(len(request_ids), 1) + 1

    def build_model(
        phase: str,
        *,
        fixed_cardinality: int | None = None,
        fixed_risk: int | None = None,
        fixed_tertiary: int | None = None,
    ):
        model = cp_model.CpModel()
        x_vars = tuple(
            model.NewBoolVar(f"x_{index}")
            for index in range(len(choices))
        )
        y_vars = {
            request_id: model.NewBoolVar(f"y_{index}")
            for index, request_id in enumerate(request_ids)
        }
        inventory_vars = {}
        generation_vars = {}
        source_vars_by_choice_edge = defaultdict(list)

        for choice_index, choice in enumerate(choices):
            plan = choice.profile.plan
            if plan.request_id in problem.preloaded_requests:
                continue
            for edge_index, elementary_edge in enumerate(
                plan.elementary_edges
            ):
                for stored_index in inventory_by_edge[elementary_edge]:
                    variable = model.NewBoolVar(
                        f"z_{choice_index}_{edge_index}_{stored_index}"
                    )
                    key = (choice_index, edge_index, stored_index)
                    inventory_vars[key] = variable
                    source_vars_by_choice_edge[
                        (choice_index, edge_index)
                    ].append(variable)
                for birth_tick, amount in enumerate(
                    arrivals[elementary_edge]
                ):
                    if amount <= 0 or birth_tick > choice.start_tick:
                        continue
                    variable = model.NewBoolVar(
                        f"g_{choice_index}_{edge_index}_{birth_tick}"
                    )
                    key = (choice_index, edge_index, birth_tick)
                    generation_vars[key] = variable
                    source_vars_by_choice_edge[
                        (choice_index, edge_index)
                    ].append(variable)
                sources = source_vars_by_choice_edge[
                    (choice_index, edge_index)
                ]
                if sources:
                    model.Add(sum(sources) == x_vars[choice_index])
                else:
                    model.Add(x_vars[choice_index] == 0)

        for request_id, choice_indices in choices_by_request.items():
            model.Add(
                sum(x_vars[index] for index in choice_indices)
                == y_vars[request_id]
            )
        for request_id in problem.required_requests:
            model.Add(y_vars[request_id] == 1)

        for stored_index in range(len(inventory)):
            relevant = tuple(
                variable
                for (choice_index, edge_index, index), variable
                in inventory_vars.items()
                if index == stored_index
            )
            if relevant:
                model.Add(sum(relevant) <= 1)
        for elementary_edge in problem.physical_edges:
            for birth_tick, amount in enumerate(arrivals[elementary_edge]):
                if amount <= 0:
                    continue
                relevant = tuple(
                    variable
                    for (choice_index, edge_index, tick), variable
                    in generation_vars.items()
                    if (
                        tick == birth_tick
                        and choices[choice_index].profile.plan.elementary_edges[
                            edge_index
                        ] == elementary_edge
                    )
                )
                if relevant:
                    model.Add(sum(relevant) <= int(amount))

        resource_constraint_count = 0
        for node, capacity in problem.capacity.items():
            for tick in range(horizon):
                terms = []
                for choice_index, choice in enumerate(choices):
                    plan = choice.profile.plan
                    if tick >= choice.start_tick:
                        offset = tick - choice.start_tick
                        values = choice.profile.memory_profile.get(node, ())
                        if offset < len(values) and values[offset]:
                            terms.append(values[offset] * x_vars[choice_index])
                    elif plan.request_id in problem.preloaded_requests:
                        incident = sum(
                            node in elementary_edge
                            for elementary_edge in plan.elementary_edges
                        )
                        if incident:
                            terms.append(incident * x_vars[choice_index])
                for (
                    choice_index, edge_index, birth_tick
                ), variable in generation_vars.items():
                    choice = choices[choice_index]
                    elementary_edge = choice.profile.plan.elementary_edges[
                        edge_index
                    ]
                    if (
                        birth_tick <= tick < choice.start_tick
                        and node in elementary_edge
                    ):
                        terms.append(variable)
                for (
                    choice_index, edge_index, stored_index
                ), variable in inventory_vars.items():
                    choice = choices[choice_index]
                    if (
                        tick >= choice.start_tick
                        and node in inventory[stored_index].elementary_edge
                    ):
                        terms.append(-variable)
                if terms:
                    model.Add(
                        sum(terms) <= capacity - baseline_memory[node]
                    )
                    resource_constraint_count += 1

        for elementary_edge, link in problem.link_by_edge.items():
            for tick in range(horizon):
                terms = []
                for choice_index, choice in enumerate(choices):
                    plan = choice.profile.plan
                    if elementary_edge not in plan.elementary_edges:
                        continue
                    if tick >= choice.start_tick:
                        offset = tick - choice.start_tick
                        values = choice.profile.edge_profile[elementary_edge]
                        if offset < len(values) and values[offset]:
                            terms.append(values[offset] * x_vars[choice_index])
                    elif plan.request_id in problem.preloaded_requests:
                        terms.append(x_vars[choice_index])
                for (
                    choice_index, edge_index, birth_tick
                ), variable in generation_vars.items():
                    choice = choices[choice_index]
                    if (
                        choice.profile.plan.elementary_edges[edge_index]
                        == elementary_edge
                        and birth_tick <= tick < choice.start_tick
                    ):
                        terms.append(variable)
                for (
                    choice_index, edge_index, stored_index
                ), variable in inventory_vars.items():
                    choice = choices[choice_index]
                    if (
                        inventory[stored_index].elementary_edge
                        == elementary_edge
                        and tick >= choice.start_tick
                    ):
                        terms.append(-variable)
                if terms:
                    model.Add(
                        sum(terms) <= link.capacity
                        - baseline_edges[elementary_edge]
                    )
                    resource_constraint_count += 1

        for node in problem.capacity:
            for tick in range(horizon):
                terms = []
                for choice_index, choice in enumerate(choices):
                    if tick < choice.start_tick:
                        continue
                    offset = tick - choice.start_tick
                    values = choice.profile.bsm_profile.get(node, ())
                    if offset < len(values) and values[offset]:
                        terms.append(values[offset] * x_vars[choice_index])
                if terms:
                    model.Add(
                        sum(terms)
                        <= problem.config.bsm_capacity_per_node
                    )
                    resource_constraint_count += 1

        cardinality = sum(y_vars.values())
        risk_terms = []
        memory_terms = []
        completion_terms = []
        stable_terms = []
        for choice_index, choice in enumerate(choices):
            plan = choice.profile.plan
            risk_terms.append(
                len(plan.swap_order) * swap_risk * x_vars[choice_index]
            )
            profile_area = sum(
                sum(values[: max(0, horizon - choice.start_tick)])
                for values in choice.profile.memory_profile.values()
            )
            if plan.request_id in problem.preloaded_requests:
                profile_area += 2 * len(plan.elementary_edges) * (
                    choice.start_tick
                )
            memory_terms.append(profile_area * x_vars[choice_index])
            completion_terms.append(
                (choice.start_tick + choice.profile.completion_ticks)
                * x_vars[choice_index]
            )
            stable_terms.append((choice_index + 1) * x_vars[choice_index])
        for source_index, (key, variable) in enumerate(
            sorted(generation_vars.items()), start=1
        ):
            choice_index, edge_index, birth_tick = key
            choice = choices[choice_index]
            elementary_edge = choice.profile.plan.elementary_edges[edge_index]
            risk_terms.append(link_risk[elementary_edge] * variable)
            memory_terms.append(
                2 * (choice.start_tick - birth_tick) * variable
            )
            stable_terms.append(
                (len(choices) + source_index) * variable
            )
        stable_offset = len(choices) + len(generation_vars)
        for source_index, (key, variable) in enumerate(
            sorted(inventory_vars.items()), start=1
        ):
            choice_index, _edge_index, _stored_index = key
            choice = choices[choice_index]
            memory_terms.append(
                -2 * (horizon - choice.start_tick) * variable
            )
            stable_terms.append(
                (stable_offset + source_index) * variable
            )

        risk = sum(risk_terms)
        memory_time = sum(memory_terms)
        completion_time = sum(completion_terms)
        tertiary = memory_time * completion_bound + completion_time
        stable = sum(stable_terms)
        if fixed_cardinality is not None:
            model.Add(cardinality == int(fixed_cardinality))
        if fixed_risk is not None:
            model.Add(risk == int(fixed_risk))
        if fixed_tertiary is not None:
            model.Add(tertiary == int(fixed_tertiary))
        if phase == "cardinality":
            model.Maximize(cardinality)
        elif phase == "risk":
            model.Minimize(risk)
        elif phase == "memory":
            model.Minimize(tertiary)
        elif phase == "stable":
            model.Minimize(stable)
        else:
            raise ValueError(f"unknown reliable MILP phase: {phase!r}")
        if x_vars:
            model.AddDecisionStrategy(
                x_vars,
                cp_model.CHOOSE_FIRST,
                cp_model.SELECT_MAX_VALUE,
            )
        return {
            "model": model,
            "x": x_vars,
            "y": y_vars,
            "inventory": inventory_vars,
            "generation": generation_vars,
            "cardinality": cardinality,
            "risk": risk,
            "memory_time": memory_time,
            "completion_time": completion_time,
            "tertiary": tertiary,
            "resource_constraint_count": resource_constraint_count,
        }

    def solve_phase(bundle, phase: str):
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.search_branching = cp_model.FIXED_SEARCH
        status = solver.Solve(bundle["model"])
        if status == cp_model.INFEASIBLE:
            raise RuntimeError(
                f"reliable-memory MILP {phase} phase is infeasible"
            )
        if status != cp_model.OPTIMAL:
            raise RuntimeError(
                "reliable-memory MILP did not prove the "
                f"{phase} optimum: status={solver.StatusName()}"
            )
        return solver

    primary = build_model("cardinality")
    primary_solver = solve_phase(primary, "cardinality")
    cardinality_optimum = int(primary_solver.Value(primary["cardinality"]))

    # Keep one variable/constraint graph for all lexicographic phases.  Each
    # proven optimum is fixed before replacing the objective for the next
    # phase.  This is exactly equivalent to rebuilding the same model four
    # times, but avoids most Python-side construction cost on formal batches.
    final = primary
    final["model"].Add(
        final["cardinality"] == cardinality_optimum
    )
    final["model"].Minimize(final["risk"])
    secondary_solver = solve_phase(final, "nominal probability")
    risk_optimum = int(secondary_solver.Value(final["risk"]))

    final["model"].Add(final["risk"] == risk_optimum)
    final["model"].Minimize(final["tertiary"])
    tertiary_solver = solve_phase(final, "memory-time")
    tertiary_optimum = int(tertiary_solver.Value(final["tertiary"]))

    final["model"].Add(final["tertiary"] == tertiary_optimum)
    stable_terms = []
    for choice_index, variable in enumerate(final["x"], start=1):
        stable_terms.append(choice_index * variable)
    stable_offset = len(final["x"])
    for source_index, (_key, variable) in enumerate(
        sorted(final["generation"].items()), start=1
    ):
        stable_terms.append((stable_offset + source_index) * variable)
    stable_offset += len(final["generation"])
    for source_index, (_key, variable) in enumerate(
        sorted(final["inventory"].items()), start=1
    ):
        stable_terms.append((stable_offset + source_index) * variable)
    final["model"].Minimize(sum(stable_terms))
    final_solver = solve_phase(final, "stable tie-break")
    selected_choice_indices = tuple(
        index for index, variable in enumerate(final["x"])
        if final_solver.Value(variable)
    )
    selected_choices = tuple(choices[index] for index in selected_choice_indices)
    selected_choices = tuple(sorted(
        selected_choices,
        key=lambda choice: (
            choice.profile.plan.priority,
            choice.profile.plan.request_id,
            choice.profile.plan.plan_id,
        ),
    ))
    selected_ids = tuple(
        choice.profile.plan.plan_id for choice in selected_choices
    )
    if len(selected_ids) != cardinality_optimum:
        raise RuntimeError(
            "reliable-memory MILP cardinality disagrees with selected plans"
        )

    inventory_assignments = []
    for (
        choice_index, edge_index, stored_index
    ), variable in final["inventory"].items():
        if not final_solver.Value(variable):
            continue
        choice = choices[choice_index]
        inventory_assignments.append((
            inventory[stored_index].pair_id,
            choice.profile.plan.plan_id,
            choice.profile.plan.elementary_edges[edge_index],
        ))
    generation_assignments = []
    for (
        choice_index, edge_index, birth_tick
    ), variable in final["generation"].items():
        if not final_solver.Value(variable):
            continue
        choice = choices[choice_index]
        generation_assignments.append((
            choice.profile.plan.plan_id,
            choice.profile.plan.elementary_edges[edge_index],
            birth_tick,
        ))
    scheduled_ticks = tuple(
        (choice.profile.plan.plan_id, choice.start_tick)
        for choice in selected_choices
    )
    baseline_memory_time = 2 * len(inventory) * horizon
    return MilpReliableMemoryPlanResult(
        selected_plan_ids=selected_ids,
        completed_count=cardinality_optimum,
        scheduled_start_ticks=scheduled_ticks,
        scheduled_start_ps=tuple(
            (plan_id, tick * quantum) for plan_id, tick in scheduled_ticks
        ),
        inventory_assignments=tuple(sorted(inventory_assignments)),
        reliable_generation_assignments=tuple(sorted(
            generation_assignments,
            key=lambda item: (
                item[2], repr(item[1]), item[0]
            ),
        )),
        reliability_confidence=float(reliability_confidence),
        time_quantum_ps=quantum,
        reliable_supply_by_edge=tuple(
            (elementary_edge, sum(arrivals[elementary_edge]))
            for elementary_edge in problem.physical_edges
        ),
        nominal_risk_micros=int(final_solver.Value(final["risk"])),
        memory_time_qubit_ticks=(
            baseline_memory_time
            + int(final_solver.Value(final["memory_time"]))
        ),
        completion_time_ticks=int(
            final_solver.Value(final["completion_time"])
        ),
        backend="ortools-cp-sat-reliable-time-indexed-memory-milp",
        eligible_candidates=len(eligible),
        feasible_candidate_starts=len(choices),
        filtered_candidates=len(eligible) - len(feasible_plan_ids),
        resource_constraint_count=int(
            final["resource_constraint_count"]
        ),
    )


class _MilpReliableMemoryPlanner:
    """Online deterministic optimizer for the reliable-memory abstraction."""

    allow_swap_orders = False
    name = "milp_reliable_memory_path"

    def __init__(self, *, reliability_confidence: float = 0.9) -> None:
        if not 0.0 < reliability_confidence <= 1.0:
            raise ValueError("reliability_confidence must lie in (0, 1]")
        self.reliability_confidence = float(reliability_confidence)
        self.last_objective = 0
        self.last_solution: MilpReliableMemoryPlanResult | None = None
        self.last_evaluations = 0
        self.last_proven_optimal = False

    def reset(self, episode_seed: int) -> None:
        del episode_seed
        self.last_objective = 0
        self.last_solution = None
        self.last_evaluations = 0
        self.last_proven_optimal = False

    def select(self, snapshot: OrderBatchSnapshot) -> tuple[str, ...]:
        solution = _solve_reliable_memory_snapshot_milp(
            snapshot,
            allow_swap_orders=self.allow_swap_orders,
            reliability_confidence=self.reliability_confidence,
        )
        self.last_solution = solution
        self.last_objective = solution.completed_count
        self.last_evaluations = 0
        self.last_proven_optimal = True
        return solution.selected_plan_ids


class MilpReliableMemoryPathPlanner(_MilpReliableMemoryPlanner):
    """Reliable-memory MILP restricted to one fixed schedule per path."""

    allow_swap_orders = False
    name = "milp_reliable_memory_path"


class MilpReliableMemoryPathOrderPlanner(_MilpReliableMemoryPlanner):
    """Reliable-memory MILP over complete cached path/schedule candidates."""

    allow_swap_orders = True
    name = "milp_reliable_memory_path_order"


class _MilpNominalPlanner:
    """Online hybrid exact optimizer for planner-owned physical scenarios.

    Link-generation and swap probabilities are *not* replaced by one.  For
    each supplied planner-owned seed, the unchanged event executor converts
    those probabilities into one deterministic physical realization.  A
    selected request is model-completable only when it completes in at least
    ``ceil(chance_threshold * K)`` of the ``K`` scenarios.  The hybrid optimizer
    returns the largest batch for which every selected request meets that
    threshold.

    HiGHS first solves a static 0-1 resource relaxation.  At every cardinality
    from that proven upper bound downward, deterministic single-worker CP-SAT
    enumerates every assignment allowed by the same relaxation.  The unchanged
    executor validates each complete assignment in every fixed planning
    scenario.  With ``oracle_workers > 1``, only those independent executor
    calls run in spawned processes; their results are committed in the original
    CP-SAT order.  A failed partial or complete subset is never generalized into
    a monotonic conflict cut.  Exhausting every higher cardinality before
    accepting the first executor-feasible assignment is therefore exact for
    this finite scenario model and candidate catalogue.  It is not a
    clairvoyant optimum for the environment's hidden physical realization.
    """

    allow_swap_orders = False
    name = "milp_nominal_path"

    def __init__(
        self,
        planning_seeds: Iterable[int] = (0,),
        *,
        chance_threshold: float = 1.0,
        oracle_workers: int = 1,
        oracle_batch_size: int = _DEFAULT_NOMINAL_ORACLE_BATCH_SIZE,
    ) -> None:
        seeds = tuple(map(int, planning_seeds))
        if not seeds:
            raise ValueError("planning_seeds cannot be empty")
        if len(set(seeds)) != len(seeds):
            raise ValueError("planning_seeds cannot contain duplicates")
        if not 0.0 < chance_threshold <= 1.0:
            raise ValueError("chance_threshold must lie in (0, 1]")
        if int(oracle_workers) != oracle_workers or oracle_workers < 1:
            raise ValueError("oracle_workers must be a positive integer")
        if int(oracle_batch_size) != oracle_batch_size or oracle_batch_size < 1:
            raise ValueError("oracle_batch_size must be a positive integer")
        self.planning_seeds = seeds
        self.chance_threshold = float(chance_threshold)
        self.oracle_workers = int(oracle_workers)
        self.oracle_batch_size = int(oracle_batch_size)
        self.required_scenarios = int(
            (
                Decimal(str(chance_threshold))
                * len(self.planning_seeds)
            ).to_integral_value(rounding=ROUND_CEILING)
        )
        self.last_objective = 0
        self.last_solution: MilpNominalPlanResult | None = None
        self.last_evaluations = 0
        self.last_proven_optimal = False

    def reset(self, episode_seed: int) -> None:
        # Planning scenarios are deliberately fixed independently of the
        # episode's hidden physical RNG stream.
        del episode_seed
        self.last_objective = 0
        self.last_solution = None
        self.last_evaluations = 0
        self.last_proven_optimal = False

    def _completion_counts(
        self,
        problem: OrderBatchProblem,
        plan_ids: tuple[str, ...],
    ) -> dict[str, int]:
        selected_ids = set(plan_ids)
        counts = {request_id: 0 for request_id in (
            plan.request_id for plan in problem.candidates
            if plan.plan_id in selected_ids
        )}
        if not plan_ids:
            return counts
        for planning_seed in self.planning_seeds:
            result = simulate_order_batch(
                problem.with_physics_seed(planning_seed),
                plan_ids,
                record_traces=False,
            )
            self.last_evaluations += 1
            for request_id in result.completed:
                counts[request_id] += 1
        return counts

    def _validate_incumbent(
        self,
        snapshot: OrderBatchSnapshot,
        eligible: tuple[OrderPlan, ...],
        request_ids: tuple[str, ...],
        plan_ids: Iterable[str],
    ) -> tuple[tuple[str, ...], tuple[tuple[str, int], ...]]:
        """Re-verify a caller-supplied feasible lower bound with this oracle."""

        supplied = tuple(plan_ids)
        if len(set(supplied)) != len(supplied):
            raise ValueError("known-feasible incumbent repeats a plan ID")
        lookup = {plan.plan_id: plan for plan in eligible}
        unknown = set(supplied) - lookup.keys()
        if unknown:
            raise ValueError(
                "known-feasible incumbent contains ineligible plan IDs: "
                f"{sorted(unknown)}"
            )
        selected_requests = tuple(
            lookup[plan_id].request_id for plan_id in supplied
        )
        if len(set(selected_requests)) != len(selected_requests):
            raise ValueError(
                "known-feasible incumbent selects multiple plans for one "
                "request"
            )
        missing_required = (
            snapshot.problem.required_requests - set(selected_requests)
        )
        if missing_required:
            raise ValueError(
                "known-feasible incumbent omits required requests: "
                f"{sorted(missing_required)}"
            )

        selected_by_request = {
            lookup[plan_id].request_id: plan_id for plan_id in supplied
        }
        normalized = tuple(
            selected_by_request[request_id]
            for request_id in request_ids
            if request_id in selected_by_request
        )
        try:
            counts = self._completion_counts(snapshot.problem, normalized)
        except ValueError as error:
            raise ValueError(
                "known-feasible incumbent was rejected by the shared "
                f"executor: {error}"
            ) from error
        infeasible = tuple(
            request_id for request_id in selected_by_request
            if counts[request_id] < self.required_scenarios
        )
        if infeasible:
            raise ValueError(
                "known-feasible incumbent fails this planner's physical "
                "scenario oracle: "
                f"requests={list(infeasible)}, counts={dict(counts)}, "
                f"required_scenarios={self.required_scenarios}"
            )
        return normalized, tuple(sorted(counts.items()))

    def select(self, snapshot: OrderBatchSnapshot) -> tuple[str, ...]:
        """Solve without a caller-provided feasible lower bound."""

        return self._select(snapshot, known_feasible_plan_ids=None)

    def select_with_incumbent(
        self,
        snapshot: OrderBatchSnapshot,
        known_feasible_plan_ids: Iterable[str],
    ) -> tuple[str, ...]:
        """Solve exactly while reusing a separately verified feasible action."""

        return self._select(
            snapshot,
            known_feasible_plan_ids=tuple(known_feasible_plan_ids),
        )

    def _select(
        self,
        snapshot: OrderBatchSnapshot,
        *,
        known_feasible_plan_ids: tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        inactive = tuple(
            plan for plan in snapshot.candidates
            if (
                plan.deadline_slot is not None
                and plan.decision_slot >= plan.deadline_slot
            )
        )
        if inactive:
            details = ", ".join(
                f"{plan.plan_id}(decision_slot={plan.decision_slot}, "
                f"deadline_slot={plan.deadline_slot})"
                for plan in inactive
            )
            raise ValueError(
                "nominal optimizer snapshot contains inactive candidates; "
                "active requests require decision_slot < deadline_slot: "
                f"{details}"
            )

        eligible = tuple(sorted((
            plan for plan in snapshot.candidates
            if self.allow_swap_orders or plan.is_fixed_order
        ), key=_nominal_candidate_key))
        missing = snapshot.problem.required_requests - {
            plan.request_id for plan in eligible
        }
        if missing:
            raise RuntimeError(
                "nominal optimizer catalogue misses required requests: "
                f"{sorted(missing)}"
            )
        raw_by_request: dict[str, list[OrderPlan]] = defaultdict(list)
        for plan in eligible:
            raw_by_request[plan.request_id].append(plan)
        request_ids = tuple(sorted(
            raw_by_request,
            key=lambda request_id: (
                min(
                    plan.priority
                    for plan in raw_by_request[request_id]
                ),
                request_id,
            ),
        ))
        self.last_evaluations = 0
        incumbent_ids: tuple[str, ...] | None = None
        incumbent_counts: tuple[tuple[str, int], ...] = ()
        if known_feasible_plan_ids is not None:
            incumbent_ids, incumbent_counts = self._validate_incumbent(
                snapshot,
                eligible,
                request_ids,
                known_feasible_plan_ids,
            )
        if not request_ids:
            solution = MilpNominalPlanResult(
                selected_plan_ids=incumbent_ids or (),
                completed_count=0,
                scenario_completion_counts=incumbent_counts,
                planning_seeds=self.planning_seeds,
                required_scenarios=self.required_scenarios,
                evaluations=0,
                milp_solves=0,
                cuts=0,
                backend=_NOMINAL_BACKEND,
                eligible_candidates=0,
                filtered_candidates=0,
                enumerated_assignments=(
                    1 if incumbent_ids is not None else 0
                ),
                static_upper_bound=0,
                proven_optimal=True,
            )
            self.last_objective = 0
            self.last_solution = solution
            self.last_proven_optimal = True
            return ()

        hard_possible_counts = {
            plan.plan_id: sum(
                _plan_hard_possible_in_scenario(
                    snapshot.problem, plan, planning_seed
                )
                for planning_seed in self.planning_seeds
            )
            for plan in eligible
        }
        plans = tuple(
            plan for plan in eligible
            if hard_possible_counts[plan.plan_id]
            >= self.required_scenarios
        )
        filtered_candidates = len(eligible) - len(plans)
        if incumbent_ids is not None:
            contradicted = set(incumbent_ids) - {
                plan.plan_id for plan in plans
            }
            if contradicted:
                raise RuntimeError(
                    "hard-possible screening rejected an executor-verified "
                    "incumbent; the nominal optimizer's necessary filter is "
                    f"unsound for plans: {sorted(contradicted)}"
                )
        plans_by_request = {
            request_id: tuple(
                plan for plan in plans
                if plan.request_id == request_id
            )
            for request_id in request_ids
        }
        unavailable_required = {
            request_id for request_id in snapshot.problem.required_requests
            if not plans_by_request[request_id]
        }
        if unavailable_required:
            raise RuntimeError(
                "no nominally completable candidate remains for required "
                f"requests: {sorted(unavailable_required)}"
            )

        searchable_request_ids = tuple(
            request_id for request_id in request_ids
            if plans_by_request[request_id]
        )
        if not searchable_request_ids:
            solution = MilpNominalPlanResult(
                selected_plan_ids=(),
                completed_count=0,
                scenario_completion_counts=(),
                planning_seeds=self.planning_seeds,
                required_scenarios=self.required_scenarios,
                evaluations=0,
                milp_solves=0,
                cuts=0,
                backend=_NOMINAL_BACKEND,
                eligible_candidates=len(eligible),
                filtered_candidates=filtered_candidates,
                enumerated_assignments=1,
                static_upper_bound=0,
                proven_optimal=True,
            )
            self.last_objective = 0
            self.last_solution = solution
            self.last_proven_optimal = True
            return ()

        plan_index = {
            plan.plan_id: index for index, plan in enumerate(plans)
        }
        request_index = {
            request_id: len(plans) + index
            for index, request_id in enumerate(searchable_request_ids)
        }
        variable_count = len(plans) + len(searchable_request_ids)
        resource_rows = _static_resource_rows(
            snapshot.problem,
            plans,
            scenario_count=len(self.planning_seeds),
            required_scenarios=self.required_scenarios,
        )
        if incumbent_ids is not None:
            incumbent_indices = {
                plan_index[plan_id] for plan_id in incumbent_ids
            }
            violated_rows = tuple(
                row_number
                for row_number, (coefficients, upper) in enumerate(
                    resource_rows
                )
                if sum(
                    value for index, value in coefficients.items()
                    if index in incumbent_indices
                ) > upper
            )
            if violated_rows:
                raise RuntimeError(
                    "static necessary constraints rejected an executor-"
                    "verified incumbent; violated rows: "
                    f"{violated_rows}"
                )

        # HiGHS proves a cardinality upper bound for the static necessary
        # conditions.  It is deliberately solved only once; black-box executor
        # failures are handled by one CP-SAT enumeration, not repeated restarts.
        row_specs: list[tuple[dict[int, float], float, float]] = []
        for request_id in searchable_request_ids:
            coefficients = {
                plan_index[plan.plan_id]: 1.0
                for plan in plans_by_request[request_id]
            }
            coefficients[request_index[request_id]] = -1.0
            row_specs.append((coefficients, 0.0, 0.0))
        for coefficients, upper in resource_rows:
            row_specs.append((
                {index: float(value) for index, value in coefficients.items()},
                -np.inf,
                float(upper),
            ))

        matrix = lil_matrix((len(row_specs), variable_count), dtype=float)
        for row_index, (coefficients, _, _) in enumerate(row_specs):
            for column, value in coefficients.items():
                matrix[row_index, column] = value
        lower_bounds = np.zeros(variable_count, dtype=float)
        upper_bounds = np.ones(variable_count, dtype=float)
        for request_id in snapshot.problem.required_requests:
            index = request_index[request_id]
            lower_bounds[index] = 1.0
            upper_bounds[index] = 1.0
        objective = np.zeros(variable_count, dtype=float)
        for request_id in searchable_request_ids:
            objective[request_index[request_id]] = -1.0
        static_result = milp(
            c=objective,
            integrality=np.ones(variable_count, dtype=int),
            bounds=Bounds(lower_bounds, upper_bounds),
            constraints=LinearConstraint(
                matrix.tocsr(),
                np.asarray([lower for _, lower, _ in row_specs]),
                np.asarray([upper for _, _, upper in row_specs]),
            ),
            options={"disp": False, "mip_rel_gap": 0.0},
        )
        if not static_result.success or static_result.x is None:
            if getattr(static_result, "status", None) == 2:
                raise RuntimeError(
                    "no nominally completable assignment satisfies all "
                    "required requests"
                )
            raise RuntimeError(
                "HiGHS nominal static master failed: "
                f"{static_result.message}"
            )
        _require_proven_mip_optimum(static_result)
        static_value = -float(static_result.fun)
        static_upper_bound = int(round(static_value))
        if abs(static_value - static_upper_bound) > 1e-7:
            raise RuntimeError(
                "nominal static master returned a fractional cardinality: "
                f"{static_value}"
            )

        required_count = len(snapshot.problem.required_requests)
        oracle_cache: dict[
            tuple[str, ...], tuple[tuple[tuple[str, int], ...], bool]
        ] = {}
        enumerated_assignments = 0
        selected_ids: tuple[str, ...] | None = None
        ordered_counts: tuple[tuple[str, int], ...] = ()

        incumbent_count: int | None = None
        if incumbent_ids is not None:
            incumbent_count = len(incumbent_ids)
            if static_upper_bound < incumbent_count:
                raise RuntimeError(
                    "nominal static upper bound fell below an executor-"
                    "verified incumbent: "
                    f"upper_bound={static_upper_bound}, "
                    f"incumbent={incumbent_count}"
                )
            oracle_cache[incumbent_ids] = (incumbent_counts, True)
            enumerated_assignments = 1
            if static_upper_bound == incumbent_count:
                selected_ids = incumbent_ids
                ordered_counts = incumbent_counts

        # The proven static-master optimum is a useful first complete action.
        # Testing it changes only search order; an oracle failure is not turned
        # into a cut or used to prune any other assignment.
        if selected_ids is None:
            master_selected_requests = tuple(
                request_id for request_id in searchable_request_ids
                if static_result.x[request_index[request_id]] > 0.5
            )
            master_ids = tuple(
                next(
                    plan.plan_id
                    for plan in plans_by_request[request_id]
                    if static_result.x[plan_index[plan.plan_id]] > 0.5
                )
                for request_id in master_selected_requests
            )
            if len(master_ids) != static_upper_bound:
                raise RuntimeError(
                    "nominal static master cardinality disagrees with selected "
                    f"plans: {static_upper_bound} != {len(master_ids)}"
                )
            master_counts = self._completion_counts(
                snapshot.problem, master_ids
            )
            enumerated_assignments += 1
            master_ordered_counts = tuple(sorted(master_counts.items()))
            master_feasible = all(
                master_counts[request_id] >= self.required_scenarios
                for request_id in master_selected_requests
            )
            oracle_cache[master_ids] = (
                master_ordered_counts, master_feasible
            )
            if master_feasible:
                selected_ids = master_ids
                ordered_counts = master_ordered_counts

        minimum_cardinality = (
            incumbent_count + 1
            if incumbent_count is not None
            else required_count
        )

        with _NominalOracleBatchEvaluator(
            self, snapshot.problem
        ) as batch_oracle:
            for cardinality in (
                range(static_upper_bound, minimum_cardinality - 1, -1)
                if selected_ids is None else ()
            ):
                model = cp_model.CpModel()
                x_vars = tuple(
                    model.NewBoolVar(f"x_{index}")
                    for index in range(len(plans))
                )
                y_vars = {
                    request_id: model.NewBoolVar(
                        f"y_{request_index_local}"
                    )
                    for request_index_local, request_id
                    in enumerate(searchable_request_ids)
                }
                for request_id in searchable_request_ids:
                    model.Add(
                        sum(
                            x_vars[plan_index[plan.plan_id]]
                            for plan in plans_by_request[request_id]
                        ) == y_vars[request_id]
                    )
                for request_id in snapshot.problem.required_requests:
                    model.Add(y_vars[request_id] == 1)
                model.Add(sum(y_vars.values()) == cardinality)
                for coefficients, upper in resource_rows:
                    model.Add(
                        sum(
                            value * x_vars[index]
                            for index, value in coefficients.items()
                        ) <= upper
                    )

                # Candidate order is request/priority/path/order stable.  CP-SAT
                # still uses exactly one deterministic search worker.  Parallel
                # executor results are committed in this callback order, so the
                # selected equal-cardinality assignment is unchanged.
                search_plan_indices = tuple(sorted(
                    range(len(plans)),
                    key=lambda index: (
                        len(plans_by_request[plans[index].request_id]),
                        _nominal_candidate_key(plans[index]),
                    ),
                ))
                model.AddDecisionStrategy(
                    tuple(x_vars[index] for index in search_plan_indices),
                    cp_model.CHOOSE_FIRST,
                    cp_model.SELECT_MAX_VALUE,
                )

                planner = self

                class _ExecutorCallback(cp_model.CpSolverSolutionCallback):
                    def __init__(self) -> None:
                        super().__init__()
                        self.assignment_count = 0
                        self.selected: tuple[str, ...] | None = None
                        self.counts: tuple[tuple[str, int], ...] = ()
                        self.pending: list[tuple[str, ...]] = []

                    def _flush(
                        self,
                        *,
                        stop_solver: bool,
                        allow_pool_start: bool,
                    ) -> None:
                        if not self.pending or self.selected is not None:
                            return
                        pending = tuple(self.pending)
                        self.pending.clear()
                        uncached = tuple(
                            candidate_ids for candidate_ids in pending
                            if candidate_ids not in oracle_cache
                        )
                        evaluated = batch_oracle.evaluate(
                            uncached,
                            allow_pool_start=allow_pool_start,
                        )
                        self.assignment_count += len(uncached)
                        for candidate_ids, result in zip(
                            uncached, evaluated, strict=True
                        ):
                            oracle_cache[candidate_ids] = (
                                result.completion_counts,
                                result.feasible,
                            )

                        # Scan the original CP-SAT order, including cached
                        # assignments.  Work later in this batch may have run
                        # speculatively, but it cannot change the first feasible
                        # assignment or the exact cardinality certificate.
                        for candidate_ids in pending:
                            counts, feasible = oracle_cache[candidate_ids]
                            if not feasible:
                                continue
                            self.selected = candidate_ids
                            self.counts = counts
                            if stop_solver:
                                self.StopSearch()
                            return

                    def flush_tail(self) -> None:
                        self._flush(
                            stop_solver=False,
                            allow_pool_start=(
                                batch_oracle.pool is not None
                                or len(self.pending) >= planner.oracle_workers
                            ),
                        )

                    def on_solution_callback(self) -> None:
                        selected_requests = tuple(
                            request_id
                            for request_id in searchable_request_ids
                            if self.Value(y_vars[request_id])
                        )
                        candidate_ids = tuple(
                            next(
                                plan.plan_id
                                for plan in plans_by_request[request_id]
                                if self.Value(
                                    x_vars[plan_index[plan.plan_id]]
                                )
                            )
                            for request_id in selected_requests
                        )
                        if planner.oracle_workers == 1:
                            cached = oracle_cache.get(candidate_ids)
                            if cached is None:
                                self.assignment_count += 1
                                counts = planner._completion_counts(
                                    snapshot.problem, candidate_ids
                                )
                                cached = (
                                    tuple(sorted(counts.items())),
                                    all(
                                        counts[request_id]
                                        >= planner.required_scenarios
                                        for request_id in selected_requests
                                    ),
                                )
                                oracle_cache[candidate_ids] = cached
                            counts, feasible = cached
                            if feasible:
                                self.selected = candidate_ids
                                self.counts = counts
                                self.StopSearch()
                            return

                        self.pending.append(candidate_ids)
                        if len(self.pending) >= planner.oracle_batch_size:
                            self._flush(
                                stop_solver=True,
                                allow_pool_start=True,
                            )

                callback = _ExecutorCallback()
                solver = cp_model.CpSolver()
                solver.parameters.enumerate_all_solutions = True
                solver.parameters.num_search_workers = 1
                solver.parameters.random_seed = 0
                solver.parameters.search_branching = cp_model.FIXED_SEARCH
                status = solver.Solve(model, callback)
                callback.flush_tail()
                enumerated_assignments += callback.assignment_count
                if callback.selected is not None:
                    selected_ids = callback.selected
                    ordered_counts = callback.counts
                    break
                if status not in (cp_model.OPTIMAL, cp_model.INFEASIBLE):
                    raise RuntimeError(
                        "CP-SAT nominal assignment enumeration did not exhaust "
                        f"cardinality {cardinality}: status={solver.StatusName()}"
                    )

        if selected_ids is None:
            if incumbent_ids is not None:
                selected_ids = incumbent_ids
                ordered_counts = incumbent_counts
            else:
                raise RuntimeError(
                    "no nominally completable assignment satisfies all "
                    "required requests"
                )
        solution = MilpNominalPlanResult(
            selected_plan_ids=selected_ids,
            completed_count=len(selected_ids),
            scenario_completion_counts=ordered_counts,
            planning_seeds=self.planning_seeds,
            required_scenarios=self.required_scenarios,
            evaluations=self.last_evaluations,
            milp_solves=1,
            cuts=0,
            backend=_NOMINAL_BACKEND,
            eligible_candidates=len(eligible),
            filtered_candidates=filtered_candidates,
            enumerated_assignments=enumerated_assignments,
            static_upper_bound=static_upper_bound,
            proven_optimal=True,
        )
        self.last_objective = solution.completed_count
        self.last_solution = solution
        self.last_proven_optimal = True
        return solution.selected_plan_ids


class MilpNominalPathPlanner(_MilpNominalPlanner):
    """Finite-scenario hybrid optimizer for canonical swap orders."""

    allow_swap_orders = False
    name = "milp_nominal_path"


class MilpNominalPathOrderPlanner(_MilpNominalPlanner):
    """Finite-scenario hybrid optimizer over path/order candidates."""

    allow_swap_orders = True
    name = "milp_nominal_path_order"
