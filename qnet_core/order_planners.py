"""Planning-only baselines for the shared order-aware batch core."""

from __future__ import annotations

from collections import defaultdict
from itertools import product

from .order_core import (
    OrderBatchSnapshot,
    OrderPlan,
    simulate_order_batch,
)
from .qcast_paper.ext import expected_throughput


class QDDCAFixedOrderPlanner:
    """Path-only Q-DDCA-style adapter with one canonical swap order.

    Q-DDCA does not expose a complete swap-order action.  This adapter keeps
    that boundary explicit: it chooses one deterministic reference-schedule
    candidate per path using path length and a local static congestion
    estimate, then leaves every physical event to the shared core.
    """

    name = "qddca_fixed"

    def reset(self, episode_seed: int) -> None:
        del episode_seed

    @staticmethod
    def _score(
        plan: OrderPlan,
        fixed_candidates: tuple[OrderPlan, ...],
        capacity: dict[object, int],
    ) -> tuple[float, int, str]:
        demand: dict[object, int] = defaultdict(int)
        for candidate in fixed_candidates:
            for node in candidate.path[1:-1]:
                demand[node] += 2
        congestion = sum(
            demand[node] / capacity[node] for node in plan.path[1:-1]
        )
        # Q-DDCA's route metric remains primary; congestion breaks equal-hop
        # path ties in this centralized one-slot adapter.
        return len(plan.path) - 1 + 1e-3 * congestion, len(plan.path), plan.plan_id

    def select(self, snapshot: OrderBatchSnapshot) -> tuple[str, ...]:
        fixed = tuple(
            plan for plan in snapshot.candidates if plan.is_fixed_order
        )
        capacity = snapshot.problem.capacity
        by_request: dict[str, list[OrderPlan]] = defaultdict(list)
        for plan in fixed:
            by_request[plan.request_id].append(plan)
        selected: list[OrderPlan] = []
        for _, candidates in sorted(
            by_request.items(),
            key=lambda item: (
                min(plan.priority for plan in item[1]), item[0]
            ),
        ):
            selected.append(min(
                candidates,
                key=lambda plan: self._score(plan, fixed, capacity),
            ))
        missing = snapshot.problem.required_requests - {
            plan.request_id for plan in selected
        }
        if missing:
            raise RuntimeError(
                f"Q-DDCA fixed catalogue misses required requests: {sorted(missing)}"
            )
        return tuple(plan.plan_id for plan in selected)


class QCASTFixedOrderPlanner:
    """Width-1 Q-CAST routing adapter with one canonical swap order.

    The order-aware catalogue does not expose Q-CAST width allocation or
    recovery paths.  This adapter therefore ports only Q-CAST's EXT path
    ranking, using the verified paper-compatibility equation already present
    in :mod:`qnet_core.qcast_paper.ext`.  Every selected path uses the same
    deterministic reference schedule as other path-only baselines, and all HEG,
    swapping, arbitration, and memory reuse remain inside the shared core.
    """

    name = "qcast_fixed"

    def reset(self, episode_seed: int) -> None:
        del episode_seed

    @staticmethod
    def _score(
        plan: OrderPlan,
        snapshot: OrderBatchSnapshot,
    ) -> tuple[float, int, int, str]:
        hops = len(plan.path) - 1
        config = snapshot.problem.config
        ext = expected_throughput(
            tuple(
                snapshot.problem.link_generation_probability(elementary_edge)
                for elementary_edge in plan.elementary_edges
            ),
            width=1,
            swap_probability=config.swap_probability,
        )
        # One width-1 path reserves one endpoint memory and two memories at
        # each internal repeater in Q-CAST's static path abstraction.
        memory_cost = 2 + 2 * max(0, len(plan.path) - 2)
        return -ext, memory_cost, hops, plan.plan_id

    def select(self, snapshot: OrderBatchSnapshot) -> tuple[str, ...]:
        fixed = tuple(
            plan for plan in snapshot.candidates if plan.is_fixed_order
        )
        by_request: dict[str, list[OrderPlan]] = defaultdict(list)
        for plan in fixed:
            by_request[plan.request_id].append(plan)

        selected = tuple(
            min(
                candidates,
                key=lambda plan: self._score(plan, snapshot),
            )
            for _, candidates in sorted(
                by_request.items(),
                key=lambda item: (
                    min(plan.priority for plan in item[1]), item[0]
                ),
            )
        )
        missing = snapshot.problem.required_requests - {
            plan.request_id for plan in selected
        }
        if missing:
            raise RuntimeError(
                "Q-CAST fixed catalogue misses required requests: "
                f"{sorted(missing)}"
            )
        return tuple(plan.plan_id for plan in selected)


class SampleAverageOrderBatchPlanner:
    """Exhaustive optimizer of a finite sample-average objective.

    ``rollout_seeds`` are planner-owned common random scenarios.  They never
    include or reveal the environment's realized physics seed.  The search is
    exact for this finite SAA sample, but it is not a true stochastic optimum
    and it is not allowed to see the test realization.
    """

    def __init__(
        self,
        *,
        allow_swap_orders: bool,
        rollout_seeds: tuple[int, ...] = (0,),
    ):
        self.allow_swap_orders = bool(allow_swap_orders)
        if not rollout_seeds:
            raise ValueError("rollout_seeds cannot be empty")
        self.rollout_seeds = tuple(map(int, rollout_seeds))
        self.last_evaluations = 0

    @property
    def name(self) -> str:
        return "saa_path_order" if self.allow_swap_orders else "saa_path"

    def reset(self, episode_seed: int) -> None:
        del episode_seed
        self.last_evaluations = 0

    def select(self, snapshot: OrderBatchSnapshot) -> tuple[str, ...]:
        eligible = tuple(
            plan for plan in snapshot.candidates
            if self.allow_swap_orders or plan.is_fixed_order
        )
        by_request: dict[str, list[OrderPlan]] = defaultdict(list)
        for plan in eligible:
            by_request[plan.request_id].append(plan)
        request_ids = tuple(sorted(
            by_request,
            key=lambda request_id: (
                min(plan.priority for plan in by_request[request_id]),
                request_id,
            ),
        ))
        missing = snapshot.problem.required_requests - set(request_ids)
        if missing:
            raise RuntimeError(
                f"SAA catalogue misses required requests: {sorted(missing)}"
            )

        choice_sets: list[tuple[OrderPlan | None, ...]] = []
        for request_id in request_ids:
            plans = tuple(sorted(
                by_request[request_id],
                key=lambda plan: (
                    len(plan.path),
                    tuple(map(repr, plan.path)),
                    plan.schedule_key,
                    plan.plan_id,
                ),
            ))
            if request_id in snapshot.problem.required_requests:
                choice_sets.append(plans)
            else:
                choice_sets.append((None, *plans))

        best_ids: tuple[str, ...] | None = None
        best_score: tuple[int, int, int, int, int] | None = None
        self.last_evaluations = 0
        for choices in product(*choice_sets):
            plans = tuple(plan for plan in choices if plan is not None)
            plan_ids = tuple(plan.plan_id for plan in plans)
            if (best_score is not None
                    and len(plans) * len(self.rollout_seeds) < best_score[0]):
                continue
            priorities = {plan.request_id: plan.priority for plan in plans}
            completed_count = 0
            completion_sum = 0
            completed_priority = 0
            failed_count = 0
            for rollout_seed in self.rollout_seeds:
                result = simulate_order_batch(
                    snapshot.problem.with_physics_seed(rollout_seed),
                    plan_ids,
                    record_traces=False,
                )
                self.last_evaluations += 1
                completed_count += result.completed_count
                completion_sum += sum(result.completion_time_ps.values())
                completed_priority += sum(
                    priorities[request_id] for request_id in result.completed
                )
                failed_count += len(result.failed)
            # Strict lexicographic objective:
            # completed requests, higher-priority completions, earlier
            # completion, then keep more executable requests.  A request that
            # happens to miss in the finite rollout set is not treated as an
            # admission penalty because the research objective is completion
            # count, not minimizing the number of selected plans.
            score = (
                completed_count,
                -completed_priority,
                -completion_sum,
                len(plans),
                -failed_count,
            )
            if (best_score is None or score > best_score
                    or (score == best_score
                        and (best_ids is None or plan_ids < best_ids))):
                best_score = score
                best_ids = plan_ids
        if best_ids is None:
            raise RuntimeError("SAA order planner found no feasible selection")
        return best_ids


class SAAPathPlanner(SampleAverageOrderBatchPlanner):
    def __init__(self, rollout_seeds: tuple[int, ...] = (0,)) -> None:
        super().__init__(
            allow_swap_orders=False,
            rollout_seeds=rollout_seeds,
        )


class SAAPathOrderPlanner(SampleAverageOrderBatchPlanner):
    def __init__(self, rollout_seeds: tuple[int, ...] = (0,)) -> None:
        super().__init__(
            allow_swap_orders=True,
            rollout_seeds=rollout_seeds,
        )


# Compatibility imports for older scripts/checkpoints.  These aliases retain
# the corrected ``saa_*`` planner names and must not be presented as stochastic
# optima in reports.
ExactOrderBatchPlanner = SampleAverageOrderBatchPlanner
OptimalPathPlanner = SAAPathPlanner
OptimalPathOrderPlanner = SAAPathOrderPlanner
