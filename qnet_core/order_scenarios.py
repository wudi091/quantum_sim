"""Deterministic hotspot workloads for isolating swap-order memory release."""

from __future__ import annotations

from itertools import permutations
import random

from .order_core import OrderBatchProblem, OrderCoreConfig, OrderPlan


def make_order_counterexample(
    *,
    hotspot_capacity: int = 2,
    seed: int = 7,
    generation_probability: float = 1.0,
    swap_probability: float = 1.0,
) -> OrderBatchProblem:
    """The three-request C-hotspot example on one event-driven control slot."""

    main_path = ("A", "B", "C", "D", "E")
    candidates = [
        OrderPlan(
            plan_id="R1:" + "-".join(order),
            request_id="R1",
            path=main_path,
            swap_order=order,
            priority=0,
        )
        for order in permutations(main_path[1:-1])
    ]
    candidates.extend((
        OrderPlan("R2:C", "R2", ("X", "C", "Y"), ("C",), priority=1),
        OrderPlan("R3:C", "R3", ("U", "C", "V"), ("C",), priority=2),
    ))
    nodes = {node for plan in candidates for node in plan.path}
    capacity = {node: 2 for node in nodes}
    capacity["C"] = hotspot_capacity
    return OrderBatchProblem.create(
        candidates=candidates,
        node_capacity=capacity,
        config=OrderCoreConfig(
            slot_duration_ps=3_000,
            generation_interval_ps=1_000,
            swap_service_ps=1_000,
            memory_reset_ps=100,
            generation_probability=generation_probability,
            swap_probability=swap_probability,
            edge_capacity=1,
            bsm_capacity_per_node=1,
            seed=seed,
        ),
        required_requests=("R1",),
        preloaded_requests=("R1",),
        name=f"counterexample-MC{hotspot_capacity}",
    )


def make_seeded_hotspot_problem(
    seed: int,
    *,
    hotspot_capacity: int = 2,
    generation_probability: float = 1.0,
    swap_probability: float = 1.0,
    physics_seed: int | None = None,
) -> OrderBatchProblem:
    """Generate one small exact-oracle motif with varied path/hotspot depth.

    The main request starts with its elementary EPRs ready.  Its canonical
    left-to-right order releases a randomly positioned internal hotspot late;
    all other requests need two memories at that hotspot.  No alternate path
    is supplied, intentionally isolating the value of swap-order choice from
    path choice.
    """

    rng = random.Random(seed)
    internal_count = rng.randint(3, 5)
    internal = tuple(f"s{seed}-I{index}" for index in range(internal_count))
    main_path = (f"s{seed}-A", *internal, f"s{seed}-Z")
    # Exclude the first internal node so the canonical order is not already
    # hotspot-first.  Different seeds vary how late the release occurs.
    hotspot = internal[rng.randrange(1, internal_count)]
    main_request = f"s{seed}-main"
    candidates = [
        OrderPlan(
            plan_id=f"{main_request}:" + "-".join(map(str, order)),
            request_id=main_request,
            path=main_path,
            swap_order=order,
            priority=0,
        )
        for order in permutations(internal)
    ]
    waiting_count = internal_count - 1
    for index in range(waiting_count):
        request_id = f"s{seed}-wait{index}"
        left = f"s{seed}-L{index}"
        right = f"s{seed}-R{index}"
        candidates.append(OrderPlan(
            plan_id=f"{request_id}:hotspot",
            request_id=request_id,
            path=(left, hotspot, right),
            swap_order=(hotspot,),
            priority=index + 1,
        ))

    nodes = {node for plan in candidates for node in plan.path}
    capacity = {node: 2 for node in nodes}
    capacity[hotspot] = hotspot_capacity
    return OrderBatchProblem.create(
        candidates=candidates,
        node_capacity=capacity,
        config=OrderCoreConfig(
            slot_duration_ps=internal_count * 1_000,
            generation_interval_ps=1_000,
            swap_service_ps=1_000,
            memory_reset_ps=100,
            generation_probability=generation_probability,
            swap_probability=swap_probability,
            edge_capacity=1,
            bsm_capacity_per_node=1,
            seed=seed if physics_seed is None else int(physics_seed),
        ),
        required_requests=(main_request,),
        preloaded_requests=(main_request,),
        name=(
            f"seeded-hotspot-s{seed}-m{internal_count}-"
            f"h{internal.index(hotspot)}-MC{hotspot_capacity}"
        ),
    )
