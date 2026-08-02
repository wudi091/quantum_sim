"""Deterministic execution and peak-memory accounting for (P, C).

Phase 1 deliberately uses con_design.md's single-slot abstraction:

* a generation layer lists elementary links built in parallel;
* each generated link occupies one memory at each endpoint during that
  layer;
* a plan's resource footprint is the per-node *peak* over all generation
  layers;
* concurrent requests in one slot reserve those peak footprints
  additively and must stay within every node's capacity.

The swap dependency tree is still validated and contributes swap depth / makespan,
but it does not create cross-slot state. This isolates the paper claim:
C_seq builds BC and CD in different layers (peak 1 at C), while C_bal
builds them together (peak 2 at C).
"""

from __future__ import annotations

from dataclasses import dataclass

from .plan import ConstructionPlan, elementary_ref, path_edges


@dataclass(frozen=True)
class PlanExecution:
    """Validated execution summary for one construction plan."""

    generation_occupancy: tuple[dict[int, int], ...]
    peak_memory: dict[int, int]
    duration_at_peak: dict[int, int]
    swap_depth: int
    makespan: int
    output_span: tuple[int, int]


def _generation_occupancy(plan: ConstructionPlan) -> tuple[dict[int, int], ...]:
    series: list[dict[int, int]] = []
    seen: set[str] = set()
    for layer in plan.gen_layers:
        occ: dict[int, int] = {}
        for e in layer:
            ref = elementary_ref(e)
            if ref in seen:
                raise ValueError(f"elementary pair generated twice: {ref}")
            seen.add(ref)
            u, v = e
            occ[u] = occ.get(u, 0) + 1
            occ[v] = occ.get(v, 0) + 1
        series.append(occ)
    expected = {elementary_ref(e) for e in plan.elementary_pairs}
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(f"generation layers mismatch: missing={missing}, extra={extra}")
    return tuple(series)


def _validate_swap_dag(plan: ConstructionPlan) -> tuple[int, tuple[int, int]]:
    spans = {elementary_ref(e): e for e in plan.elementary_pairs}
    depths = {ref: 0 for ref in spans}
    consumed: set[str] = set()
    for node in plan.swap_tree:
        if node.output_ref in spans:
            raise ValueError(f"duplicate pair ref: {node.output_ref}")
        if node.left_ref == node.right_ref:
            raise ValueError(f"swap {node.output_ref} cannot consume one pair twice")
        if node.left_ref in consumed or node.right_ref in consumed:
            raise ValueError(f"swap {node.output_ref} reuses a consumed input")
        if node.left_ref not in spans or node.right_ref not in spans:
            raise ValueError(f"swap {node.output_ref} references unavailable input")
        left_span = spans[node.left_ref]
        right_span = spans[node.right_ref]
        if node.middle not in left_span or node.middle not in right_span:
            raise ValueError(f"swap {node.output_ref} inputs do not meet at {node.middle}")
        left_outer = left_span[1] if left_span[0] == node.middle else left_span[0]
        right_outer = right_span[1] if right_span[0] == node.middle else right_span[0]
        derived_span = (left_outer, right_outer)
        if left_outer == right_outer:
            raise ValueError(f"swap {node.output_ref} would produce a self-pair")
        if node.span != derived_span:
            raise ValueError(
                f"swap {node.output_ref} declares span {node.span}, expected {derived_span}"
            )
        consumed.update((node.left_ref, node.right_ref))
        spans[node.output_ref] = derived_span
        depths[node.output_ref] = 1 + max(depths[node.left_ref], depths[node.right_ref])
    if plan.swap_tree:
        output_ref = plan.swap_tree[-1].output_ref
        if plan.output_ref != output_ref:
            raise ValueError(
                f"plan output_ref {plan.output_ref!r} does not match final swap {output_ref!r}"
            )
        return depths[output_ref], spans[output_ref]
    if len(plan.elementary_pairs) != 1:
        raise ValueError("a swap-free plan must contain exactly one elementary pair")
    expected_ref = elementary_ref(plan.elementary_pairs[0])
    if plan.output_ref != expected_ref:
        raise ValueError(
            f"plan output_ref {plan.output_ref!r} does not match elementary pair {expected_ref!r}"
        )
    return 0, (plan.path[0], plan.path[-1])


def simulate_plan(plan: ConstructionPlan) -> PlanExecution:
    """Validate ``plan`` and compute its scalar peak-memory footprint."""
    if len(plan.path) < 2:
        raise ValueError("construction path requires at least two nodes")
    if len(set(plan.path)) != len(plan.path):
        raise ValueError("construction path must be simple")
    expected_pairs = set(path_edges(plan.path))
    actual_pairs = set(plan.elementary_pairs)
    if len(actual_pairs) != len(plan.elementary_pairs):
        raise ValueError("elementary_pairs contains duplicate pair refs")
    if actual_pairs != expected_pairs:
        missing = sorted(expected_pairs - actual_pairs)
        extra = sorted(actual_pairs - expected_pairs)
        raise ValueError(f"plan edges do not match path: missing={missing}, extra={extra}")
    occupancy = _generation_occupancy(plan)
    swap_depth, output_span = _validate_swap_dag(plan)
    if output_span != (plan.path[0], plan.path[-1]):
        raise ValueError(
            f"plan output {output_span} does not complete path "
            f"{(plan.path[0], plan.path[-1])}"
        )

    nodes = set(plan.path)
    for phase in occupancy:
        nodes.update(phase)
    peaks: dict[int, int] = {}
    durations: dict[int, int] = {}
    for node in nodes:
        values = [phase.get(node, 0) for phase in occupancy]
        peak = max(values, default=0)
        peaks[node] = peak
        durations[node] = values.count(peak) if peak > 0 else 0

    # One unit per generation layer plus one unit per swap-tree level.
    return PlanExecution(
        generation_occupancy=occupancy,
        peak_memory=peaks,
        duration_at_peak=durations,
        swap_depth=swap_depth,
        makespan=len(plan.gen_layers) + swap_depth,
        output_span=output_span,
    )


def plan_footprint(execution: PlanExecution) -> dict[str, object]:
    """Serializable resource-footprint view used by demos and planners."""
    nodes = set(execution.peak_memory)
    return {
        "peak": dict(execution.peak_memory),
        "duration_at_peak": dict(execution.duration_at_peak),
        "makespan": execution.makespan,
        "swap_depth": execution.swap_depth,
        "series": {
            node: [phase.get(node, 0) for phase in execution.generation_occupancy]
            for node in nodes
        },
    }


class SlotSimulator:
    """Centralized one-slot admission under additive peak footprints."""

    def __init__(self, capacity: dict[int, int]) -> None:
        self.capacity = dict(capacity)
        self.used: dict[int, int] = {node: 0 for node in capacity}
        self.accepted: list[tuple[ConstructionPlan, PlanExecution]] = []

    def can_admit(self, execution: PlanExecution) -> bool:
        return all(
            self.used.get(node, 0) + cost <= self.capacity.get(node, 0)
            for node, cost in execution.peak_memory.items()
        )

    def admit(self, plan: ConstructionPlan, execution: PlanExecution) -> bool:
        if not self.can_admit(execution):
            return False
        for node, cost in execution.peak_memory.items():
            self.used[node] = self.used.get(node, 0) + cost
        self.accepted.append((plan, execution))
        return True

    def accepted_plans(self) -> list[ConstructionPlan]:
        return [plan for plan, _ in self.accepted]
