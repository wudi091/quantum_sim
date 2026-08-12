import unittest

from algorithms.telgen import (
    build_nominal_schedule,
    expand_construction_candidates,
)
from qnet_core.construction_api import (
    ConstructionDAG,
    ConstructionOperation,
    OperationKind,
)
from qnet_core.construction_catalog import (
    RouteConstructionCandidate,
    build_route_construction_catalogue,
)
from qnet_core.planning_spec import PlanningSpec, RequestSpec


def capacities(spec, *, memory=4):
    result = {}
    for raw_u, raw_v in spec.edges:
        u, v = sorted((raw_u, raw_v))
        result[f"link:{u}-{v}"] = 2
        result[f"genlane:{u}-{v}"] = 1
        result[f"purify:{u}-{v}"] = 1
    for node in spec.nodes:
        result[f"bsm:{node}"] = 1
        result[f"memory:{node}"] = memory
    return result


class TimeExpansionTests(unittest.TestCase):
    def path_spec(self, *, arrival=0, ttl=8, horizon=8):
        return PlanningSpec(
            seed=1,
            nodes=(0, 1, 2, 3, 4),
            edges=((0, 1), (1, 2), (2, 3), (3, 4)),
            requests=(RequestSpec("r0", 0, 4, arrival=arrival, ttl=ttl),),
            horizon=horizon,
        )

    def test_balanced_tree_finishes_before_left_deep_tree(self):
        spec = self.path_spec()
        candidates = build_route_construction_catalogue(
            spec, candidate_count=1
        )
        schedules = {
            candidate.construction_kind: build_nominal_schedule(candidate)
            for candidate in candidates
        }
        self.assertEqual(schedules["balanced"].duration_slots, 3)
        self.assertEqual(schedules["left_deep"].duration_slots, 4)
        balanced_swaps = {
            slot
            for op_id, slot in schedules["balanced"].operation_slots
            if ":swap:" in op_id
        }
        self.assertEqual(balanced_swaps, {1, 2})

    def test_generation_holds_memory_until_the_swap_round(self):
        spec = PlanningSpec(
            seed=2,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r0", 0, 2, ttl=4),),
            horizon=4,
        )
        candidate = build_route_construction_catalogue(
            spec, candidate_count=1, construction_kinds=("balanced",)
        )[0]
        schedule = build_nominal_schedule(candidate)
        usage = {
            (item.resource_id, item.slot): item.amount
            for item in schedule.resource_usage
        }
        self.assertEqual(usage[("memory:1", 0)], 2)
        self.assertEqual(usage[("memory:1", 1)], 2)
        self.assertEqual(usage[("bsm:1", 1)], 1)

    def test_disjoint_swap_nodes_use_independent_slot_resources(self):
        spec = PlanningSpec(
            seed=20,
            nodes=(0, 1, 2, 3, 4, 5),
            edges=((0, 1), (1, 2), (3, 4), (4, 5)),
            requests=(
                RequestSpec("r0", 0, 2, ttl=3),
                RequestSpec("r1", 3, 5, ttl=3),
            ),
            horizon=3,
        )
        candidates = build_route_construction_catalogue(
            spec,
            candidate_count=1,
            construction_kinds=("balanced",),
        )
        schedules = {
            item.request_id: build_nominal_schedule(item)
            for item in candidates
        }
        first = {
            (item.resource_id, item.slot): item.amount
            for item in schedules["r0"].resource_usage
        }
        second = {
            (item.resource_id, item.slot): item.amount
            for item in schedules["r1"].resource_usage
        }
        self.assertEqual(first[("bsm:1", 1)], 1)
        self.assertEqual(second[("bsm:4", 1)], 1)
        self.assertNotIn(("bsm:4", 1), first)
        self.assertNotIn(("bsm:1", 1), second)
        self.assertFalse(any(
            item.resource_id == "protocol:swap:global"
            for schedule in schedules.values()
            for item in schedule.resource_usage
        ))

    def test_input_segments_create_dependencies_even_if_predecessors_are_omitted(self):
        operations = (
            ConstructionOperation(
                op_id="r0:g0",
                request_id="r0",
                kind=OperationKind.GEN,
                output_segment_id="s0",
                output_endpoints=(0, 1),
                ordinal=0,
            ),
            ConstructionOperation(
                op_id="r0:g1",
                request_id="r0",
                kind=OperationKind.GEN,
                output_segment_id="s1",
                output_endpoints=(1, 2),
                ordinal=1,
            ),
            ConstructionOperation(
                op_id="r0:swap",
                request_id="r0",
                kind=OperationKind.SWAP,
                input_segment_ids=("s0", "s1"),
                output_segment_id="terminal",
                output_endpoints=(0, 2),
                ordinal=2,
            ),
        )
        candidate = RouteConstructionCandidate(
            candidate_id="r0:manual",
            request_id="r0",
            route_nodes=(0, 1, 2),
            construction_kind="manual",
            dag=ConstructionDAG("r0", operations),
            terminal_segment_id="terminal",
        )
        schedule = build_nominal_schedule(candidate)
        slots = dict(schedule.operation_slots)
        self.assertEqual(slots["r0:g0"], 0)
        self.assertEqual(slots["r0:g1"], 0)
        self.assertEqual(slots["r0:swap"], 1)

    def test_expansion_shifts_only_within_arrival_and_deadline(self):
        spec = self.path_spec(arrival=2, ttl=4, horizon=8)
        candidates = build_route_construction_catalogue(
            spec, candidate_count=1
        )
        expanded = expand_construction_candidates(
            spec, candidates, capacities(spec)
        )
        starts = {}
        for item in expanded.variables:
            starts.setdefault(item.construction_kind, []).append(item.start_slot)
            self.assertLessEqual(item.completion_slot, 6)
        self.assertEqual(starts["balanced"], [2, 3])
        self.assertEqual(starts["left_deep"], [2])

    def test_optional_fidelity_estimates_filter_candidates(self):
        spec = self.path_spec()
        candidates = build_route_construction_catalogue(
            spec, candidate_count=1
        )
        estimates = {
            candidate.candidate_id: (
                0.60 if candidate.construction_kind == "left_deep" else 0.90
            )
            for candidate in candidates
        }
        expanded = expand_construction_candidates(
            spec,
            candidates,
            capacities(spec),
            fidelity_estimates=estimates,
        )
        self.assertTrue(expanded.variables)
        self.assertEqual(
            {item.construction_kind for item in expanded.variables},
            {"balanced"},
        )
        self.assertEqual(len(expanded.rejections), 1)
        self.assertEqual(expanded.rejections[0].reason, "fidelity")

    def test_purification_is_pruned_when_the_unpurified_path_meets_threshold(self):
        spec = PlanningSpec(
            seed=3,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec(
                "r0", 0, 1, ttl=6, required_fidelity=0.75
            ),),
            horizon=6,
        )
        candidates = build_route_construction_catalogue(
            spec,
            candidate_count=1,
            construction_kinds=("balanced",),
            purification_kinds=("none", "elementary_once"),
        )
        estimates = {
            candidate.candidate_id: (
                0.80 if candidate.purification_kind == "none" else 0.85
            )
            for candidate in candidates
        }

        expanded = expand_construction_candidates(
            spec,
            candidates,
            capacities(spec, memory=2),
            fidelity_estimates=estimates,
        )

        self.assertEqual(
            {item.purification_kind for item in expanded.variables},
            {"none"},
        )
        self.assertIn(
            "purification_unnecessary",
            {item.reason for item in expanded.rejections},
        )

    def test_missing_capacity_is_a_model_error(self):
        spec = self.path_spec()
        candidate = build_route_construction_catalogue(
            spec, candidate_count=1, construction_kinds=("balanced",)
        )[0]
        incomplete = capacities(spec)
        del incomplete["bsm:2"]
        with self.assertRaisesRegex(ValueError, "missing capacity"):
            expand_construction_candidates(spec, (candidate,), incomplete)

    def test_online_reservation_removes_only_conflicting_start_slots(self):
        spec = PlanningSpec(
            seed=4,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r0", 0, 1, arrival=2, ttl=4),),
            horizon=8,
        )
        candidate = build_route_construction_catalogue(
            spec,
            candidate_count=1,
            construction_kinds=("balanced",),
        )[0]
        expanded = expand_construction_candidates(
            spec,
            (candidate,),
            capacities(spec),
            reserved_usage={("genlane:0-1", 2): 1},
            window_start_slot=2,
            window_end_slot=6,
        )
        self.assertEqual(
            [item.start_slot for item in expanded.variables],
            [3, 4, 5],
        )

    def test_start_window_and_completion_boundary_are_independent(self):
        spec = self.path_spec(arrival=0, ttl=8, horizon=8)
        balanced = next(
            candidate
            for candidate in build_route_construction_catalogue(
                spec,
                candidate_count=1,
                construction_kinds=("balanced",),
            )
        )
        expanded = expand_construction_candidates(
            spec,
            (balanced,),
            capacities(spec),
            window_start_slot=0,
            window_end_slot=2,
            completion_end_slot=8,
        )

        self.assertEqual(
            [item.start_slot for item in expanded.variables],
            [0, 1],
        )
        self.assertTrue(any(
            item.completion_slot > 2
            for item in expanded.variables
        ))


if __name__ == "__main__":
    unittest.main()
