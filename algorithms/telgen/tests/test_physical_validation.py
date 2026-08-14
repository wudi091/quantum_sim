import unittest

from algorithms.telgen.physical_validation import (
    compile_selected_schedule,
    evaluate_selected_physics,
)
from algorithms.telgen.time_expansion import expand_construction_candidates
from qnet_core.construction_catalog import build_route_construction_catalogue
from qnet_core.resource_catalog import build_resource_capacities
from qnet_core.spec import EpisodeSpec, PhysicalConfig, RequestSpec


def _episode(request_count: int = 1) -> EpisodeSpec:
    return EpisodeSpec(
        seed=1400,
        nodes=(0, 1),
        edges=((0, 1),),
        requests=tuple(
            RequestSpec(f"r{index}", 0, 1, ttl=4)
            for index in range(request_count)
        ),
        horizon=4,
        physical=PhysicalConfig(
            generation_probability=1.0,
            detector_efficiency=1.0,
            bsm_success_probability=1.0,
            memory_capacity=1,
            node_memory_capacity=1,
            quantum_distance_m=1.0,
            slot_duration_ps=1_000_000,
        ),
    )


def _variables(episode: EpisodeSpec):
    candidates = build_route_construction_catalogue(
        episode.planning,
        candidate_count=1,
        construction_kinds=("balanced",),
        purification_kinds=("none",),
    )
    return expand_construction_candidates(
        episode.planning,
        candidates,
        window_start_slot=0,
        window_end_slot=episode.horizon,
        resource_capacities=build_resource_capacities(episode),
    ).variables


class PhysicalValidationTests(unittest.TestCase):
    def test_selected_plan_compiles_and_completes_in_sequence(self):
        episode = _episode()
        variable = _variables(episode)[0]
        capacities = build_resource_capacities(episode)

        schedule = compile_selected_schedule(
            (variable,),
            ("r0",),
            capacities,
            horizon_slots=episode.horizon,
        )
        result = evaluate_selected_physics(
            episode,
            (variable,),
            capacities,
        )

        self.assertEqual(len(schedule.requests), 1)
        self.assertEqual(result.metrics["completed_requests"], 1.0)
        self.assertEqual(result.metrics["schedule_violation_count"], 0.0)

    def test_compile_rejects_two_capacity_conflicting_requests(self):
        episode = _episode(request_count=2)
        variables = tuple(
            next(
                variable
                for variable in _variables(episode)
                if variable.request_id == request_id
                and variable.start_slot == 0
            )
            for request_id in ("r0", "r1")
        )

        with self.assertRaisesRegex(ValueError, "infeasible"):
            compile_selected_schedule(
                variables,
                ("r0", "r1"),
                build_resource_capacities(episode),
                horizon_slots=episode.horizon,
            )

    def test_physical_seed_override_is_reproducible(self):
        episode = _episode()
        variable = _variables(episode)[0]
        capacities = build_resource_capacities(episode)

        first = evaluate_selected_physics(
            episode,
            (variable,),
            capacities,
            physical_seed=1401,
        )
        second = evaluate_selected_physics(
            episode,
            (variable,),
            capacities,
            physical_seed=1401,
        )

        self.assertEqual(first.event_trace, second.event_trace)
        self.assertEqual(first.settlements, second.settlements)


if __name__ == "__main__":
    unittest.main()
