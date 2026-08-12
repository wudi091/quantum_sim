import unittest

from algorithms.telgen import (
    HardConstraintDecoder,
    candidate_fidelity_estimate_map,
    compile_decoded_schedule,
    evaluate_decoded_physics,
    expand_construction_candidates,
    solve_teacher_episode,
    validate_decoded_physics,
)
from qnet_core.construction_catalog import build_route_construction_catalogue
from qnet_core.planning_spec import RequestSpec
from qnet_core.resource_catalog import build_resource_capacities
from qnet_core.spec import EpisodeSpec, PhysicalConfig


class PhysicalValidationTests(unittest.TestCase):
    def test_teacher_selected_purification_executes_in_sequence(self):
        episode = EpisodeSpec(
            seed=2,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec(
                "r", 0, 1, required_fidelity=0.82
            ),),
            horizon=8,
            physical=PhysicalConfig(
                initial_fidelity=0.8,
                swap_degradation=1.0,
                generation_probability=1.0,
                swap_probability=1.0,
                memory_capacity=2,
                node_memory_capacity=2,
                memory_lifetime=1000,
                quantum_distance_m=1.0,
            ),
        )
        record = solve_teacher_episode(
            episode,
            path_candidate_count=1,
            construction_kinds=("balanced",),
        )
        decoded = HardConstraintDecoder(random_restarts=0).decode(
            record.expansion,
            record.capacities,
            record.solution.stage_two.primal,
        )

        result = evaluate_decoded_physics(episode, decoded)

        self.assertEqual(
            decoded.selected_variables[0].purification_kind,
            "elementary_once",
        )
        self.assertEqual(result.metrics["completed_requests"], 1.0)
        self.assertEqual(result.metrics["schedule_adherence"], 1.0)
        purification = next(
            event for event in result.event_trace
            if event.event_kind == "purify"
        )
        self.assertTrue(purification.success)
        self.assertGreaterEqual(purification.output_fidelity or 0.0, 0.82)

    @staticmethod
    def _episode() -> EpisodeSpec:
        return EpisodeSpec(
            seed=1300,
            nodes=(0, 1, 2, 3, 4),
            edges=((0, 1), (1, 2), (2, 3), (3, 4)),
            requests=(RequestSpec("r0", 0, 4, ttl=4),),
            horizon=4,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                detector_efficiency=1.0,
                bsm_success_probability=1.0,
                quantum_distance_m=1.0,
                node_memory_capacity=4,
            ),
        )

    def _decoded(self):
        episode = self._episode()
        capacities = build_resource_capacities(episode)
        candidates = build_route_construction_catalogue(
            episode.planning,
            candidate_count=1,
            construction_kinds=("balanced",),
        )
        expansion = expand_construction_candidates(
            episode.planning,
            candidates,
            capacities,
        )
        scores = {
            variable.variable_id: float(variable.start_slot == 0)
            for variable in expansion.variables
        }
        decoded = HardConstraintDecoder(
            beam_width=32,
            random_restarts=8,
        ).decode(
            expansion,
            capacities,
            scores,
            request_ids=("r0",),
        )
        return episode, decoded

    def test_compiler_preserves_absolute_operation_slots(self):
        episode, decoded = self._decoded()

        schedule = compile_decoded_schedule(
            decoded,
            horizon_slots=episode.horizon,
        )

        selected = decoded.selected_variables[0]
        planned = schedule.requests[0]
        self.assertEqual(planned.start_slot, selected.start_slot)
        self.assertEqual(planned.completion_slot, selected.completion_slot)
        self.assertEqual(
            planned.operation_slots,
            tuple(sorted(
                (operation_id, selected.start_slot + relative_slot)
                for operation_id, relative_slot
                in selected.nominal_schedule.operation_slots
            )),
        )

    def test_sequence_serialization_stays_inside_the_coarse_swap_slot(self):
        episode, decoded = self._decoded()

        result = evaluate_decoded_physics(episode, decoded)

        self.assertEqual(result.metrics["completed_requests"], 1.0)
        self.assertEqual(result.metrics["schedule_adherence"], 1.0)
        self.assertEqual(result.violations, ())
        first_level_swaps = [
            launch
            for launch in result.launches
            if launch.planned_slot == 1 and ":swap:" in launch.operation_id
        ]
        self.assertEqual(len(first_level_swaps), 2)
        self.assertEqual(len({item.actual_time_ps for item in first_level_swaps}), 2)
        self.assertTrue(all(
            episode.physical.slot_duration_ps
            <= item.actual_time_ps
            < 2 * episode.physical.slot_duration_ps
            for item in first_level_swaps
        ))

    def test_repeated_physics_reports_retention_and_schedule_adherence(self):
        episode, decoded = self._decoded()

        report = validate_decoded_physics(
            episode,
            decoded,
            physical_seeds=(1301, 1302),
        )

        self.assertEqual(report.mean_completed_requests, 1.0)
        self.assertEqual(report.mean_completion_retention, 1.0)
        self.assertEqual(report.schedule_adherence_rate, 1.0)

    def test_conservative_fidelity_bound_is_below_independent_sequence_result(self):
        episode = EpisodeSpec(
            seed=1400,
            nodes=(0, 1, 2, 3, 4, 5),
            edges=((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)),
            requests=(RequestSpec(
                "r0", 0, 5, ttl=6, required_fidelity=0.5
            ),),
            horizon=6,
            physical=PhysicalConfig(
                generation_probability=1.0,
                swap_probability=1.0,
                detector_efficiency=1.0,
                bsm_success_probability=1.0,
                quantum_distance_m=1.0,
            ),
        )
        capacities = build_resource_capacities(episode)
        candidates = build_route_construction_catalogue(
            episode.planning,
            candidate_count=1,
            construction_kinds=("balanced",),
        )
        estimates = candidate_fidelity_estimate_map(episode, candidates)
        expansion = expand_construction_candidates(
            episode.planning,
            candidates,
            capacities,
        )
        decoded = HardConstraintDecoder(
            beam_width=16,
            random_restarts=0,
        ).decode(
            expansion,
            capacities,
            {
                variable.variable_id: float(variable.start_slot == 0)
                for variable in expansion.variables
            },
            request_ids=("r0",),
        )

        result = evaluate_decoded_physics(episode, decoded, physical_seed=1401)
        terminal_ids = set(candidates[0].all_terminal_segment_ids)
        observed = next(
            event.output_fidelity
            for event in result.event_trace
            if event.output_segment_id in terminal_ids
        )
        assert observed is not None
        estimated = estimates[candidates[0].candidate_id]
        self.assertLessEqual(estimated, observed + 1e-12)
        self.assertLess(observed - estimated, 0.03)


if __name__ == "__main__":
    unittest.main()
