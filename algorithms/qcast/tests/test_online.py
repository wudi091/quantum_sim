import unittest

from algorithms.qcast.online import OnlineQCASTConfig, run_online_qcast
from algorithms.qcast.online_planner import plan_qcast_window
from qnet_core.resource_catalog import build_resource_capacities
from qnet_core.planning_spec import RequestSpec
from qnet_core.spec import EpisodeSpec, PhysicalConfig


def deterministic_physical(**overrides):
    values = dict(
        generation_probability=1.0,
        swap_probability=1.0,
        detector_efficiency=1.0,
        bsm_success_probability=1.0,
        quantum_distance_m=1.0,
        slot_duration_ps=1_000_000,
    )
    values.update(overrides)
    return PhysicalConfig(**values)


class OnlineQCASTTests(unittest.TestCase):
    def test_qcast_window_honors_the_declared_request_subset(self):
        spec = EpisodeSpec(
            seed=1299,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(
                RequestSpec("r0", 0, 1, ttl=4),
                RequestSpec("r1", 0, 1, ttl=4),
            ),
            horizon=4,
            physical=deterministic_physical(node_memory_capacity=4),
        )
        record = plan_qcast_window(
            spec,
            window_start_slot=0,
            window_end_slot=4,
            request_ids=("r0",),
            path_candidate_count=1,
        )
        self.assertEqual(
            {candidate.request_id for candidate in record.candidates},
            {"r0"},
        )
        self.assertEqual(
            {variable.request_id for variable in record.solution.selected_variables},
            {"r0"},
        )

    def test_zero_ext_path_is_not_admitted(self):
        spec = EpisodeSpec(
            seed=1298,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r0", 0, 2, ttl=4),),
            horizon=4,
            physical=deterministic_physical(swap_probability=0.0),
        )
        record = plan_qcast_window(
            spec,
            window_start_slot=0,
            window_end_slot=4,
            path_candidate_count=1,
        )
        self.assertEqual(record.solution.completed_request_count, 0)
        self.assertEqual(record.solution.selected_variables, ())

    def test_qcast_ext_prefers_the_shorter_path(self):
        spec = EpisodeSpec(
            seed=1300,
            nodes=(0, 1, 3),
            edges=((0, 3), (0, 1), (1, 3)),
            requests=(RequestSpec("r0", 0, 3, ttl=6),),
            horizon=6,
            physical=deterministic_physical(
                generation_probability=0.8,
                swap_probability=0.9,
            ),
        )
        record = plan_qcast_window(
            spec,
            window_start_slot=0,
            window_end_slot=6,
            path_candidate_count=2,
        )
        self.assertEqual(record.solution.completed_request_count, 1)
        self.assertEqual(
            record.solution.selected_variables[0].route_nodes,
            (0, 3),
        )

    def test_qcast_moves_to_the_earliest_unreserved_start(self):
        spec = EpisodeSpec(
            seed=1301,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(RequestSpec("r0", 0, 1, ttl=4),),
            horizon=4,
            physical=deterministic_physical(),
        )
        capacities = build_resource_capacities(spec)
        first = plan_qcast_window(
            spec,
            window_start_slot=0,
            window_end_slot=4,
            resource_capacities=capacities,
            path_candidate_count=1,
        )
        initial = first.solution.selected_variables[0]
        self.assertEqual(initial.start_slot, 0)
        reserved = {
            (item.resource_id, item.slot): capacities[item.resource_id]
            for item in initial.resource_usage
        }
        shifted = plan_qcast_window(
            spec,
            window_start_slot=0,
            window_end_slot=4,
            resource_capacities=capacities,
            reserved_usage=reserved,
            path_candidate_count=1,
        )
        self.assertEqual(shifted.solution.selected_variables[0].start_slot, 1)

    def test_online_qcast_hides_future_requests(self):
        spec = EpisodeSpec(
            seed=1302,
            nodes=(0, 1),
            edges=((0, 1),),
            requests=(
                RequestSpec("r0", 0, 1, arrival=0, ttl=8),
                RequestSpec("r1", 0, 1, arrival=3, ttl=5),
            ),
            horizon=8,
            physical=deterministic_physical(node_memory_capacity=4),
        )
        result = run_online_qcast(
            spec,
            OnlineQCASTConfig(
                decision_interval=4,
                path_candidate_count=1,
            ),
        )
        self.assertEqual(
            [decision.decision_slot for decision in result.decisions],
            [0, 4],
        )
        self.assertEqual(result.decisions[0].visible_request_ids, ("r0",))
        self.assertNotIn("r1", result.decisions[0].eligible_request_ids)
        self.assertTrue(all(
            decision.decision_seconds >= decision.planner_seconds
            for decision in result.decisions
        ))
        self.assertEqual(result.metrics["completed_requests"], 2.0)

    def test_qcast_planning_time_includes_rejected_windows(self):
        spec = EpisodeSpec(
            seed=1304,
            nodes=(0, 1, 2),
            edges=((0, 1), (1, 2)),
            requests=(RequestSpec("r0", 0, 2, arrival=0, ttl=1),),
            horizon=2,
            physical=deterministic_physical(),
        )
        result = run_online_qcast(
            spec,
            OnlineQCASTConfig(
                decision_interval=1,
                path_candidate_count=1,
            ),
        )
        planner_calls = [
            decision.planner_seconds
            for decision in result.decisions
            if decision.eligible_request_ids
        ]
        self.assertEqual(result.attempts, ())
        self.assertTrue(planner_calls)
        self.assertAlmostEqual(
            result.metrics["mean_qcast_planning_seconds"],
            sum(planner_calls) / len(planner_calls),
        )
        self.assertGreater(
            result.metrics["mean_qcast_planning_seconds"],
            0.0,
        )

if __name__ == "__main__":
    unittest.main()
