import unittest

from qnet_core.workload import resolve_periodic_arrival_workload


class PeriodicArrivalWorkloadTests(unittest.TestCase):
    def test_fixed_arrival_rounds_match_qcast_style_batches(self):
        workload = resolve_periodic_arrival_workload(
            request_count=None,
            arrival_rounds=3,
            requests_per_round=5,
            arrival_interval_slots=4,
            ttl_slots=16,
            horizon_slots=None,
            default_request_count=20,
        )

        self.assertEqual(workload.mode, "fixed_arrival_rounds")
        self.assertEqual(workload.request_count, 15)
        self.assertEqual(workload.arrival_rounds, 3)
        self.assertEqual(workload.last_arrival_slot, 8)
        self.assertEqual(workload.horizon_slots, 24)
        self.assertEqual(workload.drain_slots, 16)
        self.assertEqual(workload.final_round_request_count, 5)
        self.assertAlmostEqual(workload.offered_load_requests_per_slot, 1.25)

    def test_legacy_fixed_count_keeps_a_partial_final_batch(self):
        workload = resolve_periodic_arrival_workload(
            request_count=12,
            arrival_rounds=None,
            requests_per_round=5,
            arrival_interval_slots=4,
            ttl_slots=16,
            horizon_slots=None,
            default_request_count=20,
        )

        self.assertEqual(workload.mode, "fixed_request_count")
        self.assertEqual(workload.arrival_rounds, 3)
        self.assertEqual(workload.final_round_request_count, 2)
        self.assertEqual(workload.horizon_slots, 24)
        self.assertAlmostEqual(workload.offered_load_requests_per_slot, 1.0)

    def test_default_preserves_the_previous_request_count(self):
        workload = resolve_periodic_arrival_workload(
            request_count=None,
            arrival_rounds=None,
            requests_per_round=5,
            arrival_interval_slots=4,
            ttl_slots=16,
            horizon_slots=None,
            default_request_count=20,
        )

        self.assertEqual(workload.mode, "fixed_request_count")
        self.assertEqual(workload.request_count, 20)
        self.assertEqual(workload.arrival_rounds, 4)

    def test_rejects_ambiguous_or_too_short_workloads(self):
        with self.assertRaisesRegex(ValueError, "either"):
            resolve_periodic_arrival_workload(
                request_count=20,
                arrival_rounds=4,
                requests_per_round=5,
                arrival_interval_slots=4,
                ttl_slots=16,
                horizon_slots=None,
                default_request_count=20,
            )
        with self.assertRaisesRegex(ValueError, "final arrival"):
            resolve_periodic_arrival_workload(
                request_count=None,
                arrival_rounds=4,
                requests_per_round=5,
                arrival_interval_slots=4,
                ttl_slots=16,
                horizon_slots=27,
                default_request_count=20,
            )


if __name__ == "__main__":
    unittest.main()
