import unittest

from qnet_core.construction_api import OperationKind
from qnet_core.sequence_protocol_arbiter import (
    ProtocolRequest,
    SequenceProtocolArbiter,
)


class SequenceProtocolArbiterTests(unittest.TestCase):
    @staticmethod
    def _request(
        operation_id: str,
        kind: str,
        nodes: tuple[int, ...],
        inputs: tuple[str, ...] = (),
    ) -> ProtocolRequest:
        return ProtocolRequest(
            operation_id,
            kind,
            frozenset(nodes),
            inputs,
        )

    @staticmethod
    def _conservative() -> SequenceProtocolArbiter:
        return SequenceProtocolArbiter(
            supports_inter_epoch_launch=True,
            supports_mixed_operation_concurrency=False,
            supports_concurrent_swaps=False,
        )

    def test_conservative_arbiter_rejects_mixed_launch(self):
        arbiter = self._conservative()
        result = arbiter.validate((
            self._request("g", OperationKind.GEN, (0, 1)),
            self._request("s", OperationKind.SWAP, (1, 2, 3), ("a", "b")),
        ))
        self.assertFalse(result.feasible)
        self.assertEqual(result.reason, "mixed generation/swap launch is disabled")

    def test_conservative_arbiter_rejects_swap_overlap_and_active_mixing(self):
        arbiter = self._conservative()
        overlapping = arbiter.validate((
            self._request("s0", OperationKind.SWAP, (0, 1, 2), ("a", "b")),
            self._request("s1", OperationKind.SWAP, (2, 3, 4), ("c", "d")),
        ))
        self.assertFalse(overlapping.feasible)
        self.assertEqual(overlapping.reason, "concurrent swaps are disabled")

        active_mixing = arbiter.validate(
            (self._request("s", OperationKind.SWAP, (1, 2, 3), ("a", "b")),),
            active=(self._request("g", OperationKind.GEN, (4, 5)),),
        )
        self.assertFalse(active_mixing.feasible)
        self.assertIn("mixed generation/swap", active_mixing.reason)

    def test_enabled_arbiter_allows_disjoint_mixed_and_swaps(self):
        arbiter = SequenceProtocolArbiter(
            supports_inter_epoch_launch=True,
            supports_mixed_operation_concurrency=True,
            supports_concurrent_swaps=True,
        )
        result = arbiter.validate((
            self._request("g", OperationKind.GEN, (0, 1)),
            self._request("s", OperationKind.SWAP, (2, 3, 4), ("a", "b")),
        ))
        self.assertTrue(result.feasible)
        swaps = arbiter.validate((
            self._request("s0", OperationKind.SWAP, (0, 1, 2), ("a", "b")),
            self._request("s1", OperationKind.SWAP, (3, 4, 5), ("c", "d")),
        ))
        self.assertTrue(swaps.feasible)

    def test_input_pair_conflicts_are_rejected_independently_of_family(self):
        arbiter = self._conservative()
        result = arbiter.validate(
            (self._request("s", OperationKind.SWAP, (1, 2, 3), ("a", "b")),),
            active=(self._request("g", OperationKind.GEN, (4, 5), ("b",)),),
        )
        self.assertFalse(result.feasible)
        self.assertEqual(result.reason, "input segment is already in flight")

    def test_state_is_neutral_and_serializable(self):
        arbiter = self._conservative()
        self.assertEqual(
            dict(arbiter.state()),
            {
                "supports_inter_epoch_launch": True,
                "supports_mixed_operation_concurrency": False,
                "supports_concurrent_swaps": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
