import unittest

from qnet_core.construction_api import (
    ConstructionDAG,
    ConstructionOperation,
    OperationKind,
    ResourceDemand,
)
from qnet_core.capacity_feasibility import CapacityFeasibilityOracle
from qnet_core.construction_metrics import (
    RequestSettlement,
    censored_flow_time,
)


class ConstructionContractTests(unittest.TestCase):
    def test_demand_is_canonical_and_dag_rejects_cycles(self):
        demand = ResourceDemand((("bsm:1", 1), ("memory:0", 2)))
        self.assertEqual(demand.entries, (("bsm:1", 1), ("memory:0", 2)))
        operation = ConstructionOperation(
            "a",
            "r",
            OperationKind.GEN,
            output_segment_id="s",
            output_endpoints=(0, 1),
        )
        dag = ConstructionDAG("r", (operation,))
        self.assertEqual(dag.ready_ids(set()), ("a",))
        with self.assertRaises(ValueError):
            ConstructionDAG("r", (
                ConstructionOperation(
                    "x", "r", OperationKind.RELEASE, predecessors=("y",)
                ),
                ConstructionOperation(
                    "y", "r", OperationKind.RELEASE, predecessors=("x",)
                ),
            ))

    def test_capacity_oracle_rejects_overcommit_and_double_consumption(self):
        operations = tuple(
            ConstructionOperation(
                op_id=f"operation:{index}",
                request_id="r",
                kind=OperationKind.RELEASE,
                input_segment_ids=("shared",),
                resource_demand=ResourceDemand.from_mapping({"bsm": 1}),
                ordinal=index,
            )
            for index in range(2)
        )
        oracle = CapacityFeasibilityOracle({"bsm": 1})
        self.assertTrue(oracle.check((operations[0],)).feasible)
        report = oracle.check(operations)
        self.assertFalse(report.feasible)
        self.assertIn(report.reason, {
            "input segment consumed twice",
            "capacity exceeded: bsm",
        })

    def test_censored_flow_time_uses_the_horizon_for_failures(self):
        settlements = (
            RequestSettlement("done", 0, 4, True),
            RequestSettlement("failed", 1, 6, False),
        )
        self.assertEqual(censored_flow_time(settlements, 10), 13)


if __name__ == "__main__":
    unittest.main()
