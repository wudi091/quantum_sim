"""Small construction-DAG catalogues used by tests and controlled studies."""

from __future__ import annotations

from .construction_api import ConstructionDAG, ConstructionOperation, OperationKind, ResourceDemand


def _generation_operation(
    request_id: str,
    index: int,
    left: int,
    right: int,
    *,
    required_fidelity: float = 0.0,
) -> ConstructionOperation:
    op_id = f"{request_id}:gen:{index}"
    edge = f"{min(left, right)}-{max(left, right)}"
    held = ResourceDemand.from_mapping({
        f"link:{edge}": 1,
        f"memory:{left}": 1,
        f"memory:{right}": 1,
    })
    return ConstructionOperation(
        op_id=op_id,
        request_id=request_id,
        kind=OperationKind.GEN,
        output_segment_id=f"{request_id}:seg:gen:{index}",
        output_endpoints=(left, right),
        resource_demand=ResourceDemand.from_mapping({
            f"genlane:{edge}": 1,
            **held.as_dict(),
        }),
        output_resource_hold=held,
        required_fidelity=required_fidelity,
        duration_ps=1,
        ordinal=index,
    )


def left_deep_path_dag(
    request_id: str,
    route_nodes: tuple[int, ...],
    *,
    sequential_generation: bool = False,
    required_fidelity: float = 0.0,
) -> ConstructionDAG:
    """Build a left-associated swap tree for one fixed route."""

    if len(route_nodes) < 2:
        raise ValueError("route must contain at least one edge")
    operations = [
        _generation_operation(
            request_id,
            index,
            left,
            right,
            required_fidelity=(
                required_fidelity if len(route_nodes) == 2 else 0.0
            ),
        )
        for index, (left, right) in enumerate(zip(route_nodes, route_nodes[1:]))
    ]
    if sequential_generation:
        operations = [
            ConstructionOperation(
                **{
                    **operation.__dict__,
                    "predecessors": () if index == 0 else (operations[index - 1].op_id,),
                }
            )
            for index, operation in enumerate(operations)
        ]
    if len(route_nodes) > 2:
        previous_segment = operations[0].output_segment_id
        previous_operation_id = operations[0].op_id
        assert previous_segment is not None
        for index in range(1, len(route_nodes) - 1):
            right_segment = operations[index].output_segment_id
            assert right_segment is not None
            middle = route_nodes[index]
            op_id = f"{request_id}:swap:left:{index}"
            output = f"{request_id}:seg:left:{index}"
            operations.append(ConstructionOperation(
                op_id=op_id,
                request_id=request_id,
                kind=OperationKind.SWAP,
                predecessors=(previous_operation_id, operations[index].op_id),
                input_segment_ids=(previous_segment, right_segment),
                output_segment_id=output,
                output_endpoints=(route_nodes[0], route_nodes[index + 1]),
                resource_demand=ResourceDemand.from_mapping({f"bsm:{middle}": 1}),
                output_resource_hold=ResourceDemand.from_mapping({
                    f"memory:{route_nodes[0]}": 1,
                    f"memory:{route_nodes[index + 1]}": 1,
                }),
                required_fidelity=(
                    required_fidelity
                    if index == len(route_nodes) - 2
                    else 0.0
                ),
                duration_ps=2,
                ordinal=len(operations),
            ))
            previous_segment = output
            previous_operation_id = op_id
    return ConstructionDAG(request_id, tuple(operations))


def balanced_path_dag(
    request_id: str,
    route_nodes: tuple[int, ...],
    *,
    required_fidelity: float = 0.0,
) -> ConstructionDAG:
    """Build a balanced binary swap tree for one fixed route.

    The route remains identical to :func:`left_deep_path_dag`; only the
    precedence relation between swaps changes.
    """

    if len(route_nodes) < 2:
        raise ValueError("route must contain at least one edge")
    generations = [
        _generation_operation(
            request_id,
            index,
            left,
            right,
            required_fidelity=(
                required_fidelity if len(route_nodes) == 2 else 0.0
            ),
        )
        for index, (left, right) in enumerate(zip(route_nodes, route_nodes[1:]))
    ]
    operations = list(generations)
    level: list[tuple[str, tuple[int, int], int, int]] = [
        (operation.output_segment_id or "", operation.output_endpoints or (0, 0), index, index)
        for index, operation in enumerate(generations)
    ]
    ordinal = len(operations)
    merge_index = 0
    while len(level) > 1:
        next_level: list[tuple[str, tuple[int, int], int]] = []
        index = 0
        while index < len(level):
            if index + 1 == len(level):
                next_level.append(level[index])
                index += 1
                continue
            left_segment, left_endpoints, left_start, left_end = level[index]
            right_segment, right_endpoints, right_start, right_end = level[index + 1]
            middle = route_nodes[left_end + 1]
            op_id = f"{request_id}:swap:balanced:{merge_index}"
            output = f"{request_id}:seg:balanced:{merge_index}"
            predecessors = []
            for generation_index in (left_start, right_end):
                predecessor = generations[generation_index].op_id
                if predecessor not in predecessors:
                    predecessors.append(predecessor)
            for operation in operations:
                if operation.output_segment_id in {left_segment, right_segment}:
                    predecessors.append(operation.op_id)
            operations.append(ConstructionOperation(
                op_id=op_id,
                request_id=request_id,
                kind=OperationKind.SWAP,
                predecessors=tuple(dict.fromkeys(predecessors)),
                input_segment_ids=(left_segment, right_segment),
                output_segment_id=output,
                output_endpoints=(left_endpoints[0], right_endpoints[1]),
                resource_demand=ResourceDemand.from_mapping({f"bsm:{middle}": 1}),
                output_resource_hold=ResourceDemand.from_mapping({
                    f"memory:{left_endpoints[0]}": 1,
                    f"memory:{right_endpoints[1]}": 1,
                }),
                required_fidelity=(
                    required_fidelity
                    if len(level) == 2 and index == 0
                    else 0.0
                ),
                duration_ps=2,
                ordinal=ordinal,
            ))
            ordinal += 1
            next_level.append((output, (left_endpoints[0], right_endpoints[1]), left_start, right_end))
            merge_index += 1
            index += 2
        level = next_level
    return ConstructionDAG(request_id, tuple(operations))
