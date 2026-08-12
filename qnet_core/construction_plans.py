"""Small construction-DAG catalogues used by tests and controlled studies."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import TypeAlias

from .construction_api import ConstructionDAG, ConstructionOperation, OperationKind, ResourceDemand


SwapTree: TypeAlias = int | tuple["SwapTree", "SwapTree"]


@lru_cache(maxsize=None)
def _swap_tree_metrics(tree: SwapTree) -> tuple[int, int, int]:
    """Return ``(height, held_segment_rounds, imbalance)`` for one tree.

    Elementary links are available in round zero.  Every internal node is a
    swap one round after both children are ready.  ``held_segment_rounds`` is
    the sum of the time for which child segments wait before their parent
    swap consumes them; it is proportional to the memory--time footprint of
    the construction.  The final imbalance term is only a deterministic
    tie-breaker after latency and memory occupancy.
    """

    if isinstance(tree, int):
        return 0, 0, 0
    left, right = tree
    left_height, left_hold, left_imbalance = _swap_tree_metrics(left)
    right_height, right_hold, right_imbalance = _swap_tree_metrics(right)
    height = 1 + max(left_height, right_height)
    held_segment_rounds = (
        left_hold
        + right_hold
        + height
        - left_height
        + height
        - right_height
    )
    imbalance = (
        left_imbalance
        + right_imbalance
        + abs(left_height - right_height)
    )
    return height, held_segment_rounds, imbalance


def _swap_tree_signature(tree: SwapTree) -> tuple[int, ...]:
    """Return a comparison-safe canonical preorder signature."""

    if isinstance(tree, int):
        return (0, tree)
    left, right = tree
    return (1, *_swap_tree_signature(left), 2, *_swap_tree_signature(right), 3)


def _swap_tree_rank(tree: SwapTree) -> tuple[object, ...]:
    height, held_segment_rounds, imbalance = _swap_tree_metrics(tree)
    return (
        height,
        held_segment_rounds,
        imbalance,
        _swap_tree_signature(tree),
    )


@lru_cache(maxsize=None)
def _best_ordered_swap_trees(
    start: int,
    end: int,
    limit: int | None,
) -> tuple[SwapTree, ...]:
    """Interval top-k dynamic program for order-preserving swap trees."""

    if end - start == 1:
        return (start,)
    candidates: list[SwapTree] = []
    for split in range(start + 1, end):
        left_trees = _best_ordered_swap_trees(start, split, limit)
        right_trees = _best_ordered_swap_trees(split, end, limit)
        candidates.extend(
            (left, right)
            for left in left_trees
            for right in right_trees
        )
    ranked = sorted(candidates, key=_swap_tree_rank)
    if limit is not None:
        ranked = ranked[:limit]
    return tuple(ranked)


@lru_cache(maxsize=None)
def ordered_swap_trees(
    link_count: int,
    limit: int | None = None,
) -> tuple[SwapTree, ...]:
    """Return the best order-preserving full binary trees for a path.

    Leaves are elementary links numbered from left to right.  Internal nodes
    are entanglement swaps.  Without ``limit`` the number of returned trees is
    the Catalan number ``C_(link_count - 1)``.  With ``limit``, an interval
    top-k dynamic program materializes only the strongest candidates.  Trees
    are ranked by minimum completion depth, then minimum segment holding
    time, then balance and a stable structural signature.
    """

    if link_count < 1:
        raise ValueError("link_count must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    return _best_ordered_swap_trees(0, link_count, limit)


def swap_tree_kinds(
    link_count: int,
    limit: int | None = None,
) -> tuple[str, ...]:
    """Return stable construction-kind names for all path swap trees."""

    return tuple(
        f"swap_tree_{index}"
        for index in range(len(ordered_swap_trees(link_count, limit)))
    )


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
        retry_limit=1,
        duration_ps=1,
        ordinal=index,
    )


def _swap_resource_demand(
    left: int,
    middle: int,
    right: int,
) -> ResourceDemand:
    """Declare every physical-node mutex touched by one swap.

    ``bsm`` identifies the middle-node measurement device.  The opaque
    ``swapnode`` resources mirror SeQUeNCe's stricter protocol concurrency:
    two swaps sharing either outer memory node or the middle BSM node cannot
    execute concurrently.  Keeping both declarations in the neutral DTO lets
    offline expansion, rolling reservations, and the physical executor use
    one resource contract.
    """

    return ResourceDemand.from_mapping({
        f"bsm:{middle}": 1,
        f"swapnode:{left}": 1,
        f"swapnode:{middle}": 1,
        f"swapnode:{right}": 1,
    })


def swap_tree_path_dag(
    request_id: str,
    route_nodes: tuple[int, ...],
    tree_index: int,
    *,
    required_fidelity: float = 0.0,
) -> ConstructionDAG:
    """Build one indexed, order-preserving binary swap tree for a route."""

    if len(route_nodes) < 2:
        raise ValueError("route must contain at least one edge")
    trees = ordered_swap_trees(len(route_nodes) - 1, tree_index + 1)
    if not 0 <= tree_index < len(trees):
        raise ValueError(
            f"swap tree index {tree_index} is outside [0, {len(trees)})"
        )

    generations = [
        _generation_operation(request_id, index, left, right)
        for index, (left, right) in enumerate(
            zip(route_nodes, route_nodes[1:])
        )
    ]
    operations = list(generations)
    merge_index = 0

    def compile_tree(
        tree: SwapTree,
    ) -> tuple[str, str, int, int]:
        nonlocal merge_index
        if isinstance(tree, int):
            generation = generations[tree]
            assert generation.output_segment_id is not None
            return (
                generation.output_segment_id,
                generation.op_id,
                tree,
                tree + 1,
            )

        left_tree, right_tree = tree
        left_segment, left_operation, start, split = compile_tree(left_tree)
        right_segment, right_operation, right_start, end = compile_tree(
            right_tree
        )
        if split != right_start:
            raise ValueError("swap tree leaves must form contiguous path intervals")

        current_merge = merge_index
        merge_index += 1
        middle = route_nodes[split]
        op_id = f"{request_id}:swap:tree:{tree_index}:{current_merge}"
        output = f"{request_id}:seg:tree:{tree_index}:{current_merge}"
        operations.append(ConstructionOperation(
            op_id=op_id,
            request_id=request_id,
            kind=OperationKind.SWAP,
            predecessors=(left_operation, right_operation),
            input_segment_ids=(left_segment, right_segment),
            output_segment_id=output,
            output_endpoints=(route_nodes[start], route_nodes[end]),
            resource_demand=_swap_resource_demand(
                route_nodes[start],
                middle,
                route_nodes[end],
            ),
            output_resource_hold=ResourceDemand.from_mapping({
                f"memory:{route_nodes[start]}": 1,
                f"memory:{route_nodes[end]}": 1,
            }),
            required_fidelity=(
                required_fidelity
                if start == 0 and end == len(route_nodes) - 1
                else 0.0
            ),
            retry_limit=1,
            duration_ps=2,
            ordinal=len(operations),
        ))
        return output, op_id, start, end

    compile_tree(trees[tree_index])
    if len(route_nodes) == 2:
        generations[0] = replace(
            generations[0],
            required_fidelity=required_fidelity,
        )
        operations[0] = generations[0]
    return ConstructionDAG(request_id, tuple(operations))


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
                resource_demand=_swap_resource_demand(
                    route_nodes[0],
                    middle,
                    route_nodes[index + 1],
                ),
                output_resource_hold=ResourceDemand.from_mapping({
                    f"memory:{route_nodes[0]}": 1,
                    f"memory:{route_nodes[index + 1]}": 1,
                }),
                required_fidelity=(
                    required_fidelity
                    if index == len(route_nodes) - 2
                    else 0.0
                ),
                retry_limit=1,
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
                resource_demand=_swap_resource_demand(
                    left_endpoints[0],
                    middle,
                    right_endpoints[1],
                ),
                output_resource_hold=ResourceDemand.from_mapping({
                    f"memory:{left_endpoints[0]}": 1,
                    f"memory:{right_endpoints[1]}": 1,
                }),
                required_fidelity=(
                    required_fidelity
                    if len(level) == 2 and index == 0
                    else 0.0
                ),
                retry_limit=1,
                duration_ps=2,
                ordinal=ordinal,
            ))
            ordinal += 1
            next_level.append((output, (left_endpoints[0], right_endpoints[1]), left_start, right_end))
            merge_index += 1
            index += 2
        level = next_level
    return ConstructionDAG(request_id, tuple(operations))


def elementary_purification_dag(dag: ConstructionDAG) -> ConstructionDAG:
    """Insert one BBPSSW round after every elementary-pair generation.

    The returned graph remains simulator-neutral.  Each original ``GEN`` is
    replaced by two sequential generations and one ``PURIFY`` operation.  The
    purification output reuses the original segment ID, so the existing swap
    tree does not need to know how the elementary pair was prepared.
    """

    original = dag.operations
    completion_id = {
        operation.op_id: (
            f"{operation.op_id}:purify"
            if operation.kind == OperationKind.GEN
            else operation.op_id
        )
        for operation in original
    }
    operations: list[ConstructionOperation] = []
    ordinal = 0
    for operation in original:
        predecessors = tuple(
            completion_id[predecessor]
            for predecessor in operation.predecessors
        )
        if operation.kind != OperationKind.GEN:
            operations.append(replace(
                operation,
                predecessors=predecessors,
                ordinal=ordinal,
            ))
            ordinal += 1
            continue

        if operation.output_segment_id is None or operation.output_endpoints is None:
            raise ValueError("elementary purification requires complete GEN metadata")
        left, right = operation.output_endpoints
        edge = f"{min(left, right)}-{max(left, right)}"
        keep_id = f"{operation.op_id}:purify:keep"
        measure_id = f"{operation.op_id}:purify:measure"
        purify_id = completion_id[operation.op_id]
        keep_segment = f"{operation.output_segment_id}:purify:keep"
        measure_segment = f"{operation.output_segment_id}:purify:measure"

        keep = replace(
            operation,
            op_id=keep_id,
            predecessors=predecessors,
            output_segment_id=keep_segment,
            required_fidelity=0.0,
            ordinal=ordinal,
        )
        measure = replace(
            operation,
            op_id=measure_id,
            predecessors=(keep_id,),
            output_segment_id=measure_segment,
            required_fidelity=0.0,
            ordinal=ordinal + 1,
        )
        purify = ConstructionOperation(
            op_id=purify_id,
            request_id=operation.request_id,
            kind=OperationKind.PURIFY,
            predecessors=(keep_id, measure_id),
            input_segment_ids=(keep_segment, measure_segment),
            output_segment_id=operation.output_segment_id,
            output_endpoints=operation.output_endpoints,
            resource_demand=ResourceDemand.from_mapping({
                f"purify:{edge}": 1,
            }),
            output_resource_hold=operation.output_resource_hold,
            duration_ps=1,
            required_fidelity=operation.required_fidelity,
            retry_limit=operation.retry_limit,
            ordinal=ordinal + 2,
            dag_version=operation.dag_version,
        )
        operations.extend((keep, measure, purify))
        ordinal += 3
    return ConstructionDAG(dag.request_id, tuple(operations), version=dag.version)
