"""Paper-aligned TELGEN IPM-trajectory pilot for the quantum planning LP.

This module adapts the algorithmic recipe in TELGEN to the simulator-neutral
single-stage quantum planning LP:

* one time-expanded path/construction/start candidate is one LP variable;
* request and resource-time rows are inequality-constraint vertices;
* one objective vertex is connected to all variables and constraints;
* SciPy's primal interior-point callback trajectory is the teacher signal;
* K shared outer loops imitate IPM iterations and J inner GNN layers imitate
  the Newton step;
* training uses the primal, objective, normalized constraint, request-mass,
  and within-request distribution losses;
* request uniqueness is parameterized at the readout, while one deterministic
  capacity-safe rounding shared with the IPM teacher produces an executable
  plan.

The official public TELGEN repository contains incomplete heterogeneous
aggregation code, so this is not a byte-for-byte copy.  The data generation,
trajectory sampling, tripartite graph, six directed relations, double loop,
weight sharing, and readout follow the paper and the runnable parts of the
official source.  Mean relation aggregation and request-structured losses are
quantum-network adaptations for size transfer.  The relation convolution is a
dependency-free PyTorch implementation of the intended GCN operation.

This pilot learns a continuous LP relaxation and evaluates its rounded
construction-aware plan.  The resulting checkpoint can be used by the
``ipm_gnn`` backend of the existing SeQUeNCe online controller.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import time
from typing import Iterable, Mapping, Sequence
import warnings

import networkx as nx
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import torch
from torch import nn
from torch.nn import functional as F

from qnet_core.planning_spec import RequestSpec
from qnet_core.scenario import ScenarioConfig, make_episode
from qnet_core.spec import EpisodeSpec, PhysicalConfig

from .dataset import build_planning_batch_problem
from .optimization_model import PackingModel, build_delay_model
from .time_expansion import TimeExpandedCandidate


TELGEN_REFERENCE_COMMIT = "64684ebb3a7e856de86346da46232f8ceca6666c"
TELGEN_REFERENCE_PAPER = (
    "Traffic Engineering in Large-scale Networks with Generalizable "
    "Graph Neural Networks"
)


@dataclass(frozen=True)
class IPMGraph:
    """Tripartite LP graph in the feature convention used by TELGEN.

    ``objective`` stores the normalized reduced coefficients of the
    single-stage expected-delay minimization LP.  They are generally
    non-positive because selecting an on-time candidate reduces the censoring
    penalty.  The request-censoring constant is kept separately for reporting
    the absolute expected delay; ``objective_scale`` records the normalization
    applied to the reduced coefficients.
    """

    variable_features: np.ndarray
    constraint_features: np.ndarray
    objective_features: np.ndarray
    variable_edge_indices: np.ndarray
    constraint_edge_indices: np.ndarray
    edge_features: np.ndarray
    objective_coefficients: np.ndarray
    constraint_objective_features: np.ndarray
    matrix: np.ndarray
    rhs: np.ndarray
    objective: np.ndarray
    variable_upper_bound: float = 1.0
    objective_scale: float = 1.0
    objective_constant: float = 0.0
    request_censoring_latencies: tuple[tuple[str, float], ...] = ()

    @property
    def normalized_objective_constant(self) -> float:
        """Constant expressed in the same scale as ``objective``."""

        return float(self.objective_constant) / max(float(self.objective_scale), 1e-12)


@dataclass(frozen=True)
class IPMTrajectory:
    """Raw and fixed-length primal trajectories from SciPy IPM callbacks."""

    points: np.ndarray
    raw_points: np.ndarray
    objective_values: np.ndarray
    violations: np.ndarray
    normalized_violations: np.ndarray
    lp_optimum: float
    solver_iterations: int
    solver_status: int
    objective_constant: float = 0.0

    @property
    def total_lp_optimum(self) -> float:
        """Return the full normalized objective including the constant."""

        return float(self.objective_constant) + float(self.lp_optimum)


@dataclass(frozen=True)
class PilotSample:
    graph: IPMGraph
    trajectory: IPMTrajectory
    variables: tuple[TimeExpandedCandidate, ...]
    resource_capacities: tuple[tuple[str, int], ...]
    seed: int
    topology: str
    topology_seed: int
    node_count: int
    topology_signature: tuple[tuple[int, int], ...]

    @property
    def capacities(self) -> dict[str, int]:
        return dict(self.resource_capacities)


@dataclass(frozen=True)
class _PreparedGraph:
    """Device-resident immutable tensors for one graph.

    The serialized pilot samples intentionally remain NumPy based.  This
    runtime-only view avoids rebuilding the same index and feature tensors on
    every epoch while keeping checkpoints and dataset caches device-neutral.
    """

    variable_features: torch.Tensor
    constraint_features: torch.Tensor
    objective_features: torch.Tensor
    relations: dict[
        str,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]
    relation_normalizations: dict[str, torch.Tensor]
    request_groups: tuple[torch.Tensor, ...]
    request_rows: torch.Tensor
    variable_to_request: torch.Tensor


@dataclass(frozen=True)
class _PreparedLossTensors:
    """Static tensors used by the supervised loss for one sample."""

    target: torch.Tensor
    matrix: torch.Tensor | None
    rhs: torch.Tensor
    objective: torch.Tensor
    weights: torch.Tensor
    request_groups: tuple[torch.Tensor, ...]
    discount: float


class _PilotCacheUnpickler(pickle.Unpickler):
    """Load caches produced by older ``python -m`` training invocations."""

    _LOCAL_CLASSES = {
        "IPMGraph",
        "IPMTrajectory",
        "PilotSample",
    }

    def find_class(self, module: str, name: str) -> object:
        if module == "__main__" and name in self._LOCAL_CLASSES:
            return globals()[name]
        return super().find_class(module, name)


def _load_pilot_cache(path: Path) -> object:
    with path.open("rb") as handle:
        return _PilotCacheUnpickler(handle).load()


def _as_node_counts(value: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, int):
        counts = (int(value),)
    else:
        counts = tuple(int(item) for item in value)
    if not counts or any(item < 2 for item in counts):
        raise ValueError("node counts must contain integers of at least two")
    return counts


def _subsample_ipm_points(
    raw_points: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Match TELGEN's official ``SubSample`` trajectory transform.

    The solver callback includes its initial point.  When fewer points than
    requested are available, the official implementation retains the entire
    trajectory and repeats the final point.  When more are available, it
    samples evenly from callback index 1 through the final point.
    """

    points = np.asarray(raw_points, dtype=np.float64)
    if points.ndim != 2 or len(points) < 1:
        raise ValueError("raw_points must contain at least one primal vector")
    if steps < 1:
        raise ValueError("steps must be positive")
    length = len(points)
    if steps == 1:
        return points[-1:].copy()
    if steps == length:
        return points.copy()
    if steps > length:
        padding = np.repeat(points[-1:], steps - length, axis=0)
        return np.concatenate((points, padding), axis=0)
    indices = np.linspace(1, length - 1, steps).astype(np.int64)
    return points[indices].copy()


def _maximum_feasible_prefix_scale(
    point: np.ndarray,
    matrix: np.ndarray,
    rhs: np.ndarray,
) -> float:
    """Largest common scale in [0, 1] that satisfies packing rows."""

    load = matrix @ point if len(rhs) else np.zeros(0, dtype=np.float64)
    positive = load > 1e-12
    if not np.any(positive):
        return 1.0
    return max(
        0.0,
        min(1.0, float(np.min(rhs[positive] / load[positive]))),
    )


def _deduplicate_solver_rows(
    matrix: np.ndarray,
    rhs: np.ndarray,
) -> tuple[csr_matrix, np.ndarray]:
    """Remove exact duplicate inequality rows for the LP solver only.

    Candidate/resource expansion can produce the same normalized packing row
    many times (for example, when several construction candidates touch the
    same resource--slot pattern).  Duplicate rows do not change the feasible
    set, but they make the normal equations used by SciPy's legacy interior
    point implementation singular or badly conditioned.  The graph kept for
    the GNN remains unchanged; this is an exact solver-side presolve step.
    """

    sparse = csr_matrix(matrix)
    sparse.sum_duplicates()
    if not len(rhs) or sparse.shape[0] < 2:
        return sparse, np.asarray(rhs, dtype=np.float64)
    keep: list[int] = []
    seen: set[tuple[bytes, bytes, bytes]] = set()
    for row_index in range(sparse.shape[0]):
        start, end = sparse.indptr[row_index:row_index + 2]
        indices = np.asarray(sparse.indices[start:end], dtype=np.int64)
        data = np.asarray(sparse.data[start:end], dtype=np.float64)
        rhs_value = np.asarray(rhs[row_index], dtype=np.float64)
        key = (indices.tobytes(), data.tobytes(), rhs_value.tobytes())
        if key in seen:
            continue
        seen.add(key)
        keep.append(row_index)
    if len(keep) == sparse.shape[0]:
        return sparse, np.asarray(rhs, dtype=np.float64)
    keep_array = np.asarray(keep, dtype=np.int64)
    return sparse[keep_array], np.asarray(rhs, dtype=np.float64)[keep_array]


def solve_scipy_ipm_trajectory(
    matrix: np.ndarray,
    rhs: np.ndarray,
    objective: np.ndarray,
    *,
    outer_steps: int = 16,
    tolerance: float = 1e-9,
    max_iterations: int = 1000,
    objective_constant: float = 0.0,
) -> IPMTrajectory:
    """Solve the bounded LP and record SciPy interior-point primal iterates.

    This follows the official TELGEN data path: ``linprog`` is called with
    ``method='interior-point'`` and each callback returns ``res.x``.  Current
    SciPy still provides the deprecated solver, so copying TELGEN's private
    SciPy fork is unnecessary and would be less maintainable.
    """

    matrix = np.asarray(matrix, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    objective = np.asarray(objective, dtype=np.float64)
    if matrix.ndim != 2 or rhs.shape != (matrix.shape[0],):
        raise ValueError("matrix and rhs shapes do not match")
    if objective.shape != (matrix.shape[1],):
        raise ValueError("objective has the wrong shape")
    if not np.isfinite(matrix).all() or not np.isfinite(rhs).all():
        raise ValueError("LP coefficients must be finite")
    if not np.isfinite(objective).all():
        raise ValueError("LP objective must be finite")
    if outer_steps < 1 or max_iterations < 1 or tolerance <= 0.0:
        raise ValueError("invalid IPM trajectory configuration")
    if not np.isfinite(objective_constant) or objective_constant < 0.0:
        raise ValueError("objective_constant must be finite and non-negative")

    # Keep the graph representation dense because the GNN feature builder
    # consumes the matrix directly, but give the solver an equivalent sparse
    # representation first.  The sparse code path avoids the dense
    # normal-equation factorization that becomes severely ill-conditioned for
    # the large, highly redundant resource--time matrices in the training
    # protocol.  A few tiny, deliberately degenerate fixtures are better
    # handled by the dense path, so retain it as a deterministic fallback.
    # Both attempts have exactly the same coefficients, bounds, feasible set,
    # and objective; only the numerical linear-algebra representation differs.
    solver_sparse_matrix, solver_rhs = _deduplicate_solver_rows(
        matrix, rhs
    )
    solver_attempts = (
        ("sparse", solver_sparse_matrix),
        ("dense", solver_sparse_matrix.toarray()),
    )
    result = None
    callbacks: list[np.ndarray] = []
    failures: list[str] = []
    for representation, solver_matrix in solver_attempts:
        attempt_callbacks: list[np.ndarray] = []

        def record(attempt_result: object) -> None:
            point = np.asarray(
                getattr(attempt_result, "x"), dtype=np.float64
            )
            attempt_callbacks.append(point.copy())

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="`method='interior-point'` is deprecated",
                category=DeprecationWarning,
            )
            attempt = linprog(
                objective,
                A_ub=solver_matrix if len(rhs) else None,
                b_ub=solver_rhs if len(rhs) else None,
                bounds=(0.0, 1.0),
                method="interior-point",
                callback=record,
                options={
                    "presolve": True,
                    "tol": float(tolerance),
                    "maxiter": int(max_iterations),
                },
            )
        if (
            attempt.success
            and attempt.x is not None
            and np.isfinite(attempt.fun)
            and np.isfinite(np.asarray(attempt.x, dtype=np.float64)).all()
        ):
            result = attempt
            callbacks = attempt_callbacks
            break
        failures.append(
            f"{representation}: status={attempt.status}, "
            f"message={attempt.message}"
        )
    if result is None:
        raise RuntimeError(
            "SciPy interior-point teacher failed after sparse and dense "
            "attempts: " + "; ".join(failures)
        )
    final_point = np.asarray(result.x, dtype=np.float64)
    if not callbacks:
        callbacks.append(final_point.copy())
    elif not np.allclose(callbacks[-1], final_point, rtol=1e-8, atol=1e-10):
        callbacks.append(final_point.copy())
    raw_points = np.stack(callbacks)
    points = _subsample_ipm_points(raw_points, outer_steps)
    row_gap = (
        points @ matrix.T - rhs[None, :]
        if len(rhs)
        else np.zeros((len(points), 0), dtype=np.float64)
    )
    violations = (
        np.maximum(0.0, row_gap).max(axis=1)
        if row_gap.shape[1]
        else np.zeros(len(points), dtype=np.float64)
    )
    normalized = (
        (
            np.maximum(0.0, row_gap)
            / np.maximum(np.abs(rhs)[None, :], 1e-10)
        ).max(axis=1)
        if row_gap.shape[1]
        else np.zeros(len(points), dtype=np.float64)
    )
    return IPMTrajectory(
        points=points,
        raw_points=raw_points,
        objective_values=points @ objective,
        violations=violations,
        normalized_violations=normalized,
        lp_optimum=float(result.fun),
        solver_iterations=int(result.nit),
        solver_status=int(result.status),
        objective_constant=float(objective_constant),
    )


def build_ipm_graph(
    model: PackingModel,
    variables: Sequence[object],
    horizon: int,
) -> IPMGraph:
    """Convert the quantum packing LP to TELGEN's tripartite graph.

    TELGEN's general representation uses only matrix statistics for node
    attributes.  Quantum semantics such as path length, start slot, request
    type, success probability, and constraint type are intentionally absent
    from this paper-aligned version.  Their influence remains present through
    the LP coefficients themselves.
    """

    del horizon  # Semantic time features are not part of the paper graph.
    ordered = tuple(variables)
    matrix = model.a_ub.toarray().astype(np.float64, copy=True)
    rhs = np.asarray(model.b_ub, dtype=np.float64).copy()
    objective = np.asarray(model.objective, dtype=np.float64).copy()
    if matrix.shape != (len(rhs), len(ordered)):
        raise ValueError("LP and candidate count disagree")
    if len(objective) != len(ordered):
        raise ValueError("LP objective and candidate count disagree")
    # Section IV-E of the paper clusters all inequality rows after
    # normalizing them to the same RHS.  Positive row scaling preserves the
    # LP feasible set and makes the constraint loss exactly the normalized
    # violation used by TELGEN.  This isolated batch pilot has no reservations,
    # so every active request/resource row must have positive capacity.
    if len(rhs):
        if np.any(rhs <= 0.0):
            raise ValueError("TELGEN graph rows require positive RHS values")
        matrix /= rhs[:, None]
        rhs.fill(1.0)
    objective_scale = max(
        1e-10,
        float(np.max(np.abs(objective))) if len(objective) else 1.0,
    )
    objective /= objective_scale

    variable_features = np.column_stack((
        matrix.mean(axis=0) if matrix.shape[0] else np.zeros(matrix.shape[1]),
        matrix.std(axis=0) if matrix.shape[0] else np.zeros(matrix.shape[1]),
    )).astype(np.float64, copy=False)
    constraint_features = np.column_stack((
        matrix.mean(axis=1) if matrix.shape[1] else np.zeros(matrix.shape[0]),
        matrix.std(axis=1) if matrix.shape[1] else np.zeros(matrix.shape[0]),
    )).astype(np.float64, copy=False)
    objective_features = np.asarray((
        objective.mean() if len(objective) else 0.0,
        objective.std() if len(objective) else 0.0,
    ), dtype=np.float64).reshape(1, 2)

    rows, columns = np.nonzero(matrix)
    coefficients = matrix[rows, columns]
    return IPMGraph(
        variable_features=variable_features,
        constraint_features=constraint_features,
        objective_features=objective_features,
        variable_edge_indices=columns.astype(np.int64, copy=False),
        constraint_edge_indices=rows.astype(np.int64, copy=False),
        edge_features=coefficients[:, None].astype(np.float64, copy=False),
        objective_coefficients=objective.astype(np.float64, copy=False),
        constraint_objective_features=rhs[:, None].astype(
            np.float64, copy=False
        ),
        matrix=matrix,
        rhs=rhs,
        objective=objective,
        variable_upper_bound=1.0,
        objective_scale=float(objective_scale),
        objective_constant=float(model.objective_constant),
        request_censoring_latencies=tuple(
            model.request_censoring_latencies
        ),
    )


def _episode_with_fixed_topology(
    nodes: Sequence[int],
    edges: Sequence[tuple[int, int]],
    physical: PhysicalConfig,
    horizon: int,
    seed: int,
    request_count: int,
    endpoint_mode: str = "uniform_random",
) -> EpisodeSpec:
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    if not nx.is_connected(graph):
        raise ValueError("pilot topology must be connected")
    rng = np.random.default_rng(seed)
    node_list = tuple(int(node) for node in nodes)
    if endpoint_mode not in {"uniform_random", "cut_hotspot"}:
        raise ValueError(f"unknown pilot endpoint mode: {endpoint_mode}")
    endpoint_sides: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    if endpoint_mode == "cut_hotspot":
        cut = nx.minimum_edge_cut(graph)
        if not cut:
            raise ValueError("cut-hotspot topology has no separating cut")
        separated = graph.copy()
        separated.remove_edges_from(cut)
        components = sorted(
            (
                tuple(sorted(int(node) for node in component))
                for component in nx.connected_components(separated)
            ),
            key=lambda item: (-len(item), item),
        )
        if len(components) < 2:
            raise ValueError("minimum edge cut did not separate the topology")
        endpoint_sides = (components[0], components[1])
    requests: list[RequestSpec] = []
    for index in range(request_count):
        if endpoint_sides is None:
            source, destination = (
                int(value) for value in rng.choice(node_list, 2, replace=False)
            )
        else:
            source = int(rng.choice(endpoint_sides[0]))
            destination = int(rng.choice(endpoint_sides[1]))
            if index % 2:
                source, destination = destination, source
        requests.append(RequestSpec(
            f"r{index}", source, destination, arrival=0, ttl=horizon,
        ))
    return EpisodeSpec(
        seed=int(seed),
        nodes=tuple(sorted(node_list)),
        edges=tuple(sorted(
            (min(int(u), int(v)), max(int(u), int(v))) for u, v in edges
        )),
        requests=tuple(requests),
        horizon=int(horizon),
        physical=physical,
    )


def _make_topology(
    kind: str,
    nodes: int,
    seed: int,
    physical: PhysicalConfig,
    horizon: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    scenario = ScenarioConfig(
        request_count=1,
        min_hops=None,
        max_hops=None,
        ttl=horizon,
        horizon=horizon,
        physical=physical,
        topology_nodes=nodes,
        topology_mode=kind,
        endpoint_mode="uniform_random",
        waxman_alpha=0.1,
        waxman_beta=0.4,
        waxman_add_mst=True,
        topology_attempts=256,
    )
    episode = make_episode(scenario, seed)
    return episode.nodes, episode.edges


def make_samples(
    *,
    topology: str,
    node_count: int | Sequence[int],
    sample_count: int,
    seed: int,
    request_count: int = 3,
    horizon: int = 6,
    path_count: int = 2,
    construction_plan_count: int = 3,
    outer_steps: int = 16,
    fixed_topology: bool = False,
    topology_seed: int | None = None,
    endpoint_mode: str = "uniform_random",
) -> tuple[PilotSample, ...]:
    """Generate quantum LP samples and their paper-style IPM trajectories.

    ``fixed_topology=False`` is the default because TELGEN is trained on a
    topology family with several graph configurations, not one fixed graph.
    Passing several node counts creates a multi-scale dataset within that one
    topology family.  ``fixed_topology=True`` remains available solely as a
    controlled single-graph stress test.
    """

    if sample_count < 1 or request_count < 1 or horizon < 1:
        raise ValueError("sample_count/request_count/horizon must be positive")
    if path_count < 1 or construction_plan_count < 1 or outer_steps < 1:
        raise ValueError("candidate and trajectory counts must be positive")
    node_counts = _as_node_counts(node_count)
    physical = PhysicalConfig(
        generation_probability=0.8,
        swap_probability=0.9,
        memory_capacity=2,
        max_width=1,
    )
    base_topology_seed = seed if topology_seed is None else int(topology_seed)
    fixed_graphs: dict[int, tuple[tuple[int, ...], tuple[tuple[int, int], ...]]] = {}
    if fixed_topology:
        for count_index, count in enumerate(node_counts):
            fixed_graphs[count] = _make_topology(
                topology,
                count,
                base_topology_seed + count_index * 100_003,
                physical,
                horizon,
            )

    samples: list[PilotSample] = []
    attempt = 0
    maximum_attempts = max(sample_count * 4, sample_count + 8)
    while len(samples) < sample_count and attempt < maximum_attempts:
        count = node_counts[attempt % len(node_counts)]
        sample_topology_seed = base_topology_seed + attempt * 100_003
        if fixed_topology:
            sample_nodes, sample_edges = fixed_graphs[count]
        else:
            sample_nodes, sample_edges = _make_topology(
                topology,
                count,
                sample_topology_seed,
                physical,
                horizon,
            )
        episode_seed = seed + attempt + 1
        episode = _episode_with_fixed_topology(
            sample_nodes,
            sample_edges,
            physical,
            horizon,
            episode_seed,
            request_count,
            endpoint_mode,
        )
        problem = build_planning_batch_problem(
            episode,
            path_candidate_count=path_count,
            construction_kinds=(),
            swap_tree_count=construction_plan_count,
            purification_kinds=("none",),
        )
        variables = tuple(sorted(
            problem.expansion.variables,
            key=lambda item: item.variable_id,
        ))
        attempt += 1
        if not variables:
            continue
        model = build_delay_model(
            variables,
            problem.capacities,
            request_censoring_latencies=(
                problem.request_censoring_latency_map
            ),
        )
        graph = build_ipm_graph(model, variables, episode.horizon)
        trajectory = solve_scipy_ipm_trajectory(
            graph.matrix,
            graph.rhs,
            graph.objective,
            outer_steps=outer_steps,
            objective_constant=graph.normalized_objective_constant,
        )
        samples.append(PilotSample(
            graph=graph,
            trajectory=trajectory,
            variables=variables,
            resource_capacities=tuple(sorted(problem.capacities.items())),
            seed=episode_seed,
            topology=topology,
            topology_seed=sample_topology_seed,
            node_count=count,
            topology_signature=tuple(sorted(
                (min(int(u), int(v)), max(int(u), int(v)))
                for u, v in sample_edges
            )),
        ))
    if len(samples) != sample_count:
        raise RuntimeError(
            f"generated {len(samples)} valid samples, expected {sample_count}"
        )
    return tuple(samples)


class _MLP(nn.Module):
    """Small ReLU MLP matching the official TELGEN helper's shape."""

    def __init__(
        self,
        channels: Sequence[int],
        *,
        normalization: str | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(channels) < 2:
            raise ValueError("an MLP needs at least input and output sizes")
        if normalization not in (None, "layer"):
            raise ValueError("normalization must be None or 'layer'")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(
            zip(channels[:-1], channels[1:])
        ):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(channels) - 2:
                if normalization == "layer":
                    layers.append(nn.LayerNorm(output_dim))
                layers.append(nn.ReLU())
                if dropout:
                    layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class _RelationGCN(nn.Module):
    """One directed relation in the TELGEN tripartite GCN."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        message_mlp_layers: int,
        normalization: str | None,
    ):
        super().__init__()
        self.source = nn.Linear(2 * hidden_dim, hidden_dim)
        self.destination = nn.Linear(2 * hidden_dim, hidden_dim)
        self.edge = nn.Linear(1, hidden_dim)
        self.update = _MLP(
            [hidden_dim] * (message_mlp_layers + 1),
            normalization=normalization,
        )

    def forward(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
        source_index: torch.Tensor,
        destination_index: torch.Tensor,
        edge_value: torch.Tensor,
        *,
        edge_embedding: torch.Tensor | None = None,
        destination_normalization: torch.Tensor | None = None,
    ) -> torch.Tensor:
        destination_count = len(destination)
        destination_self = self.destination(destination)
        if len(source_index) == 0:
            return self.update(destination_self)
        if destination_normalization is None:
            destination_degree = torch.bincount(
                destination_index, minlength=destination_count
            ).to(source.dtype) + 1.0
            # Mean aggregation keeps a request embedding stable when the
            # number of candidate variables grows on a larger or denser
            # topology.  The symmetric normalization used by the generic
            # TELGEN graph is degree-sensitive and caused a size-dependent
            # scale drift here.
            normalization = destination_degree[destination_index].reciprocal()
        else:
            normalization = destination_normalization
            if normalization.dtype != source.dtype:
                normalization = normalization.to(source.dtype)
        if edge_embedding is None:
            edge_embedding = self.edge(edge_value[:, None])
        message = torch.relu(
            self.source(source[source_index]) + edge_embedding
        )
        message = message * normalization[:, None]
        aggregate = destination.new_zeros((destination_count, message.shape[-1]))
        aggregate.index_add_(0, destination_index, message)
        return self.update(aggregate + destination_self)


class _TELGENInnerStep(nn.Module):
    """One ranked six-relation message-passing layer."""

    _RELATION_ORDER = (
        "constraint_to_variable",
        "variable_to_constraint",
        "variable_to_objective",
        "objective_to_variable",
        "constraint_to_objective",
        "objective_to_constraint",
    )
    _COV_RANKS = {
        "variable_to_constraint": 0,
        "objective_to_constraint": 0,
        "constraint_to_objective": 1,
        "variable_to_objective": 1,
        "constraint_to_variable": 2,
        "objective_to_variable": 2,
    }

    def __init__(
        self,
        hidden_dim: int,
        *,
        message_mlp_layers: int,
        normalization: str | None,
    ):
        super().__init__()
        for name in self._RELATION_ORDER:
            setattr(
                self,
                name,
                _RelationGCN(
                    hidden_dim,
                    message_mlp_layers=message_mlp_layers,
                    normalization=normalization,
                ),
            )

    def forward(
        self,
        variable: torch.Tensor,
        constraint: torch.Tensor,
        objective_node: torch.Tensor,
        relations: dict[
            str,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        relation_normalizations: dict[str, torch.Tensor] | None = None,
        edge_embeddings: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states = {
            "variable": variable,
            "constraint": constraint,
            "objective": objective_node,
        }
        endpoints = {
            "constraint_to_variable": ("constraint", "variable"),
            "variable_to_constraint": ("variable", "constraint"),
            "variable_to_objective": ("variable", "objective"),
            "objective_to_variable": ("objective", "variable"),
            "constraint_to_objective": ("constraint", "objective"),
            "objective_to_constraint": ("objective", "constraint"),
        }
        for rank in (0, 1, 2):
            updates: dict[str, list[torch.Tensor]] = {}
            for name in self._RELATION_ORDER:
                if self._COV_RANKS[name] != rank:
                    continue
                source_name, destination_name = endpoints[name]
                source_index, destination_index, edge_value = relations[name]
                message = getattr(self, name)(
                    states[source_name],
                    states[destination_name],
                    source_index,
                    destination_index,
                    edge_value,
                    edge_embedding=(
                        None
                        if edge_embeddings is None
                        else edge_embeddings.get(name)
                    ),
                    destination_normalization=(
                        None
                        if relation_normalizations is None
                        else relation_normalizations.get(name)
                    ),
                )
                updates.setdefault(destination_name, []).append(message)
            for destination_name, messages in updates.items():
                states[destination_name] = torch.cat(messages, dim=-1)
        return (
            states["variable"],
            states["constraint"],
            states["objective"],
        )


class TELGENPaperGNN(nn.Module):
    """Tripartite double-loop GNN with structural request normalization.

    The message-passing and shared outer loops follow TELGEN's IPM-process
    recipe.  Quantum packing adds one invariant at the readout: candidate
    mass is normalized within each request and scaled by a request-level
    admission mass.  Consequently every predicted primal already satisfies
    request uniqueness, independent of the number of candidates per request.
    """

    _GRAPH_CACHE_LIMIT_BYTES = 256 * 1024 * 1024

    def __init__(
        self,
        hidden_dim: int = 180,
        inner_layers: int = 2,
        *,
        message_mlp_layers: int = 4,
        prediction_layers: int = 4,
        normalization: str | None = "layer",
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim < 8 or inner_layers < 1:
            raise ValueError("hidden_dim/inner_layers are too small")
        if message_mlp_layers < 1 or prediction_layers < 1:
            raise ValueError("MLP layer counts must be positive")
        self.hidden_dim = int(hidden_dim)
        self.inner_layers = int(inner_layers)
        self.message_mlp_layers = int(message_mlp_layers)
        self.prediction_layers = int(prediction_layers)
        self.normalization = normalization
        self.dropout = float(dropout)
        self.variable_encoder = _MLP(
            [2, hidden_dim, 2 * hidden_dim],
            normalization=normalization,
            dropout=dropout,
        )
        self.constraint_encoder = _MLP(
            [2, hidden_dim, 2 * hidden_dim],
            normalization=normalization,
            dropout=dropout,
        )
        self.objective_encoder = _MLP(
            [2, hidden_dim, 2 * hidden_dim],
            normalization=normalization,
            dropout=dropout,
        )
        # The official configuration shares one relation-GCN set across both
        # inner layers and all outer IPM loops.
        self.inner_step = _TELGENInnerStep(
            hidden_dim,
            message_mlp_layers=message_mlp_layers,
            normalization=normalization,
        )
        self.readout = _MLP(
            [2 * hidden_dim]
            + [hidden_dim] * (prediction_layers - 1)
            + [1],
            normalization=None,
            dropout=dropout,
        )
        self.request_readout = _MLP(
            [2 * hidden_dim]
            + [hidden_dim] * (prediction_layers - 1)
            + [1],
            normalization=None,
            dropout=dropout,
        )
        # Graph structure is immutable during training.  Keep a device-local
        # tensor view keyed by graph identity; the serialized NumPy graph is
        # still the source of truth for reproducibility.
        self._graph_cache: OrderedDict[
            tuple[int, str],
            tuple[_PreparedGraph, int],
        ] = OrderedDict()
        self._graph_cache_bytes = 0

    def clear_graph_cache(self) -> None:
        """Release cached graph tensors, for phase-boundary memory control."""

        self._graph_cache.clear()
        self._graph_cache_bytes = 0

    @staticmethod
    def _relations(
        graph: IPMGraph,
        device: torch.device,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        variables = torch.arange(
            graph.matrix.shape[1], dtype=torch.long, device=device
        )
        constraints = torch.arange(
            graph.matrix.shape[0], dtype=torch.long, device=device
        )
        objective_node = torch.zeros(1, dtype=torch.long, device=device)
        variable_indices = torch.as_tensor(
            graph.variable_edge_indices, dtype=torch.long, device=device
        )
        constraint_indices = torch.as_tensor(
            graph.constraint_edge_indices, dtype=torch.long, device=device
        )
        matrix_values = torch.as_tensor(
            graph.edge_features[:, 0], dtype=torch.float32, device=device
        )
        objective_values = torch.as_tensor(
            graph.objective_coefficients, dtype=torch.float32, device=device
        )
        rhs_values = torch.as_tensor(
            graph.constraint_objective_features[:, 0],
            dtype=torch.float32,
            device=device,
        )
        return {
            "variable_to_constraint": (
                variable_indices, constraint_indices, matrix_values
            ),
            "constraint_to_variable": (
                constraint_indices, variable_indices, matrix_values
            ),
            "variable_to_objective": (
                variables,
                objective_node.expand(len(variables)),
                objective_values,
            ),
            "objective_to_variable": (
                objective_node.expand(len(variables)),
                variables,
                objective_values,
            ),
            "constraint_to_objective": (
                constraints,
                objective_node.expand(len(constraints)),
                rhs_values,
            ),
            "objective_to_constraint": (
                objective_node.expand(len(constraints)),
                constraints,
                rhs_values,
            ),
        }

    @staticmethod
    def _request_partition(
        graph: IPMGraph,
        device: torch.device,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        """Recover request rows and the variable-to-request assignment.

        ``build_delay_model`` appends resource-time rows after request
        rows.  Request rows therefore form the initial sequence of disjoint
        all-one rows and collectively cover every candidate.  The same
        invariant is retained in the serialized dataset, so this partition
        remains available without adding quantum-specific graph features.
        """

        matrix = np.asarray(graph.matrix, dtype=np.float64)
        variable_count = matrix.shape[1]
        covered = np.zeros(variable_count, dtype=bool)
        groups: list[np.ndarray] = []
        rows: list[int] = []
        for row_index in range(matrix.shape[0]):
            columns = np.flatnonzero(np.abs(matrix[row_index]) > 1e-12)
            if not len(columns):
                continue
            if not np.allclose(matrix[row_index, columns], 1.0):
                break
            if np.any(covered[columns]):
                break
            rows.append(row_index)
            groups.append(columns)
            covered[columns] = True
            if bool(np.all(covered)):
                break
        if not groups or not bool(np.all(covered)):
            raise ValueError(
                "IPM graph does not expose a complete disjoint request "
                "partition"
            )
        request_groups = tuple(
            torch.as_tensor(columns, dtype=torch.long, device=device)
            for columns in groups
        )
        variable_to_request = np.empty(variable_count, dtype=np.int64)
        for request_index, columns in enumerate(groups):
            variable_to_request[columns] = request_index
        return (
            request_groups,
            torch.as_tensor(rows, dtype=torch.long, device=device),
            torch.as_tensor(variable_to_request, dtype=torch.long, device=device),
        )

    def _prepare_graph(
        self,
        graph: IPMGraph,
        device: torch.device,
    ) -> _PreparedGraph:
        """Create and cache the device-resident graph representation."""

        key = (id(graph), str(device))
        cached = self._graph_cache.get(key)
        if cached is not None:
            self._graph_cache.move_to_end(key)
            return cached[0]

        variable_features = torch.as_tensor(
            graph.variable_features, dtype=torch.float32, device=device
        )
        constraint_features = torch.as_tensor(
            graph.constraint_features, dtype=torch.float32, device=device
        )
        objective_features = torch.as_tensor(
            graph.objective_features, dtype=torch.float32, device=device
        )
        relations = self._relations(graph, device)
        destination_counts = {
            "constraint_to_variable": len(variable_features),
            "variable_to_constraint": len(constraint_features),
            "variable_to_objective": 1,
            "objective_to_variable": len(variable_features),
            "constraint_to_objective": 1,
            "objective_to_constraint": len(constraint_features),
        }
        relation_normalizations: dict[str, torch.Tensor] = {}
        for name, (_, destination_index, _) in relations.items():
            destination_count = destination_counts[name]
            if len(destination_index) == 0:
                relation_normalizations[name] = torch.zeros(
                    0, dtype=torch.float32, device=device
                )
                continue
            destination_degree = torch.bincount(
                destination_index, minlength=destination_count
            ).to(torch.float32) + 1.0
            relation_normalizations[name] = destination_degree.index_select(
                0, destination_index
            ).reciprocal()
        request_groups, request_rows, variable_to_request = (
            self._request_partition(graph, device)
        )
        prepared = _PreparedGraph(
            variable_features=variable_features,
            constraint_features=constraint_features,
            objective_features=objective_features,
            relations=relations,
            relation_normalizations=relation_normalizations,
            request_groups=request_groups,
            request_rows=request_rows,
            variable_to_request=variable_to_request,
        )
        unique_tensors: dict[int, torch.Tensor] = {}
        direct_tensors = (
            prepared.variable_features,
            prepared.constraint_features,
            prepared.objective_features,
            prepared.request_rows,
            prepared.variable_to_request,
            *prepared.request_groups,
            *prepared.relation_normalizations.values(),
        )
        relation_tensors = (
            tensor
            for relation in prepared.relations.values()
            for tensor in relation
        )
        for tensor in (*direct_tensors, *relation_tensors):
            unique_tensors.setdefault(tensor.data_ptr(), tensor)
        prepared_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in unique_tensors.values()
        )
        if prepared_bytes <= self._GRAPH_CACHE_LIMIT_BYTES:
            while (
                self._graph_cache
                and self._graph_cache_bytes + prepared_bytes
                > self._GRAPH_CACHE_LIMIT_BYTES
            ):
                _, (_, released_bytes) = self._graph_cache.popitem(last=False)
                self._graph_cache_bytes -= released_bytes
            self._graph_cache[key] = (prepared, prepared_bytes)
            self._graph_cache_bytes += prepared_bytes
        return prepared

    def forward(self, graph: IPMGraph, *, steps: int) -> torch.Tensor:
        if steps < 1:
            raise ValueError("steps must be positive")
        device = next(self.parameters()).device
        prepared = self._prepare_graph(graph, device)
        variable = self.variable_encoder(prepared.variable_features)
        constraint = self.constraint_encoder(prepared.constraint_features)
        objective_node = self.objective_encoder(prepared.objective_features)
        relations = prepared.relations
        request_groups = prepared.request_groups
        request_rows = prepared.request_rows
        variable_to_request = prepared.variable_to_request
        # Edge values are fixed for a graph during one forward pass.  Their
        # relation-specific linear embeddings can therefore be reused across
        # all shared outer and inner iterations while retaining gradients.
        edge_embeddings = {
            name: getattr(self.inner_step, name).edge(edge_value[:, None])
            for name, (_, _, edge_value) in relations.items()
            if len(edge_value)
        }
        outputs: list[torch.Tensor] = []
        for _ in range(steps):
            last_variable_message = variable
            for _ in range(self.inner_layers):
                old_variable = variable
                old_constraint = constraint
                old_objective = objective_node
                (
                    new_variable,
                    new_constraint,
                    new_objective,
                ) = self.inner_step(
                    variable,
                    constraint,
                    objective_node,
                    relations,
                    prepared.relation_normalizations,
                    edge_embeddings,
                )
                last_variable_message = new_variable
                variable = (torch.relu(new_variable) + old_variable) / 2.0
                constraint = (torch.relu(new_constraint) + old_constraint) / 2.0
                objective_node = (
                    torch.relu(new_objective) + old_objective
                ) / 2.0
            candidate_scores = F.softplus(
                self.readout(last_variable_message).squeeze(-1)
            ) + 1e-8
            request_mass = torch.sigmoid(
                self.request_readout(
                    constraint.index_select(0, request_rows)
                ).squeeze(-1)
            )
            normalized_scores = torch.zeros_like(candidate_scores)
            for request_index, candidate_indices in enumerate(request_groups):
                request_scores = candidate_scores.index_select(
                    0, candidate_indices
                )
                normalized_scores.index_copy_(
                    0,
                    candidate_indices,
                    request_scores / request_scores.sum().clamp_min(1e-8),
                )
            outputs.append(
                normalized_scores
                * request_mass.index_select(0, variable_to_request)
            )
        return torch.stack(outputs)


class TELGENQuantumAdapterGNN(TELGENPaperGNN):
    """Reserved extension point for explicitly quantum-semantic features.

    The paper-aligned experiment does not instantiate this class.  Keeping a
    named boundary prevents future topology, path, fidelity, or slot features
    from silently changing what is reported as the TELGEN-aligned result.
    """

    def __init__(self, *args: object, **kwargs: object):
        raise NotImplementedError(
            "quantum-semantic TELGEN extensions must be implemented and "
            "reported as a separate ablation"
        )


def _prepare_loss_tensors(
    sample: PilotSample,
    device: torch.device,
    *,
    discount: float,
    cache_matrix: bool = True,
) -> _PreparedLossTensors:
    """Move sample-constant loss inputs to ``device`` once."""

    target = torch.as_tensor(
        sample.trajectory.points, dtype=torch.float32, device=device
    )
    grouped_indices: dict[str, list[int]] = {}
    for index, variable in enumerate(sample.variables):
        grouped_indices.setdefault(variable.request_id, []).append(index)
    request_groups = tuple(
        torch.as_tensor(indices, dtype=torch.long, device=device)
        for indices in grouped_indices.values()
    )
    weights = torch.as_tensor(
        [discount ** (len(target) - index - 1)
         for index in range(len(target))],
        dtype=torch.float32,
        device=device,
    )
    return _PreparedLossTensors(
        target=target,
        matrix=(
            torch.as_tensor(
                sample.graph.matrix, dtype=torch.float32, device=device
            )
            if cache_matrix else None
        ),
        rhs=torch.as_tensor(
            sample.graph.rhs, dtype=torch.float32, device=device
        ),
        objective=torch.as_tensor(
            sample.graph.objective, dtype=torch.float32, device=device
        ),
        weights=weights,
        request_groups=request_groups,
        discount=float(discount),
    )


def _sample_loss(
    prediction: torch.Tensor,
    sample: PilotSample,
    *,
    discount: float = 0.7,
    objective_weight: float = 3.43,
    constraint_weight: float = 5.8,
    request_mass_weight: float = 2.0,
    candidate_distribution_weight: float = 0.5,
    prepared: _PreparedLossTensors | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Step-supervised loss for the single-stage LP trajectory.

    Request mass and within-request distribution describe the same continuous
    primal; they do not introduce a separate admission or throughput target.
    """

    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must lie in [0, 1]")
    if (
        request_mass_weight < 0.0
        or candidate_distribution_weight < 0.0
    ):
        raise ValueError("request-structure loss weights must be non-negative")
    device = prediction.device
    if prepared is None:
        target = torch.as_tensor(
            sample.trajectory.points, dtype=torch.float32, device=device
        )
        matrix = torch.as_tensor(
            sample.graph.matrix, dtype=torch.float32, device=device
        )
        rhs = torch.as_tensor(
            sample.graph.rhs, dtype=torch.float32, device=device
        )
        objective = torch.as_tensor(
            sample.graph.objective, dtype=torch.float32, device=device
        )
        weights = torch.as_tensor(
            [discount ** (len(target) - index - 1)
             for index in range(len(target))],
            dtype=torch.float32,
            device=device,
        )
        request_groups: tuple[torch.Tensor, ...] | None = None
    else:
        if abs(float(prepared.discount) - float(discount)) > 1e-12:
            raise ValueError("prepared loss tensors use a different discount")
        target = prepared.target
        matrix = (
            prepared.matrix
            if prepared.matrix is not None
            else torch.as_tensor(
                sample.graph.matrix, dtype=torch.float32, device=device
            )
        )
        rhs = prepared.rhs
        objective = prepared.objective
        weights = prepared.weights
        request_groups = prepared.request_groups
    if prediction.shape != target.shape:
        raise ValueError("prediction and teacher trajectory shapes differ")

    primal_loss = (
        (prediction - target).pow(2) * weights[:, None]
    ).mean()
    predicted_objective = prediction @ objective
    target_objective = target @ objective
    objective_gap = (
        (predicted_objective - target_objective)
        / target_objective.abs().clamp_min(1e-8)
    )
    objective_loss = (objective_gap.pow(2) * weights).mean()
    if len(rhs):
        normalized_violation = torch.relu(
            prediction @ matrix.T - rhs[None, :]
        )
        constraint_loss = (
            normalized_violation.pow(2) * weights[:, None]
        ).mean()
        maximum_violation = float(
            normalized_violation[-1].max().detach().cpu()
        )
    else:
        constraint_loss = prediction.new_zeros(())
        maximum_violation = 0.0

    if request_groups is None:
        grouped_indices: dict[str, list[int]] = {}
        for index, variable in enumerate(sample.variables):
            grouped_indices.setdefault(variable.request_id, []).append(index)
        request_groups = tuple(
            torch.as_tensor(indices, dtype=torch.long, device=device)
            for indices in grouped_indices.values()
        )
    request_mass_terms: list[torch.Tensor] = []
    distribution_terms: list[torch.Tensor] = []
    for index_tensor in request_groups:
        predicted_request = prediction.index_select(1, index_tensor)
        target_request = target.index_select(1, index_tensor)
        predicted_mass = predicted_request.sum(dim=1)
        target_mass = target_request.sum(dim=1)
        request_mass_terms.append(
            (predicted_mass - target_mass).pow(2) * weights
        )
        predicted_distribution = (
            predicted_request
            / predicted_mass[:, None].clamp_min(1e-8)
        )
        target_distribution = (
            target_request / target_mass[:, None].clamp_min(1e-8)
        )
        distribution_terms.append(
            (
                predicted_distribution - target_distribution
            ).pow(2).sum(dim=1) * weights
        )
    request_mass_loss = (
        torch.stack(request_mass_terms).mean()
        if request_mass_terms
        else prediction.new_zeros(())
    )
    candidate_distribution_loss = (
        torch.stack(distribution_terms).mean()
        if distribution_terms
        else prediction.new_zeros(())
    )
    total = (
        primal_loss
        + objective_weight * objective_loss
        + constraint_weight * constraint_loss
        + request_mass_weight * request_mass_loss
        + candidate_distribution_weight * candidate_distribution_loss
    )
    return total, {
        "primal_loss": float(primal_loss.detach().cpu()),
        "objective_loss": float(objective_loss.detach().cpu()),
        "constraint_loss": float(constraint_loss.detach().cpu()),
        "request_mass_loss": float(request_mass_loss.detach().cpu()),
        "candidate_distribution_loss": float(
            candidate_distribution_loss.detach().cpu()
        ),
        "final_max_normalized_violation": maximum_violation,
    }


@dataclass(frozen=True)
class RoundedPlan:
    """One executable integer plan produced from a continuous primal."""

    selected_variables: tuple[TimeExpandedCandidate, ...]
    selected_indices: tuple[int, ...]
    selected_variable_ids: tuple[str, ...]
    completed_request_count: int
    expected_completed_request_mass: float
    total_completion_latency: float
    feasible: bool


def _candidate_resource_loads(
    variable: TimeExpandedCandidate,
) -> dict[tuple[str, int], int]:
    loads: dict[tuple[str, int], int] = {}
    for entry in variable.resource_usage:
        key = (entry.resource_id, int(entry.slot))
        loads[key] = loads.get(key, 0) + int(entry.amount)
    return loads


def _feasible_prefix_scale(
    point: np.ndarray,
    variables: Sequence[TimeExpandedCandidate],
    resource_capacities: Mapping[str, int],
    reserved_usage: Mapping[tuple[str, int], int],
) -> float:
    """Largest common scale that makes the non-negative primal feasible."""

    if not len(point):
        return 1.0
    scale = min(1.0, 1.0 / max(float(np.max(point)), 1.0))
    request_loads: dict[str, float] = {}
    resource_loads: dict[tuple[str, int], float] = {}
    for value, variable in zip(point, variables):
        weight = float(value)
        if weight <= 0.0:
            continue
        request_loads[variable.request_id] = (
            request_loads.get(variable.request_id, 0.0) + weight
        )
        for key, amount in _candidate_resource_loads(variable).items():
            resource_loads[key] = (
                resource_loads.get(key, 0.0) + weight * amount
            )
    for load in request_loads.values():
        if load > 1e-12:
            scale = min(scale, 1.0 / load)
    for (resource_id, slot), load in resource_loads.items():
        if load <= 1e-12:
            continue
        residual = (
            int(resource_capacities[resource_id])
            - int(reserved_usage.get((resource_id, slot), 0))
        )
        if residual <= 0:
            return 0.0
        scale = min(scale, residual / load)
    return max(0.0, min(1.0, float(scale)))


def round_continuous_plan(
    point: np.ndarray,
    sample: PilotSample,
    *,
    admission_threshold: float = 0.5,
) -> RoundedPlan:
    """Round one stored pilot sample with the shared online rule."""

    return round_candidate_scores(
        point,
        sample.variables,
        sample.capacities,
        request_censoring_latencies=dict(
            sample.graph.request_censoring_latencies
        ),
        admission_threshold=admission_threshold,
    )


def _resolve_rounding_censoring_latencies(
    variables: Sequence[TimeExpandedCandidate],
    supplied: Mapping[str, float] | None,
) -> dict[str, float]:
    """Resolve delay penalties for rounded plans.

    Low-level callers can omit the map; the largest available completion
    latency then supplies a conservative censoring boundary.  Online and
    pilot callers pass the explicit episode/deadline catalogue.
    """

    inferred: dict[str, float] = {}
    for variable in variables:
        inferred[variable.request_id] = max(
            inferred.get(variable.request_id, 0.0),
            float(variable.completion_latency),
        )
    resolved: dict[str, float] = {}
    for raw_request_id, raw_latency in (supplied or {}).items():
        request_id = str(raw_request_id)
        latency = float(raw_latency)
        if not request_id:
            raise ValueError("request censoring IDs must be non-empty")
        if not np.isfinite(latency) or latency < 0.0:
            raise ValueError(
                "request censoring latencies must be finite and non-negative"
            )
        resolved[request_id] = latency
    for request_id, latency in inferred.items():
        if request_id not in resolved:
            resolved[request_id] = latency
        elif resolved[request_id] + 1e-9 < latency:
            raise ValueError(
                f"censoring latency for {request_id} is earlier than a "
                "candidate completion"
            )
    return resolved


def round_candidate_scores(
    point: np.ndarray,
    variables: Sequence[TimeExpandedCandidate],
    resource_capacities: Mapping[str, int],
    *,
    reserved_usage: Mapping[tuple[str, int], int] | None = None,
    request_censoring_latencies: Mapping[str, float] | None = None,
    admission_threshold: float = 0.5,
) -> RoundedPlan:
    """Apply the same deterministic rounding to teacher and GNN primals.

    A request is considered eligible when its *unscaled* fractional mass
    reaches the threshold.  The common feasible-prefix scale is only used to
    rank candidates and keep the continuous diagnostic capacity-safe; it must
    not turn every request into a rejection when a dense graph has a small
    common scale.  Requests are then processed by decreasing unscaled mass,
    while candidates are tried by decreasing scaled value until one fits all
    resource--slot capacities.
    """

    values = np.asarray(point, dtype=np.float64).reshape(-1)
    ordered_variables = tuple(variables)
    if values.shape != (len(ordered_variables),):
        raise ValueError("continuous primal and candidate count differ")
    if not np.all(np.isfinite(values)):
        raise ValueError("continuous primal must be finite")
    if not 0.0 < admission_threshold <= 1.0:
        raise ValueError("admission threshold must lie in (0, 1]")
    non_negative = np.maximum(values, 0.0)
    capacities = {
        str(resource_id): int(capacity)
        for resource_id, capacity in resource_capacities.items()
    }
    reservations = {
        (str(resource_id), int(slot)): int(amount)
        for (resource_id, slot), amount in (reserved_usage or {}).items()
        if int(amount) != 0
    }
    scale = _feasible_prefix_scale(
        non_negative,
        ordered_variables,
        capacities,
        reservations,
    )
    scores = np.clip(scale * non_negative, 0.0, 1.0)
    by_request: dict[str, list[int]] = {}
    for index, variable in enumerate(ordered_variables):
        by_request.setdefault(variable.request_id, []).append(index)

    request_order: list[tuple[float, str, tuple[int, ...]]] = []
    for request_id, indices in by_request.items():
        admission_mass = min(1.0, float(np.sum(non_negative[indices])))
        if admission_mass + 1e-12 < admission_threshold:
            continue
        ranked = tuple(sorted(
            indices,
            key=lambda index: (
                -float(scores[index]),
                ordered_variables[index].completion_latency,
                ordered_variables[index].variable_id,
            ),
        ))
        request_order.append((admission_mass, request_id, ranked))
    request_order.sort(key=lambda item: (-item[0], item[1]))

    occupied = dict(reservations)
    selected: list[int] = []
    for _, _, ranked in request_order:
        for index in ranked:
            variable = ordered_variables[index]
            loads = _candidate_resource_loads(variable)
            if any(
                occupied.get(key, 0) + amount
                > capacities[key[0]]
                for key, amount in loads.items()
            ):
                continue
            for key, amount in loads.items():
                occupied[key] = occupied.get(key, 0) + amount
            selected.append(index)
            break

    feasible = all(
        amount <= capacities[resource_id]
        for (resource_id, _), amount in occupied.items()
    )
    selected_variables = tuple(ordered_variables[index] for index in selected)
    censoring = _resolve_rounding_censoring_latencies(
        ordered_variables,
        request_censoring_latencies,
    )
    expected_total_delay = sum(censoring.values()) + sum(
        variable.expected_success_probability
        * (
            float(variable.completion_latency)
            - censoring[variable.request_id]
        )
        for variable in selected_variables
    )
    if expected_total_delay < 0.0 and expected_total_delay > -1e-9:
        expected_total_delay = 0.0
    return RoundedPlan(
        selected_variables=selected_variables,
        selected_indices=tuple(selected),
        selected_variable_ids=tuple(
            variable.variable_id for variable in selected_variables
        ),
        completed_request_count=len(selected_variables),
        expected_completed_request_mass=float(sum(
            variable.expected_success_probability
            for variable in selected_variables
        )),
        total_completion_latency=float(expected_total_delay),
        feasible=feasible,
    )


def _evaluate(
    model: nn.Module,
    samples: Sequence[PilotSample],
) -> dict[str, float | int]:
    """Evaluate raw TELGEN outputs and its paper-reported scaled diagnostic."""

    model.eval()
    ratios: list[float] = []
    signed_gaps: list[float] = []
    absolute_gaps: list[float] = []
    scaled_ratios: list[float] = []
    scaled_gaps: list[float] = []
    violations: list[float] = []
    normalized_violations: list[float] = []
    bound_violations: list[float] = []
    feasible: list[float] = []
    final_mse: list[float] = []
    trajectory_mse: list[float] = []
    teacher_rounded_counts: list[float] = []
    gnn_rounded_counts: list[float] = []
    teacher_rounded_mass: list[float] = []
    gnn_rounded_mass: list[float] = []
    rounded_mass_ratios: list[float] = []
    teacher_rounded_latency: list[float] = []
    gnn_rounded_latency: list[float] = []
    rounded_jaccard: list[float] = []
    rounded_request_jaccard: list[float] = []
    rounded_feasible: list[float] = []
    with torch.no_grad():
        for sample in samples:
            trace = model(
                sample.graph, steps=len(sample.trajectory.points)
            ).cpu().numpy()
            point = trace[-1]
            # ``objective`` is the normalized reduced delay vector.  Negating
            # it is the delay reduction relative to leaving requests
            # uncensored at their deadlines.
            delay_reduction = float(-sample.graph.objective @ point)
            raw_optimum_delay_reduction = max(
                0.0,
                -float(sample.trajectory.lp_optimum),
            )
            if raw_optimum_delay_reduction <= 1e-10:
                # A graph whose every candidate completes exactly at the
                # censoring boundary has no reducible delay.  Treat an also
                # zero prediction as an exact match instead of emitting an
                # arbitrary ratio caused by a tiny denominator.
                ratio = 1.0 if abs(delay_reduction) <= 1e-8 else 0.0
                signed_gap = 0.0 if ratio == 1.0 else 1.0
                absolute_gap = signed_gap
                optimum_delay_reduction = 0.0
            else:
                optimum_delay_reduction = raw_optimum_delay_reduction
                ratio = delay_reduction / optimum_delay_reduction
                signed_gap = (
                    optimum_delay_reduction - delay_reduction
                ) / optimum_delay_reduction
                absolute_gap = abs(signed_gap)
            ratios.append(ratio)
            signed_gaps.append(signed_gap)
            absolute_gaps.append(absolute_gap)
            row_load = (
                sample.graph.matrix @ point
                if len(sample.graph.rhs)
                else np.zeros(0, dtype=np.float64)
            )
            row_gap = row_load - sample.graph.rhs
            violation = (
                max(0.0, float(np.max(row_gap))) if len(row_gap) else 0.0
            )
            normalized_violation = (
                float(np.max(
                    np.maximum(row_gap, 0.0)
                    / np.maximum(np.abs(sample.graph.rhs), 1e-10)
                ))
                if len(row_gap)
                else 0.0
            )
            bound_violation = max(
                max(0.0, float(-np.min(point))) if len(point) else 0.0,
                max(
                    0.0,
                    float(np.max(
                        point - sample.graph.variable_upper_bound
                    )),
                ) if len(point) else 0.0,
            )
            violations.append(violation)
            normalized_violations.append(normalized_violation)
            bound_violations.append(bound_violation)
            feasible.append(float(
                normalized_violation <= 1e-6 and bound_violation <= 1e-6
            ))

            # OnoCGap-style capacity scaling is a diagnostic used by TELGEN's
            # evaluation, not a training layer or execution-time decoder.
            scale = _maximum_feasible_prefix_scale(
                point, sample.graph.matrix, sample.graph.rhs
            )
            scaled_delay_reduction = float(
                -sample.graph.objective @ (scale * point)
            )
            if optimum_delay_reduction <= 1e-10:
                scaled_ratio = (
                    1.0 if abs(scaled_delay_reduction) <= 1e-8 else 0.0
                )
                scaled_gap = 0.0 if scaled_ratio == 1.0 else 1.0
            else:
                scaled_ratio = (
                    scaled_delay_reduction / optimum_delay_reduction
                )
                scaled_gap = (
                    optimum_delay_reduction - scaled_delay_reduction
                ) / optimum_delay_reduction
            scaled_ratios.append(scaled_ratio)
            scaled_gaps.append(scaled_gap)
            final_mse.append(float(np.mean(
                (point - sample.trajectory.points[-1]) ** 2
            )))
            trajectory_mse.append(float(np.mean(
                (trace - sample.trajectory.points) ** 2
            )))
            teacher_plan = round_continuous_plan(
                sample.trajectory.points[-1], sample
            )
            gnn_plan = round_continuous_plan(point, sample)
            teacher_rounded_counts.append(
                float(teacher_plan.completed_request_count)
            )
            gnn_rounded_counts.append(
                float(gnn_plan.completed_request_count)
            )
            teacher_rounded_mass.append(
                teacher_plan.expected_completed_request_mass
            )
            gnn_rounded_mass.append(
                gnn_plan.expected_completed_request_mass
            )
            if teacher_plan.expected_completed_request_mass > 1e-10:
                rounded_mass_ratios.append(
                    gnn_plan.expected_completed_request_mass
                    / teacher_plan.expected_completed_request_mass
                )
            teacher_rounded_latency.append(
                teacher_plan.total_completion_latency
            )
            gnn_rounded_latency.append(
                gnn_plan.total_completion_latency
            )
            teacher_indices = set(teacher_plan.selected_indices)
            gnn_indices = set(gnn_plan.selected_indices)
            union = teacher_indices | gnn_indices
            rounded_jaccard.append(
                1.0
                if not union
                else len(teacher_indices & gnn_indices) / len(union)
            )
            teacher_requests = {
                sample.variables[index].request_id
                for index in teacher_indices
            }
            gnn_requests = {
                sample.variables[index].request_id
                for index in gnn_indices
            }
            request_union = teacher_requests | gnn_requests
            rounded_request_jaccard.append(
                1.0
                if not request_union
                else len(teacher_requests & gnn_requests)
                / len(request_union)
            )
            rounded_feasible.append(float(
                teacher_plan.feasible and gnn_plan.feasible
            ))
    return {
        "samples": len(samples),
        "mean_objective_ratio": float(np.mean(ratios)) if ratios else 0.0,
        "mean_signed_objective_gap": (
            float(np.mean(signed_gaps)) if signed_gaps else 0.0
        ),
        "mean_absolute_objective_gap": (
            float(np.mean(absolute_gaps)) if absolute_gaps else 0.0
        ),
        "mean_onoc_scaled_objective_ratio": (
            float(np.mean(scaled_ratios)) if scaled_ratios else 0.0
        ),
        "mean_onoc_scaled_objective_gap": (
            float(np.mean(scaled_gaps)) if scaled_gaps else 0.0
        ),
        "mean_constraint_violation": (
            float(np.mean(violations)) if violations else 0.0
        ),
        "max_constraint_violation": (
            float(np.max(violations)) if violations else 0.0
        ),
        "mean_normalized_constraint_violation": (
            float(np.mean(normalized_violations))
            if normalized_violations else 0.0
        ),
        "max_normalized_constraint_violation": (
            float(np.max(normalized_violations))
            if normalized_violations else 0.0
        ),
        "max_variable_bound_violation": (
            float(np.max(bound_violations)) if bound_violations else 0.0
        ),
        "raw_feasible_rate": float(np.mean(feasible)) if feasible else 0.0,
        "mean_final_variable_mse": (
            float(np.mean(final_mse)) if final_mse else 0.0
        ),
        "mean_trajectory_variable_mse": (
            float(np.mean(trajectory_mse)) if trajectory_mse else 0.0
        ),
        "mean_teacher_rounded_request_count": (
            float(np.mean(teacher_rounded_counts))
            if teacher_rounded_counts else 0.0
        ),
        "mean_gnn_rounded_request_count": (
            float(np.mean(gnn_rounded_counts))
            if gnn_rounded_counts else 0.0
        ),
        "mean_teacher_rounded_expected_mass": (
            float(np.mean(teacher_rounded_mass))
            if teacher_rounded_mass else 0.0
        ),
        "mean_gnn_rounded_expected_mass": (
            float(np.mean(gnn_rounded_mass))
            if gnn_rounded_mass else 0.0
        ),
        "mean_gnn_to_teacher_rounded_mass_ratio": (
            float(np.mean(rounded_mass_ratios))
            if rounded_mass_ratios else 0.0
        ),
        "mean_teacher_rounded_completion_latency": (
            float(np.mean(teacher_rounded_latency))
            if teacher_rounded_latency else 0.0
        ),
        "mean_gnn_rounded_completion_latency": (
            float(np.mean(gnn_rounded_latency))
            if gnn_rounded_latency else 0.0
        ),
        "mean_rounded_selection_jaccard": (
            float(np.mean(rounded_jaccard)) if rounded_jaccard else 0.0
        ),
        "mean_rounded_request_jaccard": (
            float(np.mean(rounded_request_jaccard))
            if rounded_request_jaccard else 0.0
        ),
        "rounded_feasible_rate": (
            float(np.mean(rounded_feasible)) if rounded_feasible else 0.0
        ),
    }


def _mean_supervised_loss(
    model: nn.Module,
    samples: Sequence[PilotSample],
    *,
    objective_weight: float,
    constraint_weight: float,
    request_mass_weight: float,
    candidate_distribution_weight: float,
    prepared_cache: Mapping[int, _PreparedLossTensors] | None = None,
) -> float:
    if not samples:
        return float("inf")
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for sample in samples:
            prediction = model(
                sample.graph, steps=len(sample.trajectory.points)
            )
            loss, _ = _sample_loss(
                prediction,
                sample,
                objective_weight=objective_weight,
                constraint_weight=constraint_weight,
                request_mass_weight=request_mass_weight,
                candidate_distribution_weight=candidate_distribution_weight,
                prepared=(
                    None
                    if prepared_cache is None
                    else prepared_cache.get(id(sample))
                ),
            )
            value = float(loss.detach().cpu())
            if not np.isfinite(value):
                return float("inf")
            values.append(value)
    return float(np.mean(values))


def _topology_digest(sample: PilotSample) -> str:
    payload = json.dumps(
        {
            "topology": sample.topology,
            "nodes": sample.node_count,
            "edges": sample.topology_signature,
        },
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _sample_protocol(samples: Sequence[PilotSample]) -> dict[str, object]:
    return {
        "samples": len(samples),
        "topology_families": sorted({item.topology for item in samples}),
        "node_counts": sorted({item.node_count for item in samples}),
        "unique_topologies": len({
            (item.topology, item.node_count, item.topology_signature)
            for item in samples
        }),
        "topology_digests": sorted({_topology_digest(item) for item in samples}),
        "mean_raw_ipm_points": float(np.mean([
            len(item.trajectory.raw_points) for item in samples
        ])),
        "mean_solver_iterations": float(np.mean([
            item.trajectory.solver_iterations for item in samples
        ])),
    }


def _save_checkpoint(
    path: Path,
    model: TELGENPaperGNN,
    *,
    report: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": 5,
        "model_class": "TELGENPaperGNN",
        "method": "single_stage_delay_ipm_trajectory_with_shared_rounding",
        "objective": "expected_censored_completion_latency",
        "telgen_reference_commit": TELGEN_REFERENCE_COMMIT,
        "model_config": {
            "hidden_dim": model.hidden_dim,
            "inner_layers": model.inner_layers,
            "message_mlp_layers": model.message_mlp_layers,
            "prediction_layers": model.prediction_layers,
            "normalization": model.normalization,
            "dropout": model.dropout,
        },
        "inference_steps": int(report["ipm_steps"]),
        "state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "training_protocol": report["data_protocol"],
        "best_epoch": report["best_epoch"],
        "best_validation_loss": report["best_validation_loss"],
        "decoder": {
            "name": "shared_capacity_safe_rounding",
            "admission_threshold": 0.5,
            "admission_mass": "unscaled request mass",
            "candidate_ranking": "feasible-prefix-scaled candidate value",
            "teacher_and_gnn_share_decoder": True,
        },
    }, path)


def _resolve_device(name: str) -> torch.device:
    """Resolve the requested training device without silently falling back."""

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def run_pilot(
    *,
    train_samples: int,
    validation_samples: int | None = None,
    test_samples: int,
    epochs: int,
    seed: int,
    data_seed: int | None = None,
    hidden_dim: int = 180,
    inner_layers: int = 2,
    objective_weight: float = 3.43,
    constraint_weight: float = 5.8,
    request_mass_weight: float = 2.0,
    candidate_distribution_weight: float = 0.5,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.0,
    message_mlp_layers: int = 4,
    prediction_layers: int = 4,
    train_topology: str = "waxman",
    train_nodes: int | Sequence[int] = (10, 12, 14),
    test_topology: str | None = None,
    test_nodes: int | Sequence[int] = (18, 20),
    cross_topology: str | None = "barabasi_albert",
    cross_nodes: int | Sequence[int] | None = None,
    cross_samples: int | None = None,
    train_endpoint_mode: str = "uniform_random",
    test_endpoint_mode: str | None = None,
    request_count: int = 3,
    horizon: int = 6,
    path_count: int = 2,
    construction_plan_count: int = 3,
    ipm_steps: int = 16,
    fixed_training_topology: bool = False,
    patience: int = 40,
    min_delta: float = 1e-7,
    checkpoint_path: Path | None = None,
    dataset_cache_path: Path | None = None,
    device: str | torch.device = "auto",
) -> dict[str, object]:
    """Train and evaluate one paper-aligned TELGEN adaptation.

    The default protocol trains on many Waxman graphs and several Waxman
    sizes, then tests on larger unseen Waxman graphs.  This is what "one
    topology training" means in the original paper: one topology family, not
    one fixed graph.  A Barabasi-Albert result is reported separately as a
    cross-family stress test and is not attributed to TELGEN's original claim.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    resolved_device = (
        _resolve_device(device) if isinstance(device, str) else device
    )
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if train_samples < 1 or test_samples < 1 or epochs < 1:
        raise ValueError("sample counts and epochs must be positive")
    if validation_samples is None:
        validation_samples = max(1, train_samples // 4)
    if validation_samples < 1 or patience < 1:
        raise ValueError("validation_samples/patience must be positive")
    if min_delta < 0.0:
        raise ValueError("min_delta must be non-negative")
    if (
        objective_weight < 0.0
        or constraint_weight < 0.0
        or request_mass_weight < 0.0
        or candidate_distribution_weight < 0.0
    ):
        raise ValueError("loss weights must be non-negative")
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("optimizer configuration is invalid")
    resolved_test_topology = test_topology or train_topology
    resolved_test_endpoint_mode = test_endpoint_mode or train_endpoint_mode
    resolved_cross_nodes = cross_nodes or test_nodes
    resolved_cross_samples = test_samples if cross_samples is None else cross_samples
    resolved_data_seed = seed if data_seed is None else int(data_seed)
    if cross_topology is not None and resolved_cross_samples < 1:
        raise ValueError("cross_samples must be positive")

    started_at = time.perf_counter()
    data_started_at = time.perf_counter()
    cache_contract = {
        "schema_version": 3,
        "data_seed": resolved_data_seed,
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "test_samples": test_samples,
        "cross_samples": (
            resolved_cross_samples if cross_topology is not None else 0
        ),
        "train_topology": train_topology,
        "train_endpoint_mode": train_endpoint_mode,
        "train_nodes": list(_as_node_counts(train_nodes)),
        "test_topology": resolved_test_topology,
        "test_endpoint_mode": resolved_test_endpoint_mode,
        "test_nodes": list(_as_node_counts(test_nodes)),
        "cross_topology": cross_topology,
        "cross_nodes": (
            list(_as_node_counts(resolved_cross_nodes))
            if cross_topology is not None else None
        ),
        "request_count": request_count,
        "horizon": horizon,
        "path_count": path_count,
        "construction_plan_count": construction_plan_count,
        "ipm_steps": ipm_steps,
        "fixed_training_topology": fixed_training_topology,
    }
    cache_loaded = False
    if dataset_cache_path is not None and dataset_cache_path.is_file():
        print(f"loading IPM teacher cache: {dataset_cache_path}", flush=True)
        cached = _load_pilot_cache(dataset_cache_path)
        if not isinstance(cached, dict) or cached.get("contract") != cache_contract:
            raise ValueError("IPM teacher cache does not match this run")
        train = tuple(cached["train"])
        validation = tuple(cached["validation"])
        same_family_test = tuple(cached["same_family_test"])
        cross_family_test = tuple(cached["cross_family_test"])
        cache_loaded = True
    else:
        print(
            "generating IPM teacher data: "
            f"train={train_samples}, validation={validation_samples}, "
            f"same_family_test={test_samples}, "
            f"cross_family_test={resolved_cross_samples if cross_topology else 0}",
            flush=True,
        )
        train = make_samples(
            topology=train_topology,
            node_count=train_nodes,
            sample_count=train_samples,
            seed=resolved_data_seed + 1000,
            topology_seed=resolved_data_seed + 7000,
            fixed_topology=fixed_training_topology,
            request_count=request_count,
            horizon=horizon,
            path_count=path_count,
            construction_plan_count=construction_plan_count,
            outer_steps=ipm_steps,
            endpoint_mode=train_endpoint_mode,
        )
        validation = make_samples(
            topology=train_topology,
            node_count=train_nodes,
            sample_count=validation_samples,
            seed=resolved_data_seed + 3000,
            topology_seed=resolved_data_seed + 9000,
            fixed_topology=False,
            request_count=request_count,
            horizon=horizon,
            path_count=path_count,
            construction_plan_count=construction_plan_count,
            outer_steps=ipm_steps,
            endpoint_mode=train_endpoint_mode,
        )
        same_family_test = make_samples(
            topology=resolved_test_topology,
            node_count=test_nodes,
            sample_count=test_samples,
            seed=resolved_data_seed + 5000,
            topology_seed=resolved_data_seed + 11_000,
            fixed_topology=False,
            request_count=request_count,
            horizon=horizon,
            path_count=path_count,
            construction_plan_count=construction_plan_count,
            outer_steps=ipm_steps,
            endpoint_mode=resolved_test_endpoint_mode,
        )
        cross_family_test = (
            make_samples(
                topology=cross_topology,
                node_count=resolved_cross_nodes,
                sample_count=resolved_cross_samples,
                seed=resolved_data_seed + 6000,
                topology_seed=resolved_data_seed + 13_000,
                fixed_topology=False,
                request_count=request_count,
                horizon=horizon,
                path_count=path_count,
                construction_plan_count=construction_plan_count,
                outer_steps=ipm_steps,
                endpoint_mode=resolved_test_endpoint_mode,
            )
            if cross_topology is not None
            else ()
        )
        if dataset_cache_path is not None:
            dataset_cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_cache = dataset_cache_path.with_suffix(
                dataset_cache_path.suffix + ".tmp"
            )
            with temporary_cache.open("wb") as handle:
                pickle.dump({
                    "contract": cache_contract,
                    "train": train,
                    "validation": validation,
                    "same_family_test": same_family_test,
                    "cross_family_test": cross_family_test,
                }, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary_cache, dataset_cache_path)
    data_generation_seconds = time.perf_counter() - data_started_at
    print(
        "teacher data ready: "
        f"seconds={data_generation_seconds:.3f}, "
        f"device={resolved_device}",
        flush=True,
    )

    model = TELGENPaperGNN(
        hidden_dim=hidden_dim,
        inner_layers=inner_layers,
        message_mlp_layers=message_mlp_layers,
        prediction_layers=prediction_layers,
    ).to(resolved_device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    # Keep compact labels and request partitions resident on the device.  The
    # dense LP matrix is intentionally left uncached to avoid pinning several
    # gigabytes for a large training set; it is materialized only for the one
    # sample currently contributing a loss.
    prepared_loss_cache = {
        id(sample): _prepare_loss_tensors(
            sample,
            resolved_device,
            discount=0.7,
            cache_matrix=False,
        )
        for sample in (*train, *validation)
    }
    untrained = {
        "train": _evaluate(model, train),
        "validation": _evaluate(model, validation),
        "same_family_scale_test": _evaluate(model, same_family_test),
        "cross_family_stress_test": (
            _evaluate(model, cross_family_test) if cross_family_test else None
        ),
    }
    # Do not retain evaluation-only graphs while the training cache is being
    # populated; this keeps peak GPU residency bounded on large topologies.
    model.clear_graph_cache()

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, object]] = []
    last_epoch = 0
    rng = random.Random(seed)
    training_started_at = time.perf_counter()
    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        model.train()
        order = list(train)
        rng.shuffle(order)
        losses: list[float] = []
        for sample in order:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                sample.graph, steps=len(sample.trajectory.points)
            )
            loss, _ = _sample_loss(
                prediction,
                sample,
                objective_weight=objective_weight,
                constraint_weight=constraint_weight,
                request_mass_weight=request_mass_weight,
                candidate_distribution_weight=candidate_distribution_weight,
                prepared=prepared_loss_cache[id(sample)],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_loss = _mean_supervised_loss(
            model,
            validation,
            objective_weight=objective_weight,
            constraint_weight=constraint_weight,
            request_mass_weight=request_mass_weight,
            candidate_distribution_weight=candidate_distribution_weight,
            prepared_cache=prepared_loss_cache,
        )
        if validation_loss < best_validation_loss - min_delta:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        should_record = (
            epoch == 1
            or epoch == epochs
            or epoch % max(1, epochs // 5) == 0
        )
        if should_record:
            history.append({
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
                "train": _evaluate(model, train),
                "validation": _evaluate(model, validation),
            })
        print(
            f"epoch={epoch}/{epochs} "
            f"train_loss={float(np.mean(losses)):.8f} "
            f"validation_loss={validation_loss:.8f} "
            f"best_epoch={best_epoch} "
            f"stale_epochs={stale_epochs} "
            f"elapsed_seconds={time.perf_counter() - training_started_at:.3f}",
            flush=True,
        )
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    training_seconds = time.perf_counter() - training_started_at

    final = {
        "epoch": best_epoch,
        "validation_loss": best_validation_loss,
        "train": _evaluate(model, train),
        "validation": _evaluate(model, validation),
        "same_family_scale_test": _evaluate(model, same_family_test),
        "cross_family_stress_test": (
            _evaluate(model, cross_family_test) if cross_family_test else None
        ),
    }
    report: dict[str, object] = {
        "method": (
            "TELGEN IPM-trajectory GNN for the single-stage expected-delay LP"
        ),
        "reference": {
            "paper": TELGEN_REFERENCE_PAPER,
            "official_repository": "https://github.com/aelitazhou/TELGEN",
            "official_commit": TELGEN_REFERENCE_COMMIT,
        },
        "paper_alignment": {
            "teacher": "SciPy linprog interior-point callback primal trajectory",
            "trajectory_sampling": "official TELGEN SubSample rule",
            "graph": "variable/inequality-constraint/objective tripartite graph",
            "relations": 6,
            "outer_loop": "shared learned IPM iterations",
            "inner_loop": f"{inner_layers} shared GCN layers per outer step",
            "node_features": "row/column/objective mean and standard deviation",
            "edge_features": "A, c, and b coefficients",
            "loss": (
                "discounted primal + reduced-delay objective gap + "
                "normalized constraint violation + request-mass and "
                "within-request distribution supervision"
            ),
            "readout": (
                "request-mass sigmoid times within-request normalized "
                "candidate softplus weights"
            ),
            "request_uniqueness": (
                "satisfied structurally at every learned IPM iteration"
            ),
            "decoder": {
                "name": "shared_capacity_safe_rounding",
                "admission_threshold": 0.5,
                "admission_mass": "unscaled request mass",
                "candidate_ranking": "feasible-prefix-scaled candidate value",
                "teacher_and_gnn_share_decoder": True,
            },
        },
        "quantum_adaptation": {
            "variable": "request + path + swap tree + start slot candidate",
            "constraint": "request uniqueness or resource-time capacity row",
            "objective": "single-stage expected censored completion latency",
            "construction_candidates": construction_plan_count,
            "path_candidates": path_count,
            "physical_execution": "not part of this LP-learning pilot",
        },
        "claim_boundary": {
            "same_family_scale_transfer": (
                "aligned with TELGEN's original training/testing protocol"
            ),
            "cross_family_transfer": (
                "additional stress test; not guaranteed by the original paper"
            ),
            "discrete_quantum_schedule": (
                "produced by the same deterministic capacity-safe rounding "
                "for the IPM teacher and GNN output"
            ),
            "official_replication": (
                "algorithmically aligned adaptation, not byte-for-byte reproduction"
            ),
        },
        "training_distribution": (
            "single fixed graph"
            if fixed_training_topology
            else "one topology family with multiple independent graphs and scales"
        ),
        "train_topology": train_topology,
        "train_endpoint_mode": train_endpoint_mode,
        "seed": seed,
        "data_seed": resolved_data_seed,
        "train_nodes": list(_as_node_counts(train_nodes)),
        "test_topology": resolved_test_topology,
        "test_endpoint_mode": resolved_test_endpoint_mode,
        "test_nodes": list(_as_node_counts(test_nodes)),
        "cross_topology": cross_topology,
        "cross_nodes": (
            list(_as_node_counts(resolved_cross_nodes))
            if cross_topology is not None else None
        ),
        "request_count": request_count,
        "horizon": horizon,
        "path_count": path_count,
        "construction_plan_count": construction_plan_count,
        "ipm_steps": ipm_steps,
        "data_protocol": {
            "train": _sample_protocol(train),
            "validation": _sample_protocol(validation),
            "same_family_scale_test": _sample_protocol(same_family_test),
            "cross_family_stress_test": (
                _sample_protocol(cross_family_test)
                if cross_family_test else None
            ),
        },
        "epochs": epochs,
        "epochs_ran": last_epoch,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "early_stopped": bool(last_epoch < epochs),
        "patience": patience,
        "min_delta": min_delta,
        "hidden_dim": hidden_dim,
        "inner_layers": inner_layers,
        "message_mlp_layers": message_mlp_layers,
        "prediction_layers": prediction_layers,
        "objective_weight": objective_weight,
        "constraint_weight": constraint_weight,
        "request_mass_weight": request_mass_weight,
        "candidate_distribution_weight": candidate_distribution_weight,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "device": str(resolved_device),
        "dataset_cache": (
            str(dataset_cache_path) if dataset_cache_path is not None else None
        ),
        "dataset_cache_loaded": cache_loaded,
        "data_generation_seconds": data_generation_seconds,
        "training_seconds": training_seconds,
        "total_seconds": time.perf_counter() - started_at,
        "untrained": untrained,
        "final": final,
        "history": history,
        "teacher_reference": {
            "train_mean_optimal_delay_reduction": float(np.mean([
                -sample.trajectory.lp_optimum for sample in train
            ])),
            "validation_mean_optimal_delay_reduction": float(np.mean([
                -sample.trajectory.lp_optimum for sample in validation
            ])),
            "same_family_mean_optimal_delay_reduction": float(np.mean([
                -sample.trajectory.lp_optimum for sample in same_family_test
            ])),
            "train_mean_optimal_expected_censored_delay": float(np.mean([
                sample.trajectory.total_lp_optimum
                * sample.graph.objective_scale
                for sample in train
            ])),
            "validation_mean_optimal_expected_censored_delay": float(np.mean([
                sample.trajectory.total_lp_optimum
                * sample.graph.objective_scale
                for sample in validation
            ])),
            "same_family_mean_optimal_expected_censored_delay": float(np.mean([
                sample.trajectory.total_lp_optimum
                * sample.graph.objective_scale
                for sample in same_family_test
            ])),
            "train_final_mean_normalized_violation": float(np.mean([
                sample.trajectory.normalized_violations[-1]
                for sample in train
            ])),
        },
    }
    if checkpoint_path is not None:
        _save_checkpoint(checkpoint_path, model, report=report)
        report["checkpoint"] = str(checkpoint_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/telgen_ipm_pilot.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/telgen_ipm_pilot.pt"),
    )
    parser.add_argument("--train-samples", type=int, default=24)
    parser.add_argument("--validation-samples", type=int, default=8)
    parser.add_argument("--test-samples", type=int, default=8)
    parser.add_argument("--cross-samples", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--hidden-dim", type=int, default=180)
    parser.add_argument("--inner-layers", type=int, default=2)
    parser.add_argument("--objective-weight", type=float, default=3.43)
    parser.add_argument("--constraint-weight", type=float, default=5.8)
    parser.add_argument("--request-mass-weight", type=float, default=2.0)
    parser.add_argument(
        "--candidate-distribution-weight", type=float, default=0.5
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--message-mlp-layers", type=int, default=4)
    parser.add_argument("--prediction-layers", type=int, default=4)
    parser.add_argument("--train-topology", type=str, default="waxman")
    parser.add_argument(
        "--train-endpoint-mode",
        choices=("uniform_random", "cut_hotspot"),
        default="uniform_random",
    )
    parser.add_argument(
        "--train-nodes", type=int, nargs="+", default=[10, 12, 14]
    )
    parser.add_argument("--test-topology", type=str, default="waxman")
    parser.add_argument(
        "--test-endpoint-mode",
        choices=("uniform_random", "cut_hotspot"),
    )
    parser.add_argument(
        "--test-nodes", type=int, nargs="+", default=[18, 20]
    )
    parser.add_argument(
        "--cross-topology", type=str, default="barabasi_albert"
    )
    parser.add_argument(
        "--cross-nodes", type=int, nargs="+", default=[18, 20]
    )
    parser.add_argument("--request-count", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--path-count", type=int, default=2)
    parser.add_argument("--construction-plans", type=int, default=3)
    parser.add_argument("--ipm-steps", type=int, default=16)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--min-delta", type=float, default=1e-7)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--dataset-cache",
        type=Path,
        help="trusted local teacher cache generated by this program",
    )
    parser.add_argument(
        "--single-fixed-training-topology",
        action="store_true",
        help="controlled ablation; not the paper-style default protocol",
    )
    parser.add_argument(
        "--no-cross-family-test",
        action="store_true",
        help="skip the additional cross-topology-family stress test",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cross_topology = None if args.no_cross_family_test else args.cross_topology
    report = run_pilot(
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        cross_samples=args.cross_samples,
        epochs=args.epochs,
        seed=args.seed,
        data_seed=args.data_seed,
        hidden_dim=args.hidden_dim,
        inner_layers=args.inner_layers,
        objective_weight=args.objective_weight,
        constraint_weight=args.constraint_weight,
        request_mass_weight=args.request_mass_weight,
        candidate_distribution_weight=args.candidate_distribution_weight,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        message_mlp_layers=args.message_mlp_layers,
        prediction_layers=args.prediction_layers,
        train_topology=args.train_topology,
        train_endpoint_mode=args.train_endpoint_mode,
        train_nodes=args.train_nodes,
        test_topology=args.test_topology,
        test_endpoint_mode=args.test_endpoint_mode,
        test_nodes=args.test_nodes,
        cross_topology=cross_topology,
        cross_nodes=args.cross_nodes,
        request_count=args.request_count,
        horizon=args.horizon,
        path_count=args.path_count,
        construction_plan_count=args.construction_plans,
        ipm_steps=args.ipm_steps,
        fixed_training_topology=args.single_fixed_training_topology,
        patience=args.patience,
        min_delta=args.min_delta,
        checkpoint_path=args.checkpoint,
        dataset_cache_path=args.dataset_cache,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["final"], ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    print(f"checkpoint: {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
