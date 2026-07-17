"""Data model for the author Q-CAST simulator.

The public shared environment stores abstract plans.  Q-CAST, in contrast,
allocates individual physical channels and consumes endpoint/interior qubits
in P2.  This module keeps those resources explicit and intentionally has no
dependency on the shared environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


def canonical_edge(u: int, v: int) -> tuple[int, int]:
    if u == v:
        raise ValueError("self-loops are not valid Q-CAST channels")
    return (u, v) if u < v else (v, u)


@dataclass(frozen=True, order=True)
class ChannelRef:
    """Globally identified physical channel, as in the Kotlin simulator."""

    edge: tuple[int, int]
    channel_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "edge", canonical_edge(*self.edge))

    @property
    def id(self) -> int:
        return self.channel_id


@dataclass(frozen=True)
class Channel:
    ref: ChannelRef
    probability: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("channel probability must be in [0, 1]")

    @property
    def edge(self) -> tuple[int, int]:
        return self.ref.edge

    @property
    def p(self) -> float:
        return self.probability


@dataclass(frozen=True)
class EdgeSpec:
    """One undirected edge with ``width`` parallel channels.

    ``probability`` may be a scalar (all channels equal) or a sequence with
    one value per channel.  The latter is useful for heterogeneous-distance
    experiments and is accepted by :class:`QCastTopology`.
    """

    u: int
    v: int
    width: int
    probability: float | Sequence[float]

    def __post_init__(self) -> None:
        if self.u == self.v or self.width < 1:
            raise ValueError("an edge needs two distinct endpoints and width >= 1")
        probs = self.probabilities
        if len(probs) != self.width:
            raise ValueError("one channel probability is required for each width")

    @property
    def edge(self) -> tuple[int, int]:
        return canonical_edge(self.u, self.v)

    @property
    def probabilities(self) -> tuple[float, ...]:
        if isinstance(self.probability, (int, float)):
            values = (float(self.probability),) * self.width
        else:
            values = tuple(float(value) for value in self.probability)
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("channel probabilities must be in [0, 1]")
        return values


class QCastTopology:
    """Immutable topology and per-channel generation probabilities."""

    def __init__(
        self,
        node_qubits: Mapping[int, int],
        edges: Iterable[EdgeSpec | tuple],
        swap_probability: float = 0.9,
        link_state_range: int = 3,
    ) -> None:
        self.node_qubits = {int(node): int(capacity) for node, capacity in node_qubits.items()}
        if any(capacity < 1 for capacity in self.node_qubits.values()):
            raise ValueError("node qubit capacities must be positive")
        normalised: list[EdgeSpec] = []
        for item in edges:
            if isinstance(item, EdgeSpec):
                spec = item
            else:
                values = tuple(item)
                if len(values) == 4:
                    spec = EdgeSpec(int(values[0]), int(values[1]), int(values[2]), values[3])
                elif len(values) == 3:
                    if isinstance(values[0], (tuple, list)) and len(values[0]) == 2:
                        spec = EdgeSpec(int(values[0][0]), int(values[0][1]), int(values[1]), values[2])
                    else:
                        spec = EdgeSpec(int(values[0]), int(values[1]), int(values[2]), 1.0)
                elif len(values) == 2 and isinstance(values[0], (tuple, list)) and len(values[0]) == 2:
                    spec = EdgeSpec(int(values[0][0]), int(values[0][1]), int(values[1]), 1.0)
                else:
                    raise ValueError("edge tuples must be (u, v, width[, probability])")
            if spec.u not in self.node_qubits or spec.v not in self.node_qubits:
                raise ValueError("edge endpoint is not in node_qubits")
            normalised.append(spec)
        self.edges = tuple(normalised)
        if not 0.0 <= swap_probability <= 1.0:
            raise ValueError("swap_probability must be in [0, 1]")
        if link_state_range < 0:
            raise ValueError("link_state_range must be non-negative")
        self.swap_probability = float(swap_probability)
        self.link_state_range = int(link_state_range)
        channels: dict[ChannelRef, Channel] = {}
        by_edge: dict[tuple[int, int], list[ChannelRef]] = {}
        adjacency: dict[int, list[ChannelRef]] = {node: [] for node in self.node_qubits}
        next_id = 0
        for spec in self.edges:
            edge = spec.edge
            refs: list[ChannelRef] = []
            for probability in spec.probabilities:
                ref = ChannelRef(edge, next_id)
                next_id += 1
                channels[ref] = Channel(ref, probability)
                refs.append(ref)
                adjacency[edge[0]].append(ref)
                adjacency[edge[1]].append(ref)
            by_edge.setdefault(edge, []).extend(refs)
        self.channels = channels
        self.channels_by_edge = {edge: tuple(refs) for edge, refs in by_edge.items()}
        self.adjacency = {node: tuple(refs) for node, refs in adjacency.items()}

    @property
    def nodes(self) -> tuple[int, ...]:
        return tuple(sorted(self.node_qubits))

    @property
    def edge_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(spec.edge for spec in self.edges)

    @property
    def n(self) -> int:
        return len(self.node_qubits)

    @classmethod
    def from_edges(cls, node_qubits: Mapping[int, int], edges: Iterable[EdgeSpec | tuple], **kwargs):
        return cls(node_qubits, edges, **kwargs)

    def residual(self) -> "ResidualResources":
        return ResidualResources(self)

    def channel_probability(self, ref: ChannelRef) -> float:
        return self.channels[ref].probability

    def channels_on_edge(self, edge: tuple[int, int], available_only: bool = False) -> tuple[ChannelRef, ...]:
        """Return physical channels for an edge (topology has no reservations)."""
        del available_only
        return self.channels_by_edge.get(canonical_edge(*edge), ())


class ResidualResources:
    """Mutable residual node/channel resources used by P2 allocation."""

    def __init__(self, topology: QCastTopology | "ResidualResources") -> None:
        if isinstance(topology, ResidualResources):
            self.node_capacity = topology.node_capacity.copy()
            self.node_remaining = topology.node_remaining.copy()
            self.channels = topology.channels.copy()
            self.channels_by_edge = {
                edge: tuple(refs) for edge, refs in topology.channels_by_edge.items()
            }
            self.available = set(topology.available)
            self.swap_probability = topology.swap_probability
            self.link_state_range = topology.link_state_range
            return
        self.node_capacity = topology.node_qubits.copy()
        self.node_remaining = topology.node_qubits.copy()
        self.channels = topology.channels.copy()
        self.channels_by_edge = topology.channels_by_edge.copy()
        self.available = set(self.channels)
        self.swap_probability = topology.swap_probability
        self.link_state_range = topology.link_state_range

    def copy(self) -> "ResidualResources":
        return ResidualResources(self)

    @property
    def nodes(self) -> tuple[int, ...]:
        return tuple(sorted(self.node_capacity))

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(self.channels_by_edge))

    def channels_on_edge(self, edge: tuple[int, int], available_only: bool = True) -> tuple[ChannelRef, ...]:
        refs = self.channels_by_edge.get(canonical_edge(*edge), ())
        if available_only:
            return tuple(ref for ref in refs if ref in self.available)
        return refs

    def neighbours(self, node: int, width: int = 1, eligible_nodes: set[int] | None = None) -> tuple[int, ...]:
        result: set[int] = set()
        for ref in self._incident(node):
            if len(self.channels_on_edge(ref.edge)) < width:
                continue
            other = ref.edge[1] if ref.edge[0] == node else ref.edge[0]
            if eligible_nodes is None or other in eligible_nodes:
                result.add(other)
        return tuple(sorted(result))

    def _incident(self, node: int) -> tuple[ChannelRef, ...]:
        refs: list[ChannelRef] = []
        for edge, channels in self.channels_by_edge.items():
            if node in edge:
                refs.extend(channels)
        return tuple(refs)

    def path_channels(self, path: Sequence[int], width: int, *, available_only: bool = True) -> tuple[ChannelRef, ...]:
        if len(path) < 2 or width < 1:
            return ()
        selected: list[ChannelRef] = []
        for u, v in zip(path, path[1:]):
            refs = self.channels_on_edge((u, v), available_only)
            if len(refs) < width:
                return ()
            selected.extend(refs[:width])
        return tuple(selected)

    def can_reserve(self, path: Sequence[int], width: int) -> bool:
        if len(path) < 2 or width < 1:
            return False
        if any(node not in self.node_remaining for node in path):
            return False
        if self.node_remaining[path[0]] < width or self.node_remaining[path[-1]] < width:
            return False
        for node in path[1:-1]:
            if self.node_remaining[node] < 2 * width:
                return False
        return all(len(self.channels_on_edge((u, v))) >= width for u, v in zip(path, path[1:]))

    def reserve(self, path: Sequence[int], width: int) -> tuple[ChannelRef, ...]:
        if not self.can_reserve(path, width):
            raise ValueError("path does not fit residual node/channel resources")
        channels = self.path_channels(path, width)
        for ref in channels:
            self.available.remove(ref)
        self.node_remaining[path[0]] -= width
        self.node_remaining[path[-1]] -= width
        for node in path[1:-1]:
            self.node_remaining[node] -= 2 * width
        return channels

    def release(self, path: Sequence[int], width: int, channels: Iterable[ChannelRef] | None = None) -> None:
        refs = tuple(channels) if channels is not None else self.path_channels(path, width, available_only=False)
        self.available.update(refs)
        self.node_remaining[path[0]] += width
        self.node_remaining[path[-1]] += width
        for node in path[1:-1]:
            self.node_remaining[node] += 2 * width


@dataclass(frozen=True)
class SDPair:
    source: int
    destination: int
    id: str | int | None = None

    @property
    def src(self) -> int:
        return self.source

    @property
    def dst(self) -> int:
        return self.destination


@dataclass(frozen=True)
class PathCandidate:
    path: tuple[int, ...]
    width: int
    expected_throughput: float

    @property
    def ext(self) -> float:
        return self.expected_throughput

    @property
    def hops(self) -> int:
        return len(self.path) - 1


@dataclass(frozen=True)
class MajorReservation:
    pair: SDPair
    path: tuple[int, ...]
    width: int
    expected_throughput: float
    channels: tuple[ChannelRef, ...]

    @property
    def ext(self) -> float:
        return self.expected_throughput

    @property
    def hops(self) -> int:
        return len(self.path) - 1

    @property
    def source(self) -> int:
        return self.pair.source

    @property
    def destination(self) -> int:
        return self.pair.destination


@dataclass(frozen=True)
class RecoveryReservation:
    major: MajorReservation
    path: tuple[int, ...]
    width: int
    expected_throughput: float
    channels: tuple[ChannelRef, ...]
    start_index: int
    end_index: int

    @property
    def ext(self) -> float:
        return self.expected_throughput

    @property
    def hops(self) -> int:
        return len(self.path) - 1
