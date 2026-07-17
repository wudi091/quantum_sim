"""Immutable episode and physical configuration shared by every algorithm."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RequestSpec:
    id: str
    source: int
    destination: int
    arrival: int = 0
    ttl: int | None = None
    demand_pairs: int = 1

    def __post_init__(self) -> None:
        if self.demand_pairs < 1:
            raise ValueError("demand_pairs must be positive")

    @property
    def deadline(self) -> int | None:
        return None if self.ttl is None else self.arrival + self.ttl


@dataclass(frozen=True)
class PhysicalConfig:
    generation_probability: float = 0.5
    swap_probability: float = 0.5
    memory_capacity: int = 2
    memory_lifetime: int = 100
    initial_fidelity: float = 0.99
    swap_degradation: float = 0.95
    classical_delay_ps: int = 0
    node_memory_capacity: int | None = None
    max_width: int = 1


@dataclass(frozen=True)
class EpisodeSpec:
    seed: int
    nodes: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    requests: tuple[RequestSpec, ...]
    horizon: int
    physical: PhysicalConfig = PhysicalConfig()

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        node_set = set(self.nodes)
        if any(u not in node_set or v not in node_set or u == v for u, v in self.edges):
            raise ValueError("edges must connect distinct declared nodes")
        if not 0 < self.physical.generation_probability <= 1:
            raise ValueError("generation_probability must be in (0, 1]")
        if not 0 <= self.physical.swap_probability <= 1:
            raise ValueError("swap_probability must be in [0, 1]")
        if (self.physical.node_memory_capacity is not None
                and self.physical.node_memory_capacity < 1):
            raise ValueError("node_memory_capacity must be positive when set")
        if self.physical.max_width < 1:
            raise ValueError("max_width must be positive")
        if self.physical.max_width > self.physical.memory_capacity:
            raise ValueError("max_width cannot exceed per-edge memory_capacity")
