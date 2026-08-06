"""Immutable inputs owned by the routing and planning layer."""

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
class PlanningSpec:
    """Topology and demand data visible to routing, without physical config."""

    seed: int
    nodes: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    requests: tuple[RequestSpec, ...]
    horizon: int

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        node_set = set(self.nodes)
        if any(u not in node_set or v not in node_set or u == v for u, v in self.edges):
            raise ValueError("edges must connect distinct declared nodes")
