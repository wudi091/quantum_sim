"""Composition-level episode and physical simulator configuration."""

from __future__ import annotations

from dataclasses import dataclass

from .planning_spec import PlanningSpec, RequestSpec


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
    quantum_distance_m: float = 1000.0
    quantum_attenuation_db_per_m: float = 0.0
    quantum_polarization_fidelity: float = 1.0
    memory_frequency_hz: float = 2e8
    slot_duration_ps: int = 1_000_000
    detector_efficiency: float = 1.0
    bsm_success_probability: float = 1.0
    memory_wavelength_nm: int = 500


@dataclass(frozen=True)
class EpisodeSpec:
    seed: int
    nodes: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    requests: tuple[RequestSpec, ...]
    horizon: int
    physical: PhysicalConfig = PhysicalConfig()

    @property
    def planning(self) -> PlanningSpec:
        """Return the physical-config-free view consumed by routing."""
        return PlanningSpec(
            seed=self.seed,
            nodes=self.nodes,
            edges=self.edges,
            requests=self.requests,
            horizon=self.horizon,
        )

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
        if not 0.5 <= self.physical.initial_fidelity <= 1:
            raise ValueError("initial_fidelity must be in [0.5, 1]")
        if not 0 <= self.physical.swap_degradation <= 1:
            raise ValueError("swap_degradation must be in [0, 1]")
        if self.physical.classical_delay_ps < 0:
            raise ValueError("classical_delay_ps cannot be negative")
        if (self.physical.node_memory_capacity is not None
                and self.physical.node_memory_capacity < 1):
            raise ValueError("node_memory_capacity must be positive when set")
        if self.physical.max_width < 1:
            raise ValueError("max_width must be positive")
        if self.physical.max_width > self.physical.memory_capacity:
            raise ValueError("max_width cannot exceed per-edge memory_capacity")
        if self.physical.memory_capacity < 1:
            raise ValueError("memory_capacity must be positive")
        if self.physical.memory_lifetime < 1:
            raise ValueError("memory_lifetime must be positive")
        if self.physical.quantum_distance_m <= 0:
            raise ValueError("quantum_distance_m must be positive")
        if self.physical.quantum_attenuation_db_per_m < 0:
            raise ValueError("quantum_attenuation_db_per_m cannot be negative")
        if not 0 <= self.physical.quantum_polarization_fidelity <= 1:
            raise ValueError("quantum_polarization_fidelity must be in [0, 1]")
        if self.physical.memory_frequency_hz <= 0:
            raise ValueError("memory_frequency_hz must be positive")
        if self.physical.slot_duration_ps < 1:
            raise ValueError("slot_duration_ps must be positive")
        if not 0 <= self.physical.detector_efficiency <= 1:
            raise ValueError("detector_efficiency must be in [0, 1]")
        if not 0 <= self.physical.bsm_success_probability <= 1:
            raise ValueError("bsm_success_probability must be in [0, 1]")
        if self.physical.memory_wavelength_nm < 1:
            raise ValueError("memory_wavelength_nm must be positive")
