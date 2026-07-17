"""Small, slot-reset Q-CAST simulator suitable for paper reproductions."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .allocation import allocate_recovery_paths, geda_allocate
from .model import MajorReservation, QCastTopology, RecoveryReservation, SDPair
from .recovery import LaneOutcome, recover_lane


@dataclass(frozen=True)
class SimulationConfig:
    path_cap: int = 200
    recovery: bool = True
    compatibility: str = "author_code"
    link_state_range: int | None = None
    swap_probability: float | None = None


@dataclass(frozen=True)
class SlotResult:
    throughput: int
    successful_pairs: int
    pairs: tuple[SDPair, ...]
    major_paths: tuple[MajorReservation, ...]
    recovery_paths: tuple[RecoveryReservation, ...]
    lane_outcomes: tuple[LaneOutcome, ...]

    @property
    def eps(self) -> int:
        return self.throughput


def sample_sd_pairs(topology: QCastTopology, pair_count: int, rng: random.Random) -> tuple[SDPair, ...]:
    """Sample ``2m`` distinct nodes and pair adjacent shuffled entries."""

    if pair_count < 0 or 2 * pair_count > len(topology.nodes):
        raise ValueError("pair_count requires 2m distinct topology nodes")
    nodes = list(topology.nodes)
    rng.shuffle(nodes)
    return tuple(SDPair(nodes[index], nodes[index + 1], index // 2)
                 for index in range(0, 2 * pair_count, 2))


def _channel_successes(topology: QCastTopology, majors, recoveries, rng: random.Random):
    refs = {ref for major in majors for ref in major.channels}
    refs.update(ref for recovery in recoveries for ref in recovery.channels)
    return {
        ref: rng.random() < topology.channel_probability(ref)
        for ref in refs
    }


def run_slot(
    topology: QCastTopology,
    sd_pairs: Sequence[SDPair | Sequence[int]],
    rng: random.Random | None = None,
    *,
    config: SimulationConfig | None = None,
) -> SlotResult:
    """Run P1--P4 for one independent time slot and reset via fresh residuals."""

    rng = random.Random(0) if rng is None else rng
    config = SimulationConfig() if config is None else config
    residual = topology.residual()
    majors = geda_allocate(
        residual, sd_pairs, path_cap=config.path_cap,
        swap_probability=config.swap_probability, mutate=True,
    )
    recoveries: list[RecoveryReservation] = []
    if config.recovery:
        link_range = topology.link_state_range if config.link_state_range is None else config.link_state_range
        for major in majors:
            recoveries.extend(allocate_recovery_paths(
                residual, major, link_range,
                swap_probability=config.swap_probability,
            ))
    outcomes_map = _channel_successes(topology, majors, recoveries, rng)
    outcomes: list[LaneOutcome] = []
    successful_pairs = 0
    successful_pair_keys: set[tuple[int, int]] = set()
    throughput = 0
    for major in majors:
        owned = tuple(item for item in recoveries if item.major == major)
        q = topology.swap_probability if config.swap_probability is None else config.swap_probability
        used_channels = set()
        lane_results = tuple(
            recover_lane(
                major, tuple(item for item in owned if item.width > lane),
                outcomes_map, lane, rng=rng, swap_probability=q,
                compatibility=config.compatibility,
                used_channels=used_channels,
            )
            for lane in range(major.width)
        )
        outcomes.extend(lane_results)
        throughput += sum(item.success for item in lane_results)
        if any(item.success for item in lane_results):
            successful_pair_keys.add((major.pair.source, major.pair.destination))
    successful_pairs = len(successful_pair_keys)
    return SlotResult(
        int(throughput), successful_pairs, tuple(
            item if isinstance(item, SDPair) else SDPair(int(item[0]), int(item[1]))
            for item in sd_pairs
        ), tuple(majors), tuple(recoveries), tuple(outcomes),
    )


def run_experiment(
    topology_factory: Callable[[int, random.Random], QCastTopology],
    pair_count: int,
    *,
    topology_count: int = 10,
    slots_per_topology: int = 1000,
    seed: int = 0,
    config: SimulationConfig | None = None,
) -> dict[str, object]:
    """Run independent slots and return summary means/distribution."""

    if topology_count < 1 or slots_per_topology < 1:
        raise ValueError("experiment dimensions must be positive")
    root_rng = random.Random(seed)
    values: list[int] = []
    pair_values: list[int] = []
    for topology_index in range(topology_count):
        topology = topology_factory(topology_index, root_rng)
        for _ in range(slots_per_topology):
            pairs = sample_sd_pairs(topology, pair_count, root_rng)
            result = run_slot(topology, pairs, root_rng, config=config)
            values.append(result.throughput)
            pair_values.append(result.successful_pairs)
    return {
        "throughput_mean": sum(values) / len(values),
        "successful_pairs_mean": sum(pair_values) / len(pair_values),
        "throughput": values,
        "successful_pairs": pair_values,
        "topology_count": topology_count,
        "slots_per_topology": slots_per_topology,
        "pair_count": pair_count,
    }
