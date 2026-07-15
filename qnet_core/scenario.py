"""Deterministic workload construction shared by training and baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spec import EpisodeSpec, PhysicalConfig, RequestSpec


@dataclass(frozen=True)
class ScenarioConfig:
    request_count: int = 4
    min_hops: int = 2
    max_hops: int = 5
    ttl: int = 20
    horizon: int = 100
    arrival_rate: float = 1.0
    physical: PhysicalConfig = PhysicalConfig()


def make_episode(config: ScenarioConfig, seed: int) -> EpisodeSpec:
    if config.request_count < 1 or config.min_hops < 1 or config.max_hops < config.min_hops:
        raise ValueError("invalid request or hop configuration")
    if config.arrival_rate <= 0:
        raise ValueError("arrival_rate must be positive")
    nodes = tuple(range(config.max_hops + 1))
    edges = tuple((node, node + 1) for node in range(config.max_hops))
    if config.request_count == 1:
        hops = [config.max_hops]
    else:
        span = config.max_hops - config.min_hops
        hops = [
            config.min_hops + round(span * index / (config.request_count - 1))
            for index in range(config.request_count)
        ]
    rng = np.random.default_rng(seed)
    rng.shuffle(hops)
    # A homogeneous Poisson arrival process: exponential inter-arrival times,
    # discretized into physical slots. Multiple requests may arrive together.
    inter_arrivals = rng.exponential(1.0 / config.arrival_rate, config.request_count)
    arrivals = np.floor(np.cumsum(inter_arrivals)).astype(int)
    requests = tuple(
        RequestSpec(
            f"r{index}", 0, int(hops[index]),
            arrival=int(arrivals[index]), ttl=config.ttl,
        )
        for index in range(config.request_count)
    )
    horizon = max(config.horizon, int(arrivals[-1]) + config.ttl)
    return EpisodeSpec(seed, nodes, edges, requests, horizon, config.physical)
