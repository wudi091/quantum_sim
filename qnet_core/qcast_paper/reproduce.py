"""Command-line smoke reproduction for the independent Q-CAST simulator."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .model import EdgeSpec, QCastTopology
from .simulator import SimulationConfig, run_experiment
from .topology import AuthorTopologyConfig, generate_author_topology


def demo_topology(index: int, rng: random.Random, *, nodes: int = 20) -> QCastTopology:
    """Build a small connected Waxman-like graph for a quick smoke run."""

    del index
    node_qubits = {node: 12 for node in range(nodes)}
    edges: list[EdgeSpec] = [EdgeSpec(node, node + 1, 3, 0.82) for node in range(nodes - 1)]
    for node in range(nodes):
        for other in range(node + 2, min(nodes, node + 5)):
            if rng.random() < 0.15:
                edges.append(EdgeSpec(node, other, 3, 0.72))
    return QCastTopology(node_qubits, edges, swap_probability=0.9, link_state_range=3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--topologies", type=int, default=1)
    parser.add_argument("--slots", type=int, default=10)
    parser.add_argument("--seed", type=int, default=19900111)
    parser.add_argument("--link-probability", type=float, default=0.6)
    parser.add_argument("--swap-probability", type=float, default=0.9)
    parser.add_argument("--link-state-range", type=int, default=3)
    parser.add_argument("--average-degree", type=int, default=6)
    parser.add_argument("--no-recovery", action="store_true")
    parser.add_argument("--compatibility", choices=("author_code", "corrected"), default="author_code")
    parser.add_argument("--paper-reference", action="store_true",
                        help="use the author Topo.generate model (n=100, d=6, Ep=.6)")
    parser.add_argument("--output", type=Path, default=None,
                        help="write the complete JSON result (including slot distributions)")
    parser.add_argument("--summary-only", action="store_true",
                        help="print only aggregate means/counts to stdout")
    args = parser.parse_args(argv)
    if args.paper_reference:
        paper_config = AuthorTopologyConfig(
            node_count=args.nodes or 100, average_degree=args.average_degree,
            target_link_probability=args.link_probability,
            swap_probability=args.swap_probability,
            link_state_range=args.link_state_range,
        )
        factory = lambda index, rng: generate_author_topology(paper_config, rng)
    else:
        factory = lambda index, rng: demo_topology(index, rng, nodes=args.nodes or 20)
    result = run_experiment(
        factory, args.pairs, topology_count=args.topologies,
        slots_per_topology=args.slots, seed=args.seed,
        config=SimulationConfig(
            recovery=not args.no_recovery, compatibility=args.compatibility,
        ),
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.summary_only:
        summary = {key: value for key, value in result.items()
                   if key not in {"throughput", "successful_pairs"}}
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
