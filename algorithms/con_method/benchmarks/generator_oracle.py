"""Fair offline-generator comparison with the same online MILP oracle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Iterable

from qnet_core.order_milp import (
    MilpNominalPathOrderPlanner,
    MilpNominalPathPlanner,
    MilpReliableMemoryPathOrderPlanner,
    MilpStaticPathOrderPlanner,
)
from qnet_core.order_planners import (
    QCASTFixedOrderPlanner,
    QDDCAFixedOrderPlanner,
)
from qnet_core.order_waxman import WaxmanOrderConfig, make_waxman_order_episode
from qnet_core.order_waxman_benchmark import run_planner_episode

from ..offline_library import (
    GENERATOR_PRESETS,
    build_waxman_selection_context,
    build_waxman_topology_pool,
    compile_structural_topology_library,
    instantiate_con_library_for_episode,
    instantiate_topology_pool_for_episode,
    select_structural_library,
)


DEFAULT_GENERATORS = (
    "canonical",
    "quality",
    "banded",
    "pareto",
    "exact_kcenter",
)

BASELINE_PLANNERS = {
    "qddca_fixed": QDDCAFixedOrderPlanner,
    "qcast_fixed": QCASTFixedOrderPlanner,
}


class _PathIncumbentOrderOracle:
    """Exact path+order oracle seeded by an exact path-only incumbent.

    The path-only action is revalidated by ``select_with_incumbent`` before it
    is used.  It changes enumeration order only: the final path+order result
    still carries the same exact optimality certificate.
    """

    name = "milp_nominal_path_order_with_path_incumbent"

    def __init__(
        self,
        *,
        planning_seeds: tuple[int, ...],
        chance_threshold: float,
        oracle_workers: int,
    ) -> None:
        kwargs = {
            "planning_seeds": planning_seeds,
            "chance_threshold": chance_threshold,
            "oracle_workers": oracle_workers,
        }
        self.path_planner = MilpNominalPathPlanner(**kwargs)
        self.order_planner = MilpNominalPathOrderPlanner(**kwargs)
        self.last_objective = 0
        self.last_solution = None
        self.last_evaluations = 0

    def reset(self, episode_seed: int) -> None:
        self.path_planner.reset(episode_seed)
        self.order_planner.reset(episode_seed)
        self.last_objective = 0
        self.last_solution = None
        self.last_evaluations = 0

    def select(self, snapshot):
        incumbent = self.path_planner.select(snapshot)
        selected = self.order_planner.select_with_incumbent(
            snapshot, incumbent
        )
        self.last_objective = self.order_planner.last_objective
        self.last_solution = self.order_planner.last_solution
        self.last_evaluations = (
            self.path_planner.last_evaluations
            + self.order_planner.last_evaluations
        )
        return selected


def _make_order_oracle(
    planning_seeds: tuple[int, ...],
    *,
    online_selector: str,
    reliability_confidence: float,
    use_path_incumbent: bool,
    oracle_workers: int,
):
    if online_selector == "reliable_memory_milp":
        return MilpReliableMemoryPathOrderPlanner(
            reliability_confidence=reliability_confidence,
        )
    if online_selector in {"static_milp", "static_relaxation"}:
        return MilpStaticPathOrderPlanner(
            planning_seeds=planning_seeds,
            chance_threshold=1.0,
        )
    if online_selector != "exact_scenario_oracle":
        raise ValueError(f"unknown online_selector: {online_selector!r}")
    if use_path_incumbent:
        return _PathIncumbentOrderOracle(
            planning_seeds=planning_seeds,
            chance_threshold=1.0,
            oracle_workers=oracle_workers,
        )
    return MilpNominalPathOrderPlanner(
        planning_seeds=planning_seeds,
        chance_threshold=1.0,
        oracle_workers=oracle_workers,
    )


def _stable_seed(base: int, episode_seed: int, domain: str) -> int:
    payload = f"{domain}|{int(base)}|{int(episode_seed)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _aggregate(rows: list[dict[str, object]]) -> dict[str, float]:
    numeric_keys = tuple(sorted(set.intersection(*(
        {
            key for key, value in row.items()
            if key not in {"episode_seed", "generator"}
            and isinstance(value, (int, float))
        }
        for row in rows
    ))))
    return {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in numeric_keys
    }


def _ranking_key(item: tuple[str, dict[str, float]]):
    name, metrics = item
    return (
        -metrics.get("milp_model_objective_sum", 0.0),
        -metrics.get("executor_completed_count", 0.0),
        metrics.get("timeout_count", 0.0),
        metrics.get("mean_delay_slots", 0.0),
        metrics.get("mean_planning_ms", 0.0),
        name,
    )


def merge_generator_oracle_benchmark_results(
    partial_results: Iterable[dict[str, object]],
    *,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Merge disjoint episode shards into one formal benchmark result.

    Episode physics and planner streams are derived from the stable episode
    seed, so evaluating disjoint seed sets in separate processes is exactly
    equivalent to one serial run.  The merge always rebuilds aggregates,
    deltas, ranking, and the recommendation from raw per-episode rows.
    """

    parts = tuple(partial_results)
    if not parts:
        raise ValueError("partial_results cannot be empty")
    first = parts[0]
    invariant_keys = (
        "protocol",
        "physics_seed_base",
        "planner_seed_base",
        "workload_config",
    )
    generator_names = tuple(first["rows"])
    baseline_names = tuple(first["baseline_rows"])
    ranked_name_set = set(first["ranking"])
    ranked_generator_names = tuple(
        name for name in generator_names if name in ranked_name_set
    )
    for part in parts[1:]:
        for key in invariant_keys:
            if part[key] != first[key]:
                raise ValueError(
                    f"partial benchmark invariant differs: {key}"
                )
        if tuple(part["rows"]) != generator_names:
            raise ValueError("partial benchmark generator sets differ")
        if tuple(part["baseline_rows"]) != baseline_names:
            raise ValueError("partial benchmark baseline sets differ")

    episode_seeds = tuple(sorted(
        int(seed)
        for part in parts
        for seed in part["episode_seeds"]
    ))
    if len(set(episode_seeds)) != len(episode_seeds):
        raise ValueError("partial benchmark episode seeds overlap")
    rows = {
        name: sorted(
            (
                row
                for part in parts
                for row in part["rows"][name]
            ),
            key=lambda row: int(row["episode_seed"]),
        )
        for name in generator_names
    }
    baseline_rows = {
        name: sorted(
            (
                row
                for part in parts
                for row in part["baseline_rows"][name]
            ),
            key=lambda row: int(row["episode_seed"]),
        )
        for name in baseline_names
    }
    for name, values in (*rows.items(), *baseline_rows.items()):
        row_seeds = tuple(int(row["episode_seed"]) for row in values)
        if row_seeds != episode_seeds:
            raise ValueError(
                f"partial benchmark rows do not cover all seeds: {name}"
            )
    topologies = sorted(
        (
            row
            for part in parts
            for row in part["topologies"]
        ),
        key=lambda row: int(row["episode_seed"]),
    )
    if tuple(int(row["episode_seed"]) for row in topologies) != episode_seeds:
        raise ValueError("partial topology rows do not cover all seeds")

    aggregate = {
        name: _aggregate(values) for name, values in rows.items()
    }
    baseline_aggregate = {
        name: _aggregate(values) for name, values in baseline_rows.items()
    }
    ranked_generators = tuple(
        name for name, _ in sorted(
            (
                (name, aggregate[name])
                for name in ranked_generator_names
            ),
            key=_ranking_key,
        )
    )
    canonical_completed = aggregate.get("canonical", {}).get(
        "executor_completed_count", 0.0
    )
    paired_deltas = {
        name: {
            "completed_minus_canonical": (
                aggregate[name].get("executor_completed_count", 0.0)
                - canonical_completed
            ),
            "model_objective_minus_canonical": (
                aggregate[name].get("milp_model_objective_sum", 0.0)
                - aggregate.get("canonical", {}).get(
                    "milp_model_objective_sum", 0.0
                )
            ),
        }
        for name in ranked_generator_names
    }
    method_deltas: dict[str, dict[str, float]] = {}
    for generator_name in ranked_generator_names:
        generator_completed = aggregate[generator_name].get(
            "executor_completed_count", 0.0
        )
        generator_objective = aggregate[generator_name].get(
            "milp_model_objective_sum", 0.0
        )
        for baseline_name in baseline_names:
            baseline_completed = baseline_aggregate[baseline_name].get(
                "executor_completed_count", 0.0
            )
            method_deltas[f"{generator_name}_minus_{baseline_name}"] = {
                "executor_completed_count": (
                    generator_completed - baseline_completed
                ),
                "generator_milp_model_objective_sum": generator_objective,
            }

    result = dict(first)
    result.update({
        "episode_seeds": list(episode_seeds),
        "topologies": topologies,
        "rows": rows,
        "baseline_rows": baseline_rows,
        "aggregate": aggregate,
        "baseline_aggregate": baseline_aggregate,
        "paired_deltas": paired_deltas,
        "method_deltas": method_deltas,
        "ranking": list(ranked_generators),
        "recommended_generator": ranked_generators[0],
    })
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return result


def run_generator_oracle_benchmark(
    *,
    config: WaxmanOrderConfig,
    episode_seeds: Iterable[int],
    generator_names: Iterable[str] = DEFAULT_GENERATORS,
    baseline_names: Iterable[str] = (),
    path_pool_per_pair: int = 8,
    schedules_per_path_pool: int | None = None,
    max_hops: int | None = None,
    planning_seeds: Iterable[int] = (0,),
    physics_seed_base: int = 0x434F4E5F50485953,
    planner_seed_base: int = 0x434F4E5F504C414E,
    include_full_pool_upper_bound: bool = False,
    online_selector: str = "reliable_memory_milp",
    reliability_confidence: float = 0.9,
    use_path_incumbent: bool = True,
    oracle_workers: int = 1,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Compare libraries; only the held-out online oracle sees slot requests."""

    episode_seeds = tuple(map(int, episode_seeds))
    generator_names = tuple(generator_names)
    baseline_names = tuple(baseline_names)
    planning_seeds = tuple(map(int, planning_seeds))
    if not episode_seeds or not generator_names or not planning_seeds:
        raise ValueError("episodes, generators, and planning seeds must be non-empty")
    if int(oracle_workers) != oracle_workers or oracle_workers < 1:
        raise ValueError("oracle_workers must be a positive integer")
    if not 0.0 < reliability_confidence <= 1.0:
        raise ValueError("reliability_confidence must lie in (0, 1]")
    if online_selector not in {
        "reliable_memory_milp",
        "static_milp",
        "static_relaxation",
        "exact_scenario_oracle",
    }:
        raise ValueError(
            "online_selector must be 'reliable_memory_milp', "
            "'static_relaxation', or 'exact_scenario_oracle' "
            "('static_milp' is a legacy alias)"
        )
    unknown = set(generator_names) - set(GENERATOR_PRESETS)
    if unknown:
        raise ValueError(f"unknown structural generators: {sorted(unknown)}")
    unknown_baselines = set(baseline_names) - set(BASELINE_PLANNERS)
    if unknown_baselines:
        raise ValueError(
            f"unknown baseline planners: {sorted(unknown_baselines)}"
        )

    names = generator_names + (
        ("full_pool",) if include_full_pool_upper_bound else ()
    )
    rows: dict[str, list[dict[str, object]]] = {name: [] for name in names}
    baseline_rows: dict[str, list[dict[str, object]]] = {
        name: [] for name in baseline_names
    }
    topology_rows = []
    for episode_seed in episode_seeds:
        episode = make_waxman_order_episode(config, episode_seed)
        pool_started = perf_counter()
        pool = build_waxman_topology_pool(
            episode,
            path_pool_per_pair=path_pool_per_pair,
            schedules_per_path_pool=schedules_per_path_pool,
            max_hops=max_hops,
        )
        pool_ms = 1000.0 * (perf_counter() - pool_started)
        context = build_waxman_selection_context(episode)
        physics_seed = _stable_seed(
            physics_seed_base, episode_seed, "hidden-physics"
        )
        planner_seed = _stable_seed(
            planner_seed_base, episode_seed, "online-oracle"
        )
        topology_rows.append({
            "episode_seed": episode_seed,
            "topology_fingerprint": pool.topology_fingerprint,
            "pair_count": len(pool.pair_entries),
            "raw_path_count": len(pool.paths),
            "raw_schedule_count": len(pool.templates),
            "pool_generation_ms": pool_ms,
        })

        # Q-DDCA and Q-CAST remain path-only baselines.  They operate on the
        # episode's own configured path catalogue and project every path onto
        # one deterministic reference schedule.  They share the exact
        # topology, request trace, hidden physics root, and executor with every
        # CON generator;
        # only their routing/selection rule and candidate construction differ.
        for baseline_name in baseline_names:
            row = run_planner_episode(
                episode,
                BASELINE_PLANNERS[baseline_name](),
                physics_seed_root=physics_seed,
                planner_seed=planner_seed,
            )
            row.update({
                "method": baseline_name,
                "candidate_source": (
                    "waxman-configured-paths-reference-schedule"
                ),
            })
            baseline_rows[baseline_name].append(row)

        for generator_name in generator_names:
            generation_started = perf_counter()
            selection = select_structural_library(
                pool,
                preset=generator_name,
                context=context,
            )
            compiled = compile_structural_topology_library(pool, selection)
            generation_ms = 1000.0 * (
                perf_counter() - generation_started
            )
            online_episode = instantiate_con_library_for_episode(
                episode, compiled.library
            )
            planner = _make_order_oracle(
                planning_seeds,
                online_selector=online_selector,
                reliability_confidence=reliability_confidence,
                use_path_incumbent=use_path_incumbent,
                oracle_workers=int(oracle_workers),
            )
            row = run_planner_episode(
                online_episode,
                planner,
                physics_seed_root=physics_seed,
                planner_seed=planner_seed,
            )
            row.update({
                "generator": generator_name,
                "offline_generation_ms": generation_ms,
                "library_valid_candidates": float(sum(
                    sum(entry.valid_mask)
                    for entry in compiled.library.pair_entries
                )),
                **{
                    f"library_{key}": value
                    for key, value in selection.diagnostic_values.items()
                },
            })
            rows[generator_name].append(row)

        if include_full_pool_upper_bound:
            full_episode = instantiate_topology_pool_for_episode(episode, pool)
            planner = _make_order_oracle(
                planning_seeds,
                online_selector=online_selector,
                reliability_confidence=reliability_confidence,
                use_path_incumbent=use_path_incumbent,
                oracle_workers=int(oracle_workers),
            )
            row = run_planner_episode(
                full_episode,
                planner,
                physics_seed_root=physics_seed,
                planner_seed=planner_seed,
            )
            row.update({
                "generator": "full_pool",
                "offline_generation_ms": pool_ms,
                "library_valid_candidates": float(len(pool.templates)),
            })
            rows["full_pool"].append(row)

    aggregate = {
        name: _aggregate(values) for name, values in rows.items()
    }
    baseline_aggregate = {
        name: _aggregate(values) for name, values in baseline_rows.items()
    }
    ranked_generators = tuple(
        name for name, _ in sorted(
            (
                (name, aggregate[name]) for name in generator_names
            ),
            key=_ranking_key,
        )
    )
    canonical_completed = aggregate.get("canonical", {}).get(
        "executor_completed_count", 0.0
    )
    paired_deltas = {
        name: {
            "completed_minus_canonical": (
                aggregate[name].get("executor_completed_count", 0.0)
                - canonical_completed
            ),
            "model_objective_minus_canonical": (
                aggregate[name].get("milp_model_objective_sum", 0.0)
                - aggregate.get("canonical", {}).get(
                    "milp_model_objective_sum", 0.0
                )
            ),
        }
        for name in generator_names
    }
    method_deltas: dict[str, dict[str, float]] = {}
    for generator_name in generator_names:
        generator_completed = aggregate[generator_name].get(
            "executor_completed_count", 0.0
        )
        generator_objective = aggregate[generator_name].get(
            "milp_model_objective_sum", 0.0
        )
        for baseline_name in baseline_names:
            baseline_completed = baseline_aggregate[baseline_name].get(
                "executor_completed_count", 0.0
            )
            method_deltas[f"{generator_name}_minus_{baseline_name}"] = {
                "executor_completed_count": (
                    generator_completed - baseline_completed
                ),
                # Path-only baselines do not solve the nominal MILP, so this
                # field records the generator oracle objective without
                # pretending there is a comparable baseline model objective.
                "generator_milp_model_objective_sum": generator_objective,
            }
    result: dict[str, object] = {
        "protocol": {
            "offline_generator_observes_requests": False,
            "online_selector": (
                "MilpReliableMemoryPathOrderPlanner"
                if online_selector == "reliable_memory_milp"
                else (
                    "MilpStaticPathOrderPlanner"
                    if online_selector in {
                        "static_milp", "static_relaxation"
                    }
                    else "MilpNominalPathOrderPlanner"
                )
            ),
            "online_selector_semantics": (
                "proven optimum of a deterministic time-indexed model with "
                "per-link reliable binomial EPR supply, current inventory, "
                "schedule-dependent memory release, link buffers, and BSM "
                "capacity; not a hidden-physics completion certificate"
                if online_selector == "reliable_memory_milp"
                else (
                    "proven optimum of a static necessary-condition MILP "
                    "relaxation; not a completion certificate; physical "
                    "executor outcomes recorded separately"
                    if online_selector in {
                        "static_milp", "static_relaxation"
                    }
                    else (
                        "exact finite-scenario executor-validated oracle; "
                        "intended only for small snapshots"
                    )
                )
            ),
            "reliability_confidence": float(reliability_confidence),
            "reliability_scope": "marginal per physical link",
            "generator_ranking_primary_metric": (
                "milp_model_objective_sum"
            ),
            "same_topology_trace_and_physics_per_generator": True,
            "same_topology_trace_and_physics_for_baselines": True,
            "baseline_candidate_source": (
                "episode-configured paths with one deterministic reference "
                "schedule"
            ),
            "path_pool_per_pair": path_pool_per_pair,
            "schedules_per_path_pool": schedules_per_path_pool,
            "max_hops": max_hops,
            "planning_seeds": list(planning_seeds),
            "exact_path_incumbent_enabled": bool(
                use_path_incumbent
                and online_selector == "exact_scenario_oracle"
            ),
            "oracle_workers": int(oracle_workers),
            "physics_seed_derivation": (
                "sha256(domain|base|episode_seed), first 32 bits"
            ),
            "planner_seed_derivation": (
                "sha256(domain|base|episode_seed), first 32 bits"
                if online_selector == "exact_scenario_oracle"
                else (
                    "unused by deterministic online MILP; retained for "
                    "cross-run manifest compatibility"
                )
            ),
        },
        "physics_seed_base": int(physics_seed_base),
        "planner_seed_base": int(planner_seed_base),
        "workload_config": {
            name: getattr(config, name)
            for name in config.__dataclass_fields__
        },
        "episode_seeds": list(episode_seeds),
        "topologies": topology_rows,
        "rows": rows,
        "baseline_rows": baseline_rows,
        "aggregate": aggregate,
        "baseline_aggregate": baseline_aggregate,
        "paired_deltas": paired_deltas,
        "method_deltas": method_deltas,
        "ranking": list(ranked_generators),
        "recommended_generator": ranked_generators[0],
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return result
