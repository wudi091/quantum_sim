"""Validate the value of per-request construction selection with exact MILP.

The construction-aware model and every fixed-construction baseline are
derived from one shared time-expanded candidate set.  The aware model sees
the union of all swap-tree variables.  Fixed baseline ``k`` sees only
``swap_tree_k`` variables, while retaining the same requests, paths, start
slots, objectives, capacities, fidelity gate, and MILP solver.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from qnet_core.construction_catalog import build_route_construction_catalogue
from qnet_core.resource_catalog import build_resource_capacities
from qnet_core.scenario import ScenarioConfig, make_episode
from qnet_core.spec import EpisodeSpec, PhysicalConfig

from .fidelity import candidate_fidelity_estimate_map
from .milp_oracle import (
    ConstructionAwareMILPOracle,
    DiscreteOracleSolution,
    has_numerically_zero_mip_gap,
)
from .time_expansion import TimeExpandedCandidate, expand_construction_candidates


@dataclass(frozen=True)
class MILPPolicyOutcome:
    policy: str
    variable_count: int
    completed_request_count: int
    total_completion_latency: float
    average_completion_latency: float
    solve_seconds: float
    stage_one_mip_gap: float | None
    stage_two_mip_gap: float | None
    selected_variable_ids: tuple[str, ...]
    selected_construction_kinds: tuple[str, ...]


@dataclass(frozen=True)
class ConstructionPolicyComparison:
    construction_aware: MILPPolicyOutcome
    fixed_outcomes: tuple[MILPPolicyOutcome, ...]
    best_fixed_policy: str
    completed_request_delta: int
    comparable_latency_delta: float | None

    @property
    def best_fixed(self) -> MILPPolicyOutcome:
        return next(
            outcome
            for outcome in self.fixed_outcomes
            if outcome.policy == self.best_fixed_policy
        )


@dataclass(frozen=True)
class ConstructionMILPTrial:
    seed: int
    node_count: int
    edge_count: int
    request_count: int
    candidate_count: int
    variable_count: int
    rejected_candidate_count: int
    comparison: ConstructionPolicyComparison


@dataclass(frozen=True)
class ConstructionMILPProblem:
    """One neutral time-expanded packing problem before policy solving."""

    episode: EpisodeSpec
    resource_capacities: Mapping[str, int]
    candidate_count: int
    variables: tuple[TimeExpandedCandidate, ...]
    rejected_candidate_count: int


@dataclass(frozen=True)
class ConstructionMILPInstance:
    """One solved instance with the neutral objects needed for physical replay."""

    problem: ConstructionMILPProblem
    comparison: ConstructionPolicyComparison

    @property
    def episode(self) -> EpisodeSpec:
        return self.problem.episode

    @property
    def resource_capacities(self) -> Mapping[str, int]:
        return self.problem.resource_capacities

    @property
    def candidate_count(self) -> int:
        return self.problem.candidate_count

    @property
    def variables(self) -> tuple[TimeExpandedCandidate, ...]:
        return self.problem.variables

    @property
    def rejected_candidate_count(self) -> int:
        return self.problem.rejected_candidate_count


def _outcome(
    policy: str,
    solution: DiscreteOracleSolution,
    variable_count: int,
    solve_seconds: float,
) -> MILPPolicyOutcome:
    completed = solution.completed_request_count
    selected = solution.selected_variables
    return MILPPolicyOutcome(
        policy=policy,
        variable_count=variable_count,
        completed_request_count=completed,
        total_completion_latency=solution.total_completion_latency,
        average_completion_latency=(
            solution.total_completion_latency / completed
            if completed else 0.0
        ),
        solve_seconds=solve_seconds,
        stage_one_mip_gap=solution.stage_one.mip_gap,
        stage_two_mip_gap=solution.stage_two.mip_gap,
        selected_variable_ids=tuple(
            variable.variable_id for variable in selected
        ),
        selected_construction_kinds=tuple(sorted({
            variable.construction_kind for variable in selected
        })),
    )


def _solve_policy(
    policy: str,
    variables: Sequence[TimeExpandedCandidate],
    capacities: Mapping[str, int],
    oracle: ConstructionAwareMILPOracle,
) -> MILPPolicyOutcome:
    started = perf_counter()
    solution = oracle.solve(variables, capacities)
    return _outcome(
        policy,
        solution,
        len(variables),
        perf_counter() - started,
    )


def compare_construction_policies(
    variables: Sequence[TimeExpandedCandidate],
    capacities: Mapping[str, int],
    *,
    fixed_policies: Sequence[str],
    oracle: ConstructionAwareMILPOracle | None = None,
    tolerance: float = 1e-7,
) -> ConstructionPolicyComparison:
    """Compare the full union against exact fixed-policy MILP subsets."""

    ordered = tuple(sorted(variables, key=lambda item: item.variable_id))
    policies = tuple(str(policy) for policy in fixed_policies)
    if not ordered:
        raise ValueError("construction comparison requires at least one variable")
    if any(
        abs(variable.expected_success_probability - 1.0) > tolerance
        for variable in ordered
    ):
        raise ValueError(
            "construction-count validation requires unit candidate weights"
        )
    if not policies or len(set(policies)) != len(policies):
        raise ValueError("fixed policies must be non-empty and unique")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    observed = {variable.construction_kind for variable in ordered}
    configured = set(policies)
    if observed != configured:
        missing = sorted(observed - configured)
        extra = sorted(configured - observed)
        raise ValueError(
            "fixed policies do not partition the aware candidate set: "
            f"missing={missing}, extra={extra}"
        )

    subsets = {
        policy: tuple(
            variable
            for variable in ordered
            if variable.construction_kind == policy
        )
        for policy in policies
    }
    if any(not subset for subset in subsets.values()):
        raise ValueError("every fixed policy must contain at least one variable")
    union_ids = {
        variable.variable_id
        for subset in subsets.values()
        for variable in subset
    }
    aware_ids = {variable.variable_id for variable in ordered}
    if union_ids != aware_ids or sum(map(len, subsets.values())) != len(ordered):
        raise ValueError("fixed-policy subsets are not an exact disjoint union")

    solver = oracle or ConstructionAwareMILPOracle()
    aware = _solve_policy(
        "construction_aware",
        ordered,
        capacities,
        solver,
    )
    fixed = tuple(
        _solve_policy(policy, subsets[policy], capacities, solver)
        for policy in policies
    )
    policy_order = {policy: index for index, policy in enumerate(policies)}
    best = max(
        fixed,
        key=lambda item: (
            item.completed_request_count,
            -item.total_completion_latency,
            -policy_order[item.policy],
        ),
    )
    if aware.completed_request_count < best.completed_request_count:
        raise AssertionError(
            "construction-aware union is worse than one of its fixed subsets"
        )
    equal_primary = (
        aware.completed_request_count == best.completed_request_count
    )
    if equal_primary and aware.total_completion_latency > (
        best.total_completion_latency + tolerance
    ):
        raise AssertionError(
            "construction-aware union loses the secondary objective"
        )
    return ConstructionPolicyComparison(
        construction_aware=aware,
        fixed_outcomes=fixed,
        best_fixed_policy=best.policy,
        completed_request_delta=(
            aware.completed_request_count - best.completed_request_count
        ),
        comparable_latency_delta=(
            best.total_completion_latency - aware.total_completion_latency
            if equal_primary else None
        ),
    )


def build_construction_problem(
    seed: int,
    scenario: ScenarioConfig,
    *,
    path_candidate_count: int,
    swap_tree_count: int,
) -> ConstructionMILPProblem:
    """Generate the shared neutral candidate set used by MILP experiments."""

    episode = make_episode(scenario, seed)
    capacities = build_resource_capacities(episode)
    candidates = build_route_construction_catalogue(
        episode.planning,
        candidate_count=path_candidate_count,
        construction_kinds=(),
        purification_kinds=("none",),
        swap_tree_count=swap_tree_count,
    )
    expansion = expand_construction_candidates(
        episode.planning,
        candidates,
        capacities,
        fidelity_estimates=candidate_fidelity_estimate_map(
            episode, candidates
        ),
        # This experiment isolates resource--time scheduling.  Unit weights
        # make stage one maximize the number of admitted nominal plans; the
        # SeQUeNCe success model is validated separately after this gate.
        success_probability_estimates={
            candidate.candidate_id: 1.0 for candidate in candidates
        },
    )
    return ConstructionMILPProblem(
        episode=episode,
        resource_capacities=capacities,
        candidate_count=len(candidates),
        variables=expansion.variables,
        rejected_candidate_count=len(expansion.rejections),
    )


def build_trial_instance(
    seed: int,
    scenario: ScenarioConfig,
    *,
    path_candidate_count: int,
    swap_tree_count: int,
    time_limit_seconds: float,
) -> ConstructionMILPInstance:
    """Generate and solve one instance while preserving replay inputs."""

    problem = build_construction_problem(
        seed,
        scenario,
        path_candidate_count=path_candidate_count,
        swap_tree_count=swap_tree_count,
    )
    fixed_policies = tuple(
        f"swap_tree_{index}" for index in range(swap_tree_count)
    )
    comparison = compare_construction_policies(
        problem.variables,
        problem.resource_capacities,
        fixed_policies=fixed_policies,
        oracle=ConstructionAwareMILPOracle(
            time_limit_seconds=time_limit_seconds,
            mip_relative_gap=0.0,
        ),
    )
    return ConstructionMILPInstance(
        problem=problem,
        comparison=comparison,
    )


def run_trial(
    seed: int,
    scenario: ScenarioConfig,
    *,
    path_candidate_count: int,
    swap_tree_count: int,
    time_limit_seconds: float,
) -> ConstructionMILPTrial:
    """Generate one paired nominal-planning instance and solve all variants."""

    instance = build_trial_instance(
        seed,
        scenario,
        path_candidate_count=path_candidate_count,
        swap_tree_count=swap_tree_count,
        time_limit_seconds=time_limit_seconds,
    )
    episode = instance.episode
    return ConstructionMILPTrial(
        seed=seed,
        node_count=len(episode.nodes),
        edge_count=len(episode.edges),
        request_count=len(episode.requests),
        candidate_count=instance.candidate_count,
        variable_count=len(instance.variables),
        rejected_candidate_count=instance.rejected_candidate_count,
        comparison=instance.comparison,
    )


def _bootstrap_interval(
    values: Sequence[float],
    *,
    samples: int,
    random_seed: int,
) -> tuple[float, float]:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = np.mean(array[indices], axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _paired_randomization_p_value(
    values: Sequence[float],
    *,
    samples: int,
    random_seed: int,
) -> float:
    if samples < 1:
        raise ValueError("randomization samples must be positive")
    array = np.asarray(values, dtype=float)
    observed = abs(float(np.mean(array)))
    rng = np.random.default_rng(random_seed)
    extreme = 0
    for _ in range(samples):
        signs = rng.choice((-1.0, 1.0), size=len(array))
        extreme += int(abs(float(np.mean(array * signs))) >= observed - 1e-12)
    return float((extreme + 1) / (samples + 1))


def aggregate_trials(
    trials: Sequence[ConstructionMILPTrial],
    *,
    bootstrap_samples: int,
    randomization_samples: int,
    statistics_seed: int,
) -> dict[str, object]:
    if not trials:
        raise ValueError("at least one trial is required")
    aware_counts = np.asarray([
        trial.comparison.construction_aware.completed_request_count
        for trial in trials
    ], dtype=float)
    fixed_counts = np.asarray([
        trial.comparison.best_fixed.completed_request_count
        for trial in trials
    ], dtype=float)
    count_deltas = aware_counts - fixed_counts
    mixed = [
        len(
            trial.comparison.construction_aware.selected_construction_kinds
        ) > 1
        for trial in trials
    ]
    exact_milp = all(
        has_numerically_zero_mip_gap(outcome.stage_one_mip_gap)
        and has_numerically_zero_mip_gap(outcome.stage_two_mip_gap)
        for trial in trials
        for outcome in (
            trial.comparison.construction_aware,
            *trial.comparison.fixed_outcomes,
        )
    )
    mean_fixed = float(np.mean(fixed_counts))
    confidence_interval = _bootstrap_interval(
        count_deltas,
        samples=bootstrap_samples,
        random_seed=statistics_seed,
    )
    p_value = _paired_randomization_p_value(
        count_deltas,
        samples=randomization_samples,
        random_seed=statistics_seed + 1,
    )
    advantage_validated = bool(
        exact_milp
        and confidence_interval[0] > 0.0
        and p_value < 0.05
        and np.all(count_deltas >= 0.0)
        and all(mixed)
    )
    return {
        "trial_count": len(trials),
        "all_milp_solutions_exact": exact_milp,
        "advantage_validated": advantage_validated,
        "construction_aware_mean_completed_requests": float(
            np.mean(aware_counts)
        ),
        "best_fixed_mean_completed_requests": mean_fixed,
        "mean_completed_request_delta": float(np.mean(count_deltas)),
        "median_completed_request_delta": float(np.median(count_deltas)),
        "relative_completed_request_gain": (
            float(np.mean(count_deltas) / mean_fixed)
            if mean_fixed else None
        ),
        "completed_request_delta_bootstrap_95_ci": list(
            confidence_interval
        ),
        "completed_request_delta_randomization_p_value": p_value,
        "strict_win_count": int(np.sum(count_deltas > 0)),
        "tie_count": int(np.sum(count_deltas == 0)),
        "loss_count": int(np.sum(count_deltas < 0)),
        "mixed_construction_solution_count": int(sum(mixed)),
        "mixed_construction_solution_rate": float(np.mean(mixed)),
        "mean_construction_aware_solve_seconds": float(np.mean([
            trial.comparison.construction_aware.solve_seconds
            for trial in trials
        ])),
        "mean_best_fixed_solve_seconds": float(np.mean([
            trial.comparison.best_fixed.solve_seconds
            for trial in trials
        ])),
        "mean_all_fixed_oracle_solve_seconds": float(np.mean([
            sum(
                outcome.solve_seconds
                for outcome in trial.comparison.fixed_outcomes
            )
            for trial in trials
        ])),
    }


def _trial_row(trial: ConstructionMILPTrial) -> dict[str, object]:
    comparison = trial.comparison
    row: dict[str, object] = {
        "seed": trial.seed,
        "node_count": trial.node_count,
        "edge_count": trial.edge_count,
        "request_count": trial.request_count,
        "candidate_count": trial.candidate_count,
        "variable_count": trial.variable_count,
        "rejected_candidate_count": trial.rejected_candidate_count,
        "aware_completed_requests": (
            comparison.construction_aware.completed_request_count
        ),
        "aware_solve_seconds": comparison.construction_aware.solve_seconds,
        "aware_construction_kinds": ";".join(
            comparison.construction_aware.selected_construction_kinds
        ),
        "best_fixed_policy": comparison.best_fixed_policy,
        "best_fixed_completed_requests": (
            comparison.best_fixed.completed_request_count
        ),
        "completed_request_delta": comparison.completed_request_delta,
        "comparable_latency_delta": comparison.comparable_latency_delta,
    }
    for outcome in comparison.fixed_outcomes:
        row[f"{outcome.policy}_completed_requests"] = (
            outcome.completed_request_count
        )
        row[f"{outcome.policy}_solve_seconds"] = outcome.solve_seconds
    return row


def _markdown_summary(payload: Mapping[str, object]) -> str:
    aggregate = payload["aggregate"]
    validation_config = payload["validation_config"]
    assert isinstance(aggregate, Mapping)
    assert isinstance(validation_config, Mapping)
    low, high = aggregate["completed_request_delta_bootstrap_95_ci"]
    swap_tree_count = int(validation_config["swap_tree_count"])
    relative = aggregate["relative_completed_request_gain"]
    relative_text = (
        "n/a" if relative is None else f"{100.0 * float(relative):.2f}%"
    )
    return "\n".join((
        "# 构造感知 MILP 优势验证",
        "",
        f"- 配对实例数：{aggregate['trial_count']}",
        f"- 对照：每个实例上 {swap_tree_count} 种固定交换树 MILP 中的最优者",
        f"- 构造感知平均名义可接纳数：{float(aggregate['construction_aware_mean_completed_requests']):.4f}",
        f"- 最优固定构造平均名义可接纳数：{float(aggregate['best_fixed_mean_completed_requests']):.4f}",
        f"- 平均名义可接纳数提升：{float(aggregate['mean_completed_request_delta']):.4f}（{relative_text}）",
        f"- 配对 bootstrap 95% CI：[{float(low):.4f}, {float(high):.4f}]",
        f"- 配对随机化检验 p 值：{float(aggregate['completed_request_delta_randomization_p_value']):.6g}",
        f"- 胜/平/负：{aggregate['strict_win_count']}/{aggregate['tie_count']}/{aggregate['loss_count']}",
        f"- 使用多种交换树的实例比例：{100.0 * float(aggregate['mixed_construction_solution_rate']):.2f}%",
        f"- 所有 MILP 均为求解器数值精度内的精确解（gap≤1e-7）：{'是' if aggregate['all_milp_solutions_exact'] else '否'}",
        f"- 构造感知名义规划优势验证通过：{'是' if aggregate['advantage_validated'] else '否'}",
        "",
        "判定原则：只有全部 MILP 为求解器数值精度内的精确解（gap≤1e-7）、置信区间下界大于 0、配对随机化检验 p<0.05、无负例且每个构造感知解确实混合使用不同交换树，才认为构造感知名义规划优势成立。",
        "本实验不执行 SeQUeNCe，因此名义可接纳数不等价于物理完成请求数；物理优势需要单独验证。",
        "",
    ))


def save_report(
    payload: Mapping[str, object],
    output_directory: str | Path,
) -> dict[str, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"construction_milp_validation_{timestamp}"
    json_path = output / f"{stem}.json"
    csv_path = output / f"{stem}.csv"
    markdown_path = output / f"{stem}.md"
    serializable_payload = {
        key: value
        for key, value in payload.items()
        if key != "trial_objects"
    }
    json_path.write_text(
        json.dumps(serializable_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rows = [_trial_row(trial) for trial in payload["trial_objects"]]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(
        _markdown_summary(payload),
        encoding="utf-8",
    )
    latest = {
        "json": output / "construction_milp_validation.json",
        "csv": output / "construction_milp_validation.csv",
        "markdown": output / "construction_milp_validation.md",
    }
    for source, target in zip(
        (json_path, csv_path, markdown_path), latest.values()
    ):
        shutil.copyfile(source, target)
    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": markdown_path,
        **{f"latest_{key}": value for key, value in latest.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare construction-aware MILP with the per-instance best "
            "fixed swap-tree MILP."
        )
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=3101)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--min-hops", type=int, default=4)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--paths", type=int, default=4)
    parser.add_argument("--construction-plans", type=int, default=5)
    parser.add_argument("--time-limit-seconds", type=float, default=30.0)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--randomization-samples", type=int, default=20_000)
    parser.add_argument("--statistics-seed", type=int, default=20260812)
    parser.add_argument("--waxman-alpha", type=float, default=0.15)
    parser.add_argument("--waxman-beta", type=float, default=0.45)
    parser.add_argument("--topology-attempts", type=int, default=128)
    parser.add_argument("--generation-probability", type=float, default=0.8)
    parser.add_argument("--swap-probability", type=float, default=0.9)
    parser.add_argument("--memory-capacity", type=int, default=2)
    parser.add_argument("--memory-lifetime", type=int, default=300)
    parser.add_argument("--max-width", type=int, default=1)
    parser.add_argument("--quantum-distance-m", type=float, default=1000.0)
    parser.add_argument("--slot-duration-ps", type=int, default=50_000_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seeds < 1:
        raise ValueError("seeds must be positive")
    scenario = ScenarioConfig(
        request_count=args.requests,
        min_hops=args.min_hops,
        max_hops=args.max_hops,
        ttl=args.horizon,
        horizon=args.horizon,
        topology_nodes=args.nodes,
        waxman_alpha=args.waxman_alpha,
        waxman_beta=args.waxman_beta,
        topology_attempts=args.topology_attempts,
        waxman_add_mst=False,
        endpoint_mode="distance_stratified",
        physical=PhysicalConfig(
            generation_probability=args.generation_probability,
            swap_probability=args.swap_probability,
            memory_capacity=args.memory_capacity,
            memory_lifetime=args.memory_lifetime,
            max_width=args.max_width,
            quantum_distance_m=args.quantum_distance_m,
            slot_duration_ps=args.slot_duration_ps,
        ),
    )
    trials = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        trial = run_trial(
            seed,
            scenario,
            path_candidate_count=args.paths,
            swap_tree_count=args.construction_plans,
            time_limit_seconds=args.time_limit_seconds,
        )
        trials.append(trial)
        comparison = trial.comparison
        print(
            f"seed={seed} aware="
            f"{comparison.construction_aware.completed_request_count} "
            f"best_fixed={comparison.best_fixed.completed_request_count} "
            f"delta={comparison.completed_request_delta} "
            f"fixed={comparison.best_fixed_policy}",
            flush=True,
        )
    aggregate = aggregate_trials(
        trials,
        bootstrap_samples=args.bootstrap_samples,
        randomization_samples=args.randomization_samples,
        statistics_seed=args.statistics_seed,
    )
    serializable_trials = [asdict(trial) for trial in trials]
    payload: dict[str, object] = {
        "schema_version": 1,
        "comparison_contract": {
            "paired_instances": True,
            "same_requests_paths_slots_objective_capacities": True,
            "aware_variables_are_fixed_subset_union": True,
            "best_fixed_is_per_instance_oracle": True,
            "exact_mip_relative_gap": 0.0,
            "primary_metric": "completed_request_count",
        },
        "scenario": asdict(scenario),
        "validation_config": {
            "seed_start": args.seed_start,
            "seeds": args.seeds,
            "path_candidate_count": args.paths,
            "swap_tree_count": args.construction_plans,
            "objective": (
                "maximize nominal admitted request count, then minimize "
                "nominal completion latency"
            ),
            "time_limit_seconds": args.time_limit_seconds,
            "bootstrap_samples": args.bootstrap_samples,
            "randomization_samples": args.randomization_samples,
            "statistics_seed": args.statistics_seed,
        },
        "aggregate": aggregate,
        "trials": serializable_trials,
        # Kept only in memory for CSV serialization; removed before JSON save.
        "trial_objects": trials,
    }
    paths = save_report(payload, args.output)
    print(
        "mean delta="
        f"{aggregate['mean_completed_request_delta']:.4f} "
        f"wins/ties/losses={aggregate['strict_win_count']}/"
        f"{aggregate['tie_count']}/{aggregate['loss_count']}"
    )
    print(f"json: {paths['json']}")
    print(f"csv: {paths['csv']}")
    print(f"markdown: {paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
