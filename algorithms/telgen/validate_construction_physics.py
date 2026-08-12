"""Replay construction-aware and fixed-construction MILP plans in SeQUeNCe."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Mapping, Sequence

import numpy as np

from qnet_core.scenario import ScenarioConfig
from qnet_core.spec import PhysicalConfig

from .physical_validation import evaluate_selected_physics
from .time_expansion import TimeExpandedCandidate
from .validate_construction_milp import (
    ConstructionMILPInstance,
    MILPPolicyOutcome,
    _bootstrap_interval,
    _paired_randomization_p_value,
    build_trial_instance,
)


_NON_FATAL_SCHEDULE_CODES = frozenset({"slot_completion_overrun"})


@dataclass(frozen=True)
class PhysicalPolicyOutcome:
    policy: str
    planned_selected_requests: int
    completed_requests: int
    completion_retention: float | None
    mean_censored_latency_slots: float
    p95_completion_latency_slots: float
    schedule_violation_count: int
    unsafe_schedule_violation_count: int
    schedule_violation_codes: tuple[str, ...]
    physical_failure_count: int
    fidelity_violation_count: int
    physical_backend_rejection_count: int
    post_completion_validation_failure_count: int
    peak_physical_memory_usage: int


@dataclass(frozen=True)
class ConstructionPhysicalTrial:
    planning_seed: int
    physical_seed: int
    node_count: int
    edge_count: int
    request_count: int
    candidate_count: int
    variable_count: int
    best_fixed_policy: str
    construction_aware: PhysicalPolicyOutcome
    best_fixed: PhysicalPolicyOutcome
    completed_request_delta: int
    censored_latency_delta_slots: float


def _selected_variables(
    instance: ConstructionMILPInstance,
    outcome: MILPPolicyOutcome,
) -> tuple[TimeExpandedCandidate, ...]:
    by_id = {variable.variable_id: variable for variable in instance.variables}
    if len(by_id) != len(instance.variables):
        raise ValueError("time-expanded variable IDs must be unique")
    try:
        selected = tuple(
            by_id[variable_id] for variable_id in outcome.selected_variable_ids
        )
    except KeyError as exc:
        raise ValueError(
            f"MILP outcome references an unknown variable: {exc.args[0]}"
        ) from exc
    if len(selected) != outcome.completed_request_count:
        raise ValueError("MILP selected-variable count does not match its objective")
    if len({variable.request_id for variable in selected}) != len(selected):
        raise ValueError("MILP outcome selects one request more than once")
    return selected


def _physical_outcome(
    policy: str,
    evaluation,
    *,
    slot_duration_ps: int,
) -> PhysicalPolicyOutcome:
    metrics = evaluation.metrics
    violation_codes = tuple(sorted(
        violation.code for violation in evaluation.violations
    ))
    unsafe = sum(
        code not in _NON_FATAL_SCHEDULE_CODES for code in violation_codes
    )
    planned = int(metrics["planned_selected_requests"])
    completed = int(metrics["completed_requests"])
    return PhysicalPolicyOutcome(
        policy=policy,
        planned_selected_requests=planned,
        completed_requests=completed,
        completion_retention=(
            completed / planned if planned else None
        ),
        mean_censored_latency_slots=(
            float(metrics["mean_censored_latency_ps"]) / slot_duration_ps
        ),
        p95_completion_latency_slots=(
            float(metrics["p95_completion_latency_ps"]) / slot_duration_ps
        ),
        schedule_violation_count=len(violation_codes),
        unsafe_schedule_violation_count=int(unsafe),
        schedule_violation_codes=violation_codes,
        physical_failure_count=int(metrics["physical_failure_count"]),
        fidelity_violation_count=int(metrics["fidelity_violation_count"]),
        physical_backend_rejection_count=int(
            metrics["physical_backend_rejection_count"]
        ),
        post_completion_validation_failure_count=int(
            metrics["post_completion_validation_failure_count"]
        ),
        peak_physical_memory_usage=int(metrics["peak_physical_memory_usage"]),
    )


def replay_trial(
    planning_seed: int,
    physical_seed: int,
    scenario: ScenarioConfig,
    *,
    path_candidate_count: int,
    swap_tree_count: int,
    time_limit_seconds: float,
) -> ConstructionPhysicalTrial:
    """Solve one paired instance, then replay both fixed schedules physically."""

    instance = build_trial_instance(
        planning_seed,
        scenario,
        path_candidate_count=path_candidate_count,
        swap_tree_count=swap_tree_count,
        time_limit_seconds=time_limit_seconds,
    )
    comparison = instance.comparison
    aware_variables = _selected_variables(
        instance, comparison.construction_aware
    )
    fixed_variables = _selected_variables(instance, comparison.best_fixed)
    aware_evaluation = evaluate_selected_physics(
        instance.episode,
        aware_variables,
        instance.resource_capacities,
        physical_seed=physical_seed,
    )
    fixed_evaluation = evaluate_selected_physics(
        instance.episode,
        fixed_variables,
        instance.resource_capacities,
        physical_seed=physical_seed,
    )
    slot_duration_ps = instance.episode.physical.slot_duration_ps
    aware = _physical_outcome(
        "construction_aware",
        aware_evaluation,
        slot_duration_ps=slot_duration_ps,
    )
    fixed = _physical_outcome(
        comparison.best_fixed_policy,
        fixed_evaluation,
        slot_duration_ps=slot_duration_ps,
    )
    return ConstructionPhysicalTrial(
        planning_seed=planning_seed,
        physical_seed=physical_seed,
        node_count=len(instance.episode.nodes),
        edge_count=len(instance.episode.edges),
        request_count=len(instance.episode.requests),
        candidate_count=instance.candidate_count,
        variable_count=len(instance.variables),
        best_fixed_policy=comparison.best_fixed_policy,
        construction_aware=aware,
        best_fixed=fixed,
        completed_request_delta=(
            aware.completed_requests - fixed.completed_requests
        ),
        censored_latency_delta_slots=(
            fixed.mean_censored_latency_slots
            - aware.mean_censored_latency_slots
        ),
    )


def aggregate_physical_trials(
    trials: Sequence[ConstructionPhysicalTrial],
    *,
    bootstrap_samples: int,
    randomization_samples: int,
    statistics_seed: int,
) -> dict[str, object]:
    if not trials:
        raise ValueError("at least one physical trial is required")
    aware_completed = np.asarray([
        trial.construction_aware.completed_requests for trial in trials
    ], dtype=float)
    fixed_completed = np.asarray([
        trial.best_fixed.completed_requests for trial in trials
    ], dtype=float)
    deltas = aware_completed - fixed_completed
    aware_planned = np.asarray([
        trial.construction_aware.planned_selected_requests for trial in trials
    ], dtype=float)
    fixed_planned = np.asarray([
        trial.best_fixed.planned_selected_requests for trial in trials
    ], dtype=float)
    aware_latency = np.asarray([
        trial.construction_aware.mean_censored_latency_slots for trial in trials
    ], dtype=float)
    fixed_latency = np.asarray([
        trial.best_fixed.mean_censored_latency_slots for trial in trials
    ], dtype=float)
    confidence_interval = _bootstrap_interval(
        deltas,
        samples=bootstrap_samples,
        random_seed=statistics_seed,
    )
    p_value = _paired_randomization_p_value(
        deltas,
        samples=randomization_samples,
        random_seed=statistics_seed + 1,
    )
    unsafe_count = sum(
        trial.construction_aware.unsafe_schedule_violation_count
        + trial.best_fixed.unsafe_schedule_violation_count
        for trial in trials
    )
    backend_rejections = sum(
        trial.construction_aware.physical_backend_rejection_count
        + trial.best_fixed.physical_backend_rejection_count
        for trial in trials
    )
    post_completion_failures = sum(
        trial.construction_aware.post_completion_validation_failure_count
        + trial.best_fixed.post_completion_validation_failure_count
        for trial in trials
    )
    hard_gates_valid = (
        unsafe_count == 0
        and backend_rejections == 0
        and post_completion_failures == 0
    )
    mean_fixed_completed = float(np.mean(fixed_completed))
    return {
        "trial_count": len(trials),
        "hard_gates_valid": hard_gates_valid,
        "physical_advantage_validated": bool(
            hard_gates_valid
            and confidence_interval[0] > 0.0
            and p_value < 0.05
        ),
        "construction_aware_mean_planned_requests": float(
            np.mean(aware_planned)
        ),
        "best_fixed_mean_planned_requests": float(np.mean(fixed_planned)),
        "construction_aware_mean_completed_requests": float(
            np.mean(aware_completed)
        ),
        "best_fixed_mean_completed_requests": mean_fixed_completed,
        "mean_completed_request_delta": float(np.mean(deltas)),
        "median_completed_request_delta": float(np.median(deltas)),
        "relative_completed_request_gain": (
            float(np.mean(deltas) / mean_fixed_completed)
            if mean_fixed_completed else None
        ),
        "completed_request_delta_bootstrap_95_ci": list(confidence_interval),
        "completed_request_delta_randomization_p_value": p_value,
        "strict_win_count": int(np.sum(deltas > 0)),
        "tie_count": int(np.sum(deltas == 0)),
        "loss_count": int(np.sum(deltas < 0)),
        "construction_aware_completion_retention": float(
            np.sum(aware_completed) / np.sum(aware_planned)
        ),
        "best_fixed_completion_retention": float(
            np.sum(fixed_completed) / np.sum(fixed_planned)
        ),
        "construction_aware_mean_censored_latency_slots": float(
            np.mean(aware_latency)
        ),
        "best_fixed_mean_censored_latency_slots": float(
            np.mean(fixed_latency)
        ),
        "mean_censored_latency_improvement_slots": float(
            np.mean(fixed_latency - aware_latency)
        ),
        "unsafe_schedule_violation_count": int(unsafe_count),
        "physical_backend_rejection_count": int(backend_rejections),
        "post_completion_validation_failure_count": int(
            post_completion_failures
        ),
        "construction_aware_slot_completion_overrun_count": int(sum(
            trial.construction_aware.schedule_violation_codes.count(
                "slot_completion_overrun"
            )
            for trial in trials
        )),
        "best_fixed_slot_completion_overrun_count": int(sum(
            trial.best_fixed.schedule_violation_codes.count(
                "slot_completion_overrun"
            )
            for trial in trials
        )),
    }


def _trial_row(trial: ConstructionPhysicalTrial) -> dict[str, object]:
    return {
        "planning_seed": trial.planning_seed,
        "physical_seed": trial.physical_seed,
        "node_count": trial.node_count,
        "edge_count": trial.edge_count,
        "request_count": trial.request_count,
        "candidate_count": trial.candidate_count,
        "variable_count": trial.variable_count,
        "best_fixed_policy": trial.best_fixed_policy,
        "aware_planned_requests": (
            trial.construction_aware.planned_selected_requests
        ),
        "fixed_planned_requests": trial.best_fixed.planned_selected_requests,
        "aware_completed_requests": trial.construction_aware.completed_requests,
        "fixed_completed_requests": trial.best_fixed.completed_requests,
        "completed_request_delta": trial.completed_request_delta,
        "aware_completion_retention": (
            trial.construction_aware.completion_retention
        ),
        "fixed_completion_retention": trial.best_fixed.completion_retention,
        "aware_mean_censored_latency_slots": (
            trial.construction_aware.mean_censored_latency_slots
        ),
        "fixed_mean_censored_latency_slots": (
            trial.best_fixed.mean_censored_latency_slots
        ),
        "censored_latency_delta_slots": trial.censored_latency_delta_slots,
        "aware_schedule_violation_codes": ";".join(
            trial.construction_aware.schedule_violation_codes
        ),
        "fixed_schedule_violation_codes": ";".join(
            trial.best_fixed.schedule_violation_codes
        ),
        "aware_physical_failures": (
            trial.construction_aware.physical_failure_count
        ),
        "fixed_physical_failures": trial.best_fixed.physical_failure_count,
    }


def _markdown_summary(payload: Mapping[str, object]) -> str:
    aggregate = payload["aggregate"]
    config = payload["validation_config"]
    assert isinstance(aggregate, Mapping)
    assert isinstance(config, Mapping)
    low, high = aggregate["completed_request_delta_bootstrap_95_ci"]
    relative = aggregate["relative_completed_request_gain"]
    relative_text = (
        "n/a" if relative is None else f"{100.0 * float(relative):.2f}%"
    )
    return "\n".join((
        "# 构造感知 MILP 的 SeQUeNCe 物理回放验证",
        "",
        f"- 配对实例数：{aggregate['trial_count']}",
        f"- 每实例请求数：{config['request_count']}",
        f"- 构造感知平均计划接纳数：{float(aggregate['construction_aware_mean_planned_requests']):.4f}",
        f"- 最佳固定构造平均计划接纳数：{float(aggregate['best_fixed_mean_planned_requests']):.4f}",
        f"- 构造感知平均物理完成数：{float(aggregate['construction_aware_mean_completed_requests']):.4f}",
        f"- 最佳固定构造平均物理完成数：{float(aggregate['best_fixed_mean_completed_requests']):.4f}",
        f"- 平均物理完成数提升：{float(aggregate['mean_completed_request_delta']):.4f}（{relative_text}）",
        f"- 配对 bootstrap 95% CI：[{float(low):.4f}, {float(high):.4f}]",
        f"- 配对随机化检验 p 值：{float(aggregate['completed_request_delta_randomization_p_value']):.6g}",
        f"- 胜/平/负：{aggregate['strict_win_count']}/{aggregate['tie_count']}/{aggregate['loss_count']}",
        f"- 构造感知计划保留率：{100.0 * float(aggregate['construction_aware_completion_retention']):.2f}%",
        f"- 最佳固定构造计划保留率：{100.0 * float(aggregate['best_fixed_completion_retention']):.2f}%",
        f"- 构造感知平均删失延迟：{float(aggregate['construction_aware_mean_censored_latency_slots']):.4f} 时隙",
        f"- 最佳固定构造平均删失延迟：{float(aggregate['best_fixed_mean_censored_latency_slots']):.4f} 时隙",
        f"- 不安全调度违例：{aggregate['unsafe_schedule_violation_count']}",
        f"- 物理后端拒绝：{aggregate['physical_backend_rejection_count']}",
        f"- 完成后验证失败：{aggregate['post_completion_validation_failure_count']}",
        f"- 物理优势验证通过：{'是' if aggregate['physical_advantage_validated'] else '否'}",
        "",
        "说明：两种计划共享同一拓扑、请求、物理参数和物理 seed 标签，并在独立的 SeQUeNCe 执行器中运行。由于计划不同会消耗不同数量和顺序的随机数，这属于配对初始随机种子，而不是逐事件共同随机数。",
        "`slot_completion_overrun` 只表示物理协议跨过粗粒度名义时隙，单独报告但不作为资源不可行；其他调度违例、物理后端拒绝或完成后验证失败属于硬失败。",
        "",
    ))


def save_report(
    payload: Mapping[str, object],
    output_directory: str | Path,
) -> dict[str, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"construction_physical_validation_{timestamp}"
    json_path = output / f"{stem}.json"
    csv_path = output / f"{stem}.csv"
    markdown_path = output / f"{stem}.md"
    serializable = dict(payload)
    trials = serializable.pop("trial_objects")
    json_path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        rows = [_trial_row(trial) for trial in trials]
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(_markdown_summary(payload), encoding="utf-8")
    latest = {
        "json": output / "construction_physical_validation.json",
        "csv": output / "construction_physical_validation.csv",
        "markdown": output / "construction_physical_validation.md",
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
            "Replay construction-aware and best-fixed MILP schedules in "
            "the same SeQUeNCe environment."
        )
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=3101)
    parser.add_argument("--physical-seed-start", type=int, default=53101)
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
    parser.add_argument("--statistics-seed", type=int, default=20260813)
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
    if args.physical_seed_start < 0:
        raise ValueError("physical-seed-start must be non-negative")
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
    for index in range(args.seeds):
        planning_seed = args.seed_start + index
        physical_seed = args.physical_seed_start + index
        trial = replay_trial(
            planning_seed,
            physical_seed,
            scenario,
            path_candidate_count=args.paths,
            swap_tree_count=args.construction_plans,
            time_limit_seconds=args.time_limit_seconds,
        )
        trials.append(trial)
        print(
            f"planning_seed={planning_seed} physical_seed={physical_seed} "
            f"aware={trial.construction_aware.completed_requests}/"
            f"{trial.construction_aware.planned_selected_requests} "
            f"fixed={trial.best_fixed.completed_requests}/"
            f"{trial.best_fixed.planned_selected_requests} "
            f"delta={trial.completed_request_delta} "
            f"fixed_policy={trial.best_fixed_policy}",
            flush=True,
        )
    aggregate = aggregate_physical_trials(
        trials,
        bootstrap_samples=args.bootstrap_samples,
        randomization_samples=args.randomization_samples,
        statistics_seed=args.statistics_seed,
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "comparison_contract": {
            "paired_planning_instances": True,
            "same_topology_requests_and_physical_parameters": True,
            "same_physical_seed_label": True,
            "strict_eventwise_common_random_numbers": False,
            "independent_sequence_executors": True,
            "plans_are_fixed_before_physical_execution": True,
            "non_fatal_schedule_codes": sorted(_NON_FATAL_SCHEDULE_CODES),
            "primary_metric": "completed_requests",
        },
        "scenario": asdict(scenario),
        "validation_config": {
            "seed_start": args.seed_start,
            "physical_seed_start": args.physical_seed_start,
            "seeds": args.seeds,
            "request_count": args.requests,
            "path_candidate_count": args.paths,
            "swap_tree_count": args.construction_plans,
            "time_limit_seconds": args.time_limit_seconds,
            "bootstrap_samples": args.bootstrap_samples,
            "randomization_samples": args.randomization_samples,
            "statistics_seed": args.statistics_seed,
        },
        "aggregate": aggregate,
        "trials": [asdict(trial) for trial in trials],
        "trial_objects": trials,
    }
    paths = save_report(payload, args.output)
    print(
        f"mean physical delta={aggregate['mean_completed_request_delta']:.4f} "
        f"wins/ties/losses={aggregate['strict_win_count']}/"
        f"{aggregate['tie_count']}/{aggregate['loss_count']}"
    )
    print(f"json: {paths['json']}")
    print(f"csv: {paths['csv']}")
    print(f"markdown: {paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
