"""Resumable resource sweep for the persistent-inventory Waxman workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from .order_waxman import WaxmanOrderConfig
from .order_waxman_benchmark import run_suite


PLANNERS = (
    "qddca_fixed",
    "qcast_fixed",
    "saa_path",
    "saa_path_order",
)


def _cell_id(
    memory: int,
    arrival_rate: float,
    candidate_request_cap: int,
) -> str:
    rate = f"{arrival_rate:g}".replace(".", "p")
    return f"m{memory}_lambda{rate}_cap{candidate_request_cap}"


def _paired_order_gap(cell: dict[str, object]) -> dict[str, object]:
    rows = cell["result"]["rows"]
    order_key = (
        "saa_path_order" if "saa_path_order" in rows
        else "optimal_path_order"
    )
    path_key = "saa_path" if "saa_path" in rows else "optimal_path"
    order = rows[order_key]
    path = rows[path_key]
    differences = [
        float(left["completion_rate"]) - float(right["completion_rate"])
        for left, right in zip(order, path)
    ]
    return {
        "per_seed": differences,
        "mean": sum(differences) / len(differences),
        "min": min(differences),
        "max": max(differences),
        "wins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "losses": sum(value < 0 for value in differences),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    cells = list(payload["cells"].values())
    cells.sort(key=lambda cell: (
        int(cell["parameters"]["node_memory"]),
        float(cell["parameters"]["arrival_rate"]),
        int(cell["parameters"]["candidate_request_cap"]),
    ))
    lines = [
        "# Persistent-EPR Waxman resource sweep",
        "",
        (
            "筛选设置：100 节点、每拓扑 100 个泊松到达请求、"
            f"{payload['seeds']} 个拓扑 seed、每个 SAA planner "
            f"{payload['oracle_rollouts']} 个隐藏物理 rollout。"
        ),
        (
            "为保证 candidate_request_cap=6 时仍能穷举，本轮所有 cell "
            "固定为每请求 "
            "1 条候选路径、每路径最多 2 个完整交换顺序；因此这是资源机制筛选，"
            "不是多路径最终确认实验。"
        ),
        "",
        "| Memory | λ | Candidate request cap | Q-DDCA | Q-CAST | SAA-Path | SAA-Path+Order | Order gap | W/T/L | End inventory |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for cell in cells:
        parameters = cell["parameters"]
        aggregate = cell["result"]["aggregate"]
        gap = cell["paired_order_gap"]
        path_key = (
            "saa_path" if "saa_path" in aggregate else "optimal_path"
        )
        order_key = (
            "saa_path_order"
            if "saa_path_order" in aggregate else "optimal_path_order"
        )
        lines.append(
            "| {memory} | {rate:g} | {cap} | {qddca:.3f} | {qcast:.3f} | "
            "{path_rate:.3f} | {order_rate:.3f} | {gap:+.3f} | "
            "{wins}/{ties}/{losses} | {inventory:.2f} |".format(
                memory=int(parameters["node_memory"]),
                rate=float(parameters["arrival_rate"]),
                cap=int(parameters["candidate_request_cap"]),
                qddca=aggregate["qddca_fixed"]["completion_rate"],
                qcast=aggregate["qcast_fixed"]["completion_rate"],
                path_rate=aggregate[path_key]["completion_rate"],
                order_rate=aggregate[order_key]["completion_rate"],
                gap=gap["mean"],
                wins=gap["wins"],
                ties=gap["ties"],
                losses=gap["losses"],
                inventory=aggregate[order_key][
                    "mean_inventory_pairs_at_slot_end"
                ],
            )
        )
    ranked = sorted(
        cells,
        key=lambda cell: cell["paired_order_gap"]["mean"],
        reverse=True,
    )
    lines.extend([
        "",
        "## 筛选结论",
        "",
    ])
    if ranked:
        best = ranked[0]
        worst = ranked[-1]
        lines.extend([
            (
                "- 最大平均 order gap：{key}，{gap:+.3f}。".format(
                    key=best["cell_id"],
                    gap=best["paired_order_gap"]["mean"],
                )
            ),
            (
                "- 最小平均 order gap：{key}，{gap:+.3f}。".format(
                    key=worst["cell_id"],
                    gap=worst["paired_order_gap"]["mean"],
                )
            ),
            (
                "- 这里只能定位值得复核的区域；3 个 seed 且单路径 catalogue "
                "不足以支撑统计显著性或最终创新结论。"
            ),
        ])
    lines.extend([
        "",
        "所有方法都在同一个 `OrderEpisodeEnv` 中运行：一个 episode 固定拓扑"
        "和请求流，一次 `env.step(batch_action)` 原子推进一个 control slot。"
        "有限 rollout 方法是逐 slot 的 myopic SAA planner，不是整个 episode "
        "的离线全局最优；自回归选择和 STOP（若采用）只在 policy 内部。",
        "",
        f"原始结果：`{payload['output_json']}`。",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sweep(
    *,
    output: Path,
    summary: Path,
    seeds: int = 3,
    oracle_rollouts: int = 4,
) -> dict[str, object]:
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        environment = payload.get("model", {}).get("environment")
        if environment != "OrderEpisodeEnv":
            raise ValueError(
                "existing checkpoint predates the multi-step OrderEpisodeEnv; "
                "use a new output path instead of mixing legacy slot-loop data"
            )
        if any(
            "candidate_request_cap" not in cell.get("parameters", {})
            for cell in payload.get("cells", {}).values()
        ):
            raise ValueError(
                "existing checkpoint uses the deprecated batch_size request "
                "cap; use a new output path instead of mixing semantics"
            )
    else:
        payload = {
            "model": {
                "environment": "OrderEpisodeEnv",
                "episode_semantics": (
                    "one fixed topology/request trace per episode; one env.step "
                    "advances exactly one control slot"
                ),
                "action_semantics": (
                    "one atomic multi-hot batch of complete path/swap-order plans"
                ),
                "inventory": "persistent elementary EPR, fixed TTL=3 slots",
                "candidate_paths": 1,
                "order_variants_per_path": 2,
                "request_scope": "edf_capped_active_pending",
                "candidate_request_caps": [2, 4, 6],
                "pruning_rule": (
                    "earliest_deadline_then_arrival_then_request_id"
                ),
                "note": "finite-sample SAA, resumable mechanism-screening sweep",
            },
            "seeds": seeds,
            "oracle_rollouts": oracle_rollouts,
            "output_json": output.as_posix(),
            "cells": {},
        }
    if payload["seeds"] != seeds or payload["oracle_rollouts"] != oracle_rollouts:
        raise ValueError("existing checkpoint uses different seed/rollout settings")

    total_started = perf_counter()
    for memory in (2, 4):
        for arrival_rate in (2.0, 4.0, 6.0):
            for candidate_request_cap in (2, 4, 6):
                key = _cell_id(memory, arrival_rate, candidate_request_cap)
                if key in payload["cells"]:
                    print(f"SKIP {key}", flush=True)
                    continue
                config = WaxmanOrderConfig(
                    node_count=100,
                    average_degree=6,
                    target_link_probability=0.6,
                    request_count=100,
                    arrival_rate=arrival_rate,
                    request_ttl_slots=10,
                    min_hops=2,
                    max_hops=6,
                    candidate_paths=1,
                    order_variants_per_path=2,
                    candidate_request_cap=candidate_request_cap,
                    node_memory_cap=memory,
                    slot_duration_ps=6_000,
                    generation_interval_ps=1_000,
                    swap_service_ps=1_000,
                    memory_reset_ps=100,
                    swap_probability=0.9,
                    bsm_capacity_per_node=1,
                    epr_ttl_slots=3,
                )
                started = perf_counter()
                print(f"START {key}", flush=True)
                result = run_suite(
                    seeds=seeds,
                    config=config,
                    oracle_rollouts=oracle_rollouts,
                    planner_names=PLANNERS,
                )
                cell = {
                    "cell_id": key,
                    "parameters": {
                        "node_memory": memory,
                        "arrival_rate": arrival_rate,
                        "candidate_request_cap": candidate_request_cap,
                    },
                    "elapsed_seconds": perf_counter() - started,
                    "result": result,
                }
                cell["paired_order_gap"] = _paired_order_gap(cell)
                payload["cells"][key] = cell
                payload["elapsed_seconds"] = sum(
                    float(value["elapsed_seconds"])
                    for value in payload["cells"].values()
                )
                _write_json(output, payload)
                _write_summary(summary, payload)
                print(
                    f"DONE {key} gap={cell['paired_order_gap']['mean']:+.3f} "
                    f"seconds={cell['elapsed_seconds']:.1f}",
                    flush=True,
                )
    del total_started
    payload["elapsed_seconds"] = sum(
        float(value["elapsed_seconds"])
        for value in payload["cells"].values()
    )
    _write_json(output, payload)
    _write_summary(summary, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Persistent-EPR Waxman memory/arrival/candidate-request-cap sweep"
        )
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--oracle-rollouts", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/order_waxman_episode_sweep_3seed_r4.json"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/order_waxman_episode_sweep_3seed_r4.md"),
    )
    args = parser.parse_args()
    run_sweep(
        output=args.output,
        summary=args.summary,
        seeds=args.seeds,
        oracle_rollouts=args.oracle_rollouts,
    )


if __name__ == "__main__":
    main()
