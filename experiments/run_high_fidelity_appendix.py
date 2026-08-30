from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import shutil
from statistics import fmean
from time import perf_counter

from algorithms.baselines.online import OnlineBaselineConfig, run_online_baseline
from algorithms.telgen.online import OnlineTELGENConfig, run_online_telgen
from qnet_core.scenario import ScenarioConfig, make_episode
from qnet_core.spec import EpisodeSpec, PhysicalConfig
from qnet_core.workload import resolve_periodic_arrival_workload

CHECKPOINT = Path(
    "results/generalization_formal_v1/models_pooled/seed_20260821/"
    "online_milp_gnn.pt"
)
OUTPUT_DIR = Path("results/r015_high_fidelity")
SEED_START = 75_000
SEEDS = 20

def physical_for(threshold: float) -> PhysicalConfig:
    initial_fidelity = {0.70: 0.90, 0.80: 0.94, 0.90: 0.965}[threshold]
    return PhysicalConfig(
        generation_probability=1.0,
        swap_probability=1.0,
        memory_capacity=2,
        max_width=1,
        memory_lifetime=100_000,
        initial_fidelity=initial_fidelity,
        swap_degradation=1.0,
        slot_duration_ps=1_000_000_000,
    )

def episode_for(seed: int, threshold: float) -> EpisodeSpec:
    workload = resolve_periodic_arrival_workload(
        request_count=20,
        arrival_rounds=None,
        requests_per_round=5,
        arrival_interval_slots=4,
        ttl_slots=16,
        horizon_slots=None,
        default_request_count=20,
    )
    scenario = ScenarioConfig(
        request_count=workload.request_count,
        min_hops=4,
        max_hops=4,
        ttl=16,
        horizon=workload.horizon_slots,
        physical=physical_for(threshold),
        topology_nodes=64,
        topology_mode="waxman",
        waxman_alpha=0.15,
        waxman_beta=0.45,
        waxman_add_mst=False,
        endpoint_mode="distance_stratified",
        arrival_batch_size=5,
        arrival_interval=4,
    )
    episode = make_episode(scenario, seed)
    return replace(
        episode,
        requests=tuple(
            replace(request, required_fidelity=threshold)
            for request in episode.requests
        ),
    )

def gnn_config() -> OnlineTELGENConfig:
    return OnlineTELGENConfig(
        decision_interval=4,
        path_candidate_count=4,
        construction_kinds=(),
        swap_tree_count=5,
        purification_kinds=("none", "elementary_once"),
        decision_backend="gnn",
        gnn_checkpoint=str(CHECKPOINT),
        gnn_device="cpu",
    )

def baseline_config(algorithm: str) -> OnlineBaselineConfig:
    return OnlineBaselineConfig(
        algorithm=algorithm,
        decision_interval=4,
        path_candidate_count=4,
        construction_kind="left_deep",
    )

def method_payload(result, wall_seconds: float) -> dict:
    attempts = tuple(getattr(result, "attempts", ()))
    purifications = tuple(
        item
        for item in attempts
        if getattr(item, "purification_kind", "none") != "none"
    )
    successes = tuple(item for item in purifications if item.success is True)
    return {
        "metrics": dict(result.metrics),
        "wall_seconds": wall_seconds,
        "violation_count": len(result.violations),
        "violations": [asdict(item) for item in result.violations],
        "attempt_count": len(attempts),
        "purification_attempt_count": len(purifications),
        "purification_success_count": len(successes),
    }

def mean_metrics(rows: list[dict]) -> dict:
    keys = sorted(set().union(*(row.keys() for row in rows)))
    return {
        key: fmean(float(row.get(key, 0.0)) for row in rows)
        for key in keys
    }

def run_threshold(threshold: float) -> dict:
    methods = {
        "gnn": ("telgen", gnn_config()),
        "qpath": ("baseline", baseline_config("qpath")),
        "qleap": ("baseline", baseline_config("qleap")),
    }
    trials = []
    started = perf_counter()
    for index, seed in enumerate(range(SEED_START, SEED_START + SEEDS)):
        episode = episode_for(seed, threshold)
        entry = {"seed": seed, "episode": asdict(episode), "methods": {}}
        for name, (kind, config) in methods.items():
            t0 = perf_counter()
            result = (
                run_online_telgen(episode, config)
                if kind == "telgen"
                else run_online_baseline(episode, config)
            )
            entry["methods"][name] = method_payload(result, perf_counter() - t0)
        trials.append(entry)
        summary = " ".join(
            f"{name}={int(item['metrics']['completed_requests'])}"
            for name, item in entry["methods"].items()
        )
        print(
            f"F={threshold:.2f} seed={seed} ({index + 1}/{SEEDS}) {summary}",
            flush=True,
        )
    aggregate = {}
    for name in methods:
        aggregate[name] = {
            "metrics": mean_metrics([
                trial["methods"][name]["metrics"] for trial in trials
            ]),
            "violation_count": sum(
                trial["methods"][name]["violation_count"] for trial in trials
            ),
            "purification_attempt_count": fmean(
                trial["methods"][name]["purification_attempt_count"]
                for trial in trials
            ),
            "purification_success_count": fmean(
                trial["methods"][name]["purification_success_count"]
                for trial in trials
            ),
            "wall_seconds": fmean(
                trial["methods"][name]["wall_seconds"] for trial in trials
            ),
        }
    return {
        "threshold": threshold,
        "seed_start": SEED_START,
        "seed_count": SEEDS,
        "physical": asdict(physical_for(threshold)),
        "methods": sorted(methods),
        "aggregate": aggregate,
        "trials": trials,
        "elapsed_seconds": perf_counter() - started,
    }

def save(payload: dict) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned = OUTPUT_DIR / f"high_fidelity_appendix_{stamp}.json"
    latest = OUTPUT_DIR / "high_fidelity_appendix.json"
    versioned.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copyfile(versioned, latest)
    return versioned, latest

def main() -> int:
    payload = {
        "schema_version": 1,
        "experiment": "r015_high_fidelity_appendix",
        "checkpoint": str(CHECKPOINT),
        "contract": {
            "paired_episode_spec": True,
            "independent_persistent_executors": True,
            "future_requests_hidden": True,
            "methods": ["gnn", "qpath", "qleap"],
            "purification_protocol": "elementary_once",
            "hard_violation_gate": "zero",
        },
        "thresholds": {},
    }
    for threshold in (0.70, 0.80, 0.90):
        payload["thresholds"][f"{threshold:.2f}"] = run_threshold(threshold)
    versioned, _ = save(payload)
    for threshold in (0.70, 0.80, 0.90):
        block = payload["thresholds"][f"{threshold:.2f}"]
        print(
            f"F={threshold:.2f}: "
            + " ".join(
                f"{name}="
                f"{block['aggregate'][name]['metrics']['completed_requests']:.3f}/"
                f"{block['aggregate'][name]['purification_attempt_count']:.1f}pur/"
                f"{int(block['aggregate'][name]['violation_count'])}viol"
                for name in ("gnn", "qpath", "qleap")
            )
        )
    print(f"json: {versioned}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
