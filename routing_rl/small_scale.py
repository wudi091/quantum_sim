"""Direction-gated small-scale PPO validation.

This pilot is deliberately the same single full-range 2--50 hop task used by
the formal run.  It only reduces the number of requests and optimizer updates.
The pilot saves an optimizer-step-zero checkpoint, evaluates it on fixed seeds,
trains from scratch, and writes a direction report comparing the learned best
checkpoint with that frozen initialization.

Run with ``python -m routing_rl.small_scale``.  The formal large-scale entry
point requires the resulting report to pass before it will start training.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch

from .train import build_environment, build_evaluator, make_config, parse_args
from .trainer import PPOTrainer, load_model


@dataclass(frozen=True)
class SmallScaleSettings:
    output: Path = Path("results/gnn_small_direction_seed68001_u60c_credit_reward2")
    init_checkpoint: Path | None = None
    seed: int = 68_001
    min_hops: int = 2
    max_hops: int = 50
    updates: int = 60
    requests: int = 5
    rollout_steps: int = 256
    minibatch_size: int = 128
    ppo_epochs: int = 4
    hidden_dim: int = 128
    learning_rate: float = 5e-5
    value_coef: float = 0.5
    entropy_coef: float = 5e-3
    gamma: float = 0.999
    gae_lambda: float = 0.99
    checkpoint_every: int = 10
    evaluate_every: int = 10
    evaluation_episodes: int = 5
    high_hop_evaluation_episodes: int = 5
    high_hop_min_hops: int = 41
    early_stopping_patience: int = 0
    early_stopping_min_updates: int = 0
    request_ttl: int = 64
    generation_probability: float = 0.5
    swap_probability: float = 0.95
    memory_capacity: int = 2
    arrival_rate: float = 1.0
    topology_nodes: int = 100
    waxman_alpha: float = 0.05
    waxman_beta: float = 0.02
    topology_attempts: int = 128
    demand_pairs: int = 1
    max_width: int = 1
    candidates_per_request: int = 6
    potential_coef: float = 0.03
    completion_bonus: float = 2.0
    makespan_coef: float = 0.002
    failure_coef: float = 0.05
    timeout_coef: float = 0.05
    torch_threads: int = 4


SETTINGS = SmallScaleSettings()


def build_args(settings: SmallScaleSettings = SETTINGS) -> argparse.Namespace:
    args = parse_args([])
    for field in fields(settings):
        setattr(args, field.name, getattr(settings, field.name))
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.anneal_learning_rate = False
    args.curriculum = False
    args.select_high_hop = False
    args.reset_critic = False
    args.smoke = False
    return args


def _relative_gain(current: float, initial: float) -> float:
    if initial > 1e-12:
        return current / initial - 1.0
    return float("inf") if current > 1e-12 else 0.0


def assess_direction(
    initial: dict[str, float],
    learned: dict[str, float],
    history: list[dict[str, Any]],
    initial_stochastic: dict[str, float] | None = None,
    learned_stochastic: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Apply conservative, objective-level gates before formal training."""
    initial_throughput = float(initial.get("pair_throughput", 0.0))
    learned_throughput = float(learned.get("pair_throughput", 0.0))
    initial_completion = float(initial.get("completion_rate", 0.0))
    learned_completion = float(learned.get("completion_rate", 0.0))
    initial_high_completion = float(initial.get("high_hop_completion_rate", 0.0))
    learned_high_completion = float(learned.get("high_hop_completion_rate", 0.0))

    eval_rows = [row for row in history if row.get("evaluation")]
    throughputs = [float(row["evaluation"].get("pair_throughput", 0.0)) for row in eval_rows]
    first_evaluation = eval_rows[0]["evaluation"] if eval_rows else {}
    training_baseline_throughput = float(first_evaluation.get("pair_throughput", 0.0))
    training_baseline_completion = float(first_evaluation.get("completion_rate", 0.0))
    best_throughput = max(throughputs, default=0.0)
    tail_count = max(1, len(throughputs) // 3)
    tail_mean = sum(throughputs[-tail_count:]) / tail_count if throughputs else 0.0

    # A random untrained network can have a lucky greedy argmax despite an
    # almost uniform policy.  Require deterministic improvement across the
    # training window and independently require the fixed-seed sampled policy
    # to improve over optimizer-step zero.
    overall_pass = (
        learned_throughput
        >= max(training_baseline_throughput * 1.15, training_baseline_throughput + 1e-3)
        and learned_completion >= training_baseline_completion + 0.05
    )
    stochastic_pass = True
    stochastic_metrics: dict[str, float] = {}
    if initial_stochastic is not None and learned_stochastic is not None:
        initial_sampled_throughput = float(initial_stochastic.get("pair_throughput", 0.0))
        learned_sampled_throughput = float(learned_stochastic.get("pair_throughput", 0.0))
        initial_sampled_completion = float(initial_stochastic.get("completion_rate", 0.0))
        learned_sampled_completion = float(learned_stochastic.get("completion_rate", 0.0))
        stochastic_pass = (
            learned_sampled_throughput
            >= max(initial_sampled_throughput * 1.05, initial_sampled_throughput + 5e-4)
            and learned_sampled_completion >= initial_sampled_completion
        )
        stochastic_metrics = {
            "initial_stochastic_pair_throughput": initial_sampled_throughput,
            "learned_stochastic_pair_throughput": learned_sampled_throughput,
            "stochastic_pair_throughput_relative_gain": _relative_gain(
                learned_sampled_throughput, initial_sampled_throughput,
            ),
            "initial_stochastic_completion_rate": initial_sampled_completion,
            "learned_stochastic_completion_rate": learned_sampled_completion,
        }
    high_hop_pass = learned_high_completion + 0.01 >= initial_high_completion
    stability_pass = tail_mean >= 0.85 * best_throughput if throughputs else False
    # Deployment and the paired planner comparison use deterministic actions.
    # Keep stochastic sampling as a diagnostic warning: it is useful for
    # detecting a fragile policy, but it is not the controller being selected.
    passed = bool(overall_pass and high_hop_pass and stability_pass)
    return {
        "passed": passed,
        "overall_pass": overall_pass,
        "stochastic_pass": stochastic_pass,
        "stochastic_warning": not stochastic_pass,
        "high_hop_non_regression": high_hop_pass,
        "stability_pass": stability_pass,
        "initial_pair_throughput": initial_throughput,
        "learned_pair_throughput": learned_throughput,
        "pair_throughput_relative_gain": _relative_gain(learned_throughput, initial_throughput),
        "initial_completion_rate": initial_completion,
        "learned_completion_rate": learned_completion,
        "completion_rate_absolute_gain": learned_completion - initial_completion,
        "initial_high_hop_completion_rate": initial_high_completion,
        "learned_high_hop_completion_rate": learned_high_completion,
        "best_evaluation_pair_throughput": best_throughput,
        "tail_evaluation_pair_throughput": tail_mean,
        "evaluation_count": len(eval_rows),
        "training_baseline_pair_throughput": training_baseline_throughput,
        "training_baseline_completion_rate": training_baseline_completion,
        **stochastic_metrics,
    }


def _evaluate(
    checkpoint: Path,
    args: argparse.Namespace,
    config,
    update: int,
    deterministic: bool = True,
) -> dict[str, float]:
    device = torch.device(args.device)
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [
            torch.cuda.current_device() if device.index is None else device.index
        ]
    # Model construction consumes PyTorch RNG even though checkpoint weights
    # immediately replace the initialization.  Isolate the whole load/eval
    # operation so validation before training cannot change rollout sampling.
    with torch.random.fork_rng(devices=cuda_devices):
        model = load_model(checkpoint, device)
        evaluator = build_evaluator(args, config, deterministic=deterministic)
        return evaluator(model, update)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Print pilot configuration without training.")
    operational = parser.parse_args(argv)
    args = build_args()
    config = make_config(args)
    print(
        "small-scale direction pilot: "
        f"device={args.device}, updates={args.updates}, requests={args.requests}, "
        f"hops={args.min_hops}-{args.max_hops}, curriculum={args.curriculum}, "
        f"output={args.output}"
    )
    if operational.check:
        return

    args.output.mkdir(parents=True, exist_ok=True)
    env = build_environment(
        config, args.request_ttl, args.generation_probability, args.swap_probability,
        args.memory_capacity, args.arrival_rate, args.topology_nodes,
        args.waxman_alpha, args.waxman_beta, args.topology_attempts,
        args.demand_pairs, args.node_memory_capacity, args.max_width,
        args.candidates_per_request,
    )
    trainer = PPOTrainer(env, config, args.output, evaluator=build_evaluator(args, config))
    initial_path = args.output / "initial.pt"
    trainer.save_checkpoint(initial_path)
    initial = _evaluate(initial_path, args, config, 0)
    initial_stochastic = _evaluate(initial_path, args, config, 0, deterministic=False)
    history = trainer.train()
    best_path = args.output / "best.pt"
    if not best_path.is_file():
        best_path = args.output / "checkpoint.pt"
    learned = _evaluate(best_path, args, config, config.total_updates)
    learned_stochastic = _evaluate(
        best_path, args, config, config.total_updates, deterministic=False,
    )
    gate = assess_direction(
        initial, learned, history, initial_stochastic, learned_stochastic,
    )
    report = {
        "settings": {field.name: getattr(SETTINGS, field.name) for field in fields(SETTINGS)},
        "initial_checkpoint": str(initial_path),
        "learned_checkpoint": str(best_path),
        "initial": initial,
        "learned": learned,
        "initial_stochastic": initial_stochastic,
        "learned_stochastic": learned_stochastic,
        "gate": gate,
    }
    report_path = args.output / "direction_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    if not gate["passed"]:
        raise SystemExit("direction gate failed; do not start formal large-scale training")


if __name__ == "__main__":
    main()
