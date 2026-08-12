"""Command-line entry point for static-batch purification A/B evaluation."""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import subprocess
import sys

from qnet_core.scenario import ScenarioConfig, make_episode
from qnet_core.spec import PhysicalConfig

from .purification_ab import run_purification_ab, save_purification_ab_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="比较静态 batch 中禁用纯化与按需 BBPSSW 纯化",
    )
    parser.add_argument("--episode-seed", type=int, default=2026)
    parser.add_argument("--physical-seed-start", type=int, default=10000)
    parser.add_argument("--physical-trials", type=int, default=8)
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--min-hops", type=int, default=2)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--ttl", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--topology-nodes", type=int, default=16)
    parser.add_argument("--path-candidates", type=int, default=3)
    parser.add_argument("--required-fidelity", type=float, default=0.69)
    parser.add_argument(
        "--required-fidelity-pattern",
        default="",
        help=(
            "逗号分隔的保真度阈值，按请求顺序循环应用；"
            "例如 0.60,0.69"
        ),
    )
    parser.add_argument("--initial-fidelity", type=float, default=0.80)
    parser.add_argument("--swap-degradation", type=float, default=1.0)
    parser.add_argument("--generation-probability", type=float, default=1.0)
    parser.add_argument("--swap-probability", type=float, default=1.0)
    parser.add_argument("--memory-capacity", type=int, default=2)
    parser.add_argument("--node-memory-capacity", type=int, default=8)
    parser.add_argument("--memory-lifetime", type=int, default=1000)
    parser.add_argument("--slot-duration-ps", type=int, default=1_000_000)
    parser.add_argument("--decoder-beam-width", type=int, default=256)
    parser.add_argument("--decoder-random-restarts", type=int, default=64)
    parser.add_argument(
        "--construction-kinds",
        default="left_deep,balanced",
        help="逗号分隔的交换树类型",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "purification_ab",
    )
    return parser


def _resolve_required_fidelities(
    request_count: int,
    uniform_fidelity: float,
    pattern_text: str,
) -> tuple[float, ...]:
    """Resolve one uniform threshold or a cyclic mixed-threshold pattern."""

    if request_count < 1:
        raise ValueError("request-count must be positive")
    if pattern_text.strip():
        pieces = tuple(item.strip() for item in pattern_text.split(","))
        if any(not item for item in pieces):
            raise ValueError("required-fidelity-pattern contains an empty item")
        try:
            pattern = tuple(float(item) for item in pieces)
        except ValueError as exc:
            raise ValueError(
                "required-fidelity-pattern must contain only numbers"
            ) from exc
    else:
        pattern = (float(uniform_fidelity),)
    if any(not 0.5 <= value <= 1.0 for value in pattern):
        raise ValueError("required fidelities must be in [0.5, 1.0]")
    return tuple(
        pattern[index % len(pattern)]
        for index in range(request_count)
    )


def _repository_revision() -> str | None:
    """Return the repository commit with a dirty marker when available."""

    repository = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return f"{revision}-dirty" if dirty else revision


def _source_snapshot() -> tuple[str, tuple[tuple[str, str], ...]]:
    """Hash the planning, physical-boundary, and environment source inputs."""

    repository = Path(__file__).resolve().parents[2]
    targets = [
        repository / "algorithms" / "__init__.py",
        *sorted((repository / "algorithms" / "telgen").rglob("*.py")),
        *sorted((repository / "qnet_core").rglob("*.py")),
        repository / "environment.yml",
        repository / "requirements.txt",
        repository / "pytest.ini",
    ]
    file_hashes = []
    combined = sha256()
    for target in targets:
        relative = target.relative_to(repository).as_posix()
        digest = sha256(target.read_bytes()).hexdigest()
        file_hashes.append((relative, digest))
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\n")
    return combined.hexdigest(), tuple(file_hashes)


def main() -> None:
    args = _parser().parse_args()
    if args.physical_trials < 1:
        raise ValueError("physical-trials must be positive")
    physical = PhysicalConfig(
        generation_probability=args.generation_probability,
        swap_probability=args.swap_probability,
        memory_capacity=args.memory_capacity,
        memory_lifetime=args.memory_lifetime,
        initial_fidelity=args.initial_fidelity,
        swap_degradation=args.swap_degradation,
        node_memory_capacity=args.node_memory_capacity,
        max_width=1,
        quantum_distance_m=1.0,
        slot_duration_ps=args.slot_duration_ps,
        detector_efficiency=1.0,
        bsm_success_probability=1.0,
    )
    scenario = ScenarioConfig(
        request_count=args.request_count,
        min_hops=args.min_hops,
        max_hops=args.max_hops,
        ttl=args.ttl,
        horizon=args.horizon,
        physical=physical,
        topology_nodes=args.topology_nodes,
    )
    episode = make_episode(scenario, args.episode_seed)
    required_fidelities = _resolve_required_fidelities(
        len(episode.requests),
        args.required_fidelity,
        args.required_fidelity_pattern,
    )
    episode = replace(
        episode,
        requests=tuple(
            replace(request, required_fidelity=required_fidelities[index])
            for index, request in enumerate(episode.requests)
        ),
    )
    physical_seeds = tuple(
        args.physical_seed_start + index
        for index in range(args.physical_trials)
    )
    construction_kinds = tuple(
        item.strip()
        for item in args.construction_kinds.split(",")
        if item.strip()
    )
    source_tree_sha256, source_file_hashes = _source_snapshot()
    report = run_purification_ab(
        episode,
        physical_seeds,
        path_candidate_count=args.path_candidates,
        construction_kinds=construction_kinds,
        decoder_beam_width=args.decoder_beam_width,
        decoder_random_restarts=args.decoder_random_restarts,
        code_revision=_repository_revision(),
        working_directory=str(Path.cwd()),
        run_command=tuple(sys.orig_argv),
        source_tree_sha256=source_tree_sha256,
        source_file_hashes=source_file_hashes,
    )
    paths = save_purification_ab_report(report, args.output_dir)
    baseline = report.baseline
    on_demand = report.on_demand
    print(
        "请求保真度阈值："
        + ", ".join(f"{value:.3f}" for value in required_fidelities)
    )
    print(
        "无纯化：计划接纳 "
        f"{baseline.planned_selected_requests}，平均物理完成 "
        f"{baseline.mean_completed_requests:.3f}"
    )
    print(
        "按需纯化：计划接纳 "
        f"{on_demand.planned_selected_requests}，其中纯化 "
        f"{on_demand.planned_purified_requests}，平均物理完成 "
        f"{on_demand.mean_completed_requests:.3f}"
    )
    if on_demand.purification_success_rate is not None:
        print(
            "纯化事件成功率："
            f"{100.0 * on_demand.purification_success_rate:.2f}%"
        )
    print(
        "说明：两组使用相同 episode 与 seed 标签，但不是严格共同随机数；"
        "当前结果仅用于单 episode sanity 检查。"
    )
    print(f"结果 JSON：{paths.latest_json}")
    print(f"逐次实验 CSV：{paths.latest_csv}")


if __name__ == "__main__":
    main()
