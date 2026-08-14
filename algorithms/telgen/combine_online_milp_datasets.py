"""Combine completed online MILP collections into one validated suite."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil

from .online_milp_dataset import (
    load_online_milp_dataset,
    resolve_online_milp_dataset_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine disjoint online MILP collections."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--input",
        action="append",
        nargs=3,
        metavar=("NAME", "ROLE", "PATH"),
        required=True,
        help="Collection name, split role, and manifest/directory path.",
    )
    return parser


def _relative_manifest(path: Path, output: Path) -> str:
    return Path(os.path.relpath(path.resolve(), output.resolve())).as_posix()


def combine_collections(
    output: Path,
    profile: str,
    inputs: list[tuple[str, str, Path]],
) -> dict[str, object]:
    if not profile:
        raise ValueError("profile cannot be empty")
    if not inputs:
        raise ValueError("at least one collection is required")
    names = [name for name, _, _ in inputs]
    if len(set(names)) != len(names):
        raise ValueError("collection names must be unique")
    valid_roles = {"train", "validation", "test"}
    invalid_roles = sorted({role for _, role, _ in inputs} - valid_roles)
    if invalid_roles:
        raise ValueError(f"unsupported split role: {invalid_roles[0]}")

    output.mkdir(parents=True, exist_ok=True)
    groups: list[dict[str, object]] = []
    seen_seeds: set[int] = set()
    seen_samples: set[Path] = set()
    for name, role, raw_path in inputs:
        manifest = resolve_online_milp_dataset_manifest(raw_path)
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        if manifest_payload.get("dataset_kind") != "online_milp_teacher_collection":
            raise ValueError(f"{name} is not a completed collection manifest")
        if manifest_payload.get("collection_complete", True) is not True:
            raise ValueError(f"{name} collection is incomplete")
        loaded = load_online_milp_dataset(manifest)
        seeds = loaded.episode_seeds
        overlap = seen_seeds.intersection(seeds)
        if overlap:
            raise ValueError(f"duplicate episode seed: {min(overlap)}")
        sample_paths = {path.resolve() for path in loaded.sample_paths}
        duplicated_samples = seen_samples.intersection(sample_paths)
        if duplicated_samples:
            raise ValueError(
                f"duplicate graph sample: {next(iter(duplicated_samples))}"
            )
        seen_seeds.update(seeds)
        seen_samples.update(sample_paths)
        groups.append({
            "name": name,
            "role": role,
            "manifest": _relative_manifest(manifest, output),
            "episode_count": len(seeds),
            "sample_count": len(loaded.samples),
            "seeds": list(seeds),
            "scenario": manifest_payload.get("scenario"),
            "configuration": manifest_payload.get("configuration"),
        })

    roles = {
        role: sorted(
            seed
            for group in groups
            if group["role"] == role
            for seed in group["seeds"]
        )
        for role in sorted(valid_roles)
    }
    if any(not roles[role] for role in valid_roles):
        raise ValueError("train, validation, and test roles must all be non-empty")
    return {
        "schema_version": 1,
        "dataset_kind": "online_milp_teacher_collection",
        "collection_complete": True,
        "collection_profile": profile,
        "episode_count": len(seen_seeds),
        "sample_count": len(seen_samples),
        "scenario_group_count": len(groups),
        "split_episode_seeds": roles,
        "episodes": groups,
    }


def save_combined_collection(
    output: Path,
    payload: dict[str, object],
) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned = output / f"online_milp_dataset_{timestamp}.json"
    latest = output / "online_milp_dataset.json"
    versioned.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    shutil.copyfile(versioned, latest)
    return versioned, latest


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = [
        (str(name), str(role), Path(path))
        for name, role, path in args.input
    ]
    payload = combine_collections(args.output, args.profile, inputs)
    versioned, latest = save_combined_collection(args.output, payload)
    print(
        f"groups={payload['scenario_group_count']} "
        f"episodes={payload['episode_count']} samples={payload['sample_count']}"
    )
    print(f"manifest: {versioned}")
    print(f"latest: {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
