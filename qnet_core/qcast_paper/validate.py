"""Validate official Q-CAST sweep output against SIGCOMM 2020 figures.

Canonical input schema::

    {
      "source_commit": "...",
      "slots_per_point": 1000,
      "sweeps": {
        "k": {"x": [...], "algorithms": {"QCAST": [...], ...}},
        "p": {"x": [...], "algorithms": {"QCAST": [...], ...}},
        "q": {"x": [...], "algorithms": {"QCAST": [...], ...}},
        "n": {"x": [...], "algorithms": {"QCAST": [...], ...}},
        "m": {"x": [...], "algorithms": {"QCAST": [...], ...}}
      },
      "recovery": {
        "QCAST": {"on": 0.0, "off": 0.0},
        "QPASS_CR": {"on": 0.0, "off": 0.0}
      }
    }

The validator also accepts the official runner's row-list output.  Each row is
``{fig, parameter, value, algorithm, n, p, q, k, m, slots, topologies,
throughput, success_pairs}``; author names ``Online/CR/Greedy_H/SL`` are
mapped to the canonical algorithm names below.

The paper does not publish raw tables.  ``PAPER_TARGETS`` therefore records
values extracted from the PDF's vector figures, not claimed author data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


OFFICIAL_COMMIT = "2db9e716fe948aed198a4fb1f5ee1335e7d7d6e1"
ALGORITHMS = ("QCAST", "QPASS_CR", "Greedy", "SLMP")
ALGORITHM_ALIASES = {
    "QCAST": "QCAST", "Online": "QCAST",
    "QPASS_CR": "QPASS_CR", "CR": "QPASS_CR",
    "Greedy": "Greedy", "Greedy_H": "Greedy",
    "SLMP": "SLMP", "SL": "SLMP",
}
PAPER_TARGETS: dict[str, Any] = {
    "provenance": "PDF vector extraction from SIGCOMM 2020 figures 16--20",
    "sweeps": {
        "k": {
            "x": [0, 3, 6, 10000],
            "algorithms": {
                "QCAST": [12.300, 13.063, 13.430, 13.226],
                "QPASS_CR": [9.524, 9.080, 9.697, 9.290],
                "Greedy": [8.263, 8.085, 8.427, 8.150],
                "SLMP": [2.323, 2.559, 2.598, 2.568],
            },
        },
        "p": {
            "x": [0.1, 0.3, 0.6, 0.9],
            "algorithms": {
                "QCAST": [0.340, 3.471, 13.063, 29.043],
                "QPASS_CR": [0.240, 2.287, 9.079, 20.805],
                "Greedy": [0.160, 1.839, 8.084, 20.477],
                "SLMP": [0.026, 0.343, 2.561, 12.702],
            },
        },
        "q": {
            "x": [0.8, 0.85, 0.9, 0.95, 1.0],
            "algorithms": {
                "QCAST": [9.839, 11.259, 13.064, 15.920, 18.663],
                "QPASS_CR": [6.723, 7.745, 9.080, 11.003, 13.707],
                "Greedy": [6.354, 6.985, 8.085, 9.641, 11.195],
                "SLMP": [2.092, 2.266, 2.559, 3.058, 3.410],
            },
        },
        "n": {
            "x": [50, 100, 200, 400, 800],
            "algorithms": {
                "QCAST": [14.332, 13.065, 10.916, 8.915, 6.008],
                "QPASS_CR": [9.962, 9.081, 7.659, 6.112, 4.388],
                "Greedy": [9.274, 8.087, 6.281, 4.637, 3.228],
                "SLMP": [3.910, 2.560, 1.505, 0.821, 0.489],
            },
        },
        "m": {
            "x": list(range(1, 11)),
            "algorithms": {
                "QCAST": [2.084, 3.937, 5.610, 6.981, 8.265, 9.517, 10.700, 11.629, 12.460, 13.065],
                "QPASS_CR": [2.179, 3.880, 5.234, 6.269, 6.872, 7.692, 8.094, 8.482, 8.747, 9.081],
                "Greedy": [1.749, 2.962, 4.000, 4.921, 5.562, 6.210, 6.948, 7.471, 7.766, 8.086],
                "SLMP": [0.375, 0.652, 0.960, 1.199, 1.471, 1.711, 1.938, 2.139, 2.341, 2.560],
            },
        },
    },
    "recovery": {
        "QCAST": {"on": 13.062, "off": 11.346},
        "QPASS_CR": {"on": 9.077, "off": 8.634},
    },
}


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _numbers(value: Any, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(_is_finite_number(item) for item in value)
    )


def _same_numbers(left: Sequence[float], right: Sequence[float], tolerance: float = 1e-9) -> bool:
    return len(left) == len(right) and all(
        math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)
        for a, b in zip(left, right)
    )


def _ranks(values: Sequence[float]) -> list[float]:
    """Return average ranks, including correct handling of ties."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            result[index] = rank
        start = end
    return result


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    x = _ranks(left)
    y = _ranks(right)
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_norm = math.sqrt(sum((a - x_mean) ** 2 for a in x))
    y_norm = math.sqrt(sum((b - y_mean) ** 2 for b in y))
    if x_norm == 0.0 or y_norm == 0.0:
        return 1.0 if _same_numbers(left, right) else 0.0
    return numerator / (x_norm * y_norm)


def _curve_stats(target: Sequence[float], observed: Sequence[float]) -> dict[str, float]:
    errors = [float(value) - float(reference) for reference, value in zip(target, observed)]
    scale = max(sum(abs(float(value)) for value in target) / len(target), 1e-12)
    return {
        "spearman": spearman(target, observed),
        "mae_eps": sum(abs(value) for value in errors) / len(errors),
        "rmse_eps": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "normalized_mae": sum(abs(value) for value in errors) / len(errors) / scale,
        "mean_relative_error": sum(
            abs(error) / max(abs(float(reference)), 1e-12)
            for reference, error in zip(target, errors)
        ) / len(errors),
        "bias_eps": sum(errors) / len(errors),
    }


def _ordering(algorithms: Mapping[str, Sequence[float]], index: int) -> list[str]:
    return sorted(ALGORITHMS, key=lambda name: (-float(algorithms[name][index]), ALGORITHMS.index(name)))


def _recovery_state(value: Any) -> str | None:
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value).strip().lower()
    if text in {"on", "true", "1", "enabled", "with"}:
        return "on"
    if text in {"off", "false", "0", "disabled", "without", "no_recovery"}:
        return "off"
    return None


def _rows_to_payload(rows: list[Any]) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Normalize the official runner's long rows to the canonical schema."""
    errors: list[str] = []
    required = {
        "fig", "parameter", "value", "algorithm", "n", "p", "q", "k", "m",
        "slots", "topologies", "throughput", "success_pairs",
    }
    buckets: dict[tuple[str, str, int], list[float]] = {}
    recovery_buckets: dict[tuple[str, str], list[float]] = {}
    slots_seen: set[Any] = set()
    topologies_seen: set[Any] = set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row {row_index} must be an object")
            continue
        missing = sorted(required - row.keys())
        if missing:
            errors.append(f"row {row_index} is missing fields: {', '.join(missing)}")
            continue
        slots_seen.add(row["slots"])
        topologies_seen.add(row["topologies"])
        if not _is_finite_number(row["throughput"]):
            errors.append(f"row {row_index}.throughput must be finite numeric")
            continue
        if not _is_finite_number(row["success_pairs"]):
            errors.append(f"row {row_index}.success_pairs must be finite numeric")
        raw_algorithm = str(row["algorithm"])
        no_recovery_suffix = raw_algorithm.endswith("-R")
        algorithm = ALGORITHM_ALIASES.get(raw_algorithm[:-2] if no_recovery_suffix else raw_algorithm)
        if algorithm is None:
            errors.append(f"row {row_index} has unknown algorithm {raw_algorithm!r}")
            continue
        parameter = str(row["parameter"]).strip().lower()
        if parameter in PAPER_TARGETS["sweeps"]:
            target_x = PAPER_TARGETS["sweeps"][parameter]["x"]
            if not _is_finite_number(row["value"]):
                errors.append(f"row {row_index}.value must be numeric for sweep {parameter}")
                continue
            matches = [
                index for index, target_value in enumerate(target_x)
                if math.isclose(float(row["value"]), float(target_value), rel_tol=1e-9, abs_tol=1e-9)
            ]
            if not matches:
                errors.append(f"row {row_index} has unexpected {parameter} value {row['value']!r}")
                continue
            buckets.setdefault((parameter, algorithm, matches[0]), []).append(float(row["throughput"]))
        elif parameter in {"recovery", "r"}:
            if algorithm not in PAPER_TARGETS["recovery"]:
                continue
            state = "off" if no_recovery_suffix else _recovery_state(row["value"])
            if state is None:
                errors.append(f"row {row_index} has unknown recovery state {row['value']!r}")
                continue
            recovery_buckets.setdefault((algorithm, state), []).append(float(row["throughput"]))
        else:
            errors.append(f"row {row_index} has unknown parameter {row['parameter']!r}")

    sweeps: dict[str, Any] = {}
    for sweep_name, target in PAPER_TARGETS["sweeps"].items():
        algorithms: dict[str, list[float | None]] = {}
        for algorithm in ALGORITHMS:
            values: list[float | None] = []
            for index in range(len(target["x"])):
                samples = buckets.get((sweep_name, algorithm, index), [])
                values.append(sum(samples) / len(samples) if samples else None)
            algorithms[algorithm] = values
        sweeps[sweep_name] = {"x": list(target["x"]), "algorithms": algorithms}
    recovery: dict[str, Any] = {}
    for algorithm in PAPER_TARGETS["recovery"]:
        recovery[algorithm] = {}
        for state in ("on", "off"):
            samples = recovery_buckets.get((algorithm, state), [])
            recovery[algorithm][state] = sum(samples) / len(samples) if samples else None
    if slots_seen != {1000}:
        errors.append(f"all row slots must equal 1000; observed {sorted(map(str, slots_seen))}")
    if topologies_seen != {10}:
        errors.append(f"all row topologies must equal 10; observed {sorted(map(str, topologies_seen))}")
    canonical = {
        "source_commit": None,
        "slots_per_point": 1000 if slots_seen == {1000} else None,
        "sweeps": sweeps,
        "recovery": recovery,
    }
    metadata = {
        "row_count": len(rows),
        "slots_observed": sorted(slots_seen, key=str),
        "topologies_observed": sorted(topologies_seen, key=str),
        "source_commit_available": False,
    }
    return canonical, errors, metadata


def _config_report(
    payload: Any,
    *,
    require_commit: bool = True,
    initial_errors: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = list(initial_errors)
    if not isinstance(payload, dict):
        return {"pass": False, "errors": ["root must be a JSON object"]}
    if require_commit and payload.get("source_commit") != OFFICIAL_COMMIT:
        errors.append(f"source_commit must equal {OFFICIAL_COMMIT}")
    if payload.get("slots_per_point") != 1000:
        errors.append("slots_per_point must equal the paper protocol value 1000")
    sweeps = payload.get("sweeps")
    if not isinstance(sweeps, dict):
        errors.append("sweeps must be an object")
        sweeps = {}
    for sweep_name, target in PAPER_TARGETS["sweeps"].items():
        sweep = sweeps.get(sweep_name)
        if not isinstance(sweep, dict):
            errors.append(f"sweeps.{sweep_name} must be an object")
            continue
        x = sweep.get("x")
        if not _numbers(x, len(target["x"])) or not _same_numbers(x, target["x"]):
            errors.append(f"sweeps.{sweep_name}.x must equal {target['x']}")
        algorithms = sweep.get("algorithms")
        if not isinstance(algorithms, dict):
            errors.append(f"sweeps.{sweep_name}.algorithms must be an object")
            continue
        for algorithm in ALGORITHMS:
            if not _numbers(algorithms.get(algorithm), len(target["x"])):
                errors.append(
                    f"sweeps.{sweep_name}.algorithms.{algorithm} must contain "
                    f"{len(target['x'])} finite numbers"
                )
    recovery = payload.get("recovery")
    if not isinstance(recovery, dict):
        errors.append("recovery must be an object")
        recovery = {}
    for algorithm in PAPER_TARGETS["recovery"]:
        levels = recovery.get(algorithm)
        if not isinstance(levels, dict) or not all(_is_finite_number(levels.get(key)) for key in ("on", "off")):
            errors.append(f"recovery.{algorithm} must contain finite numeric on/off values")
    result = {
        "pass": not errors,
        "errors": errors,
        "source_commit_matches": (
            payload.get("source_commit") == OFFICIAL_COMMIT if require_commit else None
        ),
        "slots_per_point_matches": payload.get("slots_per_point") == 1000,
    }
    if metadata is not None:
        result["row_protocol"] = dict(metadata)
    return result


def validate_sweep_payload(
    payload: Any,
    *,
    spearman_threshold: float = 0.8,
    value_tolerance: float = 0.20,
    recovery_gain_tolerance: float = 0.50,
) -> dict[str, Any]:
    """Return independent configuration, trend, and value judgements."""
    input_schema = "official_row_list" if isinstance(payload, list) else "canonical_object"
    if isinstance(payload, list):
        canonical, row_errors, row_metadata = _rows_to_payload(payload)
        config = _config_report(
            canonical, require_commit=False, initial_errors=row_errors, metadata=row_metadata,
        )
        payload = canonical
    else:
        config = _config_report(payload)
    report: dict[str, Any] = {
        "input_schema": input_schema,
        "target_provenance": PAPER_TARGETS["provenance"],
        "thresholds": {
            "spearman_min": spearman_threshold,
            "normalized_mae_max": value_tolerance,
            "ordering_match_rate_min": 0.8,
            "recovery_gain_relative_error_max": recovery_gain_tolerance,
        },
        "config": config,
        "sweeps": {},
        "recovery": {},
        "config_pass": bool(config["pass"]),
        "trend_pass": False,
        "value_pass": False,
        "overall_pass": False,
    }
    if not config["pass"]:
        return report

    curve_trends: list[bool] = []
    curve_values: list[bool] = []
    ordering_trends: list[bool] = []
    payload_sweeps = payload["sweeps"]
    for sweep_name, target in PAPER_TARGETS["sweeps"].items():
        observed = payload_sweeps[sweep_name]["algorithms"]
        algorithm_reports: dict[str, Any] = {}
        for algorithm in ALGORITHMS:
            stats = _curve_stats(target["algorithms"][algorithm], observed[algorithm])
            stats["trend_pass"] = stats["spearman"] >= spearman_threshold - 1e-12
            stats["value_pass"] = stats["normalized_mae"] <= value_tolerance + 1e-12
            algorithm_reports[algorithm] = stats
            curve_trends.append(stats["trend_pass"])
            curve_values.append(stats["value_pass"])

        order_points = []
        for index, x_value in enumerate(target["x"]):
            target_order = _ordering(target["algorithms"], index)
            observed_order = _ordering(observed, index)
            order_points.append({
                "x": x_value,
                "target": target_order,
                "observed": observed_order,
                "matches": target_order == observed_order,
            })
        match_rate = sum(point["matches"] for point in order_points) / len(order_points)
        order_pass = match_rate >= 0.8 - 1e-12
        ordering_trends.append(order_pass)

        semantic: dict[str, bool] = {}
        if sweep_name in ("p", "q", "m"):
            semantic["throughput_increases"] = all(
                spearman(target["x"], observed[algorithm]) >= spearman_threshold - 1e-12
                for algorithm in ALGORITHMS
            )
        elif sweep_name == "n":
            semantic["throughput_decreases"] = all(
                spearman(target["x"], observed[algorithm]) <= -spearman_threshold + 1e-12
                for algorithm in ALGORITHMS
            )
        elif sweep_name == "k":
            qcast = observed["QCAST"]
            semantic["k3_is_sufficient"] = qcast[1] >= 0.95 * max(qcast)
            semantic["unbounded_not_better_than_k6"] = qcast[-1] <= qcast[2] + 1e-12
        if sweep_name == "m":
            gap = [a - b for a, b in zip(observed["QCAST"], observed["QPASS_CR"])]
            semantic["qcast_advantage_expands"] = gap[-1] > gap[0]
        semantic_pass = all(semantic.values())
        curve_trends.append(semantic_pass)
        report["sweeps"][sweep_name] = {
            "x": target["x"],
            "algorithms": algorithm_reports,
            "ordering": {"points": order_points, "match_rate": match_rate, "pass": order_pass},
            "semantic_trends": semantic,
            "trend_pass": all(item["trend_pass"] for item in algorithm_reports.values()) and order_pass and semantic_pass,
            "value_pass": all(item["value_pass"] for item in algorithm_reports.values()),
        }

    recovery_trends: list[bool] = []
    recovery_values: list[bool] = []
    for algorithm, target in PAPER_TARGETS["recovery"].items():
        observed = payload["recovery"][algorithm]
        target_gain = target["on"] - target["off"]
        observed_gain = float(observed["on"]) - float(observed["off"])
        level_stats = _curve_stats([target["off"], target["on"]], [observed["off"], observed["on"]])
        gain_relative_error = abs(observed_gain - target_gain) / abs(target_gain)
        trend_ok = observed_gain > 0.0
        value_ok = (
            level_stats["normalized_mae"] <= value_tolerance + 1e-12
            and gain_relative_error <= recovery_gain_tolerance + 1e-12
        )
        recovery_trends.append(trend_ok)
        recovery_values.append(value_ok)
        report["recovery"][algorithm] = {
            "target": target,
            "observed": {"on": float(observed["on"]), "off": float(observed["off"])},
            "target_gain_eps": target_gain,
            "observed_gain_eps": observed_gain,
            "gain_absolute_error_eps": abs(observed_gain - target_gain),
            "gain_relative_error": gain_relative_error,
            "level_normalized_mae": level_stats["normalized_mae"],
            "trend_pass": trend_ok,
            "value_pass": value_ok,
        }

    report["trend_pass"] = all(curve_trends) and all(ordering_trends) and all(recovery_trends)
    report["value_pass"] = all(curve_values) and all(recovery_values)
    report["overall_pass"] = report["config_pass"] and report["trend_pass"] and report["value_pass"]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="official sweep JSON")
    parser.add_argument("--output", type=Path, default=None, help="write the full validation report")
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = validate_sweep_payload(payload)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("config_pass", "trend_pass", "value_pass", "overall_pass")}, indent=2))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
