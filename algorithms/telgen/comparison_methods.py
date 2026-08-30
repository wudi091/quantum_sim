"""Frozen method sets used by the formal TELGEN experiments.

The fixed-construction variant is a construction-awareness ablation of the
same GNN policy.  It is evaluated in paired B5 reports rather than presented
as an unrelated routing baseline in the main comparison.
"""

from __future__ import annotations


FORMAL_METHOD_ORDER = ("gnn", "milp", "qpass", "greedy")
SCALABLE_METHOD_ORDER = ("gnn", "qpass", "greedy")
ROUTING_BASELINE_METHODS = ("qpass", "greedy")
CONSTRUCTION_ABLATION_METHODS = ("gnn",)
CONSTRUCTION_ABLATION_VARIANTS = ("adaptive", "fixed")

COMPARISON_PROFILES = {
    "formal": FORMAL_METHOD_ORDER,
    "scalable": SCALABLE_METHOD_ORDER,
    "construction_ablation": CONSTRUCTION_ABLATION_METHODS,
}


def methods_for_profile(profile: str) -> tuple[str, ...]:
    """Return the exact frozen method set for an experiment profile."""

    try:
        return COMPARISON_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown comparison profile: {profile!r}") from exc


def validate_profile_methods(profile: str, methods: set[str]) -> tuple[str, ...]:
    """Reject method drift and return the profile's canonical order."""

    expected = methods_for_profile(profile)
    if methods != set(expected):
        raise ValueError(
            f"comparison profile {profile!r} requires {list(expected)}, "
            f"got {sorted(methods)}"
        )
    return expected


def ordered_present_methods(methods: set[str]) -> tuple[str, ...]:
    """Return known formal methods in their frozen paper order."""

    unknown = methods.difference(FORMAL_METHOD_ORDER)
    if unknown:
        raise ValueError(f"unsupported formal comparison methods: {sorted(unknown)}")
    return tuple(method for method in FORMAL_METHOD_ORDER if method in methods)


__all__ = [
    "COMPARISON_PROFILES",
    "CONSTRUCTION_ABLATION_METHODS",
    "CONSTRUCTION_ABLATION_VARIANTS",
    "FORMAL_METHOD_ORDER",
    "ROUTING_BASELINE_METHODS",
    "SCALABLE_METHOD_ORDER",
    "methods_for_profile",
    "ordered_present_methods",
    "validate_profile_methods",
]
