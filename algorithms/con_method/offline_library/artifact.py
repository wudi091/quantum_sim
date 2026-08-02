"""Immutable fixed-grid artifact consumed by CON's online selector."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import TypeAlias

from qnet_core.contracts.complete_schedule import CompleteSchedule

from .models import LibraryScheduleTemplate, library_digest
from .topology_pool import Node, canonical_pair


SCHEMA_VERSION = "con-offline-library-v1"
PATH_SLOTS = 4
SCHEDULE_SLOTS = 4
CANDIDATE_SLOTS = PATH_SLOTS * SCHEDULE_SLOTS


@dataclass(frozen=True)
class CachedScheduleCandidate:
    """One executable path plus complete swap-group schedule."""

    pair_id: str
    path_id: str
    template_id: str
    schedule: CompleteSchedule

    def __post_init__(self) -> None:
        if not self.pair_id or not self.path_id or not self.template_id:
            raise ValueError("cached candidate IDs must be non-empty")

    @property
    def path(self) -> tuple[Node, ...]:
        return self.schedule.path  # type: ignore[return-value]

    @property
    def groups(self):
        return self.schedule.groups

    def oriented(self, source: Node, destination: Node) -> "CachedScheduleCandidate":
        if self.path[0] == source and self.path[-1] == destination:
            return self
        if self.path[0] == destination and self.path[-1] == source:
            reversed_path = tuple(reversed(self.path))
            return CachedScheduleCandidate(
                pair_id=self.pair_id,
                path_id=self.path_id,
                template_id=self.template_id,
                schedule=CompleteSchedule(
                    path=reversed_path,
                    groups=self.schedule.groups,
                ),
            )
        raise ValueError("cached candidate does not belong to the query pair")


CandidateSlot: TypeAlias = CachedScheduleCandidate | None


def _validate_slots(
    pair_id: str,
    endpoints: tuple[Node, Node],
    candidates: tuple[CandidateSlot, ...],
) -> None:
    if len(candidates) != CANDIDATE_SLOTS:
        raise ValueError("a pair artifact must contain exactly 16 slots")
    seen_templates: set[str] = set()
    encountered_empty_path_row = False
    for path_slot in range(PATH_SLOTS):
        row = candidates[
            path_slot * SCHEDULE_SLOTS:(path_slot + 1) * SCHEDULE_SLOTS
        ]
        nonempty = tuple(candidate for candidate in row if candidate is not None)
        if not nonempty:
            encountered_empty_path_row = True
            continue
        if encountered_empty_path_row:
            raise ValueError("valid path rows must precede padding rows")
        first_padding = next(
            (index for index, candidate in enumerate(row) if candidate is None),
            SCHEDULE_SLOTS,
        )
        if any(candidate is not None for candidate in row[first_padding:]):
            raise ValueError("valid schedules must precede padding in each row")
        path_ids = {candidate.path_id for candidate in nonempty}
        paths = {candidate.path for candidate in nonempty}
        if len(path_ids) != 1 or len(paths) != 1:
            raise ValueError("one grid row must represent exactly one path")
        path = next(iter(paths))
        if (path[0], path[-1]) != endpoints:
            raise ValueError("stored candidates must use canonical pair direction")
        for candidate in nonempty:
            if candidate.pair_id != pair_id:
                raise ValueError("candidate pair ID does not match its grid")
            if candidate.template_id in seen_templates:
                raise ValueError("candidate templates cannot be duplicated as padding")
            seen_templates.add(candidate.template_id)


@dataclass(frozen=True)
class StoredPairLibrary:
    pair_id: str
    endpoints: tuple[Node, Node]
    candidates: tuple[CandidateSlot, ...]

    def __post_init__(self) -> None:
        endpoints = tuple(self.endpoints)
        candidates = tuple(self.candidates)
        if not self.pair_id:
            raise ValueError("pair_id must be non-empty")
        if len(endpoints) != 2 or canonical_pair(*endpoints) != endpoints:
            raise ValueError("stored endpoints must be one canonical unordered pair")
        object.__setattr__(self, "endpoints", endpoints)
        object.__setattr__(self, "candidates", candidates)
        _validate_slots(self.pair_id, endpoints, candidates)

    @property
    def valid_mask(self) -> tuple[bool, ...]:
        return tuple(candidate is not None for candidate in self.candidates)


@dataclass(frozen=True)
class PairCandidateGrid:
    """An oriented, immutable 4-path by 4-schedule online view."""

    pair_id: str
    source: Node
    destination: Node
    candidates: tuple[CandidateSlot, ...]

    def __post_init__(self) -> None:
        if self.source == self.destination:
            raise ValueError("lookup endpoints must be different")
        if len(self.candidates) != CANDIDATE_SLOTS:
            raise ValueError("online candidate grids always have 16 slots")
        for candidate in self.candidates:
            if candidate is None:
                continue
            if candidate.path[0] != self.source or candidate.path[-1] != self.destination:
                raise ValueError("online candidate is not oriented to the query")

    @property
    def valid_mask(self) -> tuple[bool, ...]:
        return tuple(candidate is not None for candidate in self.candidates)

    @property
    def path_valid_mask(self) -> tuple[bool, ...]:
        return tuple(
            any(self.valid_mask[
                path_slot * SCHEDULE_SLOTS:(path_slot + 1) * SCHEDULE_SLOTS
            ])
            for path_slot in range(PATH_SLOTS)
        )

    @property
    def schedule_valid_mask(self) -> tuple[tuple[bool, ...], ...]:
        mask = self.valid_mask
        return tuple(
            mask[
                path_slot * SCHEDULE_SLOTS:(path_slot + 1) * SCHEDULE_SLOTS
            ]
            for path_slot in range(PATH_SLOTS)
        )

    @property
    def valid_candidates(self) -> tuple[CachedScheduleCandidate, ...]:
        return tuple(
            candidate for candidate in self.candidates if candidate is not None
        )

    def resolve(
        self, path_slot: int, schedule_slot: int
    ) -> CachedScheduleCandidate:
        if not 0 <= path_slot < PATH_SLOTS:
            raise IndexError("path slot is outside the 4x4 candidate grid")
        if not 0 <= schedule_slot < SCHEDULE_SLOTS:
            raise IndexError("schedule slot is outside the 4x4 candidate grid")
        candidate = self.candidates[
            path_slot * SCHEDULE_SLOTS + schedule_slot
        ]
        if candidate is None:
            raise ValueError("candidate slot is masked padding")
        return candidate


@dataclass(frozen=True)
class SolverCertificate:
    solver: str
    status: str
    objective: float
    mip_gap: float

    def __post_init__(self) -> None:
        if not self.solver or not self.status:
            raise ValueError("solver certificate labels must be non-empty")
        if self.mip_gap != 0.0:
            raise ValueError("CON artifacts require a zero-gap MILP certificate")


def compute_layout_digest(entries: tuple[StoredPairLibrary, ...]) -> str:
    payload = tuple(
        (
            entry.pair_id,
            tuple(map(repr, entry.endpoints)),
            tuple(
                None if candidate is None else (
                    candidate.path_id,
                    candidate.template_id,
                    tuple(map(repr, candidate.path)),
                    candidate.schedule.structural_key,
                )
                for candidate in entry.candidates
            ),
        )
        for entry in entries
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def compute_artifact_structural_digest(
    entries: tuple[StoredPairLibrary, ...],
) -> str:
    templates = tuple(
        LibraryScheduleTemplate(
            template_id=candidate.template_id,
            path_id=candidate.path_id,
            schedule=candidate.schedule,
            pair_id=candidate.pair_id,
        )
        for entry in entries
        for candidate in entry.candidates
        if candidate is not None
    )
    return library_digest(templates)


@dataclass(frozen=True)
class ConLibrary:
    """Topology-specific, read-only CON candidate cache."""

    topology_fingerprint: str
    pool_structural_digest: str
    library_structural_digest: str
    layout_digest: str
    compiler_fingerprint: str
    request_distribution_fingerprint: str
    physics_fingerprint: str
    training_scenario_ids: tuple[str, ...]
    training_trace_digests: tuple[str, ...]
    pair_entries: tuple[StoredPairLibrary, ...]
    solver_certificate: SolverCertificate
    selection_mode: str
    schema_version: str = SCHEMA_VERSION
    directionality: str = "undirected-symmetric-v1"
    paths_per_pair: int = PATH_SLOTS
    schedules_per_path: int = SCHEDULE_SLOTS

    def __post_init__(self) -> None:
        for name in (
            "topology_fingerprint",
            "pool_structural_digest",
            "library_structural_digest",
            "layout_digest",
            "compiler_fingerprint",
            "request_distribution_fingerprint",
            "physics_fingerprint",
            "selection_mode",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported CON artifact schema version")
        if self.paths_per_pair != PATH_SLOTS or self.schedules_per_path != SCHEDULE_SLOTS:
            raise ValueError("CON online artifacts require a fixed 4x4 grid")
        pair_ids = tuple(entry.pair_id for entry in self.pair_entries)
        endpoints = tuple(entry.endpoints for entry in self.pair_entries)
        if len(set(pair_ids)) != len(pair_ids) or len(set(endpoints)) != len(endpoints):
            raise ValueError("artifact pair entries must be unique")
        if len(set(self.training_scenario_ids)) != len(self.training_scenario_ids):
            raise ValueError("training scenario IDs must be unique")
        if len(set(self.training_trace_digests)) != len(self.training_trace_digests):
            raise ValueError("training trace digests must be unique")
        entries = tuple(self.pair_entries)
        object.__setattr__(self, "pair_entries", entries)
        if compute_layout_digest(entries) != self.layout_digest:
            raise ValueError("artifact layout digest does not match its 4x4 slots")
        if compute_artifact_structural_digest(entries) != self.library_structural_digest:
            raise ValueError("artifact structural digest does not match candidates")

    @cached_property
    def pair_by_endpoints(self) -> dict[tuple[Node, Node], StoredPairLibrary]:
        return {entry.endpoints: entry for entry in self.pair_entries}

    def lookup_grid(self, source: Node, destination: Node) -> PairCandidateGrid:
        endpoints = canonical_pair(source, destination)
        try:
            stored = self.pair_by_endpoints[endpoints]
        except KeyError as exc:
            raise KeyError("endpoint pair is not present in this topology cache") from exc
        oriented = tuple(
            None if candidate is None else candidate.oriented(source, destination)
            for candidate in stored.candidates
        )
        return PairCandidateGrid(
            pair_id=stored.pair_id,
            source=source,
            destination=destination,
            candidates=oriented,
        )

    def lookup(
        self, source: Node, destination: Node
    ) -> tuple[tuple[CandidateSlot, ...], tuple[bool, ...]]:
        """Return the fixed 16 slots and their immutable structural mask."""

        grid = self.lookup_grid(source, destination)
        return grid.candidates, grid.valid_mask

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "directionality": self.directionality,
            "paths_per_pair": self.paths_per_pair,
            "schedules_per_path": self.schedules_per_path,
            "topology_fingerprint": self.topology_fingerprint,
            "pool_structural_digest": self.pool_structural_digest,
            "library_structural_digest": self.library_structural_digest,
            "layout_digest": self.layout_digest,
            "compiler_fingerprint": self.compiler_fingerprint,
            "request_distribution_fingerprint": self.request_distribution_fingerprint,
            "physics_fingerprint": self.physics_fingerprint,
            "training_scenario_ids": list(self.training_scenario_ids),
            "training_trace_digests": list(self.training_trace_digests),
            "selection_mode": self.selection_mode,
            "solver_certificate": {
                "solver": self.solver_certificate.solver,
                "status": self.solver_certificate.status,
                "objective": self.solver_certificate.objective,
                "mip_gap": self.solver_certificate.mip_gap,
            },
            "pairs": [
                {
                    "pair_id": entry.pair_id,
                    "endpoints": list(entry.endpoints),
                    "valid_mask": list(entry.valid_mask),
                    "candidates": [
                        None if candidate is None else {
                            "pair_id": candidate.pair_id,
                            "path_id": candidate.path_id,
                            "template_id": candidate.template_id,
                            "path": list(candidate.path),
                            "groups": [list(group) for group in candidate.groups],
                        }
                        for candidate in entry.candidates
                    ],
                }
                for entry in self.pair_entries
            ],
        }

    def save(self, path: str | os.PathLike[str]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        try:
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        return target

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        expected_topology_fingerprint: str | None = None,
        expected_compiler_fingerprint: str | None = None,
    ) -> "ConLibrary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported CON artifact schema version")
        entries = []
        for pair_data in data["pairs"]:
            candidates = []
            for candidate_data in pair_data["candidates"]:
                if candidate_data is None:
                    candidates.append(None)
                    continue
                candidates.append(CachedScheduleCandidate(
                    pair_id=candidate_data["pair_id"],
                    path_id=candidate_data["path_id"],
                    template_id=candidate_data["template_id"],
                    schedule=CompleteSchedule(
                        path=tuple(candidate_data["path"]),
                        groups=tuple(
                            tuple(group) for group in candidate_data["groups"]
                        ),
                    ),
                ))
            entry = StoredPairLibrary(
                pair_id=pair_data["pair_id"],
                endpoints=tuple(pair_data["endpoints"]),
                candidates=tuple(candidates),
            )
            if tuple(pair_data["valid_mask"]) != entry.valid_mask:
                raise ValueError("serialized valid_mask disagrees with padding slots")
            entries.append(entry)
        certificate_data = data["solver_certificate"]
        library = cls(
            topology_fingerprint=data["topology_fingerprint"],
            pool_structural_digest=data["pool_structural_digest"],
            library_structural_digest=data["library_structural_digest"],
            layout_digest=data["layout_digest"],
            compiler_fingerprint=data["compiler_fingerprint"],
            request_distribution_fingerprint=(
                data["request_distribution_fingerprint"]
            ),
            physics_fingerprint=data["physics_fingerprint"],
            training_scenario_ids=tuple(data["training_scenario_ids"]),
            training_trace_digests=tuple(data["training_trace_digests"]),
            pair_entries=tuple(entries),
            solver_certificate=SolverCertificate(
                solver=certificate_data["solver"],
                status=certificate_data["status"],
                objective=float(certificate_data["objective"]),
                mip_gap=float(certificate_data["mip_gap"]),
            ),
            selection_mode=data["selection_mode"],
            schema_version=data["schema_version"],
            directionality=data["directionality"],
            paths_per_pair=int(data["paths_per_pair"]),
            schedules_per_path=int(data["schedules_per_path"]),
        )
        if (
            expected_topology_fingerprint is not None
            and library.topology_fingerprint != expected_topology_fingerprint
        ):
            raise ValueError("artifact topology fingerprint does not match")
        if (
            expected_compiler_fingerprint is not None
            and library.compiler_fingerprint != expected_compiler_fingerprint
        ):
            raise ValueError("artifact compiler fingerprint does not match")
        return library
