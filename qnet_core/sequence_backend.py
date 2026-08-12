"""SeQUeNCe-backed physical resource kernel.

The routing layer only sees pair identifiers and immutable metadata.  All
physical behaviour lives in SeQUeNCe entities and protocols: memories, photon
loss, detector/BSM success, swapping, and memory expiration are driven by the
Timeline event queue.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import math
from typing import Iterable

from .physical_api import LaneExecutionResult, PhysicalCapabilities, PhysicalResource
from .command_api import ResourceClaim, SwapAction, SwapLane
from .spec import EpisodeSpec
from .sequence_protocol_arbiter import SequenceProtocolArbiter


@dataclass
class ResourcePair:
    pair_id: str
    left: int
    right: int
    left_memory: object
    right_memory: object
    fidelity: float
    born: int
    owner_request: str | None = None
    reserved_by: str | None = None
    lane: int | None = None

    @property
    def endpoints(self) -> tuple[int, int]:
        return self.left, self.right


@dataclass(frozen=True)
class PreparedGeneration:
    """Opaque handle for a batch generation attempt."""

    claim: ResourceClaim
    allocation_id: str
    pair_id: str
    context: tuple[object, ...] | None
    started_time_ps: int
    failure_cause: str = ""


@dataclass(frozen=True)
class PreparedSwap:
    """Opaque handle for a swap protocol started on the SeQUeNCe timeline."""

    action: SwapAction
    attempt_id: str
    left_pair_id: str
    right_pair_id: str
    left_outer: object
    right_outer: object
    left_outer_node: int
    right_outer_node: int
    middle_protocol: object
    end_left_protocol: object
    end_right_protocol: object
    started_time_ps: int


@dataclass(frozen=True)
class PreparedPurification:
    """Opaque handle for one two-pair BBPSSW attempt."""

    attempt_id: str
    keep_pair_id: str
    measure_pair_id: str
    left_protocol: object
    right_protocol: object
    kept_left_memory: object
    kept_right_memory: object
    measured_left_memory: object
    measured_right_memory: object
    kept_state: tuple[float, ...]
    measured_state: tuple[float, ...]
    kept_fidelities: tuple[float, float]
    measured_fidelities: tuple[float, float]
    started_time_ps: int


class SequenceBackend:
    """Common physical state shared by all planners."""

    # The construction executor now places launches through an explicit
    # resource/protocol scheduler.  These capabilities describe what that
    # scheduler is allowed to validate; subclasses can still opt out.
    # Multiple protocol families still share SeQUeNCe's global timeline and
    # Bell-diagonal state manager.  In SeQUeNCe 1.0.0, starting GEN and SWAP
    # in the same launch can race even on disjoint links, so the adapter keeps
    # same-epoch mixed-family packing conservative.  Independent SWAPs are
    # safe once reserved input pairs are excluded from ordinary pair-index
    # synchronization; the scheduler and protocol arbiter separately reject
    # shared BSM nodes, input segments, memories, and physical node scopes.
    supports_concurrent_swaps = True
    supports_mixed_operation_concurrency = False
    supports_inter_epoch_launch = True

    def __init__(self, spec: EpisodeSpec):
        self.spec = spec
        self.capabilities = PhysicalCapabilities(
            max_width=spec.physical.max_width,
            memory_capacity=spec.physical.memory_capacity,
            node_memory_capacity=spec.physical.node_memory_capacity,
        )
        self.protocol_arbiter = SequenceProtocolArbiter(
            supports_inter_epoch_launch=self.supports_inter_epoch_launch,
            supports_mixed_operation_concurrency=(
                self.supports_mixed_operation_concurrency
            ),
            supports_concurrent_swaps=self.supports_concurrent_swaps,
        )
        self.time = 0  # logical routing slots; SeQUeNCe uses ps internally
        self._counter = 0
        self.pairs: dict[str, ResourcePair] = {}
        self._protocol_owned_pair_ids: set[str] = set()
        self._claim_results: dict[
            tuple[int, str, tuple[int, int], int], str | None
        ] = {}
        self._topology_edges = {
            (min(u, v), max(u, v)) for u, v in self.spec.edges
        }
        self._hop_distances = self._build_hop_distances()
        self._sequence_ready = False
        self._build_sequence_world()

    def _build_hop_distances(self) -> dict[int, dict[int, int]]:
        adjacency = {node: set() for node in self.spec.nodes}
        for u, v in self.spec.edges:
            adjacency[u].add(v)
            adjacency[v].add(u)
        distances: dict[int, dict[int, int]] = {}
        for source in self.spec.nodes:
            rows = {source: 0}
            frontier = deque([source])
            while frontier:
                current = frontier.popleft()
                for neighbor in adjacency[current]:
                    if neighbor in rows:
                        continue
                    rows[neighbor] = rows[current] + 1
                    frontier.append(neighbor)
            distances[source] = rows
        return distances

    def _build_sequence_world(self) -> None:
        try:
            from sequence.components.optical_channel import (
                ClassicalChannel,
                QuantumChannel,
            )
            from sequence.entanglement_management.generation import (
                EntanglementGenerationA,
                EntanglementGenerationB,
            )
            from sequence.entanglement_management.swapping import (
                EntanglementSwappingA,
                EntanglementSwappingB,
            )
            from sequence.entanglement_management.purification import (
                BBPSSWProtocol,
            )
            from sequence.kernel.timeline import Timeline
            from sequence.resource_management.memory_manager import MemoryInfo
            from sequence.topology.node import BSMNode, QuantumRouter
        except ImportError as exc:  # pragma: no cover - deployment guard
            raise RuntimeError(
                "SeQUeNCe is required; install requirements.txt in Python 3.12+"
            ) from exc

        # Single-heralded generation and BDS swapping are the physical model
        # used by this adapter.  These are process-global SeQUeNCe factories.
        EntanglementGenerationA.set_global_type("single_heralded")
        EntanglementGenerationB.set_global_type("single_heralded")
        EntanglementSwappingA.set_formalism("bell_diagonal")
        EntanglementSwappingB.set_formalism("bell_diagonal")
        BBPSSWProtocol.set_formalism("bell_diagonal")

        self._MemoryInfo = MemoryInfo
        self._EntanglementGenerationA = EntanglementGenerationA
        self._EntanglementSwappingA = EntanglementSwappingA
        self._EntanglementSwappingB = EntanglementSwappingB
        self._BBPSSWProtocol = BBPSSWProtocol
        # SeQUeNCe 1.0.0 still draws protocol outcomes from Python's
        # process-global RNG.  Reset it before constructing the world so a
        # physical episode is reproducible and independent of the execution
        # order of other policy evaluations.
        self._timeline_seed = self._event_seed("timeline") % (1 << 32)
        Timeline.seed(self._timeline_seed)
        self.timeline = Timeline(stop_time=10 ** 23, formalism="bell_diagonal")

        physical = self.spec.physical
        coherence_time_s = physical.memory_lifetime * physical.slot_duration_ps * 1e-12
        memory_template = {
            "MemoryArray": {
                "frequency": physical.memory_frequency_hz,
                "coherence_time": coherence_time_s,
                "efficiency": math.sqrt(physical.generation_probability),
                "fidelity": physical.initial_fidelity,
                "wavelength": physical.memory_wavelength_nm,
            },
            "EntanglementSwapping": {
                "swapping_success_prob": physical.swap_probability,
                "swapping_degradation": physical.swap_degradation,
            },
        }

        self.nodes = {}
        self._memory_owner_by_name: dict[str, object] = {}
        for node in self.spec.nodes:
            degree = sum(node in edge for edge in self.spec.edges)
            memo_size = (
                physical.node_memory_capacity
                if physical.node_memory_capacity is not None
                else max(1, degree * physical.memory_capacity)
            )
            router = QuantumRouter(
                str(node),
                self.timeline,
                memo_size=memo_size,
                seed=self.spec.seed + int(node),
                component_templates=memory_template,
                gate_fid=physical.swap_degradation,
                meas_fid=1.0,
            )
            self.nodes[node] = router

        self._edge_bsm: dict[tuple[int, int], object] = {}
        self._edge_quantum_channels: dict[tuple[int, int], tuple[object, object]] = {}
        detector = {
            "efficiency": physical.detector_efficiency,
            "dark_count": 0,
            "time_resolution": 1,
            "count_rate": max(physical.memory_frequency_hz, 1e12),
        }
        for raw_u, raw_v in self.spec.edges:
            u, v = sorted((raw_u, raw_v))
            edge = (u, v)
            middle = BSMNode(
                f"bsm-{u}-{v}",
                self.timeline,
                [str(u), str(v)],
                seed=self._event_seed("bsm", u, v),
                component_templates={
                    "encoding_type": "single_heralded",
                    "SingleHeraldedBSM": {
                        "success_rate": physical.bsm_success_probability,
                        "detectors": [detector.copy(), detector.copy()],
                    },
                },
            )
            # BSMNode creates the protocol but sequence 1.0 does not register
            # it in ``protocols`` automatically.
            middle.protocols.append(middle.eg)
            self._edge_bsm[edge] = middle

            quantum_channels = []
            for endpoint in (u, v):
                channel = QuantumChannel(
                    f"qc-{endpoint}-{middle.name}",
                    self.timeline,
                    physical.quantum_attenuation_db_per_m,
                    physical.quantum_distance_m / 2,
                    physical.quantum_polarization_fidelity,
                    frequency=physical.memory_frequency_hz,
                )
                channel.set_ends(self.nodes[endpoint], middle.name)
                quantum_channels.append(channel)
            self._edge_quantum_channels[edge] = tuple(quantum_channels)

        # Entanglement generation and swapping communicate over classical
        # channels.  Use physical propagation distance unless an explicit
        # delay is configured.
        routers = list(self.nodes.values())
        for source in routers:
            for target in routers:
                if source is target:
                    continue
                source_id = int(source.name)
                target_id = int(target.name)
                hops = self._hop_distances[source_id].get(target_id)
                if hops is None:
                    continue
                delay = (
                    physical.classical_delay_ps * hops
                    if physical.classical_delay_ps > 0
                    else 0
                )
                channel = ClassicalChannel(
                    f"cc-{source.name}-{target.name}",
                    self.timeline,
                    physical.quantum_distance_m * hops,
                    delay,
                )
                channel.set_ends(source, target.name)

        for (u, v), middle in self._edge_bsm.items():
            for endpoint in (u, v):
                router = self.nodes[endpoint]
                for source, target in ((router, middle), (middle, router)):
                    channel = ClassicalChannel(
                        f"cc-{source.name}-{target.name}",
                        self.timeline,
                        physical.quantum_distance_m / 2,
                        physical.classical_delay_ps,
                    )
                    channel.set_ends(source, target.name)

        self.timeline.init()
        for router in self.nodes.values():
            for memory in router.components[router.memo_arr_name]:
                self._memory_owner_by_name[memory.name] = router
        self._physical_memory_current_usage = 0
        self._physical_memory_peak_usage = 0
        self._physical_memory_time_unit_ps = 0
        self._physical_memory_last_time_ps = self.physical_time_ps
        self._install_memory_telemetry()
        self._edge_generation_probabilities = {
            edge: self._effective_generation_probability(edge)
            for edge in self._topology_edges
        }
        self._sequence_ready = True

    def _new_id(self, prefix: str = "epr") -> str:
        value = f"{prefix}-{self._counter}"
        self._counter += 1
        return value

    @property
    def physical_time_ps(self) -> int:
        """Current SeQUeNCe timeline timestamp in picoseconds."""

        return int(self.timeline.now())

    def _flush_memory_telemetry(self) -> None:
        now = self.physical_time_ps
        if now < self._physical_memory_last_time_ps:
            raise RuntimeError("SeQUeNCe memory telemetry time moved backwards")
        self._physical_memory_time_unit_ps += (
            self._physical_memory_current_usage
            * (now - self._physical_memory_last_time_ps)
        )
        self._physical_memory_last_time_ps = now

    def _record_memory_state_transition(
        self,
        old_state: str,
        new_state: str,
    ) -> None:
        self._flush_memory_telemetry()
        old_occupied = str(old_state).upper() != "RAW"
        new_occupied = str(new_state).upper() != "RAW"
        self._physical_memory_current_usage += (
            int(new_occupied) - int(old_occupied)
        )
        if self._physical_memory_current_usage < 0:
            raise RuntimeError("physical memory usage became negative")
        self._physical_memory_peak_usage = max(
            self._physical_memory_peak_usage,
            self._physical_memory_current_usage,
        )

    def _install_memory_telemetry(self) -> None:
        """Observe every SeQUeNCe memory-manager state transition."""

        for router in self.nodes.values():
            manager = router.resource_manager.memory_manager
            original_update = manager.update

            def tracked_update(
                memory,
                state,
                *,
                _manager=manager,
                _original_update=original_update,
            ):
                old_state = str(
                    _manager.get_info_by_memory(memory).state
                )
                _original_update(memory, state)
                self._record_memory_state_transition(old_state, str(state))

            manager.update = tracked_update

    def _event_seed(self, *parts: object) -> int:
        payload = "|".join(map(str, (self.spec.seed, *parts))).encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    def _run_until(self, target_ps: int, *, advance_clock: bool = True) -> None:
        """Run all SeQUeNCe events through ``target_ps`` (inclusive)."""
        target_ps = max(int(target_ps), int(self.timeline.now()))
        old_stop = self.timeline.stop_time
        self.timeline.stop_time = target_ps + 1
        self.timeline.run()
        if advance_clock and self.timeline.now() < target_ps:
            self.timeline.time = target_ps
        self.timeline.stop_time = old_stop

    def _run_protocols(self, protocols: Iterable[object], window_ps: int) -> None:
        """Process events until the supplied protocols finish or time out."""
        tracked = tuple(protocols)
        deadline = self.timeline.now() + window_ps
        while any(protocol in protocol.owner.protocols for protocol in tracked):
            if self.timeline.events.isempty():
                break
            next_time = self.timeline.events.top().time
            if next_time > deadline:
                break
            self._run_until(next_time, advance_clock=False)

    def _generation_window_ps(self) -> int:
        delays = [
            channel.delay
            for router in self.nodes.values()
            for channel in router.qchannels.values()
        ]
        delays.extend(
            channel.delay
            for router in list(self.nodes.values()) + list(self._edge_bsm.values())
            for channel in router.cchannels.values()
        )
        maximum = max(delays, default=1)
        return max(10_000, 8 * maximum + 100)

    def _swap_window_ps(self) -> int:
        delays = [
            channel.delay
            for router in list(self.nodes.values()) + list(self._edge_bsm.values())
            for channel in router.cchannels.values()
        ]
        return max(1_000, 2 * max(delays, default=1) + 100)

    def _purification_window_ps(self) -> int:
        delays = [
            channel.delay
            for router in self.nodes.values()
            for channel in router.cchannels.values()
        ]
        return max(1_000, 2 * max(delays, default=1) + 100)

    @property
    def generation_duration_ps(self) -> int:
        return self._generation_window_ps()

    @property
    def swap_duration_ps(self) -> int:
        return self._swap_window_ps()

    @property
    def purification_duration_ps(self) -> int:
        return self._purification_window_ps()

    def advance_physical_to(self, target_ps: int, *, synchronize: bool = True) -> None:
        """Advance SeQUeNCe without changing the legacy logical slot clock."""

        self._run_until(target_ps)
        if synchronize:
            self._sync_pairs()

    def _memory_owner(self, memory: object) -> object:
        return self._memory_owner_by_name[memory.name]

    def _memory_info(self, memory: object) -> object:
        owner = self._memory_owner(memory)
        return owner.resource_manager.memory_manager.get_info_by_memory(memory)

    def _reserve_memory(self, memory: object) -> bool:
        info = self._memory_info(memory)
        if info.state != self._MemoryInfo.RAW:
            return False
        owner = self._memory_owner(memory)
        owner.resource_manager.memory_manager.update(
            memory, self._MemoryInfo.OCCUPIED
        )
        return True

    def _release_memory(self, memory: object) -> None:
        owner = self._memory_owner(memory)
        owner.resource_manager.update(None, memory, self._MemoryInfo.RAW)

    def _raw_memory(self, node: int) -> object | None:
        owner = self.nodes[node]
        for info in owner.resource_manager.memory_manager:
            if info.state == self._MemoryInfo.RAW:
                return info.memory
        return None

    @staticmethod
    def _is_entangled(memory: object, node: int, remote_memory: object) -> bool:
        entangled = memory.entangled_memory
        return (
            str(entangled.get("node_id")) == str(node)
            and entangled.get("memo_id") == remote_memory.name
            and memory.fidelity > 0
        )

    def _effective_generation_probability(self, edge: tuple[int, int]) -> float:
        physical = self.spec.physical
        transmission = math.prod(
            max(0.0, 1.0 - channel.loss)
            for channel in self._edge_quantum_channels[edge]
        )
        probability = (
            physical.generation_probability
            * transmission
            * physical.detector_efficiency ** 2
            * physical.bsm_success_probability
        )
        return min(max(float(probability), 0.0), 1.0)

    def _sync_pairs(self) -> None:
        """Mirror SeQUeNCe memory state into the routing index."""
        for pair_id, pair in list(self.pairs.items()):
            # An in-flight SWAP/PURIFY protocol owns its input pair.
            # SeQUeNCe legitimately rewrites or resets those memories before
            # the adapter finalizes the protocol.  Treating that transient
            # state as expiration would discard the pair and, for BDS swaps,
            # can remove a newly-created remote state belonging to another
            # concurrently prepared swap.  The protocol finalizer is the only
            # component allowed to settle reserved inputs.
            if pair_id in self._protocol_owned_pair_ids:
                continue
            if not self._is_entangled(pair.left_memory, pair.right, pair.right_memory):
                self.discard_pair(pair_id)
                continue
            if not self._is_entangled(pair.right_memory, pair.left, pair.left_memory):
                self.discard_pair(pair_id)
                continue
            pair.left_memory.bds_decohere()
            pair.right_memory.bds_decohere()
            pair.fidelity = min(
                pair.left_memory.get_bds_fidelity(),
                pair.right_memory.get_bds_fidelity(),
            )

    def synchronize(self) -> None:
        """Refresh the routing index after Timeline-driven physical events."""
        self._sync_pairs()

    @staticmethod
    def _resource_view(pair: ResourcePair) -> PhysicalResource:
        return PhysicalResource(
            pair_id=pair.pair_id,
            left=pair.left,
            right=pair.right,
            fidelity=pair.fidelity,
            born=pair.born,
            owner_request=pair.owner_request,
            reserved_by=pair.reserved_by,
            lane=pair.lane,
        )

    def resources(self) -> tuple[PhysicalResource, ...]:
        self._sync_pairs()
        return tuple(
            self._resource_view(pair)
            for pair in sorted(self.pairs.values(), key=lambda item: item.pair_id)
        )

    def resource(self, pair_id: str) -> PhysicalResource | None:
        self._sync_pairs()
        pair = self.pairs.get(pair_id)
        return None if pair is None else self._resource_view(pair)

    def resource_without_sync(self, pair_id: str) -> PhysicalResource | None:
        """Refresh and read one pair without running the SeQUeNCe timeline.

        This is used while another protocol owns different memories.  The
        local BDS update makes fidelity/expiration observations consistent at
        the current physical timestamp without synchronizing swap inputs.
        """

        pair = self.pairs.get(pair_id)
        if pair is None:
            return None
        if pair_id in self._protocol_owned_pair_ids:
            return self._resource_view(pair)
        if not (
            self._is_entangled(pair.left_memory, pair.right, pair.right_memory)
            and self._is_entangled(pair.right_memory, pair.left, pair.left_memory)
        ):
            return None
        pair.left_memory.bds_decohere()
        pair.right_memory.bds_decohere()
        pair.fidelity = min(
            pair.left_memory.get_bds_fidelity(),
            pair.right_memory.get_bds_fidelity(),
        )
        if not (
            self._is_entangled(pair.left_memory, pair.right, pair.right_memory)
            and self._is_entangled(pair.right_memory, pair.left, pair.left_memory)
        ):
            return None
        return self._resource_view(pair)

    def resource_ids(self) -> frozenset[str]:
        self._sync_pairs()
        return frozenset(self.pairs)

    def resource_count(self) -> int:
        self._sync_pairs()
        return len(self.pairs)

    def has_resource(self, pair_id: str) -> bool:
        self._sync_pairs()
        return pair_id in self.pairs

    def edge_occupancy(self, u: int, v: int) -> int:
        return self._edge_occupancy(u, v)

    def assign_owner(self, pair_id: str, request_id: str) -> None:
        pair = self.pairs[pair_id]
        pair.owner_request = request_id
        pair.reserved_by = None

    def validate_claim_batch(self, claims: Iterable[ResourceClaim]) -> None:
        self._sync_pairs()
        claim_list = tuple(claims)
        edge_claims: dict[tuple[int, int], int] = {}
        node_claims: dict[int, int] = {}
        for claim in claim_list:
            if claim.endpoints not in self._topology_edges:
                raise ValueError(
                    f"claim references non-topology edge {claim.endpoints}"
                )
            edge_claims[claim.endpoints] = edge_claims.get(claim.endpoints, 0) + 1
            for node in claim.endpoints:
                node_claims[node] = node_claims.get(node, 0) + 1
        for edge, count in edge_claims.items():
            if self._edge_occupancy(*edge) + count > self.capabilities.memory_capacity:
                raise ValueError(f"edge {edge} memory capacity exceeded")
        capacity = self.capabilities.node_memory_capacity
        if capacity is not None:
            for node, count in node_claims.items():
                if self.node_occupancy(node) + count > capacity:
                    raise ValueError(f"node {node} memory capacity exceeded")

    def can_allocate_claims(self, claims: Iterable[ResourceClaim]) -> bool:
        try:
            self.validate_claim_batch(claims)
        except ValueError:
            return False
        return True

    def estimate_route_throughput(
        self,
        route_nodes: tuple[int, ...],
        width: int,
    ) -> float:
        edges = [
            (min(u, v), max(u, v))
            for u, v in zip(route_nodes, route_nodes[1:])
        ]
        if not edges:
            return 0.0
        expected_bottleneck = 0.0
        for minimum in range(1, width + 1):
            all_links_tail = 1.0
            for edge in edges:
                probability = self._edge_generation_probabilities.get(edge, 0.0)
                tail = sum(
                    math.comb(width, successes)
                    * probability ** successes
                    * (1.0 - probability) ** (width - successes)
                    for successes in range(minimum, width + 1)
                )
                all_links_tail *= tail
            expected_bottleneck += all_links_tail
        swaps = max(len(route_nodes) - 2, 0)
        return float(
            expected_bottleneck
            * self.spec.physical.swap_probability ** swaps
        )

    def estimate_swap_throughput(self, swap_counts: Iterable[int]) -> float:
        probability = self.spec.physical.swap_probability
        return float(sum(probability ** count for count in swap_counts))

    def link_capacities(self) -> tuple[dict[str, object], ...]:
        return tuple({
            "left": min(u, v),
            "right": max(u, v),
            "max_width": self.capabilities.max_width,
            "generation_probability": self._edge_generation_probabilities[
                (min(u, v), max(u, v))
            ],
        } for u, v in self.spec.edges)

    def construction_state(self) -> tuple[tuple[str, object], ...]:
        """Return a side-effect-free neutral summary for construction DTOs."""

        self._flush_memory_telemetry()

        node_memory = []
        for node, router in sorted(self.nodes.items()):
            counts: dict[str, int] = {}
            for info in router.resource_manager.memory_manager:
                state = str(info.state)
                counts[state] = counts.get(state, 0) + 1
            node_memory.append((node, tuple(sorted(counts.items()))))

        pair_reservations = []
        expirations = []
        link_occupancy: dict[tuple[int, int], int] = {}
        for pair in sorted(self.pairs.values(), key=lambda item: item.pair_id):
            edge = tuple(sorted(pair.endpoints))
            if edge in self._topology_edges:
                link_occupancy[edge] = link_occupancy.get(edge, 0) + 1
            raw_expiration = min(
                pair.left_memory.get_expire_time(),
                pair.right_memory.get_expire_time(),
            )
            expiration = (
                int(raw_expiration) if math.isfinite(raw_expiration) else -1
            )
            pair_reservations.append((
                pair.pair_id,
                pair.left,
                pair.right,
                pair.reserved_by or "",
                -1 if pair.lane is None else pair.lane,
                float(pair.fidelity),
            ))
            expirations.append((pair.pair_id, expiration))

        event_times = tuple(sorted(
            int(event.time) for event in self.timeline.events.data
        ))
        return (
            ("episode_seed", int(self.spec.seed)),
            ("physical_rng_seed", int(self._timeline_seed)),
            ("expiration_events", tuple(expirations)),
            ("link_occupancy", tuple(
                (u, v, amount)
                for (u, v), amount in sorted(link_occupancy.items())
            )),
            ("node_memory", tuple(node_memory)),
            (
                "physical_memory_usage",
                int(self._physical_memory_current_usage),
            ),
            (
                "peak_physical_memory_usage",
                int(self._physical_memory_peak_usage),
            ),
            (
                "physical_memory_time_unit_ps",
                int(self._physical_memory_time_unit_ps),
            ),
            ("pair_reservations", tuple(pair_reservations)),
            ("physical_formalism", "bell_diagonal"),
            ("stochastic_model", "seeded_conditionally_independent_protocol_outcomes"),
            ("supports_concurrent_swaps", bool(self.supports_concurrent_swaps)),
            (
                "supports_mixed_operation_concurrency",
                bool(self.supports_mixed_operation_concurrency),
            ),
            ("protocol_arbiter", self.protocol_arbiter.state()),
            ("timeline_next_event_ps", event_times[0] if event_times else -1),
            ("timeline_pending_event_count", len(event_times)),
        )

    def fidelity_hop_bound(
        self, required_fidelity: float, max_storage_slots: int
    ) -> int:
        """Return the paper's conservative Werner-state hop bound.

        The planner receives only this integer.  Initial fidelity and memory
        coherence remain properties of the SeQUeNCe-backed physical adapter.
        ``max_storage_slots`` is the application policy for one distribution
        attempt (the paper uses ``T / t``).
        """
        if not 0.5 <= required_fidelity <= 1.0:
            raise ValueError("required_fidelity must be in [0.5, 1]")
        if max_storage_slots < 1:
            raise ValueError("max_storage_slots must be positive")
        physical = self.spec.physical
        w0 = (4.0 * physical.initial_fidelity - 1.0) / 3.0
        wk = (4.0 * required_fidelity - 1.0) / 3.0
        coherence_slots = max(float(physical.memory_lifetime), 1.0)
        alpha = math.exp(-float(max_storage_slots) / coherence_slots)
        numerator = alpha * wk
        denominator = alpha * w0
        if numerator >= 1.0:
            return max(1, len(self._hop_distances))
        if denominator <= 0.0 or denominator >= 1.0:
            return 1
        bound = math.floor(math.log(numerator) / math.log(denominator))
        return max(1, int(bound))

    def discard_pair(self, pair_id: str) -> PhysicalResource | None:
        """Remove a pair and release both SeQUeNCe memories."""
        if pair_id in self._protocol_owned_pair_ids:
            raise RuntimeError(
                "cannot discard a pair owned by an in-flight physical protocol"
            )
        pair = self.pairs.pop(pair_id, None)
        if pair is None:
            return None
        view = self._resource_view(pair)
        self._release_memory(pair.left_memory)
        self._release_memory(pair.right_memory)
        return view

    def node_occupancy(self, node: int) -> int:
        self._sync_pairs()
        if node not in self.nodes:
            raise ValueError(f"unknown node {node}")
        return sum(node in pair.endpoints for pair in self.pairs.values())

    def node_free_slots(self, node: int) -> int | None:
        capacity = self.spec.physical.node_memory_capacity
        if capacity is None:
            if node not in self.nodes:
                raise ValueError(f"unknown node {node}")
            return None
        return max(0, capacity - self.node_occupancy(node))

    def _edge_occupancy(self, u: int, v: int) -> int:
        self._sync_pairs()
        edge = {u, v}
        return sum(set(pair.endpoints) == edge for pair in self.pairs.values())

    def _prepare_generation(
        self,
        u: int,
        v: int,
        lane: int,
        pair_id: str,
    ) -> tuple[object, ...] | None:
        left_memory = self._raw_memory(u)
        right_memory = self._raw_memory(v)
        if left_memory is None or right_memory is None:
            return None
        left_reserved = False
        right_reserved = False
        protocols: list[object] = []
        try:
            left_reserved = self._reserve_memory(left_memory)
            right_reserved = self._reserve_memory(right_memory)
            if not left_reserved or not right_reserved:
                if left_reserved:
                    self._release_memory(left_memory)
                if right_reserved:
                    self._release_memory(right_memory)
                return None

            middle = self._edge_bsm[(min(u, v), max(u, v))]
            left = self.nodes[u]
            right = self.nodes[v]
            left_protocol = self._EntanglementGenerationA.create(
                left, f"{pair_id}-a", middle.name, str(v), left_memory
            )
            right_protocol = self._EntanglementGenerationA.create(
                right, f"{pair_id}-b", middle.name, str(u), right_memory
            )
            left.protocols.append(left_protocol)
            protocols.append(left_protocol)
            right.protocols.append(right_protocol)
            protocols.append(right_protocol)
            left_protocol.set_others(right_protocol.name, right.name, [right_memory.name])
            right_protocol.set_others(left_protocol.name, left.name, [left_memory.name])
            left_protocol.start()
            right_protocol.start()
            return (
                u, v, lane, pair_id, left_memory, right_memory,
                left_protocol, right_protocol,
            )
        except Exception:
            for protocol in reversed(protocols):
                self._cancel_protocol(protocol)
            if left_reserved:
                self._release_memory(left_memory)
            if right_reserved:
                self._release_memory(right_memory)
            raise

    def _cancel_protocol(self, protocol: object) -> None:
        if protocol in protocol.owner.protocols:
            protocol.owner.protocols.remove(protocol)
        for event in tuple(getattr(protocol, "scheduled_events", ())):
            if event.time >= self.timeline.now():
                self.timeline.remove_event(event)

    def cancel_generation(self, prepared: Iterable[PreparedGeneration]) -> None:
        """Cancel generation handles before their timeline epoch is run."""

        for item in tuple(prepared):
            if item.context is None:
                continue
            if self.physical_time_ps != item.started_time_ps:
                raise RuntimeError("cannot cancel generation after physical time advanced")
            (
                _u, _v, _lane, pair_id, left_memory, right_memory,
                left_protocol, right_protocol,
            ) = item.context
            self.pairs.pop(pair_id, None)
            self._cancel_protocol(left_protocol)
            self._cancel_protocol(right_protocol)
            self._release_memory(left_memory)
            self._release_memory(right_memory)

    def _finalize_generation(self, context: tuple[object, ...]) -> str | None:
        (
            u, v, lane, pair_id, left_memory, right_memory,
            left_protocol, right_protocol,
        ) = context
        for protocol in (left_protocol, right_protocol):
            if protocol in protocol.owner.protocols:
                protocol.owner.protocols.remove(protocol)
            for event in protocol.scheduled_events:
                if event.time >= self.timeline.now():
                    self.timeline.remove_event(event)
        success = (
            self._memory_info(left_memory).state == self._MemoryInfo.ENTANGLED
            and self._memory_info(right_memory).state == self._MemoryInfo.ENTANGLED
            and self._is_entangled(left_memory, v, right_memory)
            and self._is_entangled(right_memory, u, left_memory)
        )
        if not success:
            self._release_memory(left_memory)
            self._release_memory(right_memory)
            return None
        self.pairs[pair_id] = ResourcePair(
            pair_id,
            u,
            v,
            left_memory,
            right_memory,
            min(left_memory.fidelity, right_memory.fidelity),
            self.physical_time_ps,
            lane=lane,
        )
        return pair_id

    def begin_generation(
        self,
        claims: Iterable[ResourceClaim],
        allocation_id: str,
    ) -> tuple[PreparedGeneration, ...]:
        """Start all generation protocols in one physical-time epoch.

        Unlike ``generate_claimed_pairs`` this method does not run the
        timeline or finalize a pair.  The caller can start several batches,
        advance the timeline once, and then call ``finish_generation`` for
        event-level aggregation.
        """

        if not allocation_id:
            raise ValueError("allocation_id must be non-empty")
        claim_list = tuple(claims)
        if len(set(claim_list)) != len(claim_list):
            raise ValueError("duplicate resource claim")
        for claim in claim_list:
            if claim.endpoints not in self._topology_edges:
                raise ValueError(
                    f"claim references non-topology edge {claim.endpoints}"
                )
            if claim.lane >= self.spec.physical.max_width:
                raise ValueError("claim lane exceeds physical max_width")
        self._sync_pairs()
        pending_nodes: dict[int, int] = {}
        pending_edges: dict[tuple[int, int], int] = {}
        prepared: list[PreparedGeneration] = []
        try:
            for claim in sorted(claim_list, key=lambda item: (*item.endpoints, item.lane)):
                u, v = claim.endpoints
                edge = (u, v)
                left_free = self.node_free_slots(u)
                right_free = self.node_free_slots(v)
                rejected = (
                    self._edge_occupancy(u, v) + pending_edges.get(edge, 0)
                    >= self.spec.physical.memory_capacity
                    or (left_free is not None and pending_nodes.get(u, 0) >= left_free)
                    or (right_free is not None and pending_nodes.get(v, 0) >= right_free)
                )
                digest = hashlib.sha256(
                    f"{self.physical_time_ps}|{allocation_id}|{u}|{v}|{claim.lane}".encode()
                ).hexdigest()[:16]
                pair_id = f"event-epr-{digest}"
                if rejected:
                    prepared.append(PreparedGeneration(
                        claim,
                        allocation_id,
                        pair_id,
                        None,
                        self.physical_time_ps,
                        "physical_backend_rejection",
                    ))
                    continue
                context = self._prepare_generation(u, v, claim.lane, pair_id)
                if context is None:
                    prepared.append(PreparedGeneration(
                        claim,
                        allocation_id,
                        pair_id,
                        None,
                        self.physical_time_ps,
                        "physical_backend_rejection",
                    ))
                    continue
                pending_edges[edge] = pending_edges.get(edge, 0) + 1
                pending_nodes[u] = pending_nodes.get(u, 0) + 1
                pending_nodes[v] = pending_nodes.get(v, 0) + 1
                prepared.append(PreparedGeneration(
                    claim, allocation_id, pair_id, context, self.physical_time_ps
                ))
        except Exception:
            self.cancel_generation(prepared)
            raise
        return tuple(prepared)

    def finish_generation(self, prepared: Iterable[PreparedGeneration]) -> dict[ResourceClaim, str | None]:
        """Finalize prepared generation protocols after timeline advancement."""

        outcomes: dict[ResourceClaim, str | None] = {}
        for item in prepared:
            if item.context is None:
                outcomes[item.claim] = None
                continue
            pair_id = self._finalize_generation(item.context)
            if pair_id is not None:
                self.pairs[pair_id].reserved_by = item.allocation_id
            outcomes[item.claim] = pair_id
        self._sync_pairs()
        return outcomes

    def run_prepared_protocols(
        self,
        generations: Iterable[PreparedGeneration] = (),
        swaps: Iterable[PreparedSwap] = (),
        purifications: Iterable[PreparedPurification] = (),
        deadline_ps: int | None = None,
    ) -> None:
        """Run all supplied protocol instances until they terminate."""

        generation_list = tuple(generations)
        swap_list = tuple(swaps)
        purification_list = tuple(purifications)
        protocols: list[object] = []
        for item in generation_list:
            if item.context is not None:
                protocols.extend(item.context[-2:])
        for item in swap_list:
            protocols.extend((item.end_left_protocol, item.middle_protocol, item.end_right_protocol))
        for item in purification_list:
            protocols.extend((item.left_protocol, item.right_protocol))
        if protocols:
            windows = []
            if generation_list:
                windows.append(self._generation_window_ps())
            if swap_list:
                windows.append(self._swap_window_ps())
            if purification_list:
                windows.append(self._purification_window_ps())
            window = max(windows)
            if deadline_ps is not None:
                window = min(window, max(0, int(deadline_ps) - self.physical_time_ps))
            self._run_protocols(protocols, window)

    def prepared_complete(
        self,
        generations: Iterable[PreparedGeneration] = (),
        swaps: Iterable[PreparedSwap] = (),
        purifications: Iterable[PreparedPurification] = (),
    ) -> bool:
        """Report whether all supplied physical protocols have terminated."""

        protocols: list[object] = []
        for item in generations:
            if item.context is not None:
                protocols.extend(item.context[-2:])
        for item in swaps:
            protocols.extend((item.end_left_protocol, item.middle_protocol, item.end_right_protocol))
        for item in purifications:
            protocols.extend((item.left_protocol, item.right_protocol))
        return all(protocol not in protocol.owner.protocols for protocol in protocols)

    def _generate_batch(
        self,
        attempts: Iterable[tuple[int, int, int, str]],
    ) -> dict[str, str | None]:
        contexts: list[tuple[object, ...]] = []
        outcomes: dict[str, str | None] = {}
        for u, v, lane, pair_id in attempts:
            context = self._prepare_generation(u, v, lane, pair_id)
            if context is None:
                outcomes[pair_id] = None
            else:
                contexts.append(context)
        if contexts:
            protocols = [
                protocol
                for context in contexts
                for protocol in context[-2:]
            ]
            self._run_protocols(protocols, self._generation_window_ps())
            for context in contexts:
                requested_id = context[3]
                outcomes[requested_id] = self._finalize_generation(context)
        return outcomes

    def generate_claimed_pairs(
        self,
        claims: Iterable[ResourceClaim],
        allocation_id: str,
    ) -> dict[ResourceClaim, str | None]:
        """Run one SeQUeNCe generation attempt for every explicit claim."""
        if not allocation_id:
            raise ValueError("allocation_id must be non-empty")
        self._sync_pairs()
        claim_list = tuple(claims)
        if len(set(claim_list)) != len(claim_list):
            raise ValueError("duplicate resource claim")
        for claim in claim_list:
            if claim.endpoints not in self._topology_edges:
                raise ValueError(f"claim references non-topology edge {claim.endpoints}")
            if claim.lane >= self.spec.physical.max_width:
                raise ValueError("claim lane exceeds physical max_width")

        results: dict[ResourceClaim, str | None] = {}
        pending: list[tuple[int, int, int, str]] = []
        pending_claims: dict[str, ResourceClaim] = {}
        pending_edges: dict[tuple[int, int], int] = {}
        pending_nodes: dict[int, int] = {}
        for claim in sorted(claim_list, key=lambda item: (*item.endpoints, item.lane)):
            u, v = claim.endpoints
            attempt_key = (self.time, allocation_id, claim.endpoints, claim.lane)
            if attempt_key in self._claim_results:
                pair_id = self._claim_results[attempt_key]
                results[claim] = pair_id if pair_id in self.pairs else None
                continue
            edge = (u, v)
            left_free = self.node_free_slots(u)
            right_free = self.node_free_slots(v)
            if (
                self._edge_occupancy(u, v) + pending_edges.get(edge, 0)
                >= self.spec.physical.memory_capacity
                or (left_free is not None and pending_nodes.get(u, 0) >= left_free)
                or (right_free is not None and pending_nodes.get(v, 0) >= right_free)
            ):
                self._claim_results[attempt_key] = None
                results[claim] = None
            else:
                digest = hashlib.sha256(
                    f"{self.time}|{allocation_id}|{u}|{v}|{claim.lane}".encode()
                ).hexdigest()[:16]
                requested_id = f"claim-{digest}"
                pending.append((u, v, claim.lane, requested_id))
                pending_claims[requested_id] = claim
                pending_edges[edge] = pending_edges.get(edge, 0) + 1
                pending_nodes[u] = pending_nodes.get(u, 0) + 1
                pending_nodes[v] = pending_nodes.get(v, 0) + 1

        outcomes = self._generate_batch(pending)
        for requested_id, claim in pending_claims.items():
            pair_id = outcomes[requested_id]
            if pair_id is not None:
                self.pairs[pair_id].reserved_by = allocation_id
            attempt_key = (self.time, allocation_id, claim.endpoints, claim.lane)
            self._claim_results[attempt_key] = pair_id
            results[claim] = pair_id
        return results

    def generate_elementary_pairs(self) -> tuple[str, ...]:
        """Attempt one physical generation round on every free topology edge."""
        self._sync_pairs()
        attempts: list[tuple[int, int, int, str]] = []
        pending_nodes: dict[int, int] = {}
        for edge_index, (raw_u, raw_v) in enumerate(self.spec.edges):
            u, v = sorted((raw_u, raw_v))
            left_free = self.node_free_slots(u)
            right_free = self.node_free_slots(v)
            if (
                self._edge_occupancy(u, v) >= self.spec.physical.memory_capacity
                or (left_free is not None and pending_nodes.get(u, 0) >= left_free)
                or (right_free is not None and pending_nodes.get(v, 0) >= right_free)
            ):
                continue
            pair_id = f"epr-{self.time}-{edge_index}-{self._counter}"
            self._counter += 1
            attempts.append((u, v, edge_index, pair_id))
            pending_nodes[u] = pending_nodes.get(u, 0) + 1
            pending_nodes[v] = pending_nodes.get(v, 0) + 1
        outcomes = self._generate_batch(attempts)
        return tuple(
            pair_id for *_, pair_id in attempts
            if outcomes[pair_id] is not None
        )

    def can_begin_purification(
        self,
        keep_pair_id: str,
        measure_pair_id: str,
    ) -> bool:
        """Read-only preflight for a two-pair BBPSSW reservation."""

        if keep_pair_id == measure_pair_id:
            return False
        self._sync_pairs()
        keep = self.pairs.get(keep_pair_id)
        measure = self.pairs.get(measure_pair_id)
        if (
            keep is None
            or measure is None
            or keep.endpoints != measure.endpoints
            or keep.reserved_by is not None
            or measure.reserved_by is not None
            or min(keep.fidelity, measure.fidelity) <= 0.5
        ):
            return False
        left, right = keep.endpoints
        return (
            self._is_entangled(keep.left_memory, right, keep.right_memory)
            and self._is_entangled(keep.right_memory, left, keep.left_memory)
            and self._is_entangled(measure.left_memory, right, measure.right_memory)
            and self._is_entangled(measure.right_memory, left, measure.left_memory)
        )

    def begin_purification(
        self,
        keep_pair_id: str,
        measure_pair_id: str,
        attempt_id: str,
    ) -> PreparedPurification | None:
        """Start one native SeQUeNCe BBPSSW protocol pair."""

        if not attempt_id:
            raise ValueError("attempt_id must be non-empty")
        if not self.can_begin_purification(keep_pair_id, measure_pair_id):
            return None
        keep = self.pairs.get(keep_pair_id)
        measure = self.pairs.get(measure_pair_id)
        assert keep is not None and measure is not None
        left_node_id, right_node_id = keep.endpoints

        quantum_manager = self.timeline.quantum_manager
        kept_state = tuple(float(value) for value in quantum_manager.get(
            keep.left_memory.qstate_key
        ).state)
        measured_state = tuple(float(value) for value in quantum_manager.get(
            measure.left_memory.qstate_key
        ).state)
        keep.reserved_by = attempt_id
        measure.reserved_by = attempt_id
        self._protocol_owned_pair_ids.update((keep_pair_id, measure_pair_id))
        protocols: list[object] = []
        try:
            left_node = self.nodes[left_node_id]
            right_node = self.nodes[right_node_id]
            left_node.set_seed(self._event_seed(
                "purify", self.physical_time_ps, attempt_id, left_node_id
            ))
            right_node.set_seed(self._event_seed(
                "purify", self.physical_time_ps, attempt_id, right_node_id
            ))
            suffix = f"purify-{self._counter}"
            left_protocol = self._BBPSSWProtocol.create(
                left_node,
                f"{suffix}-l",
                keep.left_memory,
                measure.left_memory,
                is_twirled=True,
            )
            right_protocol = self._BBPSSWProtocol.create(
                right_node,
                f"{suffix}-r",
                keep.right_memory,
                measure.right_memory,
                is_twirled=True,
            )
            left_node.protocols.append(left_protocol)
            protocols.append(left_protocol)
            right_node.protocols.append(right_protocol)
            protocols.append(right_protocol)
            left_protocol.set_others(
                right_protocol.name,
                right_node.name,
                [keep.right_memory.name, measure.right_memory.name],
            )
            right_protocol.set_others(
                left_protocol.name,
                left_node.name,
                [keep.left_memory.name, measure.left_memory.name],
            )
            left_protocol.start()
            right_protocol.start()
            self._counter += 1
            return PreparedPurification(
                attempt_id=attempt_id,
                keep_pair_id=keep_pair_id,
                measure_pair_id=measure_pair_id,
                left_protocol=left_protocol,
                right_protocol=right_protocol,
                kept_left_memory=keep.left_memory,
                kept_right_memory=keep.right_memory,
                measured_left_memory=measure.left_memory,
                measured_right_memory=measure.right_memory,
                kept_state=kept_state,
                measured_state=measured_state,
                kept_fidelities=(
                    float(keep.left_memory.fidelity),
                    float(keep.right_memory.fidelity),
                ),
                measured_fidelities=(
                    float(measure.left_memory.fidelity),
                    float(measure.right_memory.fidelity),
                ),
                started_time_ps=self.physical_time_ps,
            )
        except Exception:
            for protocol in reversed(protocols):
                self._cancel_protocol(protocol)
            quantum_manager.set(
                [keep.left_memory.qstate_key, keep.right_memory.qstate_key],
                list(kept_state),
            )
            quantum_manager.set(
                [measure.left_memory.qstate_key, measure.right_memory.qstate_key],
                list(measured_state),
            )
            keep.reserved_by = None
            measure.reserved_by = None
            self._protocol_owned_pair_ids.difference_update(
                (keep_pair_id, measure_pair_id)
            )
            raise

    def cancel_purification(self, prepared: PreparedPurification) -> None:
        """Roll back a purification handle before physical time advances."""

        if self.physical_time_ps != prepared.started_time_ps:
            raise RuntimeError("cannot cancel purification after physical time advanced")
        for protocol in (prepared.left_protocol, prepared.right_protocol):
            self._cancel_protocol(protocol)
        quantum_manager = self.timeline.quantum_manager
        quantum_manager.set(
            [
                prepared.kept_left_memory.qstate_key,
                prepared.kept_right_memory.qstate_key,
            ],
            list(prepared.kept_state),
        )
        quantum_manager.set(
            [
                prepared.measured_left_memory.qstate_key,
                prepared.measured_right_memory.qstate_key,
            ],
            list(prepared.measured_state),
        )
        prepared.kept_left_memory.fidelity = prepared.kept_fidelities[0]
        prepared.kept_right_memory.fidelity = prepared.kept_fidelities[1]
        prepared.measured_left_memory.fidelity = prepared.measured_fidelities[0]
        prepared.measured_right_memory.fidelity = prepared.measured_fidelities[1]
        for pair_id in (prepared.keep_pair_id, prepared.measure_pair_id):
            pair = self.pairs.get(pair_id)
            if pair is not None and pair.reserved_by == prepared.attempt_id:
                pair.reserved_by = None
        self._protocol_owned_pair_ids.difference_update(
            (prepared.keep_pair_id, prepared.measure_pair_id)
        )

    def finish_purification(
        self,
        prepared: PreparedPurification,
    ) -> str | None:
        """Finalize BBPSSW, retaining only the kept pair on success."""

        complete = self.prepared_complete(purifications=(prepared,))
        if not complete:
            for protocol in (prepared.left_protocol, prepared.right_protocol):
                self._cancel_protocol(protocol)
        keep = self.pairs.pop(prepared.keep_pair_id, None)
        measure = self.pairs.pop(prepared.measure_pair_id, None)
        self._protocol_owned_pair_ids.difference_update(
            (prepared.keep_pair_id, prepared.measure_pair_id)
        )
        success = (
            complete
            and keep is not None
            and measure is not None
            and prepared.left_protocol.meas_res is not None
            and prepared.left_protocol.meas_res
            == prepared.right_protocol.meas_res
            and self._is_entangled(
                prepared.kept_left_memory,
                keep.right,
                prepared.kept_right_memory,
            )
            and self._is_entangled(
                prepared.kept_right_memory,
                keep.left,
                prepared.kept_left_memory,
            )
        )
        if not success:
            for memory in (
                prepared.kept_left_memory,
                prepared.kept_right_memory,
                prepared.measured_left_memory,
                prepared.measured_right_memory,
            ):
                self._release_memory(memory)
            return None

        prepared.kept_left_memory.bds_decohere()
        prepared.kept_right_memory.bds_decohere()
        keep.fidelity = min(
            prepared.kept_left_memory.get_bds_fidelity(),
            prepared.kept_right_memory.get_bds_fidelity(),
        )
        keep.born = self.physical_time_ps
        keep.reserved_by = None
        self.pairs[keep.pair_id] = keep
        # Successful BBPSSW has already returned both measured memories to RAW.
        return keep.pair_id

    def begin_swap(
        self,
        action: SwapAction,
        attempt_id: str,
        *,
        allow_existing_reservation: bool = False,
    ) -> PreparedSwap | None:
        """Start one swap protocol without advancing the SeQUeNCe timeline."""

        if not attempt_id:
            raise ValueError("attempt_id must be non-empty")
        self._sync_pairs()
        left = self.pairs.get(action.left_pair_id)
        right = self.pairs.get(action.right_pair_id)
        if (
            left is None
            or right is None
            or set(left.endpoints) & set(right.endpoints) != {action.middle}
            or (
                not allow_existing_reservation
                and (left.reserved_by is not None or right.reserved_by is not None)
            )
        ):
            return None
        left_outer = left.right_memory if left.left == action.middle else left.left_memory
        right_outer = right.right_memory if right.left == action.middle else right.left_memory
        left_middle = left.left_memory if left.left == action.middle else left.right_memory
        right_middle = right.left_memory if right.left == action.middle else right.right_memory
        middle_node = self.nodes[action.middle]
        left_outer_node = left.right if left.left == action.middle else left.left
        right_outer_node = right.right if right.left == action.middle else right.left
        if (
            str(left_outer_node) != str(left_middle.entangled_memory["node_id"])
            or str(right_outer_node) != str(right_middle.entangled_memory["node_id"])
        ):
            return None

        left.reserved_by = attempt_id
        right.reserved_by = attempt_id
        self._protocol_owned_pair_ids.update((left.pair_id, right.pair_id))
        protocols: list[object] = []
        try:
            middle_node.set_seed(self._event_seed(
                "swap", self.physical_time_ps, action.middle,
                min(action.left_pair_id, action.right_pair_id),
                max(action.left_pair_id, action.right_pair_id),
            ))
            left_node = self.nodes[left_outer_node]
            right_node = self.nodes[right_outer_node]
            suffix = f"swap-{self._counter}"
            end_left = self._EntanglementSwappingB.create(
                left_node, f"{suffix}-l", left_outer
            )
            middle = self._EntanglementSwappingA.create(
                middle_node,
                f"{suffix}-m",
                left_middle,
                right_middle,
                success_prob=self.spec.physical.swap_probability,
            )
            end_right = self._EntanglementSwappingB.create(
                right_node, f"{suffix}-r", right_outer
            )
            for node, protocol in (
                (left_node, end_left),
                (middle_node, middle),
                (right_node, end_right),
            ):
                node.protocols.append(protocol)
                protocols.append(protocol)
            end_left.set_others(
                middle.name, middle_node.name, [left_middle.name, right_middle.name]
            )
            end_right.set_others(
                middle.name, middle_node.name, [left_middle.name, right_middle.name]
            )
            middle.set_others(end_left.name, left_node.name, [left_outer.name])
            middle.set_others(end_right.name, right_node.name, [right_outer.name])
            middle.start()
            self._counter += 1
            return PreparedSwap(
                action=action,
                attempt_id=attempt_id,
                left_pair_id=left.pair_id,
                right_pair_id=right.pair_id,
                left_outer=left_outer,
                right_outer=right_outer,
                left_outer_node=int(left_outer_node),
                right_outer_node=int(right_outer_node),
                middle_protocol=middle,
                end_left_protocol=end_left,
                end_right_protocol=end_right,
                started_time_ps=self.physical_time_ps,
            )
        except Exception:
            for protocol in reversed(protocols):
                self._cancel_protocol(protocol)
            if left.reserved_by == attempt_id:
                left.reserved_by = None
            if right.reserved_by == attempt_id:
                right.reserved_by = None
            self._protocol_owned_pair_ids.difference_update(
                (left.pair_id, right.pair_id)
            )
            raise

    def cancel_swap(self, prepared: PreparedSwap) -> None:
        """Cancel a swap handle before its timeline epoch is run."""

        if self.physical_time_ps != prepared.started_time_ps:
            raise RuntimeError("cannot cancel swap after physical time advanced")
        for protocol in (
            prepared.end_left_protocol,
            prepared.middle_protocol,
            prepared.end_right_protocol,
        ):
            self._cancel_protocol(protocol)
        for pair_id in (prepared.left_pair_id, prepared.right_pair_id):
            pair = self.pairs.get(pair_id)
            if pair is not None and pair.reserved_by == prepared.attempt_id:
                pair.reserved_by = None
        self._protocol_owned_pair_ids.difference_update(
            (prepared.left_pair_id, prepared.right_pair_id)
        )

    def can_begin_swap(self, action: SwapAction) -> bool:
        """Read-only preflight for an event-level swap reservation."""

        self._sync_pairs()
        left = self.pairs.get(action.left_pair_id)
        right = self.pairs.get(action.right_pair_id)
        if (
            left is None
            or right is None
            or left.reserved_by is not None
            or right.reserved_by is not None
            or set(left.endpoints) & set(right.endpoints) != {action.middle}
        ):
            return False
        left_outer = left.right_memory if left.left == action.middle else left.left_memory
        right_outer = right.right_memory if right.left == action.middle else right.left_memory
        left_middle = left.left_memory if left.left == action.middle else left.right_memory
        right_middle = right.left_memory if right.left == action.middle else right.right_memory
        left_outer_node = left.right if left.left == action.middle else left.left
        right_outer_node = right.right if right.left == action.middle else right.left
        return (
            str(left_outer_node) == str(left_middle.entangled_memory.get("node_id"))
            and str(right_outer_node) == str(right_middle.entangled_memory.get("node_id"))
            and left_middle.fidelity > 0
            and right_middle.fidelity > 0
        )

    def finish_swap(self, prepared: PreparedSwap) -> str | None:
        """Complete a previously started swap at the current timeline time."""

        left = self.pairs.pop(prepared.left_pair_id, None)
        right = self.pairs.pop(prepared.right_pair_id, None)
        self._protocol_owned_pair_ids.difference_update(
            (prepared.left_pair_id, prepared.right_pair_id)
        )
        success = bool(prepared.middle_protocol.is_success)

        if not success:
            self._release_memory(prepared.left_outer)
            self._release_memory(prepared.right_outer)
            return None
        if not (
            self._is_entangled(
                prepared.left_outer, prepared.right_outer_node, prepared.right_outer
            )
            and self._is_entangled(
                prepared.right_outer, prepared.left_outer_node, prepared.left_outer
            )
        ):
            self._release_memory(prepared.left_outer)
            self._release_memory(prepared.right_outer)
            return None

        pair_id = "long-" + hashlib.sha256(
            "|".join(sorted((prepared.left_pair_id, prepared.right_pair_id))).encode()
        ).hexdigest()[:16]
        if prepared.left_outer_node <= prepared.right_outer_node:
            low_memory, high_memory = prepared.left_outer, prepared.right_outer
        else:
            low_memory, high_memory = prepared.right_outer, prepared.left_outer
        self.pairs[pair_id] = ResourcePair(
            pair_id,
            min(prepared.left_outer_node, prepared.right_outer_node),
            max(prepared.left_outer_node, prepared.right_outer_node),
            low_memory,
            high_memory,
            min(low_memory.fidelity, high_memory.fidelity),
            self.physical_time_ps,
        )
        return pair_id

    def _execute_swap(self, action: SwapAction) -> str | None:
        """Execute one adjacent-pair swap through SeQUeNCe."""

        prepared = self.begin_swap(
            action,
            f"atomic:{self.physical_time_ps}:{action.left_pair_id}:{action.right_pair_id}",
            allow_existing_reservation=True,
        )
        if prepared is None:
            return None
        self._run_protocols(
            (prepared.end_left_protocol, prepared.middle_protocol, prepared.end_right_protocol),
            self._swap_window_ps(),
        )
        return self.finish_swap(prepared)

    def execute_swap(self, action: SwapAction) -> bool:
        return self._execute_swap(action) is not None

    def execute_actions(self, actions: Iterable[SwapAction]) -> str | None:
        outputs: dict[str, str] = {}
        last: str | None = None
        for index, action in enumerate(actions):
            left_id = outputs.get(action.left_pair_id, action.left_pair_id)
            right_id = outputs.get(action.right_pair_id, action.right_pair_id)
            last = self._execute_swap(
                SwapAction(action.request_id, action.middle, left_id, right_id)
            )
            if last is None:
                return None
            outputs[f"@{index}"] = last
        return last

    def execute_lane(
        self,
        lane: SwapLane,
        allocation_id: str | None = None,
    ) -> LaneExecutionResult:
        original_ids = set(lane.elementary_pair_ids)
        lane_scope = set(original_ids)
        consumed: set[str] = set()
        outputs: dict[str, str] = {}
        last: str | None = None
        failed_action_index: int | None = None
        attempted_swaps = 0

        if not lane.swap_actions:
            if len(lane.elementary_pair_ids) == 1:
                candidate = lane.elementary_pair_ids[0]
                if candidate in self.pairs:
                    last = candidate
                    if allocation_id is not None:
                        self.pairs[candidate].reserved_by = allocation_id
            untouched = tuple(sorted(pair_id for pair_id in original_ids if pair_id in self.pairs))
            return LaneExecutionResult(lane.lane, last, (), untouched, untouched, None, 0)

        for index, action in enumerate(lane.swap_actions):
            attempted_swaps += 1
            left_id = outputs.get(action.left_pair_id, action.left_pair_id)
            right_id = outputs.get(action.right_pair_id, action.right_pair_id)
            before = {pair_id for pair_id in (left_id, right_id) if pair_id in self.pairs}
            last = self._execute_swap(
                SwapAction(action.request_id, action.middle, left_id, right_id)
            )
            consumed.update(pair_id for pair_id in before if pair_id not in self.pairs)
            if last is None:
                failed_action_index = index
                break
            outputs[f"@{index}"] = last
            lane_scope.add(last)
            if allocation_id is not None:
                self.pairs[last].reserved_by = allocation_id

        untouched = tuple(sorted(pair_id for pair_id in original_ids if pair_id in self.pairs))
        surviving = tuple(sorted(pair_id for pair_id in lane_scope if pair_id in self.pairs))
        return LaneExecutionResult(
            lane.lane,
            last,
            tuple(sorted(consumed)),
            untouched,
            surviving,
            failed_action_index,
            attempted_swaps,
        )

    def execute_lanes(
        self,
        lanes: Iterable[SwapLane],
        allocation_id: str | None = None,
    ) -> tuple[LaneExecutionResult, ...]:
        lane_list = tuple(lanes)
        lane_ids = [lane.lane for lane in lane_list]
        if len(set(lane_ids)) != len(lane_ids):
            raise ValueError("duplicate swap lane")
        return tuple(
            self.execute_lane(lane, allocation_id)
            for lane in sorted(lane_list, key=lambda item: item.lane)
        )

    def release_allocation(self, allocation_id: str) -> tuple[str, ...]:
        released = []
        for pair in self.pairs.values():
            if pair.reserved_by == allocation_id:
                pair.reserved_by = None
                released.append(pair.pair_id)
        return tuple(sorted(released))

    def advance_slot(self) -> None:
        """Advance logical time and let SeQUeNCe process physical events."""
        self.time += 1
        self._run_until(self.timeline.now() + self.spec.physical.slot_duration_ps)
        self._sync_pairs()
