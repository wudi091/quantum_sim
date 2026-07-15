"""Small SeQUeNCe-backed resource kernel.

Routing code never receives SeQUeNCe objects.  This module owns memories,
randomness, exchange execution, and resource lifetime; callers only use pair
IDs and immutable snapshots.  Elementary generation is deliberately sampled
by this shared kernel so every planner sees the same trace.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

from .planner_api import SwapAction
from .spec import EpisodeSpec


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

    @property
    def endpoints(self) -> tuple[int, int]:
        return self.left, self.right


class _ResourceManager:
    def update(self, _protocol: object, memory: object, state: str) -> None:
        if state == "RAW":
            memory.reset()


class _Node:  # replaced with sequence.topology.node.Node at runtime
    pass


class SequenceBackend:
    """Common physical state for all planners."""

    def __init__(self, spec: EpisodeSpec):
        self.spec = spec
        self.time = 0
        self._counter = 0
        self.pairs: dict[str, ResourcePair] = {}
        self._sequence_ready = False
        self._build_sequence_world()

    def _build_sequence_world(self) -> None:
        try:
            from sequence.components.memory import Memory
            from sequence.components.optical_channel import ClassicalChannel
            from sequence.kernel.timeline import Timeline
            from sequence.topology.node import Node
        except ImportError as exc:  # pragma: no cover - exercised in deployment env
            raise RuntimeError(
                "SeQUeNCe is required; install requirements.txt in Python 3.12+"
            ) from exc

        class ResourceNode(Node):
            def __init__(self, name: str, timeline: object, seed: int):
                super().__init__(name, timeline, seed=seed)
                self.resource_manager = _ResourceManager()

        self._Memory = Memory
        self._SwappingA = None
        self._SwappingB = None
        self.timeline = Timeline(stop_time=10 ** 23)
        self.nodes = {
            node: ResourceNode(str(node), self.timeline, self.spec.seed + node)
            for node in self.spec.nodes
        }
        self._memories: dict[tuple[int, int], list[object]] = {}
        # A zero-delay classical mesh keeps the physical result delivery in
        # SeQUeNCe while avoiding a routing policy hidden in the backend.
        for src in self.spec.nodes:
            for dst in self.spec.nodes:
                if src == dst:
                    continue
                channel = ClassicalChannel(
                    f"cc-{src}-{dst}", self.timeline, 0, self.spec.physical.classical_delay_ps
                )
                channel.set_ends(self.nodes[src], self.nodes[dst].name)
        self.timeline.init()
        self._sequence_ready = True

    def _new_id(self, prefix: str = "epr") -> str:
        value = f"{prefix}-{self._counter}"
        self._counter += 1
        return value

    def _event_seed(self, *parts: object) -> int:
        payload = "|".join(map(str, (self.spec.seed, *parts))).encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    def _uniform(self, *parts: object) -> float:
        return self._event_seed(*parts) / 2**64

    def _memory(self, node: int, pair_id: str) -> object:
        memory = self._Memory(
            f"mem-{pair_id}-{node}", self.timeline,
            self.spec.physical.initial_fidelity, 1e9, 1.0, -1, 500,
            cutoff_flag=False,
        )
        memory.fidelity = self.spec.physical.initial_fidelity
        return memory

    def generate_elementary_pairs(self) -> tuple[str, ...]:
        """Sample one generation attempt per free topology edge."""
        generated: list[str] = []
        for edge_index, (raw_u, raw_v) in enumerate(self.spec.edges):
            u, v = sorted((raw_u, raw_v))
            occupied = sum(
                1 for pair in self.pairs.values()
                if set(pair.endpoints) == {u, v}
            )
            if occupied >= self.spec.physical.memory_capacity:
                continue
            if self._uniform("generation", self.time, edge_index) > self.spec.physical.generation_probability:
                continue
            pair_id = f"epr-{self.time}-{edge_index}"
            left_memory, right_memory = self._memory(u, pair_id), self._memory(v, pair_id)
            left_memory.entangled_memory = {"node_id": str(v), "memo_id": right_memory.name}
            right_memory.entangled_memory = {"node_id": str(u), "memo_id": left_memory.name}
            phi_plus = [2 ** -0.5, 0, 0, 2 ** -0.5]
            self.timeline.quantum_manager.set(
                [left_memory.qstate_key, right_memory.qstate_key], phi_plus
            )
            self.pairs[pair_id] = ResourcePair(
                pair_id, u, v, left_memory, right_memory,
                self.spec.physical.initial_fidelity, self.time,
            )
            generated.append(pair_id)
        return tuple(generated)

    def _load_protocols(self):
        if self._SwappingA is None:
            from sequence.entanglement_management.swapping import EntanglementSwappingA, EntanglementSwappingB
            self._SwappingA, self._SwappingB = EntanglementSwappingA, EntanglementSwappingB

    def _execute_swap(self, action: SwapAction) -> str | None:
        """Execute one adjacent-pair swap and return the output pair ID."""
        self._load_protocols()
        left = self.pairs.get(action.left_pair_id)
        right = self.pairs.get(action.right_pair_id)
        if (left is None or right is None
                or set(left.endpoints) & set(right.endpoints) != {action.middle}):
            return None
        left_outer = left.right_memory if left.left == action.middle else left.left_memory
        right_outer = right.right_memory if right.left == action.middle else right.left_memory
        left_middle = left.left_memory if left.left == action.middle else left.right_memory
        right_middle = right.left_memory if right.left == action.middle else right.right_memory
        middle_node = self.nodes[action.middle]
        middle_node.set_seed(self._event_seed(
            "swap", self.time, action.middle,
            min(action.left_pair_id, action.right_pair_id),
            max(action.left_pair_id, action.right_pair_id),
        ))
        left_outer_node = left.right if left.left == action.middle else left.left
        right_outer_node = right.right if right.left == action.middle else right.left
        if (str(left_outer_node) != str(left_middle.entangled_memory["node_id"])
                or str(right_outer_node) != str(right_middle.entangled_memory["node_id"])):
            return None
        left_node = self.nodes[left_outer_node]
        right_node = self.nodes[right_outer_node]
        suffix = self._new_id("swap")
        end_left = self._SwappingB.create(left_node, f"{suffix}-l", left_outer)
        middle = self._SwappingA.create(
            middle_node, f"{suffix}-m", left_middle, right_middle,
            success_prob=self.spec.physical.swap_probability,
            degradation=self.spec.physical.swap_degradation,
        )
        end_right = self._SwappingB.create(right_node, f"{suffix}-r", right_outer)
        for node, protocol in ((left_node, end_left), (middle_node, middle), (right_node, end_right)):
            node.protocols.append(protocol)
        end_left.set_others(middle.name, middle_node.name, [left_middle.name, right_middle.name])
        end_right.set_others(middle.name, middle_node.name, [left_middle.name, right_middle.name])
        middle.set_others(end_left.name, left_node.name, [left_outer.name])
        middle.set_others(end_right.name, right_node.name, [right_outer.name])
        middle.start()
        self.timeline.run()
        success = bool(middle.is_success)
        for node, protocols in ((left_node, (end_left,)), (middle_node, (middle,)), (right_node, (end_right,))):
            node.protocols[:] = [protocol for protocol in node.protocols if protocol not in protocols]
        self.pairs.pop(left.pair_id, None)
        self.pairs.pop(right.pair_id, None)
        if not success:
            return None
        pair_id = "long-" + hashlib.sha256(
            "|".join(sorted((left.pair_id, right.pair_id))).encode()
        ).hexdigest()[:16]
        if left_outer_node <= right_outer_node:
            low_memory, high_memory = left_outer, right_outer
        else:
            low_memory, high_memory = right_outer, left_outer
        self.pairs[pair_id] = ResourcePair(
            pair_id,
            min(left_outer_node, right_outer_node),
            max(left_outer_node, right_outer_node),
            low_memory, high_memory,
            min(left_outer.fidelity, right_outer.fidelity), self.time,
        )
        return pair_id

    def execute_swap(self, action: SwapAction) -> bool:
        """Execute one adjacent-pair swap through SeQUeNCe."""
        return self._execute_swap(action) is not None

    def execute_actions(self, actions: Iterable[SwapAction]) -> str | None:
        """Execute a symbolic swap chain; ``@N`` references operation N."""
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

    def advance_slot(self) -> None:
        self.time += 1
        lifetime = self.spec.physical.memory_lifetime
        for pair_id, pair in list(self.pairs.items()):
            if self.time - pair.born >= lifetime:
                pair.left_memory.reset()
                pair.right_memory.reset()
                self.pairs.pop(pair_id, None)
