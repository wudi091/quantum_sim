"""Feasibility-preserving autoregressive graph actor--critic for ARC-Q."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from .environment import (
    STOP_ACTION,
    FeasiblePlanBuilder,
    RoutingAction,
    RoutingObservation,
)
from .graph import (
    EDGE_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    NODE_FEATURE_DIM,
    NODE_TYPE_COUNT,
    RELATION_COUNT,
    RoutingGraph,
    build_routing_graph,
)


class RelationalMessageLayer(nn.Module):
    """One lightweight relation-aware message-passing layer."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.source = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relation = nn.Embedding(RELATION_COUNT, hidden_dim)
        self.edge = nn.Linear(EDGE_FEATURE_DIM, hidden_dim, bias=False)
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_embeddings: Tensor,
        edge_index: Tensor,
        edge_types: Tensor,
        edge_features: Tensor,
    ) -> Tensor:
        if edge_index.shape[1] == 0:
            aggregate = torch.zeros_like(node_embeddings)
        else:
            sources, destinations = edge_index
            messages = (
                self.source(node_embeddings[sources])
                + self.relation(edge_types)
                + self.edge(edge_features)
            )
            aggregate = torch.zeros_like(node_embeddings)
            aggregate.index_add_(0, destinations, messages)
            counts = torch.zeros(
                node_embeddings.shape[0],
                dtype=node_embeddings.dtype,
                device=node_embeddings.device,
            )
            counts.index_add_(
                0,
                destinations,
                torch.ones_like(destinations, dtype=node_embeddings.dtype),
            )
            aggregate = aggregate / counts.clamp_min(1.0).unsqueeze(-1)
        update = self.update(torch.cat((node_embeddings, aggregate), dim=-1))
        return self.normalization(node_embeddings + update)


class GraphActorCritic(nn.Module):
    """Scores candidate nodes and STOP while estimating macro-state value."""

    def __init__(
        self,
        hidden_dim: int = 96,
        message_passing_layers: int = 3,
    ) -> None:
        super().__init__()
        if hidden_dim < 8:
            raise ValueError("hidden_dim must be at least 8")
        if message_passing_layers < 1:
            raise ValueError("message_passing_layers must be positive")
        self.node_encoder = nn.Sequential(
            nn.Linear(NODE_FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_type_embedding = nn.Embedding(NODE_TYPE_COUNT, hidden_dim)
        self.layers = nn.ModuleList(
            RelationalMessageLayer(hidden_dim)
            for _ in range(message_passing_layers)
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.candidate_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph: RoutingGraph) -> tuple[Tensor, Tensor, Tensor]:
        embeddings = (
            self.node_encoder(graph.node_features)
            + self.node_type_embedding(graph.node_types)
        )
        for layer in self.layers:
            embeddings = layer(
                embeddings,
                graph.edge_index,
                graph.edge_types,
                graph.edge_features,
            )
        context = embeddings.mean(dim=0) + self.global_encoder(
            graph.global_features
        )
        candidate_embeddings = embeddings[graph.candidate_node_indices]
        if candidate_embeddings.shape[0]:
            repeated_context = context.unsqueeze(0).expand(
                candidate_embeddings.shape[0], -1
            )
            candidate_logits = self.candidate_head(torch.cat(
                (candidate_embeddings, repeated_context), dim=-1
            )).squeeze(-1)
        else:
            candidate_logits = embeddings.new_empty((0,))
        stop_logit = self.stop_head(context).squeeze(-1)
        value = self.value_head(context).squeeze(-1)
        return candidate_logits, stop_logit, value


@dataclass(frozen=True)
class PolicyEvaluation:
    action: RoutingAction
    log_probability: Tensor
    entropy: Tensor
    value: Tensor
    token_count: int


class ARCQPolicy(nn.Module):
    """Autoregressive policy whose only non-learned operation is masking."""

    def __init__(
        self,
        hidden_dim: int = 96,
        message_passing_layers: int = 3,
    ) -> None:
        super().__init__()
        self.actor_critic = GraphActorCritic(
            hidden_dim=hidden_dim,
            message_passing_layers=message_passing_layers,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _distribution(
        self,
        graph: RoutingGraph,
    ) -> tuple[Categorical, Tensor]:
        candidate_logits, stop_logit, value = self.actor_critic(graph)
        masked_candidate_logits = candidate_logits.masked_fill(
            ~graph.candidate_legal_mask,
            -torch.inf,
        )
        logits = torch.cat((masked_candidate_logits, stop_logit.reshape(1)))
        return Categorical(logits=logits), value

    def sample_action(
        self,
        observation: RoutingObservation,
        *,
        deterministic: bool = False,
    ) -> PolicyEvaluation:
        builder = FeasiblePlanBuilder(observation)
        action_ids: list[str] = []
        log_probabilities: list[Tensor] = []
        entropies: list[Tensor] = []
        initial_value: Tensor | None = None

        while not builder.stopped:
            graph = build_routing_graph(
                observation,
                builder,
                device=self.device,
            )
            distribution, value = self._distribution(graph)
            if initial_value is None:
                initial_value = value
            if deterministic:
                action_index = distribution.logits.argmax()
            else:
                action_index = distribution.sample()
            log_probabilities.append(distribution.log_prob(action_index))
            entropies.append(distribution.entropy())
            index = int(action_index.item())
            action_id = (
                STOP_ACTION
                if index == len(graph.candidate_variable_ids)
                else graph.candidate_variable_ids[index]
            )
            action_ids.append(action_id)
            builder.select(action_id)

        if initial_value is None:
            raise RuntimeError("policy produced no autoregressive token")
        action = RoutingAction(
            decision_slot=observation.slot,
            action_ids=tuple(action_ids),
        )
        return PolicyEvaluation(
            action=action,
            log_probability=torch.stack(log_probabilities).sum(),
            entropy=torch.stack(entropies).mean(),
            value=initial_value,
            token_count=len(action_ids),
        )

    def evaluate_action(
        self,
        observation: RoutingObservation,
        action: RoutingAction,
    ) -> PolicyEvaluation:
        if action.decision_slot != observation.slot:
            raise ValueError("routing action was produced for a stale slot")
        builder = FeasiblePlanBuilder(observation)
        log_probabilities: list[Tensor] = []
        entropies: list[Tensor] = []
        initial_value: Tensor | None = None

        for action_id in action.action_ids:
            graph = build_routing_graph(
                observation,
                builder,
                device=self.device,
            )
            distribution, value = self._distribution(graph)
            if initial_value is None:
                initial_value = value
            if action_id == STOP_ACTION:
                action_index = len(graph.candidate_variable_ids)
            else:
                try:
                    action_index = graph.candidate_variable_ids.index(action_id)
                except ValueError as exc:
                    raise KeyError(f"unknown routing action: {action_id}") from exc
                if not bool(graph.candidate_legal_mask[action_index]):
                    raise ValueError(f"infeasible routing action: {action_id}")
            action_tensor = torch.tensor(action_index, device=self.device)
            log_probabilities.append(distribution.log_prob(action_tensor))
            entropies.append(distribution.entropy())
            builder.select(action_id)

        if not builder.stopped or initial_value is None:
            raise ValueError("routing action must contain a terminal STOP")
        return PolicyEvaluation(
            action=action,
            log_probability=torch.stack(log_probabilities).sum(),
            entropy=torch.stack(entropies).mean(),
            value=initial_value,
            token_count=len(action.action_ids),
        )


__all__ = [
    "ARCQPolicy",
    "GraphActorCritic",
    "PolicyEvaluation",
    "RelationalMessageLayer",
]
