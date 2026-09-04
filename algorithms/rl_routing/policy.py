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

    RESOURCE_DEMAND_RELATION = 10

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
            aggregate = self._aggregate_messages(
                messages,
                destinations,
                edge_types,
                edge_features,
                node_count=node_embeddings.shape[0],
            )
        update = self.update(torch.cat((node_embeddings, aggregate), dim=-1))
        return self.normalization(node_embeddings + update)

    @classmethod
    def _aggregate_messages(
        cls,
        messages: Tensor,
        destinations: Tensor,
        edge_types: Tensor,
        edge_features: Tensor,
        *,
        node_count: int,
    ) -> Tensor:
        """Respect topology symmetry and additive resource-capacity semantics."""

        aggregate = messages.new_zeros((node_count, messages.shape[-1]))
        additive = edge_types == cls.RESOURCE_DEMAND_RELATION
        ordinary = ~additive
        if bool(ordinary.any()):
            ordinary_sum = torch.zeros_like(aggregate)
            ordinary_sum.index_add_(
                0,
                destinations[ordinary],
                messages[ordinary],
            )
            counts = messages.new_zeros((node_count,))
            counts.index_add_(
                0,
                destinations[ordinary],
                torch.ones_like(
                    destinations[ordinary],
                    dtype=messages.dtype,
                ),
            )
            aggregate = aggregate + ordinary_sum / counts.clamp_min(
                1.0
            ).unsqueeze(-1)
        if bool(additive.any()):
            # Column 0 is demand / capacity.  Column 4 divides one unit of
            # prior mass hierarchically over each request's legal routes,
            # constructions, and starts.  Summation therefore measures
            # inter-request capacity pressure without counting mutually
            # exclusive alternatives as independent traffic; illegal
            # alternatives carry zero mass.
            weights = (
                edge_features[additive, 0]
                * edge_features[additive, 4]
            ).unsqueeze(-1)
            aggregate.index_add_(
                0,
                destinations[additive],
                messages[additive] * weights,
            )
        return aggregate


class RoutingGraphEncoder(nn.Module):
    """Encode one heterogeneous routing graph into nodes and global context."""

    def __init__(self, hidden_dim: int, message_passing_layers: int) -> None:
        super().__init__()
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
        self.type_pool = nn.Sequential(
            nn.Linear(NODE_TYPE_COUNT * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, graph: RoutingGraph) -> tuple[Tensor, Tensor]:
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
        pooled_by_type = []
        for node_type in range(NODE_TYPE_COUNT):
            selected = embeddings[graph.node_types == node_type]
            pooled_by_type.append(
                embeddings.new_zeros((embeddings.shape[-1],))
                if selected.shape[0] == 0
                else selected.mean(dim=0)
            )
        context = self.type_pool(torch.cat(pooled_by_type))
        context = context + self.global_encoder(graph.global_features)
        return embeddings, context


class GraphActorCritic(nn.Module):
    """Independent graph actor and critic for stable clipped optimization."""

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
        self.actor_encoder = RoutingGraphEncoder(
            hidden_dim,
            message_passing_layers,
        )
        self.critic_encoder = RoutingGraphEncoder(
            hidden_dim,
            message_passing_layers,
        )
        self.request_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.route_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.construction_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.start_head = nn.Sequential(
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

    def actor_forward(
        self,
        graph: RoutingGraph,
    ) -> tuple[Tensor, Tensor, Tensor]:
        actor_embeddings, actor_context = self.actor_encoder(graph)
        stop_logit = self.stop_head(actor_context).squeeze(-1)
        return actor_embeddings, actor_context, stop_logit

    def critic_forward(self, graph: RoutingGraph) -> Tensor:
        _critic_embeddings, critic_context = self.critic_encoder(graph)
        value = self.value_head(critic_context).squeeze(-1)
        return value

    def forward(
        self,
        graph: RoutingGraph,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        embeddings, context, stop_logit = self.actor_forward(graph)
        value = self.critic_forward(graph)
        return embeddings, context, stop_logit, value


@dataclass(frozen=True)
class PolicyTokenEvaluation:
    """One atomic action in the augmented autoregressive MDP."""

    prefix_action_ids: tuple[str, ...]
    action_id: str
    log_probability: Tensor
    entropy: Tensor
    value: Tensor


@dataclass(frozen=True)
class PolicyEvaluation:
    action: RoutingAction
    log_probability: Tensor
    entropy: Tensor
    value: Tensor
    token_count: int
    tokens: tuple[PolicyTokenEvaluation, ...]


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

    def actor_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            self.actor_critic.actor_encoder.parameters()
        ) + tuple(
            self.actor_critic.request_head.parameters()
        ) + tuple(
            self.actor_critic.route_head.parameters()
        ) + tuple(
            self.actor_critic.construction_head.parameters()
        ) + tuple(
            self.actor_critic.start_head.parameters()
        ) + tuple(
            self.actor_critic.stop_head.parameters()
        )

    def critic_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            self.actor_critic.critic_encoder.parameters()
        ) + tuple(self.actor_critic.value_head.parameters())

    @staticmethod
    def _legal_hierarchy(
        graph: RoutingGraph,
    ) -> dict[
        str,
        dict[tuple[int, ...], dict[str, tuple[int, ...]]],
    ]:
        """Group legal joint candidates without weighting large groups more."""

        mutable: dict[
            str,
            dict[tuple[int, ...], dict[str, list[int]]],
        ] = {}
        for candidate_index, legal in enumerate(
            graph.candidate_legal_mask.tolist()
        ):
            if not legal:
                continue
            request_id = graph.candidate_request_ids[candidate_index]
            route_nodes = graph.candidate_route_nodes[candidate_index]
            construction_id = graph.candidate_construction_ids[
                candidate_index
            ]
            mutable.setdefault(request_id, {}).setdefault(
                route_nodes, {}
            ).setdefault(construction_id, []).append(candidate_index)
        return {
            request_id: {
                route_nodes: {
                    construction_id: tuple(candidate_indices)
                    for construction_id, candidate_indices in constructions.items()
                }
                for route_nodes, constructions in routes.items()
            }
            for request_id, routes in mutable.items()
        }

    @staticmethod
    def _group_embedding(
        embeddings: Tensor,
        node_indices: Tensor,
        candidate_indices: tuple[int, ...],
    ) -> Tensor:
        indices = torch.tensor(
            candidate_indices,
            dtype=torch.long,
            device=embeddings.device,
        )
        return embeddings[node_indices[indices]].mean(dim=0)

    def _token_choice(
        self,
        graph: RoutingGraph,
        *,
        action_id: str | None = None,
        deterministic: bool = False,
        include_value: bool = True,
        include_entropy: bool = True,
    ) -> tuple[str, Tensor, Tensor, Tensor]:
        """Choose one feasible joint action through a semantic hierarchy."""

        embeddings, context, stop_logit = self.actor_critic.actor_forward(
            graph
        )
        value = (
            self.actor_critic.critic_forward(graph)
            if include_value
            else context.new_zeros(())
        )
        hierarchy = self._legal_hierarchy(graph)
        if not hierarchy:
            if action_id not in {None, STOP_ACTION}:
                raise ValueError(f"infeasible routing action: {action_id}")
            zero = value.new_zeros(())
            return STOP_ACTION, zero, zero, value

        request_node_by_id = {
            request_id: int(node_index)
            for request_id, node_index in zip(
                graph.request_ids,
                graph.request_node_indices.tolist(),
                strict=True,
            )
        }
        request_ids = tuple(hierarchy)
        if set(request_ids) - set(request_node_by_id):
            raise ValueError("legal candidate has no request graph node")

        def score(head: nn.Module, embedding: Tensor) -> Tensor:
            return head(torch.cat((embedding, context), dim=-1)).squeeze(-1)

        request_logits = torch.stack(tuple(
            score(
                self.actor_critic.request_head,
                embeddings[request_node_by_id[request_id]],
            )
            for request_id in request_ids
        ))
        # STOP and one branch per legal request share the same top-level
        # categorical decision.  With equal logits, STOP and every request
        # receive equal probability, so neutral exploration is not geometric
        # in plan length and a request is not rewarded for owning more joint
        # candidates.
        top_distribution = Categorical(logits=torch.cat((
            stop_logit.reshape(1),
            request_logits,
        )))
        route_cache: dict[
            str,
            tuple[tuple[tuple[int, ...], ...], Categorical],
        ] = {}
        construction_cache: dict[
            tuple[str, tuple[int, ...]],
            tuple[tuple[str, ...], Categorical],
        ] = {}
        start_cache: dict[
            tuple[str, tuple[int, ...], str],
            tuple[tuple[int, ...], Categorical],
        ] = {}

        def route_distribution(
            request_id: str,
        ) -> tuple[tuple[tuple[int, ...], ...], Categorical]:
            cached = route_cache.get(request_id)
            if cached is not None:
                return cached
            routes = hierarchy[request_id]
            route_keys = tuple(routes)
            logits = []
            for route_key in route_keys:
                member_indices = tuple(
                    candidate_index
                    for indices in routes[route_key].values()
                    for candidate_index in indices
                )
                logits.append(score(
                    self.actor_critic.route_head,
                    self._group_embedding(
                        embeddings,
                        graph.candidate_node_indices,
                        member_indices,
                    ),
                ))
            result = route_keys, Categorical(logits=torch.stack(logits))
            route_cache[request_id] = result
            return result

        def construction_distribution(
            request_id: str,
            route_key: tuple[int, ...],
        ) -> tuple[tuple[str, ...], Categorical]:
            cache_key = request_id, route_key
            cached = construction_cache.get(cache_key)
            if cached is not None:
                return cached
            constructions = hierarchy[request_id][route_key]
            construction_ids = tuple(constructions)
            logits = tuple(
                score(
                    self.actor_critic.construction_head,
                    self._group_embedding(
                        embeddings,
                        graph.candidate_node_indices,
                        constructions[construction_id],
                    ),
                )
                for construction_id in construction_ids
            )
            result = construction_ids, Categorical(
                logits=torch.stack(logits)
            )
            construction_cache[cache_key] = result
            return result

        def start_distribution(
            request_id: str,
            route_key: tuple[int, ...],
            construction_id: str,
        ) -> tuple[tuple[int, ...], Categorical]:
            cache_key = request_id, route_key, construction_id
            cached = start_cache.get(cache_key)
            if cached is not None:
                return cached
            candidate_indices = hierarchy[request_id][route_key][
                construction_id
            ]
            logits = tuple(
                score(
                    self.actor_critic.start_head,
                    embeddings[graph.candidate_node_indices[candidate_index]],
                )
                for candidate_index in candidate_indices
            )
            result = candidate_indices, Categorical(
                logits=torch.stack(logits)
            )
            start_cache[cache_key] = result
            return result

        def choose(distribution: Categorical) -> int:
            selected = (
                distribution.logits.argmax()
                if deterministic
                else distribution.sample()
            )
            return int(selected.item())

        candidate_index: int | None
        if action_id is None:
            top_position = choose(top_distribution)
            if top_position == 0:
                selected_action_id = STOP_ACTION
                candidate_index = None
                selected_request_id = None
                selected_route = None
                selected_construction = None
                request_position = None
                route_position = None
                construction_position = None
                start_position = None
            else:
                request_position = top_position - 1
                selected_request_id = request_ids[request_position]
                route_keys, routes = route_distribution(selected_request_id)
                route_position = choose(routes)
                selected_route = route_keys[route_position]
                construction_ids, constructions = construction_distribution(
                    selected_request_id,
                    selected_route,
                )
                construction_position = choose(constructions)
                selected_construction = construction_ids[
                    construction_position
                ]
                candidate_indices, starts = start_distribution(
                    selected_request_id,
                    selected_route,
                    selected_construction,
                )
                start_position = choose(starts)
                candidate_index = candidate_indices[start_position]
                selected_action_id = graph.candidate_variable_ids[
                    candidate_index
                ]
        elif action_id == STOP_ACTION:
            selected_action_id = STOP_ACTION
            candidate_index = None
            top_position = 0
            selected_request_id = None
            selected_route = None
            selected_construction = None
            request_position = None
            route_position = None
            construction_position = None
            start_position = None
        else:
            try:
                candidate_index = graph.candidate_variable_ids.index(action_id)
            except ValueError as exc:
                raise KeyError(f"unknown routing action: {action_id}") from exc
            if not bool(graph.candidate_legal_mask[candidate_index]):
                raise ValueError(f"infeasible routing action: {action_id}")
            selected_action_id = action_id
            selected_request_id = graph.candidate_request_ids[candidate_index]
            selected_route = graph.candidate_route_nodes[candidate_index]
            selected_construction = graph.candidate_construction_ids[
                candidate_index
            ]
            request_position = request_ids.index(selected_request_id)
            top_position = request_position + 1
            route_keys, _routes = route_distribution(selected_request_id)
            route_position = route_keys.index(selected_route)
            construction_ids, _constructions = construction_distribution(
                selected_request_id,
                selected_route,
            )
            construction_position = construction_ids.index(
                selected_construction
            )
            candidate_indices, _starts = start_distribution(
                selected_request_id,
                selected_route,
                selected_construction,
            )
            start_position = candidate_indices.index(candidate_index)

        top_target = torch.tensor(
            top_position,
            dtype=torch.long,
            device=self.device,
        )
        log_probability = top_distribution.log_prob(top_target)
        if candidate_index is not None:
            stage_positions = (
                request_position,
                route_position,
                construction_position,
                start_position,
            )
            if any(position is None for position in stage_positions):
                raise RuntimeError("incomplete hierarchical action")
            route_keys, routes = route_distribution(selected_request_id)
            construction_ids, constructions = construction_distribution(
                selected_request_id,
                selected_route,
            )
            candidate_indices, starts = start_distribution(
                selected_request_id,
                selected_route,
                selected_construction,
            )
            distributions = (
                routes,
                constructions,
                starts,
            )
            for distribution, position in zip(
                distributions,
                stage_positions[1:],
                strict=True,
            ):
                target = torch.tensor(
                    position,
                    dtype=torch.long,
                    device=self.device,
                )
                log_probability = log_probability + distribution.log_prob(
                    target
                )

        if include_entropy:
            entropy = top_distribution.entropy()
            for request_position_index, request_id in enumerate(request_ids):
                request_probability = top_distribution.probs[
                    request_position_index + 1
                ]
                route_keys, routes = route_distribution(request_id)
                request_conditional_entropy = routes.entropy()
                for route_position_index, route_key in enumerate(route_keys):
                    construction_ids, constructions = (
                        construction_distribution(request_id, route_key)
                    )
                    route_conditional_entropy = constructions.entropy()
                    for construction_position_index, construction_id in (
                        enumerate(construction_ids)
                    ):
                        _candidate_indices, starts = start_distribution(
                            request_id,
                            route_key,
                            construction_id,
                        )
                        route_conditional_entropy = (
                            route_conditional_entropy
                            + constructions.probs[construction_position_index]
                            * starts.entropy()
                        )
                    request_conditional_entropy = (
                        request_conditional_entropy
                        + routes.probs[route_position_index]
                        * route_conditional_entropy
                    )
                entropy = (
                    entropy
                    + request_probability * request_conditional_entropy
                )
        else:
            entropy = context.new_zeros(())
        return selected_action_id, log_probability, entropy, value

    def sample_action(
        self,
        observation: RoutingObservation,
        *,
        deterministic: bool = False,
        include_value: bool = True,
    ) -> PolicyEvaluation:
        builder = FeasiblePlanBuilder(observation)
        action_ids: list[str] = []
        log_probabilities: list[Tensor] = []
        entropies: list[Tensor] = []
        token_evaluations: list[PolicyTokenEvaluation] = []
        initial_value: Tensor | None = None

        while not builder.stopped:
            graph = build_routing_graph(
                observation,
                builder,
                device=self.device,
            )
            action_id, log_probability, entropy, value = self._token_choice(
                graph,
                deterministic=deterministic,
                include_value=include_value,
                include_entropy=include_value,
            )
            if initial_value is None:
                initial_value = value
            log_probabilities.append(log_probability)
            entropies.append(entropy)
            token_evaluations.append(PolicyTokenEvaluation(
                prefix_action_ids=tuple(action_ids),
                action_id=action_id,
                log_probability=log_probabilities[-1],
                entropy=entropies[-1],
                value=value,
            ))
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
            tokens=tuple(token_evaluations),
        )

    def evaluate_token(
        self,
        observation: RoutingObservation,
        prefix_action_ids: tuple[str, ...],
        action_id: str,
    ) -> PolicyTokenEvaluation:
        """Re-evaluate one action from an exact feasible plan prefix."""

        if STOP_ACTION in prefix_action_ids:
            raise ValueError("a token prefix cannot contain STOP")
        builder = FeasiblePlanBuilder(observation)
        for prefix_action_id in prefix_action_ids:
            builder.select(prefix_action_id)
        graph = build_routing_graph(
            observation,
            builder,
            device=self.device,
        )
        _, log_probability, entropy, value = self._token_choice(
            graph,
            action_id=action_id,
        )
        return PolicyTokenEvaluation(
            prefix_action_ids=prefix_action_ids,
            action_id=action_id,
            log_probability=log_probability,
            entropy=entropy,
            value=value,
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
        token_evaluations: list[PolicyTokenEvaluation] = []
        initial_value: Tensor | None = None

        for action_id in action.action_ids:
            graph = build_routing_graph(
                observation,
                builder,
                device=self.device,
            )
            _, log_probability, entropy, value = self._token_choice(
                graph,
                action_id=action_id,
            )
            if initial_value is None:
                initial_value = value
            log_probabilities.append(log_probability)
            entropies.append(entropy)
            token_evaluations.append(PolicyTokenEvaluation(
                prefix_action_ids=tuple(
                    item.action_id for item in token_evaluations
                ),
                action_id=action_id,
                log_probability=log_probabilities[-1],
                entropy=entropies[-1],
                value=value,
            ))
            builder.select(action_id)

        if not builder.stopped or initial_value is None:
            raise ValueError("routing action must contain a terminal STOP")
        return PolicyEvaluation(
            action=action,
            log_probability=torch.stack(log_probabilities).sum(),
            entropy=torch.stack(entropies).mean(),
            value=initial_value,
            token_count=len(action.action_ids),
            tokens=tuple(token_evaluations),
        )


__all__ = [
    "ARCQPolicy",
    "GraphActorCritic",
    "PolicyEvaluation",
    "PolicyTokenEvaluation",
    "RelationalMessageLayer",
    "RoutingGraphEncoder",
]
