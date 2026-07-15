from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class PolicyOutput:
    logits: Tensor
    value: Tensor


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _masked_max(values: Tensor, mask: Tensor) -> Tensor:
    negative = torch.finfo(values.dtype).min
    result = values.masked_fill(~mask.unsqueeze(-1), negative).amax(dim=1)
    empty = ~mask.any(dim=1)
    if empty.any():
        result = torch.where(empty.unsqueeze(-1), torch.zeros_like(result), result)
    return result


class DynamicPlanActorCritic(nn.Module):
    """Centralized actor-critic over a variable-size set of candidate plans.

    Candidate plans are encoded independently and pooled with a permutation-
    invariant DeepSets context.  The actor emits one score per candidate and a
    learned STOP score.  The critic sees the global observation plus the full
    pooled candidate set, making it centralized without requiring fixed action
    dimensions.
    """

    def __init__(
        self,
        plan_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int = 128,
        request_feature_dim: int = 0,
    ):
        super().__init__()
        self.plan_feature_dim = plan_feature_dim
        self.global_feature_dim = global_feature_dim
        self.request_feature_dim = request_feature_dim
        self.hidden_dim = hidden_dim

        self.plan_encoder = nn.Sequential(
            nn.Linear(plan_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_dim), nn.Tanh()
        )
        self.request_encoder = (
            nn.Sequential(
                nn.Linear(request_feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            if request_feature_dim > 0
            else None
        )
        context_inputs = 5 if self.request_encoder is not None else 3
        self.context = nn.Sequential(
            nn.Linear(hidden_dim * context_inputs, hidden_dim), nn.Tanh()
        )
        self.plan_actor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        self.stop_actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.apply(self._initialize)
        nn.init.orthogonal_(self.plan_actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.stop_actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=2**0.5)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        plan_features: Tensor,
        global_features: Tensor,
        plan_mask: Tensor,
        action_mask: Tensor | None = None,
        request_features: Tensor | None = None,
        request_mask: Tensor | None = None,
    ) -> PolicyOutput:
        """Evaluate a padded batch.

        Args:
            plan_features: ``[batch, max_plans, plan_feature_dim]``.
            global_features: ``[batch, global_feature_dim]``.
            plan_mask: True for real (unpadded) plans, shape ``[batch, max_plans]``.
            action_mask: Optional legality mask with STOP last, shape
                ``[batch, max_plans + 1]``.  STOP is forced legal.
        """
        if plan_features.ndim != 3:
            raise ValueError("plan_features must have shape [batch, plans, features]")
        if global_features.ndim != 2:
            raise ValueError("global_features must have shape [batch, features]")
        if plan_mask.shape != plan_features.shape[:2]:
            raise ValueError("plan_mask shape must match the first two plan feature dimensions")

        plan_hidden = self.plan_encoder(plan_features)
        global_hidden = self.global_encoder(global_features)
        mean_hidden = _masked_mean(plan_hidden, plan_mask)
        max_hidden = _masked_max(plan_hidden, plan_mask)
        context_parts = [global_hidden, mean_hidden, max_hidden]
        if self.request_encoder is not None:
            if request_features is None or request_mask is None:
                raise ValueError("request features and mask are required by this model")
            request_hidden = self.request_encoder(request_features)
            context_parts.extend(
                (_masked_mean(request_hidden, request_mask), _masked_max(request_hidden, request_mask))
            )
        context = self.context(torch.cat(context_parts, dim=-1))

        expanded_context = context.unsqueeze(1).expand(-1, plan_features.shape[1], -1)
        plan_logits = self.plan_actor(torch.cat((plan_hidden, expanded_context), dim=-1)).squeeze(-1)
        stop_logit = self.stop_actor(context)
        logits = torch.cat((plan_logits, stop_logit), dim=-1)

        if action_mask is None:
            action_mask = torch.cat(
                (plan_mask, torch.ones((plan_mask.shape[0], 1), dtype=torch.bool, device=plan_mask.device)),
                dim=-1,
            )
        else:
            if action_mask.shape != logits.shape:
                raise ValueError("action_mask must contain one entry per plan plus STOP")
            action_mask = action_mask.bool().clone()
            if not action_mask.any(dim=-1).all():
                raise ValueError("each batch row must expose at least one legal action")
        logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        return PolicyOutput(logits=logits, value=self.critic(context).squeeze(-1))

    def distribution_and_value(
        self,
        plan_features: Tensor,
        global_features: Tensor,
        plan_mask: Tensor,
        action_mask: Tensor | None = None,
        request_features: Tensor | None = None,
        request_mask: Tensor | None = None,
    ) -> tuple[torch.distributions.Categorical, Tensor]:
        output = self(
            plan_features,
            global_features,
            plan_mask,
            action_mask,
            request_features,
            request_mask,
        )
        return torch.distributions.Categorical(logits=output.logits), output.value
