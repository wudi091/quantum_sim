"""ARC-Q: feasibility-preserving autoregressive quantum routing."""

from .environment import (
    STOP_ACTION,
    ConstructionAwareRoutingEnvironment,
    FeasiblePlanBuilder,
    RoutingAction,
    RoutingObservation,
    RoutingTransition,
)
from .graph import RoutingGraph, build_routing_graph
from .policy import ARCQPolicy, GraphActorCritic, PolicyEvaluation
from .rollout import EpisodeRollout, PolicyRolloutStep, collect_episode
from .training import PPOConfig, PPODiagnostics, PPOTrainer
from .checkpoint import load_arcq_checkpoint, save_arcq_checkpoint
from .evaluation import (
    BaselineDefinition,
    EvaluationRecord,
    default_baselines,
    run_paired_evaluation,
    save_evaluation_records,
)

__all__ = [
    "STOP_ACTION",
    "ConstructionAwareRoutingEnvironment",
    "FeasiblePlanBuilder",
    "RoutingAction",
    "RoutingObservation",
    "RoutingTransition",
    "RoutingGraph",
    "build_routing_graph",
    "ARCQPolicy",
    "GraphActorCritic",
    "PolicyEvaluation",
    "EpisodeRollout",
    "PolicyRolloutStep",
    "collect_episode",
    "PPOConfig",
    "PPODiagnostics",
    "PPOTrainer",
    "load_arcq_checkpoint",
    "save_arcq_checkpoint",
    "BaselineDefinition",
    "EvaluationRecord",
    "default_baselines",
    "run_paired_evaluation",
    "save_evaluation_records",
]
