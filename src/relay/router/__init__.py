"""P1 routing layer: rule-based scorer + policy chain + bounded execution.

The `apply_router` entry point is the seam where a P3 trained LightGBM strategy
can drop in (same signature, swap `score_features`).
"""

from .runtime import DecisionStore, RoutingDecision, RoutingHistory, apply_router

__all__ = ["apply_router", "RoutingDecision", "RoutingHistory", "DecisionStore"]
