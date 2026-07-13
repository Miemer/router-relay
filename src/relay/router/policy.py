"""Post-classifier policy chain: confidence_gate → large_context_floor → sticky.

Mirrors the stage order of OpenSquilla's `engine/routing/policy.py` (a subset):
we keep `confidence_gate` and `large_context_floor` verbatim in spirit, and
implement `sticky` as a simplified `anti_downgrade` (prevent flapping *down* to a
cheaper tier when borderline — the dangerous under-routing direction).
"""

from __future__ import annotations

from .features import FeatureBundle
from .scorer import ScoreResult

_TIERS = ("c0", "c1", "c2", "c3")


def _idx(tier: str) -> int:
    return _TIERS.index(tier)


def _bump(tier: str, steps: int = 1) -> str:
    return _TIERS[min(_idx(tier) + steps, len(_TIERS) - 1)]


def _max(a: str, b: str) -> str:
    return a if _idx(a) >= _idx(b) else b


def apply_policy(
    score: ScoreResult,
    features: FeatureBundle,
    session_key: str,
    history,  # RoutingHistory
    settings,
) -> tuple[str, list]:
    """Return (final_tier, trail). `trail` is a list of (stage_name, payload)."""
    tier = score.tier
    trail: list[tuple[str, dict]] = []

    # 1. confidence_gate: low margin near a tier boundary → upgrade one tier
    #    (safer to send the stronger model when uncertain).
    if score.confidence < settings.router_confidence_threshold:
        before = tier
        tier = _bump(tier, 1)
        trail.append(("confidence_gate", {
            "from": before, "to": tier, "confidence": score.confidence,
        }))

    # 2. complaint_upgrade: the user is complaining about the previous answer →
    #    upgrade one tier (stronger model to fix it). The signal is captured in
    #    the decision record so offline realignment can mark the PREVIOUS turn as
    #    under-routed (OpenSquilla retrospective_under_routing).
    if features.complaint_detected:
        before = tier
        tier = _bump(tier, 1)
        trail.append(("complaint_upgrade", {
            "from": before, "to": tier, "hits": features.complaint_hits,
        }))

    # 3. large_context_floor: very long context → floor at c2 (small context
    #    windows on cheap models would truncate).
    if features.total_context_chars > settings.router_large_context_chars:
        before = tier
        tier = _max(tier, "c2")
        if tier != before:
            trail.append(("large_context_floor", {
                "from": before, "to": tier, "chars": features.total_context_chars,
            }))

    # 4. sticky (anti-downgrade): if the previous turn in this session ran on a
    #    higher tier and this turn is borderline-down by exactly one, keep the
    #    previous tier to avoid flapping between models mid-conversation.
    prev = history.recent_tier(session_key)
    if (
        prev is not None
        and _idx(prev) - _idx(tier) == 1
        and score.confidence < settings.router_confidence_threshold
    ):
        trail.append(("sticky", {"proposed": tier, "kept": prev}))
        tier = prev

    return tier, trail
