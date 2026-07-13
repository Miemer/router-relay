"""Deterministic rule-based difficulty scorer → tier (c0..c3) + confidence.

This is the P1 "classifier". It needs no training data and is the substrate a P3
LightGBM head will eventually replace at the same `score_features` call site
(see `runtime.apply_router`). Mirrors the role of OpenSquilla's
`InferenceCore.predict` but with a hand-tuned linear model instead of a trained
ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass

from .features import FeatureBundle

# Tier thresholds on the [0, 1] difficulty score. Tunable; mirror the 4-tier
# scheme of OpenSquilla's `ROUTE_CLASS_TO_TIER` (R0..R3 → c0..c3).
TIER_BOUNDS = (0.20, 0.45, 0.70)
TIERS = ("c0", "c1", "c2", "c3")
# Half-band width used to derive confidence (distance to nearest boundary).
_HALF_BAND = 0.10


@dataclass
class ScoreResult:
    difficulty: float
    tier: str
    confidence: float
    signals: dict


def _tier_for(score: float) -> str:
    if score < TIER_BOUNDS[0]:
        return "c0"
    if score < TIER_BOUNDS[1]:
        return "c1"
    if score < TIER_BOUNDS[2]:
        return "c2"
    return "c3"


def _confidence(score: float) -> float:
    """High in the middle of a band, low at boundaries (where the gate fires)."""
    dist = min(abs(score - bound) for bound in TIER_BOUNDS)
    return min(dist / _HALF_BAND, 1.0)


def score_features(features: FeatureBundle) -> ScoreResult:
    # Length: longer turns tend to be harder (capped contribution).
    len_score = min(features.char_len / 1500.0, 1.0) * 0.20
    # Code presence raises difficulty.
    code_score = features.code_ratio * 0.35 + (0.15 if features.has_code_block else 0.0)
    # Hard keywords raise, easy keywords lower.
    kw_score = (
        min(features.hard_kw_hits * 0.20, 0.50)
        - min(features.easy_kw_hits * 0.12, 0.30)
    )
    # Heavy context raises difficulty.
    ctx_score = min(features.total_context_chars / 40000.0, 1.0) * 0.15

    difficulty = max(0.0, min(1.0, len_score + code_score + kw_score + ctx_score))
    return ScoreResult(
        difficulty=round(difficulty, 4),
        tier=_tier_for(difficulty),
        confidence=round(_confidence(difficulty), 4),
        signals={
            "len_score": round(len_score, 4),
            "code_score": round(code_score, 4),
            "kw_score": round(kw_score, 4),
            "ctx_score": round(ctx_score, 4),
            "hard_kw_hits": features.hard_kw_hits,
            "easy_kw_hits": features.easy_kw_hits,
        },
    )
