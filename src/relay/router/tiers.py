"""Tier → model mapping. Built-in preset targets the marketingforce upstream
discovered via GET /v1/models (qwen3-max / deepseek-r1 / gpt-5.4 / claude-opus-4-8).
Override any tier with `ROUTER_TIERS` JSON env.
"""

from __future__ import annotations

DEFAULT_TIERS: dict[str, dict] = {
    # All four support BOTH the OpenAI (/chat/completions) and Anthropic (/messages)
    # endpoints on marketingforce, so the same preset works for opencode (OpenAI
    # path) and ZCode (Anthropic path). If you only use one path you may swap in
    # endpoint-specific models (e.g. qwen3-max is OpenAI-only — cheaper for c0).
    "c0": {"model": "stepfun/step-3.7-flash", "description": "cheap / fast"},
    "c1": {"model": "qwen3.7-plus", "description": "medium"},
    "c2": {"model": "glm-5.2", "description": "strong"},
    "c3": {"model": "gpt-5.5", "description": "strongest"},
}

_TIERS = ("c0", "c1", "c2", "c3")


def tier_rank(tier: str) -> int:
    """Return 0..3 for a tier id; 0 for unknown (cheapest)."""
    return _TIERS.index(tier) if tier in _TIERS else 0


def effective_tiers(user_cfg: dict | None) -> dict[str, dict]:
    """Merge user-provided tier overrides over the built-in defaults."""
    merged = {k: dict(v) for k, v in DEFAULT_TIERS.items()}
    if user_cfg:
        for tier, entry in user_cfg.items():
            tier = str(tier).strip().lower()
            if tier in _TIERS and isinstance(entry, dict):
                merged[tier] = {**merged.get(tier, {}), **entry}
    return merged


def resolve_model(user_cfg: dict | None, tier: str) -> str:
    """Return the model id for a tier, or "" if unset (caller falls back)."""
    entry = effective_tiers(user_cfg).get(tier) or {}
    return str(entry.get("model") or "")
