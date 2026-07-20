"""P3 ML head: LightGBM-based tier classifier (drop-in for score_features).

This module replaces the rule-based ``score_features`` when a trained model is
available. It loads a LightGBM model saved by ``scripts/train_p3.py`` and
produces the same ``ScoreResult`` the rule scorer outputs, so ``apply_router`` /
``apply_policy`` work unchanged.

Model selection (hot reload, no restart needed):
  1. If the model registry (``models/registry.json``) has an active version,
     that model is used — this is what the self-learning loop promotes to.
  2. Else if ``ROUTER_ML_MODEL_PATH`` is set, that model is used (legacy path).
  3. Else the rule scorer is used.

Hot reload: the loaded ``MLHead`` is cached keyed by ``(path, mtime)`` so a
registry promote (or an overwritten model file) is picked up on the next
routing decision without a restart. ``reload_ml_head()`` / ``invalidate_ml_head()``
force a refresh on demand (used by the reload endpoint / self-learn loop).

Inference is ~0.1ms (numpy array → LightGBM predict). On any load failure the
caller falls back to the rule scorer (see ``runtime.apply_router``) — never
block a request for ML.

The difficulty score is derived from the predicted class probabilities as a
weighted expectation: difficulty = Σ(tier_rank_i × prob_i) / 3, mapping the
[c0..c3] distribution to a continuous [0, 1] score that TIER_BOUNDS can slice.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import TYPE_CHECKING

import lightgbm as lgb
import numpy as np

from .registry import active_model_path, registry_path_for
from .scorer import ScoreResult, TIER_BOUNDS

if TYPE_CHECKING:
    from ..config import Settings
    from .features import FeatureBundle

logger = logging.getLogger("relay.router.ml_head")

# Feature order — must match train_p3.FEATURE_ORDER exactly.
FEATURE_ORDER = [
    "char_len", "word_count", "zh_ratio", "code_ratio",
    "has_code_block", "has_json", "has_yaml", "has_table",
    "easy_kw_hits", "hard_kw_hits", "has_url", "has_file_ref",
    "n_messages", "total_context_chars", "turn_index",
    "complaint_detected", "complaint_hits",
]

_TIERS = ("c0", "c1", "c2", "c3")


class MLHead:
    """Loaded LightGBM model + metadata, producing ScoreResult from features."""

    def __init__(self, model_path: str) -> None:
        self._model = lgb.Booster(model_file=model_path)
        # Load sibling metadata file for feature-order verification.
        meta_path = self._find_meta(model_path)
        self._meta: dict = {}
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self._meta = json.load(f)
        # Verify feature alignment if metadata is available.
        meta_features = self._meta.get("feature_order")
        if meta_features and meta_features != FEATURE_ORDER:
            logger.warning(
                "ml_head: feature order mismatch! model=%s code=%s — "
                "predictions may be incorrect",
                meta_features, FEATURE_ORDER,
            )
        logger.info("ml_head: loaded %s (features=%d)", model_path, len(FEATURE_ORDER))

    @staticmethod
    def _find_meta(model_path: str) -> str | None:
        """Find the sibling .meta.json file written by train_p3.py."""
        base, ext = os.path.splitext(model_path)
        candidates = [
            base + ".meta.json",
            model_path + ".meta.json",
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def score(self, features: "FeatureBundle") -> ScoreResult:
        """Predict tier + confidence + difficulty from the feature bundle.

        This is the drop-in replacement for ``scorer.score_features``.
        """
        snapshot = features.to_snapshot()
        row = np.array([[
            _coerce(snapshot.get(key, 0)) for key in FEATURE_ORDER
        ]], dtype=np.float64)

        proba = self._model.predict(row)[0]  # shape: (4,) for multiclass
        tier_idx = int(np.argmax(proba))
        tier = _TIERS[tier_idx]

        # Difficulty = weighted expectation of tier rank, normalized to [0, 1].
        # c0→0, c1→1/3, c2→2/3, c3→1. This aligns with the rule scorer's
        # TIER_BOUNDS so apply_policy's confidence_gate still works.
        difficulty = float(np.dot(proba, np.arange(4)) / 3.0)

        # Confidence = max probability (how certain the model is about the tier).
        confidence = float(proba[tier_idx])

        # Signals = raw probabilities (useful for debugging / future features).
        signals = {f"proba_{t}": round(float(p), 4) for t, p in zip(_TIERS, proba)}

        return ScoreResult(
            difficulty=round(difficulty, 4),
            tier=tier,
            confidence=round(confidence, 4),
            signals=signals,
        )


def _coerce(val) -> float:
    """Coerce a feature value to float (bool→0/1, None→0)."""
    if isinstance(val, bool):
        return float(val)
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


# ── Hot-reload cache ─────────────────────────────────────────────────────────
# Cache a loaded MLHead keyed by (path, mtime). When the registry active
# pointer flips to a new version (different path) or the model file is
# overwritten (different mtime), the key changes and a fresh head is loaded on
# the next routing decision — no restart required. The lock guards concurrent
# routing threads; loading is idempotent so a benign race is harmless.

_cache_lock = threading.Lock()
_cache: tuple[str, float] | None = None
_cache_head: "MLHead | None" = None


def _load_head(model_path: str) -> "MLHead | None":
    """Load an MLHead for a path, or None on any failure (caller falls back)."""
    if not model_path or not os.path.exists(model_path):
        if model_path:
            logger.warning("ml_head: model not found at %s — using rule scorer", model_path)
        return None
    try:
        return MLHead(model_path)
    except Exception as exc:
        logger.warning("ml_head: failed to load %s (%s) — using rule scorer", model_path, exc)
        return None


def _head_for_path(model_path: str) -> "MLHead | None":
    """Return the cached head for (path, mtime), reloading when either changes."""
    global _cache, _cache_head
    try:
        mtime = os.path.getmtime(model_path)
    except OSError:
        return _load_head(model_path)  # missing file → None (no caching of failure)
    key = (model_path, mtime)
    with _cache_lock:
        if _cache == key:
            return _cache_head
        head = _load_head(model_path)
        _cache = key
        _cache_head = head
        return head


def get_ml_head(model_path: str) -> "MLHead | None":
    """Hot-reloading loader for an explicit model path (legacy / env override).

    Picks up file overwrites (mtime change) on the next call. Returns None if
    loading fails so the caller falls back to ``score_features``.
    """
    return _head_for_path(model_path)


def get_active_ml_head(settings: "Settings") -> "MLHead | None":
    """Resolve the serving ML head: registry active model first, then env path.

    Registry takes precedence so the self-learning loop can promote new models
    without editing env or restarting. ``ROUTER_ML_MODEL_PATH`` remains as a
    manual override / bootstrap path. Returns None → caller uses the rule scorer.
    """
    models_dir = settings.router_models_dir
    reg_path = registry_path_for(models_dir)
    if os.path.exists(reg_path):
        active_path = active_model_path(models_dir)
        if active_path:
            return _head_for_path(active_path)
    if settings.router_ml_model_path:
        return _head_for_path(settings.router_ml_model_path)
    return None


def invalidate_ml_head() -> None:
    """Drop the cached head so the next routing decision reloads from disk."""
    global _cache, _cache_head
    with _cache_lock:
        _cache = None
        _cache_head = None


def reload_ml_head(settings: "Settings") -> "MLHead | None":
    """Force-refresh and return the current serving head (used by reload endpoint)."""
    invalidate_ml_head()
    return get_active_ml_head(settings)
