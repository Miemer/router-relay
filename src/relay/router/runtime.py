"""Routing runtime: decision record, in-memory history, bounded apply_router.

`apply_router` is the single entry point the gateway calls per turn. It runs
feature extraction + scoring + policy under a time budget (`anyio.fail_after`),
thread-isolated (`anyio.to_thread.run_sync`) so a future P3 LightGBM/ONNX head
cannot block the event loop. On timeout/error it returns None and the caller
transparently passes the client's original model through — the OpenSquilla
pattern (`engine/runtime.py:_bounded_apply_squilla_router`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .features import extract_features
from .policy import apply_policy
from .scorer import score_features
from .tiers import resolve_model

if TYPE_CHECKING:
    from ..config import Settings

logger = logging.getLogger("relay.router")


def _derive_source(trail: list, base: str = "rule_scorer") -> str:
    """Derive the decision `source` from which policy stages fired.

    Diversifies the captured provenance so P3 training data can distinguish
    rule_scorer base decisions from those adjusted by a policy stage. Priority
    follows the OpenSquilla retrospective label intent: complaint_upgrade and
    sticky are the most informative (they signal under-routing risk), then
    confidence_gate, then large_context_floor.

    ``base`` is ``"rule_scorer"`` (default) or ``"ml_head"`` (when the P3
    LightGBM model replaces ``score_features``). The policy stages that fire
    after scoring are the same regardless of scorer.
    """
    stages = {t[0] for t in trail}
    if "complaint_upgrade" in stages:
        return f"{base}:complaint_upgrade"
    if "sticky" in stages:
        return f"{base}:sticky"
    if "confidence_gate" in stages:
        return f"{base}:confidence_gate"
    if "large_context_floor" in stages:
        return f"{base}:large_context_floor"
    return base


@dataclass
class RoutingDecision:
    decision_id: str
    session_key: str
    ts_ms: int
    tier: str
    model: str
    confidence: float
    difficulty: float
    source: str
    trail: list  # list[tuple[str, dict]]
    feature_snapshot: dict
    client_model: str
    signals: dict = None  # scorer sub-scores (len/code/kw/ctx + kw hits)

    def __post_init__(self) -> None:
        if self.signals is None:
            self.signals = {}

    def summarize(self) -> str:
        trail_names = ",".join(t[0] for t in self.trail) or "-"
        return (
            f"tier={self.tier} model={self.model} conf={self.confidence:.2f} "
            f"diff={self.difficulty:.2f} client={self.client_model or '-'} "
            f"trail={trail_names} sess={self.session_key[:8]}"
        )

    def to_record(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "ts_ms": self.ts_ms,
            "session_key": self.session_key,
            "tier": self.tier,
            "model": self.model,
            "confidence": self.confidence,
            "difficulty": self.difficulty,
            "source": self.source,
            "trail": [{"stage": stage, **payload} for stage, payload in self.trail],
            "feature_snapshot": self.feature_snapshot,
            "signals": self.signals,
            "client_model": self.client_model,
            "executed_kind": "single",  # overridden to "ensemble" when P2 wraps
        }


class RoutingHistory:
    """In-memory per-session tier history (for sticky) + ring buffer of decisions."""

    def __init__(self, sticky_turns: int = 3, ring_size: int = 100) -> None:
        self._sticky_turns = max(1, sticky_turns)
        self._recent: dict[str, deque] = {}
        self._decisions: deque[RoutingDecision] = deque(maxlen=ring_size)

    def recent_tier(self, session_key: str) -> str | None:
        dq = self._recent.get(session_key)
        return dq[-1] if dq else None

    def record(self, session_key: str, tier: str, decision: RoutingDecision) -> None:
        dq = self._recent.setdefault(
            session_key, deque(maxlen=self._sticky_turns)
        )
        dq.append(tier)
        self._decisions.append(decision)

    def recent_decisions(self, limit: int = 20) -> list[RoutingDecision]:
        items = list(self._decisions)
        return list(reversed(items))[:limit] if limit < len(items) else list(reversed(items))

    def previous_decision(
        self, session_key: str, exclude_id: str | None = None
    ) -> RoutingDecision | None:
        """Return the most recent decision for a session (for complaint backfill).

        Used by the retrospective_under_routing label path: when turn N detects
        a complaint, the decision for turn N-1 in the same session is retrieved
        so its label can be marked `under_routed`. ``exclude_id`` is the current
        turn's decision id — it must be skipped because `apply_router` records
        the current decision into the ring buffer *before* returning it.
        """
        for d in reversed(self._decisions):
            if d.session_key == session_key and d.decision_id != exclude_id:
                return d
        return None


class DecisionStore:
    """Optional SQLite persistence for decision records.

    Only aggregate feature scalars are stored — never raw prompt text — mirroring
    OpenSquilla's privacy stance (`router_decision_record.build_trail` sanitization).
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._db = None
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        import aiosqlite  # local import: only needed when persistence is on

        self._db = await aiosqlite.connect(self._path)
        await self._db.execute(
            """CREATE TABLE IF NOT EXISTS router_decisions (
                decision_id TEXT PRIMARY KEY,
                ts_ms INTEGER NOT NULL,
                session_key TEXT,
                tier TEXT,
                model TEXT,
                source TEXT,
                confidence REAL,
                difficulty REAL,
                trail TEXT,
                feature_snapshot TEXT,
                signals TEXT,
                client_model TEXT,
                executed_kind TEXT
            )"""
        )
        # Lightweight migration: add `signals` column to pre-existing tables.
        try:
            await self._db.execute("ALTER TABLE router_decisions ADD COLUMN signals TEXT")
            logger.info("router: decision store migrated (added signals column)")
        except Exception:
            pass  # column already exists — expected on fresh tables
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def write(self, decision: RoutingDecision) -> None:
        if self._db is None:
            return
        rec = decision.to_record()
        try:
            async with self._lock:
                await self._db.execute(
                    """INSERT OR REPLACE INTO router_decisions
                       (decision_id, ts_ms, session_key, tier, model, source,
                        confidence, difficulty, trail, feature_snapshot, signals,
                        client_model, executed_kind)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        rec["decision_id"], rec["ts_ms"], rec["session_key"],
                        rec["tier"], rec["model"], rec["source"], rec["confidence"],
                        rec["difficulty"], json.dumps(rec["trail"]),
                        json.dumps(rec["feature_snapshot"]), json.dumps(rec["signals"]),
                        rec["client_model"], rec["executed_kind"],
                    ),
                )
                await self._db.commit()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("router: decision store write failed: %s", exc)


async def apply_router(body: dict, settings: "Settings", history: RoutingHistory) -> RoutingDecision | None:
    """Classify the turn and pick a tier/model. None → caller passthrough."""
    try:
        features = extract_features(body)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("router: feature extraction failed: %s", exc)
        return None

    session_key = features.session_key
    client_model = str(body.get("model") or "")

    try:
        # Bounded execution: run the (sync) scorer in a worker thread with a
        # timeout. For the rule scorer this is sub-millisecond; for the P3 ML
        # head it's ~0.1ms (LightGBM predict). Timeout/exception → passthrough.
        #
        # Scorer selection (hot reload): the registry's active model is used
        # when present, else ROUTER_ML_MODEL_PATH, else the rule scorer. The
        # registry active pointer is read on every call, so a self-learn
        # promote is picked up on the next decision without a restart. On any
        # load failure, falls back to the rule scorer transparently (never
        # block a request for ML).
        from .ml_head import get_active_ml_head
        ml_head = get_active_ml_head(settings)

        if ml_head is not None:
            score = await asyncio.wait_for(
                asyncio.to_thread(ml_head.score, features),
                timeout=settings.router_timeout_seconds,
            )
            scorer_base = "ml_head"
        else:
            score = await asyncio.wait_for(
                asyncio.to_thread(score_features, features),
                timeout=settings.router_timeout_seconds,
            )
            scorer_base = "rule_scorer"
    except asyncio.TimeoutError:
        logger.warning(
            "router: scoring timed out after %ss; passthrough",
            settings.router_timeout_seconds,
        )
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("router: scoring failed: %s", exc)
        return None

    final_tier, trail = apply_policy(score, features, session_key, history, settings)
    model = resolve_model(settings.router_tiers, final_tier)

    decision = RoutingDecision(
        decision_id=uuid.uuid4().hex,
        session_key=session_key,
        ts_ms=int(time.time() * 1000),
        tier=final_tier,
        model=model or client_model,
        confidence=score.confidence,
        difficulty=score.difficulty,
        source=_derive_source(trail, base=scorer_base),
        trail=trail,
        feature_snapshot=features.to_snapshot(),
        client_model=client_model,
        signals=score.signals,
    )
    history.record(session_key, final_tier, decision)
    return decision
