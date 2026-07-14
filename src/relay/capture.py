"""P3-prep capture: append one training sample per turn to a date-partitioned
JSONL file (router-samples-YYYY-MM-DD.jsonl), plus a sidecar outcome file
(router-outcomes-YYYY-MM-DD.jsonl) written after the upstream call.

This is the data source for a future P3 LightGBM head and is independently useful
for audit / cost / troubleshooting. Mirrors OpenSquilla's
`self_learning/store.write_sample` → `samples-YYYYMMDD.jsonl`.

Decision records are written at routing time (pre-call); outcome records are
written after the upstream response. The two are joined on `decision_id` by the
offline realignment script (`scripts/realign_labels.py`) which populates the
`label` field. Both files are append-only — a write failure never affects the
request path (the caller fires these off in a background task).

Only aggregate feature scalars are written — never raw prompt text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .router.runtime import RoutingDecision

logger = logging.getLogger("relay.capture")

_SCHEMA_VERSION = 1


class CaptureStore:
    """Append-only, date-rotated JSONL of routing decisions + outcomes."""

    def __init__(self, dir_path: str) -> None:
        self._dir = dir_path
        self._date: str | None = None
        self._fp = None   # decision file handle (router-samples-*.jsonl)
        self._ofp = None  # outcome file handle (router-outcomes-*.jsonl)
        self._rfp = None  # raw content file handle (router-raw-*.jsonl)
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        os.makedirs(self._dir, exist_ok=True)

    async def close(self) -> None:
        async with self._lock:
            if self._fp is not None:
                self._fp.close()
                self._fp = None
            if self._ofp is not None:
                self._ofp.close()
                self._ofp = None
            if self._rfp is not None:
                self._rfp.close()
                self._rfp = None
            self._date = None

    async def _ensure_open(self, ts_ms: int) -> None:
        """Rotate all files at the date boundary. Caller holds the lock."""
        today = datetime.fromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")
        if today == self._date and self._fp is not None:
            return
        if self._fp is not None:
            self._fp.close()
        if self._ofp is not None:
            self._ofp.close()
        if self._rfp is not None:
            self._rfp.close()
        samples_path = os.path.join(self._dir, f"router-samples-{today}.jsonl")
        outcomes_path = os.path.join(self._dir, f"router-outcomes-{today}.jsonl")
        raw_path = os.path.join(self._dir, f"router-raw-{today}.jsonl")
        self._fp = open(samples_path, "a", encoding="utf-8")
        self._ofp = open(outcomes_path, "a", encoding="utf-8")
        self._rfp = open(raw_path, "a", encoding="utf-8")
        self._date = today
        logger.info("capture: rotating to %s + %s + %s", samples_path, outcomes_path, raw_path)

    async def write(self, decision: "RoutingDecision") -> None:
        """Append one decision sample line. Rotates to a new file at the date boundary."""
        rec = decision.to_record()
        rec["schema_version"] = _SCHEMA_VERSION
        rec["label"] = None  # filled by offline realignment (outcome join)
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        async with self._lock:
            await self._ensure_open(decision.ts_ms)
            self._fp.write(line)
            self._fp.flush()

    async def write_outcome(self, decision_id: str, outcome: dict) -> None:
        """Append an outcome line to the sidecar outcomes file.

        Called after the upstream response. ``outcome`` must include at least
        ``outcome`` and ``executed_kind`` keys; ``decision_id`` links it back
        to the decision record. Multiple outcome lines may exist for one
        decision_id (e.g. an upstream outcome + a complaint_followup backfill).
        """
        rec = {
            "decision_id": decision_id,
            "ts_ms": int(time.time() * 1000),
            "schema_version": _SCHEMA_VERSION,
            **outcome,
        }
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        async with self._lock:
            await self._ensure_open(rec["ts_ms"])
            self._ofp.write(line)
            self._ofp.flush()

    async def write_passthrough(
        self, session_key: str, client_model: str, reason: str, feature_snapshot: dict | None = None
    ) -> None:
        """Append a minimal decision record when apply_router returned None.

        These are the cases where the router failed to act (timeout/exception).
        Recording them makes passthrough failures visible in the capture log and
        gives P3 training data on the failure mode. ``source`` is "passthrough".
        """
        ts_ms = int(time.time() * 1000)
        rec = {
            "decision_id": uuid.uuid4().hex,
            "ts_ms": ts_ms,
            "session_key": session_key,
            "tier": "",
            "model": client_model,
            "confidence": 0.0,
            "difficulty": 0.0,
            "source": "passthrough",
            "trail": [{"stage": "passthrough", "reason": reason}],
            "feature_snapshot": feature_snapshot or {},
            "signals": {},
            "client_model": client_model,
            "executed_kind": "passthrough",
            "schema_version": _SCHEMA_VERSION,
            "label": None,
        }
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        async with self._lock:
            await self._ensure_open(ts_ms)
            self._fp.write(line)
            self._fp.flush()

    async def write_raw(
        self, decision: "RoutingDecision", user_message: str,
        max_chars: int = 8000,
    ) -> None:
        """Append the last user message text to the raw-content sidecar file.

        This is the input for LLM-as-judge offline labeling: the judge reads the
        user's message and independently assesses absolute difficulty (optimal
        tier c0..c3), breaking the dependence on the rule scorer's own pick.

        Only the **last user message** is stored — not the full conversation
        history, not the model response. Text is truncated to ``max_chars`` to
        bound file size. This file is opt-in (``CAPTURE_RAW_CONTENT=true``) and
        can be deleted after judging; judge labels persist separately.
        """
        text = user_message[:max_chars] if len(user_message) > max_chars else user_message
        rec = {
            "decision_id": decision.decision_id,
            "ts_ms": decision.ts_ms,
            "session_key": decision.session_key,
            "tier": decision.tier,
            "model": decision.model,
            "user_message": text,
            "truncated": len(user_message) > max_chars,
            "schema_version": _SCHEMA_VERSION,
        }
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        async with self._lock:
            await self._ensure_open(decision.ts_ms)
            self._rfp.write(line)
            self._rfp.flush()
