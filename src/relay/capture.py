"""P3-prep capture: append one training sample per turn to a date-partitioned
JSONL file (router-samples-YYYY-MM-DD.jsonl).

This is the data source for a future P3 LightGBM head and is independently useful
for audit / cost / troubleshooting. Mirrors OpenSquilla's
`self_learning/store.write_sample` → `samples-YYYYMMDD.jsonl`.

Only aggregate feature scalars are written — never raw prompt text. Each line is
self-contained and append-only, so a write failure never affects the request path
(the caller fires this off in a background task).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .router.runtime import RoutingDecision

logger = logging.getLogger("relay.capture")

_SCHEMA_VERSION = 1


class CaptureStore:
    """Append-only, date-rotated JSONL of routing decisions."""

    def __init__(self, dir_path: str) -> None:
        self._dir = dir_path
        self._date: str | None = None
        self._fp = None
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        os.makedirs(self._dir, exist_ok=True)

    async def close(self) -> None:
        async with self._lock:
            if self._fp is not None:
                self._fp.close()
                self._fp = None
                self._date = None

    async def write(self, decision: "RoutingDecision") -> None:
        """Append one sample line. Rotates to a new file at the date boundary."""
        rec = decision.to_record()
        rec["schema_version"] = _SCHEMA_VERSION
        rec["label"] = None  # filled by offline realignment later
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        async with self._lock:
            today = datetime.fromtimestamp(decision.ts_ms / 1000).strftime("%Y-%m-%d")
            if today != self._date:
                if self._fp is not None:
                    self._fp.close()
                path = os.path.join(self._dir, f"router-samples-{today}.jsonl")
                self._fp = open(path, "a", encoding="utf-8")
                self._date = today
                logger.info("capture: rotating to %s", path)
            self._fp.write(line)
            self._fp.flush()
