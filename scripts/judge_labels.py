"""LLM-as-judge: independently label each turn's absolute difficulty (optimal tier).

The core problem this solves: the existing `optimal_tier` labels are *relative*
(derived from the rule scorer's own pick ± 1), so a model trained on them would
just imitate the rule scorer. This script asks a strong LLM to read the user's
message and independently judge "what tier SHOULD this go to?" — producing an
absolute label that can discover the rule scorer's own mistakes.

Input files (same date):
  - ``router-samples-YYYY-MM-DD.jsonl``  (decisions: decision_id, tier, feature_snapshot)
  - ``router-raw-YYYY-MM-DD.jsonl``      (raw user message text, keyed by decision_id)

Output file:
  - ``router-judge-YYYY-MM-DD.jsonl``    (one line per judged decision)

The output is consumed by ``scripts/realign_labels.py`` which merges judge labels
with outcome signals (complaint > judge > error > heuristic) to produce the final
``router-labeled-*.jsonl`` training file.

Usage:
    uv run python scripts/judge_labels.py --date 2026-07-14
    uv run python scripts/judge_labels.py --limit 50 --delay 1.0
    uv run python scripts/judge_labels.py --dry-run
    uv run python scripts/judge_labels.py --skip-judged --model glm-5.2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

# Make the relay package importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from relay.config import get_settings  # noqa: E402
from relay.upstream import UpstreamClient  # noqa: E402

logger = logging.getLogger("judge")

_TIERS = ("c0", "c1", "c2", "c3")

_JUDGE_SYSTEM = """You are a routing difficulty judge for a coding assistant.
Classify the task into one of 4 tiers based on what model capability it needs:

c0 (cheap/fast): greetings, simple factual Q&A, translations, single-word answers, chitchat
c1 (medium): explanations, summaries, simple code edits, straightforward debugging, basic refactoring
c2 (strong): architecture design, multi-file refactoring, complex debugging, SQL optimization, system design
c3 (strongest): novel algorithm design, security audits, production deployment, deep analysis, distributed systems

Consider: task complexity, code presence, context length, and whether it requires multi-step reasoning.
A long conversation context may warrant a higher tier even for a simple-seeming question.

Respond ONLY with a JSON object, no other text:
{"optimal_tier": "c0" | "c1" | "c2" | "c3", "confidence": 0.0-1.0, "reason": "one sentence"}"""


def _build_judge_prompt(user_message: str, features: dict) -> list[dict]:
    """Build the chat messages for the judge LLM call."""
    ctx = (
        f"Context: turn {features.get('turn_index', 0)}, "
        f"{features.get('n_messages', 0)} messages, "
        f"{features.get('total_context_chars', 0)} chars total, "
        f"has_code_block={features.get('has_code_block', False)}, "
        f"hard_kw_hits={features.get('hard_kw_hits', 0)}"
    )
    user_content = (
        f"{ctx}\n\nUser message:\n\"\"\"\n{user_message}\n\"\"\""
    )
    return [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def _parse_judge_response(text: str) -> dict | None:
    """Extract {optimal_tier, confidence, reason} from the judge LLM's response.

    Handles: clean JSON, JSON wrapped in markdown fences, and bare regex fallback.
    """
    # Try direct JSON parse first.
    try:
        data = json.loads(text)
        return _validate_judgment(data)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting from markdown code fences.
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return _validate_judgment(json.loads(m.group(1)))
        except (json.JSONDecodeError, TypeError):
            pass

    # Try finding a bare JSON object in the text.
    m = re.search(r'\{[^{}]*"optimal_tier"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return _validate_judgment(json.loads(m.group(0)))
        except (json.JSONDecodeError, TypeError):
            pass

    # Last resort: regex for the tier value.
    m = re.search(r'"?optimal_tier"?\s*[:=]\s*"?(c[0-3])"', text, re.IGNORECASE)
    if m:
        tier = m.group(1).lower()
        conf_m = re.search(r'"?confidence"?\s*[:=]\s*([0-9.]+)', text)
        conf = float(conf_m.group(1)) if conf_m else 0.5
        return {"optimal_tier": tier, "confidence": conf, "reason": "regex fallback"}

    return None


def _validate_judgment(data: dict) -> dict | None:
    """Validate and normalize a parsed judge response. Returns None if invalid."""
    tier = str(data.get("optimal_tier", "")).strip().lower()
    if tier not in _TIERS:
        return None
    try:
        conf = float(data.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
    except (ValueError, TypeError):
        conf = 0.5
    reason = str(data.get("reason", ""))[:200]
    return {"optimal_tier": tier, "confidence": round(conf, 2), "reason": reason}


def _load_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("%s:%d: invalid JSON (%s); skipped", path, lineno, exc)
    return records


def _load_existing_judged(path: str) -> set[str]:
    """Load decision_ids already judged (for --skip-judged)."""
    ids: set[str] = set()
    if not os.path.exists(path):
        return ids
    for rec in _load_jsonl(path):
        did = rec.get("decision_id")
        if did:
            ids.add(did)
    return ids


async def _judge_one(
    client: UpstreamClient, model: str, messages: list[dict], timeout: float
) -> dict | None:
    """Call the judge model and return parsed verdict, or None on failure."""
    try:
        resp = await client.client.post(
            "/chat/completions",
            json={"model": model, "messages": messages, "stream": False, "temperature": 0},
            timeout=timeout,
        )
        if resp.status_code >= 400:
            logger.warning("judge: upstream %d — skipped", resp.status_code)
            return None
        data = resp.json()
        text = ""
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            pass
        if not text:
            logger.warning("judge: empty response — skipped")
            return None
        return _parse_judge_response(text)
    except Exception as exc:
        logger.warning("judge: API call failed (%s) — skipped", exc)
        return None


async def run_judge(
    date: str,
    capture_dir: str,
    model: str,
    limit: int | None,
    delay: float,
    max_chars: int,
    dry_run: bool,
    skip_judged: bool,
    judge_timeout: float,
) -> dict:
    """Run the LLM-as-judge on all decisions for a date. Returns summary."""
    samples_path = os.path.join(capture_dir, f"router-samples-{date}.jsonl")
    raw_path = os.path.join(capture_dir, f"router-raw-{date}.jsonl")
    judge_path = os.path.join(capture_dir, f"router-judge-{date}.jsonl")

    decisions = _load_jsonl(samples_path)
    raws = {r["decision_id"]: r for r in _load_jsonl(raw_path) if "decision_id" in r}
    already = _load_existing_judged(judge_path) if skip_judged else set()

    # Filter to decisions that have raw content and aren't already judged.
    candidates = [
        d for d in decisions
        if d.get("decision_id") in raws and d["decision_id"] not in already
    ]
    if limit:
        candidates = candidates[:limit]

    total = len(candidates)
    no_raw = len(decisions) - len([d for d in decisions if d.get("decision_id") in raws])
    skipped = len(decisions) - total - no_raw

    print(f"\n{'='*60}")
    print(f"  LLM-as-Judge — {date}")
    print(f"{'='*60}")
    print(f"  Total decisions:     {len(decisions)}")
    print(f"  Without raw content: {no_raw} (cannot judge)")
    print(f"  Already judged:      {len(already)} (skipped)")
    print(f"  To judge:            {total}")
    print(f"  Judge model:         {model}")
    if dry_run:
        print(f"  Mode:                DRY RUN (no API calls)")
    print(f"{'='*60}\n")

    if dry_run:
        for d in candidates[:20]:
            raw = raws.get(d["decision_id"], {})
            msg = (raw.get("user_message") or "")[:80]
            print(f"  {d['decision_id'][:12]}  tier={d.get('tier','?')}  msg=\"{msg}...\"")
        if total > 20:
            print(f"  ... and {total - 20} more")
        return {"total": total, "judged": 0, "failed": 0}

    if not candidates:
        print("  Nothing to judge.")
        return {"total": 0, "judged": 0, "failed": 0}

    # Open upstream client.
    settings = get_settings()
    client = UpstreamClient(settings)
    client.open()

    judged = 0
    failed = 0
    out_lines: list[str] = []

    # Load existing judge records (to preserve on re-run without --skip-judged).
    existing_records = _load_jsonl(judge_path) if os.path.exists(judge_path) else []
    existing_ids = {r["decision_id"] for r in existing_records if "decision_id" in r}

    tier_dist: dict[str, int] = defaultdict(int)

    try:
        for i, dec in enumerate(candidates):
            did = dec["decision_id"]
            raw = raws[did]
            user_msg = (raw.get("user_message") or "")[:max_chars]
            features = dec.get("feature_snapshot") or {}

            messages = _build_judge_prompt(user_msg, features)
            verdict = await _judge_one(client, model, messages, judge_timeout)

            if verdict is None:
                failed += 1
                logger.warning("judge: %s failed (attempt %d/%d)", did[:12], i + 1, total)
            else:
                judged += 1
                tier_dist[verdict["optimal_tier"]] += 1
                rec = {
                    "decision_id": did,
                    "ts_ms": int(time.time() * 1000),
                    "judge_model": model,
                    **verdict,
                }
                out_lines.append(json.dumps(rec, ensure_ascii=False))
                if (judged % 10 == 0) or (i == len(candidates) - 1):
                    print(f"  [{judged + failed}/{total}] judged={judged} failed={failed} "
                          f"latest: tier={verdict['optimal_tier']} conf={verdict['confidence']}")

            if delay > 0 and i < len(candidates) - 1:
                await asyncio.sleep(delay)
    except KeyboardInterrupt:
        print("\n  Interrupted — writing partial results...")
    finally:
        await client.close()

    # Write output (append mode preserves existing records on re-run).
    if out_lines:
        with open(judge_path, "a", encoding="utf-8") as f:
            for line in out_lines:
                f.write(line + "\n")
        logger.info("wrote %d judge labels to %s", len(out_lines), judge_path)

    print(f"\n{'='*60}")
    print(f"  Judge complete — {date}")
    print(f"{'='*60}")
    print(f"  Judged:     {judged}")
    print(f"  Failed:     {failed}")
    print(f"  Tier dist:  {dict(tier_dist)}")
    print(f"  Output:     {judge_path}")
    print(f"{'='*60}")

    return {"total": total, "judged": judged, "failed": failed, "tier_dist": dict(tier_dist)}


def _parse_args() -> argparse.Namespace:
    today = datetime.now().strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(description="LLM-as-judge absolute difficulty labeling for P3.")
    p.add_argument("--date", default=today, help="YYYY-MM-DD (default: today)")
    p.add_argument("--dir", dest="capture_dir", default="logs", help="capture directory")
    p.add_argument("--model", default="gpt-5.5", help="judge model id (default: gpt-5.5)")
    p.add_argument("--limit", type=int, default=None, help="max decisions to judge")
    p.add_argument("--delay", type=float, default=0.5, help="seconds between API calls (default: 0.5)")
    p.add_argument("--max-chars", type=int, default=8000, help="truncate user message (default: 8000)")
    p.add_argument("--timeout", type=float, default=60.0, help="per-call timeout seconds (default: 60)")
    p.add_argument("--dry-run", action="store_true", help="print what would be judged, no API calls")
    p.add_argument("--skip-judged", action="store_true", help="skip already-judged decision_ids")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    summary = asyncio.run(run_judge(
        date=args.date,
        capture_dir=args.capture_dir,
        model=args.model,
        limit=args.limit,
        delay=args.delay,
        max_chars=args.max_chars,
        dry_run=args.dry_run,
        skip_judged=args.skip_judged,
        judge_timeout=args.timeout,
    ))
    return 0 if summary["judged"] > 0 or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
