"""Offline label realignment: join decision + outcome capture → labeled P3 data.

Reads the date-partitioned capture files produced by ``relay.capture``:
  - ``router-samples-YYYY-MM-DD.jsonl``  (decisions, written at routing time)
  - ``router-outcomes-YYYY-MM-DD.jsonl``  (outcomes, written after the upstream call)

Joins them on ``decision_id`` and writes ``router-labeled-YYYY-MM-DD.jsonl``
where each line is the original decision record with the ``label`` field
populated (``under_routed`` / ``appropriate`` / ``over_routed``) plus an
``optimal_tier`` derived from the label and the actual tier.

Label priority (strongest signal first):
  1. ``complaint_followup`` outcome  → ``under_routed``  (user explicitly complained)
  2. ``upstream_error`` outcome      → ``under_routed``  (model couldn't serve the request)
  3. ``label_hint`` from the upstream outcome (token-efficiency heuristic)
  4. ``None``  (no outcome record → cannot label; left for future realignment)

Usage:
    uv run python scripts/realign_labels.py --date 2026-07-13
    uv run python scripts/realign_labels.py                # today
    uv run python scripts/realign_labels.py --dir logs --date 2026-07-13 --dry-run

This script is read-only against the capture files; it never modifies them.
The labeled file is a separate output suitable as P3 LightGBM training input.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime

# Run without requiring the relay package on sys.path (this script is standalone).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logger = logging.getLogger("realign")

_TIERS = ("c0", "c1", "c2", "c3")


def _clamp_tier(tier: str) -> str:
    return tier if tier in _TIERS else "c0"


def _tier_idx(tier: str) -> int:
    """Return 0..3 for a tier id; 0 for unknown."""
    return _TIERS.index(_clamp_tier(tier))


def _bump_tier(tier: str, delta: int) -> str:
    """Return tier shifted by delta, clamped to c0..c3."""
    idx = _tier_idx(tier)
    return _TIERS[max(0, min(len(_TIERS) - 1, idx + delta))]


def _derive_label_and_optimal(
    outcomes: list[dict], judge: dict | None, actual_tier: str
) -> tuple[str | None, str | None, str]:
    """Derive (label, optimal_tier, source) from outcome + judge signals.

    Priority (strongest first):
    1. ``complaint_followup`` → ``under_routed``; optimal = max(judge, actual+1)
    2. judge label → use judge's ``optimal_tier`` directly (absolute, independent
       of the rule scorer's pick — this is the key P3 improvement)
    3. ``upstream_error`` → ``under_routed``; optimal = actual+1
    4. ``label_hint`` from outcomes → ±1 heuristic (weakest)
    5. None → cannot label
    """
    has_complaint = any(o.get("outcome") == "complaint_followup" for o in outcomes)
    judge_optimal = judge.get("optimal_tier") if judge else None

    # 1. Complaint (explicit user feedback) — always wins for the label.
    if has_complaint:
        bumped = _bump_tier(actual_tier, 1)
        if judge_optimal and _tier_idx(judge_optimal) > _tier_idx(bumped):
            optimal = judge_optimal  # judge says even higher — respect it
        else:
            optimal = bumped
        return "under_routed", optimal, "complaint_followup"

    # 2. Judge label (absolute difficulty, independent of rule scorer).
    if judge_optimal:
        actual_idx = _tier_idx(actual_tier)
        judge_idx = _tier_idx(judge_optimal)
        if judge_idx > actual_idx:
            label = "under_routed"
        elif judge_idx < actual_idx:
            label = "over_routed"
        else:
            label = "appropriate"
        return label, judge_optimal, "judge"

    # 3. Upstream error (model couldn't serve the request).
    has_error = any(
        o.get("outcome") == "upstream_error" or (o.get("upstream_status") or 0) >= 400
        for o in outcomes
    )
    if has_error:
        return "under_routed", _bump_tier(actual_tier, 1), "upstream_error"

    # 4. Label hint (token-efficiency heuristic).
    hints = [o.get("label_hint") for o in outcomes if o.get("label_hint")]
    if hints:
        hint = hints[0]
        if hint == "under_routed":
            return hint, _bump_tier(actual_tier, 1), "label_hint"
        if hint == "over_routed":
            return hint, _bump_tier(actual_tier, -1), "label_hint"
        return hint, _clamp_tier(actual_tier), "label_hint"

    return None, None, "none"


def _load_jsonl(path: str) -> list[dict]:
    """Read a JSONL file, skipping blank/invalid lines."""
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


def realign(date: str, capture_dir: str, dry_run: bool = False) -> dict:
    """Join decisions + outcomes + judge labels for a date and write the labeled file.

    Reads three sidecar files (all optional — missing judge file means no
    absolute labels; missing outcomes means no heuristic labels):
      - ``router-samples-*.jsonl``  (decisions)
      - ``router-outcomes-*.jsonl``  (outcomes)
      - ``router-judge-*.jsonl``     (LLM-as-judge absolute labels)

    Returns a summary dict with counts.
    """
    samples_path = os.path.join(capture_dir, f"router-samples-{date}.jsonl")
    outcomes_path = os.path.join(capture_dir, f"router-outcomes-{date}.jsonl")
    judge_path = os.path.join(capture_dir, f"router-judge-{date}.jsonl")
    labeled_path = os.path.join(capture_dir, f"router-labeled-{date}.jsonl")

    decisions = _load_jsonl(samples_path)
    outcomes = _load_jsonl(outcomes_path)
    judges = _load_jsonl(judge_path)

    # Index outcomes by decision_id (one decision may have multiple outcome records).
    by_decision: dict[str, list[dict]] = defaultdict(list)
    for o in outcomes:
        did = o.get("decision_id")
        if did:
            by_decision[did].append(o)

    # Index judge labels by decision_id (one per decision; last wins on dup).
    by_judge: dict[str, dict] = {}
    for j in judges:
        did = j.get("decision_id")
        if did:
            by_judge[did] = j

    labeled_count = 0
    unlabeled_count = 0
    label_dist: dict[str, int] = defaultdict(int)
    tier_dist: dict[str, int] = defaultdict(int)
    source_dist: dict[str, int] = defaultdict(int)
    label_source_dist: dict[str, int] = defaultdict(int)

    out_lines: list[str] = []
    for dec in decisions:
        did = dec.get("decision_id")
        outs = by_decision.get(did, [])
        judge = by_judge.get(did)
        actual_tier = dec.get("tier", "c0")

        label, optimal, label_source = _derive_label_and_optimal(outs, judge, actual_tier)

        labeled = dict(dec)
        labeled["label"] = label
        labeled["optimal_tier"] = optimal
        labeled["label_source"] = label_source  # which signal won
        # Attach compact summaries of the signals that informed the label.
        labeled["outcome_summary"] = [
            {
                "outcome": o.get("outcome"),
                "executed_kind": o.get("executed_kind"),
                "latency_ms": o.get("latency_ms"),
                "label_hint": o.get("label_hint"),
                "upstream_status": o.get("upstream_status"),
            }
            for o in outs
        ]
        if judge:
            labeled["judge_summary"] = {
                "optimal_tier": judge.get("optimal_tier"),
                "confidence": judge.get("confidence"),
                "reason": judge.get("reason"),
                "judge_model": judge.get("judge_model"),
            }

        if label is not None:
            labeled_count += 1
            label_dist[label] += 1
            label_source_dist[label_source] += 1
        else:
            unlabeled_count += 1
        tier_dist[dec.get("tier", "")] += 1
        source_dist[dec.get("source", "")] += 1
        out_lines.append(json.dumps(labeled, ensure_ascii=False))

    if not dry_run and out_lines:
        with open(labeled_path, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines))
            f.write("\n")
        logger.info("wrote %s", labeled_path)

    summary = {
        "date": date,
        "decisions": len(decisions),
        "outcomes": len(outcomes),
        "judges": len(judges),
        "labeled": labeled_count,
        "unlabeled": unlabeled_count,
        "label_distribution": dict(label_dist),
        "label_source_distribution": dict(label_source_dist),
        "tier_distribution": dict(tier_dist),
        "source_distribution": dict(source_dist),
    }
    return summary


def _parse_args() -> argparse.Namespace:
    today = datetime.now().strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(description="Realign router capture labels for P3 training.")
    p.add_argument("--date", default=today, help="YYYY-MM-DD (default: today)")
    p.add_argument("--dir", dest="capture_dir", default="logs", help="capture directory (default: logs)")
    p.add_argument("--dry-run", action="store_true", help="print summary only, don't write file")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    summary = realign(args.date, args.capture_dir, dry_run=args.dry_run)

    print(f"\n{'='*60}")
    print(f"  P3 label realignment — {summary['date']}")
    print(f"{'='*60}")
    print(f"  Decisions:      {summary['decisions']}")
    print(f"  Outcomes:       {summary['outcomes']}")
    print(f"  Judge labels:   {summary['judges']}")
    print(f"  Labeled:        {summary['labeled']}")
    print(f"  Unlabeled:      {summary['unlabeled']}")
    print(f"  Label dist:     {summary['label_distribution']}")
    print(f"  Label sources:  {summary['label_source_distribution']}")
    print(f"  Tier dist:      {summary['tier_distribution']}")
    print(f"  Source dist:    {summary['source_distribution']}")
    if summary['decisions'] and summary['labeled']:
        pct = 100.0 * summary['labeled'] / summary['decisions']
        print(f"  Label rate:     {pct:.1f}%")
    print(f"{'='*60}")

    # Exit code: 0 if at least one decision was labeled, else 1.
    return 0 if summary["labeled"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
