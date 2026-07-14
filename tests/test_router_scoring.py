"""Scoring + routing smoke test: crafted prompts → expected tiers/sources/labels.

Run: uv run python tests/test_router_scoring.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src/relay` importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from relay.config import Settings  # noqa: E402
from relay.router.features import extract_features  # noqa: E402
from relay.router.runtime import (  # noqa: E402
    RoutingDecision,
    RoutingHistory,
    _derive_source,
    apply_router,
)
from relay.router.scorer import score_features  # noqa: E402


def _score(text: str, history_msgs: list[dict] | None = None) -> tuple[str, float, float]:
    messages = list(history_msgs or []) + [{"role": "user", "content": text}]
    body = {"messages": messages}
    feats = extract_features(body)
    result = score_features(feats)
    return result.tier, result.confidence, result.difficulty


CASES = [
    # (label, last_user_text, expected_tier_subset)
    ("greeting", "hi", {"c0"}),
    ("simple qa", "what is the capital of france", {"c0", "c1"}),
    ("explain task", "explain photosynthesis in one paragraph", {"c0", "c1"}),
    ("translate", "translate this to english: bonjour", {"c0", "c1"}),
    ("refactor + code", "refactor this auth module for concurrency and add tests\n```python\ndef login(u,p):\n    pass\n```", {"c2", "c3"}),
    ("architecture design", "design a distributed rate limiter architecture with redis and analyze race conditions", {"c2", "c3"}),
    ("long context", "review this\n" + ("lorem ipsum " * 6000), {"c2", "c3"}),  # large_context_floor
]


# ── Source derivation tests ──
SOURCE_CASES = [
    # (label, trail, expected_source)
    ("empty trail", [], "rule_scorer"),
    ("unknown stage", [("unknown", {})], "rule_scorer"),
    ("confidence_gate only", [("confidence_gate", {"from": "c0", "to": "c1", "confidence": 0.3})], "rule_scorer:confidence_gate"),
    ("large_context_floor only", [("large_context_floor", {"from": "c1", "to": "c2", "chars": 70000})], "rule_scorer:large_context_floor"),
    ("sticky only", [("sticky", {"proposed": "c1", "kept": "c2"})], "rule_scorer:sticky"),
    ("complaint_upgrade only", [("complaint_upgrade", {"from": "c0", "to": "c1", "hits": 2})], "rule_scorer:complaint_upgrade"),
    # Priority: complaint_upgrade beats the others even when multiple stages fire.
    ("complaint + confidence", [
        ("confidence_gate", {"from": "c0", "to": "c1", "confidence": 0.3}),
        ("complaint_upgrade", {"from": "c1", "to": "c2", "hits": 2}),
    ], "rule_scorer:complaint_upgrade"),
    # Priority: sticky beats confidence_gate.
    ("sticky + confidence", [
        ("confidence_gate", {"from": "c0", "to": "c1", "confidence": 0.3}),
        ("sticky", {"proposed": "c1", "kept": "c2"}),
    ], "rule_scorer:sticky"),
]


async def _test_apply_router_signals_and_source() -> tuple[bool, str]:
    """Verify apply_router populates signals + derives source from the trail."""
    settings = Settings(router_enabled=True)
    history = RoutingHistory()
    body = {"messages": [{"role": "user", "content": "hi"}], "model": "test-model"}
    decision = await apply_router(body, settings, history)
    if decision is None:
        return False, "apply_router returned None for a simple prompt"
    # signals must be a non-empty dict (scorer sub-scores).
    if not isinstance(decision.signals, dict) or not decision.signals:
        return False, f"signals empty/missing: {decision.signals!r}"
    expected_keys = {"len_score", "code_score", "kw_score", "ctx_score"}
    if not expected_keys.issubset(decision.signals):
        return False, f"signals missing keys: {set(decision.signals)}"
    # source must be a derived string (not the old hardcoded "rule_scorer" alone
    # for a greeting — it should be plain "rule_scorer" since no policy fires).
    if not decision.source.startswith("rule_scorer"):
        return False, f"source not derived: {decision.source!r}"
    # to_record must include signals.
    rec = decision.to_record()
    if "signals" not in rec:
        return False, "to_record() missing 'signals' key"
    return True, f"tier={decision.tier} source={decision.source} signals={list(decision.signals)}"


async def _test_previous_decision_excludes_current() -> tuple[bool, str]:
    """Verify previous_decision skips the current turn's decision."""
    history = RoutingHistory()
    settings = Settings(router_enabled=True)

    # Turn 1: a single "hello" user message.
    d1 = await apply_router(
        {"messages": [{"role": "user", "content": "hello"}], "model": "m"}, settings, history
    )
    # Turn 2: full history resent (stateless protocol) + a complaint follow-up.
    # The first user message is still "hello" → same session_key as turn 1.
    d2 = await apply_router(
        {"messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there, how can I help?"},
            {"role": "user", "content": "that's wrong, try again"},
        ], "model": "m"}, settings, history
    )

    if d1 is None or d2 is None:
        return False, "a decision was None"
    if d1.session_key != d2.session_key:
        return False, f"session keys differ: {d1.session_key} != {d2.session_key}"

    # previous_decision for d2's session, excluding d2, should return d1.
    prev = history.previous_decision(d2.session_key, exclude_id=d2.decision_id)
    if prev is None:
        return False, "previous_decision returned None (expected d1)"
    if prev.decision_id != d1.decision_id:
        return False, f"previous_decision returned wrong decision: {prev.decision_id} != {d1.decision_id}"

    # Without exclusion, it returns d2 (the most recent).
    latest = history.previous_decision(d2.session_key, exclude_id=None)
    if latest is None or latest.decision_id != d2.decision_id:
        return False, "previous_decision(exclude_id=None) didn't return the latest"
    return True, f"prev={prev.decision_id[:8]} latest={latest.decision_id[:8]}"


def main() -> int:
    failures = 0
    print(f"{'label':<22} {'tier':<4} {'conf':>5} {'diff':>5}")
    print("-" * 44)
    for label, text, expected in CASES:
        tier, conf, diff = _score(text)
        ok = tier in expected
        flag = "OK " if ok else "XX "
        print(f"{flag}{label:<20} {tier:<4} {conf:>5.2f} {diff:>5.2f}")
        if not ok:
            failures += 1
    print("-" * 44)

    # ── Source derivation ──
    print("\n--- source derivation ---")
    for label, trail, expected in SOURCE_CASES:
        got = _derive_source(trail)
        ok = got == expected
        flag = "OK " if ok else "XX "
        print(f"{flag}{label:<28} → {got}")
        if not ok:
            print(f"   expected: {expected}")
            failures += 1

    # ── Async: apply_router signals + source ──
    import asyncio
    print("\n--- apply_router: signals + source ---")
    ok, msg = asyncio.run(_test_apply_router_signals_and_source())
    flag = "OK " if ok else "XX "
    print(f"{flag}{msg}")
    if not ok:
        failures += 1

    # ── Async: previous_decision ──
    print("\n--- previous_decision (complaint backfill seam) ---")
    ok, msg = asyncio.run(_test_previous_decision_excludes_current())
    flag = "OK " if ok else "XX "
    print(f"{flag}{msg}")
    if not ok:
        failures += 1

    print("-" * 44)
    print("ALL PASS" if failures == 0 else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
