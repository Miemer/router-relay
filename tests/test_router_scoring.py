"""Scoring smoke test: crafted prompts → expected tiers.

Run: uv run python tests/test_router_scoring.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src/relay` importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from relay.router.features import extract_features  # noqa: E402
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
    print("ALL PASS" if failures == 0 else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
