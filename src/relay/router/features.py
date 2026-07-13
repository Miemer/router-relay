"""Handcrafted per-turn feature extraction (no embeddings, no I/O).

The feature set mirrors the *handcrafted* channel of OpenSquilla's SquillaRouter
(`squilla_router/runtime_src/src/router/inference/features.py:extract_handcrafted`)
but is far smaller and dependency-free. It is the substrate a P3 LightGBM head will
train on, so keep the fields stable and aggregate-only (never prompt text).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha1

# Keyword buckets. Lowercased substring match against the last user message.
# easy_kw → simple Q&A / chitchat (lowers difficulty); hard_kw → engineering
# tasks (raises difficulty).
_EASY_KEYWORDS = (
    "explain", "summarize", "summarise", "translate", "list", "define",
    "what is", "what's", "meaning", "hi", "hello", "hey", "thanks", "thank",
    "please", "greet",
)
_HARD_KEYWORDS = (
    "refactor", "architecture", "design", "optimize", "optimise", "debug",
    "analyze", "analyse", "compare", "implement", "production", "security",
    "deploy", "algorithm", "performance", "concurrency", "async",
    "race condition", "deadlock", "scale", "distributed", "migration",
    "integration", "audit", "review", "benchmark", "test", "tests",
)

# Complaint cues: the user is unhappy with the previous answer. Detected on the
# CURRENT user message; recorded per-turn so offline realignment can mark the
# PREVIOUS turn as under-routed (OpenSquilla's retrospective_under_routing).
_COMPLAINT_EN = [
    re.compile(r"\bthat'?s wrong\b", re.IGNORECASE),
    re.compile(r"\bthat is wrong\b", re.IGNORECASE),
    re.compile(r"\bnot right\b", re.IGNORECASE),
    re.compile(r"\btry again\b", re.IGNORECASE),
    re.compile(r"\bredo\b", re.IGNORECASE),
    re.compile(r"\bincorrect\b", re.IGNORECASE),
    re.compile(r"\bdidn'?t work\b", re.IGNORECASE),
    re.compile(r"\bdoesn'?t work\b", re.IGNORECASE),
    re.compile(r"\bbad answer\b", re.IGNORECASE),
    re.compile(r"\bwrong answer\b", re.IGNORECASE),
    re.compile(r"\byou'?re wrong\b", re.IGNORECASE),
    re.compile(r"\bnot what i (asked|wanted)\b", re.IGNORECASE),
]
_COMPLAINT_ZH = ("不对", "错了", "重做", "重来", "答非所问", "不要这样", "不是这样")

_URL_RE = re.compile(r"https?://", re.IGNORECASE)
_FILE_RE = re.compile(
    r"\b[\w./-]+\.(py|js|ts|tsx|jsx|go|rs|java|rb|md|json|ya?ml|toml|sql|sh|css|html|c|cpp|h)\b",
    re.IGNORECASE,
)
_CJK_START = 0x4E00
_CJK_END = 0x9FFF
# Compiled keyword patterns (filled lazily by _count_keyword_hits).
_KW_CACHE: dict[str, "re.Pattern[str]"] = {}


@dataclass
class FeatureBundle:
    """Aggregate per-turn features. No prompt text is stored here."""

    session_key: str
    char_len: int
    word_count: int
    zh_ratio: float
    code_ratio: float
    has_code_block: bool
    has_json: bool
    has_yaml: bool
    has_table: bool
    easy_kw_hits: int
    hard_kw_hits: int
    has_url: bool
    has_file_ref: bool
    n_messages: int
    total_context_chars: int
    turn_index: int
    # Complaint signal (current user message) — feeds complaint_upgrade policy
    # and offline label realignment for the previous turn.
    complaint_detected: bool
    complaint_hits: int

    def to_snapshot(self) -> dict:
        """Return the aggregate values for the decision record (no prompt text)."""
        return {
            "char_len": self.char_len,
            "word_count": self.word_count,
            "zh_ratio": round(self.zh_ratio, 3),
            "code_ratio": round(self.code_ratio, 3),
            "has_code_block": self.has_code_block,
            "has_json": self.has_json,
            "has_yaml": self.has_yaml,
            "has_table": self.has_table,
            "easy_kw_hits": self.easy_kw_hits,
            "hard_kw_hits": self.hard_kw_hits,
            "has_url": self.has_url,
            "has_file_ref": self.has_file_ref,
            "n_messages": self.n_messages,
            "total_context_chars": self.total_context_chars,
            "turn_index": self.turn_index,
            "complaint_detected": self.complaint_detected,
            "complaint_hits": self.complaint_hits,
        }


def _message_text(content: object) -> str:
    """Flatten a message `content` (str or multimodal list) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


def _last_user_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return _message_text(msg.get("content"))
    return ""


def _first_user_text(messages: list) -> str:
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return _message_text(msg.get("content"))
    return ""


def derive_session_key(body: dict) -> str:
    """Stable per-conversation id from the first user message.

    opencode resends the full history each turn (stateless OpenAI protocol), so
    hashing the first user turn yields a stable key without client cooperation.
    This is what makes per-session sticky routing work.
    """
    text = _first_user_text(body.get("messages") or [])
    if not text:
        return "default"
    return sha1(text.encode("utf-8")).hexdigest()[:16]


def _code_chars(text: str) -> tuple[int, bool]:
    """Return (chars inside fenced code blocks, has_code_block)."""
    if "```" not in text:
        return 0, False
    parts = text.split("```")
    # Segments at odd indices sit between fences → code.
    inside = sum(len(p) for p in parts[1::2])
    return inside, True


def _count_keyword_hits(text_lower: str, keywords: tuple[str, ...]) -> int:
    # Word-boundary match so short keywords like "hi"/"list"/"test" don't fire
    # inside "this"/"architecture"/"latest". Multi-word keywords (e.g. "what is")
    # still work because spaces sit between word boundaries.
    count = 0
    for kw in keywords:
        rx = _KW_CACHE.get(kw)
        if rx is None:
            rx = re.compile(r"\b" + re.escape(kw) + r"\b")
            _KW_CACHE[kw] = rx
        if rx.search(text_lower):
            count += 1
    return count


def _detect_complaint(text: str) -> tuple[bool, int]:
    """Return (detected, hit_count) for complaint cues in the user message."""
    hits = sum(1 for rx in _COMPLAINT_EN if rx.search(text))
    hits += sum(1 for zh in _COMPLAINT_ZH if zh in text)
    return hits > 0, hits


def extract_features(body: dict) -> FeatureBundle:
    """Build the feature bundle from a chat-completion request body."""
    raw_messages = body.get("messages") or []
    messages = [m for m in raw_messages if isinstance(m, dict)]

    last_user = _last_user_text(messages)
    text_lower = last_user.lower()

    char_len = len(last_user)
    word_count = len(last_user.split())
    cjk = sum(1 for ch in last_user if _CJK_START <= ord(ch) <= _CJK_END)
    alpha = sum(1 for ch in last_user if ch.isalpha())
    zh_ratio = (cjk / alpha) if alpha else 0.0

    code_inside, has_code_block = _code_chars(last_user)
    code_ratio = (code_inside / char_len) if char_len else 0.0

    has_json = ("```json" in text_lower) or ('{"' in last_user and "}" in last_user)
    has_yaml = "```yaml" in text_lower
    has_table = last_user.count("|") >= 4

    easy_kw_hits = _count_keyword_hits(text_lower, _EASY_KEYWORDS)
    hard_kw_hits = _count_keyword_hits(text_lower, _HARD_KEYWORDS)

    has_url = bool(_URL_RE.search(last_user))
    has_file_ref = bool(_FILE_RE.search(last_user))

    total_context_chars = sum(len(_message_text(m.get("content"))) for m in messages)
    n_messages = len(messages)
    turn_index = n_messages // 2

    complaint_detected, complaint_hits = _detect_complaint(last_user)

    return FeatureBundle(
        session_key=derive_session_key(body),
        char_len=char_len,
        word_count=word_count,
        zh_ratio=zh_ratio,
        code_ratio=code_ratio,
        has_code_block=has_code_block,
        has_json=has_json,
        has_yaml=has_yaml,
        has_table=has_table,
        easy_kw_hits=easy_kw_hits,
        hard_kw_hits=hard_kw_hits,
        has_url=has_url,
        has_file_ref=has_file_ref,
        n_messages=n_messages,
        total_context_chars=total_context_chars,
        turn_index=turn_index,
        complaint_detected=complaint_detected,
        complaint_hits=complaint_hits,
    )
