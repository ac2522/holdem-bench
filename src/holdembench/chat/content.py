"""Content analysis + validation for chat messages.

Design note: card claims are ALLOWED (core to poker bluffing). They are
logged to analysis/card_claims.jsonl for calibration metrics post-hoc.
Identity-leakage strings are also allowed but counted.
Prompt-injection / tool-call / HTML attempts ARE rejected as malformed.
"""
from __future__ import annotations

import re

_CARD_RANK_WORDS = (
    r"ace|king|queen|jack|ten|nine|eight|seven|six|five|four|three|two|deuce"
)
_CARD_SUIT_WORDS = r"hearts?|diamonds?|clubs?|spades?"

_CARD_CLAIM_PATTERNS = (
    re.compile(r"[AKQJT2-9][cdhs]"),                                        # "Ah", "Ks", "AhKs"
    re.compile(r"\b[AKQJT2-9]\s?[♠♥♦♣]", re.UNICODE),                       # "A♠"
    re.compile(rf"\b({_CARD_RANK_WORDS})\s+of\s+({_CARD_SUIT_WORDS})\b", re.IGNORECASE),
    re.compile(r"\brocket[s-]?rocket\b|\bpocket\s+(kings|aces|queens|jacks)\b", re.IGNORECASE),
    re.compile(r"\brockets\b", re.IGNORECASE),
)

_IDENTITY_PATTERNS = (
    re.compile(r"\bas an? (ai|language model|llm)\b", re.IGNORECASE),
    re.compile(
        r"\b(I am|I'm)\s+"
        r"(claude|gpt|gemini|llama|grok|deepseek|qwen|kimi|yi|glm|ernie)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bas (claude|gpt|gemini|llama|grok)\b", re.IGNORECASE),
)

_INJECTION_PATTERNS = (
    re.compile(r"<\s*[a-z][a-z0-9-]*\b[^>]*>", re.IGNORECASE),   # any HTML-like tag
    re.compile(r"\bfunction_call\b|\btool_call\b"),
    re.compile(r"```"),                                          # code fences
)


class ContentRejection(ValueError):
    """Raised when a message fails content validation (injection attempt, HTML)."""


def detect_card_claims(text: str) -> list[str]:
    """Return all card-claim matches (empty list if none)."""
    hits: list[str] = []
    for pat in _CARD_CLAIM_PATTERNS:
        hits.extend(m.group(0) for m in pat.finditer(text))
    return hits


def detect_identity_leaks(text: str) -> list[str]:
    hits: list[str] = []
    for pat in _IDENTITY_PATTERNS:
        hits.extend(m.group(0) for m in pat.finditer(text))
    return hits


def validate_content(text: str) -> None:
    """Reject structural injection attempts. Raises ContentRejection."""
    for pat in _INJECTION_PATTERNS:
        if pat.search(text):
            msg = f"message contains disallowed html/tag/tool-call pattern: {pat.pattern}"
            raise ContentRejection(msg)
