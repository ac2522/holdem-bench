"""Canonical token counter and truncator, using tiktoken cl100k_base.

One ruler for all models regardless of their native tokenizer; this is the
HELM-style fairness approach. Documented in docs/prompting.md.
"""

from __future__ import annotations

import functools

import tiktoken


@functools.lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the cl100k_base token count for `text`."""
    if not text:
        return 0
    return len(_encoding().encode(text))


def truncate_to_budget(text: str, max_tokens: int) -> str:
    """Truncate `text` to at most `max_tokens` tokens by token prefix."""
    if max_tokens <= 0:
        return ""
    enc = _encoding()
    ids = enc.encode(text)
    if len(ids) <= max_tokens:
        return text
    return enc.decode(ids[:max_tokens])
