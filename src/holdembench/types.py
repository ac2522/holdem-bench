"""Canonical type aliases shared across the package.

Imported by `events.schema`, `engine.validator`, `chat.protocol`, and
`agents.base`.  Keep narrow: only literal/string aliases live here.  Non-literal
abstractions (e.g. RawDecision) stay in their domain module.
"""

from __future__ import annotations

from typing import Literal

type ActionName = Literal["fold", "check", "call", "raise"]
type ActionKind = Literal["action", "probe", "probe_reply"]
type ChatKind = Literal["action", "probe", "probe_reply"]
type Street = Literal["preflop", "flop", "turn", "river"]

__all__ = ["ActionName", "ActionKind", "ChatKind", "Street"]
