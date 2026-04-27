"""Canonical prompt bundle — identical blocks across every provider.

Providers set their own cache-control / caching knobs in their adapter; this
module just carves the prompt into the four cacheable blocks described in
spec §8.3:

    1. ``system_tournament``   — per-tournament system prompt (cache whole run)
    2. ``system_session``      — per-session system prompt (cache per session)
    3. ``user_session_log``    — canonical action log so far (cache per-hand)
    4. ``user_volatile``       — current-decision block; ~500 tok per call
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from holdembench.agents.base import DecisionContext
from holdembench.chat.tokenizer import count_tokens


@dataclass(frozen=True)
class TournamentContext:
    tournament_id: str
    seat: str
    seat_count: int = 9


@dataclass(frozen=True)
class SessionContext:
    session_id: int
    small_blind: int
    big_blind: int
    ante: int
    starting_stack_bb: int
    orbit_budget_tokens: int
    canonical_action_log: str = ""


@dataclass(frozen=True)
class PromptBundle:
    system_tournament: str
    system_session: str
    user_session_log: str
    user_volatile: str
    user_volatile_token_count: int
    prompt_hash: str


# Prompt templates are sent verbatim to providers; line length matches the
# rendered text so breaking lines would change the prompt — noqa is intentional.
_SYSTEM_TOURNAMENT = """You are playing No-Limit Texas Hold'em as {seat} at a {seat_count}-seat table.

Rules: standard NLHE, TDA-compliant.  The table is anonymized — seats are labelled Seat1..Seat{seat_count}; you must NOT claim to know other players' identities or guess their models.  Claims about your own hole cards count as strategic chat and are allowed only within the per-orbit chat budget.

Output protocol:
- Respond with a single JSON object matching the schema:
    {{
      "kind":     "action" | "probe" | "probe_reply",
      "action":   "fold" | "check" | "call" | "raise" | null,
      "amount":   integer chip amount for "raise", else null,
      "message":  optional natural-language chat, <=80 tokens,
      "thinking": optional private reasoning (not shown to opponents)
    }}
- For kind="action" you MUST fill in "action" (and "amount" when raising).
- For kind="probe" or kind="probe_reply" you MUST fill in a non-empty "message".
- Chat is OPTIONAL for kind="action".  Over-budget chat will be rejected and an auto-fold applied on the second violation.
- Parse failures are retried once; a second failure results in auto-fold."""  # noqa: E501


_SYSTEM_SESSION = """Session {session_id}: blinds {sb}/{bb}, ante {ante}, starting stack {ss_bb}bb.
Seat roster: Seat1..Seat{seat_count}  (identities salted; do not infer).
Your seat: {seat}.  Orbit budget: {orbit_bb} tokens (shared across action-attached chat, probes, and probe replies)."""  # noqa: E501


_USER_SESSION_LOG_HEADER = "Canonical action log so far this session:\n"


def render_prompt(
    *,
    tournament: TournamentContext,
    session: SessionContext,
    decision: DecisionContext,
) -> PromptBundle:
    system_tournament = _SYSTEM_TOURNAMENT.format(
        seat=tournament.seat,
        seat_count=tournament.seat_count,
    )
    system_session = _SYSTEM_SESSION.format(
        session_id=session.session_id,
        sb=session.small_blind,
        bb=session.big_blind,
        ante=session.ante,
        ss_bb=session.starting_stack_bb,
        seat_count=tournament.seat_count,
        seat=tournament.seat,
        orbit_bb=session.orbit_budget_tokens,
    )
    # Prefer the per-decision log (kept fresh by the runner); fall back to
    # the per-session legacy slot for callers that haven't been migrated.
    log_body = decision.canonical_action_log or session.canonical_action_log
    body = log_body if log_body else "(no actions yet)"
    user_session_log = _USER_SESSION_LOG_HEADER + body
    user_volatile = _render_volatile(decision)
    volatile_tokens = count_tokens(user_volatile)

    digest = hashlib.sha256()
    for block in (system_tournament, system_session, user_session_log, user_volatile):
        digest.update(block.encode("utf-8"))
        digest.update(b"\x00")
    prompt_hash = f"sha256:{digest.hexdigest()[:16]}"

    return PromptBundle(
        system_tournament=system_tournament,
        system_session=system_session,
        user_session_log=user_session_log,
        user_volatile=user_volatile,
        user_volatile_token_count=volatile_tokens,
        prompt_hash=prompt_hash,
    )


def _render_volatile(d: DecisionContext) -> str:
    chat_block = "\n".join(f"  {line}" for line in d.chat_log) if d.chat_log else "  (none)"
    role_note = (
        "You are answering an OPPONENT'S PROBE (probe_reply).  "
        "Set kind='probe_reply' and provide a message >=20 tokens."
        if d.is_probe_reply
        else "You are making a regular action or optional probe."
    )
    board = " ".join(d.board) if d.board else "(none)"
    return (
        f"Current hand {d.hand_id} on street '{d.street}'.\n"
        f"Your seat: {d.seat}\n"
        f"Stacks: {_fmt_stacks(d.stacks)}\n"
        f"Board: {board}\n"
        f"Your hole cards: {' '.join(d.hole)}\n"
        f"Legal actions: {', '.join(d.legal)}\n"
        f"Chat budget remaining this orbit: {d.budget_remaining} tokens\n"
        f"Chat this hand so far:\n{chat_block}\n"
        f"{role_note}\n"
        f"Respond with a single JSON object — nothing else."
    )


def _fmt_stacks(stacks: dict[str, int]) -> str:
    return ", ".join(f"{s}={v}" for s, v in sorted(stacks.items()))
