"""Chat protocol: per-orbit token budgets, probe mechanics, folded-silencing.

Enforcement layer that sits between agents and the event log.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from holdembench.chat.tokenizer import count_tokens

ChatKind = Literal["action", "probe", "probe_reply"]

MIN_PROBE_TOKENS = 10
MIN_PROBE_REPLY_TOKENS = 20
MAX_PLAYERS_FOR_PROBE_BAN = 2
MAX_PROBES_PER_HAND = 2


class ChatRuleViolation(ValueError):
    """Raised when a chat event would break the protocol."""


@dataclass
class _SeatState:
    budget_remaining: int
    probes_this_hand: int = 0
    folded_this_hand: bool = False
    last_was_probe: bool = False


@dataclass
class ChatProtocol:
    """Per-orbit chat budgets + probe and content rules.

    Lifecycle:
        start_orbit() -> start_hand(in_hand) -> spend(...) -> mark_folded(...) -> ...
        -> end_hand() -> (repeat hand) -> end_orbit() -> start_orbit() ...
    """

    seats: tuple[str, ...]
    budget_per_orbit: int = 400
    per_action_cap: int = 80
    _state: dict[str, _SeatState] = field(init=False)
    _in_hand: set[str] = field(init=False)

    def __post_init__(self) -> None:
        self._state = {}
        self._in_hand = set()
        self.start_orbit()

    def start_orbit(self) -> None:
        self._state = {
            s: _SeatState(budget_remaining=self.budget_per_orbit) for s in self.seats
        }

    def start_hand(self, in_hand: set[str]) -> None:
        for _seat, st in self._state.items():
            st.probes_this_hand = 0
            st.folded_this_hand = False
            st.last_was_probe = False
        self._in_hand = set(in_hand)

    def mark_folded(self, seat: str) -> None:
        self._state[seat].folded_this_hand = True
        self._in_hand.discard(seat)

    def budget_remaining(self, seat: str) -> int:
        return self._state[seat].budget_remaining

    def spend(self, seat: str, message: str, kind: ChatKind) -> int:
        st = self._state[seat]
        if st.folded_this_hand:
            raise ChatRuleViolation(f"{seat} is folded for the hand; cannot chat")

        tokens = count_tokens(message)
        if tokens > self.per_action_cap:
            raise ChatRuleViolation(
                f"message exceeds per-action cap of {self.per_action_cap} tokens"
            )

        if kind == "probe":
            if len(self._in_hand) <= MAX_PLAYERS_FOR_PROBE_BAN:
                raise ChatRuleViolation("probe not allowed when only 2 players remain")
            if st.probes_this_hand >= MAX_PROBES_PER_HAND:
                raise ChatRuleViolation("max 2 probes per player per hand")
            if st.last_was_probe:
                raise ChatRuleViolation("cannot probe twice in a row")
            if tokens < MIN_PROBE_TOKENS:
                raise ChatRuleViolation(
                    f"probe requires min {MIN_PROBE_TOKENS} tokens, got {tokens}"
                )

        if kind == "probe_reply" and tokens < MIN_PROBE_REPLY_TOKENS:
            raise ChatRuleViolation(
                f"probe_reply requires min {MIN_PROBE_REPLY_TOKENS} tokens"
            )

        if tokens > st.budget_remaining:
            raise ChatRuleViolation(
                f"budget remaining ({st.budget_remaining}) insufficient for {tokens} tokens"
            )

        st.budget_remaining -= tokens
        if kind == "probe":
            st.probes_this_hand += 1
            st.last_was_probe = True
        else:
            st.last_was_probe = False

        return tokens
