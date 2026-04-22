"""Thin wrapper around pokerkit.NoLimitTexasHoldem.

We never subclass or monkey-patch pokerkit; this module exposes just the
surface the harness needs (next_actor, apply_*, pot state, hand/end flags).

Pokerkit API notes (v0.7.3):
- ``State.actor_index`` (int | None) — current actor seat, or None.
- ``State.actor_indices`` (deque[int]) — queue of upcoming actors.
- ``State.bets`` (list[int]) — current street bets per seat.
- ``State.total_pot_amount`` (int) — total pot including bets and collected
  pots (replaces sum(bets) + sum(pots) — pots is a generator in 0.7.3).
- ``State.status`` (bool) — True while hand is in progress, False when over.
- ``Automation.HOLE_DEALING`` — auto-deals hole cards so actors are available
  immediately after construction without a separate deal step.
"""
from __future__ import annotations

from typing import cast

from pokerkit import Automation, NoLimitTexasHoldem, State

from holdembench.engine.config import TableConfig

# Automation is a StrEnum; pyright infers the tuple as tuple[str, ...] without
# the explicit cast.  The cast is safe — all members are genuine Automation values.
_AUTOMATIONS: tuple[Automation, ...] = cast(
    "tuple[Automation, ...]",
    (
        Automation.ANTE_POSTING,
        Automation.BET_COLLECTION,
        Automation.BLIND_OR_STRADDLE_POSTING,
        Automation.CARD_BURNING,
        Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
        Automation.HAND_KILLING,
        Automation.CHIPS_PUSHING,
        Automation.CHIPS_PULLING,
        Automation.HOLE_DEALING,
    ),
)


class Table:
    """Owns a pokerkit State. Callers drive the hand turn-by-turn."""

    def __init__(self, config: TableConfig) -> None:
        self._config = config
        self._state: State = NoLimitTexasHoldem.create_state(
            automations=_AUTOMATIONS,
            ante_trimming_status=True,
            raw_antes=config.ante,
            raw_blinds_or_straddles=(config.small_blind, config.big_blind),
            min_bet=config.big_blind,
            raw_starting_stacks=config.starting_stacks,
            player_count=config.seat_count,
        )

    @property
    def seat_count(self) -> int:
        return self._config.seat_count

    @property
    def big_blind(self) -> int:
        return self._config.big_blind

    def next_actor(self) -> int | None:
        """Return the seat index (0-based) of the next actor, or None if hand over."""
        return self._state.actor_index

    def pot_committed(self) -> int:
        """Total chips committed to the pot including current street bets."""
        return self._state.total_pot_amount

    def current_bet(self) -> int:
        """Highest bet on the current street (the amount to call or exceed)."""
        return max(self._state.bets) if self._state.bets else 0

    def hand_is_over(self) -> bool:
        """True once pokerkit has settled the hand (status flips to False)."""
        return not self._state.status

    def apply_fold(self, seat: int) -> None:
        if self._state.actor_index != seat:
            raise ValueError(f"seat {seat} is not the current actor")
        self._state.fold()

    def apply_check_or_call(self, seat: int) -> None:
        if self._state.actor_index != seat:
            raise ValueError(f"seat {seat} is not the current actor")
        self._state.check_or_call()

    def min_raise_to(self) -> int:
        """Return the minimum total amount for a legal raise on the current street.

        Delegates to pokerkit's ``min_completion_betting_or_raising_to_amount``
        which implements TDA Rule 40 correctly (current_bet + last_raise_size).
        Returns ``big_blind`` as the floor when no raises have occurred yet.
        """
        amount = self._state.min_completion_betting_or_raising_to_amount
        if amount is None:
            return self._config.big_blind
        return int(amount)

    def apply_raise(self, seat: int, to: int) -> None:
        if self._state.actor_index != seat:
            raise ValueError(f"seat {seat} is not the current actor")
        try:
            self._state.complete_bet_or_raise_to(to)
        except Exception as exc:
            raise ValueError(f"invalid raise to {to}: {exc}") from exc
