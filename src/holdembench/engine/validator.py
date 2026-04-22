"""TDA action-protocol validator — enforces rules pokerkit doesn't cover."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from holdembench.engine.table import Table

ActionName = Literal["fold", "check", "call", "raise"]
ActionKind = Literal["action", "probe", "probe_reply"]

_VALID_ACTIONS = {"fold", "check", "call", "raise"}


class ValidationError(ValueError):
    """Raised when an action fails TDA-protocol validation."""


@dataclass(frozen=True)
class RawDecision:
    kind: ActionKind
    action: ActionName | None = None
    amount: int | None = None
    message: str | None = None


class TDAValidator:
    """Checks a RawDecision against TDA rules before it's applied.

    Engine-level rules (side-pot math, etc.) are enforced by pokerkit.
    This class covers the action-protocol rules we layer on top.
    """

    def __init__(self, table: Table) -> None:
        self._table = table

    def check(self, seat: int, decision: RawDecision) -> None:
        if self._table.next_actor() != seat:
            raise ValidationError(f"seat {seat} is not the current actor")

        if decision.kind in {"probe", "probe_reply"}:
            if not decision.message or not decision.message.strip():
                raise ValidationError(f"{decision.kind} requires a non-empty message")
            return

        if decision.kind != "action":
            raise ValidationError(f"unknown decision kind: {decision.kind!r}")

        if decision.action not in _VALID_ACTIONS:
            raise ValidationError(f"unknown action: {decision.action!r}")

        if decision.action == "raise":
            amount = decision.amount
            if amount is None or amount <= 0:
                raise ValidationError("raise requires positive amount")
            current_bet = self._table.current_bet()
            min_raise = current_bet * 2 if current_bet > 0 else self._table.big_blind
            if amount < min_raise:
                raise ValidationError(
                    f"min raise is {min_raise}, got {amount}"
                )
