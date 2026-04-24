"""TDA action-protocol validator — enforces rules pokerkit doesn't cover."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from holdembench.engine.table import Table
from holdembench.types import ActionKind, ActionName

_VALID_ACTIONS: frozenset[ActionName] = frozenset({"fold", "check", "call", "raise"})


class ValidationError(ValueError):
    """Raised when an action fails TDA-protocol validation."""


class RawDecision(BaseModel):
    """An agent's raw decision before TDA validation.

    Frozen + extra=forbid so adapters cannot smuggle arbitrary fields through.
    Invariants match ActionResponse so the two types cannot drift.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ActionKind
    action: ActionName | None = None
    amount: int | None = None
    message: str | None = None

    @model_validator(mode="after")
    def _check_kind_shape(self) -> RawDecision:
        if self.kind == "action" and self.action is None:
            raise ValueError("kind=action requires `action` field")
        if self.kind in {"probe", "probe_reply"} and not self.message:
            raise ValueError(f"kind={self.kind} requires non-empty `message`")
        return self


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
            # Message invariant already enforced at construction time.
            return
        if decision.kind != "action":
            raise ValidationError(f"unknown decision kind: {decision.kind!r}")
        if decision.action not in _VALID_ACTIONS:
            raise ValidationError(f"unknown action: {decision.action!r}")
        if decision.action == "raise":
            amount = decision.amount
            if amount is None or amount <= 0:
                raise ValidationError("raise requires positive amount")
            min_raise = self._table.min_raise_to()
            if amount < min_raise:
                raise ValidationError(f"min raise is {min_raise}, got {amount}")
