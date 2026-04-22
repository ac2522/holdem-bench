"""Canonical event schema (pydantic v2 models)."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"

__all__ = [
    "_Base",
    "ActionRequest",
    "ActionResponse",
    "AutoFold",
    "BudgetCircuitBreak",
    "CommunityDeal",
    "Deal",
    "Event",
    "HandEnd",
    "HandStart",
    "ProbeResponseRequest",
    "SessionEnd",
    "SessionStart",
    "Showdown",
    "TournamentEnd",
    "TournamentStart",
    "ValidatorRejection",
    "parse_event",
]


class _Base(BaseModel):
    """Common pydantic config for all events."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TournamentStart(_Base):
    type: Literal["tournament_start"] = "tournament_start"
    tournament_id: str
    schema_version: str
    holdembench_version: str
    pokerkit_version: str
    git_sha: str
    seat_assignments: dict[str, str]
    master_seed: int
    anonymization_salt: str
    canary_uuid: str


class SessionStart(_Base):
    type: Literal["session_start"] = "session_start"
    session_id: int
    hand_cap: int
    small_blind: int
    big_blind: int
    ante: int
    deal_pack_seed: int


class HandStart(_Base):
    type: Literal["hand_start"] = "hand_start"
    hand_id: str
    button_seat: int
    stacks: dict[str, int]
    cards_hash: str
    chat_budgets_remaining: dict[str, int]


class Deal(_Base):
    type: Literal["deal"] = "deal"
    street: Literal["preflop"]
    to_seat: str
    cards: list[str]


class CommunityDeal(_Base):
    type: Literal["community_deal"] = "community_deal"
    street: Literal["flop", "turn", "river"]
    cards: list[str]


type Street = Literal["preflop", "flop", "turn", "river"]
type ActionKind = Literal["action", "probe", "probe_reply"]
type ActionName = Literal["fold", "check", "call", "raise"]


class ActionRequest(_Base):
    type: Literal["action_request"] = "action_request"
    hand_id: str
    to_seat: str
    street: Street
    legal: list[ActionName]
    timeout_s: float
    budget_remaining: int


class ActionResponse(_Base):
    type: Literal["action_response"] = "action_response"
    hand_id: str
    seat: str
    kind: ActionKind
    action: ActionName | None = None
    amount: int | None = None
    message: str | None = None
    tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    model_id: str
    prompt_hash: str = ""
    thinking: str | None = None
    auto_generated: bool = False
    timeout: bool = False

    @model_validator(mode="after")
    def _check_kind_shape(self) -> ActionResponse:
        if self.kind == "action" and self.action is None:
            raise ValueError("kind=action requires `action` field")
        if self.kind in {"probe", "probe_reply"} and not self.message:
            raise ValueError(f"kind={self.kind} requires non-empty `message`")
        return self


class ProbeResponseRequest(_Base):
    type: Literal["probe_response_request"] = "probe_response_request"
    hand_id: str
    from_seat: str
    responders: list[str]


class Showdown(_Base):
    type: Literal["showdown"] = "showdown"
    hand_id: str
    revealed: dict[str, list[str]]
    winners: list[dict[str, str | int]]
    all_in_ev_adjusted: bool = False
    stack_deltas_actual: dict[str, int] | None = None


class HandEnd(_Base):
    type: Literal["hand_end"] = "hand_end"
    hand_id: str
    stack_deltas: dict[str, int]
    elapsed_s: float
    total_cost_usd: float


class SessionEnd(_Base):
    type: Literal["session_end"] = "session_end"
    session_id: int
    final_stacks: dict[str, int]
    total_hands: int
    total_cost_usd: float


class TournamentEnd(_Base):
    type: Literal["tournament_end"] = "tournament_end"
    tournament_id: str
    final_chip_totals: dict[str, int]
    winner_seat: str
    winner_model: str
    total_cost_usd: float
    wall_clock_s: float


class ValidatorRejection(_Base):
    type: Literal["validator_rejection"] = "validator_rejection"
    seat: str
    reason: str
    original_response: dict[str, object]
    retry_allowed: bool


class AutoFold(_Base):
    type: Literal["auto_fold"] = "auto_fold"
    seat: str
    reason: Literal["timeout", "invalid_after_retry", "budget_circuit_break"]


class BudgetCircuitBreak(_Base):
    type: Literal["budget_circuit_break"] = "budget_circuit_break"
    seat: str
    threshold_usd: float
    actual_usd: float


type Event = Annotated[
    TournamentStart
    | SessionStart
    | HandStart
    | Deal
    | CommunityDeal
    | ActionRequest
    | ActionResponse
    | ProbeResponseRequest
    | Showdown
    | HandEnd
    | SessionEnd
    | TournamentEnd
    | ValidatorRejection
    | AutoFold
    | BudgetCircuitBreak,
    Field(discriminator="type"),
]


_EVENT_TYPES: dict[str, type[_Base]] = {
    cls.model_fields["type"].default: cls  # type: ignore[assignment]
    for cls in (
        TournamentStart,
        SessionStart,
        HandStart,
        Deal,
        CommunityDeal,
        ActionRequest,
        ActionResponse,
        ProbeResponseRequest,
        Showdown,
        HandEnd,
        SessionEnd,
        TournamentEnd,
        ValidatorRejection,
        AutoFold,
        BudgetCircuitBreak,
    )
}


def parse_event(data: dict[str, object]) -> _Base:
    t = data.get("type")
    if t is None:
        raise ValueError("event has no `type` field")
    cls = _EVENT_TYPES.get(str(t))  # type: ignore[arg-type]
    if cls is None:
        raise ValueError(f"unknown event type: {t!r}")
    return cls.model_validate(data)
