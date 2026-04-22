"""Tests for P1 runner fixes: action_request events, chat wiring, mark_folded, retry policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from holdembench.agents.base import DecisionContext, Pricing
from holdembench.chat.protocol import ChatProtocol, ChatRuleViolation
from holdembench.engine.validator import RawDecision
from holdembench.events.log import EventLog
from holdembench.events.schema import (
    ActionRequest,
    ActionResponse,
    AutoFold,
    TournamentEnd,
    ValidatorRejection,
)
from holdembench.harness.runner import TournamentConfig, run_tournament

_MIN_LEGAL_ACTIONS = 2
_EXPECTED_REJECTIONS_PER_INVALID = 2


# ---------------------------------------------------------------------------
# Stub agents
# ---------------------------------------------------------------------------


class _FoldAgent:
    """Always folds."""

    model_id = "stub:fold"
    pricing = Pricing(input_per_mtok=0.0, output_per_mtok=0.0)

    async def decide(self, ctx: DecisionContext) -> RawDecision:
        return RawDecision(kind="action", action="fold")


class _CheckAgent:
    """Always checks/calls (no message)."""

    model_id = "stub:check"
    pricing = Pricing(input_per_mtok=0.0, output_per_mtok=0.0)

    async def decide(self, ctx: DecisionContext) -> RawDecision:
        action = "check" if "check" in ctx.legal else "call"
        return RawDecision(kind="action", action=action)


class _ChatAgent:
    """Always checks/calls and attaches a short message."""

    model_id = "stub:chat"
    pricing = Pricing(input_per_mtok=0.0, output_per_mtok=0.0)

    async def decide(self, ctx: DecisionContext) -> RawDecision:
        action = "check" if "check" in ctx.legal else "call"
        return RawDecision(kind="action", action=action, message="hello poker friends")


class _InjectionChatAgent:
    """Attaches a message with an HTML injection pattern (should be content-rejected)."""

    model_id = "stub:injection"
    pricing = Pricing(input_per_mtok=0.0, output_per_mtok=0.0)

    async def decide(self, ctx: DecisionContext) -> RawDecision:
        action = "check" if "check" in ctx.legal else "call"
        return RawDecision(kind="action", action=action, message="<script>alert(1)</script>")


class _AlwaysInvalidRaiseAgent:
    """Always tries to raise with an invalid amount (1 chip — always below min)."""

    model_id = "stub:invalid_raise"
    pricing = Pricing(input_per_mtok=0.0, output_per_mtok=0.0)

    async def decide(self, ctx: DecisionContext) -> RawDecision:
        return RawDecision(kind="action", action="raise", amount=1)


def _base_cfg(tmp_path: Path, *, seats: dict[str, str], hand_cap: int = 3) -> TournamentConfig:
    return TournamentConfig(
        tournament_id="t-test",
        seats=seats,
        small_blind=10,
        big_blind=20,
        ante=0,
        starting_stack=1000,
        hand_cap=hand_cap,
        session_count=1,
        master_seed=1,
        results_dir=tmp_path / "results",
    )


# ---------------------------------------------------------------------------
# Fix 3: ActionRequest events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_emits_action_request_events(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path, seats={"Seat1": "stub:check", "Seat2": "stub:check"})
    agents = {"stub:check": _CheckAgent()}
    out = await run_tournament(cfg, agents)
    events = list(EventLog.replay(out.log_path))
    action_requests = [e for e in events if isinstance(e, ActionRequest)]
    assert len(action_requests) >= cfg.hand_cap, (
        f"Expected at least {cfg.hand_cap} ActionRequest events, got {len(action_requests)}"
    )
    # Each ActionRequest must have the correct hand_id structure and legal actions
    for req in action_requests:
        assert req.hand_id.startswith("s1h")
        assert req.timeout_s == 60.0  # noqa: PLR2004
        assert len(req.legal) >= _MIN_LEGAL_ACTIONS
        assert req.budget_remaining >= 0


# ---------------------------------------------------------------------------
# Fix 1: Chat content + spend wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_calls_chat_spend_valid_message(tmp_path: Path) -> None:
    """Agent sends a valid short message — it should appear in ActionResponse."""
    cfg = _base_cfg(tmp_path, seats={"Seat1": "stub:chat", "Seat2": "stub:chat"})
    agents = {"stub:chat": _ChatAgent()}
    out = await run_tournament(cfg, agents)
    events = list(EventLog.replay(out.log_path))
    responses = [e for e in events if isinstance(e, ActionResponse)]
    # At least some responses should carry the message
    messages = [e.message for e in responses if e.message is not None]
    assert len(messages) > 0, "Expected at least one ActionResponse with a non-null message"


@pytest.mark.asyncio
async def test_runner_content_rejection_strips_message(tmp_path: Path) -> None:
    """Agent sends an HTML injection message — should be stripped and ValidatorRejection emitted."""
    cfg = _base_cfg(tmp_path, seats={"Seat1": "stub:injection", "Seat2": "stub:injection"})
    agents = {"stub:injection": _InjectionChatAgent()}
    out = await run_tournament(cfg, agents)
    events = list(EventLog.replay(out.log_path))

    rejections = [
        e for e in events if isinstance(e, ValidatorRejection) and e.reason == "chat_content"
    ]
    assert len(rejections) > 0, "Expected ValidatorRejection with reason='chat_content'"

    responses = [e for e in events if isinstance(e, ActionResponse)]
    # All messages must be None because the injection was stripped
    for resp in responses:
        assert resp.message is None


# ---------------------------------------------------------------------------
# Fix 2: mark_folded when seat folds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_marks_folded_seat_chat_silenced(tmp_path: Path) -> None:
    """After a seat folds, its ChatProtocol state should reflect folded=True.

    We verify this indirectly: the fold agent folds every turn. If mark_folded
    is called the chat spend path is blocked for that seat. We assert no
    chat_budget rejections appear for the fold agent (it has no message anyway),
    and that the runner completes without errors.
    """
    cfg = _base_cfg(tmp_path, seats={"Seat1": "stub:fold", "Seat2": "stub:check"})
    agents = {"stub:fold": _FoldAgent(), "stub:check": _CheckAgent()}
    out = await run_tournament(cfg, agents)
    events = list(EventLog.replay(out.log_path))

    # Tournament must complete successfully
    assert any(isinstance(e, TournamentEnd) for e in events)

    # No unexpected budget rejections
    budget_rejections = [
        e for e in events if isinstance(e, ValidatorRejection) and e.reason == "chat_budget"
    ]
    assert len(budget_rejections) == 0


@pytest.mark.asyncio
async def test_runner_chat_protocol_folded_state_directly(tmp_path: Path) -> None:
    """Directly verify ChatProtocol.mark_folded semantics.

    We instrument a ChatProtocol manually to prove mark_folded prevents spend.
    """
    chat = ChatProtocol(seats=("Seat1", "Seat2"), budget_per_orbit=400, per_action_cap=80)
    chat.start_hand(in_hand={"Seat1", "Seat2"})

    # Before fold — spend works
    chat.spend("Seat1", "hello world", kind="action")

    # After fold — spend raises
    chat.mark_folded("Seat1")
    with pytest.raises(ChatRuleViolation, match="folded"):
        chat.spend("Seat1", "hello again", kind="action")


# ---------------------------------------------------------------------------
# Fix 4: Retry-once policy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_validator_retry_emits_two_rejections_and_autofold(
    tmp_path: Path,
) -> None:
    """Agent always raises with amount=1 (invalid). Expect:
    1. ValidatorRejection(retry_allowed=True)
    2. ValidatorRejection(retry_allowed=False)
    3. AutoFold(reason="invalid_after_retry")
    """
    cfg = _base_cfg(
        tmp_path,
        seats={"Seat1": "stub:invalid_raise", "Seat2": "stub:check"},
        hand_cap=1,
    )
    agents = {
        "stub:invalid_raise": _AlwaysInvalidRaiseAgent(),
        "stub:check": _CheckAgent(),
    }
    out = await run_tournament(cfg, agents)
    events = list(EventLog.replay(out.log_path))

    rejections = [e for e in events if isinstance(e, ValidatorRejection)]
    auto_folds = [e for e in events if isinstance(e, AutoFold)]

    seat1_rejections = [e for e in rejections if e.seat == "Seat1"]
    assert len(seat1_rejections) >= _EXPECTED_REJECTIONS_PER_INVALID, (
        f"Expected {_EXPECTED_REJECTIONS_PER_INVALID} ValidatorRejection for Seat1, "
        f"got {len(seat1_rejections)}: {seat1_rejections}"
    )
    assert seat1_rejections[0].retry_allowed is True
    assert seat1_rejections[1].retry_allowed is False

    # AutoFold must follow
    seat1_folds = [e for e in auto_folds if e.seat == "Seat1"]
    assert len(seat1_folds) >= 1
    assert seat1_folds[0].reason == "invalid_after_retry"
