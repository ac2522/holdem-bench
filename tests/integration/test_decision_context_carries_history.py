"""DecisionContext must carry the live action log, chat history, board, and
street to every agent — without these, models play blind every turn.

These are end-to-end runner-driven assertions: run a small stub tournament
with a Spy agent that records every DecisionContext it receives, then
assert the recorded contexts have the expected populated fields.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from holdembench.agents.base import Agent, DecisionContext, Pricing
from holdembench.agents.prompt import SessionContext, TournamentContext
from holdembench.engine.validator import RawDecision
from holdembench.harness.runner import TournamentConfig, run_tournament

pytestmark = pytest.mark.asyncio

_SEED = 11
_MIN_CHAT_DECISIONS = 2  # need at least 2 decisions for the chat round-trip
_FLOP_CARDS = 3  # any postflop street has >= 3 board cards


class _SpyAgent(Agent):
    """Records every DecisionContext received; replays a scripted reply.

    The agent emits a probe message on its first invocation so chat
    routing can be tested.  Subsequent calls just check / call.
    """

    pricing = Pricing(input_per_mtok=0.0, output_per_mtok=0.0)

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.received: list[DecisionContext] = []
        self._call_count = 0

    def set_context(
        self, *, tournament: TournamentContext, session: SessionContext
    ) -> None:
        _ = tournament, session  # ignored; we test DecisionContext only

    async def decide(self, ctx: DecisionContext) -> RawDecision:
        self.received.append(ctx)
        self._call_count += 1
        action = "check" if "check" in ctx.legal else "call"
        # First call: attach a probe message so we can verify routing.
        if self._call_count == 1:
            return RawDecision(
                kind="action",
                action=action,
                message=f"hello-from-{ctx.seat}-shittalking",
            )
        return RawDecision(kind="action", action=action)


def _config(tmp_path: Path, seat_count: int = 3, hand_cap: int = 2) -> TournamentConfig:
    return TournamentConfig(
        tournament_id="ctx-history",
        seats={f"Seat{i}": "stub:spy" for i in range(1, seat_count + 1)},
        small_blind=10,
        big_blind=20,
        ante=0,
        starting_stack=1000,
        hand_cap=hand_cap,
        session_count=1,
        master_seed=_SEED,
        results_dir=tmp_path,
    )


async def test_action_log_grows_across_decisions(tmp_path: Path) -> None:
    """By the second hand, ctx.canonical_action_log mentions hand 1."""
    spy = _SpyAgent("stub:spy")
    cfg = _config(tmp_path, seat_count=3, hand_cap=2)
    await run_tournament(cfg, {"stub:spy": spy})

    # Find a context from hand 2 (any seat)
    hand2_ctx = next(
        (c for c in spy.received if c.hand_id.endswith("h002")), None
    )
    assert hand2_ctx is not None, "no decision happened on hand 2"
    assert "h001" in hand2_ctx.canonical_action_log, (
        f"hand 2 prompt missing hand 1 actions; got log:\n{hand2_ctx.canonical_action_log!r}"
    )


async def test_chat_log_contains_other_seats_messages(tmp_path: Path) -> None:
    """Seat A's first-turn chat message must appear in the chat_log of the
    next decision delivered to Seat B."""
    spy = _SpyAgent("stub:spy")
    cfg = _config(tmp_path, seat_count=3, hand_cap=1)
    await run_tournament(cfg, {"stub:spy": spy})

    # spy.received[0] = first decision (no chat yet).
    # spy.received[1] = second decision; should see seat-from-first's chat.
    assert len(spy.received) >= _MIN_CHAT_DECISIONS, (
        "need at least 2 decisions for chat round-trip test"
    )
    second_ctx = spy.received[1]
    first_speaker = spy.received[0].seat
    haystack = "\n".join(second_ctx.chat_log)
    assert f"hello-from-{first_speaker}" in haystack, (
        f"second decision's chat_log missing first speaker's message; "
        f"chat_log={second_ctx.chat_log!r}"
    )


async def test_board_and_street_advance_postflop(tmp_path: Path) -> None:
    """Once the hand crosses to the flop, ctx.board has cards and street advances."""
    spy = _SpyAgent("stub:spy")
    # 3 seats checking through preflop guarantees a flop.
    cfg = _config(tmp_path, seat_count=3, hand_cap=1)
    await run_tournament(cfg, {"stub:spy": spy})

    postflop = [c for c in spy.received if c.street != "preflop"]
    assert postflop, (
        f"no postflop decision was made (got streets: "
        f"{[c.street for c in spy.received]})"
    )
    flop_ctx = postflop[0]
    assert len(flop_ctx.board) >= _FLOP_CARDS, (
        f"flop decision has board={flop_ctx.board!r}, expected >=3 cards"
    )
