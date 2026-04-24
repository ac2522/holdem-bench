"""Prompt renderer — canonical form across all models."""

from __future__ import annotations

from holdembench.agents.base import DecisionContext
from holdembench.agents.prompt import (
    PromptBundle,
    SessionContext,
    TournamentContext,
    render_prompt,
)

_VOLATILE_TOKEN_UPPER_BOUND = 1000


def _ctx() -> DecisionContext:
    return DecisionContext(
        seat="Seat3",
        hand_id="s1h007",
        street="flop",
        legal=("fold", "call", "raise"),
        stacks={"Seat1": 980, "Seat2": 1020, "Seat3": 1000},
        board=("As", "Kd", "2c"),
        hole=("Qh", "Qs"),
        budget_remaining=320,
        is_probe_reply=False,
        deadline_s=60.0,
    )


def _tournament_ctx() -> TournamentContext:
    return TournamentContext(tournament_id="tour-1", seat="Seat3", seat_count=9)


def _session_ctx() -> SessionContext:
    return SessionContext(
        session_id=1,
        small_blind=10,
        big_blind=20,
        ante=0,
        starting_stack_bb=100,
        orbit_budget_tokens=400,
        canonical_action_log=(
            "s1h001 preflop Seat1:fold\ns1h001 preflop Seat2:call 20\n"
        ),
    )


def test_bundle_shape_is_four_blocks() -> None:
    bundle = render_prompt(
        tournament=_tournament_ctx(), session=_session_ctx(), decision=_ctx()
    )
    assert isinstance(bundle, PromptBundle)
    assert bundle.system_tournament
    assert bundle.system_session
    assert bundle.user_session_log
    assert bundle.user_volatile


def test_volatile_block_mentions_seat_and_board() -> None:
    bundle = render_prompt(
        tournament=_tournament_ctx(), session=_session_ctx(), decision=_ctx()
    )
    assert "Seat3" in bundle.user_volatile
    assert "As Kd 2c" in bundle.user_volatile


def test_volatile_is_reasonably_small() -> None:
    bundle = render_prompt(
        tournament=_tournament_ctx(), session=_session_ctx(), decision=_ctx()
    )
    assert bundle.user_volatile_token_count <= _VOLATILE_TOKEN_UPPER_BOUND


def test_probe_reply_flag_shows_in_volatile() -> None:
    ctx = _ctx()
    ctx_probe = DecisionContext(
        seat=ctx.seat,
        hand_id=ctx.hand_id,
        street=ctx.street,
        legal=ctx.legal,
        stacks=ctx.stacks,
        board=ctx.board,
        hole=ctx.hole,
        budget_remaining=ctx.budget_remaining,
        is_probe_reply=True,
        deadline_s=ctx.deadline_s,
        chat_log=("Seat5 (probe): Are you bluffing?",),
    )
    bundle = render_prompt(
        tournament=_tournament_ctx(), session=_session_ctx(), decision=ctx_probe
    )
    assert "probe" in bundle.user_volatile.lower()


def test_prompt_hash_stable_for_identical_input() -> None:
    t = _tournament_ctx()
    s = _session_ctx()
    d = _ctx()
    a = render_prompt(tournament=t, session=s, decision=d)
    b = render_prompt(tournament=t, session=s, decision=d)
    assert a.prompt_hash == b.prompt_hash


def test_prompt_hash_differs_when_hole_cards_differ() -> None:
    t = _tournament_ctx()
    s = _session_ctx()
    d1 = _ctx()
    d2 = DecisionContext(
        seat=d1.seat,
        hand_id=d1.hand_id,
        street=d1.street,
        legal=d1.legal,
        stacks=d1.stacks,
        board=d1.board,
        hole=("7c", "2h"),
        budget_remaining=d1.budget_remaining,
        is_probe_reply=False,
        deadline_s=d1.deadline_s,
    )
    a = render_prompt(tournament=t, session=s, decision=d1)
    b = render_prompt(tournament=t, session=s, decision=d2)
    assert a.prompt_hash != b.prompt_hash


def test_empty_canonical_log_renders_placeholder() -> None:
    s = SessionContext(
        session_id=1,
        small_blind=10,
        big_blind=20,
        ante=0,
        starting_stack_bb=100,
        orbit_budget_tokens=400,
        canonical_action_log="",
    )
    bundle = render_prompt(tournament=_tournament_ctx(), session=s, decision=_ctx())
    assert "(no actions yet)" in bundle.user_session_log
