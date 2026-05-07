"""Independent adversarial integration tests for the match-based tournament.

These tests are written purely from the documented public-API contract,
without reading the harness implementation.  Each test asserts a single
invariant the contract promises and prints a clear failure message so a
break tells you exactly which contract clause regressed.

Conventions:
    * Stub agents only (no LLM calls, fully deterministic).
    * Fixed master_seed values for reproducibility.
    * Per-test ``tmp_path`` so logs don't collide.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from holdembench.agents.base import Agent, DecisionContext, Pricing
from holdembench.baselines.random_agent import RandomAgent
from holdembench.baselines.tight_passive import TightPassiveAgent
from holdembench.engine.validator import RawDecision
from holdembench.events.log import EventLog
from holdembench.harness.runner import (
    BlindLevel,
    TournamentConfig,
    run_tournament,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_config(
    tmp_path: Path,
    *,
    seat_count: int,
    session_count: int = 1,
    hand_cap: int = 8,
    starting_stack: int = 1000,
    small_blind: int = 10,
    big_blind: int = 20,
    ante: int = 0,
    blind_levels: tuple[BlindLevel, ...] | None = None,
    master_seed: int = 11,
    model_id: str = "stub:tight_passive",
    tournament_id: str = "test",
) -> TournamentConfig:
    return TournamentConfig(
        tournament_id=tournament_id,
        seats={f"Seat{i}": model_id for i in range(1, seat_count + 1)},
        small_blind=small_blind,
        big_blind=big_blind,
        ante=ante,
        starting_stack=starting_stack,
        hand_cap=hand_cap,
        session_count=session_count,
        master_seed=master_seed,
        results_dir=tmp_path,
        blind_levels=blind_levels,
    )


def _replay(log_path: Path) -> list:
    return list(EventLog.replay(log_path))


def _bank(seat_count: int, starting_stack: int) -> int:
    return seat_count * starting_stack


# ---------------------------------------------------------------------------
# Spy agent for DecisionContext snapshots (reset verification)
# ---------------------------------------------------------------------------


class _SpyAgent(Agent):
    """Records every DecisionContext received; replays a benign reply."""

    pricing = Pricing(input_per_mtok=0.0, output_per_mtok=0.0)

    def __init__(self, model_id: str = "stub:spy") -> None:
        self.model_id = model_id
        self.received: list[DecisionContext] = []

    def set_context(self, *, tournament: object, session: object) -> None:
        # Optional; we only read DecisionContext.
        _ = tournament, session

    async def decide(self, ctx: DecisionContext) -> RawDecision:
        self.received.append(ctx)
        action = "check" if "check" in ctx.legal else (
            "call" if "call" in ctx.legal else ctx.legal[0]
        )
        return RawDecision(kind="action", action=action)


# ---------------------------------------------------------------------------
# 1. Seat count edge cases (heads-up, 3, 9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seat_count", [2, 3, 9])
async def test_seat_count_edge_cases_complete_normally(
    seat_count: int, tmp_path: Path
) -> None:
    """A tournament with any supported seat count completes with a tournament_end."""
    cfg = _make_config(
        tmp_path, seat_count=seat_count, hand_cap=4, session_count=1,
        tournament_id=f"seats-{seat_count}",
    )
    result = await run_tournament(cfg, {"stub:tight_passive": TightPassiveAgent()})
    events = _replay(result.log_path)
    types = [e.type for e in events]
    assert "tournament_start" in types, (
        f"seat_count={seat_count}: missing tournament_start; types={types}"
    )
    assert "session_start" in types, (
        f"seat_count={seat_count}: missing session_start; types={types}"
    )
    assert "session_end" in types, (
        f"seat_count={seat_count}: missing session_end; types={types}"
    )
    assert "tournament_end" in types, (
        f"seat_count={seat_count}: missing tournament_end; types={types}"
    )
    # session_end stack count must equal seat count.
    session_end = next(e for e in events if e.type == "session_end")
    assert len(session_end.final_stacks) == seat_count, (  # type: ignore[attr-defined]
        f"seat_count={seat_count}: session_end has "
        f"{len(session_end.final_stacks)} stacks, expected {seat_count}"  # type: ignore[attr-defined]
    )


# ---------------------------------------------------------------------------
# 2. Match (session) count edge cases — 1, 2, 3, 5
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("session_count", [1, 2, 3, 5])
async def test_session_count_emits_correct_number_of_sessions(
    session_count: int, tmp_path: Path
) -> None:
    """Exactly session_count session_start and session_end events are emitted."""
    cfg = _make_config(
        tmp_path, seat_count=3, hand_cap=3, session_count=session_count,
        tournament_id=f"sessions-{session_count}",
    )
    result = await run_tournament(cfg, {"stub:tight_passive": TightPassiveAgent()})
    events = _replay(result.log_path)
    starts = [e for e in events if e.type == "session_start"]
    ends = [e for e in events if e.type == "session_end"]
    assert len(starts) == session_count, (
        f"expected {session_count} session_start events, got {len(starts)}"
    )
    assert len(ends) == session_count, (
        f"expected {session_count} session_end events, got {len(ends)}"
    )
    # Session ids should be 1..session_count, monotonic.
    start_ids = [e.session_id for e in starts]  # type: ignore[attr-defined]
    end_ids = [e.session_id for e in ends]  # type: ignore[attr-defined]
    assert start_ids == sorted(start_ids), (
        f"session_start ids not monotonic: {start_ids}"
    )
    assert end_ids == sorted(end_ids), (
        f"session_end ids not monotonic: {end_ids}"
    )
    assert start_ids == end_ids, (
        f"session_start ids {start_ids} != session_end ids {end_ids}"
    )


# ---------------------------------------------------------------------------
# 3. Blind ladder edge cases
# ---------------------------------------------------------------------------


async def test_blind_levels_none_uses_config_blinds_throughout(tmp_path: Path) -> None:
    """When blind_levels is None, every HandStart uses TournamentConfig.small/big_blind."""
    cfg = _make_config(
        tmp_path, seat_count=3, hand_cap=5, session_count=1,
        small_blind=10, big_blind=20, blind_levels=None,
        tournament_id="blinds-none",
    )
    result = await run_tournament(cfg, {"stub:tight_passive": TightPassiveAgent()})
    events = _replay(result.log_path)
    hand_starts = [e for e in events if e.type == "hand_start"]
    assert hand_starts, "no hand_start events"
    for hs in hand_starts:
        assert hs.small_blind == 10, (  # type: ignore[attr-defined]
            f"hand {hs.hand_id} sb={hs.small_blind}, expected 10 "  # type: ignore[attr-defined]
            f"(blind_levels=None should fall through to config blinds)"
        )
        assert hs.big_blind == 20, (  # type: ignore[attr-defined]
            f"hand {hs.hand_id} bb={hs.big_blind}, expected 20"  # type: ignore[attr-defined]
        )


async def test_blind_level_unreachable_never_triggers(tmp_path: Path) -> None:
    """A blind level whose start_hand exceeds hand_cap should never be in effect."""
    # hand_cap=4, but a level at start_hand=99 — should never apply.
    cfg = _make_config(
        tmp_path, seat_count=3, hand_cap=4, session_count=1,
        small_blind=10, big_blind=20,
        blind_levels=(
            BlindLevel(start_hand=1, small_blind=10, big_blind=20),
            BlindLevel(start_hand=99, small_blind=500, big_blind=1000),
        ),
        tournament_id="blinds-unreachable",
    )
    result = await run_tournament(cfg, {"stub:tight_passive": TightPassiveAgent()})
    events = _replay(result.log_path)
    for hs in (e for e in events if e.type == "hand_start"):
        assert (hs.small_blind, hs.big_blind) == (10, 20), (  # type: ignore[attr-defined]
            f"hand {hs.hand_id} unexpectedly used blinds "  # type: ignore[attr-defined]
            f"({hs.small_blind}/{hs.big_blind}); the level at "  # type: ignore[attr-defined]
            f"start_hand=99 should never trigger when hand_cap=4"
        )


async def test_single_blind_level_at_start_hand_one(tmp_path: Path) -> None:
    """A single level at start_hand=1 governs every hand of the match."""
    cfg = _make_config(
        tmp_path, seat_count=3, hand_cap=5, session_count=1,
        small_blind=10, big_blind=20,
        blind_levels=(BlindLevel(start_hand=1, small_blind=25, big_blind=50),),
        tournament_id="blinds-single",
    )
    result = await run_tournament(cfg, {"stub:tight_passive": TightPassiveAgent()})
    events = _replay(result.log_path)
    hand_starts = [e for e in events if e.type == "hand_start"]
    assert hand_starts, "no hand_start events"
    for hs in hand_starts:
        assert (hs.small_blind, hs.big_blind) == (25, 50), (  # type: ignore[attr-defined]
            f"hand {hs.hand_id}: blinds {hs.small_blind}/{hs.big_blind}, "  # type: ignore[attr-defined]
            f"expected single-level 25/50"
        )


async def test_blinds_reset_to_level_one_at_start_of_each_match(tmp_path: Path) -> None:
    """Hand 1 of every match must use level-1 blinds (10/20), not whatever
    the previous match was on when it ended."""
    cfg = _make_config(
        tmp_path, seat_count=3, hand_cap=6, session_count=2,
        small_blind=10, big_blind=20,
        blind_levels=(
            BlindLevel(start_hand=1, small_blind=10, big_blind=20),
            BlindLevel(start_hand=3, small_blind=50, big_blind=100),
        ),
        tournament_id="blinds-reset",
    )
    result = await run_tournament(cfg, {"stub:tight_passive": TightPassiveAgent()})
    events = _replay(result.log_path)
    # First HandStart inside each session.
    first_hs_per_session: dict[int, object] = {}
    current_session = 0
    for ev in events:
        if ev.type == "session_start":
            current_session = ev.session_id  # type: ignore[attr-defined]
        elif ev.type == "hand_start" and current_session not in first_hs_per_session:
            first_hs_per_session[current_session] = ev
    assert len(first_hs_per_session) == 2, (
        f"expected 2 sessions, got {sorted(first_hs_per_session)}"
    )
    for sid, hs in first_hs_per_session.items():
        assert (hs.small_blind, hs.big_blind) == (10, 20), (  # type: ignore[attr-defined]
            f"session {sid} hand 1 used blinds "
            f"{hs.small_blind}/{hs.big_blind}, expected level-1 10/20 "  # type: ignore[attr-defined]
            f"(blinds must reset between matches)"
        )


async def test_session_start_carries_level_one_blinds(tmp_path: Path) -> None:
    """SessionStart event must report the level-1 blinds (small_blind, big_blind)."""
    cfg = _make_config(
        tmp_path, seat_count=3, hand_cap=4, session_count=2,
        small_blind=10, big_blind=20,
        blind_levels=(
            BlindLevel(start_hand=1, small_blind=15, big_blind=30),
            BlindLevel(start_hand=3, small_blind=50, big_blind=100),
        ),
        tournament_id="session-start-blinds",
    )
    result = await run_tournament(cfg, {"stub:tight_passive": TightPassiveAgent()})
    events = _replay(result.log_path)
    session_starts = [e for e in events if e.type == "session_start"]
    assert len(session_starts) == 2
    for ss in session_starts:
        assert (ss.small_blind, ss.big_blind) == (15, 30), (  # type: ignore[attr-defined]
            f"session_start session_id={ss.session_id} reports "  # type: ignore[attr-defined]
            f"blinds {ss.small_blind}/{ss.big_blind}, "  # type: ignore[attr-defined]
            f"expected level-1 (15/30)"
        )


async def test_blind_rises_mid_match_take_effect(tmp_path: Path) -> None:
    """A multi-level ladder where a rise happens within hand_cap actually
    rises the blinds at the prescribed hand."""
    cfg = _make_config(
        tmp_path, seat_count=3, hand_cap=6, session_count=1,
        small_blind=10, big_blind=20,
        blind_levels=(
            BlindLevel(start_hand=1, small_blind=10, big_blind=20),
            BlindLevel(start_hand=3, small_blind=20, big_blind=40),
            BlindLevel(start_hand=5, small_blind=40, big_blind=80),
        ),
        tournament_id="blinds-rise",
    )
    result = await run_tournament(cfg, {"stub:tight_passive": TightPassiveAgent()})
    events = _replay(result.log_path)
    # Build (hand-index-within-session-1, sb, bb).
    seq: list[tuple[int, int, int]] = []
    session = 0
    idx = 0
    for ev in events:
        if ev.type == "session_start":
            session = ev.session_id  # type: ignore[attr-defined]
            idx = 0
        elif ev.type == "hand_start" and session == 1:
            idx += 1
            seq.append((idx, ev.small_blind, ev.big_blind))  # type: ignore[attr-defined]
    # Build expected: per spec, level with highest start_hand <= H wins.
    levels = [(1, 10, 20), (3, 20, 40), (5, 40, 80)]
    def expected(h: int) -> tuple[int, int]:
        sb, bb = 10, 20
        for sh, s, b in levels:
            if sh <= h:
                sb, bb = s, b
        return sb, bb

    for h, sb, bb in seq:
        ex_sb, ex_bb = expected(h)
        assert (sb, bb) == (ex_sb, ex_bb), (
            f"session 1 hand {h}: blinds {sb}/{bb}, expected {ex_sb}/{ex_bb}"
        )


# ---------------------------------------------------------------------------
# 4. Reset verification — chat & action log do NOT leak across matches.
# ---------------------------------------------------------------------------


async def test_action_log_resets_between_matches(tmp_path: Path) -> None:
    """First DecisionContext of match 2 must have an empty canonical_action_log,
    even though match 1 generated lots of actions."""
    spy = _SpyAgent("stub:spy")
    # Need enough hands in match 1 to create action history.
    cfg = _make_config(
        tmp_path, seat_count=3, hand_cap=3, session_count=2,
        master_seed=42, model_id="stub:spy",
        tournament_id="reset-action-log",
    )
    await run_tournament(cfg, {"stub:spy": spy})

    # Identify match boundary by hand_id pattern.  After match 1 ends, hand_ids
    # for match 2 should not appear in match-1 history.  We pick any context
    # whose canonical_action_log mentions h001 ("hand 1") -- the FIRST decision
    # of match 2 must NOT.  We approximate "first decision of match 2" as:
    # the first ctx whose hand_id corresponds to session 2.
    # Hand IDs in this codebase look like "<tournament>:s<session>:h<seq>" or
    # similar; we match on a substring containing "s002" or "s2" etc.
    def session_index(hand_id: str) -> int | None:
        # Try common patterns: ":s001:" or ":s1:" or "_s2_"
        for tok in ("s002", "s2", "session2", "session_2", "_s002_", ":s002:"):
            if tok in hand_id:
                return 2
        for tok in ("s001", "s1", "session1", "session_1", "_s001_", ":s001:"):
            if tok in hand_id:
                return 1
        return None

    # Fall back: find the boundary by looking for a context where hole-card seat
    # is the same but action_log shrinks back to empty.
    first_match2_ctx: DecisionContext | None = None
    prev_log_lengths: list[int] = []
    for ctx in spy.received:
        sid = session_index(ctx.hand_id)
        if sid == 2:
            first_match2_ctx = ctx
            break
        prev_log_lengths.append(len(ctx.canonical_action_log))

    if first_match2_ctx is None:
        # Fallback heuristic: detect a sudden drop in action-log length.
        for i, ctx in enumerate(spy.received[1:], start=1):
            prev = spy.received[i - 1].canonical_action_log
            if len(prev) > 50 and len(ctx.canonical_action_log) < len(prev) // 2:
                first_match2_ctx = ctx
                break

    assert first_match2_ctx is not None, (
        "could not identify the first decision of match 2 from spy.received; "
        f"hand_ids seen: {[c.hand_id for c in spy.received]}"
    )
    assert first_match2_ctx.canonical_action_log == "", (
        f"first decision of match 2 had non-empty canonical_action_log "
        f"({first_match2_ctx.canonical_action_log!r}); "
        "action history must reset between matches per contract"
    )


async def test_chat_log_resets_between_matches(tmp_path: Path) -> None:
    """Chat from match 1 must not appear in DecisionContext.chat_log of match 2."""
    sentinel = "MATCH1-CHAT-SENTINEL-XYZ"

    class _ChattyOnceSpy:
        model_id = "stub:chatty"
        pricing = Pricing(input_per_mtok=0.0, output_per_mtok=0.0)

        def __init__(self) -> None:
            self.received: list[DecisionContext] = []
            self._sent = False

        def set_context(self, *, tournament: object, session: object) -> None:
            _ = tournament, session

        async def decide(self, ctx: DecisionContext) -> RawDecision:
            self.received.append(ctx)
            action = "check" if "check" in ctx.legal else (
                "call" if "call" in ctx.legal else ctx.legal[0]
            )
            if not self._sent:
                self._sent = True
                return RawDecision(
                    kind="action", action=action, message=sentinel,
                )
            return RawDecision(kind="action", action=action)

    spy = _ChattyOnceSpy()
    cfg = _make_config(
        tmp_path, seat_count=3, hand_cap=2, session_count=2,
        master_seed=99, model_id="stub:chatty",
        tournament_id="reset-chat-log",
    )
    await run_tournament(cfg, {"stub:chatty": spy})

    # Identify match-2 contexts.  Use heuristic: chat_log length suddenly
    # drops to 0 after having had entries.
    saw_sentinel_in_match1 = any(
        any(sentinel in s for s in c.chat_log) for c in spy.received
    )
    if not saw_sentinel_in_match1:
        pytest.skip(
            "could not generate chat in match 1 to test reset against; "
            "this is a sub-component (chat round-trip), not the reset itself"
        )

    # Find the latest contiguous run of contexts with empty chat_log following
    # a non-empty chat_log run — that boundary is between matches.
    boundary_idx = None
    for i in range(1, len(spy.received)):
        prev_has_sentinel = any(sentinel in s for s in spy.received[i - 1].chat_log)
        curr_has_sentinel = any(sentinel in s for s in spy.received[i].chat_log)
        if prev_has_sentinel and not curr_has_sentinel:
            boundary_idx = i
            break

    assert boundary_idx is not None, (
        "never observed the sentinel disappearing from chat_log; "
        "either chat never propagated or it never resets"
    )
    # All decisions from boundary onward must NOT contain the match-1 sentinel.
    for ctx in spy.received[boundary_idx:]:
        assert not any(sentinel in s for s in ctx.chat_log), (
            f"match-1 chat sentinel leaked into match-2 chat_log: "
            f"{ctx.chat_log!r}"
        )


# ---------------------------------------------------------------------------
# 5. Conservation under bust / aggressive play.
# ---------------------------------------------------------------------------


async def test_chips_conserved_per_session_with_random_aggressive(
    tmp_path: Path,
) -> None:
    """Random agents (often all-in) across multiple matches preserve the bank
    at every session_end."""
    cfg = _make_config(
        tmp_path, seat_count=4, hand_cap=10, session_count=3,
        starting_stack=1000, small_blind=10, big_blind=20,
        master_seed=7, model_id="stub:random",
        tournament_id="bust-conservation",
    )
    result = await run_tournament(
        cfg, {"stub:random": RandomAgent(seed=7, big_blind=20)}
    )
    events = _replay(result.log_path)
    bank = _bank(4, 1000)
    session_ends = [e for e in events if e.type == "session_end"]
    assert len(session_ends) == 3, f"expected 3 session_ends, got {len(session_ends)}"
    for se in session_ends:
        total = sum(se.final_stacks.values())  # type: ignore[attr-defined]
        assert total == bank, (
            f"session {se.session_id} final_stacks sum to {total}, "  # type: ignore[attr-defined]
            f"expected {bank} (chip conservation under busts violated)"
        )


async def test_hand_end_stack_deltas_always_zero_sum(tmp_path: Path) -> None:
    """Per-hand chip conservation across multiple matches and aggressive play."""
    cfg = _make_config(
        tmp_path, seat_count=3, hand_cap=8, session_count=2,
        starting_stack=1000, small_blind=10, big_blind=20,
        master_seed=2026, model_id="stub:random",
        tournament_id="per-hand-conservation",
    )
    result = await run_tournament(
        cfg, {"stub:random": RandomAgent(seed=2026, big_blind=20)}
    )
    events = _replay(result.log_path)
    hand_ends = [e for e in events if e.type == "hand_end"]
    assert hand_ends, "no hand_end events emitted"
    for he in hand_ends:
        s = sum(he.stack_deltas.values())  # type: ignore[attr-defined]
        assert s == 0, (
            f"hand_end {he.hand_id} stack_deltas sum to {s}: "  # type: ignore[attr-defined]
            f"{he.stack_deltas} (per-hand conservation broken)"  # type: ignore[attr-defined]
        )


async def test_busted_seat_remains_zero_for_remainder_of_match(tmp_path: Path) -> None:
    """Once a seat reaches 0 chips inside a match, it must remain 0 at every
    subsequent hand_start of that match (no rebuys)."""
    cfg = _make_config(
        tmp_path, seat_count=4, hand_cap=20, session_count=1,
        starting_stack=1000, small_blind=10, big_blind=20,
        master_seed=3, model_id="stub:random",
        tournament_id="bust-no-rebuy",
    )
    result = await run_tournament(
        cfg, {"stub:random": RandomAgent(seed=3, big_blind=20)}
    )
    events = _replay(result.log_path)
    busted_at: dict[str, int] = {}  # seat -> first hand index busted
    hand_idx = 0
    for ev in events:
        if ev.type == "hand_start":
            hand_idx += 1
            for seat, chips in ev.stacks.items():  # type: ignore[attr-defined]
                if chips == 0 and seat not in busted_at:
                    busted_at[seat] = hand_idx
                if chips != 0 and seat in busted_at:
                    pytest.fail(
                        f"seat {seat} busted at hand {busted_at[seat]} but "
                        f"reappears with {chips} chips at hand {hand_idx} "
                        f"(rebuys forbidden inside a match)"
                    )


# ---------------------------------------------------------------------------
# 6. Final-score correctness: sum of session_end final_stacks per seat.
# ---------------------------------------------------------------------------


async def test_final_score_equals_sum_of_session_finals(tmp_path: Path) -> None:
    """tournament_end.final_score[seat] == sum(session_end.final_stacks[seat])
    across every session, even when seats bust in some matches."""
    cfg = _make_config(
        tmp_path, seat_count=4, hand_cap=8, session_count=3,
        starting_stack=1000, small_blind=10, big_blind=20,
        master_seed=5, model_id="stub:random",
        tournament_id="final-score-sum",
    )
    result = await run_tournament(
        cfg, {"stub:random": RandomAgent(seed=5, big_blind=20)}
    )
    events = _replay(result.log_path)
    per_seat_total: dict[str, int] = defaultdict(int)
    for ev in events:
        if ev.type == "session_end":
            for seat, chips in ev.final_stacks.items():  # type: ignore[attr-defined]
                per_seat_total[seat] += chips
    end = next((e for e in events if e.type == "tournament_end"), None)
    assert end is not None, "tournament_end missing"
    final_score = end.final_score  # type: ignore[attr-defined]
    assert dict(per_seat_total) == final_score, (
        f"final_score {final_score} != sum-of-session-finals "
        f"{dict(per_seat_total)}"
    )


async def test_winner_seat_has_max_final_score(tmp_path: Path) -> None:
    """tournament_end.winner_seat must be argmax of final_score (with ties
    being broken in some deterministic manner; we assert the winner's score
    equals the max)."""
    cfg = _make_config(
        tmp_path, seat_count=3, hand_cap=8, session_count=2,
        starting_stack=1000, small_blind=10, big_blind=20,
        master_seed=13, model_id="stub:random",
        tournament_id="winner-argmax",
    )
    result = await run_tournament(
        cfg, {"stub:random": RandomAgent(seed=13, big_blind=20)}
    )
    events = _replay(result.log_path)
    end = next((e for e in events if e.type == "tournament_end"), None)
    assert end is not None
    fs = end.final_score  # type: ignore[attr-defined]
    winner = end.winner_seat  # type: ignore[attr-defined]
    assert winner in fs, f"winner {winner} not in final_score {fs}"
    max_score = max(fs.values())
    assert fs[winner] == max_score, (
        f"winner_seat={winner} has final_score={fs[winner]}, "
        f"but max in final_score is {max_score}: {fs}"
    )


async def test_total_bank_across_matches_matches_starting_total(tmp_path: Path) -> None:
    """Sum of all final_score values must equal seat_count * starting_stack *
    session_count -- every match starts with a fresh bank, all chips are
    accounted for at session_end."""
    seats = 3
    starting = 1000
    sessions = 4
    cfg = _make_config(
        tmp_path, seat_count=seats, hand_cap=6, session_count=sessions,
        starting_stack=starting, small_blind=10, big_blind=20,
        master_seed=21, model_id="stub:random",
        tournament_id="bank-across",
    )
    result = await run_tournament(
        cfg, {"stub:random": RandomAgent(seed=21, big_blind=20)}
    )
    events = _replay(result.log_path)
    end = next((e for e in events if e.type == "tournament_end"), None)
    assert end is not None
    expected_total = seats * starting * sessions
    actual_total = sum(end.final_score.values())  # type: ignore[attr-defined]
    assert actual_total == expected_total, (
        f"sum(final_score) = {actual_total}, expected "
        f"{expected_total} = {seats}*{starting}*{sessions}"
    )


# ---------------------------------------------------------------------------
# 7. Stack-reset invariant — every match's first hand has every seat at the
#    starting_stack regardless of what happened before.
# ---------------------------------------------------------------------------


async def test_first_hand_of_every_match_resets_to_starting_stack(
    tmp_path: Path,
) -> None:
    """Even after busts in match N, match N+1's hand 1 must show every seat
    at starting_stack."""
    starting = 1000
    cfg = _make_config(
        tmp_path, seat_count=4, hand_cap=12, session_count=3,
        starting_stack=starting, small_blind=10, big_blind=20,
        master_seed=4, model_id="stub:random",
        tournament_id="reset-stacks",
    )
    result = await run_tournament(
        cfg, {"stub:random": RandomAgent(seed=4, big_blind=20)}
    )
    events = _replay(result.log_path)
    current_session = 0
    seen_first_in_session: set[int] = set()
    for ev in events:
        if ev.type == "session_start":
            current_session = ev.session_id  # type: ignore[attr-defined]
        elif ev.type == "hand_start" and current_session not in seen_first_in_session:
            seen_first_in_session.add(current_session)
            for seat, chips in ev.stacks.items():  # type: ignore[attr-defined]
                assert chips == starting, (
                    f"session {current_session} hand 1: seat {seat} = {chips}, "
                    f"expected {starting} (stacks must reset every match)"
                )
    assert seen_first_in_session == {1, 2, 3}, (
        f"missing first-hand entries for some sessions: {seen_first_in_session}"
    )


# ---------------------------------------------------------------------------
# 8. Heads-up + high-blind forced-bust scenario (stress on hand_cap termination
#    when fewer than 2 active seats remain — the tournament ends the match).
# ---------------------------------------------------------------------------


async def test_heads_up_short_stack_completes(tmp_path: Path) -> None:
    """Heads-up with high blinds: match must complete cleanly with a
    session_end and total chips = bank, even when blinds eat stacks fast."""
    cfg = _make_config(
        tmp_path, seat_count=2, hand_cap=20, session_count=1,
        starting_stack=1000, small_blind=10, big_blind=20,
        blind_levels=(
            BlindLevel(start_hand=1, small_blind=10, big_blind=20),
            BlindLevel(start_hand=3, small_blind=100, big_blind=200),
            BlindLevel(start_hand=5, small_blind=300, big_blind=600),
        ),
        master_seed=8, model_id="stub:tight_passive",
        tournament_id="heads-up-short",
    )
    result = await run_tournament(
        cfg, {"stub:tight_passive": TightPassiveAgent()}
    )
    events = _replay(result.log_path)
    session_end = next((e for e in events if e.type == "session_end"), None)
    assert session_end is not None, "heads-up short-stack: no session_end emitted"
    total = sum(session_end.final_stacks.values())  # type: ignore[attr-defined]
    assert total == 2000, (
        f"heads-up short-stack: total chips {total}, expected 2000"
    )


# ---------------------------------------------------------------------------
# 9. tournament_id propagation — TournamentStart and TournamentEnd agree on id.
# ---------------------------------------------------------------------------


async def test_tournament_id_consistent_in_start_and_end(tmp_path: Path) -> None:
    cfg = _make_config(
        tmp_path, seat_count=3, hand_cap=3, session_count=2,
        master_seed=17, tournament_id="my-cool-tournament",
    )
    result = await run_tournament(cfg, {"stub:tight_passive": TightPassiveAgent()})
    events = _replay(result.log_path)
    start = next((e for e in events if e.type == "tournament_start"), None)
    end = next((e for e in events if e.type == "tournament_end"), None)
    assert start is not None and end is not None
    assert start.tournament_id == "my-cool-tournament", (  # type: ignore[attr-defined]
        f"tournament_start.tournament_id={start.tournament_id}, "  # type: ignore[attr-defined]
        f"expected 'my-cool-tournament'"
    )
    assert end.tournament_id == start.tournament_id, (  # type: ignore[attr-defined]
        f"tournament_end.tournament_id={end.tournament_id} != "  # type: ignore[attr-defined]
        f"tournament_start.tournament_id={start.tournament_id}"  # type: ignore[attr-defined]
    )


# ---------------------------------------------------------------------------
# 10. hand_cap respected per match.
# ---------------------------------------------------------------------------


async def test_hand_count_per_match_at_most_hand_cap(tmp_path: Path) -> None:
    """No match emits more than hand_cap hand_start events."""
    hand_cap = 5
    cfg = _make_config(
        tmp_path, seat_count=3, hand_cap=hand_cap, session_count=3,
        master_seed=23, tournament_id="hand-cap",
    )
    result = await run_tournament(cfg, {"stub:tight_passive": TightPassiveAgent()})
    events = _replay(result.log_path)
    hands_per_session: dict[int, int] = defaultdict(int)
    current = 0
    for ev in events:
        if ev.type == "session_start":
            current = ev.session_id  # type: ignore[attr-defined]
        elif ev.type == "hand_start":
            hands_per_session[current] += 1
    for sid, count in hands_per_session.items():
        assert count <= hand_cap, (
            f"session {sid} played {count} hands, exceeds hand_cap={hand_cap}"
        )


# ---------------------------------------------------------------------------
# 11. session_end.total_hands matches actual hand_start count for that match.
# ---------------------------------------------------------------------------


async def test_session_end_total_hands_matches_hand_start_count(
    tmp_path: Path,
) -> None:
    cfg = _make_config(
        tmp_path, seat_count=3, hand_cap=6, session_count=2,
        master_seed=29, tournament_id="total-hands",
    )
    result = await run_tournament(cfg, {"stub:tight_passive": TightPassiveAgent()})
    events = _replay(result.log_path)
    counts: dict[int, int] = defaultdict(int)
    current = 0
    for ev in events:
        if ev.type == "session_start":
            current = ev.session_id  # type: ignore[attr-defined]
        elif ev.type == "hand_start":
            counts[current] += 1
        elif ev.type == "session_end":
            sid = ev.session_id  # type: ignore[attr-defined]
            reported = ev.total_hands  # type: ignore[attr-defined]
            assert reported == counts[sid], (
                f"session {sid}: total_hands={reported} but counted "
                f"{counts[sid]} hand_start events"
            )
