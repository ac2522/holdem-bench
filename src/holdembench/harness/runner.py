"""Tournament runner — game loop wiring engine + chat + event log + agents.

Determinism notes
-----------------
``TournamentConfig.deterministic_time`` (default ``True``) zeroes out all
wall-clock-dependent fields (``HandEnd.elapsed_s``, ``TournamentEnd.wall_clock_s``)
and replaces the live ``_git_sha()`` call with the constant ``"deterministic"``
so that byte-identical logs are produced for the same ``master_seed`` +
identical agent RNG seeds, regardless of when the run executes.

Phase-0 simplifications
-----------------------
- ``DecisionContext.street`` is always ``"preflop"``; street tracking arrives
  in Phase 1.
- ``_legal_actions()`` is conservative: always ``["fold", "check"|"call",
  "raise"]``; fine-grained legality is already enforced by pokerkit/TDAValidator.

Pokerkit ``state.stacks`` semantics (v0.7.3)
--------------------------------------------
With the ``CHIPS_PUSHING`` + ``CHIPS_PULLING`` automations enabled, pokerkit
distributes the pot immediately when the hand ends.  Reading ``state.stacks``
after the hand loop terminates therefore gives correct final chip counts.
Empirically verified: after a fold with SB=10/BB=20 starting 1000 each,
``state.stacks`` shows ``[1010, 990]`` (winner gained 20 net, loser lost 10 net
relative to pre-post starting stacks; absolute from the 1000 each starting point
they are 1010 and 990).
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pokerkit

import holdembench
from holdembench.agents.base import Agent, DecisionContext
from holdembench.agents.prompt import SessionContext, TournamentContext
from holdembench.chat.content import ContentRejection, validate_content
from holdembench.chat.protocol import ChatProtocol, ChatRuleViolation
from holdembench.engine.config import TableConfig
from holdembench.engine.deck import shuffled_deck
from holdembench.engine.table import Table
from holdembench.engine.validator import RawDecision, TDAValidator, ValidationError
from holdembench.events.log import EventLog
from holdembench.events.schema import (
    ActionRequest,
    ActionResponse,
    AutoFold,
    BudgetCircuitBreak,
    HandEnd,
    HandStart,
    SessionEnd,
    SessionStart,
    TournamentEnd,
    TournamentStart,
    ValidatorRejection,
)
from holdembench.harness.manifest import write_manifest
from holdembench.types import ActionName

_SEAT_KEY_RE = re.compile(r"^Seat\d+$")


@dataclass(frozen=True)
class TournamentConfig:
    """Immutable tournament configuration."""

    tournament_id: str
    seats: dict[str, str]  # "SeatN" -> model_id
    small_blind: int
    big_blind: int
    ante: int
    starting_stack: int
    hand_cap: int
    session_count: int
    master_seed: int
    results_dir: Path
    schema_version: str = "1.0"
    # When True (default), zero out elapsed_s / wall_clock_s and use a fixed
    # git_sha so byte-identical logs are produced for identical seeds.
    deterministic_time: bool = True
    # Per-model USD ceilings (spec §8.5).  Breach of 2x ceiling triggers
    # BudgetCircuitBreak + AutoFold for the seat's remaining hands.
    budget_ceilings_usd: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if not self.seats:
            raise ValueError("TournamentConfig.seats must be non-empty")
        for key in self.seats:
            if not _SEAT_KEY_RE.match(key):
                raise ValueError(
                    f"TournamentConfig.seats key {key!r} does not match 'SeatN' pattern"
                )
        if self.small_blind <= 0:
            raise ValueError(f"small_blind must be > 0, got {self.small_blind}")
        if self.big_blind < self.small_blind:
            raise ValueError(
                f"big_blind ({self.big_blind}) must be >= small_blind ({self.small_blind})"
            )
        if self.ante < 0:
            raise ValueError(f"ante must be >= 0, got {self.ante}")
        if self.starting_stack <= 0:
            raise ValueError(f"starting_stack must be > 0, got {self.starting_stack}")
        if self.hand_cap <= 0:
            raise ValueError(f"hand_cap must be > 0, got {self.hand_cap}")
        if self.budget_ceilings_usd is not None:
            for mid, v in self.budget_ceilings_usd.items():
                if v <= 0:
                    raise ValueError(f"budget_ceilings_usd[{mid}] must be > 0, got {v}")
        if self.session_count <= 0:
            raise ValueError(f"session_count must be > 0, got {self.session_count}")


@dataclass(frozen=True)
class TournamentResult:
    """Returned by :func:`run_tournament`."""

    log_path: Path
    manifest_path: Path
    final_chip_totals: dict[str, int]
    per_model_cost: dict[str, dict[str, float]]
    total_cost_usd: float


def _empty_stats() -> dict[str, float]:
    return {
        "input_tokens": 0.0,
        "output_tokens": 0.0,
        "cache_read_tokens": 0.0,
        "cache_write_tokens": 0.0,
        "thinking_tokens": 0.0,
        "usd_total": 0.0,
        "retries": 0.0,
        "timeouts": 0.0,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _canary_uuid(tid: str, seed: int) -> str:
    digest = hashlib.sha256(f"{tid}:{seed}".encode()).digest()[:16]
    return str(uuid.UUID(bytes=digest))


def _anon_salt(tid: str, seed: int) -> str:
    return hashlib.sha256(f"anon:{tid}:{seed}".encode()).hexdigest()[:16]


def _get_version() -> str:
    return holdembench.__version__


def _get_pokerkit_version() -> str:
    return getattr(pokerkit, "__version__", "unknown")


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _legal_actions(table: Table) -> list[ActionName]:
    """Return a conservative legal-action list.

    "raise" is omitted when every opponent is already all-in-covered;
    pokerkit signals this via ``min_completion_betting_or_raising_to_amount=None``.
    See :meth:`Table.can_raise`.
    """
    actions: list[ActionName] = ["fold"]
    if table.current_bet() == 0:
        actions.append("check")
    else:
        actions.append("call")
    if table.can_raise():
        actions.append("raise")
    return actions


_POKERKIT_NO_REASON_TO_FOLD = "no reason for this player to fold"
_POKERKIT_ALREADY_COVERED = "already covered by a previous bet/raise"


def _apply_raw_to_table(table: Table, idx: int, raw: RawDecision) -> None:
    """Translate a validated ``RawDecision`` into a pokerkit state mutation.

    Pokerkit (in TOURNAMENT mode) raises ``ValueError("There is no reason for
    this player to fold.")`` when a player tries to fold with no outstanding
    bet to call (e.g. BB checking back preflop).  We downgrade that specific
    case to ``check_or_call``.  Any other ValueError from ``apply_fold`` is
    re-raised so future pokerkit invariants are not silently swallowed.
    """
    if raw.kind != "action":
        return  # probes are not driven through pokerkit in Phase 0
    if raw.action == "fold":
        try:
            table.apply_fold(idx)
        except ValueError as exc:
            if _POKERKIT_NO_REASON_TO_FOLD not in str(exc).lower():
                raise
            # Known pokerkit rejection: downgrade to check_or_call.
            table.apply_check_or_call(idx)
    elif raw.action in {"check", "call"}:
        table.apply_check_or_call(idx)
    elif raw.action == "raise" and raw.amount is not None:
        try:
            table.apply_raise(idx, to=raw.amount)
        except ValueError as exc:
            # If the LLM picked "raise" while every opponent is already
            # all-in-covered (a corner case _legal_actions now blocks but
            # could still surface via stale prompt context after a re-deal),
            # downgrade to check_or_call rather than crashing the run.
            if _POKERKIT_ALREADY_COVERED not in str(exc).lower():
                raise
            table.apply_check_or_call(idx)


def _compute_stack_deltas(
    table: Table,
    seat_list: list[str],
    running: dict[str, int],
) -> dict[str, int]:
    """Compute per-seat chip delta for the hand just completed.

    With ``CHIPS_PUSHING`` + ``CHIPS_PULLING`` automations active, pokerkit
    updates ``state.stacks`` immediately when the hand concludes, so we can
    read final chip totals directly.
    """
    state = table._state  # noqa: SLF001 — intentional read-only introspection  # type: ignore[reportPrivateUsage]
    return {s: int(state.stacks[i]) - running[s] for i, s in enumerate(seat_list)}


def _raw_to_dict(raw: RawDecision) -> dict[str, object]:
    return {
        "kind": raw.kind,
        "action": raw.action,
        "amount": raw.amount,
        "message": raw.message,
    }


async def _validate_with_retry(
    *,
    log: EventLog,
    agent: Agent,
    validator: TDAValidator,
    table: Table,
    chat: ChatProtocol,
    idx: int,
    seat_name: str,
    ctx: DecisionContext,
    raw: RawDecision,
) -> RawDecision | None:
    """Validate *raw* against the TDA rules.  Returns the validated decision or None on auto-fold.

    On first failure emits ValidatorRejection(retry_allowed=True) and re-calls the agent once.
    On second failure emits ValidatorRejection(retry_allowed=False) + AutoFold and applies
    an emergency fold (falling back to check_or_call when fold is not legal).

    Returns ``None`` when the action was auto-folded (caller should ``continue``).
    """
    try:
        validator.check(idx, raw)
        return raw
    except ValidationError as exc:
        log.emit(
            ValidatorRejection(
                seat=seat_name,
                reason=str(exc),
                original_response=_raw_to_dict(raw),
                retry_allowed=True,
            )
        )

    # Retry
    raw2 = await agent.decide(ctx)
    try:
        validator.check(idx, raw2)
        return raw2
    except ValidationError as exc2:
        log.emit(
            ValidatorRejection(
                seat=seat_name,
                reason=str(exc2),
                original_response=_raw_to_dict(raw2),
                retry_allowed=False,
            )
        )
        log.emit(AutoFold(seat=seat_name, reason="invalid_after_retry"))
        try:
            table.apply_fold(idx)
        except ValueError as exc:
            if _POKERKIT_NO_REASON_TO_FOLD not in str(exc).lower():
                raise
            # Cannot fold (BB checking back with no need to fold); downgrade.
            table.apply_check_or_call(idx)
        chat.mark_folded(seat_name)
        return None


def _filter_chat_message(
    *,
    log: EventLog,
    seat_name: str,
    raw: RawDecision,
    chat: ChatProtocol,
) -> str | None:
    """Validate content + spend chat budget.  Returns the message to attach (possibly None)."""
    message = raw.message
    if message is None:
        return None

    try:
        validate_content(message)
    except ContentRejection:
        log.emit(
            ValidatorRejection(
                seat=seat_name,
                reason="chat_content",
                original_response=_raw_to_dict(raw),
                retry_allowed=False,
            )
        )
        return None

    try:
        chat.spend(seat_name, message, kind=raw.kind)  # type: ignore[arg-type]
    except ChatRuleViolation:
        log.emit(
            ValidatorRejection(
                seat=seat_name,
                reason="chat_budget",
                original_response=_raw_to_dict(raw),
                retry_allowed=False,
            )
        )
        return None

    return message


_ACTION_TIMEOUT_S = 60.0
_BUDGET_BREACH_MULTIPLIER = 2.0


def _check_circuit_breaker(
    *,
    log: EventLog,
    seat_name: str,
    model_id: str,
    running_cost_by_seat: dict[str, float],
    ceilings: dict[str, float] | None,
    breached: set[str],
) -> bool:
    """Return True if *seat_name* is under the budget breaker.

    Emits :class:`BudgetCircuitBreak` exactly once, on first breach.
    The caller is responsible for the subsequent auto-fold.
    """
    if seat_name in breached:
        return True
    if ceilings is None:
        return False
    ceiling = ceilings.get(model_id)
    if ceiling is None:
        return False
    actual = running_cost_by_seat.get(seat_name, 0.0)
    threshold = _BUDGET_BREACH_MULTIPLIER * ceiling
    if actual > threshold:
        breached.add(seat_name)
        log.emit(
            BudgetCircuitBreak(
                seat=seat_name,
                threshold_usd=threshold,
                actual_usd=actual,
            )
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Hand-level helpers (extracted to keep run_tournament statement count low)
# ---------------------------------------------------------------------------


def _emit_action_response_and_update_stats(
    *,
    log: EventLog,
    agent_obj: Agent,
    hand_id: str,
    seat_name: str,
    raw: RawDecision,
    final_message: str | None,
    running_cost_by_seat: dict[str, float],
    per_model_stats: dict[str, dict[str, float]],
) -> float:
    """Emit ``ActionResponse`` + accumulate per-seat / per-model cost & usage.

    Returns the USD cost for this single decision (caller adds it to ``hand_cost``).
    """
    last_usage = getattr(agent_obj, "last_usage", None)
    tokens = int(getattr(last_usage, "output_tokens", 0) or 0)
    cost = float(getattr(agent_obj, "last_cost_usd", 0.0) or 0.0)
    thinking = getattr(agent_obj, "last_thinking", None)
    prompt_hash = str(getattr(agent_obj, "last_prompt_hash", "") or "")
    latency_ms = int(getattr(agent_obj, "last_latency_ms", 0) or 0)
    log.emit(
        ActionResponse(
            hand_id=hand_id,
            seat=seat_name,
            kind=raw.kind,
            action=raw.action,
            amount=raw.amount,
            message=final_message,
            tokens=tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            model_id=agent_obj.model_id,
            prompt_hash=prompt_hash,
            thinking=thinking,
        )
    )
    running_cost_by_seat[seat_name] = running_cost_by_seat.get(seat_name, 0.0) + cost
    if last_usage is not None:
        stats = per_model_stats.setdefault(agent_obj.model_id, _empty_stats())
        stats["input_tokens"] += int(getattr(last_usage, "input_tokens", 0) or 0)
        stats["output_tokens"] += int(getattr(last_usage, "output_tokens", 0) or 0)
        stats["cache_read_tokens"] += int(getattr(last_usage, "cache_read_tokens", 0) or 0)
        stats["cache_write_tokens"] += int(getattr(last_usage, "cache_write_tokens", 0) or 0)
        stats["thinking_tokens"] += int(getattr(last_usage, "thinking_tokens", 0) or 0)
        stats["usd_total"] += cost
        stats["retries"] += int(getattr(agent_obj, "last_parse_retries", 0) or 0)
    else:
        # Ensure stub-only models still show in the summary with zero totals.
        per_model_stats.setdefault(agent_obj.model_id, _empty_stats())
    return cost


async def _run_hand(
    *,
    cfg: TournamentConfig,
    log: EventLog,
    seat_list: list[str],
    seat_count: int,
    agents_by_seat: dict[str, Agent],
    chat: ChatProtocol,
    running_stacks: dict[str, int],
    running_cost_by_seat: dict[str, float],
    breached: set[str],
    per_model_stats: dict[str, dict[str, float]],
    session_id: int,
    hand_num: int,
) -> float:
    """Run one hand and return the hand cost (USD).  Mutates *running_stacks* in-place."""
    hand_id = f"s{session_id}h{hand_num:03d}"
    table_cfg = TableConfig(
        seat_count=seat_count,
        small_blind=cfg.small_blind,
        big_blind=cfg.big_blind,
        ante=cfg.ante,
        starting_stacks=tuple(running_stacks[s] for s in seat_list),
    )
    table = Table(table_cfg)
    validator = TDAValidator(table)

    if hand_num % seat_count == 1:
        chat.start_orbit()
    chat.start_hand(in_hand=set(seat_list))

    deck = shuffled_deck(seed=cfg.master_seed * 10_000 + hand_num)
    cards_hash = hashlib.sha256(",".join(deck[: seat_count * 2]).encode()).hexdigest()
    log.emit(
        HandStart(
            hand_id=hand_id,
            button_seat=(hand_num - 1) % seat_count,
            stacks=dict(running_stacks),
            cards_hash=f"sha256:{cards_hash}",
            chat_budgets_remaining={s: chat.budget_remaining(s) for s in seat_list},
        )
    )

    hand_wall_start = time.time()
    hand_cost = 0.0

    while not table.hand_is_over():
        idx = table.next_actor()
        if idx is None:
            break
        seat_name = seat_list[idx]
        agent_obj_pre = agents_by_seat[seat_name]
        if _check_circuit_breaker(
            log=log,
            seat_name=seat_name,
            model_id=agent_obj_pre.model_id,
            running_cost_by_seat=running_cost_by_seat,
            ceilings=cfg.budget_ceilings_usd,
            breached=breached,
        ):
            log.emit(AutoFold(seat=seat_name, reason="budget_circuit_break"))
            try:
                table.apply_fold(idx)
            except ValueError as exc:
                if _POKERKIT_NO_REASON_TO_FOLD not in str(exc).lower():
                    raise
                table.apply_check_or_call(idx)
            chat.mark_folded(seat_name)
            continue
        legal = _legal_actions(table)
        log.emit(
            ActionRequest(
                hand_id=hand_id,
                to_seat=seat_name,
                street="preflop",  # Phase 0: placeholder — street tracking in Phase 1
                legal=legal,
                timeout_s=_ACTION_TIMEOUT_S,
                budget_remaining=chat.budget_remaining(seat_name),
            )
        )
        ctx = DecisionContext(
            seat=seat_name,
            hand_id=hand_id,
            street="preflop",  # Phase 0: placeholder — street tracking in Phase 1
            legal=tuple(legal),
            stacks=dict(running_stacks),
            board=(),
            hole=tuple(deck[idx * 2 : idx * 2 + 2]),
            budget_remaining=chat.budget_remaining(seat_name),
            is_probe_reply=False,
            deadline_s=_ACTION_TIMEOUT_S,
        )
        raw = await agents_by_seat[seat_name].decide(ctx)
        validated = await _validate_with_retry(
            log=log,
            agent=agents_by_seat[seat_name],
            validator=validator,
            table=table,
            chat=chat,
            idx=idx,
            seat_name=seat_name,
            ctx=ctx,
            raw=raw,
        )
        if validated is None:
            continue
        raw = validated
        final_message = _filter_chat_message(log=log, seat_name=seat_name, raw=raw, chat=chat)
        cost = _emit_action_response_and_update_stats(
            log=log,
            agent_obj=agents_by_seat[seat_name],
            hand_id=hand_id,
            seat_name=seat_name,
            raw=raw,
            final_message=final_message,
            running_cost_by_seat=running_cost_by_seat,
            per_model_stats=per_model_stats,
        )
        hand_cost += cost
        _apply_raw_to_table(table, idx, raw)

        # Mark folded seats so they cannot chat for the rest of this hand
        if raw.action == "fold":
            chat.mark_folded(seat_name)

    elapsed = 0.0 if cfg.deterministic_time else time.time() - hand_wall_start
    deltas = _compute_stack_deltas(table, seat_list, running_stacks)
    for s, d in deltas.items():
        running_stacks[s] += d
    log.emit(
        HandEnd(hand_id=hand_id, stack_deltas=deltas, elapsed_s=elapsed, total_cost_usd=hand_cost)
    )
    return hand_cost


def _refresh_adapter_contexts(
    *,
    cfg: TournamentConfig,
    seat_list: list[str],
    agents_by_seat: dict[str, Agent],
    session_id: int,
) -> None:
    """Call ``set_context(tournament=..., session=...)`` on each adapter that
    supports it, so the rendered prompt reflects the current session id.

    Adapters keyed by the same ``model_id`` share a single agent instance; the
    first seat encountered wins the seat-identity slot (tracked as P1.1-B).
    """
    seen_models: set[str] = set()
    for seat in seat_list:
        agent = agents_by_seat[seat]
        if not hasattr(agent, "set_context"):
            continue
        if agent.model_id in seen_models:
            continue
        seen_models.add(agent.model_id)
        tournament = TournamentContext(
            tournament_id=cfg.tournament_id,
            seat=seat,
            seat_count=len(cfg.seats),
        )
        session = SessionContext(
            session_id=session_id,
            small_blind=cfg.small_blind,
            big_blind=cfg.big_blind,
            ante=cfg.ante,
            starting_stack_bb=max(1, cfg.starting_stack // cfg.big_blind),
            orbit_budget_tokens=400,
        )
        agent.set_context(tournament=tournament, session=session)  # type: ignore[attr-defined]


async def _run_session(
    *,
    cfg: TournamentConfig,
    log: EventLog,
    seat_list: list[str],
    seat_count: int,
    agents_by_seat: dict[str, Agent],
    running_stacks: dict[str, int],
    running_cost_by_seat: dict[str, float],
    breached: set[str],
    per_model_stats: dict[str, dict[str, float]],
    session_id: int,
) -> float:
    """Run one session and return the session cost (USD).  Mutates *running_stacks* in-place."""
    _refresh_adapter_contexts(
        cfg=cfg,
        seat_list=seat_list,
        agents_by_seat=agents_by_seat,
        session_id=session_id,
    )
    log.emit(
        SessionStart(
            session_id=session_id,
            hand_cap=cfg.hand_cap,
            small_blind=cfg.small_blind,
            big_blind=cfg.big_blind,
            ante=cfg.ante,
            deal_pack_seed=cfg.master_seed * 1000 + session_id,
        )
    )
    chat = ChatProtocol(seats=tuple(seat_list), budget_per_orbit=400, per_action_cap=80)
    session_cost = 0.0
    hands_played = 0
    for hand_num in range(1, cfg.hand_cap + 1):
        # Pokerkit refuses to deal a hand with any non-positive starting stack.
        # When a seat busts we end the session early — chip-EV semantics are
        # preserved (no auto-rebuy) and downstream scoring sees the true totals.
        # Proper tournament elimination (re-seating, button rotation) is tracked
        # as a Phase 1.1 follow-up (P1.1-C).
        if any(running_stacks[s] <= 0 for s in seat_list):
            break
        session_cost += await _run_hand(
            cfg=cfg,
            log=log,
            seat_list=seat_list,
            seat_count=seat_count,
            agents_by_seat=agents_by_seat,
            chat=chat,
            running_stacks=running_stacks,
            running_cost_by_seat=running_cost_by_seat,
            breached=breached,
            per_model_stats=per_model_stats,
            session_id=session_id,
            hand_num=hand_num,
        )
        hands_played += 1
    log.emit(
        SessionEnd(
            session_id=session_id,
            final_stacks=dict(running_stacks),
            total_hands=hands_played,
            total_cost_usd=session_cost,
        )
    )
    return session_cost


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_tournament(
    cfg: TournamentConfig,
    agents: Mapping[str, Agent],
) -> TournamentResult:
    """Run a full tournament and return the paths to the resulting log + manifest.

    Parameters
    ----------
    cfg:
        Immutable tournament configuration.
    agents:
        Mapping from *model_id* (the values in ``cfg.seats``) to :class:`Agent`
        instances.  The caller owns the agent objects; this function never
        constructs or destroys them.
    """
    results_dir = cfg.results_dir / cfg.tournament_id
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "events.jsonl"
    manifest_path = results_dir / "manifest.json"
    if log_path.exists():
        log_path.unlink()  # Always start fresh so re-runs are idempotent.

    seat_list = sorted(cfg.seats.keys(), key=lambda s: int(s.removeprefix("Seat")))
    seat_count = len(seat_list)
    agents_by_seat = {s: agents[cfg.seats[s]] for s in seat_list}

    holdembench_ver = _get_version()
    pokerkit_ver = _get_pokerkit_version()
    git_sha = "deterministic" if cfg.deterministic_time else _git_sha()
    wall_clock_start = time.time()

    with EventLog(log_path) as log:
        log.emit(
            TournamentStart(
                tournament_id=cfg.tournament_id,
                schema_version=cfg.schema_version,
                holdembench_version=holdembench_ver,
                pokerkit_version=pokerkit_ver,
                git_sha=git_sha,
                seat_assignments=dict(cfg.seats),
                master_seed=cfg.master_seed,
                anonymization_salt=_anon_salt(cfg.tournament_id, cfg.master_seed),
                canary_uuid=_canary_uuid(cfg.tournament_id, cfg.master_seed),
            )
        )
        running_stacks = {s: cfg.starting_stack for s in seat_list}
        running_cost_by_seat: dict[str, float] = {s: 0.0 for s in seat_list}
        breached: set[str] = set()
        per_model_stats: dict[str, dict[str, float]] = {}
        total_cost = 0.0
        for session_id in range(1, cfg.session_count + 1):
            total_cost += await _run_session(
                cfg=cfg,
                log=log,
                seat_list=seat_list,
                seat_count=seat_count,
                agents_by_seat=agents_by_seat,
                running_stacks=running_stacks,
                running_cost_by_seat=running_cost_by_seat,
                breached=breached,
                per_model_stats=per_model_stats,
                session_id=session_id,
            )
        winner_seat = max(running_stacks, key=lambda s: running_stacks[s])
        wall_clock_s = 0.0 if cfg.deterministic_time else time.time() - wall_clock_start
        log.emit(
            TournamentEnd(
                tournament_id=cfg.tournament_id,
                final_chip_totals=dict(running_stacks),
                winner_seat=winner_seat,
                winner_model=cfg.seats[winner_seat],
                total_cost_usd=total_cost,
                wall_clock_s=wall_clock_s,
            )
        )

    write_manifest(
        log_path=log_path,
        manifest_path=manifest_path,
        tournament_id=cfg.tournament_id,
        schema_version=cfg.schema_version,
        holdembench_version=holdembench_ver,
        pokerkit_version=pokerkit_ver,
        seat_assignments=dict(cfg.seats),
        master_seed=cfg.master_seed,
        canary_uuid=_canary_uuid(cfg.tournament_id, cfg.master_seed),
    )
    return TournamentResult(
        log_path=log_path,
        manifest_path=manifest_path,
        final_chip_totals=dict(running_stacks),
        per_model_cost=per_model_stats,
        total_cost_usd=total_cost,
    )
