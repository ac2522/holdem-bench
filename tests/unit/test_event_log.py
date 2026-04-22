"""Tests for append-only JSONL event log."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from holdembench.events.log import EventLog
from holdembench.events.schema import HandEnd, SessionStart, TournamentStart

NUM_LINES_TWO_EVENTS = 2
NUM_EVENTS_TWO = 2


def _ts() -> TournamentStart:
    return TournamentStart(
        tournament_id="t1",
        schema_version="1.0",
        holdembench_version="0.1.0",
        pokerkit_version="0.7.3",
        git_sha="deadbeef",
        seat_assignments={"Seat1": "stub:random"},
        master_seed=42,
        anonymization_salt="cafebabe",
        canary_uuid="11111111-1111-1111-1111-111111111111",
    )


def test_eventlog_appends_jsonl_lines(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with EventLog(path) as log:
        log.emit(_ts())
        log.emit(SessionStart(
            session_id=1, hand_cap=10, small_blind=10, big_blind=20,
            ante=0, deal_pack_seed=1,
        ))

    lines = path.read_text().splitlines()
    assert len(lines) == NUM_LINES_TWO_EVENTS
    first = json.loads(lines[0])
    assert first["type"] == "tournament_start"
    assert first["tournament_id"] == "t1"


def test_eventlog_replay_yields_typed_events(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with EventLog(path) as log:
        log.emit(_ts())
    events = list(EventLog.replay(path))
    assert len(events) == 1
    assert isinstance(events[0], TournamentStart)


def test_eventlog_is_append_only(tmp_path: Path) -> None:
    """A second open() session appends without truncating."""
    path = tmp_path / "events.jsonl"
    with EventLog(path) as log:
        log.emit(_ts())
    with EventLog(path) as log:
        log.emit(HandEnd(
            hand_id="s1h001",
            stack_deltas={"Seat1": 0},
            elapsed_s=0.1,
            total_cost_usd=0.0,
        ))
    events = list(EventLog.replay(path))
    assert len(events) == NUM_EVENTS_TWO


def test_eventlog_sha256_hex_matches_file(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with EventLog(path) as log:
        log.emit(_ts())
    digest = EventLog.sha256_hex(path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == expected


def test_eventlog_rejects_extra_keys(tmp_path: Path) -> None:
    """Raw dict missing `type` rejected on replay."""
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({"not_type": "oops"}) + "\n")
    with pytest.raises(ValueError, match="type"):
        list(EventLog.replay(path))
