"""Tests for manifest.json writer + integrity check."""

from __future__ import annotations

import json
from pathlib import Path

from holdembench.events.log import EventLog
from holdembench.events.schema import SessionStart, TournamentStart
from holdembench.harness.manifest import verify_manifest, write_manifest


def _write_simple_log(path: Path) -> None:
    with EventLog(path) as log:
        log.emit(
            TournamentStart(
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
        )
        log.emit(
            SessionStart(
                session_id=1,
                hand_cap=10,
                small_blind=10,
                big_blind=20,
                ante=0,
                deal_pack_seed=1,
            )
        )


def test_write_manifest_produces_matching_sha(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    _write_simple_log(log_path)
    manifest_path = tmp_path / "manifest.json"
    m = write_manifest(
        log_path=log_path,
        manifest_path=manifest_path,
        tournament_id="t1",
        holdembench_version="0.1.0",
        pokerkit_version="0.7.3",
        schema_version="1.0",
        seat_assignments={"Seat1": "stub:random"},
        master_seed=42,
        canary_uuid="11111111-1111-1111-1111-111111111111",
    )
    raw = json.loads(manifest_path.read_text())
    assert raw["events_sha256"] == m.events_sha256
    assert m.tournament_id == "t1"


def test_verify_manifest_passes_on_unmodified_file(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    _write_simple_log(log_path)
    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        log_path=log_path,
        manifest_path=manifest_path,
        tournament_id="t1",
        holdembench_version="0.1.0",
        pokerkit_version="0.7.3",
        schema_version="1.0",
        seat_assignments={"Seat1": "stub:random"},
        master_seed=42,
        canary_uuid="11111111-1111-1111-1111-111111111111",
    )
    assert verify_manifest(log_path=log_path, manifest_path=manifest_path)


def test_verify_manifest_fails_after_tamper(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    _write_simple_log(log_path)
    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        log_path=log_path,
        manifest_path=manifest_path,
        tournament_id="t1",
        holdembench_version="0.1.0",
        pokerkit_version="0.7.3",
        schema_version="1.0",
        seat_assignments={"Seat1": "stub:random"},
        master_seed=42,
        canary_uuid="11111111-1111-1111-1111-111111111111",
    )
    # Tamper with the log
    log_path.write_text(log_path.read_text() + "\n")
    assert not verify_manifest(log_path=log_path, manifest_path=manifest_path)
