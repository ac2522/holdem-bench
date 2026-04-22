"""Manifest.json writer + verifier — binds tournament metadata to event-log SHA."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from holdembench.events.log import EventLog


@dataclass(frozen=True)
class Manifest:
    tournament_id: str
    schema_version: str
    holdembench_version: str
    pokerkit_version: str
    seat_assignments: dict[str, str]
    master_seed: int
    canary_uuid: str
    events_sha256: str


def write_manifest(
    *,
    log_path: Path,
    manifest_path: Path,
    tournament_id: str,
    schema_version: str,
    holdembench_version: str,
    pokerkit_version: str,
    seat_assignments: dict[str, str],
    master_seed: int,
    canary_uuid: str,
) -> Manifest:
    m = Manifest(
        tournament_id=tournament_id,
        schema_version=schema_version,
        holdembench_version=holdembench_version,
        pokerkit_version=pokerkit_version,
        seat_assignments=dict(seat_assignments),
        master_seed=master_seed,
        canary_uuid=canary_uuid,
        events_sha256=EventLog.sha256_hex(log_path),
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(asdict(m), indent=2, sort_keys=True))
    return m


def verify_manifest(*, log_path: Path, manifest_path: Path) -> bool:
    claimed = json.loads(manifest_path.read_text())["events_sha256"]
    actual = EventLog.sha256_hex(log_path)
    return claimed == actual
