"""Stub-only end-to-end integration — full tournament via CLI."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest


@pytest.mark.slow
def test_cli_end_to_end_produces_valid_artifacts(tmp_path: Path) -> None:
    """Run a stub tournament via CLI and verify artifacts are valid.

    Checks:
    - CLI returns 0
    - events.jsonl and manifest.json are produced
    - manifest SHA-256 matches actual events.jsonl content
    """
    cfg = Path(__file__).parents[2] / "evals" / "stub-phase0-smoke.yaml"
    results = tmp_path / "results"
    r = subprocess.run(
        ["uv", "run", "holdembench", "run", "--config", str(cfg),
         "--results-dir", str(results)],
        capture_output=True, text=True, timeout=120,
        check=False,
    )
    assert r.returncode == 0, f"CLI failed: {r.stderr}"

    base = results / "stub-phase0-smoke"
    log = base / "events.jsonl"
    manifest = base / "manifest.json"

    assert log.exists(), f"events.jsonl not found at {log}"
    assert manifest.exists(), f"manifest.json not found at {manifest}"
    # Manifest SHA matches log content
    claimed = json.loads(manifest.read_text())["events_sha256"]
    actual = hashlib.sha256(log.read_bytes()).hexdigest()
    assert claimed == actual, "manifest SHA does not match log contents"


@pytest.mark.slow
def test_byte_identical_on_reseed(tmp_path: Path) -> None:
    """Run the same stub tournament twice and verify byte-identical output.

    With deterministic_time=True (CLI default) and fixed seeds, the output
    should be byte-for-byte identical.
    """
    cfg = Path(__file__).parents[2] / "evals" / "stub-phase0-smoke.yaml"
    for name in ("a", "b"):
        subprocess.run(
            ["uv", "run", "holdembench", "run", "--config", str(cfg),
             "--results-dir", str(tmp_path / name)],
            check=True, timeout=120,
        )
    a_log = (tmp_path / "a" / "stub-phase0-smoke" / "events.jsonl").read_bytes()
    b_log = (tmp_path / "b" / "stub-phase0-smoke" / "events.jsonl").read_bytes()
    assert a_log == b_log, "stub tournament is non-deterministic under same seed"
