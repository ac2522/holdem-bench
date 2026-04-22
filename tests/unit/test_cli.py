"""Tests for the holdembench CLI entry point."""
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from holdembench.cli import cli


def test_cli_shows_help() -> None:
    r = CliRunner().invoke(cli, ["--help"])
    assert r.exit_code == 0
    assert "run" in r.output


@pytest.mark.slow
def test_cli_run_stub_config(tmp_path: Path) -> None:
    cfg = Path(__file__).parents[2] / "evals" / "stub-phase0-smoke.yaml"
    r = CliRunner().invoke(
        cli,
        ["run", "--config", str(cfg), "--results-dir", str(tmp_path / "results")],
    )
    assert r.exit_code == 0, r.output
    log = tmp_path / "results" / "stub-phase0-smoke" / "events.jsonl"
    manifest = tmp_path / "results" / "stub-phase0-smoke" / "manifest.json"
    assert log.exists()
    assert manifest.exists()
