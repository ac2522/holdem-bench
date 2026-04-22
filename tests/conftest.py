"""Shared pytest fixtures for HoldEmBench tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def deterministic_rng() -> np.random.Generator:
    """A fixed-seed numpy RNG for deterministic unit tests."""
    return np.random.default_rng(42)


@pytest.fixture
def tmp_event_log_path(tmp_path: Path) -> Path:
    """A fresh path for an event log under pytest tmp_path."""
    return tmp_path / "events.jsonl"
