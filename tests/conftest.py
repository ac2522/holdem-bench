"""Shared pytest fixtures + --runslow flag."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def deterministic_rng() -> np.random.Generator:
    """A fixed-seed numpy RNG for deterministic unit tests."""
    return np.random.default_rng(42)


@pytest.fixture
def tmp_event_log_path(tmp_path: Path) -> Path:
    """A fresh path for an event log under pytest tmp_path."""
    return tmp_path / "events.jsonl"
