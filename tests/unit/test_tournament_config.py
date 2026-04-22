"""Tests for TournamentConfig.__post_init__ validation (Fix 6)."""
from __future__ import annotations

from pathlib import Path

import pytest

from holdembench.harness.runner import TournamentConfig


def _valid_kwargs(tmp_path: Path) -> dict[str, object]:
    return dict(
        tournament_id="t-valid",
        seats={"Seat1": "model:a", "Seat2": "model:b"},
        small_blind=10,
        big_blind=20,
        ante=0,
        starting_stack=1000,
        hand_cap=5,
        session_count=1,
        master_seed=42,
        results_dir=tmp_path / "results",
    )


def test_tournament_config_accepts_valid(tmp_path: Path) -> None:
    cfg = TournamentConfig(**_valid_kwargs(tmp_path))  # type: ignore[arg-type]
    assert cfg.tournament_id == "t-valid"


def test_tournament_config_rejects_empty_seats(tmp_path: Path) -> None:
    kwargs = _valid_kwargs(tmp_path)
    kwargs["seats"] = {}
    with pytest.raises(ValueError, match="seats must be non-empty"):
        TournamentConfig(**kwargs)  # type: ignore[arg-type]


def test_tournament_config_rejects_bad_seat_key_format(tmp_path: Path) -> None:
    kwargs = _valid_kwargs(tmp_path)
    kwargs["seats"] = {"Player1": "model:a", "Seat2": "model:b"}
    with pytest.raises(ValueError, match="SeatN"):
        TournamentConfig(**kwargs)  # type: ignore[arg-type]


def test_tournament_config_rejects_bad_seat_key_no_digit(tmp_path: Path) -> None:
    kwargs = _valid_kwargs(tmp_path)
    kwargs["seats"] = {"Seat": "model:a"}
    with pytest.raises(ValueError, match="SeatN"):
        TournamentConfig(**kwargs)  # type: ignore[arg-type]


def test_tournament_config_rejects_zero_small_blind(tmp_path: Path) -> None:
    kwargs = _valid_kwargs(tmp_path)
    kwargs["small_blind"] = 0
    with pytest.raises(ValueError, match="small_blind"):
        TournamentConfig(**kwargs)  # type: ignore[arg-type]


def test_tournament_config_rejects_big_blind_below_small_blind(tmp_path: Path) -> None:
    kwargs = _valid_kwargs(tmp_path)
    kwargs["small_blind"] = 20
    kwargs["big_blind"] = 10
    with pytest.raises(ValueError, match="big_blind"):
        TournamentConfig(**kwargs)  # type: ignore[arg-type]


def test_tournament_config_rejects_negative_ante(tmp_path: Path) -> None:
    kwargs = _valid_kwargs(tmp_path)
    kwargs["ante"] = -1
    with pytest.raises(ValueError, match="ante"):
        TournamentConfig(**kwargs)  # type: ignore[arg-type]


def test_tournament_config_rejects_non_positive_stacks_or_blinds(tmp_path: Path) -> None:
    kwargs = _valid_kwargs(tmp_path)
    kwargs["starting_stack"] = 0
    with pytest.raises(ValueError, match="starting_stack"):
        TournamentConfig(**kwargs)  # type: ignore[arg-type]


def test_tournament_config_rejects_zero_hand_cap(tmp_path: Path) -> None:
    kwargs = _valid_kwargs(tmp_path)
    kwargs["hand_cap"] = 0
    with pytest.raises(ValueError, match="hand_cap"):
        TournamentConfig(**kwargs)  # type: ignore[arg-type]


def test_tournament_config_rejects_zero_session_count(tmp_path: Path) -> None:
    kwargs = _valid_kwargs(tmp_path)
    kwargs["session_count"] = 0
    with pytest.raises(ValueError, match="session_count"):
        TournamentConfig(**kwargs)  # type: ignore[arg-type]


def test_tournament_config_accepts_equal_blinds(tmp_path: Path) -> None:
    """big_blind == small_blind is valid (straddle-like config)."""
    equal_blind = 20
    kwargs = _valid_kwargs(tmp_path)
    kwargs["small_blind"] = equal_blind
    kwargs["big_blind"] = equal_blind
    cfg = TournamentConfig(**kwargs)  # type: ignore[arg-type]
    assert cfg.small_blind == equal_blind


def test_tournament_config_accepts_zero_ante(tmp_path: Path) -> None:
    kwargs = _valid_kwargs(tmp_path)
    kwargs["ante"] = 0
    cfg = TournamentConfig(**kwargs)  # type: ignore[arg-type]
    assert cfg.ante == 0
