"""TournamentConfig blind-level validation.

Without ordering / value validation, ``_blinds_for_hand`` would silently
pick the last qualifying level in tuple-iteration order rather than the
highest ``start_hand``.  See audit finding "blind levels last-wins".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from holdembench.harness.runner import BlindLevel, TournamentConfig


def _base_kwargs(tmp_path: Path) -> dict[str, object]:
    return {
        "tournament_id": "t",
        "seats": {"Seat1": "stub:tight_passive", "Seat2": "stub:tight_passive"},
        "small_blind": 10,
        "big_blind": 20,
        "ante": 0,
        "starting_stack": 1000,
        "hand_cap": 5,
        "session_count": 1,
        "master_seed": 1,
        "results_dir": tmp_path,
    }


def test_unordered_blind_levels_rejected(tmp_path: Path) -> None:
    """If levels are not strictly ascending by start_hand, reject."""
    with pytest.raises(ValueError, match="strictly ascending"):
        TournamentConfig(
            **_base_kwargs(tmp_path),  # type: ignore[arg-type]
            blind_levels=(
                BlindLevel(start_hand=1, small_blind=10, big_blind=20),
                BlindLevel(start_hand=81, small_blind=40, big_blind=80),
                BlindLevel(start_hand=41, small_blind=20, big_blind=40),
            ),
        )


def test_zero_start_hand_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="start_hand must be >= 1"):
        TournamentConfig(
            **_base_kwargs(tmp_path),  # type: ignore[arg-type]
            blind_levels=(BlindLevel(start_hand=0, small_blind=10, big_blind=20),),
        )


def test_bb_less_than_sb_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="big_blind"):
        TournamentConfig(
            **_base_kwargs(tmp_path),  # type: ignore[arg-type]
            blind_levels=(BlindLevel(start_hand=1, small_blind=20, big_blind=10),),
        )


def test_sb_zero_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="small_blind must be > 0"):
        TournamentConfig(
            **_base_kwargs(tmp_path),  # type: ignore[arg-type]
            blind_levels=(BlindLevel(start_hand=1, small_blind=0, big_blind=20),),
        )


def test_well_formed_ladder_accepted(tmp_path: Path) -> None:
    cfg = TournamentConfig(
        **_base_kwargs(tmp_path),  # type: ignore[arg-type]
        blind_levels=(
            BlindLevel(start_hand=1, small_blind=10, big_blind=20),
            BlindLevel(start_hand=41, small_blind=20, big_blind=40),
            BlindLevel(start_hand=81, small_blind=40, big_blind=80),
        ),
    )
    assert cfg.blind_levels is not None
