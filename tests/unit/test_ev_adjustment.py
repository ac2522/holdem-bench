"""Tests for all-in Monte Carlo equity computation."""
from __future__ import annotations

import pytest

from holdembench.engine.ev_adjustment import monte_carlo_equity

# Equity bounds for AA vs KK pre-flop (true equity ≈ 82%)
AA_VS_KK_ACES_LO = 0.78
AA_VS_KK_ACES_HI = 0.86
AA_VS_KK_KINGS_LO = 0.14
AA_VS_KK_KINGS_HI = 0.22

# Equity bounds for AhKh flush draw vs QsQd overpair on turn (≈ 20%)
FLUSH_DRAW_LO = 0.12
FLUSH_DRAW_HI = 0.30

# Tolerance for sum-to-1 check
SUM_TOLERANCE = 1e-6


def test_aces_vs_kings_preflop_aces_win_majority() -> None:
    """AA vs KK pre-flop: aces win ≈ 82%."""
    eq = monte_carlo_equity(
        hole_cards=[["As", "Ad"], ["Ks", "Kd"]],
        board=[],
        samples=5_000,
        seed=1,
    )
    assert AA_VS_KK_ACES_LO <= eq[0] <= AA_VS_KK_ACES_HI
    assert AA_VS_KK_KINGS_LO <= eq[1] <= AA_VS_KK_KINGS_HI
    assert abs(sum(eq) - 1.0) < SUM_TOLERANCE


def test_flush_draw_vs_overpair_turn() -> None:
    """AhKh (flush draw) vs QsQd on a Qh-7h-2c-3d turn.

    Flush draw has roughly 9/44 ≈ 20% to river a flush. Accept a wide band.
    """
    eq = monte_carlo_equity(
        hole_cards=[["Ah", "Kh"], ["Qs", "Qd"]],
        board=["Qh", "7h", "2c", "3d"],
        samples=5_000,
        seed=1,
    )
    assert FLUSH_DRAW_LO <= eq[0] <= FLUSH_DRAW_HI
    assert abs(sum(eq) - 1.0) < SUM_TOLERANCE


def test_river_already_decided() -> None:
    """On the river with no cards to come, one player has exactly 100% equity."""
    eq = monte_carlo_equity(
        hole_cards=[["As", "Ks"], ["2d", "3d"]],
        board=["Ah", "Kh", "Qc", "9d", "5c"],
        samples=100,
        seed=1,
    )
    assert eq[0] == pytest.approx(1.0)
    assert eq[1] == pytest.approx(0.0)


def test_equity_is_deterministic_with_seed() -> None:
    eq1 = monte_carlo_equity([["As", "Ad"], ["Ks", "Kd"]], [], 1000, seed=42)
    eq2 = monte_carlo_equity([["As", "Ad"], ["Ks", "Kd"]], [], 1000, seed=42)
    assert eq1 == eq2
