"""Tests for multi-way Plackett-Luce rating."""
from __future__ import annotations

import pytest

from holdembench.scoring.multi_way_elo import fit_plackett_luce


def test_pl_single_ranking_produces_ratings() -> None:
    rankings = [["A", "B", "C", "D"]]  # A beat B beat C beat D
    ratings = fit_plackett_luce(rankings)
    assert ratings["A"] > ratings["B"] > ratings["C"] > ratings["D"]


def test_pl_consistent_winner_ranks_highest() -> None:
    rankings = [
        ["A", "B", "C"],
        ["A", "C", "B"],
        ["A", "B", "C"],
        ["A", "C", "B"],
    ]
    ratings = fit_plackett_luce(rankings)
    assert max(ratings, key=lambda k: ratings[k]) == "A"


def test_pl_symmetric_tournaments_give_equal_ratings() -> None:
    rankings = [["A", "B"], ["B", "A"]]  # perfectly symmetric
    ratings = fit_plackett_luce(rankings)
    assert pytest.approx(ratings["A"], rel=0.01) == ratings["B"]


def test_pl_handles_missing_players_gracefully() -> None:
    rankings = [["A", "B"], ["C", "A"]]  # B absent from round 2
    ratings = fit_plackett_luce(rankings)
    assert set(ratings) == {"A", "B", "C"}
