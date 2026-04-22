"""Tests for deterministic deck shuffling."""
from __future__ import annotations

from holdembench.engine.deck import STANDARD_DECK, shuffled_deck

STANDARD_DECK_SIZE = 52
MIN_TOP_CARD_COUNT = 5
MAX_TOP_CARD_COUNT = 50
NUM_SHUFFLES = 1000


def test_standard_deck_is_52_unique_cards() -> None:
    assert len(STANDARD_DECK) == STANDARD_DECK_SIZE
    assert len(set(STANDARD_DECK)) == STANDARD_DECK_SIZE


def test_shuffled_deck_is_permutation() -> None:
    deck = shuffled_deck(seed=1)
    assert sorted(deck) == sorted(STANDARD_DECK)


def test_shuffle_is_deterministic_for_same_seed() -> None:
    a = shuffled_deck(seed=42)
    b = shuffled_deck(seed=42)
    assert a == b


def test_different_seeds_produce_different_decks() -> None:
    a = shuffled_deck(seed=1)
    b = shuffled_deck(seed=2)
    assert a != b


def test_shuffle_is_uniform_ish() -> None:
    """Sanity: the top card distribution over many seeds is roughly uniform."""
    tops: dict[str, int] = {}
    for s in range(NUM_SHUFFLES):
        top = shuffled_deck(seed=s)[0]
        tops[top] = tops.get(top, 0) + 1
    # Each card appears as top for ~1000/52 ≈ 19 runs. Allow wide band.
    assert all(MIN_TOP_CARD_COUNT <= n <= MAX_TOP_CARD_COUNT for n in tops.values())
    assert len(tops) == STANDARD_DECK_SIZE
