"""Coarse 9-max push-fold chart — shove ranges by effective stack (BB)."""

from __future__ import annotations

SHOVE_RANGE_10BB = frozenset(
    {
        "AA",
        "KK",
        "QQ",
        "JJ",
        "TT",
        "99",
        "88",
        "77",
        "66",
        "55",
        "44",
        "33",
        "22",
        "AK",
        "AQ",
        "AJ",
        "AT",
        "A9",
        "A8",
        "A7",
        "A6",
        "A5",
        "A4",
        "A3",
        "A2",
        "KQ",
        "KJ",
        "KT",
        "K9",
        "QJ",
        "QT",
        "Q9",
        "JT",
        "J9",
        "T9",
        "T8",
        "98",
        "87",
        "76",
    }
)

SHOVE_RANGE_15BB = frozenset(
    {
        "AA",
        "KK",
        "QQ",
        "JJ",
        "TT",
        "99",
        "88",
        "77",
        "AK",
        "AQ",
        "AJ",
        "AT",
        "A9",
        "KQ",
        "KJ",
        "KT",
        "QJ",
        "QT",
        "JT",
    }
)

SHOVE_RANGE_20BB = frozenset(
    {
        "AA",
        "KK",
        "QQ",
        "JJ",
        "TT",
        "99",
        "AK",
        "AQ",
        "AJ",
        "AT",
        "KQ",
        "KJ",
    }
)
