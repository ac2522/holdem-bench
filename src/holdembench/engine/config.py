"""Table/session configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

_MIN_SEAT_COUNT = 2


@dataclass(frozen=True)
class TableConfig:
    seat_count: int
    small_blind: int
    big_blind: int
    ante: int
    starting_stacks: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.seat_count < _MIN_SEAT_COUNT:
            raise ValueError("seat_count must be >= 2")
        if len(self.starting_stacks) != self.seat_count:
            raise ValueError("starting_stacks length must equal seat_count")
        if self.big_blind < self.small_blind:
            raise ValueError("big_blind must be >= small_blind")
