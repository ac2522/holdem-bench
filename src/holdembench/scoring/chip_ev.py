"""Chip EV and mbb/100 computation from an event log."""
from __future__ import annotations

from pathlib import Path

from holdembench.events.log import EventLog
from holdembench.events.schema import HandEnd


def compute_chip_ev(log_path: Path) -> dict[str, int]:
    """Return total chip delta per seat across all HandEnd events."""
    totals: dict[str, int] = {}
    for event in EventLog.replay(log_path):
        if isinstance(event, HandEnd):
            for seat, d in event.stack_deltas.items():
                totals[seat] = totals.get(seat, 0) + d
    return totals


def mbb_per_100(chip_delta: int, hands: int, big_blind: int) -> float:
    """Milli-big-blinds per 100 hands."""
    if hands == 0:
        return 0.0
    bb_delta = chip_delta / big_blind
    return (bb_delta * 1000) / hands * 100
