"""Hypothesis: EventLog round-trip is lossless."""
from __future__ import annotations

from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from holdembench.events.log import EventLog
from holdembench.events.schema import HandEnd


@given(
    st.dictionaries(
        keys=st.sampled_from([f"Seat{i}" for i in range(1, 10)]),
        values=st.integers(min_value=-10_000, max_value=10_000),
        min_size=2, max_size=9,
    ),
    st.floats(min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=None)
def test_hand_end_roundtrip(deltas: dict[str, int], elapsed: float, cost: float) -> None:
    p = Path(f"/tmp/holdembench_prop_{hash(frozenset(deltas.items())) & 0xFFFFFF:06x}.jsonl")
    try:
        with EventLog(p) as log:
            log.emit(HandEnd(
                hand_id="h", stack_deltas=deltas, elapsed_s=elapsed, total_cost_usd=cost,
            ))
        events = list(EventLog.replay(p))
        assert len(events) == 1
        e = events[0]
        assert isinstance(e, HandEnd)
        assert e.stack_deltas == deltas
    finally:
        if p.exists():
            p.unlink()
