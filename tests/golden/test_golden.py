"""Golden regression tests — scripted hands with known expected outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from holdembench.engine.config import TableConfig
from holdembench.engine.table import Table
from holdembench.engine.validator import RawDecision, TDAValidator, ValidationError

_GOLDEN_DIR = Path(__file__).parent / "hands"


def _scenario_files() -> list[Path]:
    return sorted(_GOLDEN_DIR.glob("*.json"))


@pytest.mark.parametrize("scenario_path", _scenario_files(), ids=lambda p: p.stem)
def test_golden_hand(scenario_path: Path) -> None:
    scenario = json.loads(scenario_path.read_text())

    # Skip scenarios marked with "skip"
    if scenario.get("skip"):
        pytest.skip(scenario["skip"])

    table_cfg = TableConfig(**scenario["table"])
    table = Table(table_cfg)
    validator = TDAValidator(table)

    for step in scenario["script"]:
        seat = step["seat"]
        kind = step.get("kind", "action")
        action = step.get("action")
        amount = step.get("amount")
        decision = RawDecision(kind=kind, action=action, amount=amount)

        if scenario.get("expect_rejection") and step.get("expect_rejection"):
            with pytest.raises(ValidationError):
                validator.check(seat, decision)
            continue

        validator.check(seat, decision)
        if action == "fold":
            table.apply_fold(seat)
        elif action in {"check", "call"}:
            table.apply_check_or_call(seat)
        elif action == "raise":
            table.apply_raise(seat, to=amount)

    # Verify chip deltas match expected
    starting = scenario["table"]["starting_stacks"]
    actual = [s - starting[i] for i, s in enumerate(table._state.stacks)]
    assert actual == scenario["expected_stack_deltas"], (
        f"{scenario['name']}: actual={actual}, expected={scenario['expected_stack_deltas']}"
    )
