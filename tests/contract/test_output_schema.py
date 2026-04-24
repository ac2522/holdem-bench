"""Agent output JSON schema + parser."""

from __future__ import annotations

import pytest

from holdembench.agents.output_schema import (
    AgentOutputParseError,
    parse_agent_output,
)


def test_minimal_action_parses() -> None:
    out = parse_agent_output('{"kind": "action", "action": "fold"}')
    assert out.kind == "action"
    assert out.action == "fold"
    raw = out.to_raw_decision()
    assert raw.kind == "action"
    assert raw.action == "fold"


def test_raise_requires_amount() -> None:
    with pytest.raises(AgentOutputParseError):
        parse_agent_output('{"kind": "action", "action": "raise"}')


def test_raise_rejects_non_positive_amount() -> None:
    with pytest.raises(AgentOutputParseError):
        parse_agent_output('{"kind": "action", "action": "raise", "amount": 0}')
    with pytest.raises(AgentOutputParseError):
        parse_agent_output('{"kind": "action", "action": "raise", "amount": -10}')


def test_probe_requires_message() -> None:
    with pytest.raises(AgentOutputParseError):
        parse_agent_output('{"kind": "probe"}')
    ok = parse_agent_output(
        '{"kind": "probe", "message": "Big bet here, are you actually strong?"}'
    )
    assert ok.to_raw_decision().message is not None


def test_ignores_trailing_text_after_json_block() -> None:
    out = parse_agent_output(
        '{"kind": "action", "action": "check"}\n\nsorry, small commentary'
    )
    assert out.action == "check"


def test_empty_string_raises_parse_error() -> None:
    with pytest.raises(AgentOutputParseError):
        parse_agent_output("")


def test_invalid_action_rejected() -> None:
    with pytest.raises(AgentOutputParseError):
        parse_agent_output('{"kind": "action", "action": "all_in"}')


def test_thinking_stored_but_not_in_raw() -> None:
    out = parse_agent_output(
        '{"kind": "action", "action": "call", "thinking": "I have middle pair..."}'
    )
    assert out.thinking == "I have middle pair..."
    raw = out.to_raw_decision()
    assert raw.kind == "action"


def test_extra_unknown_field_rejected() -> None:
    with pytest.raises(AgentOutputParseError):
        parse_agent_output('{"kind": "action", "action": "fold", "cheating": true}')


def test_handles_markdown_code_fence_wrapper() -> None:
    out = parse_agent_output('```json\n{"kind": "action", "action": "check"}\n```')
    assert out.action == "check"


def test_no_json_object_raises() -> None:
    with pytest.raises(AgentOutputParseError, match="no JSON"):
        parse_agent_output("hello there")


def test_unterminated_json_raises() -> None:
    with pytest.raises(AgentOutputParseError):
        parse_agent_output('{"kind": "action", "action": "fold"')
