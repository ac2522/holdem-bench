"""Canonical agent output JSON schema + parser.

Every adapter must produce JSON matching :class:`AgentOutput`.  The runner
catches :class:`AgentOutputParseError` and applies retry-once-then-autofold.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, model_validator
from pydantic import ValidationError as PydanticValidationError

from holdembench.engine.validator import RawDecision
from holdembench.types import ActionKind, ActionName


class AgentOutputParseError(ValueError):
    """Raised when an adapter's JSON output cannot be validated."""


class AgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ActionKind
    action: ActionName | None = None
    amount: int | None = None
    message: str | None = None
    thinking: str | None = None

    @model_validator(mode="after")
    def _check(self) -> AgentOutput:
        if self.kind == "action":
            if self.action is None:
                raise ValueError("kind=action requires `action`")
            if self.action == "raise" and (self.amount is None or self.amount <= 0):
                raise ValueError("action=raise requires positive `amount`")
        elif not self.message:
            raise ValueError(f"kind={self.kind} requires non-empty `message`")
        return self

    def to_raw_decision(self) -> RawDecision:
        return RawDecision(
            kind=self.kind,
            action=self.action,
            amount=self.amount,
            message=self.message,
        )


def parse_agent_output(text: str) -> AgentOutput:
    stripped = text.strip()
    if not stripped:
        raise AgentOutputParseError("empty response")
    payload = _extract_json_object(stripped)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        raise AgentOutputParseError(f"invalid JSON: {e}") from e
    try:
        return AgentOutput.model_validate(data)
    except PydanticValidationError as e:
        raise AgentOutputParseError(
            f"schema violation: {e.errors(include_url=False)}"
        ) from e


def _extract_json_object(text: str) -> str:
    """Return the first balanced ``{...}`` block in *text*.

    Tolerates markdown code fences and trailing commentary after the JSON.
    Raises :class:`AgentOutputParseError` if no balanced object is found.
    """
    start = text.find("{")
    if start < 0:
        raise AgentOutputParseError("no JSON object found")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise AgentOutputParseError("unterminated JSON object")
