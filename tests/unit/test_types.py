"""Types module canonical source."""

from __future__ import annotations

import typing

from holdembench import types as t
from holdembench.agents import base
from holdembench.chat import protocol
from holdembench.engine import validator
from holdembench.events import schema
from holdembench.events.schema import TournamentStart, parse_event


def test_canonical_type_aliases_exported() -> None:
    assert typing.get_args(t.ActionName.__value__) == ("fold", "check", "call", "raise")
    assert typing.get_args(t.ActionKind.__value__) == ("action", "probe", "probe_reply")
    assert typing.get_args(t.ChatKind.__value__) == ("action", "probe", "probe_reply")
    assert typing.get_args(t.Street.__value__) == ("preflop", "flop", "turn", "river")


def test_schema_reexports_types_from_canonical_module() -> None:
    assert schema.ActionName is t.ActionName
    assert schema.ActionKind is t.ActionKind
    assert schema.Street is t.Street
    assert validator.ActionName is t.ActionName
    assert validator.ActionKind is t.ActionKind
    assert protocol.ChatKind is t.ChatKind
    assert base.Street is t.Street


def test_parse_event_returns_event_union() -> None:
    obj = parse_event(
        {
            "type": "tournament_start",
            "tournament_id": "t",
            "schema_version": "1.0",
            "holdembench_version": "x",
            "pokerkit_version": "y",
            "git_sha": "z",
            "seat_assignments": {},
            "master_seed": 0,
            "anonymization_salt": "a",
            "canary_uuid": "00000000-0000-0000-0000-000000000000",
        }
    )
    assert isinstance(obj, TournamentStart)


def test_base_class_no_longer_exported() -> None:
    assert "_Base" not in schema.__all__
