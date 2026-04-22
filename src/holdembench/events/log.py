"""Append-only JSONL event log writer + reader."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import IO, Self

from holdembench.events.schema import _Base, parse_event  # type: ignore[private]


class EventLog:
    """Append-only writer for events.jsonl.

    Usage:
        with EventLog(path) as log:
            log.emit(event)

    Never re-opens in truncate mode. Re-opening existing files appends.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] | None = None

    def __enter__(self) -> Self:
        self._fh = self._path.open("a", encoding="utf-8")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def emit(self, event: _Base) -> None:
        if self._fh is None:
            raise RuntimeError("EventLog used outside of with-block")
        line = event.model_dump_json()
        self._fh.write(line + "\n")

    @staticmethod
    def replay(path: Path) -> Iterator[_Base]:
        """Iterate events from a JSONL file, parsing each into its typed model."""
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                stripped = raw.strip()
                if not stripped:
                    continue
                yield parse_event(json.loads(stripped))

    @staticmethod
    def sha256_hex(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
