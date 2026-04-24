"""Credential loader for Phase 1+ runs.

Reads ``~/.holdembench/credentials.toml`` by default.  File is never committed;
``.gitignore`` covers ``credentials*`` and ``.holdembench/``.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast


class CredentialsFileMissing(FileNotFoundError):
    """Raised when the credentials file is absent at the expected path."""


@dataclass(frozen=True)
class ProviderCredentials:
    api_key: str
    base_url: str | None = None


@dataclass(frozen=True)
class Credentials:
    _by_provider: dict[str, ProviderCredentials] = field(
        default_factory=lambda: dict[str, ProviderCredentials]()
    )

    def get(self, provider: str) -> ProviderCredentials:
        try:
            return self._by_provider[provider]
        except KeyError as e:
            raise KeyError(
                f"no credentials for provider {provider!r}; "
                f"add a [{provider}] section to credentials.toml"
            ) from e

    def has(self, provider: str) -> bool:
        return provider in self._by_provider

    def providers(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_provider))


def default_credentials_path() -> Path:
    home = os.environ.get("HOME")
    base = Path(home) if home else Path.home()
    return base / ".holdembench" / "credentials.toml"


def load_credentials(path: Path | None = None) -> Credentials:
    resolved = path if path is not None else default_credentials_path()
    if not resolved.exists():
        raise CredentialsFileMissing(f"credentials file not found: {resolved}")
    raw = cast("dict[str, Any]", tomllib.loads(resolved.read_text()))
    by_provider: dict[str, ProviderCredentials] = {}
    for provider, section in raw.items():
        if not isinstance(section, dict):
            continue
        section_typed = cast("dict[str, Any]", section)
        api_key_val = section_typed.get("api_key")
        if not isinstance(api_key_val, str):
            raise ValueError(f"provider {provider!r} missing required string `api_key`")
        base_url_val = section_typed.get("base_url")
        if base_url_val is not None and not isinstance(base_url_val, str):
            raise ValueError(f"provider {provider!r} `base_url` must be a string if set")
        by_provider[provider] = ProviderCredentials(api_key=api_key_val, base_url=base_url_val)
    return Credentials(_by_provider=by_provider)
