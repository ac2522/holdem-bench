"""Provider credentials, sourced from environment variables.

We let each provider SDK read its own canonical env var
(``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``GOOGLE_API_KEY``).  For
OpenRouter / xAI / Moonshot — which don't have a single official SDK — we
follow the convention ``<PROVIDER>_API_KEY`` and ``<PROVIDER>_BASE_URL``.

A ``.env`` file in the project root is loaded automatically (via
``python-dotenv``) so contributors don't have to ``export`` keys in their
shell every session.  ``.env`` is gitignored.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_PROVIDERS: dict[str, tuple[str, str | None, str | None]] = {
    # provider     -> (api_key_env,        base_url_env,           default_base_url)
    "anthropic":     ("ANTHROPIC_API_KEY",  None,                   None),
    "openai":        ("OPENAI_API_KEY",     "OPENAI_BASE_URL",      None),
    "google":        ("GOOGLE_API_KEY",     None,                   None),
    "xai":           ("XAI_API_KEY",        "XAI_BASE_URL",         "https://api.x.ai/v1"),
    "moonshot":      ("MOONSHOT_API_KEY",   "MOONSHOT_BASE_URL",    "https://api.moonshot.ai/v1"),
    "openrouter":    ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL",  "https://openrouter.ai/api/v1"),
}


class MissingCredentialError(RuntimeError):
    """Raised when a required provider env var is unset/empty."""


@dataclass(frozen=True)
class ProviderCredentials:
    api_key: str
    base_url: str | None = None


def load_dotenv_from_repo(repo_root: Path | None = None) -> None:
    """Load ``.env`` from the repo root, if present.

    Idempotent — safe to call multiple times.  Existing env vars take
    precedence over file values (``override=False``).
    """
    if repo_root is None:
        repo_root = _find_repo_root(Path.cwd())
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def _find_repo_root(start: Path) -> Path:
    for d in (start, *start.parents):
        if (d / "pyproject.toml").exists() or (d / ".git").exists():
            return d
    return start


def get_provider_credentials(provider: str) -> ProviderCredentials:
    """Return the api_key (+ optional base_url) for ``provider``.

    Raises :class:`MissingCredentialError` if the api-key env var is unset
    or empty.  ``base_url`` falls back to the provider's documented default
    when the override env var is unset.
    """
    if provider not in _PROVIDERS:
        raise KeyError(f"unknown provider {provider!r}")
    key_env, base_env, default_base = _PROVIDERS[provider]
    api_key = os.environ.get(key_env, "").strip()
    if not api_key:
        raise MissingCredentialError(
            f"missing {key_env}; set it in .env or export it in your shell"
        )
    base_url: str | None = None
    if base_env is not None:
        override = os.environ.get(base_env, "").strip()
        base_url = override or default_base
    elif default_base is not None:
        base_url = default_base
    return ProviderCredentials(api_key=api_key, base_url=base_url)


def has_provider_credentials(provider: str) -> bool:
    """True if the provider's api-key env var is set and non-empty."""
    if provider not in _PROVIDERS:
        return False
    key_env, _, _ = _PROVIDERS[provider]
    return bool(os.environ.get(key_env, "").strip())


def known_providers() -> tuple[str, ...]:
    return tuple(_PROVIDERS)
