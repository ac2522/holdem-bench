"""Tests for env-var-based credential lookup."""

from __future__ import annotations

from pathlib import Path

import pytest

from holdembench.credentials import (
    MissingCredentialError,
    ProviderCredentials,
    get_provider_credentials,
    has_provider_credentials,
    known_providers,
    load_dotenv_from_repo,
)


def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "GOOGLE_API_KEY",
        "XAI_API_KEY",
        "XAI_BASE_URL",
        "MOONSHOT_API_KEY",
        "MOONSHOT_BASE_URL",
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)


def test_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    with pytest.raises(MissingCredentialError, match="ANTHROPIC_API_KEY"):
        get_provider_credentials("anthropic")


def test_returns_api_key_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")
    creds = get_provider_credentials("anthropic")
    assert creds == ProviderCredentials(api_key="sk-ant-xxx", base_url=None)


def test_openrouter_uses_default_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-xxx")
    creds = get_provider_credentials("openrouter")
    assert creds.api_key == "sk-or-xxx"
    assert creds.base_url == "https://openrouter.ai/api/v1"


def test_base_url_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("XAI_API_KEY", "sk-xai-xxx")
    monkeypatch.setenv("XAI_BASE_URL", "https://example.test/v1")
    creds = get_provider_credentials("xai")
    assert creds.base_url == "https://example.test/v1"


def test_unknown_provider_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    with pytest.raises(KeyError, match="unknown provider"):
        get_provider_credentials("not-a-provider")


def test_has_provider_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    assert not has_provider_credentials("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert has_provider_credentials("openai")
    assert not has_provider_credentials("not-a-provider")


def test_empty_string_treated_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert not has_provider_credentials("openai")
    with pytest.raises(MissingCredentialError):
        get_provider_credentials("openai")


def test_whitespace_only_treated_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    with pytest.raises(MissingCredentialError):
        get_provider_credentials("openai")


def test_known_providers_lists_all() -> None:
    providers = known_providers()
    assert "anthropic" in providers
    assert "openai" in providers
    assert "google" in providers
    assert "xai" in providers
    assert "moonshot" in providers
    assert "openrouter" in providers


def test_load_dotenv_from_repo_reads_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_provider_env(monkeypatch)
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-from-dotenv\n")
    load_dotenv_from_repo(tmp_path)
    creds = get_provider_credentials("anthropic")
    assert creds.api_key == "sk-from-dotenv"


def test_load_dotenv_does_not_override_existing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell")
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=from-file\n")
    load_dotenv_from_repo(tmp_path)
    creds = get_provider_credentials("anthropic")
    assert creds.api_key == "from-shell"


def test_load_dotenv_no_file_is_noop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_provider_env(monkeypatch)
    (tmp_path / "pyproject.toml").write_text("")
    load_dotenv_from_repo(tmp_path)  # no .env present — should not raise
    assert not has_provider_credentials("anthropic")
