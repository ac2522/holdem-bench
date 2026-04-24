"""Tests for credential loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from holdembench.credentials import (
    CredentialsFileMissing,
    ProviderCredentials,
    load_credentials,
)


def test_raises_if_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nope.toml"
    with pytest.raises(CredentialsFileMissing):
        load_credentials(missing)


def test_loads_provider_section(tmp_path: Path) -> None:
    p = tmp_path / "credentials.toml"
    p.write_text(
        """
        [anthropic]
        api_key = "sk-ant-xxx"

        [openai]
        api_key = "sk-oai-xxx"

        [openrouter]
        api_key = "sk-or-xxx"
        """
    )
    creds = load_credentials(p)
    assert creds.get("anthropic") == ProviderCredentials(api_key="sk-ant-xxx", base_url=None)
    assert creds.get("openai").api_key == "sk-oai-xxx"
    assert creds.get("openrouter").api_key == "sk-or-xxx"


def test_supports_optional_base_url(tmp_path: Path) -> None:
    p = tmp_path / "credentials.toml"
    p.write_text(
        """
        [xai]
        api_key = "sk-xai-xxx"
        base_url = "https://api.x.ai/v1"
        """
    )
    creds = load_credentials(p)
    assert creds.get("xai").base_url == "https://api.x.ai/v1"


def test_default_path_is_home_holdembench(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_home = tmp_path / "home"
    (fake_home / ".holdembench").mkdir(parents=True)
    (fake_home / ".holdembench" / "credentials.toml").write_text('[anthropic]\napi_key = "k"\n')
    monkeypatch.setenv("HOME", str(fake_home))
    creds = load_credentials()
    assert creds.get("anthropic").api_key == "k"


def test_has_and_providers(tmp_path: Path) -> None:
    p = tmp_path / "credentials.toml"
    p.write_text('[anthropic]\napi_key = "k"\n[openai]\napi_key = "k2"\n')
    creds = load_credentials(p)
    assert creds.has("anthropic")
    assert not creds.has("google")
    assert creds.providers() == ("anthropic", "openai")


def test_missing_api_key_raises(tmp_path: Path) -> None:
    p = tmp_path / "credentials.toml"
    p.write_text('[anthropic]\nbase_url = "http://x"\n')
    with pytest.raises(ValueError, match="api_key"):
        load_credentials(p)
