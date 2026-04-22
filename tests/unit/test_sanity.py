"""Sanity check — package imports and version is set."""

import tomllib
from pathlib import Path

import holdembench


def test_package_version_is_set() -> None:
    assert isinstance(holdembench.__version__, str)
    assert holdembench.__version__ == "0.1.0"


def test_ruff_and_pyright_configured() -> None:
    """Just check pyproject.toml has our tool sections loaded."""
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    assert "ruff" in data["tool"]
    assert "pyright" in data["tool"]
    assert "pytest" in data["tool"]
