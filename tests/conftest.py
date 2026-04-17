"""Shared pytest fixtures for egt."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def tests_dir() -> Path:
    """Absolute path to the tests/ directory."""
    return Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def testdb_dir(tests_dir: Path) -> Path:
    """Absolute path to tests/testdb/."""
    return tests_dir / "testdb"


@pytest.fixture(scope="session")
def mini_hydra(testdb_dir: Path) -> Path:
    """Absolute path to the mini_hydra fixture directory."""
    path = testdb_dir / "mini_hydra"
    if not path.is_dir():
        pytest.skip(f"mini_hydra fixture missing at {path}")
    return path


@pytest.fixture(scope="session")
def mini_urchin(testdb_dir: Path) -> Path:
    """Absolute path to the mini_urchin fixture directory."""
    path = testdb_dir / "mini_urchin"
    if not path.is_dir():
        pytest.skip(f"mini_urchin fixture missing at {path}")
    return path
