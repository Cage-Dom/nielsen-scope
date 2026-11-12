"""Shared test settings."""

import pytest


@pytest.fixture(autouse=True)
def _no_project_database(monkeypatch):
    """Unset NIELSEN_DB_URL in every test so a node under test can never reach the
    project database. Tests that write set it to the disposable test database."""
    monkeypatch.delenv("NIELSEN_DB_URL", raising=False)