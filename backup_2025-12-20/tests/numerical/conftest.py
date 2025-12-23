"""Configuration for numerical correctness tests."""

from __future__ import annotations

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "numerical: Numerical correctness tests comparing against gold standards"
    )
