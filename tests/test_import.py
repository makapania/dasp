"""Test basic package imports."""

import pytest


def test_import_package():
    """Test that the package can be imported."""
    import spectral_predict

    assert spectral_predict is not None


def test_version_exists():
    """Test that VERSION is defined."""
    from spectral_predict import VERSION

    assert VERSION is not None
    assert isinstance(VERSION, str)
    assert len(VERSION) > 0


def test_version_format():
    """Test that version follows semantic versioning."""
    from spectral_predict import __version__

    parts = __version__.split(".")
    assert len(parts) >= 2  # At least major.minor
