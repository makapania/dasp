"""Test optional R bridge functionality.

These tests are skipped if R (Rscript) is not available in the PATH.
"""

import pytest
from pathlib import Path

from spectral_predict.readers.asd_r_bridge import (
    check_r_available,
    read_asd_with_r,
    read_asd_with_asdreader,
    read_asd_with_prospectr,
)
from spectral_predict.readers.asd_native import read_binary_asd


# Skip all tests in this file if R is not available
pytestmark = pytest.mark.skipif(not check_r_available(), reason="R (Rscript) not available in PATH")


def test_check_r_available():
    """Test R availability check."""
    # This test only runs if R is available
    assert check_r_available() is True


def test_read_asd_with_r_not_implemented(tmp_path):
    """Test that R bridge raises NotImplementedError."""
    # Create a dummy file
    dummy_file = tmp_path / "test.asd"
    dummy_file.write_bytes(b"dummy binary data")

    # Should raise NotImplementedError (not yet implemented)
    with pytest.raises(NotImplementedError, match="R bridge.*not yet implemented"):
        read_asd_with_r(dummy_file, r_package="asdreader")


def test_read_asd_with_asdreader_not_implemented(tmp_path):
    """Test asdreader wrapper raises NotImplementedError."""
    dummy_file = tmp_path / "test.asd"
    dummy_file.write_bytes(b"dummy")

    with pytest.raises(NotImplementedError, match="not yet implemented"):
        read_asd_with_asdreader(dummy_file)


def test_read_asd_with_prospectr_not_implemented(tmp_path):
    """Test prospectr wrapper raises NotImplementedError."""
    dummy_file = tmp_path / "test.asd"
    dummy_file.write_bytes(b"dummy")

    with pytest.raises(NotImplementedError, match="not yet implemented"):
        read_asd_with_prospectr(dummy_file)


def test_native_reader_not_implemented(tmp_path):
    """Test that native binary reader raises NotImplementedError."""
    dummy_file = tmp_path / "test.asd"
    dummy_file.write_bytes(b"dummy")

    with pytest.raises(NotImplementedError, match="Native Python.*not yet implemented"):
        read_binary_asd(dummy_file)


# Tests that always run (no R required)
@pytest.mark.skipif(False, reason="Always run")
def test_check_r_available_returns_bool():
    """Test that check_r_available returns a boolean."""
    result = check_r_available()
    assert isinstance(result, bool)
