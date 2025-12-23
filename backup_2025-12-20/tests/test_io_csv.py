"""Test CSV I/O functions."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from spectral_predict.io import read_csv_spectra, read_reference_csv, align_xy


def test_read_csv_wide(tmp_path):
    """Test reading wide-format CSV."""
    # Create a synthetic wide CSV
    csv_path = tmp_path / "spectra_wide.csv"

    # Create wavelengths (400-2400 nm, 2001 points for 1nm resolution)
    wavelengths = np.linspace(400, 2400, 2001)
    wl_cols = [str(wl) for wl in wavelengths]

    # Create 5 samples
    data = {"sample_id": ["S1", "S2", "S3", "S4", "S5"]}
    for wl in wl_cols:
        data[wl] = np.random.rand(5)

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    # Read it back
    result = read_csv_spectra(csv_path)

    assert result.shape == (5, 2001)
    assert list(result.index) == ["S1", "S2", "S3", "S4", "S5"]
    assert result.columns[0] == 400.0
    assert result.columns[-1] == 2400.0

    # Check wavelengths are sorted and increasing
    wls = np.array(result.columns)
    assert np.all(wls[1:] > wls[:-1])


def test_read_csv_long(tmp_path):
    """Test reading long-format CSV (single spectrum)."""
    csv_path = tmp_path / "spectrum_single.csv"

    # Create a long-format CSV with wavelength and value
    wavelengths = np.linspace(400, 2400, 2001)
    values = np.random.rand(2001)

    df = pd.DataFrame({"wavelength": wavelengths, "value": values})
    df.to_csv(csv_path, index=False)

    # Read it back
    result = read_csv_spectra(csv_path)

    assert result.shape == (1, 2001)  # Single row
    assert result.index[0] == "spectrum_single"  # Uses filename as ID
    assert result.columns[0] == 400.0
    assert result.columns[-1] == 2400.0


def test_read_csv_wide_validation_too_few_wavelengths(tmp_path):
    """Test that validation catches too few wavelengths."""
    csv_path = tmp_path / "spectra_short.csv"

    # Create CSV with only 50 wavelengths (< 100 minimum)
    wavelengths = np.linspace(400, 450, 50)
    wl_cols = [str(wl) for wl in wavelengths]

    data = {"sample_id": ["S1"]}
    for wl in wl_cols:
        data[wl] = [np.random.rand()]

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    # Should raise error
    with pytest.raises(ValueError, match="at least 100 wavelengths"):
        read_csv_spectra(csv_path)


def test_read_reference_csv(tmp_path):
    """Test reading reference CSV."""
    ref_path = tmp_path / "reference.csv"

    df = pd.DataFrame(
        {"sample_id": ["S1", "S2", "S3"], "nitrogen": [2.5, 3.1, 2.8], "carbon": [45.2, 43.8, 44.5]}
    )
    df.to_csv(ref_path, index=False)

    result = read_reference_csv(ref_path, "sample_id")

    assert list(result.index) == ["S1", "S2", "S3"]
    assert "nitrogen" in result.columns
    assert "carbon" in result.columns


def test_read_reference_missing_id_column(tmp_path):
    """Test error when ID column doesn't exist."""
    ref_path = tmp_path / "reference.csv"

    df = pd.DataFrame({"sample_id": ["S1", "S2"], "nitrogen": [2.5, 3.1]})
    df.to_csv(ref_path, index=False)

    with pytest.raises(ValueError, match="Column 'wrong_col' not found"):
        read_reference_csv(ref_path, "wrong_col")


def test_align_xy(tmp_path):
    """Test alignment of X and y."""
    # Create spectral data
    X = pd.DataFrame(
        np.random.rand(5, 100),
        index=["S1", "S2", "S3", "S4", "S5"],
        columns=np.linspace(400, 2400, 100),
    )

    # Create reference
    ref = pd.DataFrame(
        {"nitrogen": [2.5, 3.1, 2.8, 3.0, 2.7]}, index=["S1", "S2", "S3", "S4", "S5"]
    )

    X_aligned, y = align_xy(X, ref, "sample_id", "nitrogen")

    assert X_aligned.shape == (5, 100)
    assert len(y) == 5
    assert list(y.index) == ["S1", "S2", "S3", "S4", "S5"]


def test_align_xy_partial_overlap(tmp_path):
    """Test alignment with partial overlap."""
    # X has S1-S5, ref has S3-S7
    X = pd.DataFrame(
        np.random.rand(5, 100),
        index=["S1", "S2", "S3", "S4", "S5"],
        columns=np.linspace(400, 2400, 100),
    )

    ref = pd.DataFrame(
        {"nitrogen": [2.8, 3.0, 2.7, 3.2, 2.9]}, index=["S3", "S4", "S5", "S6", "S7"]
    )

    X_aligned, y = align_xy(X, ref, "sample_id", "nitrogen")

    # Should only have S3, S4, S5
    assert X_aligned.shape == (3, 100)
    assert len(y) == 3
    assert set(y.index) == {"S3", "S4", "S5"}


def test_align_xy_missing_target(tmp_path):
    """Test error when target doesn't exist."""
    X = pd.DataFrame(
        np.random.rand(3, 100), index=["S1", "S2", "S3"], columns=np.linspace(400, 2400, 100)
    )

    ref = pd.DataFrame({"nitrogen": [2.5, 3.1, 2.8]}, index=["S1", "S2", "S3"])

    with pytest.raises(ValueError, match="Target 'carbon' not found"):
        align_xy(X, ref, "sample_id", "carbon")


def test_align_xy_no_overlap(tmp_path):
    """Test error when no IDs overlap."""
    X = pd.DataFrame(
        np.random.rand(3, 100), index=["S1", "S2", "S3"], columns=np.linspace(400, 2400, 100)
    )

    ref = pd.DataFrame({"nitrogen": [2.5, 3.1, 2.8]}, index=["S4", "S5", "S6"])

    with pytest.raises(ValueError, match="No matching IDs"):
        align_xy(X, ref, "sample_id", "nitrogen")
