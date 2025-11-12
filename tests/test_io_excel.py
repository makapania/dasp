"""Test Excel I/O functions."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from spectral_predict.io import (
    read_excel_spectra,
    write_excel_spectra,
    read_spectra,
    write_spectra
)


def test_read_excel_wide_format(tmp_path):
    """Test reading wide-format Excel file."""
    excel_path = tmp_path / "spectra_wide.xlsx"

    # Create wavelengths (400-2400 nm, 2001 points)
    wavelengths = np.linspace(400, 2400, 2001)
    wl_cols = [str(wl) for wl in wavelengths]

    # Create 5 samples
    data = {"sample_id": ["S1", "S2", "S3", "S4", "S5"]}
    for wl in wl_cols:
        data[wl] = np.random.rand(5) * 0.5 + 0.2  # 0.2-0.7 range

    df = pd.DataFrame(data)
    df.to_excel(excel_path, index=False)

    # Read it back
    result, metadata = read_excel_spectra(excel_path)

    assert result.shape == (5, 2001)
    assert list(result.index) == ["S1", "S2", "S3", "S4", "S5"]
    assert result.columns[0] == 400.0
    assert result.columns[-1] == 2400.0
    assert metadata['file_format'] == 'excel'
    assert metadata['n_spectra'] == 5

    # Check wavelengths are sorted
    wls = np.array(result.columns)
    assert np.all(wls[1:] > wls[:-1])


def test_read_excel_long_format(tmp_path):
    """Test reading long-format Excel file."""
    excel_path = tmp_path / "spectrum_single.xlsx"

    # Create long-format data
    wavelengths = np.linspace(400, 2400, 2001)
    values = np.random.rand(2001) * 0.5 + 0.2

    df = pd.DataFrame({"wavelength": wavelengths, "value": values})
    df.to_excel(excel_path, index=False)

    # Read it back
    result, metadata = read_excel_spectra(excel_path)

    assert result.shape == (1, 2001)
    assert result.index[0] == "spectrum_single"
    assert result.columns[0] == 400.0
    assert result.columns[-1] == 2400.0


def test_write_read_excel_roundtrip(tmp_path):
    """Test write and read Excel roundtrip."""
    excel_path = tmp_path / "roundtrip.xlsx"

    # Create test data
    wavelengths = np.linspace(400, 2400, 2001)
    data = pd.DataFrame(
        np.random.rand(3, 2001) * 0.5 + 0.2,
        index=["Sample_1", "Sample_2", "Sample_3"],
        columns=wavelengths
    )

    # Write
    write_excel_spectra(data, excel_path, sheet_name='TestSpectra')

    # Read back
    result, metadata = read_excel_spectra(excel_path, sheet_name='TestSpectra')

    # Compare
    assert result.shape == data.shape
    assert list(result.index) == list(data.index)
    np.testing.assert_array_almost_equal(result.values, data.values, decimal=5)


def test_excel_too_few_wavelengths(tmp_path):
    """Test validation catches too few wavelengths."""
    excel_path = tmp_path / "short.xlsx"

    # Create Excel with only 50 wavelengths
    wavelengths = np.linspace(400, 450, 50)
    data = {"sample_id": ["S1"]}
    for wl in wavelengths:
        data[str(wl)] = [np.random.rand()]

    df = pd.DataFrame(data)
    df.to_excel(excel_path, index=False)

    # Should raise error
    with pytest.raises(ValueError, match="at least 100 wavelengths"):
        read_excel_spectra(excel_path)


def test_excel_empty_file(tmp_path):
    """Test error on empty Excel file."""
    excel_path = tmp_path / "empty.xlsx"

    # Create empty Excel file
    df = pd.DataFrame()
    df.to_excel(excel_path, index=False)

    with pytest.raises(ValueError, match="Empty Excel file"):
        read_excel_spectra(excel_path)


def test_excel_via_unified_api(tmp_path):
    """Test reading Excel through unified read_spectra API."""
    excel_path = tmp_path / "unified_test.xlsx"

    # Create test data
    wavelengths = np.linspace(400, 2400, 2001)
    data = {"sample_id": ["S1", "S2", "S3"]}
    for wl in wavelengths:
        data[str(wl)] = np.random.rand(3) * 0.5 + 0.2

    df = pd.DataFrame(data)
    df.to_excel(excel_path, index=False)

    # Read with auto-detection
    result, metadata = read_spectra(excel_path, format='auto')

    assert result.shape == (3, 2001)
    assert metadata['file_format'] == 'excel'

    # Read with explicit format
    result2, metadata2 = read_spectra(excel_path, format='excel')
    pd.testing.assert_frame_equal(result, result2)


def test_write_excel_via_unified_api(tmp_path):
    """Test writing Excel through unified write_spectra API."""
    excel_path = tmp_path / "unified_write.xlsx"

    # Create test data
    wavelengths = np.linspace(400, 2400, 2001)
    data = pd.DataFrame(
        np.random.rand(3, 2001) * 0.5 + 0.2,
        index=["S1", "S2", "S3"],
        columns=wavelengths
    )

    # Write via unified API
    write_spectra(data, excel_path, format='excel', sheet_name='TestData')

    # Read back
    result, _ = read_excel_spectra(excel_path, sheet_name='TestData')

    assert result.shape == data.shape
    np.testing.assert_array_almost_equal(result.values, data.values, decimal=5)


def test_excel_multiple_sheets(tmp_path):
    """Test reading specific sheet from multi-sheet Excel."""
    excel_path = tmp_path / "multi_sheet.xlsx"

    # Create test data for two sheets
    wavelengths = np.linspace(400, 2400, 2001)
    data1 = {"sample_id": ["S1", "S2"]}
    data2 = {"sample_id": ["S3", "S4"]}

    for wl in wavelengths:
        data1[str(wl)] = np.random.rand(2) * 0.5 + 0.2
        data2[str(wl)] = np.random.rand(2) * 0.5 + 0.2

    with pd.ExcelWriter(excel_path) as writer:
        pd.DataFrame(data1).to_excel(writer, sheet_name='Sheet1', index=False)
        pd.DataFrame(data2).to_excel(writer, sheet_name='Sheet2', index=False)

    # Read Sheet1
    result1, meta1 = read_excel_spectra(excel_path, sheet_name='Sheet1')
    assert list(result1.index) == ["S1", "S2"]
    assert meta1['sheet_name'] == 'Sheet1'

    # Read Sheet2
    result2, meta2 = read_excel_spectra(excel_path, sheet_name='Sheet2')
    assert list(result2.index) == ["S3", "S4"]
    assert meta2['sheet_name'] == 'Sheet2'


def test_excel_freeze_panes(tmp_path):
    """Test Excel export with frozen panes."""
    excel_path = tmp_path / "frozen.xlsx"

    wavelengths = np.linspace(400, 2400, 2001)
    data = pd.DataFrame(
        np.random.rand(5, 2001) * 0.5 + 0.2,
        index=[f"S{i}" for i in range(5)],
        columns=wavelengths
    )

    # Write with frozen panes
    write_excel_spectra(data, excel_path, freeze_panes=(1, 1))

    # Read back (just verify it works)
    result, _ = read_excel_spectra(excel_path)
    assert result.shape == data.shape


def test_excel_custom_float_format(tmp_path):
    """Test Excel export with custom float formatting."""
    excel_path = tmp_path / "formatted.xlsx"

    wavelengths = np.linspace(400, 2400, 2001)
    data = pd.DataFrame(
        np.random.rand(2, 2001) * 0.5 + 0.2,
        index=["S1", "S2"],
        columns=wavelengths
    )

    # Write with custom format
    write_excel_spectra(data, excel_path, float_format='0.0000')

    # Read back
    result, _ = read_excel_spectra(excel_path)
    assert result.shape == data.shape
    np.testing.assert_array_almost_equal(result.values, data.values, decimal=4)


def test_excel_metadata_preservation(tmp_path):
    """Test that metadata is correctly extracted."""
    excel_path = tmp_path / "meta_test.xlsx"

    wavelengths = np.linspace(400, 2400, 2001)
    data = {"sample_id": ["S1"]}
    for wl in wavelengths:
        data[str(wl)] = [np.random.rand() * 0.5 + 0.2]

    df = pd.DataFrame(data)
    df.to_excel(excel_path, index=False)

    # Read and check metadata
    result, metadata = read_excel_spectra(excel_path)

    assert 'file_format' in metadata
    assert metadata['file_format'] == 'excel'
    assert 'n_spectra' in metadata
    assert metadata['n_spectra'] == 1
    assert 'wavelength_range' in metadata
    assert metadata['wavelength_range'] == (400.0, 2400.0)
    assert 'data_type' in metadata
    assert 'type_confidence' in metadata
    assert 'detection_method' in metadata
