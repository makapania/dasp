"""Test unified I/O dispatcher (read_spectra and write_spectra)."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from spectral_predict.io import (
    read_spectra,
    write_spectra,
    detect_format,
    _detect_directory_format
)


def create_test_data(n_samples=3, n_wavelengths=2001):
    """Create synthetic spectral data for testing."""
    wavelengths = np.linspace(400, 2400, n_wavelengths)
    data = pd.DataFrame(
        np.random.rand(n_samples, n_wavelengths) * 0.5 + 0.2,
        index=[f"Sample_{i+1}" for i in range(n_samples)],
        columns=wavelengths
    )
    return data


# ============================================================================
# Auto-Detection Tests
# ============================================================================


def test_auto_detect_csv(tmp_path):
    """Test auto-detection of CSV format."""
    csv_path = tmp_path / "test.csv"
    data = create_test_data()

    # Write CSV
    csv_data = data.copy()
    csv_data.index.name = 'sample_id'
    csv_data.to_csv(csv_path)

    # Read with auto-detection
    result, metadata = read_spectra(csv_path, format='auto')

    assert metadata['file_format'] == 'csv'
    assert result.shape[0] == 3


def test_auto_detect_excel(tmp_path):
    """Test auto-detection of Excel format."""
    excel_path = tmp_path / "test.xlsx"
    data = create_test_data()

    # Write Excel
    excel_data = data.copy()
    excel_data.index.name = 'sample_id'
    excel_data.to_excel(excel_path)

    # Read with auto-detection
    result, metadata = read_spectra(excel_path, format='auto')

    assert metadata['file_format'] == 'excel'
    assert result.shape[0] == 3


def test_auto_detect_ascii(tmp_path):
    """Test auto-detection of ASCII format."""
    ascii_path = tmp_path / "test.txt"

    wavelengths = np.linspace(400, 2400, 2001)
    intensities = np.random.rand(2001) * 0.5 + 0.2

    with open(ascii_path, 'w') as f:
        for wl, intensity in zip(wavelengths, intensities):
            f.write(f"{wl}\t{intensity}\n")

    # Read with auto-detection
    result, metadata = read_spectra(ascii_path, format='auto')

    assert metadata['file_format'] == 'ascii'
    assert result.shape[0] == 1


def test_detect_directory_asd(tmp_path):
    """Test directory format detection for ASD files."""
    asd_dir = tmp_path / "asd_data"
    asd_dir.mkdir()

    # Create mock ASD files
    (asd_dir / "sample1.asd").touch()
    (asd_dir / "sample2.sig").touch()

    detected = _detect_directory_format(asd_dir)
    assert detected == 'asd'


def test_detect_directory_spc(tmp_path):
    """Test directory format detection for SPC files."""
    spc_dir = tmp_path / "spc_data"
    spc_dir.mkdir()

    # Create mock SPC files
    (spc_dir / "sample1.spc").touch()
    (spc_dir / "sample2.spc").touch()

    detected = _detect_directory_format(spc_dir)
    assert detected == 'spc'


def test_detect_directory_csv(tmp_path):
    """Test directory format detection for CSV files."""
    csv_dir = tmp_path / "csv_data"
    csv_dir.mkdir()

    (csv_dir / "data.csv").touch()

    detected = _detect_directory_format(csv_dir)
    assert detected == 'csv'


def test_detect_directory_unknown(tmp_path):
    """Test directory format detection for unknown contents."""
    unknown_dir = tmp_path / "unknown"
    unknown_dir.mkdir()

    (unknown_dir / "file.xyz").touch()

    detected = _detect_directory_format(unknown_dir)
    assert detected == 'unknown'


# ============================================================================
# Unified Write API Tests
# ============================================================================


def test_write_spectra_csv(tmp_path):
    """Test unified write API for CSV."""
    csv_path = tmp_path / "output.csv"
    data = create_test_data()

    write_spectra(data, csv_path, format='csv')

    assert csv_path.exists()

    # Read back
    result, _ = read_spectra(csv_path)
    assert result.shape == data.shape


def test_write_spectra_excel(tmp_path):
    """Test unified write API for Excel."""
    excel_path = tmp_path / "output.xlsx"
    data = create_test_data()

    write_spectra(data, excel_path, format='excel')

    assert excel_path.exists()

    # Read back
    result, _ = read_spectra(excel_path)
    assert result.shape == data.shape


def test_write_spectra_ascii(tmp_path):
    """Test unified write API for ASCII."""
    ascii_path = tmp_path / "output.txt"
    data = create_test_data(n_samples=1)  # ASCII supports single spectrum

    write_spectra(data, ascii_path, format='ascii')

    assert ascii_path.exists()

    # Read back
    result, _ = read_spectra(ascii_path)
    assert result.shape[0] == 1


def test_write_spectra_unsupported_format(tmp_path):
    """Test that unsupported write format raises error."""
    output_path = tmp_path / "output.xyz"
    data = create_test_data()

    with pytest.raises(ValueError, match="Unsupported export format"):
        write_spectra(data, output_path, format='unsupported')


# ============================================================================
# Unified Read API Tests
# ============================================================================


def test_read_spectra_explicit_csv(tmp_path):
    """Test explicit CSV format specification."""
    csv_path = tmp_path / "data.csv"
    data = create_test_data()

    data.index.name = 'sample_id'
    data.to_csv(csv_path)

    # Read with explicit format
    result, metadata = read_spectra(csv_path, format='csv')

    assert metadata['file_format'] == 'csv'
    assert result.shape[0] == 3


def test_read_spectra_explicit_excel(tmp_path):
    """Test explicit Excel format specification."""
    excel_path = tmp_path / "data.xlsx"
    data = create_test_data()

    data.index.name = 'sample_id'
    data.to_excel(excel_path)

    result, metadata = read_spectra(excel_path, format='excel')

    assert metadata['file_format'] == 'excel'


def test_read_spectra_explicit_ascii(tmp_path):
    """Test explicit ASCII format specification."""
    ascii_path = tmp_path / "data.txt"

    wavelengths = np.linspace(400, 2400, 2001)
    intensities = np.random.rand(2001) * 0.5 + 0.2

    with open(ascii_path, 'w') as f:
        for wl, intensity in zip(wavelengths, intensities):
            f.write(f"{wl}\t{intensity}\n")

    result, metadata = read_spectra(ascii_path, format='ascii')

    assert metadata['file_format'] == 'ascii'


def test_read_spectra_unsupported_format(tmp_path):
    """Test that unsupported read format raises error."""
    with pytest.raises(ValueError, match="Unsupported or unknown format"):
        read_spectra(tmp_path / "dummy.xyz", format='unsupported')


def test_read_spectra_unknown_auto_format(tmp_path):
    """Test that unknown format in auto mode raises error."""
    unknown_path = tmp_path / "file.xyz"
    unknown_path.write_text("unknown content")

    with pytest.raises(ValueError, match="Unsupported or unknown format"):
        read_spectra(unknown_path, format='auto')


# ============================================================================
# Format-Specific Parameter Passing Tests
# ============================================================================


def test_read_excel_with_sheet_name(tmp_path):
    """Test passing sheet_name parameter through unified API."""
    excel_path = tmp_path / "multi.xlsx"
    data = create_test_data()

    # Write to specific sheet
    with pd.ExcelWriter(excel_path) as writer:
        data.to_excel(writer, sheet_name='CustomSheet')

    # Read with sheet name
    result, metadata = read_spectra(excel_path, format='excel', sheet_name='CustomSheet')

    assert metadata['sheet_name'] == 'CustomSheet'
    assert result.shape == data.shape


def test_write_excel_with_custom_options(tmp_path):
    """Test passing custom options through unified write API."""
    excel_path = tmp_path / "custom.xlsx"
    data = create_test_data()

    # Write with custom options
    write_spectra(
        data,
        excel_path,
        format='excel',
        sheet_name='Data',
        freeze_panes=(1, 1)
    )

    assert excel_path.exists()


def test_write_csv_with_float_format(tmp_path):
    """Test passing float_format through unified write API."""
    csv_path = tmp_path / "formatted.csv"
    data = create_test_data()

    write_spectra(data, csv_path, format='csv', float_format='%.4f')

    assert csv_path.exists()


def test_read_ascii_with_delimiter(tmp_path):
    """Test passing delimiter parameter through unified API."""
    ascii_path = tmp_path / "delimited.txt"

    wavelengths = np.linspace(400, 2400, 2001)
    intensities = np.random.rand(2001) * 0.5 + 0.2

    # Write with semicolon
    with open(ascii_path, 'w') as f:
        for wl, intensity in zip(wavelengths, intensities):
            f.write(f"{wl};{intensity}\n")

    # Read with delimiter specified
    result, _ = read_spectra(ascii_path, format='ascii', delimiter=';')

    assert result.shape == (1, 2001)


# ============================================================================
# Metadata Consistency Tests
# ============================================================================


def test_metadata_structure_csv(tmp_path):
    """Test that CSV metadata has consistent structure."""
    csv_path = tmp_path / "meta.csv"
    data = create_test_data()
    data.index.name = 'sample_id'
    data.to_csv(csv_path)

    result, metadata = read_spectra(csv_path)

    # Check required metadata fields
    required_fields = [
        'file_format',
        'n_spectra',
        'wavelength_range',
        'data_type',
        'type_confidence',
        'detection_method'
    ]

    for field in required_fields:
        assert field in metadata, f"Missing metadata field: {field}"


def test_metadata_structure_excel(tmp_path):
    """Test that Excel metadata has consistent structure."""
    excel_path = tmp_path / "meta.xlsx"
    data = create_test_data()
    data.index.name = 'sample_id'
    data.to_excel(excel_path)

    result, metadata = read_spectra(excel_path)

    required_fields = [
        'file_format',
        'n_spectra',
        'wavelength_range',
        'data_type',
        'type_confidence',
        'detection_method',
        'sheet_name'  # Excel-specific
    ]

    for field in required_fields:
        assert field in metadata


def test_metadata_structure_ascii(tmp_path):
    """Test that ASCII metadata has consistent structure."""
    ascii_path = tmp_path / "meta.txt"

    wavelengths = np.linspace(400, 2400, 2001)
    intensities = np.random.rand(2001) * 0.5 + 0.2

    with open(ascii_path, 'w') as f:
        for wl, intensity in zip(wavelengths, intensities):
            f.write(f"{wl}\t{intensity}\n")

    result, metadata = read_spectra(ascii_path)

    required_fields = [
        'file_format',
        'n_spectra',
        'wavelength_range',
        'data_type',
        'type_confidence',
        'detection_method'
    ]

    for field in required_fields:
        assert field in metadata


# ============================================================================
# Roundtrip Tests
# ============================================================================


def test_roundtrip_csv(tmp_path):
    """Test write-read roundtrip for CSV."""
    csv_path = tmp_path / "roundtrip.csv"
    original = create_test_data()

    write_spectra(original, csv_path, format='csv')
    result, _ = read_spectra(csv_path)

    assert result.shape == original.shape
    np.testing.assert_array_almost_equal(result.values, original.values, decimal=5)


def test_roundtrip_excel(tmp_path):
    """Test write-read roundtrip for Excel."""
    excel_path = tmp_path / "roundtrip.xlsx"
    original = create_test_data()

    write_spectra(original, excel_path, format='excel')
    result, _ = read_spectra(excel_path)

    assert result.shape == original.shape
    np.testing.assert_array_almost_equal(result.values, original.values, decimal=5)


def test_roundtrip_ascii(tmp_path):
    """Test write-read roundtrip for ASCII."""
    ascii_path = tmp_path / "roundtrip.txt"
    original = create_test_data(n_samples=1)

    write_spectra(original, ascii_path, format='ascii')
    result, _ = read_spectra(ascii_path)

    assert result.shape == original.shape
    np.testing.assert_array_almost_equal(result.values, original.values, decimal=5)


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_read_nonexistent_file():
    """Test error when reading non-existent file."""
    with pytest.raises((FileNotFoundError, ValueError)):
        read_spectra(Path("nonexistent.csv"))


def test_write_to_invalid_path():
    """Test error when writing to invalid path."""
    data = create_test_data()

    # Try to write to a directory that doesn't exist
    with pytest.raises((FileNotFoundError, OSError)):
        write_spectra(data, Path("nonexistent/dir/file.csv"), format='csv')


def test_read_with_metadata_option(tmp_path):
    """Test that metadata can be passed through read function."""
    csv_path = tmp_path / "test.csv"
    data = create_test_data()
    data.index.name = 'sample_id'
    data.to_csv(csv_path)

    # Read (metadata is output, not input, but test the API works)
    result, metadata = read_spectra(csv_path)

    assert isinstance(metadata, dict)
    assert len(metadata) > 0


def test_write_with_metadata_option(tmp_path):
    """Test that metadata can be passed through write function."""
    csv_path = tmp_path / "test.csv"
    data = create_test_data()

    custom_metadata = {
        'source': 'test',
        'date': '2025-01-01'
    }

    # Write with metadata (may not be stored in CSV, but API should accept it)
    write_spectra(data, csv_path, format='csv', metadata=custom_metadata)

    assert csv_path.exists()


# ============================================================================
# Integration Tests
# ============================================================================


def test_multiple_format_detection(tmp_path):
    """Test detecting multiple file formats in sequence."""
    # Create files of different formats
    csv_path = tmp_path / "data.csv"
    excel_path = tmp_path / "data.xlsx"
    ascii_path = tmp_path / "data.txt"

    data = create_test_data()

    # Write in different formats
    data.index.name = 'id'
    data.to_csv(csv_path)
    data.to_excel(excel_path)

    wavelengths = np.linspace(400, 2400, 2001)
    with open(ascii_path, 'w') as f:
        for wl in wavelengths:
            f.write(f"{wl}\t{data.iloc[0, list(data.columns).index(wl)]}\n")

    # Detect each
    assert detect_format(csv_path) == 'csv'
    assert detect_format(excel_path) == 'excel'
    assert detect_format(ascii_path) == 'ascii'


def test_format_agnostic_pipeline(tmp_path):
    """Test that the same code works for different formats."""
    data = create_test_data()

    formats_and_paths = [
        ('csv', tmp_path / "data.csv"),
        ('excel', tmp_path / "data.xlsx"),
    ]

    for fmt, path in formats_and_paths:
        # Write
        write_spectra(data, path, format=fmt)

        # Read
        result, metadata = read_spectra(path, format='auto')

        # Verify
        assert metadata['file_format'] == fmt
        assert result.shape[0] == data.shape[0]
        np.testing.assert_array_almost_equal(result.values, data.values, decimal=4)
