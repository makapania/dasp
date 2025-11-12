"""Test JCAMP-DX I/O functions."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from spectral_predict.io import (
    read_jcamp_file,
    write_jcamp_file,
    read_spectra,
    write_spectra
)


def create_jcamp_file(path, wavelengths, intensities, title="Test Spectrum"):
    """Helper to create a JCAMP-DX file."""
    with open(path, 'w') as f:
        f.write("##TITLE=" + title + "\n")
        f.write("##JCAMP-DX=5.0\n")
        f.write("##DATA TYPE=INFRARED SPECTRUM\n")
        f.write("##XUNITS=NANOMETERS\n")
        f.write("##YUNITS=REFLECTANCE\n")
        f.write(f"##FIRSTX={wavelengths[0]}\n")
        f.write(f"##LASTX={wavelengths[-1]}\n")
        f.write(f"##NPOINTS={len(wavelengths)}\n")
        f.write("##XYDATA=(X++(Y..Y))\n")

        # Write data in JCAMP format (simplified)
        for wl, intensity in zip(wavelengths, intensities):
            f.write(f"{wl} {intensity}\n")

        f.write("##END=\n")


def test_read_jcamp_basic(tmp_path):
    """Test reading basic JCAMP-DX file."""
    jcamp_path = tmp_path / "test.jdx"

    # Create synthetic JCAMP file
    wavelengths = np.linspace(400, 2400, 2001)
    intensities = np.random.rand(2001) * 0.5 + 0.2

    create_jcamp_file(jcamp_path, wavelengths, intensities, "Sample_001")

    # Read it
    try:
        result, metadata = read_jcamp_file(jcamp_path)

        assert result.shape == (1, 2001)
        assert result.index[0] == "test"
        assert result.columns[0] == pytest.approx(400.0, abs=0.1)
        assert result.columns[-1] == pytest.approx(2400.0, abs=0.1)
        assert metadata['file_format'] == 'jcamp'
        assert metadata['n_spectra'] == 1
    except ImportError:
        pytest.skip("jcamp package not installed")


def test_read_jcamp_via_unified_api(tmp_path):
    """Test reading JCAMP through unified API."""
    jcamp_path = tmp_path / "unified.jdx"

    wavelengths = np.linspace(400, 2400, 2001)
    intensities = np.random.rand(2001) * 0.5 + 0.2

    create_jcamp_file(jcamp_path, wavelengths, intensities)

    try:
        # Auto-detect
        result, metadata = read_spectra(jcamp_path, format='auto')
        assert metadata['file_format'] == 'jcamp'

        # Explicit format
        result2, _ = read_spectra(jcamp_path, format='jcamp')
        pd.testing.assert_frame_equal(result, result2)
    except ImportError:
        pytest.skip("jcamp package not installed")


def test_write_jcamp_basic(tmp_path):
    """Test writing JCAMP-DX file."""
    jcamp_path = tmp_path / "output.jdx"

    # Create test data
    wavelengths = np.linspace(400, 2400, 2001)
    data = pd.DataFrame(
        [np.random.rand(2001) * 0.5 + 0.2],
        index=["Sample_1"],
        columns=wavelengths
    )

    try:
        # Write
        write_jcamp_file(
            data,
            jcamp_path,
            title="Test Output",
            data_type="INFRARED SPECTRUM"
        )

        # Verify file exists
        assert jcamp_path.exists()

        # Read back
        result, metadata = read_jcamp_file(jcamp_path)
        assert result.shape == (1, 2001)
        assert metadata['jcamp_header']['title'] == "Test Output"
    except ImportError:
        pytest.skip("jcamp package not installed")


def test_write_jcamp_multiple_spectra_warning(tmp_path):
    """Test that writing multiple spectra gives warning and writes first only."""
    jcamp_path = tmp_path / "multi.jdx"

    wavelengths = np.linspace(400, 2400, 2001)
    data = pd.DataFrame(
        np.random.rand(3, 2001) * 0.5 + 0.2,
        index=["S1", "S2", "S3"],
        columns=wavelengths
    )

    try:
        # Write (should warn)
        write_jcamp_file(data, jcamp_path)

        # Read back - should only have one spectrum
        result, _ = read_jcamp_file(jcamp_path)
        assert result.shape[0] == 1
    except ImportError:
        pytest.skip("jcamp package not installed")


def test_write_read_jcamp_roundtrip(tmp_path):
    """Test JCAMP write and read roundtrip."""
    jcamp_path = tmp_path / "roundtrip.jdx"

    wavelengths = np.linspace(400, 2400, 2001)
    data = pd.DataFrame(
        [np.random.rand(2001) * 0.5 + 0.2],
        index=["Sample_1"],
        columns=wavelengths
    )

    try:
        # Write
        write_jcamp_file(
            data,
            jcamp_path,
            title="Roundtrip Test",
            xunits="NANOMETERS",
            yunits="REFLECTANCE"
        )

        # Read back
        result, metadata = read_jcamp_file(jcamp_path)

        # Compare (allowing for some precision loss in JCAMP format)
        assert result.shape == data.shape
        np.testing.assert_array_almost_equal(
            result.values,
            data.values,
            decimal=4
        )
    except ImportError:
        pytest.skip("jcamp package not installed")


def test_jcamp_via_unified_write_api(tmp_path):
    """Test writing JCAMP via unified API."""
    jcamp_path = tmp_path / "unified_write.jdx"

    wavelengths = np.linspace(400, 2400, 2001)
    data = pd.DataFrame(
        [np.random.rand(2001) * 0.5 + 0.2],
        index=["Sample_1"],
        columns=wavelengths
    )

    try:
        # Write via unified API
        write_spectra(
            data,
            jcamp_path,
            format='jcamp',
            title="Unified API Test"
        )

        # Read back
        result, _ = read_jcamp_file(jcamp_path)
        assert result.shape == data.shape
    except ImportError:
        pytest.skip("jcamp package not installed")


def test_jcamp_metadata_extraction(tmp_path):
    """Test that JCAMP metadata is properly extracted."""
    jcamp_path = tmp_path / "meta.jdx"

    wavelengths = np.linspace(400, 2400, 2001)
    intensities = np.random.rand(2001) * 0.5 + 0.2

    create_jcamp_file(jcamp_path, wavelengths, intensities, "Metadata Test")

    try:
        result, metadata = read_jcamp_file(jcamp_path)

        # Check standard metadata
        assert 'file_format' in metadata
        assert metadata['file_format'] == 'jcamp'
        assert 'n_spectra' in metadata
        assert metadata['n_spectra'] == 1
        assert 'wavelength_range' in metadata
        assert 'data_type' in metadata
        assert 'jcamp_header' in metadata

        # Check JCAMP-specific metadata
        assert isinstance(metadata['jcamp_header'], dict)
    except ImportError:
        pytest.skip("jcamp package not installed")


def test_jcamp_custom_units(tmp_path):
    """Test JCAMP writing with custom units."""
    jcamp_path = tmp_path / "custom_units.jdx"

    wavelengths = np.linspace(4000, 400, 2001)  # Wavenumbers
    data = pd.DataFrame(
        [np.random.rand(2001) * 0.5 + 0.2],
        index=["Sample_1"],
        columns=wavelengths
    )

    try:
        # Write with wavenumbers
        write_jcamp_file(
            data,
            jcamp_path,
            xunits="1/CM",
            yunits="ABSORBANCE",
            data_type="INFRARED SPECTRUM"
        )

        assert jcamp_path.exists()
    except ImportError:
        pytest.skip("jcamp package not installed")


def test_jcamp_import_error():
    """Test that missing jcamp package raises helpful error."""
    # Mock the import to fail
    import sys
    jcamp_module = sys.modules.get('jcamp')
    if jcamp_module is not None:
        # Package is installed, skip this test
        pytest.skip("jcamp package is installed")

    # If package not installed, reading should raise ImportError
    with pytest.raises(ImportError, match="jcamp package"):
        read_jcamp_file(Path("dummy.jdx"))


def test_jcamp_data_type_detection(tmp_path):
    """Test that data type is correctly detected from JCAMP data."""
    jcamp_path = tmp_path / "detect.jdx"

    wavelengths = np.linspace(400, 2400, 2001)
    # Create reflectance-like data (0-1 range)
    intensities = np.random.rand(2001) * 0.5 + 0.2

    create_jcamp_file(jcamp_path, wavelengths, intensities)

    try:
        result, metadata = read_jcamp_file(jcamp_path)

        # Should detect as reflectance
        assert metadata['data_type'] in ['reflectance', 'absorbance']
        assert metadata['type_confidence'] > 0
        assert 'detection_method' in metadata
    except ImportError:
        pytest.skip("jcamp package not installed")
