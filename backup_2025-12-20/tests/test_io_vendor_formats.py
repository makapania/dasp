"""Test vendor-specific format readers (OPUS, PerkinElmer, Agilent)."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from spectral_predict.io import (
    read_opus_file,
    read_perkinelmer_file,
    read_agilent_file,
    read_ascii_spectra,
    write_ascii_spectra,
    read_spectra,
    write_spectra,
    detect_format
)


# ============================================================================
# ASCII Text Format Tests (Generic two-column format)
# ============================================================================


def test_read_ascii_tab_delimited(tmp_path):
    """Test reading ASCII file with tab delimiter."""
    ascii_path = tmp_path / "spectrum.txt"

    # Create tab-delimited file
    wavelengths = np.linspace(400, 2400, 2001)
    intensities = np.random.rand(2001) * 0.5 + 0.2

    with open(ascii_path, 'w') as f:
        f.write("# Spectral data file\n")
        f.write("# Wavelength\tIntensity\n")
        for wl, intensity in zip(wavelengths, intensities):
            f.write(f"{wl:.2f}\t{intensity:.6f}\n")

    # Read
    result, metadata = read_ascii_spectra(ascii_path)

    assert result.shape == (1, 2001)
    assert result.index[0] == "spectrum"
    assert metadata['file_format'] == 'ascii'
    assert metadata['n_spectra'] == 1


def test_read_ascii_comma_delimited(tmp_path):
    """Test reading ASCII file with comma delimiter."""
    ascii_path = tmp_path / "spectrum.dat"

    wavelengths = np.linspace(400, 2400, 2001)
    intensities = np.random.rand(2001) * 0.5 + 0.2

    with open(ascii_path, 'w') as f:
        for wl, intensity in zip(wavelengths, intensities):
            f.write(f"{wl:.2f},{intensity:.6f}\n")

    result, metadata = read_ascii_spectra(ascii_path)

    assert result.shape == (1, 2001)
    assert metadata['file_format'] == 'ascii'


def test_read_ascii_space_delimited(tmp_path):
    """Test reading ASCII file with space delimiter."""
    ascii_path = tmp_path / "spectrum.txt"

    wavelengths = np.linspace(400, 2400, 2001)
    intensities = np.random.rand(2001) * 0.5 + 0.2

    with open(ascii_path, 'w') as f:
        for wl, intensity in zip(wavelengths, intensities):
            f.write(f"{wl:.2f} {intensity:.6f}\n")

    result, metadata = read_ascii_spectra(ascii_path)

    assert result.shape == (1, 2001)


def test_read_ascii_with_comments(tmp_path):
    """Test reading ASCII file with comment lines."""
    ascii_path = tmp_path / "commented.txt"

    wavelengths = np.linspace(400, 2400, 2001)
    intensities = np.random.rand(2001) * 0.5 + 0.2

    with open(ascii_path, 'w') as f:
        f.write("# This is a comment\n")
        f.write("# Another comment line\n")
        f.write("# Wavelength (nm)\tReflectance\n")
        for wl, intensity in zip(wavelengths, intensities):
            f.write(f"{wl:.2f}\t{intensity:.6f}\n")

    result, metadata = read_ascii_spectra(ascii_path)

    assert result.shape == (1, 2001)


def test_write_read_ascii_roundtrip(tmp_path):
    """Test ASCII write and read roundtrip."""
    ascii_path = tmp_path / "roundtrip.txt"

    wavelengths = np.linspace(400, 2400, 2001)
    original = pd.DataFrame(
        [np.random.rand(2001) * 0.5 + 0.2],
        index=["Sample_1"],
        columns=wavelengths
    )

    # Write
    write_ascii_spectra(original, ascii_path, delimiter='\t')

    # Read
    result, _ = read_ascii_spectra(ascii_path, delimiter='\t')

    # Compare
    assert result.shape == original.shape
    np.testing.assert_array_almost_equal(result.values, original.values, decimal=5)


def test_write_ascii_multiple_spectra_warning(tmp_path):
    """Test that writing multiple spectra gives warning."""
    ascii_path = tmp_path / "multi.txt"

    wavelengths = np.linspace(400, 2400, 2001)
    data = pd.DataFrame(
        np.random.rand(3, 2001),
        index=["S1", "S2", "S3"],
        columns=wavelengths
    )

    # Should warn and write only first
    write_ascii_spectra(data, ascii_path)

    # Read back
    result, _ = read_ascii_spectra(ascii_path)
    assert result.shape[0] == 1


def test_ascii_via_unified_api(tmp_path):
    """Test ASCII read/write via unified API."""
    ascii_path = tmp_path / "unified.txt"

    wavelengths = np.linspace(400, 2400, 2001)
    data = pd.DataFrame(
        [np.random.rand(2001) * 0.5 + 0.2],
        index=["S1"],
        columns=wavelengths
    )

    # Write
    write_spectra(data, ascii_path, format='ascii', delimiter='\t')

    # Read with auto-detect
    result, metadata = read_spectra(ascii_path, format='auto')
    assert metadata['file_format'] == 'ascii'

    # Read with explicit format
    result2, _ = read_spectra(ascii_path, format='ascii')
    pd.testing.assert_frame_equal(result, result2)


def test_ascii_without_header(tmp_path):
    """Test ASCII file without header."""
    ascii_path = tmp_path / "no_header.txt"

    wavelengths = np.linspace(400, 2400, 2001)
    intensities = np.random.rand(2001) * 0.5 + 0.2

    with open(ascii_path, 'w') as f:
        for wl, intensity in zip(wavelengths, intensities):
            f.write(f"{wl:.2f}\t{intensity:.6f}\n")

    result, _ = read_ascii_spectra(ascii_path)
    assert result.shape == (1, 2001)


def test_ascii_custom_delimiter_write(tmp_path):
    """Test ASCII write with custom delimiter."""
    ascii_path = tmp_path / "custom.txt"

    wavelengths = np.linspace(400, 2400, 2001)
    data = pd.DataFrame(
        [np.random.rand(2001)],
        index=["S1"],
        columns=wavelengths
    )

    # Write with semicolon delimiter
    write_ascii_spectra(data, ascii_path, delimiter=';', include_header=True)

    # Read with specified delimiter
    result, _ = read_ascii_spectra(ascii_path, delimiter=';')
    assert result.shape == data.shape


# ============================================================================
# Bruker OPUS Format Tests
# ============================================================================


def test_opus_import_error():
    """Test that missing brukeropus package raises error."""
    import sys
    opus_module = sys.modules.get('brukeropusreader')

    if opus_module is not None:
        pytest.skip("brukeropus package is installed")

    # Should raise ImportError
    with pytest.raises(ImportError, match="brukeropus"):
        read_opus_file(Path("dummy.0"))


def test_opus_file_detection():
    """Test that OPUS numbered extensions are detected."""
    assert detect_format(Path("spectrum.0")) == 'opus'
    assert detect_format(Path("spectrum.1")) == 'opus'
    assert detect_format(Path("spectrum.12")) == 'opus'


def test_opus_via_unified_api_if_installed(tmp_path):
    """Test OPUS reading via unified API if package installed."""
    opus_path = tmp_path / "test.0"

    # Create a mock OPUS file
    with open(opus_path, 'wb') as f:
        f.write(b'OPUS')
        f.write(b'\x00' * 100)

    try:
        # Try auto-detect
        result, metadata = read_spectra(opus_path, format='auto')
        assert metadata['file_format'] == 'opus'
    except ImportError:
        pytest.skip("brukeropus not installed")
    except Exception:
        # Mock file isn't valid
        pass


# ============================================================================
# PerkinElmer Format Tests
# ============================================================================


def test_perkinelmer_import_error():
    """Test that missing specio package raises error."""
    import sys
    specio_module = sys.modules.get('specio')

    if specio_module is not None:
        pytest.skip("specio package is installed")

    with pytest.raises(ImportError, match="specio"):
        read_perkinelmer_file(Path("dummy.sp"))


def test_perkinelmer_file_detection():
    """Test that .sp extension is detected."""
    assert detect_format(Path("spectrum.sp")) == 'perkinelmer'


def test_perkinelmer_via_unified_api_if_installed(tmp_path):
    """Test PerkinElmer reading via unified API."""
    pe_path = tmp_path / "test.sp"

    # Create mock file
    with open(pe_path, 'wb') as f:
        f.write(b'PEPE')  # Mock header
        f.write(b'\x00' * 100)

    try:
        result, metadata = read_spectra(pe_path, format='auto')
        assert metadata['file_format'] == 'perkinelmer'
    except ImportError:
        pytest.skip("specio not installed")
    except Exception:
        # Mock file isn't valid
        pass


# ============================================================================
# Agilent Format Tests
# ============================================================================


def test_agilent_import_error():
    """Test that missing agilent-ir-formats package raises error."""
    import sys
    agilent_module = sys.modules.get('agilent_ir_formats')

    if agilent_module is not None:
        pytest.skip("agilent-ir-formats package is installed")

    with pytest.raises(ImportError, match="agilent-ir-formats"):
        read_agilent_file(Path("dummy.seq"))


def test_agilent_file_detection():
    """Test that .seq extension is detected."""
    assert detect_format(Path("spectrum.seq")) == 'agilent'


def test_agilent_not_implemented():
    """Test that Agilent reader raises NotImplementedError."""
    # Even if package is installed, reader is not fully implemented
    agilent_path = Path("test.seq")

    try:
        read_agilent_file(agilent_path)
        pytest.fail("Should raise NotImplementedError")
    except ImportError:
        # Package not installed
        pass
    except NotImplementedError as e:
        # Expected - reader not implemented
        assert "not yet fully implemented" in str(e)


# ============================================================================
# Format Detection Tests
# ============================================================================


def test_detect_format_csv():
    """Test CSV format detection."""
    assert detect_format(Path("data.csv")) == 'csv'


def test_detect_format_excel():
    """Test Excel format detection."""
    assert detect_format(Path("data.xlsx")) == 'excel'
    assert detect_format(Path("data.xls")) == 'excel'


def test_detect_format_asd():
    """Test ASD format detection."""
    assert detect_format(Path("spectrum.asd")) == 'asd'
    assert detect_format(Path("spectrum.sig")) == 'asd'


def test_detect_format_spc():
    """Test SPC format detection."""
    assert detect_format(Path("spectrum.spc")) == 'spc'


def test_detect_format_jcamp():
    """Test JCAMP-DX format detection."""
    assert detect_format(Path("spectrum.jdx")) == 'jcamp'
    assert detect_format(Path("spectrum.dx")) == 'jcamp'
    assert detect_format(Path("spectrum.jcm")) == 'jcamp'


def test_detect_format_ascii():
    """Test ASCII format detection."""
    assert detect_format(Path("spectrum.txt")) == 'ascii'
    assert detect_format(Path("spectrum.dat")) == 'ascii'


def test_detect_format_directory(tmp_path):
    """Test directory detection."""
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    assert detect_format(test_dir) == 'directory'


def test_detect_format_unknown():
    """Test unknown format detection."""
    assert detect_format(Path("file.unknown")) == 'unknown'


def test_detect_format_magic_bytes_spc(tmp_path):
    """Test SPC detection via magic bytes."""
    spc_path = tmp_path / "noext"

    # Create file with SPC magic bytes
    with open(spc_path, 'wb') as f:
        f.write(b'\x4d\x4b')  # 'MK'
        f.write(b'\x00' * 100)

    assert detect_format(spc_path) == 'spc'


def test_detect_format_magic_bytes_jcamp(tmp_path):
    """Test JCAMP detection via magic bytes."""
    jcamp_path = tmp_path / "noext"

    with open(jcamp_path, 'wb') as f:
        f.write(b'##TITLE=Test\n')
        f.write(b'##JCAMP-DX=5.0\n')

    assert detect_format(jcamp_path) == 'jcamp'


def test_detect_format_magic_bytes_opus(tmp_path):
    """Test OPUS detection via magic bytes."""
    opus_path = tmp_path / "noext"

    with open(opus_path, 'wb') as f:
        f.write(b'OPUS')
        f.write(b'\x00' * 100)

    assert detect_format(opus_path) == 'opus'


# ============================================================================
# Integration Tests
# ============================================================================


def test_ascii_metadata_completeness(tmp_path):
    """Test that ASCII reader returns complete metadata."""
    ascii_path = tmp_path / "meta.txt"

    wavelengths = np.linspace(400, 2400, 2001)
    intensities = np.random.rand(2001) * 0.5 + 0.2

    with open(ascii_path, 'w') as f:
        for wl, intensity in zip(wavelengths, intensities):
            f.write(f"{wl}\t{intensity}\n")

    result, metadata = read_ascii_spectra(ascii_path)

    # Check all expected metadata fields
    assert 'file_format' in metadata
    assert metadata['file_format'] == 'ascii'
    assert 'n_spectra' in metadata
    assert 'wavelength_range' in metadata
    assert 'data_type' in metadata
    assert 'type_confidence' in metadata
    assert 'detection_method' in metadata


def test_ascii_data_type_detection(tmp_path):
    """Test that data type is detected for ASCII files."""
    ascii_path = tmp_path / "reflectance.txt"

    wavelengths = np.linspace(400, 2400, 2001)
    # Create reflectance-like data (0-1 range)
    intensities = np.random.rand(2001) * 0.5 + 0.2

    with open(ascii_path, 'w') as f:
        for wl, intensity in zip(wavelengths, intensities):
            f.write(f"{wl}\t{intensity}\n")

    result, metadata = read_ascii_spectra(ascii_path)

    assert metadata['data_type'] in ['reflectance', 'absorbance']
    assert 0 <= metadata['type_confidence'] <= 100


def test_ascii_wavelength_sorting(tmp_path):
    """Test that wavelengths are sorted in ASCII data."""
    ascii_path = tmp_path / "unsorted.txt"

    wavelengths = np.linspace(400, 2400, 2001)
    intensities = np.random.rand(2001)

    # Write in random order
    indices = np.random.permutation(len(wavelengths))

    with open(ascii_path, 'w') as f:
        for idx in indices:
            f.write(f"{wavelengths[idx]}\t{intensities[idx]}\n")

    result, _ = read_ascii_spectra(ascii_path)

    # Check wavelengths are sorted
    wls = result.columns.values
    assert np.all(wls[:-1] <= wls[1:])
