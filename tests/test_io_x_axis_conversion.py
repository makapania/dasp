"""Unit tests for x-axis unit conversion in spectral_predict.io."""

import pytest
import numpy as np
import pandas as pd

from spectral_predict.io import convert_x_axis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vis_nir_wavelengths_nm():
    """Typical VIS-NIR wavelengths in nm (ascending)."""
    return np.array([400.0, 500.0, 800.0, 1000.0, 2000.0, 2500.0])


@pytest.fixture
def ftir_wavenumbers_cm1():
    """Typical FTIR wavenumbers in cm-1 (descending)."""
    return np.array([4000.0, 2500.0, 2000.0, 1500.0, 1000.0, 500.0])


# ---------------------------------------------------------------------------
# Basic conversion tests
# ---------------------------------------------------------------------------

def test_convert_nm_to_cm1_known_values():
    """500 nm should equal 20000 cm-1; 1000 nm should equal 10000 cm-1."""
    nm_values = np.array([500.0, 1000.0, 2000.0])
    result = convert_x_axis(nm_values, "nm", "cm-1")
    expected = np.array([20000.0, 10000.0, 5000.0])
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_convert_cm1_to_nm_known_values():
    """20000 cm-1 should equal 500 nm; 10000 cm-1 should equal 1000 nm."""
    cm1_values = np.array([20000.0, 10000.0, 5000.0])
    result = convert_x_axis(cm1_values, "cm-1", "nm")
    expected = np.array([500.0, 1000.0, 2000.0])
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_convert_same_unit_returns_unchanged():
    """Converting from nm to nm should return the same array."""
    values = np.array([400.0, 600.0, 800.0])
    result_nm = convert_x_axis(values, "nm", "nm")
    np.testing.assert_array_equal(result_nm, values)

    result_cm1 = convert_x_axis(values, "cm-1", "cm-1")
    np.testing.assert_array_equal(result_cm1, values)


# ---------------------------------------------------------------------------
# Roundtrip conversion
# ---------------------------------------------------------------------------

def test_roundtrip_nm_cm1_nm(vis_nir_wavelengths_nm):
    """nm -> cm-1 -> nm should recover the original values."""
    intermediate = convert_x_axis(vis_nir_wavelengths_nm, "nm", "cm-1")
    recovered = convert_x_axis(intermediate, "cm-1", "nm")
    np.testing.assert_allclose(recovered, vis_nir_wavelengths_nm, rtol=1e-12)


def test_roundtrip_cm1_nm_cm1(ftir_wavenumbers_cm1):
    """cm-1 -> nm -> cm-1 should recover the original values."""
    intermediate = convert_x_axis(ftir_wavenumbers_cm1, "cm-1", "nm")
    recovered = convert_x_axis(intermediate, "nm", "cm-1")
    np.testing.assert_allclose(recovered, ftir_wavenumbers_cm1, rtol=1e-12)


# ---------------------------------------------------------------------------
# Order inversion
# ---------------------------------------------------------------------------

def test_ascending_nm_becomes_descending_cm1(vis_nir_wavelengths_nm):
    """Ascending nm values should produce descending cm-1 values."""
    assert np.all(np.diff(vis_nir_wavelengths_nm) > 0), "Precondition: nm ascending"
    result = convert_x_axis(vis_nir_wavelengths_nm, "nm", "cm-1")
    assert np.all(np.diff(result) < 0), "cm-1 should be descending"


def test_descending_cm1_becomes_ascending_nm(ftir_wavenumbers_cm1):
    """Descending cm-1 values should produce ascending nm values."""
    assert np.all(np.diff(ftir_wavenumbers_cm1) < 0), "Precondition: cm-1 descending"
    result = convert_x_axis(ftir_wavenumbers_cm1, "cm-1", "nm")
    assert np.all(np.diff(result) > 0), "nm should be ascending"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_convert_single_value():
    """Conversion should work on a single-element array."""
    result = convert_x_axis(np.array([250.0]), "nm", "cm-1")
    expected = np.array([1e7 / 250.0])
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_convert_very_small_wavelength():
    """Very small nm (UV) should produce very large cm-1."""
    uv_nm = np.array([100.0])  # Deep UV
    result = convert_x_axis(uv_nm, "nm", "cm-1")
    np.testing.assert_allclose(result, np.array([100000.0]), rtol=1e-12)


def test_convert_very_large_wavelength():
    """Very large nm (mid-IR) should produce small cm-1."""
    mir_nm = np.array([25000.0])  # 25 microns in nm
    result = convert_x_axis(mir_nm, "nm", "cm-1")
    np.testing.assert_allclose(result, np.array([400.0]), rtol=1e-12)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_unsupported_unit_raises_value_error():
    """Unsupported unit strings should raise ValueError."""
    values = np.array([500.0, 1000.0])
    with pytest.raises(ValueError, match="Unsupported conversion"):
        convert_x_axis(values, "nm", "micrometers")

    with pytest.raises(ValueError, match="Unsupported conversion"):
        convert_x_axis(values, "inches", "nm")


# ---------------------------------------------------------------------------
# Input type flexibility
# ---------------------------------------------------------------------------

def test_convert_accepts_python_list():
    """convert_x_axis should accept a plain Python list, not just ndarray."""
    result = convert_x_axis([500.0, 1000.0], "nm", "cm-1")
    expected = np.array([20000.0, 10000.0])
    np.testing.assert_allclose(result, expected, rtol=1e-12)
    assert isinstance(result, np.ndarray)
