"""Unit tests for spectral data type detection."""

import pytest
import numpy as np
import pandas as pd

from spectral_predict.io import detect_spectral_data_type


def _gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def test_detect_reflectance_with_low_negative_fraction():
    wavelengths = np.arange(1000, 1100)
    baseline = 0.6
    dip1 = 0.15 * _gaussian(wavelengths, 1030, 4)
    dip2 = 0.12 * _gaussian(wavelengths, 1070, 6)
    spectrum = baseline - dip1 - dip2

    rng = np.random.default_rng(42)
    spectra = []
    for _ in range(12):
        noise = rng.normal(scale=0.005, size=spectrum.size)
        spectra.append(spectrum + noise)
    X = pd.DataFrame(spectra, columns=wavelengths)

    data_type, confidence, _ = detect_spectral_data_type(X)
    assert data_type == "reflectance"
    assert confidence >= 50


def test_detect_absorbance_when_values_exceed_reflectance_range():
    wavelengths = np.arange(1000, 1100)
    baseline = 1.1
    peak1 = 0.5 * _gaussian(wavelengths, 1030, 4)
    peak2 = 0.4 * _gaussian(wavelengths, 1070, 6)
    spectrum = baseline + peak1 + peak2

    X = pd.DataFrame([spectrum], columns=wavelengths)
    data_type, confidence, _ = detect_spectral_data_type(X)
    assert data_type == "absorbance"
    assert confidence >= 60


# ===========================================================================
# Additional tests for broader coverage
# ===========================================================================

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nir_wavelengths():
    """NIR wavelength range (900-2500 nm)."""
    return np.arange(900, 2501, 4)


@pytest.fixture
def vis_wavelengths():
    """VIS wavelength range (400-700 nm)."""
    return np.arange(400, 701, 1)


# ---------------------------------------------------------------------------
# NIR range spectra
# ---------------------------------------------------------------------------

def test_detect_reflectance_nir_range(nir_wavelengths):
    """Typical NIR reflectance: values in 0.2-0.8 range over 900-2500 nm."""
    rng = np.random.default_rng(123)
    baseline = 0.55
    spectra = []
    for _ in range(10):
        dip = 0.15 * _gaussian(nir_wavelengths, 1450, 30)
        dip2 = 0.10 * _gaussian(nir_wavelengths, 1930, 40)
        noise = rng.normal(scale=0.003, size=nir_wavelengths.size)
        spectra.append(baseline - dip - dip2 + noise)

    X = pd.DataFrame(spectra, columns=nir_wavelengths.astype(float))
    data_type, confidence, _ = detect_spectral_data_type(X)
    assert data_type == "reflectance"
    assert confidence >= 50


def test_detect_absorbance_nir_range(nir_wavelengths):
    """NIR absorbance: values spanning 0-2+ AU range over 900-2500 nm."""
    rng = np.random.default_rng(456)
    baseline = 0.8
    spectra = []
    for _ in range(8):
        peak = 0.9 * _gaussian(nir_wavelengths, 1450, 30)
        peak2 = 0.7 * _gaussian(nir_wavelengths, 1930, 40)
        noise = rng.normal(scale=0.005, size=nir_wavelengths.size)
        spectra.append(baseline + peak + peak2 + noise)

    X = pd.DataFrame(spectra, columns=nir_wavelengths.astype(float))
    data_type, confidence, _ = detect_spectral_data_type(X)
    assert data_type == "absorbance"
    assert confidence >= 50


# ---------------------------------------------------------------------------
# Negative-log transformed spectra
# ---------------------------------------------------------------------------

def test_detect_negative_log_transformed_as_absorbance(nir_wavelengths):
    """Negative-log transform of low-reflectance spectra creates absorbance values
    well above 1.0, which the detector should classify as absorbance.

    Using a low reflectance baseline (0.08) so -log10(0.08) ~ 1.1, with
    absorption features dipping to ~0.02 giving -log10(0.02) ~ 1.7.
    """
    rng = np.random.default_rng(789)
    # Low reflectance sample: baseline ~0.08, strong absorption dips to ~0.02
    reflectance = 0.08 - 0.06 * _gaussian(nir_wavelengths, 1450, 80)
    reflectance = np.clip(reflectance, 0.005, 1.0)
    absorbance = -np.log10(reflectance)

    spectra = []
    for _ in range(5):
        noise = rng.normal(scale=0.02, size=nir_wavelengths.size)
        spectra.append(absorbance + noise)

    X = pd.DataFrame(spectra, columns=nir_wavelengths.astype(float))
    data_type, confidence, _ = detect_spectral_data_type(X)
    assert data_type == "absorbance"


# ---------------------------------------------------------------------------
# Noisy spectra
# ---------------------------------------------------------------------------

def test_detect_reflectance_with_heavy_noise(vis_wavelengths):
    """Reflectance with heavy noise should still be detected as reflectance
    when the mean and range remain in the reflectance domain."""
    rng = np.random.default_rng(101)
    baseline = 0.6
    noise = rng.normal(scale=0.05, size=vis_wavelengths.size)
    spectrum = np.clip(baseline + noise, 0.0, 1.0)

    X = pd.DataFrame([spectrum] * 5, columns=vis_wavelengths.astype(float))
    data_type, confidence, _ = detect_spectral_data_type(X)
    assert data_type == "reflectance"
    assert confidence >= 50


def test_detect_absorbance_with_noise_induced_negatives(nir_wavelengths):
    """Low-absorbance spectra with noise can go negative. These should still
    be detected as absorbance due to the presence of negative values."""
    rng = np.random.default_rng(202)
    baseline = 0.15
    peak = 0.5 * _gaussian(nir_wavelengths, 1450, 30)
    spectra = []
    for _ in range(10):
        noise = rng.normal(scale=0.08, size=nir_wavelengths.size)
        spectra.append(baseline + peak + noise)

    X = pd.DataFrame(spectra, columns=nir_wavelengths.astype(float))
    data_type, confidence, _ = detect_spectral_data_type(X)
    # With negative values and peaks going well above baseline,
    # this should lean absorbance
    assert data_type == "absorbance"


# ---------------------------------------------------------------------------
# Borderline / ambiguous cases
# ---------------------------------------------------------------------------

def test_detect_borderline_max_near_one():
    """Spectra with max just above 1.0 (e.g., 1.05) are ambiguous but should
    still return a valid result without crashing."""
    wavelengths = np.arange(400, 500)
    spectrum = np.full_like(wavelengths, 0.9, dtype=float)
    spectrum[50] = 1.05  # slight overshoot
    X = pd.DataFrame([spectrum], columns=wavelengths.astype(float))

    data_type, confidence, method = detect_spectral_data_type(X)
    assert data_type in ("reflectance", "absorbance")
    assert 0 <= confidence <= 100
    assert isinstance(method, str)


def test_detect_percent_reflectance():
    """Percent reflectance (0-100 scale) should be identified as reflectance."""
    wavelengths = np.arange(350, 2501, 5)
    rng = np.random.default_rng(303)
    spectrum = 60.0 + 10.0 * rng.standard_normal(wavelengths.size)
    spectrum = np.clip(spectrum, 5.0, 95.0)

    X = pd.DataFrame([spectrum], columns=wavelengths.astype(float))
    data_type, confidence, _ = detect_spectral_data_type(X)
    assert data_type == "reflectance"
    assert confidence >= 50


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_detect_single_wavelength():
    """A single-column DataFrame should not crash the detector."""
    X = pd.DataFrame({"500.0": [0.5, 0.6, 0.55]})
    data_type, confidence, method = detect_spectral_data_type(X)
    assert data_type in ("reflectance", "absorbance")
    assert 0 <= confidence <= 100


def test_detect_all_nan_returns_default():
    """All-NaN data should return a default with low confidence."""
    wavelengths = np.arange(400, 500)
    nan_row = np.full(len(wavelengths), np.nan)
    X = pd.DataFrame([nan_row], columns=wavelengths.astype(float))

    data_type, confidence, method = detect_spectral_data_type(X)
    # The function returns reflectance/50% for no_valid_data
    assert data_type == "reflectance"
    assert confidence == 50.0
    assert "no_valid_data" in method


def test_detect_very_large_absorbance_values():
    """Very large absorbance values (saturated detector) should still be absorbance."""
    wavelengths = np.arange(1000, 1100)
    spectrum = np.full(len(wavelengths), 3.5)
    spectrum[40:60] = 5.0  # saturation peak

    X = pd.DataFrame([spectrum], columns=wavelengths.astype(float))
    data_type, confidence, _ = detect_spectral_data_type(X)
    # Values well above 1.5 are clearly absorbance (not percent reflectance
    # since they are in 3-5 range, not 0-100 range)
    assert data_type == "absorbance"


# ---------------------------------------------------------------------------
# Metadata keyword influence
# ---------------------------------------------------------------------------

def test_metadata_reflectance_keyword_boosts_reflectance():
    """Metadata containing reflectance keywords should increase reflectance confidence."""
    wavelengths = np.arange(400, 500)
    # Ambiguous values around 0.5
    spectrum = np.full(len(wavelengths), 0.5)
    X = pd.DataFrame([spectrum], columns=wavelengths.astype(float))

    _, conf_no_meta, _ = detect_spectral_data_type(X)
    _, conf_meta, _ = detect_spectral_data_type(
        X, metadata={"column_names": ["pct_reflect_500"]}
    )
    # With reflectance keyword, confidence should be at least as high
    assert conf_meta >= conf_no_meta
