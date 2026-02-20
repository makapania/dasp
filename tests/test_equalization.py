"""Comprehensive tests for spectral_predict.equalization module.

Tests cover:
- choose_common_grid: common wavelength grid selection from instrument profiles
- build_equalization_mapping_for_instrument: mapping callable construction
- equalize_dataset: multi-instrument equalization with shape/wavelength verification
"""

import numpy as np
import pytest

from spectral_predict.equalization import (
    build_equalization_mapping_for_instrument,
    choose_common_grid,
    equalize_dataset,
)
from spectral_predict.instrument_profiles import InstrumentProfile


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def instrument_a_profile():
    """InstrumentProfile for a high-resolution instrument (400-2500nm, 1nm spacing)."""
    wavelengths = np.arange(400.0, 2501.0, 1.0)
    return InstrumentProfile(
        instrument_id="InstrumentA",
        wavelengths=wavelengths,
        delta_lambda_med=1.0,
    )


@pytest.fixture
def instrument_b_profile():
    """InstrumentProfile for a lower-resolution instrument (500-2400nm, 2nm spacing)."""
    wavelengths = np.arange(500.0, 2401.0, 2.0)
    return InstrumentProfile(
        instrument_id="InstrumentB",
        wavelengths=wavelengths,
        delta_lambda_med=2.0,
    )


@pytest.fixture
def instrument_c_profile():
    """InstrumentProfile with a narrow range (1000-1500nm, 5nm spacing)."""
    wavelengths = np.arange(1000.0, 1501.0, 5.0)
    return InstrumentProfile(
        instrument_id="InstrumentC",
        wavelengths=wavelengths,
        delta_lambda_med=5.0,
    )


@pytest.fixture
def profiles_ab(instrument_a_profile, instrument_b_profile):
    """Dictionary of two instrument profiles (A and B)."""
    return {
        "InstrumentA": instrument_a_profile,
        "InstrumentB": instrument_b_profile,
    }


@pytest.fixture
def profiles_abc(instrument_a_profile, instrument_b_profile, instrument_c_profile):
    """Dictionary of three instrument profiles (A, B, and C)."""
    return {
        "InstrumentA": instrument_a_profile,
        "InstrumentB": instrument_b_profile,
        "InstrumentC": instrument_c_profile,
    }


# =============================================================================
# choose_common_grid tests
# =============================================================================


def test_choose_common_grid_overlapping(profiles_ab):
    """Common grid should cover the overlapping region of two instruments."""
    common = choose_common_grid(profiles_ab, ["InstrumentA", "InstrumentB"])

    # Overlap of [400, 2500] and [500, 2400] is [500, 2400]
    assert common[0] >= 500.0 - 1e-6, "Grid should start at max of minima"
    assert common[-1] <= 2400.0 + 2.0 + 1e-6, "Grid should end near min of maxima"


def test_choose_common_grid_uses_coarsest_spacing(profiles_ab):
    """Common grid should use the coarsest spacing (largest delta_lambda_med)."""
    common = choose_common_grid(profiles_ab, ["InstrumentA", "InstrumentB"])

    # InstrumentB has delta_lambda_med=2.0 (coarsest)
    spacing = np.diff(common)
    median_spacing = np.median(spacing)
    assert abs(median_spacing - 2.0) < 0.1, (
        f"Median spacing should be ~2.0 (coarsest), got {median_spacing}"
    )


def test_choose_common_grid_three_instruments(profiles_abc):
    """Common grid for three instruments should cover the tightest overlap."""
    common = choose_common_grid(profiles_abc, ["InstrumentA", "InstrumentB", "InstrumentC"])

    # Overlap of [400,2500], [500,2400], [1000,1500] is [1000, 1500]
    assert common[0] >= 1000.0 - 1e-6
    assert common[-1] <= 1500.0 + 5.0 + 1e-6

    # Coarsest spacing is 5.0 (InstrumentC)
    spacing = np.diff(common)
    assert abs(np.median(spacing) - 5.0) < 0.5


def test_choose_common_grid_returns_1d_array(profiles_ab):
    """Result should be a 1D numpy array."""
    common = choose_common_grid(profiles_ab, ["InstrumentA", "InstrumentB"])

    assert isinstance(common, np.ndarray)
    assert common.ndim == 1


def test_choose_common_grid_monotonically_increasing(profiles_ab):
    """Common grid wavelengths should be monotonically increasing."""
    common = choose_common_grid(profiles_ab, ["InstrumentA", "InstrumentB"])

    assert np.all(np.diff(common) > 0), "Wavelengths should be monotonically increasing"


# =============================================================================
# build_equalization_mapping_for_instrument tests
# =============================================================================


def test_mapping_returns_callable(instrument_a_profile):
    """build_equalization_mapping_for_instrument should return a callable."""
    common_wl = np.arange(500.0, 2401.0, 2.0)
    mapping = build_equalization_mapping_for_instrument(
        instrument_profile=instrument_a_profile,
        wavelengths_common=common_wl,
    )

    assert callable(mapping)


def test_mapping_produces_correct_shape(instrument_a_profile):
    """Mapping should produce output with n_samples x len(common_wl)."""
    common_wl = np.arange(500.0, 2401.0, 2.0)
    mapping = build_equalization_mapping_for_instrument(
        instrument_profile=instrument_a_profile,
        wavelengths_common=common_wl,
    )

    n_samples = 10
    X = np.random.randn(n_samples, len(instrument_a_profile.wavelengths))
    X_common = mapping(X, instrument_a_profile.wavelengths)

    assert X_common.shape == (n_samples, len(common_wl))


# =============================================================================
# equalize_dataset tests
# =============================================================================


def test_equalize_dataset_output_shape(profiles_ab):
    """Equalized output should have correct shape: total samples x common wavelengths."""
    wl_a = profiles_ab["InstrumentA"].wavelengths
    wl_b = profiles_ab["InstrumentB"].wavelengths

    n_a, n_b = 5, 7
    X_a = np.random.randn(n_a, len(wl_a))
    X_b = np.random.randn(n_b, len(wl_b))

    spectra_by_instrument = {
        "InstrumentA": (wl_a, X_a),
        "InstrumentB": (wl_b, X_b),
    }

    wl_common, X_common = equalize_dataset(spectra_by_instrument, profiles_ab)

    assert X_common.shape[0] == n_a + n_b, "Total samples should be sum of all instruments"
    assert X_common.shape[1] == len(wl_common), "Columns should match common wavelength grid"


def test_equalize_dataset_common_wavelengths_returned(profiles_ab):
    """equalize_dataset should return a valid common wavelength array."""
    wl_a = profiles_ab["InstrumentA"].wavelengths
    wl_b = profiles_ab["InstrumentB"].wavelengths

    X_a = np.random.randn(3, len(wl_a))
    X_b = np.random.randn(4, len(wl_b))

    spectra_by_instrument = {
        "InstrumentA": (wl_a, X_a),
        "InstrumentB": (wl_b, X_b),
    }

    wl_common, X_common = equalize_dataset(spectra_by_instrument, profiles_ab)

    assert isinstance(wl_common, np.ndarray)
    assert wl_common.ndim == 1
    assert len(wl_common) > 0
    assert np.all(np.diff(wl_common) > 0), "Common wavelengths should be increasing"
