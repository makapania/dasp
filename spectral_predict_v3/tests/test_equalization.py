"""
Comprehensive tests for spectral equalization.

Tests:
- Wavelength interpolation between instruments
- Intensity normalization
- Combined equalization pipeline
- Common grid selection
- Multiple instrument equalization
"""

import pytest
import numpy as np

from spectral_predict_v3.core.equalization import (
    choose_common_grid,
    build_equalization_mapping_for_instrument,
    equalize_dataset,
)
from spectral_predict_v3.core.instrument_profiles import (
    InstrumentProfile,
    characterize_instrument,
)
from spectral_predict_v3.core.calibration_transfer import (
    TransferModel,
    estimate_ds,
)


@pytest.fixture
def multi_instrument_data():
    """Generate synthetic data from multiple instruments."""
    np.random.seed(42)

    # Instrument 1: High resolution, 1000-2500 nm
    wavelengths_1 = np.linspace(1000, 2500, 300)
    n_samples_1 = 30
    X_1 = np.zeros((n_samples_1, len(wavelengths_1)))

    for i in range(n_samples_1):
        center = np.random.uniform(1200, 2300)
        width = np.random.uniform(50, 150)
        amplitude = np.random.uniform(0.5, 2.0)
        peak = amplitude * np.exp(-0.5 * ((wavelengths_1 - center) / width) ** 2)
        X_1[i, :] = peak + 0.2 + np.random.normal(0, 0.01, len(wavelengths_1))

    # Instrument 2: Medium resolution, 1100-2400 nm
    wavelengths_2 = np.linspace(1100, 2400, 200)
    n_samples_2 = 25
    X_2 = np.zeros((n_samples_2, len(wavelengths_2)))

    for i in range(n_samples_2):
        center = np.random.uniform(1200, 2300)
        width = np.random.uniform(50, 150)
        amplitude = np.random.uniform(0.5, 2.0)
        peak = amplitude * np.exp(-0.5 * ((wavelengths_2 - center) / width) ** 2)
        X_2[i, :] = peak + 0.3 + np.random.normal(0, 0.01, len(wavelengths_2))

    # Instrument 3: Low resolution, 1050-2450 nm
    wavelengths_3 = np.linspace(1050, 2450, 150)
    n_samples_3 = 20
    X_3 = np.zeros((n_samples_3, len(wavelengths_3)))

    for i in range(n_samples_3):
        center = np.random.uniform(1200, 2300)
        width = np.random.uniform(50, 150)
        amplitude = np.random.uniform(0.5, 2.0)
        peak = amplitude * np.exp(-0.5 * ((wavelengths_3 - center) / width) ** 2)
        X_3[i, :] = peak + 0.25 + np.random.normal(0, 0.01, len(wavelengths_3))

    return {
        "inst1": (wavelengths_1, X_1),
        "inst2": (wavelengths_2, X_2),
        "inst3": (wavelengths_3, X_3),
    }


@pytest.fixture
def instrument_profiles(multi_instrument_data):
    """Create instrument profiles from multi-instrument data."""
    profiles = {}

    for inst_id, (wavelengths, X) in multi_instrument_data.items():
        profiles[inst_id] = characterize_instrument(
            instrument_id=inst_id,
            wavelengths=wavelengths,
            spectra=X
        )

    return profiles


class TestCommonGrid:
    """Tests for common grid selection."""

    def test_choose_common_grid(self, instrument_profiles):
        """Test common grid selection from multiple instruments."""
        instrument_ids = list(instrument_profiles.keys())

        common_wl = choose_common_grid(instrument_profiles, instrument_ids)

        assert isinstance(common_wl, np.ndarray)
        assert len(common_wl) > 0

        # Should be within overlapping range
        min_wl_all = max(profile.wavelengths.min() for profile in instrument_profiles.values())
        max_wl_all = min(profile.wavelengths.max() for profile in instrument_profiles.values())

        assert common_wl.min() >= min_wl_all
        assert common_wl.max() <= max_wl_all

    def test_common_grid_spacing(self, instrument_profiles):
        """Test common grid uses coarsest spacing."""
        instrument_ids = list(instrument_profiles.keys())

        common_wl = choose_common_grid(instrument_profiles, instrument_ids)

        # Spacing should be close to coarsest instrument spacing
        coarsest_spacing = max(profile.delta_lambda_med for profile in instrument_profiles.values())

        actual_spacing = np.median(np.diff(common_wl))

        assert abs(actual_spacing - coarsest_spacing) < 0.1


class TestEqualizationMapping:
    """Tests for equalization mapping construction."""

    def test_build_mapping_basic(self, instrument_profiles):
        """Test basic mapping construction."""
        profile = instrument_profiles["inst1"]
        common_wl = np.linspace(1100, 2400, 200)

        mapping_func = build_equalization_mapping_for_instrument(
            instrument_profile=profile,
            wavelengths_common=common_wl
        )

        assert callable(mapping_func)

        # Test mapping
        X_test = np.random.randn(5, len(profile.wavelengths))
        X_mapped = mapping_func(X_test, profile.wavelengths)

        assert X_mapped.shape == (5, len(common_wl))

    def test_mapping_with_transfer_model(self, instrument_profiles):
        """Test mapping with calibration transfer model."""
        np.random.seed(42)

        # Create synthetic master/slave for transfer model
        common_wl = np.linspace(1100, 2400, 200)

        n_samples = 20
        X_master = np.random.randn(n_samples, len(common_wl))
        X_slave = 0.95 * X_master + 0.1

        A = estimate_ds(X_master, X_slave)

        transfer_model = TransferModel(
            master_id="master",
            slave_id="inst1",
            method="ds",
            wavelengths_common=common_wl,
            params={'A': A}
        )

        profile = instrument_profiles["inst1"]

        mapping_func = build_equalization_mapping_for_instrument(
            instrument_profile=profile,
            wavelengths_common=common_wl,
            transfer_model=transfer_model
        )

        X_test = np.random.randn(5, len(profile.wavelengths))
        X_mapped = mapping_func(X_test, profile.wavelengths)

        assert X_mapped.shape == (5, len(common_wl))


class TestDatasetEqualization:
    """Tests for full dataset equalization."""

    def test_equalize_dataset_basic(self, multi_instrument_data, instrument_profiles):
        """Test equalizing dataset from multiple instruments."""
        wavelengths_common, X_common = equalize_dataset(
            multi_instrument_data,
            instrument_profiles
        )

        assert isinstance(wavelengths_common, np.ndarray)
        assert isinstance(X_common, np.ndarray)

        # Should have combined all samples
        total_samples = sum(X.shape[0] for _, X in multi_instrument_data.values())
        assert X_common.shape[0] == total_samples

        # Should be on common grid
        assert X_common.shape[1] == len(wavelengths_common)

    def test_equalized_wavelength_range(self, multi_instrument_data, instrument_profiles):
        """Test equalized wavelengths are in valid range."""
        wavelengths_common, X_common = equalize_dataset(
            multi_instrument_data,
            instrument_profiles
        )

        # Find overlapping range
        all_min = max(wl.min() for wl, _ in multi_instrument_data.values())
        all_max = min(wl.max() for wl, _ in multi_instrument_data.values())

        assert wavelengths_common.min() >= all_min
        assert wavelengths_common.max() <= all_max

    def test_equalized_data_quality(self, multi_instrument_data, instrument_profiles):
        """Test equalized data has no NaN or inf values."""
        wavelengths_common, X_common = equalize_dataset(
            multi_instrument_data,
            instrument_profiles
        )

        assert not np.any(np.isnan(X_common))
        assert not np.any(np.isinf(X_common))


class TestWavelengthInterpolation:
    """Tests for wavelength interpolation."""

    def test_interpolation_preserves_features(self):
        """Test that interpolation preserves spectral features."""
        # Create spectrum with known peak
        wl_src = np.linspace(1000, 2000, 200)
        wl_target = np.linspace(1000, 2000, 150)

        center = 1500
        width = 50
        peak = np.exp(-0.5 * ((wl_src - center) / width) ** 2)

        X = peak.reshape(1, -1)

        # Create temporary profile for mapping
        from spectral_predict_v3.core.calibration_transfer import resample_to_grid

        X_interp = resample_to_grid(X, wl_src, wl_target)

        # Peak should still be near center
        peak_idx_src = np.argmax(peak)
        peak_idx_target = np.argmax(X_interp[0, :])

        # Wavelengths of peaks should be close
        assert abs(wl_src[peak_idx_src] - wl_target[peak_idx_target]) < 20

    def test_extrapolation_handling(self):
        """Test extrapolation at edges."""
        from spectral_predict_v3.core.calibration_transfer import resample_to_grid

        wl_src = np.linspace(1000, 2000, 100)
        wl_target = np.linspace(900, 2100, 120)  # Extends beyond source

        X = np.random.randn(5, len(wl_src))

        X_resampled = resample_to_grid(X, wl_src, wl_target)

        assert X_resampled.shape == (5, len(wl_target))
        assert not np.any(np.isnan(X_resampled))


class TestIntensityNormalization:
    """Tests for intensity normalization during equalization."""

    def test_normalization_effect(self, multi_instrument_data, instrument_profiles):
        """Test that equalization normalizes intensity differences."""
        # Get individual instrument data
        inst1_wl, inst1_X = multi_instrument_data["inst1"]
        inst2_wl, inst2_X = multi_instrument_data["inst2"]

        # Equalize
        wavelengths_common, X_common = equalize_dataset(
            multi_instrument_data,
            instrument_profiles
        )

        # Extract equalized data for each instrument
        n1 = inst1_X.shape[0]
        X_inst1_equalized = X_common[:n1]
        X_inst2_equalized = X_common[n1:n1 + inst2_X.shape[0]]

        # Mean intensities should be more similar after equalization
        # (though not identical since they're different samples)
        mean1 = np.mean(X_inst1_equalized)
        mean2 = np.mean(X_inst2_equalized)

        # Should be within reasonable range
        assert abs(mean1 - mean2) < 1.0


class TestEdgeCases:
    """Tests for edge cases in equalization."""

    def test_single_instrument(self):
        """Test equalization with single instrument."""
        np.random.seed(42)

        wavelengths = np.linspace(1000, 2000, 100)
        X = np.random.randn(20, 100)

        profile = characterize_instrument("inst1", wavelengths, X)

        spectra_by_instrument = {
            "inst1": (wavelengths, X)
        }

        profiles = {"inst1": profile}

        wavelengths_common, X_common = equalize_dataset(
            spectra_by_instrument,
            profiles
        )

        # Should still work
        assert X_common.shape[0] == X.shape[0]

    def test_non_overlapping_wavelengths(self):
        """Test with minimal wavelength overlap."""
        np.random.seed(42)

        # Barely overlapping instruments
        wl1 = np.linspace(1000, 1500, 100)
        wl2 = np.linspace(1400, 2000, 100)

        X1 = np.random.randn(10, 100)
        X2 = np.random.randn(10, 100)

        profile1 = characterize_instrument("inst1", wl1, X1)
        profile2 = characterize_instrument("inst2", wl2, X2)

        spectra_by_instrument = {
            "inst1": (wl1, X1),
            "inst2": (wl2, X2)
        }

        profiles = {
            "inst1": profile1,
            "inst2": profile2
        }

        wavelengths_common, X_common = equalize_dataset(
            spectra_by_instrument,
            profiles
        )

        # Common grid should be in overlapping region (1400-1500)
        assert wavelengths_common.min() >= 1400
        assert wavelengths_common.max() <= 1500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
