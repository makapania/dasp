"""
Comprehensive tests for instrument profiling and characterization.

Tests:
- Profile creation from spectra
- Wavelength spacing computation
- Roughness metrics
- Peak detection and FWHM
- Interpolation detection
- Profile comparison
- Save/load functionality
"""

import pytest
import numpy as np
from pathlib import Path

from spectral_predict_v3.core.instrument_profiles import (
    InstrumentProfile,
    compute_wavelength_spacing,
    compute_roughness,
    detect_interpolation,
    compute_peak_fwhm,
    analyze_peaks,
    characterize_instrument,
    save_instrument_profiles,
    load_instrument_profiles,
    rank_instruments_by_detail,
    estimate_smoothing_between_instruments,
)


@pytest.fixture
def synthetic_spectra():
    """Generate synthetic spectral data for testing."""
    np.random.seed(42)

    n_samples = 30
    n_wavelengths = 150
    wavelengths = np.linspace(1000, 2500, n_wavelengths)

    # Generate spectra with multiple Gaussian peaks
    X = np.zeros((n_samples, n_wavelengths))

    for i in range(n_samples):
        # Add 2-4 peaks
        n_peaks = np.random.randint(2, 5)

        for _ in range(n_peaks):
            center = np.random.uniform(1000, 2500)
            width = np.random.uniform(50, 150)
            amplitude = np.random.uniform(0.5, 2.0)

            peak = amplitude * np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
            X[i, :] += peak

        # Add baseline
        X[i, :] += np.random.uniform(0.1, 0.3)

        # Add noise
        X[i, :] += np.random.normal(0, 0.02, n_wavelengths)

    return wavelengths, X


@pytest.fixture
def interpolated_wavelengths():
    """Generate uniformly interpolated wavelengths."""
    return np.linspace(1000, 2500, 200)


@pytest.fixture
def non_uniform_wavelengths():
    """Generate non-uniform wavelengths."""
    # Simulate variable spacing (e.g., from spectrometer)
    base = np.linspace(1000, 2500, 150)
    jitter = np.random.uniform(-0.5, 0.5, 150)
    jitter[0] = 0  # Keep endpoints fixed
    jitter[-1] = 0
    return base + jitter


class TestWavelengthSpacing:
    """Tests for wavelength spacing computation."""

    def test_uniform_spacing(self, interpolated_wavelengths):
        """Test spacing computation for uniform grid."""
        stats = compute_wavelength_spacing(interpolated_wavelengths)

        assert 'delta_lambda_med' in stats
        assert 'delta_lambda_min' in stats
        assert 'delta_lambda_max' in stats

        # Should be nearly constant
        assert abs(stats['delta_lambda_max'] - stats['delta_lambda_min']) < 0.01

    def test_non_uniform_spacing(self, non_uniform_wavelengths):
        """Test spacing computation for non-uniform grid."""
        stats = compute_wavelength_spacing(non_uniform_wavelengths)

        # Should have some variation
        assert stats['delta_lambda_max'] > stats['delta_lambda_min']
        assert stats['delta_lambda_med'] > 0


class TestRoughness:
    """Tests for spectral roughness metrics."""

    def test_roughness_smooth_spectra(self):
        """Test roughness on smooth spectra."""
        wavelengths = np.linspace(1000, 2500, 100)
        X = np.ones((10, 100))  # Flat spectra

        roughness = compute_roughness(X, wavelengths)

        # Flat spectra should have very low roughness
        assert roughness < 0.01

    def test_roughness_noisy_spectra(self):
        """Test roughness on noisy spectra."""
        np.random.seed(42)
        wavelengths = np.linspace(1000, 2500, 100)
        X = np.random.randn(10, 100)  # Noisy spectra

        roughness = compute_roughness(X, wavelengths)

        # Noisy spectra should have higher roughness
        assert roughness > 0.1


class TestInterpolationDetection:
    """Tests for interpolation detection."""

    def test_detect_uniform_interpolation(self, interpolated_wavelengths):
        """Test detection of uniformly interpolated wavelengths."""
        is_interp = detect_interpolation(interpolated_wavelengths)

        assert is_interp is True

    def test_detect_non_uniform(self, non_uniform_wavelengths):
        """Test detection of non-uniform wavelengths."""
        is_interp = detect_interpolation(non_uniform_wavelengths)

        assert is_interp is False

    def test_short_wavelength_array(self):
        """Test with very short wavelength array."""
        short_wl = np.array([1000, 1100])

        is_interp = detect_interpolation(short_wl)

        # Should return False for short arrays
        assert is_interp is False


class TestPeakAnalysis:
    """Tests for peak detection and FWHM."""

    def test_compute_peak_fwhm_gaussian(self):
        """Test FWHM computation for Gaussian peak."""
        wavelengths = np.linspace(0, 100, 200)

        # Create perfect Gaussian with known FWHM
        # FWHM = 2.355 * sigma for Gaussian
        sigma = 5.0
        theoretical_fwhm = 2.355 * sigma

        center = 50
        peak = np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)

        peak_idx = np.argmax(peak)

        fwhm, sharpness = compute_peak_fwhm(peak, wavelengths, peak_idx)

        # Should be close to theoretical FWHM
        assert abs(fwhm - theoretical_fwhm) < 1.0
        assert sharpness > 0

    def test_analyze_peaks(self, synthetic_spectra):
        """Test peak analysis on synthetic spectra."""
        wavelengths, X = synthetic_spectra

        peak_stats = analyze_peaks(X, wavelengths)

        assert 'peak_count' in peak_stats
        assert 'avg_peak_fwhm' in peak_stats
        assert 'avg_peak_sharpness' in peak_stats

        # Should detect some peaks
        assert peak_stats['peak_count'] > 0
        assert peak_stats['avg_peak_fwhm'] > 0


class TestInstrumentProfile:
    """Tests for instrument profile creation and management."""

    def test_characterize_instrument(self, synthetic_spectra):
        """Test instrument characterization."""
        wavelengths, X = synthetic_spectra

        profile = characterize_instrument(
            instrument_id="test_inst",
            wavelengths=wavelengths,
            spectra=X,
            vendor="Test Vendor",
            model="Model X",
            description="Test instrument"
        )

        assert profile.instrument_id == "test_inst"
        assert profile.vendor == "Test Vendor"
        assert profile.model == "Model X"
        assert profile.description == "Test instrument"

        # Check metrics are computed
        assert profile.delta_lambda_med is not None
        assert profile.roughness_R is not None
        assert profile.detail_score is not None
        assert profile.peak_count is not None
        assert profile.avg_peak_fwhm is not None
        assert profile.avg_peak_sharpness is not None
        assert profile.is_interpolated is not None

    def test_profile_wavelengths_copied(self, synthetic_spectra):
        """Test that wavelengths are copied, not referenced."""
        wavelengths, X = synthetic_spectra

        profile = characterize_instrument("test", wavelengths, X)

        # Modify original wavelengths
        wavelengths[0] = 9999

        # Profile wavelengths should be unchanged
        assert profile.wavelengths[0] != 9999


class TestSaveLoadProfiles:
    """Tests for saving and loading profiles."""

    def test_save_load_single_profile(self, synthetic_spectra, tmp_path):
        """Test saving and loading a single profile."""
        wavelengths, X = synthetic_spectra

        profile = characterize_instrument("test_inst", wavelengths, X)

        profiles = {"test_inst": profile}

        # Save
        save_path = tmp_path / "profiles.json"
        save_instrument_profiles(profiles, save_path)

        assert save_path.exists()

        # Load
        loaded_profiles = load_instrument_profiles(save_path)

        assert "test_inst" in loaded_profiles
        loaded = loaded_profiles["test_inst"]

        assert loaded.instrument_id == profile.instrument_id
        assert np.allclose(loaded.wavelengths, profile.wavelengths)
        assert loaded.delta_lambda_med == profile.delta_lambda_med
        assert loaded.roughness_R == profile.roughness_R

    def test_save_load_multiple_profiles(self, synthetic_spectra, tmp_path):
        """Test saving and loading multiple profiles."""
        wavelengths, X = synthetic_spectra

        profile1 = characterize_instrument("inst1", wavelengths, X)
        profile2 = characterize_instrument("inst2", wavelengths, X)

        profiles = {
            "inst1": profile1,
            "inst2": profile2
        }

        save_path = tmp_path / "profiles.json"
        save_instrument_profiles(profiles, save_path)

        loaded_profiles = load_instrument_profiles(save_path)

        assert len(loaded_profiles) == 2
        assert "inst1" in loaded_profiles
        assert "inst2" in loaded_profiles


class TestProfileComparison:
    """Tests for comparing instrument profiles."""

    def test_rank_by_detail(self):
        """Test ranking instruments by detail score."""
        # Create profiles with different detail scores
        profile1 = InstrumentProfile(
            instrument_id="low_detail",
            detail_score=0.5
        )
        profile2 = InstrumentProfile(
            instrument_id="high_detail",
            detail_score=1.5
        )
        profile3 = InstrumentProfile(
            instrument_id="medium_detail",
            detail_score=1.0
        )

        profiles = {
            "low_detail": profile1,
            "high_detail": profile2,
            "medium_detail": profile3
        }

        ranked = rank_instruments_by_detail(profiles)

        # Should be sorted by descending detail score
        assert ranked == ["high_detail", "medium_detail", "low_detail"]

    def test_rank_with_none_detail(self):
        """Test ranking with None detail scores."""
        profile1 = InstrumentProfile(
            instrument_id="has_detail",
            detail_score=1.0
        )
        profile2 = InstrumentProfile(
            instrument_id="no_detail",
            detail_score=None
        )

        profiles = {
            "has_detail": profile1,
            "no_detail": profile2
        }

        ranked = rank_instruments_by_detail(profiles)

        # Profile with detail score should rank higher
        assert ranked[0] == "has_detail"


class TestSmoothingEstimation:
    """Tests for smoothing parameter estimation."""

    def test_estimate_smoothing(self):
        """Test smoothing parameter estimation between instruments."""
        np.random.seed(42)

        # High-resolution instrument
        wavelengths_high = np.linspace(1000, 2500, 300)
        n_samples = 20
        X_high = np.zeros((n_samples, len(wavelengths_high)))

        for i in range(n_samples):
            center = np.random.uniform(1200, 2300)
            width = np.random.uniform(50, 150)
            amplitude = np.random.uniform(0.5, 2.0)

            peak = amplitude * np.exp(-0.5 * ((wavelengths_high - center) / width) ** 2)
            X_high[i, :] = peak + np.random.normal(0, 0.01, len(wavelengths_high))

        # Low-resolution instrument (smoothed version)
        from scipy.ndimage import gaussian_filter1d

        wavelengths_low = np.linspace(1000, 2500, 150)
        true_sigma = 2.0

        X_low_smooth = gaussian_filter1d(X_high, sigma=true_sigma, axis=1)

        # Resample to low-res grid
        from scipy.interpolate import interp1d

        X_low = np.zeros((n_samples, len(wavelengths_low)))
        for i in range(n_samples):
            interpolator = interp1d(wavelengths_high, X_low_smooth[i, :],
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
            X_low[i, :] = interpolator(wavelengths_low)

        # Estimate smoothing
        sigma_candidates = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        best_sigma = estimate_smoothing_between_instruments(
            wavelengths_high, X_high,
            wavelengths_low, X_low,
            sigma_candidates
        )

        # Should be close to true sigma
        assert abs(best_sigma - true_sigma) <= 0.5


class TestDataclass:
    """Tests for InstrumentProfile dataclass."""

    def test_default_values(self):
        """Test dataclass with default values."""
        profile = InstrumentProfile(instrument_id="test")

        assert profile.instrument_id == "test"
        assert profile.vendor is None
        assert profile.model is None
        assert profile.wavelengths is None
        assert profile.extra == {}

    def test_extra_field(self):
        """Test extra metadata field."""
        profile = InstrumentProfile(
            instrument_id="test",
            extra={"custom_key": "custom_value"}
        )

        assert profile.extra["custom_key"] == "custom_value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
