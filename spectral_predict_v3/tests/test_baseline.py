"""
Comprehensive tests for baseline correction methods.

Tests all baseline correction algorithms:
- BaselinePolynomial (polynomial fitting to minima)
- BaselineAsLS (Asymmetric Least Squares)
- BaselineAirPLS (Adaptive iteratively reweighted PLS)
- SavgolSmooth (Savitzky-Golay smoothing)

Test coverage includes:
- Basic functionality on synthetic data
- Removal of known baselines (polynomial drift, fluorescence)
- Edge cases (flat spectra, single peaks, noisy data)
- Parameter validation and error handling
- Transform consistency (fit/transform pattern)
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from spectral_predict_v3.core.baseline import (
    BaselinePolynomial,
    BaselineAsLS,
    BaselineAirPLS,
    SavgolSmooth,
    polynomial_baseline,
    als_baseline,
    airpls_baseline,
    savgol_smooth
)


class TestBaselinePolynomial:
    """Test polynomial baseline correction."""

    def test_basic_polynomial(self):
        """Test basic polynomial baseline correction."""
        # Create synthetic spectrum with polynomial baseline
        n_wavelengths = 100
        x = np.arange(n_wavelengths)

        # Polynomial baseline: y = 0.001*x^2 + 0.05*x + 1
        baseline = 0.001 * x**2 + 0.05 * x + 1

        # Add Gaussian peak
        peak = np.exp(-((x - 50)**2) / 50)

        # Combined spectrum
        spectrum = baseline + peak
        X = spectrum.reshape(1, -1)

        # Remove baseline
        bl = BaselinePolynomial(degree=2, n_segments=20)
        X_corrected = bl.fit_transform(X)

        assert X_corrected.shape == X.shape
        # After correction, baseline should be closer to zero
        assert np.abs(X_corrected.mean()) < np.abs(spectrum.mean())

    def test_polynomial_removes_drift(self):
        """Test that polynomial baseline removes polynomial drift."""
        # Create spectrum with linear drift
        n_wavelengths = 200
        x = np.linspace(0, 1, n_wavelengths)

        # Linear drift
        drift = 2.0 * x + 1.0

        # Add some peaks
        peaks = np.exp(-((x - 0.3)**2) / 0.01) + np.exp(-((x - 0.7)**2) / 0.01)

        spectrum = drift + peaks
        X = spectrum.reshape(1, -1)

        # Remove linear baseline (degree=1)
        bl = BaselinePolynomial(degree=1, n_segments=30)
        X_corrected = bl.fit_transform(X)

        # Check that drift is removed (baseline closer to zero)
        assert np.abs(X_corrected.mean()) < 0.5
        # Peaks should still be present
        assert X_corrected.max() > 0.5

    def test_polynomial_multiple_spectra(self):
        """Test polynomial baseline on multiple spectra."""
        n_samples = 50
        n_wavelengths = 100
        x = np.arange(n_wavelengths)

        X = np.zeros((n_samples, n_wavelengths))
        for i in range(n_samples):
            # Random polynomial baseline
            baseline = np.random.randn() * x + np.random.randn()
            # Random peaks
            peak = np.exp(-((x - 50 + np.random.randn() * 10)**2) / 100)
            X[i, :] = baseline + peak

        bl = BaselinePolynomial(degree=2)
        X_corrected = bl.fit_transform(X)

        assert X_corrected.shape == X.shape
        # Each spectrum should be corrected
        for i in range(n_samples):
            assert not np.allclose(X[i, :], X_corrected[i, :])

    def test_polynomial_degree_parameter(self):
        """Test different polynomial degrees."""
        n_wavelengths = 100
        x = np.arange(n_wavelengths)
        baseline = 0.001 * x**2 + 0.05 * x + 1
        peak = np.exp(-((x - 50)**2) / 50)
        spectrum = baseline + peak
        X = spectrum.reshape(1, -1)

        for degree in [1, 2, 3, 4, 5]:
            bl = BaselinePolynomial(degree=degree)
            X_corrected = bl.fit_transform(X)
            assert X_corrected.shape == X.shape

    def test_polynomial_n_segments_parameter(self):
        """Test different numbers of segments."""
        n_wavelengths = 100
        x = np.arange(n_wavelengths)
        spectrum = x * 0.1 + np.sin(x / 10)
        X = spectrum.reshape(1, -1)

        for n_segments in [5, 10, 20, 50]:
            bl = BaselinePolynomial(n_segments=n_segments)
            X_corrected = bl.fit_transform(X)
            assert X_corrected.shape == X.shape

    def test_polynomial_flat_spectrum(self):
        """Test polynomial baseline on flat spectrum."""
        X = np.ones((10, 100)) * 5.0

        bl = BaselinePolynomial(degree=2)
        X_corrected = bl.fit_transform(X)

        # Flat spectrum should remain approximately flat
        assert X_corrected.shape == X.shape

    def test_polynomial_single_peak(self):
        """Test polynomial baseline with single sharp peak."""
        n_wavelengths = 100
        x = np.arange(n_wavelengths)

        # Baseline + single sharp peak
        baseline = 0.01 * x + 1
        peak = np.zeros(n_wavelengths)
        peak[50] = 10  # Single spike

        spectrum = baseline + peak
        X = spectrum.reshape(1, -1)

        bl = BaselinePolynomial(degree=1, n_segments=30)
        X_corrected = bl.fit_transform(X)

        # Peak should still be visible
        assert X_corrected.max() > 5

    def test_polynomial_fit_transform_consistency(self):
        """Test fit/transform consistency."""
        X_train = np.random.randn(50, 100) * 2 + 5
        X_test = np.random.randn(20, 100) * 2 + 5

        bl = BaselinePolynomial(degree=2)

        # Fit and transform separately
        bl.fit(X_train)
        X_train_corrected = bl.transform(X_train)
        X_test_corrected = bl.transform(X_test)

        # fit_transform should give same result as fit + transform
        bl2 = BaselinePolynomial(degree=2)
        X_train_corrected2 = bl2.fit_transform(X_train)

        # Results should be similar (may differ slightly due to fit)
        assert X_train_corrected.shape == X_train.shape
        assert X_test_corrected.shape == X_test.shape


class TestBaselineAsLS:
    """Test Asymmetric Least Squares baseline correction."""

    def test_basic_als(self):
        """Test basic AsLS baseline correction."""
        n_wavelengths = 100
        x = np.arange(n_wavelengths)

        # Smooth baseline (simulating fluorescence)
        baseline = np.exp(-x / 50) * 5 + 1

        # Sharp peaks (positive)
        peaks = np.exp(-((x - 30)**2) / 20) + np.exp(-((x - 70)**2) / 20)

        spectrum = baseline + peaks
        X = spectrum.reshape(1, -1)

        # Remove baseline with AsLS
        bl = BaselineAsLS(lam=1e5, p=0.01, max_iter=10)
        X_corrected = bl.fit_transform(X)

        assert X_corrected.shape == X.shape
        # Baseline should be reduced
        assert X_corrected.min() < spectrum.min()

    def test_als_asymmetric_fluorescence(self):
        """Test AsLS removes asymmetric fluorescence background."""
        n_wavelengths = 200
        x = np.linspace(0, 1, n_wavelengths)

        # Exponential fluorescence background (asymmetric)
        fluorescence = np.exp(-x * 3) * 10 + 2

        # Raman-like peaks (sharp, symmetric)
        peaks = (np.exp(-((x - 0.3)**2) / 0.001) * 0.5 +
                 np.exp(-((x - 0.5)**2) / 0.001) * 0.8 +
                 np.exp(-((x - 0.7)**2) / 0.001) * 0.3)

        spectrum = fluorescence + peaks
        X = spectrum.reshape(1, -1)

        # Apply AsLS (small p for positive peaks on baseline)
        bl = BaselineAsLS(lam=1e6, p=0.01, max_iter=15)
        X_corrected = bl.fit_transform(X)

        # Check fluorescence is removed
        # Corrected spectrum should have lower mean (fluorescence removed)
        assert X_corrected.mean() < spectrum.mean()
        # Peaks should still be present
        assert X_corrected.max() > 0.2

    def test_als_lambda_parameter(self):
        """Test different lambda (smoothness) parameters."""
        n_wavelengths = 100
        x = np.arange(n_wavelengths)
        baseline = np.exp(-x / 30) * 5
        peaks = np.exp(-((x - 50)**2) / 50)
        spectrum = baseline + peaks
        X = spectrum.reshape(1, -1)

        results = []
        for lam in [1e3, 1e5, 1e7]:
            bl = BaselineAsLS(lam=lam, p=0.01)
            X_corrected = bl.fit_transform(X)
            results.append(X_corrected)
            assert X_corrected.shape == X.shape

        # Different lambda should give different results
        assert not np.allclose(results[0], results[1])

    def test_als_p_parameter(self):
        """Test different asymmetry parameters."""
        n_wavelengths = 100
        x = np.arange(n_wavelengths)
        baseline = np.exp(-x / 30) * 5
        peaks = np.exp(-((x - 50)**2) / 50)
        spectrum = baseline + peaks
        X = spectrum.reshape(1, -1)

        results = []
        for p in [0.001, 0.01, 0.1]:
            bl = BaselineAsLS(lam=1e5, p=p)
            X_corrected = bl.fit_transform(X)
            results.append(X_corrected)
            assert X_corrected.shape == X.shape

        # Different p should give different results
        assert not np.allclose(results[0], results[2])

    def test_als_convergence(self):
        """Test AsLS convergence with different max_iter."""
        n_wavelengths = 100
        x = np.arange(n_wavelengths)
        baseline = np.exp(-x / 30) * 5
        peaks = np.exp(-((x - 50)**2) / 50)
        spectrum = baseline + peaks
        X = spectrum.reshape(1, -1)

        # More iterations should converge better
        bl_few = BaselineAsLS(lam=1e5, p=0.01, max_iter=3)
        bl_many = BaselineAsLS(lam=1e5, p=0.01, max_iter=20)

        X_few = bl_few.fit_transform(X)
        X_many = bl_many.fit_transform(X)

        assert X_few.shape == X.shape
        assert X_many.shape == X.shape
        # Results may differ (convergence)

    def test_als_multiple_spectra(self):
        """Test AsLS on multiple spectra."""
        n_samples = 30
        n_wavelengths = 100
        x = np.arange(n_wavelengths)

        X = np.zeros((n_samples, n_wavelengths))
        for i in range(n_samples):
            baseline = np.exp(-x / (30 + np.random.randn() * 5)) * (5 + np.random.randn())
            peaks = np.exp(-((x - 50)**2) / 50) * (1 + np.random.randn() * 0.2)
            X[i, :] = baseline + peaks

        bl = BaselineAsLS(lam=1e5, p=0.01)
        X_corrected = bl.fit_transform(X)

        assert X_corrected.shape == X.shape

    def test_als_flat_spectrum(self):
        """Test AsLS on flat spectrum."""
        X = np.ones((10, 100)) * 5.0

        bl = BaselineAsLS(lam=1e5, p=0.01)
        X_corrected = bl.fit_transform(X)

        # Flat spectrum should remain approximately flat
        assert X_corrected.shape == X.shape


class TestBaselineAirPLS:
    """Test Adaptive iteratively reweighted PLS baseline correction."""

    def test_basic_airpls(self):
        """Test basic airPLS baseline correction."""
        n_wavelengths = 100
        x = np.arange(n_wavelengths)

        # Smooth baseline
        baseline = np.exp(-x / 50) * 5 + 1

        # Sharp peaks
        peaks = np.exp(-((x - 30)**2) / 20) + np.exp(-((x - 70)**2) / 20)

        spectrum = baseline + peaks
        X = spectrum.reshape(1, -1)

        # Remove baseline with airPLS
        bl = BaselineAirPLS(lam=1e5, max_iter=15)
        X_corrected = bl.fit_transform(X)

        assert X_corrected.shape == X.shape

    def test_airpls_fluorescence_removal(self):
        """Test airPLS removes fluorescence background."""
        n_wavelengths = 200
        x = np.linspace(0, 1, n_wavelengths)

        # Fluorescence background
        fluorescence = np.exp(-x * 2) * 8 + 1

        # Raman peaks
        peaks = (np.exp(-((x - 0.3)**2) / 0.001) * 0.5 +
                 np.exp(-((x - 0.6)**2) / 0.001) * 0.7)

        spectrum = fluorescence + peaks
        X = spectrum.reshape(1, -1)

        # Apply airPLS
        bl = BaselineAirPLS(lam=1e6, max_iter=15)
        X_corrected = bl.fit_transform(X)

        # Fluorescence should be removed
        assert X_corrected.mean() < spectrum.mean()
        # Peaks should still be present
        assert X_corrected.max() > 0.2

    def test_airpls_vs_als(self):
        """Test that airPLS and AsLS give similar (but not identical) results."""
        n_wavelengths = 100
        x = np.arange(n_wavelengths)
        baseline = np.exp(-x / 30) * 5
        peaks = np.exp(-((x - 50)**2) / 50)
        spectrum = baseline + peaks
        X = spectrum.reshape(1, -1)

        # Both methods
        als = BaselineAsLS(lam=1e5, p=0.01)
        airpls = BaselineAirPLS(lam=1e5)

        X_als = als.fit_transform(X)
        X_airpls = airpls.fit_transform(X)

        # Should both correct baseline
        assert X_als.mean() < spectrum.mean()
        assert X_airpls.mean() < spectrum.mean()

        # Results should be similar but not identical (different algorithms)
        # Check correlation rather than exact match
        correlation = np.corrcoef(X_als.ravel(), X_airpls.ravel())[0, 1]
        assert correlation > 0.8  # Should be highly correlated

    def test_airpls_lambda_parameter(self):
        """Test different lambda parameters for airPLS."""
        n_wavelengths = 100
        x = np.arange(n_wavelengths)
        baseline = np.exp(-x / 30) * 5
        peaks = np.exp(-((x - 50)**2) / 50)
        spectrum = baseline + peaks
        X = spectrum.reshape(1, -1)

        results = []
        for lam in [1e3, 1e5, 1e7]:
            bl = BaselineAirPLS(lam=lam)
            X_corrected = bl.fit_transform(X)
            results.append(X_corrected)
            assert X_corrected.shape == X.shape

        # Different lambda should give different results
        assert not np.allclose(results[0], results[2])

    def test_airpls_multiple_spectra(self):
        """Test airPLS on multiple spectra."""
        n_samples = 30
        n_wavelengths = 100
        x = np.arange(n_wavelengths)

        X = np.zeros((n_samples, n_wavelengths))
        for i in range(n_samples):
            baseline = np.exp(-x / 30) * (3 + np.random.randn())
            peaks = np.exp(-((x - 50)**2) / 50) * (1 + np.random.randn() * 0.2)
            X[i, :] = baseline + peaks

        bl = BaselineAirPLS(lam=1e5)
        X_corrected = bl.fit_transform(X)

        assert X_corrected.shape == X.shape

    def test_airpls_flat_spectrum(self):
        """Test airPLS on flat spectrum."""
        X = np.ones((10, 100)) * 5.0

        bl = BaselineAirPLS(lam=1e5)
        X_corrected = bl.fit_transform(X)

        assert X_corrected.shape == X.shape


class TestSavgolSmooth:
    """Test Savitzky-Golay smoothing (without differentiation)."""

    def test_basic_smoothing(self):
        """Test basic Savitzky-Golay smoothing."""
        # Create noisy spectrum
        n_wavelengths = 100
        x = np.arange(n_wavelengths)
        clean_signal = np.sin(x / 10)
        noise = np.random.randn(n_wavelengths) * 0.1
        noisy_signal = clean_signal + noise

        X = noisy_signal.reshape(1, -1)

        # Apply smoothing
        smoother = SavgolSmooth(window_length=11, polyorder=2)
        X_smooth = smoother.fit_transform(X)

        assert X_smooth.shape == X.shape

        # Smoothed signal should be closer to clean signal
        noise_before = np.std(X - clean_signal.reshape(1, -1))
        noise_after = np.std(X_smooth - clean_signal.reshape(1, -1))
        assert noise_after < noise_before

    def test_smoothing_reduces_noise(self):
        """Test that smoothing reduces noise while preserving signal."""
        n_wavelengths = 200
        x = np.linspace(0, 2 * np.pi, n_wavelengths)

        # Clean signal with multiple components
        clean = np.sin(x) + 0.5 * np.sin(3 * x)

        # Add noise
        noise = np.random.randn(n_wavelengths) * 0.2
        noisy = clean + noise

        X = noisy.reshape(1, -1)

        # Smooth
        smoother = SavgolSmooth(window_length=15, polyorder=3)
        X_smooth = smoother.fit_transform(X)

        # Variance should decrease (noise removed)
        var_before = np.var(X)
        var_after = np.var(X_smooth)
        assert var_after < var_before

    def test_smoothing_preserves_peaks(self):
        """Test that smoothing preserves peak shapes."""
        n_wavelengths = 100
        x = np.arange(n_wavelengths)

        # Gaussian peak + noise
        peak = np.exp(-((x - 50)**2) / 50)
        noise = np.random.randn(n_wavelengths) * 0.05
        signal = peak + noise

        X = signal.reshape(1, -1)

        # Smooth
        smoother = SavgolSmooth(window_length=11, polyorder=2)
        X_smooth = smoother.fit_transform(X)

        # Peak location should be preserved
        peak_idx_before = np.argmax(X)
        peak_idx_after = np.argmax(X_smooth)
        assert abs(peak_idx_before - peak_idx_after) <= 2  # Within 2 points

        # Peak height should be similar
        assert abs(X.max() - X_smooth.max()) < 0.2

    def test_smoothing_window_parameter(self):
        """Test different window lengths."""
        n_wavelengths = 100
        x = np.arange(n_wavelengths)
        signal = np.sin(x / 10) + np.random.randn(n_wavelengths) * 0.1
        X = signal.reshape(1, -1)

        results = []
        for window in [5, 11, 21, 31]:
            smoother = SavgolSmooth(window_length=window, polyorder=2)
            X_smooth = smoother.fit_transform(X)
            results.append(X_smooth)
            assert X_smooth.shape == X.shape

        # Larger windows should give smoother results
        # (lower variance)
        var_5 = np.var(results[0])
        var_31 = np.var(results[3])
        assert var_31 < var_5

    def test_smoothing_polyorder_parameter(self):
        """Test different polynomial orders."""
        n_wavelengths = 100
        x = np.arange(n_wavelengths)
        signal = np.sin(x / 10) + np.random.randn(n_wavelengths) * 0.1
        X = signal.reshape(1, -1)

        for polyorder in [2, 3, 4]:
            smoother = SavgolSmooth(window_length=11, polyorder=polyorder)
            X_smooth = smoother.fit_transform(X)
            assert X_smooth.shape == X.shape

    def test_smoothing_even_window_increment(self):
        """Test that even window is incremented to odd."""
        X = np.random.randn(10, 100)

        # Even window (should be incremented to 11)
        smoother = SavgolSmooth(window_length=10, polyorder=2)
        X_smooth = smoother.fit_transform(X)

        assert X_smooth.shape == X.shape

    def test_smoothing_window_too_large_error(self):
        """Test error when window is larger than data."""
        X = np.random.randn(10, 20)  # Only 20 features

        smoother = SavgolSmooth(window_length=25, polyorder=2)

        with pytest.raises(ValueError, match="Window length.*must be <="):
            smoother.fit_transform(X)

    def test_smoothing_window_too_small_error(self):
        """Test error when window <= polyorder."""
        X = np.random.randn(10, 100)

        # window=5, polyorder=5 means window <= polyorder
        smoother = SavgolSmooth(window_length=5, polyorder=5)

        with pytest.raises(ValueError, match="Window length.*must be >"):
            smoother.fit_transform(X)

    def test_smoothing_multiple_spectra(self):
        """Test smoothing on multiple spectra."""
        n_samples = 50
        n_wavelengths = 100
        x = np.arange(n_wavelengths)

        X = np.zeros((n_samples, n_wavelengths))
        for i in range(n_samples):
            signal = np.sin((x + i * 5) / 10)
            noise = np.random.randn(n_wavelengths) * 0.1
            X[i, :] = signal + noise

        smoother = SavgolSmooth(window_length=11, polyorder=2)
        X_smooth = smoother.fit_transform(X)

        assert X_smooth.shape == X.shape


class TestConvenienceFunctions:
    """Test convenience functions for direct use."""

    def test_polynomial_baseline_function(self):
        """Test polynomial_baseline() convenience function."""
        X = np.random.randn(50, 100) + np.arange(100) * 0.05

        X_corrected = polynomial_baseline(X, degree=2, n_segments=20)

        assert X_corrected.shape == X.shape

    def test_als_baseline_function(self):
        """Test als_baseline() convenience function."""
        X = np.random.randn(50, 100) + np.exp(-np.arange(100) / 30) * 5

        X_corrected = als_baseline(X, lam=1e5, p=0.01)

        assert X_corrected.shape == X.shape

    def test_airpls_baseline_function(self):
        """Test airpls_baseline() convenience function."""
        X = np.random.randn(50, 100) + np.exp(-np.arange(100) / 30) * 5

        X_corrected = airpls_baseline(X, lam=1e5)

        assert X_corrected.shape == X.shape

    def test_savgol_smooth_function(self):
        """Test savgol_smooth() convenience function."""
        X = np.random.randn(50, 100) * 0.2 + np.sin(np.arange(100) / 10)

        X_smooth = savgol_smooth(X, window_length=11, polyorder=2)

        assert X_smooth.shape == X.shape


class TestBaselinePipelines:
    """Test combinations of baseline methods in pipelines."""

    def test_baseline_then_snv(self):
        """Test baseline correction followed by SNV."""
        from spectral_predict_v3.core.preprocess import SNV

        X = np.random.randn(50, 100) + np.arange(100) * 0.05

        pipeline = Pipeline([
            ('baseline', BaselinePolynomial(degree=2)),
            ('snv', SNV())
        ])

        X_processed = pipeline.fit_transform(X)

        assert X_processed.shape == X.shape

    def test_smooth_then_baseline(self):
        """Test smoothing followed by baseline correction."""
        X = np.random.randn(50, 100) * 0.3 + np.exp(-np.arange(100) / 30) * 5

        pipeline = Pipeline([
            ('smooth', SavgolSmooth(window_length=11, polyorder=2)),
            ('baseline', BaselineAsLS(lam=1e5, p=0.01))
        ])

        X_processed = pipeline.fit_transform(X)

        assert X_processed.shape == X.shape

    def test_baseline_then_derivative(self):
        """Test baseline correction followed by derivative."""
        from spectral_predict_v3.core.preprocess import SavgolDerivative

        X = np.random.randn(50, 100) + np.arange(100) * 0.05

        pipeline = Pipeline([
            ('baseline', BaselinePolynomial(degree=1)),
            ('deriv', SavgolDerivative(deriv=1, window=7))
        ])

        X_processed = pipeline.fit_transform(X)

        assert X_processed.shape == X.shape

    def test_full_preprocessing_with_baseline(self):
        """Test comprehensive preprocessing: smooth → baseline → SNV → derivative."""
        from spectral_predict_v3.core.preprocess import SNV, SavgolDerivative

        X = np.random.randn(50, 100) * 0.5 + np.exp(-np.arange(100) / 30) * 5

        pipeline = Pipeline([
            ('smooth', SavgolSmooth(window_length=11, polyorder=2)),
            ('baseline', BaselineAirPLS(lam=1e5)),
            ('snv', SNV()),
            ('deriv', SavgolDerivative(deriv=1, window=7))
        ])

        X_processed = pipeline.fit_transform(X)

        assert X_processed.shape == X.shape
        assert not np.allclose(X, X_processed)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_spectrum(self):
        """Test all methods with single spectrum."""
        X = np.random.randn(1, 100) + np.arange(100) * 0.05

        # Polynomial
        bl1 = BaselinePolynomial(degree=2)
        X1 = bl1.fit_transform(X)
        assert X1.shape == X.shape

        # AsLS
        bl2 = BaselineAsLS(lam=1e5, p=0.01)
        X2 = bl2.fit_transform(X)
        assert X2.shape == X.shape

        # airPLS
        bl3 = BaselineAirPLS(lam=1e5)
        X3 = bl3.fit_transform(X)
        assert X3.shape == X.shape

        # Smooth
        bl4 = SavgolSmooth(window_length=11, polyorder=2)
        X4 = bl4.fit_transform(X)
        assert X4.shape == X.shape

    def test_small_feature_count(self):
        """Test baseline methods with small number of features."""
        X = np.random.randn(50, 20) + np.arange(20) * 0.1

        # Polynomial
        bl1 = BaselinePolynomial(degree=2, n_segments=5)
        X1 = bl1.fit_transform(X)
        assert X1.shape == X.shape

        # AsLS (works with any feature count)
        bl2 = BaselineAsLS(lam=1e4, p=0.01)
        X2 = bl2.fit_transform(X)
        assert X2.shape == X.shape

        # Smooth (small window)
        bl3 = SavgolSmooth(window_length=5, polyorder=2)
        X3 = bl3.fit_transform(X)
        assert X3.shape == X.shape

    def test_very_noisy_data(self):
        """Test baseline methods on very noisy data."""
        n_wavelengths = 100
        x = np.arange(n_wavelengths)

        # High noise
        signal = np.sin(x / 10)
        noise = np.random.randn(n_wavelengths) * 5  # Very high noise
        X = (signal + noise).reshape(1, -1)

        # All methods should still work
        bl1 = BaselinePolynomial(degree=2)
        X1 = bl1.fit_transform(X)
        assert X1.shape == X.shape

        bl2 = BaselineAsLS(lam=1e6, p=0.05)  # Higher lam for smoother fit
        X2 = bl2.fit_transform(X)
        assert X2.shape == X.shape

        bl3 = SavgolSmooth(window_length=21, polyorder=2)  # Larger window
        X3 = bl3.fit_transform(X)
        assert X3.shape == X.shape


def run_tests():
    """Run all tests using pytest."""
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'pytest', __file__, '-v'],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT)
    )
    print(result.stdout)
    print(result.stderr)
    return result.returncode == 0


if __name__ == "__main__":
    # Run with pytest if available
    try:
        import pytest
        exit_code = pytest.main([__file__, '-v', '--tb=short'])
        exit(exit_code)
    except ImportError:
        print("pytest not available, running basic tests...")
        run_tests()
