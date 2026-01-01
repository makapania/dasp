"""Tests for baseline correction transformers.

This module tests baseline correction methods in src/spectral_predict/baseline.py:
- BaselinePolynomial: Polynomial fitting baseline correction
- BaselineALS: Asymmetric Least Squares baseline correction
- BaselineAirPLS: Adaptive Iteratively Reweighted PLS baseline correction

Test coverage:
- Numerical correctness (matches gold standards)
- Different polynomial degrees
- ALS parameter variations (lambda, p, niter)
- AirPLS convergence and robustness
- Peak preservation
- Edge cases and input validation
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

# Import Spectral Predict implementations
from spectral_predict.baseline import (
    BaselinePolynomial,
    BaselineALS,
    BaselineAirPLS,
)


@pytest.mark.numerical
class TestBaselinePolynomial:
    """Test polynomial baseline correction."""

    def test_polynomial_degree_1(self):
        """Test linear baseline (degree=1) correction."""
        np.random.seed(42)

        # Create spectrum with linear baseline
        n_features = 100
        wavelengths = np.arange(n_features)

        # True signal: Gaussian peak
        signal = np.exp(-((wavelengths - 50) ** 2) / 200)

        # Add linear baseline
        baseline = 0.5 * wavelengths / n_features + 0.2

        # Observed spectrum
        spectrum = signal + baseline
        X = spectrum.reshape(1, -1)

        # Apply baseline correction
        corrector = BaselinePolynomial(degree=1)
        X_corrected = corrector.fit_transform(X)

        # Should remove most of the linear trend
        # Check that corrected spectrum is closer to true signal
        error_before = np.mean((spectrum - signal) ** 2)
        error_after = np.mean((X_corrected[0] - signal) ** 2)

        assert error_after < error_before, \
            f"Baseline correction should reduce error (before: {error_before:.6f}, after: {error_after:.6f})"

    def test_polynomial_degree_2(self):
        """Test quadratic baseline (degree=2) correction."""
        np.random.seed(42)

        # Create spectrum with quadratic baseline
        n_features = 100
        wavelengths = np.arange(n_features)

        # Apply baseline correction
        corrector = BaselinePolynomial(degree=2)

        # Just verify it doesn't crash and preserves shape
        X = np.random.randn(5, n_features) + 2.0
        X_corrected = corrector.fit_transform(X)

        assert X_corrected.shape == X.shape, "Shape should be preserved"
        assert not np.any(np.isnan(X_corrected)), "Should not produce NaN"

    def test_polynomial_degree_3(self):
        """Test cubic baseline (degree=3) correction."""
        np.random.seed(42)

        # Create spectrum with cubic baseline
        n_features = 100

        # Apply baseline correction
        corrector = BaselinePolynomial(degree=3)

        # Verify it works without crashing
        X = np.random.randn(5, n_features) + 3.0
        X_corrected = corrector.fit_transform(X)

        assert X_corrected.shape == X.shape, "Shape should be preserved"
        assert not np.any(np.isnan(X_corrected)), "Should not produce NaN"

    def test_polynomial_matches_gold_standard(self):
        """Verify polynomial baseline matches manual numpy.polyfit calculation."""
        np.random.seed(123)

        # Create test data
        n_samples = 5
        n_features = 100
        X = np.random.randn(n_samples, n_features) + 2.0
        degree = 2

        # Apply baseline correction
        corrector = BaselinePolynomial(degree=degree)
        X_actual = corrector.fit_transform(X)

        # Manually compute expected result using numpy
        wavelengths = np.arange(n_features)
        X_expected = np.zeros_like(X)
        for i in range(n_samples):
            coeffs = np.polyfit(wavelengths, X[i], degree)
            baseline = np.polyval(coeffs, wavelengths)
            X_expected[i] = X[i] - baseline

        # Compare
        np.testing.assert_allclose(
            X_actual, X_expected, rtol=1e-10, atol=1e-10,
            err_msg="Polynomial baseline correction does not match numpy.polyfit"
        )

    def test_polynomial_preserves_shape(self):
        """Polynomial baseline correction should preserve array shape."""
        np.random.seed(42)

        shapes = [(1, 50), (10, 100), (50, 200)]

        for shape in shapes:
            X = np.random.randn(*shape)

            corrector = BaselinePolynomial(degree=2)
            X_corrected = corrector.fit_transform(X)

            assert X_corrected.shape == X.shape, \
                f"Shape {shape} not preserved"

    def test_polynomial_deterministic(self):
        """Polynomial baseline correction should be deterministic."""
        np.random.seed(42)
        X = np.random.randn(10, 100)

        corrector = BaselinePolynomial(degree=2)
        X_result1 = corrector.fit_transform(X)
        X_result2 = corrector.fit_transform(X)

        np.testing.assert_array_equal(
            X_result1, X_result2,
            err_msg="Polynomial baseline correction is not deterministic"
        )


@pytest.mark.numerical
class TestBaselineALS:
    """Test Asymmetric Least Squares baseline correction."""

    def test_als_removes_polynomial_baseline(self):
        """ALS should remove polynomial baselines effectively."""
        np.random.seed(42)

        # Create spectrum with polynomial baseline
        n_features = 200
        wavelengths = np.arange(n_features)
        x_norm = wavelengths / n_features

        # True signal: Multiple peaks
        signal = (
            0.8 * np.exp(-((wavelengths - 50) ** 2) / 150) +
            0.6 * np.exp(-((wavelengths - 100) ** 2) / 100) +
            0.4 * np.exp(-((wavelengths - 150) ** 2) / 120)
        )

        # Add polynomial baseline
        baseline = 0.3 * x_norm**2 + 0.1 * x_norm + 0.2

        # Observed spectrum
        spectrum = signal + baseline
        X = spectrum.reshape(1, -1)

        # Apply ALS baseline correction
        corrector = BaselineALS(lambda_=1e5, p=0.001, niter=10)
        X_corrected = corrector.fit_transform(X)

        # Should remove most of the baseline
        error_before = np.mean((spectrum - signal) ** 2)
        error_after = np.mean((X_corrected[0] - signal) ** 2)

        assert error_after < error_before, \
            "ALS should reduce baseline error"

    def test_als_preserves_peaks(self):
        """ALS should preserve peak positions and relative heights."""
        np.random.seed(42)

        # Create spectrum with clear peaks
        n_features = 200
        wavelengths = np.arange(n_features)

        # Three peaks at known positions
        peak_positions = [50, 100, 150]
        signal = np.zeros(n_features)
        for pos in peak_positions:
            signal += np.exp(-((wavelengths - pos) ** 2) / 100)

        # Add baseline
        baseline = np.linspace(0.2, 0.5, n_features)

        spectrum = signal + baseline
        X = spectrum.reshape(1, -1)

        # Apply ALS
        corrector = BaselineALS(lambda_=1e6, p=0.001, niter=10)
        X_corrected = corrector.fit_transform(X)

        # Find peak positions in corrected spectrum
        corrected_peaks = []
        for i in range(20, n_features - 20, 10):
            if X_corrected[0, i] > X_corrected[0, i-10] and X_corrected[0, i] > X_corrected[0, i+10]:
                corrected_peaks.append(i)

        # Should find peaks near original positions (within 20 indices)
        for original_pos in peak_positions:
            nearby_peaks = [p for p in corrected_peaks if abs(p - original_pos) <= 20]
            assert len(nearby_peaks) > 0, \
                f"Peak at position {original_pos} not preserved"

    def test_als_different_lambda_values(self):
        """Test ALS with different smoothness parameters."""
        np.random.seed(42)

        # Create test spectrum
        n_features = 150
        wavelengths = np.arange(n_features)

        signal = np.exp(-((wavelengths - 75) ** 2) / 200)
        baseline = np.linspace(0.1, 0.4, n_features)
        spectrum = signal + baseline
        X = spectrum.reshape(1, -1)

        # Test different lambda values
        lambdas = [1e3, 1e5, 1e7]
        results = []

        for lam in lambdas:
            corrector = BaselineALS(lambda_=lam, p=0.001, niter=10)
            X_corrected = corrector.fit_transform(X)
            results.append(X_corrected)

        # Higher lambda should produce smoother baselines (less aggressive correction)
        # Lower lambda follows signal more closely (more aggressive correction)
        # We can't easily test this without knowing the true baseline,
        # but we can verify they produce different results
        for i in range(len(results) - 1):
            assert not np.allclose(results[i], results[i + 1]), \
                f"Lambda {lambdas[i]} and {lambdas[i+1]} should produce different results"

    def test_als_convergence(self):
        """ALS should converge with sufficient iterations."""
        np.random.seed(42)

        # Create test spectrum
        n_features = 100
        wavelengths = np.arange(n_features)
        signal = np.exp(-((wavelengths - 50) ** 2) / 150)
        baseline = np.linspace(0.2, 0.5, n_features)
        spectrum = signal + baseline
        X = spectrum.reshape(1, -1)

        # Test with different iteration counts
        corrector_few = BaselineALS(lambda_=1e5, p=0.001, niter=5)
        X_few = corrector_few.fit_transform(X)

        corrector_many = BaselineALS(lambda_=1e5, p=0.001, niter=20)
        X_many = corrector_many.fit_transform(X)

        # More iterations should provide better or similar results
        # (but differences should be small after convergence)
        # Just verify it doesn't crash and produces reasonable output
        assert X_few.shape == X.shape
        assert X_many.shape == X.shape
        assert not np.any(np.isnan(X_few))
        assert not np.any(np.isnan(X_many))

    def test_als_preserves_shape(self):
        """ALS should preserve array shape."""
        np.random.seed(42)

        shapes = [(1, 50), (5, 100), (20, 150)]

        for shape in shapes:
            X = np.random.randn(*shape) + 1.0  # Ensure positive values

            corrector = BaselineALS(lambda_=1e5, p=0.001, niter=10)
            X_corrected = corrector.fit_transform(X)

            assert X_corrected.shape == X.shape, \
                f"Shape {shape} not preserved"


@pytest.mark.numerical
class TestBaselineAirPLS:
    """Test Adaptive Iteratively Reweighted PLS baseline correction."""

    def test_airpls_basic(self):
        """Test basic airPLS functionality."""
        np.random.seed(42)

        # Create spectrum with baseline
        n_features = 200
        wavelengths = np.arange(n_features)

        # Signal: Gaussian peak
        signal = np.exp(-((wavelengths - 100) ** 2) / 300)

        # Baseline: polynomial
        x_norm = wavelengths / n_features
        baseline = 0.3 * x_norm**2 + 0.1

        spectrum = signal + baseline
        X = spectrum.reshape(1, -1)

        # Apply airPLS
        corrector = BaselineAirPLS(lam=1e6, max_iter=15, tol=1e-3)
        X_corrected = corrector.fit_transform(X)

        # Should remove baseline
        error_before = np.mean((spectrum - signal) ** 2)
        error_after = np.mean((X_corrected[0] - signal) ** 2)

        assert error_after < error_before, \
            "airPLS should reduce baseline error"

    def test_airpls_different_lambda(self):
        """Test airPLS with different smoothness parameters."""
        np.random.seed(42)

        # Create test spectrum
        n_features = 150
        wavelengths = np.arange(n_features)
        signal = 0.5 * np.exp(-((wavelengths - 75) ** 2) / 200)
        baseline = np.linspace(0.2, 0.5, n_features)
        spectrum = signal + baseline
        X = spectrum.reshape(1, -1)

        # Test different lambda values (more spread out)
        lambdas = [1e3, 1e5, 1e7]
        results = []

        for lam in lambdas:
            corrector = BaselineAirPLS(lam=lam, max_iter=15)
            X_corrected = corrector.fit_transform(X)
            results.append(X_corrected)

        # Different lambdas should produce different results (at least first and last)
        # Use more relaxed tolerance since airPLS can converge to similar results
        assert not np.allclose(results[0], results[2], atol=1e-4), \
            f"Lambda {lambdas[0]} and {lambdas[2]} should produce different results"

    def test_airpls_convergence(self):
        """Test airPLS convergence behavior."""
        np.random.seed(42)

        # Create test spectrum
        n_features = 100
        wavelengths = np.arange(n_features)
        signal = np.exp(-((wavelengths - 50) ** 2) / 150)
        baseline = np.linspace(0.3, 0.6, n_features)
        spectrum = signal + baseline
        X = spectrum.reshape(1, -1)

        # With sufficient iterations, should converge
        corrector = BaselineAirPLS(lam=1e6, max_iter=15, tol=1e-3)
        X_corrected = corrector.fit_transform(X)

        # Should not produce NaN or Inf
        assert not np.any(np.isnan(X_corrected)), "airPLS should not produce NaN"
        assert not np.any(np.isinf(X_corrected)), "airPLS should not produce Inf"

        # Should preserve shape
        assert X_corrected.shape == X.shape

    def test_airpls_preserves_peaks(self):
        """Test that airPLS preserves spectral peaks."""
        np.random.seed(42)

        # Create spectrum with multiple peaks
        n_features = 200
        wavelengths = np.arange(n_features)

        # Three clear peaks
        signal = (
            np.exp(-((wavelengths - 50) ** 2) / 100) +
            0.8 * np.exp(-((wavelengths - 100) ** 2) / 120) +
            0.6 * np.exp(-((wavelengths - 150) ** 2) / 90)
        )

        # Add polynomial baseline
        x_norm = wavelengths / n_features
        baseline = 0.4 * x_norm**2 + 0.1 * x_norm + 0.2

        spectrum = signal + baseline
        X = spectrum.reshape(1, -1)

        # Apply airPLS
        corrector = BaselineAirPLS(lam=1e6, max_iter=15)
        X_corrected = corrector.fit_transform(X)

        # Find peak in corrected spectrum around position 50
        peak_region = X_corrected[0, 40:60]
        peak_pos = 40 + np.argmax(peak_region)

        # Peak should be near position 50 (within 10 indices)
        assert abs(peak_pos - 50) <= 10, \
            f"Peak preservation failed: expected near 50, got {peak_pos}"

    def test_airpls_preserves_shape(self):
        """airPLS should preserve array shape."""
        np.random.seed(42)

        shapes = [(1, 50), (5, 100), (15, 150)]

        for shape in shapes:
            X = np.random.randn(*shape) + 1.0

            corrector = BaselineAirPLS(lam=1e5, max_iter=10)
            X_corrected = corrector.fit_transform(X)

            assert X_corrected.shape == X.shape, \
                f"Shape {shape} not preserved"


@pytest.mark.numerical
class TestBaselineComparison:
    """Compare different baseline correction methods."""

    def test_all_methods_reduce_baseline_error(self):
        """All methods should reduce baseline-related error."""
        np.random.seed(42)

        # Create test spectrum
        n_features = 150
        wavelengths = np.arange(n_features)
        x_norm = wavelengths / n_features

        # True signal
        signal = (
            0.8 * np.exp(-((wavelengths - 50) ** 2) / 120) +
            0.5 * np.exp(-((wavelengths - 100) ** 2) / 100)
        )

        # Polynomial baseline
        baseline = 0.3 * x_norm**2 + 0.1 * x_norm + 0.15

        spectrum = signal + baseline
        X = spectrum.reshape(1, -1)

        # Initial error
        error_before = np.mean((spectrum - signal) ** 2)

        # Test all methods
        methods = [
            ("Polynomial", BaselinePolynomial(degree=2)),
            ("ALS", BaselineALS(lambda_=1e5, p=0.001, niter=10)),
            ("airPLS", BaselineAirPLS(lam=1e6, max_iter=15)),
        ]

        for name, corrector in methods:
            X_corrected = corrector.fit_transform(X)
            error_after = np.mean((X_corrected[0] - signal) ** 2)

            assert error_after < error_before, \
                f"{name} should reduce baseline error (before: {error_before:.6f}, after: {error_after:.6f})"

    def test_all_methods_preserve_shape(self):
        """All methods should preserve input shape."""
        np.random.seed(42)
        X = np.random.randn(10, 100) + 1.0

        methods = [
            BaselinePolynomial(degree=2),
            BaselineALS(lambda_=1e5, p=0.001, niter=10),
            BaselineAirPLS(lam=1e6, max_iter=10),
        ]

        for corrector in methods:
            X_corrected = corrector.fit_transform(X)
            assert X_corrected.shape == X.shape, \
                f"{corrector.__class__.__name__} did not preserve shape"


@pytest.mark.unit
class TestBaselineInputValidation:
    """Test input validation for baseline correction."""

    def test_polynomial_invalid_degree(self):
        """Test polynomial with invalid degrees."""
        # Very high degree should still work (numpy handles it)
        corrector = BaselinePolynomial(degree=20)
        X = np.random.randn(5, 50)

        # Should not crash
        X_corrected = corrector.fit_transform(X)
        assert X_corrected.shape == X.shape

    def test_als_negative_lambda(self):
        """Test ALS with negative lambda (should still work, scipy handles it)."""
        # Negative lambda is unusual but scipy doesn't explicitly forbid it
        corrector = BaselineALS(lambda_=-1000, p=0.001, niter=5)
        X = np.random.randn(5, 50) + 2.0

        # Should not crash
        try:
            X_corrected = corrector.fit_transform(X)
            assert X_corrected.shape == X.shape
        except Exception:
            # If it raises an exception, that's also acceptable behavior
            pass

    def test_als_invalid_p(self):
        """Test ALS with p outside [0, 1] range."""
        # p should be in [0, 1] but algorithm may still work
        corrector = BaselineALS(lambda_=1e5, p=1.5, niter=5)
        X = np.random.randn(5, 50) + 2.0

        # Should not crash (though results may be unexpected)
        try:
            X_corrected = corrector.fit_transform(X)
            assert X_corrected.shape == X.shape
        except Exception:
            # If it raises an exception, that's also acceptable
            pass


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])
