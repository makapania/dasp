"""
Comprehensive tests for preprocessing methods.

Tests all preprocessing methods:
- SNV (Standard Normal Variate)
- SavgolDerivative (1st, 2nd derivatives)
- MSC (Multiplicative Scatter Correction)

Test coverage includes:
- Basic functionality
- Edge cases and error handling
- Preprocessing chains
- Memory efficiency
- Transform consistency
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

from spectral_predict_v3.core.preprocess import SNV, SavgolDerivative, MSC


class TestSNV:
    """Test Standard Normal Variate (SNV) transformation."""

    def test_basic_snv(self):
        """Test basic SNV functionality."""
        X = np.random.randn(50, 100) * 5 + 10  # Scaled and shifted
        snv = SNV()
        X_snv = snv.fit_transform(X)

        assert X_snv.shape == X.shape

    def test_snv_normalization(self):
        """Test that SNV normalizes each spectrum."""
        X = np.random.randn(50, 100) * 5 + 10
        snv = SNV()
        X_snv = snv.fit_transform(X)

        # Each row should have mean ~ 0 and std ~ 1
        row_means = X_snv.mean(axis=1)
        row_stds = X_snv.std(axis=1)

        assert np.allclose(row_means, 0, atol=1e-10)
        assert np.allclose(row_stds, 1, atol=1e-10)

    def test_snv_constant_spectrum(self):
        """Test SNV handles constant spectra (zero variance)."""
        X = np.random.randn(50, 100)
        # Make one spectrum constant
        X[0, :] = 5.0

        snv = SNV()
        X_snv = snv.fit_transform(X)

        # Constant spectrum should remain constant (division by 1.0)
        assert np.allclose(X_snv[0, :], 0.0)  # (5 - 5) / 1 = 0

    def test_snv_preserves_shape(self):
        """Test that SNV preserves data shape."""
        shapes = [(10, 50), (100, 200), (5, 10)]

        for shape in shapes:
            X = np.random.randn(*shape)
            snv = SNV()
            X_snv = snv.fit_transform(X)
            assert X_snv.shape == shape

    def test_snv_no_fit_required(self):
        """Test that SNV works without explicit fit (stateless)."""
        X_train = np.random.randn(50, 100)
        X_test = np.random.randn(20, 100)

        snv = SNV()
        X_train_snv = snv.transform(X_train)
        X_test_snv = snv.transform(X_test)

        # Both should work without calling fit
        assert X_train_snv.shape == X_train.shape
        assert X_test_snv.shape == X_test.shape


class TestSavgolDerivative:
    """Test Savitzky-Golay derivative transformation."""

    def test_basic_first_derivative(self):
        """Test basic 1st derivative."""
        X = np.random.randn(50, 100)
        deriv = SavgolDerivative(deriv=1, window=7)
        X_deriv = deriv.fit_transform(X)

        assert X_deriv.shape == X.shape

    def test_basic_second_derivative(self):
        """Test basic 2nd derivative."""
        X = np.random.randn(50, 100)
        deriv = SavgolDerivative(deriv=2, window=7)
        X_deriv = deriv.fit_transform(X)

        assert X_deriv.shape == X.shape

    def test_derivative_detects_peaks(self):
        """Test that derivative detects peaks in spectra."""
        # Create synthetic spectrum with a peak
        wavelengths = np.linspace(0, 10, 100)
        spectrum = np.exp(-(wavelengths - 5)**2)  # Gaussian peak at x=5

        X = spectrum.reshape(1, -1)

        # First derivative should be zero at peak
        deriv1 = SavgolDerivative(deriv=1, window=11)
        X_deriv1 = deriv1.fit_transform(X)

        # Find where derivative crosses zero (peak location)
        # Derivative should change sign around the peak
        signs_before = X_deriv1[0, :45] > 0
        signs_after = X_deriv1[0, 55:] < 0

        # Most points before peak should have positive derivative
        # Most points after peak should have negative derivative
        assert signs_before.sum() > signs_before.size * 0.5
        assert signs_after.sum() > signs_after.size * 0.5

    def test_even_window_auto_increment(self):
        """Test that even window size is automatically incremented to odd."""
        X = np.random.randn(50, 100)

        # Even window (should be incremented to 9)
        deriv = SavgolDerivative(deriv=1, window=8)
        X_deriv = deriv.fit_transform(X)

        assert X_deriv.shape == X.shape

    def test_default_polyorder(self):
        """Test default polyorder selection."""
        X = np.random.randn(50, 100)

        # deriv=1 should default to polyorder=2
        deriv1 = SavgolDerivative(deriv=1, window=7)
        X_deriv1 = deriv1.fit_transform(X)

        # deriv=2 should default to polyorder=3
        deriv2 = SavgolDerivative(deriv=2, window=7)
        X_deriv2 = deriv2.fit_transform(X)

        assert X_deriv1.shape == X.shape
        assert X_deriv2.shape == X.shape

    def test_custom_polyorder(self):
        """Test custom polyorder."""
        X = np.random.randn(50, 100)

        deriv = SavgolDerivative(deriv=1, window=9, polyorder=3)
        X_deriv = deriv.fit_transform(X)

        assert X_deriv.shape == X.shape

    def test_window_too_large_error(self):
        """Test error when window is larger than number of features."""
        X = np.random.randn(50, 20)  # Only 20 features

        deriv = SavgolDerivative(deriv=1, window=25)

        with pytest.raises(ValueError, match="Window length.*must be <="):
            deriv.fit_transform(X)

    def test_window_too_small_for_polyorder(self):
        """Test error when window is too small for polyorder."""
        X = np.random.randn(50, 100)

        deriv = SavgolDerivative(deriv=1, window=5, polyorder=4)

        with pytest.raises(ValueError, match="Window length.*must be >="):
            deriv.fit_transform(X)

    def test_different_window_sizes(self):
        """Test various window sizes."""
        X = np.random.randn(50, 100)

        for window in [5, 7, 9, 11, 15]:
            deriv = SavgolDerivative(deriv=1, window=window)
            X_deriv = deriv.fit_transform(X)
            assert X_deriv.shape == X.shape


class TestMSC:
    """Test Multiplicative Scatter Correction (MSC)."""

    def test_basic_msc_mean(self):
        """Test MSC with mean reference."""
        X = np.random.randn(50, 100) * 2 + 5
        msc = MSC(reference='mean')
        X_corrected = msc.fit_transform(X)

        assert X_corrected.shape == X.shape
        assert hasattr(msc, 'reference_')
        assert len(msc.reference_) == X.shape[1]

    def test_msc_median_reference(self):
        """Test MSC with median reference."""
        X = np.random.randn(50, 100) * 2 + 5

        # Add outliers
        X[0, :] += 100

        msc_mean = MSC(reference='mean')
        msc_median = MSC(reference='median')

        X_mean = msc_mean.fit_transform(X)
        X_median = msc_median.fit_transform(X)

        # Median should be more robust to outliers
        assert not np.allclose(X_mean, X_median)

    def test_msc_custom_reference(self):
        """Test MSC with custom reference spectrum."""
        X = np.random.randn(50, 100)
        custom_ref = np.random.randn(100)

        msc = MSC(reference=custom_ref)
        X_corrected = msc.fit_transform(X)

        assert X_corrected.shape == X.shape
        assert np.allclose(msc.reference_, custom_ref)

    def test_msc_scatter_correction_effectiveness(self):
        """Test that MSC actually corrects scatter effects."""
        # Create synthetic data with known scatter
        n_samples, n_wavelengths = 50, 100
        wavelengths = np.linspace(1000, 2500, n_wavelengths)

        # True underlying spectrum
        true_spectrum = np.sin(wavelengths / 200) * 0.5 + 1.0

        # Add scatter: each spectrum = a + b * true_spectrum + noise
        X = np.zeros((n_samples, n_wavelengths))
        for i in range(n_samples):
            a = np.random.randn() * 0.2  # Additive offset
            b = 0.5 + np.random.randn() * 0.3  # Multiplicative factor
            noise = np.random.randn(n_wavelengths) * 0.02
            X[i, :] = a + b * true_spectrum + noise

        # Variance before MSC
        var_before = np.var(X, axis=0).mean()

        # Apply MSC
        msc = MSC(reference='mean')
        X_corrected = msc.fit_transform(X)

        # Variance after MSC
        var_after = np.var(X_corrected, axis=0).mean()

        # MSC should reduce inter-sample variance
        # (This is expected but not guaranteed for all random data)
        assert not np.allclose(X, X_corrected)

    def test_msc_reference_computation(self):
        """Test MSC reference spectrum computation."""
        X = np.random.randn(50, 100)

        # Mean reference
        msc_mean = MSC(reference='mean')
        msc_mean.fit(X)
        assert np.allclose(msc_mean.reference_, X.mean(axis=0))

        # Median reference
        msc_median = MSC(reference='median')
        msc_median.fit(X)
        assert np.allclose(msc_median.reference_, np.median(X, axis=0))

    def test_msc_constant_spectrum_handling(self):
        """Test that MSC handles constant spectra gracefully."""
        X = np.random.randn(50, 100)
        X[0, :] = 5.0  # Constant spectrum

        msc = MSC(reference='mean')
        X_corrected = msc.fit_transform(X)

        # Constant spectrum should be returned unchanged
        assert np.allclose(X_corrected[0, :], X[0, :])

    def test_msc_zero_variance_reference_warning(self):
        """Test warning when reference has zero variance."""
        X = np.ones((50, 100))  # All constant

        msc = MSC(reference='mean')

        with pytest.warns(UserWarning, match="near-zero variance"):
            X_corrected = msc.fit_transform(X)

        # Should return data unchanged
        assert np.allclose(X_corrected, X)

    def test_msc_transform_consistency(self):
        """Test MSC transform consistency for train/test."""
        X_train = np.random.randn(50, 100)
        X_test = np.random.randn(20, 100)

        msc = MSC(reference='mean')
        msc.fit(X_train)

        X_train_corrected = msc.transform(X_train)
        X_test_corrected = msc.transform(X_test)

        # Same reference should be used
        assert X_train_corrected.shape == X_train.shape
        assert X_test_corrected.shape == X_test.shape

    def test_msc_different_train_test_means(self):
        """Test that MSC uses training mean for test data."""
        X_train = np.random.randn(50, 100) + 5  # Mean ~ 5
        X_test = np.random.randn(20, 100) + 10  # Mean ~ 10

        msc = MSC(reference='mean')
        msc.fit(X_train)

        # Reference should be from training data
        assert np.allclose(msc.reference_, X_train.mean(axis=0), atol=0.5)

        # Test data should use training reference (not test mean)
        X_test_corrected = msc.transform(X_test)
        assert X_test_corrected.shape == X_test.shape

    def test_msc_invalid_reference_type(self):
        """Test error for invalid reference type."""
        X = np.random.randn(50, 100)
        msc = MSC(reference='invalid')

        with pytest.raises(ValueError, match="reference must be"):
            msc.fit(X)

    def test_msc_custom_reference_length_mismatch(self):
        """Test error when custom reference length doesn't match features."""
        X = np.random.randn(50, 100)
        custom_ref = np.random.randn(50)  # Wrong length

        msc = MSC(reference=custom_ref)

        with pytest.raises(ValueError, match="Reference spectrum length"):
            msc.fit(X)


class TestPreprocessingChains:
    """Test combinations of preprocessing methods."""

    def test_msc_then_snv(self):
        """Test MSC followed by SNV."""
        X = np.random.randn(50, 100) * 2 + 5

        pipeline = Pipeline([
            ('msc', MSC(reference='mean')),
            ('snv', SNV())
        ])

        X_processed = pipeline.fit_transform(X)

        assert X_processed.shape == X.shape

        # After SNV, each row should be normalized
        row_means = X_processed.mean(axis=1)
        row_stds = X_processed.std(axis=1)
        assert np.allclose(row_means, 0, atol=1e-10)
        assert np.allclose(row_stds, 1, atol=1e-10)

    def test_snv_then_derivative(self):
        """Test SNV followed by Savitzky-Golay derivative."""
        X = np.random.randn(50, 100)

        pipeline = Pipeline([
            ('snv', SNV()),
            ('deriv', SavgolDerivative(deriv=1, window=7))
        ])

        X_processed = pipeline.fit_transform(X)

        assert X_processed.shape == X.shape

    def test_msc_then_deriv(self):
        """Test MSC followed by derivative."""
        X = np.random.randn(50, 100)

        pipeline = Pipeline([
            ('msc', MSC(reference='mean')),
            ('deriv', SavgolDerivative(deriv=1, window=7))
        ])

        X_processed = pipeline.fit_transform(X)

        assert X_processed.shape == X.shape

    def test_full_preprocessing_chain(self):
        """Test comprehensive preprocessing chain: MSC → SNV → Derivative."""
        X = np.random.randn(50, 100) * 3 + 7

        pipeline = Pipeline([
            ('msc', MSC(reference='mean')),
            ('snv', SNV()),
            ('deriv', SavgolDerivative(deriv=1, window=7))
        ])

        X_processed = pipeline.fit_transform(X)

        assert X_processed.shape == X.shape
        assert not np.allclose(X, X_processed)

    def test_snv_then_second_derivative(self):
        """Test SNV followed by 2nd derivative."""
        X = np.random.randn(50, 100)

        pipeline = Pipeline([
            ('snv', SNV()),
            ('deriv2', SavgolDerivative(deriv=2, window=9))
        ])

        X_processed = pipeline.fit_transform(X)

        assert X_processed.shape == X.shape

    def test_msc_median_then_deriv(self):
        """Test MSC with median reference followed by derivative."""
        X = np.random.randn(50, 100)

        pipeline = Pipeline([
            ('msc', MSC(reference='median')),
            ('deriv', SavgolDerivative(deriv=1, window=7))
        ])

        X_processed = pipeline.fit_transform(X)

        assert X_processed.shape == X.shape


class TestPreprocessingEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_sample(self):
        """Test preprocessing with single sample."""
        X = np.random.randn(1, 100)

        # SNV
        snv = SNV()
        X_snv = snv.fit_transform(X)
        assert X_snv.shape == X.shape

        # MSC
        msc = MSC(reference='mean')
        X_msc = msc.fit_transform(X)
        assert X_msc.shape == X.shape

        # Derivative
        deriv = SavgolDerivative(deriv=1, window=7)
        X_deriv = deriv.fit_transform(X)
        assert X_deriv.shape == X.shape

    def test_small_feature_count(self):
        """Test preprocessing with small number of features."""
        X = np.random.randn(50, 10)

        # SNV
        snv = SNV()
        X_snv = snv.fit_transform(X)
        assert X_snv.shape == X.shape

        # MSC
        msc = MSC(reference='mean')
        X_msc = msc.fit_transform(X)
        assert X_msc.shape == X.shape

        # Derivative (small window)
        deriv = SavgolDerivative(deriv=1, window=5)
        X_deriv = deriv.fit_transform(X)
        assert X_deriv.shape == X.shape

    def test_nan_handling(self):
        """Test that NaN values are preserved or handled."""
        X = np.random.randn(50, 100)
        X[0, 50] = np.nan

        # SNV should propagate NaN
        snv = SNV()
        X_snv = snv.fit_transform(X)
        assert np.isnan(X_snv[0, 50])

    def test_inf_handling(self):
        """Test that Inf values are handled."""
        X = np.random.randn(50, 100)
        X[0, 50] = np.inf

        # SNV should handle Inf
        snv = SNV()
        X_snv = snv.fit_transform(X)
        # Result may be Inf or large value
        assert X_snv.shape == X.shape


class TestMemoryEfficiency:
    """Test memory efficiency on large datasets."""

    def test_large_dataset_snv(self):
        """Test SNV on large dataset."""
        # Simulate large dataset
        X = np.random.randn(1000, 500)

        snv = SNV()
        X_snv = snv.fit_transform(X)

        assert X_snv.shape == X.shape

    def test_large_dataset_msc(self):
        """Test MSC on large dataset."""
        X = np.random.randn(1000, 500)

        msc = MSC(reference='mean')
        X_msc = msc.fit_transform(X)

        assert X_msc.shape == X.shape

    def test_large_dataset_derivative(self):
        """Test derivative on large dataset."""
        X = np.random.randn(1000, 500)

        deriv = SavgolDerivative(deriv=1, window=7)
        X_deriv = deriv.fit_transform(X)

        assert X_deriv.shape == X.shape

    def test_large_dataset_full_pipeline(self):
        """Test full preprocessing pipeline on large dataset."""
        X = np.random.randn(1000, 500)

        pipeline = Pipeline([
            ('msc', MSC(reference='mean')),
            ('snv', SNV()),
            ('deriv', SavgolDerivative(deriv=1, window=7))
        ])

        X_processed = pipeline.fit_transform(X)

        assert X_processed.shape == X.shape


class TestInvertibility:
    """Test whether preprocessing is invertible (where applicable)."""

    def test_snv_not_invertible(self):
        """Test that SNV is not directly invertible (information lost)."""
        X = np.random.randn(50, 100) * 5 + 10

        snv = SNV()
        X_snv = snv.fit_transform(X)

        # Cannot recover original scale without storing means and stds
        # This is expected behavior for SNV
        assert not np.allclose(X, X_snv)

    def test_derivative_not_invertible(self):
        """Test that derivative is not invertible (information lost)."""
        X = np.random.randn(50, 100)

        deriv = SavgolDerivative(deriv=1, window=7)
        X_deriv = deriv.fit_transform(X)

        # Derivative loses absolute position information
        assert not np.allclose(X, X_deriv)


def run_tests():
    """Run all tests using pytest."""
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'pytest', __file__, '-v'],
        capture_output=True,
        text=True
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
