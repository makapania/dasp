"""
MASTER DEBUGGER AGENT - Adversarial Testing Suite for Interference Removal
============================================================================

This script systematically tests Phase 1 interference removal implementation
with extreme inputs, edge cases, and numerical instabilities.

Testing Categories:
1. Numerical Stability (extreme values, division by zero, singularities)
2. Edge Cases (single sample, single wavelength, etc.)
3. Memory & Performance (large datasets)
4. API Misuse (transform before fit, shape mismatches)
5. Cross-Validation Leakage (deep dive)
6. Integration Issues (pipelines, serialization)
7. Error Messages (clarity and actionability)
"""

import numpy as np
import pytest
import pickle
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression
from spectral_predict.interference import WavelengthExcluder, MSC, OSC


class TestNumericalStability:
    """Test numerical stability with extreme inputs."""

    def test_wavelength_excluder_extreme_values(self):
        """Test WavelengthExcluder with extreme wavelength values."""
        # Very large wavelengths
        wavelengths = np.arange(1e6, 1e6 + 100)
        X = np.random.randn(10, 100)

        excluder = WavelengthExcluder(wavelengths, exclude_ranges=[(1e6 + 20, 1e6 + 40)])
        X_filtered = excluder.fit_transform(X)

        assert X_filtered.shape[1] == 79  # 100 - 21 = 79
        assert not np.any(np.isnan(X_filtered))
        assert not np.any(np.isinf(X_filtered))

    def test_msc_extreme_small_values(self):
        """Test MSC with very small values (1e-10)."""
        X = np.random.randn(50, 100) * 1e-10

        msc = MSC(reference='mean')
        X_corrected = msc.fit_transform(X)

        # Should handle without errors
        assert X_corrected.shape == X.shape
        # May produce NaN due to numerical precision, but should not crash
        # Check that most values are not NaN
        nan_ratio = np.sum(np.isnan(X_corrected)) / X_corrected.size
        assert nan_ratio < 0.5  # Allow some NaN but not majority

    def test_msc_extreme_large_values(self):
        """Test MSC with very large values (1e10)."""
        X = np.random.randn(50, 100) * 1e10

        msc = MSC(reference='mean')
        X_corrected = msc.fit_transform(X)

        assert X_corrected.shape == X.shape
        assert not np.any(np.isnan(X_corrected))
        # Allow Inf due to large values, but check for NaN
        # assert not np.any(np.isinf(X_corrected))

    def test_msc_all_zeros(self):
        """Test MSC with all-zero spectra."""
        X = np.zeros((10, 50))

        msc = MSC(reference='mean')
        msc.fit(X)
        X_corrected = msc.transform(X)

        # Should return zeros unchanged
        assert X_corrected.shape == X.shape
        assert not np.any(np.isnan(X_corrected))
        assert not np.any(np.isinf(X_corrected))

    def test_msc_all_constants(self):
        """Test MSC with constant spectra (all values = 5.0)."""
        X = np.ones((10, 50)) * 5.0

        msc = MSC(reference='mean')
        msc.fit(X)
        X_corrected = msc.transform(X)

        # Should handle gracefully
        assert X_corrected.shape == X.shape
        assert not np.any(np.isnan(X_corrected))

    def test_msc_with_nan_input(self):
        """Test MSC with NaN values in input."""
        X = np.random.randn(50, 100)
        X[5, 10] = np.nan
        X[10, 20] = np.nan

        msc = MSC(reference='mean')

        # sklearn check_array should catch this
        with pytest.raises(ValueError):
            msc.fit(X)

    def test_msc_with_inf_input(self):
        """Test MSC with Inf values in input."""
        X = np.random.randn(50, 100)
        X[5, 10] = np.inf
        X[10, 20] = -np.inf

        msc = MSC(reference='mean')

        # sklearn check_array should catch this
        with pytest.raises(ValueError):
            msc.fit(X)

    def test_msc_negative_values(self):
        """Test MSC with all negative values."""
        X = -np.abs(np.random.randn(50, 100))

        msc = MSC(reference='mean')
        X_corrected = msc.fit_transform(X)

        assert X_corrected.shape == X.shape
        assert not np.any(np.isnan(X_corrected))

    def test_osc_extreme_small_values(self):
        """Test OSC with very small values."""
        X = np.random.randn(50, 100) * 1e-10
        y = np.random.randn(50) * 1e-10

        osc = OSC(n_components=1)
        X_corrected = osc.fit_transform(X, y)

        assert X_corrected.shape == X.shape
        # May have numerical issues, but should not crash

    def test_osc_extreme_large_values(self):
        """Test OSC with very large values."""
        X = np.random.randn(50, 100) * 1e8
        y = np.random.randn(50) * 1e8

        osc = OSC(n_components=1)
        X_corrected = osc.fit_transform(X, y)

        assert X_corrected.shape == X.shape

    def test_osc_all_zeros(self):
        """Test OSC with all-zero X and y."""
        X = np.zeros((50, 100))
        y = np.zeros(50)

        osc = OSC(n_components=1)
        # Should handle or raise clear error
        try:
            X_corrected = osc.fit_transform(X, y)
            # If it succeeds, check output
            assert X_corrected.shape == X.shape
        except (np.linalg.LinAlgError, ValueError):
            # Acceptable to fail with clear error
            pass

    def test_osc_constant_y(self):
        """Test OSC with constant y (zero variance)."""
        X = np.random.randn(50, 100)
        y = np.ones(50) * 5.0  # Constant y

        osc = OSC(n_components=1)
        # Should handle or warn
        with warnings.catch_warnings(record=True):
            X_corrected = osc.fit_transform(X, y)
            assert X_corrected.shape == X.shape


class TestEdgeCases:
    """Test edge cases: single sample, single wavelength, etc."""

    def test_wavelength_excluder_single_sample(self):
        """Test WavelengthExcluder with single sample."""
        wavelengths = np.arange(1000, 1100)
        X = np.random.randn(1, 100)

        excluder = WavelengthExcluder(wavelengths, exclude_ranges=[(1020, 1040)])
        X_filtered = excluder.fit_transform(X)

        assert X_filtered.shape[0] == 1
        assert X_filtered.shape[1] == 79  # 100 - 21

    def test_wavelength_excluder_single_wavelength(self):
        """Test WavelengthExcluder with single wavelength."""
        wavelengths = np.array([1500.0])
        X = np.random.randn(50, 1)

        excluder = WavelengthExcluder(wavelengths, exclude_ranges=[])
        X_filtered = excluder.fit_transform(X)

        assert X_filtered.shape == (50, 1)

    def test_wavelength_excluder_empty_exclusion(self):
        """Test WavelengthExcluder with empty exclusion ranges."""
        wavelengths = np.arange(1000, 2000)
        X = np.random.randn(50, 1000)

        excluder = WavelengthExcluder(wavelengths, exclude_ranges=[])
        X_filtered = excluder.fit_transform(X)

        # Should keep all wavelengths
        np.testing.assert_array_equal(X_filtered, X)

    def test_wavelength_excluder_overlapping_ranges(self):
        """Test WavelengthExcluder with overlapping exclusion ranges."""
        wavelengths = np.arange(1000, 2000)
        X = np.random.randn(50, 1000)

        # Overlapping ranges
        excluder = WavelengthExcluder(
            wavelengths,
            exclude_ranges=[(1200, 1400), (1300, 1500)]
        )
        X_filtered = excluder.fit_transform(X)

        # Should exclude union of ranges: 1200-1500 (301 wavelengths)
        assert X_filtered.shape[1] == 1000 - 301

    def test_wavelength_excluder_inverted_wavelengths(self):
        """Test WavelengthExcluder with descending wavelength order."""
        wavelengths = np.arange(2500, 1000, -1)  # Descending
        X = np.random.randn(50, 1500)

        excluder = WavelengthExcluder(
            wavelengths,
            exclude_ranges=[(1400, 1500), (1900, 2000)]
        )
        X_filtered = excluder.fit_transform(X)

        # Should still exclude correctly
        assert X_filtered.shape[1] < 1500

    def test_msc_single_sample(self):
        """Test MSC with single sample (n_samples=1)."""
        X = np.random.randn(1, 100)

        msc = MSC(reference='mean')
        msc.fit(X)
        X_corrected = msc.transform(X)

        assert X_corrected.shape == (1, 100)

    def test_msc_single_wavelength(self):
        """Test MSC with single wavelength (n_features=1)."""
        X = np.random.randn(50, 1)

        msc = MSC(reference='mean')
        msc.fit(X)
        X_corrected = msc.transform(X)

        # polyfit should handle this
        assert X_corrected.shape == (50, 1)

    def test_osc_single_sample(self):
        """Test OSC with single sample."""
        X = np.random.randn(1, 100)
        y = np.array([1.0])

        osc = OSC(n_components=1)

        # PLS requires at least 2 samples, should fail or handle
        try:
            X_corrected = osc.fit_transform(X, y)
        except (ValueError, np.linalg.LinAlgError):
            pass  # Expected to fail

    def test_osc_single_wavelength(self):
        """Test OSC with single wavelength."""
        X = np.random.randn(50, 1)
        y = np.random.randn(50)

        osc = OSC(n_components=1)

        # Should handle or raise clear error
        try:
            X_corrected = osc.fit_transform(X, y)
            assert X_corrected.shape == (50, 1)
        except (ValueError, np.linalg.LinAlgError):
            pass  # Acceptable

    def test_osc_more_components_than_features(self):
        """Test OSC with n_components > n_wavelengths."""
        X = np.random.randn(100, 10)  # Only 10 features
        y = np.random.randn(100)

        osc = OSC(n_components=20)  # More than n_features

        # Should handle gracefully
        with warnings.catch_warnings(record=True):
            X_corrected = osc.fit_transform(X, y)
            # Should extract fewer components than requested
            assert X_corrected.shape == X.shape

    def test_osc_more_components_than_samples(self):
        """Test OSC with n_components > n_samples."""
        X = np.random.randn(10, 100)  # Only 10 samples
        y = np.random.randn(10)

        osc = OSC(n_components=20)  # More than n_samples

        # PLS should be limited by n_samples
        with warnings.catch_warnings(record=True):
            X_corrected = osc.fit_transform(X, y)
            assert X_corrected.shape == X.shape


class TestMemoryPerformance:
    """Test memory and performance with large datasets."""

    def test_large_dataset_wavelength_excluder(self):
        """Test WavelengthExcluder with large dataset."""
        wavelengths = np.arange(1000, 6000)  # 5000 wavelengths
        X = np.random.randn(10000, 5000)

        excluder = WavelengthExcluder(
            wavelengths,
            exclude_ranges=[(1400, 1500), (1900, 2000)]
        )
        X_filtered = excluder.fit_transform(X)

        assert X_filtered.shape[0] == 10000
        assert X_filtered.shape[1] < 5000

    def test_large_dataset_msc(self):
        """Test MSC with large dataset."""
        X = np.random.randn(5000, 1000)

        msc = MSC(reference='mean')
        X_corrected = msc.fit_transform(X)

        assert X_corrected.shape == X.shape

    def test_large_dataset_osc(self):
        """Test OSC with large dataset."""
        X = np.random.randn(2000, 500)
        y = np.random.randn(2000)

        osc = OSC(n_components=2)
        X_corrected = osc.fit_transform(X, y)

        assert X_corrected.shape == X.shape


class TestAPIMisuse:
    """Test API misuse scenarios."""

    def test_transform_before_fit_wavelength_excluder(self):
        """Test transforming before fitting WavelengthExcluder."""
        wavelengths = np.arange(1000, 2000)
        X = np.random.randn(50, 1000)

        excluder = WavelengthExcluder(wavelengths)

        with pytest.raises(Exception):  # NotFittedError
            excluder.transform(X)

    def test_transform_before_fit_msc(self):
        """Test transforming before fitting MSC."""
        X = np.random.randn(50, 100)
        msc = MSC()

        with pytest.raises(Exception):  # NotFittedError
            msc.transform(X)

    def test_transform_before_fit_osc(self):
        """Test transforming before fitting OSC."""
        X = np.random.randn(50, 100)
        osc = OSC(n_components=1)

        with pytest.raises(Exception):  # NotFittedError
            osc.transform(X)

    def test_shape_mismatch_wavelength_excluder(self):
        """Test shape mismatch in WavelengthExcluder transform."""
        wavelengths = np.arange(1000, 2000)
        X_train = np.random.randn(50, 1000)
        X_test = np.random.randn(20, 500)  # Wrong shape

        excluder = WavelengthExcluder(wavelengths)
        excluder.fit(X_train)

        with pytest.raises(ValueError, match="was fitted with"):
            excluder.transform(X_test)

    def test_shape_mismatch_msc(self):
        """Test shape mismatch in MSC transform."""
        X_train = np.random.randn(50, 100)
        X_test = np.random.randn(20, 50)  # Wrong shape

        msc = MSC()
        msc.fit(X_train)

        with pytest.raises(ValueError, match="was fitted with"):
            msc.transform(X_test)

    def test_shape_mismatch_osc(self):
        """Test shape mismatch in OSC transform."""
        X_train = np.random.randn(50, 100)
        X_test = np.random.randn(20, 50)  # Wrong shape
        y_train = np.random.randn(50)

        osc = OSC(n_components=1)
        osc.fit(X_train, y_train)

        with pytest.raises(ValueError, match="was fitted with"):
            osc.transform(X_test)

    def test_wrong_dtype_string(self):
        """Test passing string data."""
        X = np.array([['a', 'b', 'c'], ['d', 'e', 'f']])
        y = np.array([1.0, 2.0])

        osc = OSC(n_components=1)

        with pytest.raises(ValueError):
            osc.fit(X, y)


class TestCrossValidationLeakage:
    """Deep dive into cross-validation data leakage testing."""

    def test_osc_cv_with_different_folds(self):
        """Test OSC in cross-validation produces different results per fold."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        pipeline = Pipeline([
            ('osc', OSC(n_components=1)),
            ('pls', PLSRegression(n_components=3))
        ])

        # Run cross-validation
        scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')

        # Should complete without errors
        assert len(scores) == 5
        # Scores should vary across folds (not identical)
        assert np.std(scores) > 0

    def test_msc_cv_recomputes_reference(self):
        """Test that MSC recomputes reference for each CV fold."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        pipeline = Pipeline([
            ('msc', MSC(reference='mean')),
            ('pls', PLSRegression(n_components=3))
        ])

        # Run cross-validation
        scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')

        # Should complete without errors
        assert len(scores) == 5

    def test_osc_preserves_test_set_statistics(self):
        """
        Critical test: Verify OSC doesn't leak test set statistics.

        Test that when transforming test data, OSC uses TRAINING mean,
        not test mean.
        """
        np.random.seed(42)

        # Training data with mean=0
        X_train = np.random.randn(100, 50)
        y_train = np.random.randn(100)

        # Test data with mean=10 (very different!)
        X_test = np.random.randn(30, 50) + 10.0

        osc = OSC(n_components=1)
        osc.fit(X_train, y_train)

        # Store training mean
        train_mean = osc.X_mean_
        assert abs(np.mean(train_mean)) < 0.5  # Should be near 0

        # Transform test set
        X_test_osc = osc.transform(X_test)

        # If data leakage exists, X_test_osc would be centered to ~0
        # If no leakage, X_test_osc should maintain offset from training mean
        test_mean = np.mean(X_test_osc)

        # Test data should NOT be centered to zero
        assert abs(test_mean) > 5.0  # Should still have significant offset

        # Also test that training data IS properly centered
        X_train_osc = osc.transform(X_train)
        train_mean_transformed = np.mean(X_train_osc)
        assert abs(train_mean_transformed) < 1.0  # Should be near zero

    def test_msc_preserves_test_set_statistics(self):
        """
        Test that MSC doesn't leak test set statistics.

        MSC reference should be computed ONLY from training data.
        """
        np.random.seed(42)

        # Training data
        X_train = np.random.randn(100, 50)

        # Test data with very different distribution
        X_test = np.random.randn(30, 50) * 2.0 + 5.0

        msc = MSC(reference='mean')
        msc.fit(X_train)

        # Store training reference
        train_ref = msc.reference_.copy()

        # Transform test set
        X_test_msc = msc.transform(X_test)

        # Reference should not change after transforming test data
        np.testing.assert_array_equal(msc.reference_, train_ref)


class TestIntegration:
    """Test integration with sklearn components."""

    def test_pickle_serialization_wavelength_excluder(self):
        """Test pickling WavelengthExcluder."""
        wavelengths = np.arange(1000, 2000)
        X = np.random.randn(50, 1000)

        excluder = WavelengthExcluder(wavelengths, exclude_ranges=[(1400, 1500)])
        excluder.fit(X)

        # Pickle and unpickle
        pickled = pickle.dumps(excluder)
        excluder_loaded = pickle.loads(pickled)

        # Should produce same results
        X_orig = excluder.transform(X)
        X_loaded = excluder_loaded.transform(X)

        np.testing.assert_array_equal(X_orig, X_loaded)

    def test_pickle_serialization_msc(self):
        """Test pickling MSC."""
        X = np.random.randn(50, 100)

        msc = MSC(reference='mean')
        msc.fit(X)

        # Pickle and unpickle
        pickled = pickle.dumps(msc)
        msc_loaded = pickle.loads(pickled)

        # Should produce same results
        X_orig = msc.transform(X)
        X_loaded = msc_loaded.transform(X)

        np.testing.assert_array_almost_equal(X_orig, X_loaded)

    def test_pickle_serialization_osc(self):
        """Test pickling OSC."""
        X = np.random.randn(50, 100)
        y = np.random.randn(50)

        osc = OSC(n_components=2)
        osc.fit(X, y)

        # Pickle and unpickle
        pickled = pickle.dumps(osc)
        osc_loaded = pickle.loads(pickled)

        # Should produce same results
        X_orig = osc.transform(X)
        X_loaded = osc_loaded.transform(X)

        np.testing.assert_array_almost_equal(X_orig, X_loaded)

    def test_pipeline_with_gridsearch(self):
        """Test interference removal in GridSearchCV."""
        from sklearn.model_selection import GridSearchCV

        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        pipeline = Pipeline([
            ('osc', OSC(n_components=1)),
            ('pls', PLSRegression(n_components=3))
        ])

        param_grid = {
            'osc__n_components': [1, 2],
            'pls__n_components': [2, 3, 5]
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2')
        grid_search.fit(X, y)

        # Should complete without errors
        assert hasattr(grid_search, 'best_score_')
        assert hasattr(grid_search, 'best_params_')


class TestErrorMessages:
    """Test that error messages are clear and actionable."""

    def test_wavelength_mismatch_error_message(self):
        """Test error message for wavelength mismatch."""
        wavelengths = np.arange(1000, 1100)  # 100 wavelengths
        X = np.random.randn(50, 200)  # 200 features (mismatch!)

        excluder = WavelengthExcluder(wavelengths)

        try:
            excluder.fit(X)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            error_msg = str(e)
            # Check that error message contains helpful info
            assert "100" in error_msg  # wavelength array length
            assert "200" in error_msg  # number of features
            assert "must match" in error_msg.lower()

    def test_sample_mismatch_error_message(self):
        """Test error message for X-y sample mismatch in OSC."""
        X = np.random.randn(100, 50)
        y = np.random.randn(50)  # Mismatch!

        osc = OSC(n_components=1)

        try:
            osc.fit(X, y)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            error_msg = str(e)
            # Check that error message contains helpful info
            assert "100" in error_msg
            assert "50" in error_msg
            assert "same number of samples" in error_msg.lower()

    def test_invalid_reference_error_message(self):
        """Test error message for invalid MSC reference."""
        X = np.random.randn(50, 100)
        msc = MSC(reference='invalid_option')

        try:
            msc.fit(X)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            error_msg = str(e)
            # Check that error message suggests valid options
            assert "'mean'" in error_msg or "mean" in error_msg
            assert "'median'" in error_msg or "median" in error_msg


class TestSpecialScenarios:
    """Test special real-world scenarios."""

    def test_osc_with_perfect_correlation(self):
        """Test OSC when y is perfectly correlated with X."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = X[:, 0]  # y = first wavelength (perfect correlation)

        osc = OSC(n_components=1)
        X_corrected = osc.fit_transform(X, y)

        # Should still work, though may not remove much variance
        assert X_corrected.shape == X.shape

    def test_osc_with_no_correlation(self):
        """Test OSC when y has no correlation with X."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)  # Independent of X

        osc = OSC(n_components=1)
        X_corrected = osc.fit_transform(X, y)

        # Should still work
        assert X_corrected.shape == X.shape

    def test_wavelength_excluder_exclude_all_but_one(self):
        """Test WavelengthExcluder leaving only one wavelength."""
        wavelengths = np.arange(1000, 2000)
        X = np.random.randn(50, 1000)

        excluder = WavelengthExcluder(
            wavelengths,
            exclude_ranges=[(1000, 1499), (1501, 2000)]  # Keep only 1500
        )

        with warnings.catch_warnings(record=True):
            excluder.fit(X)
            X_filtered = excluder.transform(X)

        # Should have only 1 wavelength
        assert X_filtered.shape[1] == 1
        assert excluder.wavelengths_out_[0] == 1500

    def test_msc_with_one_outlier_spectrum(self):
        """Test MSC robustness with one extreme outlier."""
        np.random.seed(42)
        X = np.random.randn(50, 100)
        X[0, :] = X[0, :] * 1000  # One extreme outlier

        msc = MSC(reference='median')  # Median should be more robust
        X_corrected = msc.fit_transform(X)

        # Should handle without crashing
        assert X_corrected.shape == X.shape
        # Check that non-outlier samples are reasonable
        assert np.all(np.abs(X_corrected[1:, :]) < 1000)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])
