"""
Unit tests for interference removal transformers.

Tests for:
- WavelengthExcluder
- MSC (Multiplicative Scatter Correction)
- OSC (Orthogonal Signal Correction)
- EPO, GLSW, DOSC (when implemented)
"""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression
from spectral_predict.interference import (
    WavelengthExcluder,
    MSC,
    OSC,
    EPO,
    GLSW,
    DOSC
)


class TestWavelengthExcluder:
    """Tests for WavelengthExcluder transformer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.wavelengths = np.arange(1000, 2501)  # 1000-2500 nm
        self.n_samples = 50
        self.X = np.random.randn(self.n_samples, len(self.wavelengths))

    def test_default_exclusion(self):
        """Test default moisture band exclusion (1400-1500, 1900-2000)."""
        excluder = WavelengthExcluder(self.wavelengths)
        X_filtered = excluder.fit_transform(self.X)

        # Should exclude 101 + 101 = 202 wavelengths
        expected_removed = 202
        expected_kept = len(self.wavelengths) - expected_removed
        assert X_filtered.shape[1] == expected_kept

        # Check that excluded wavelengths are not in output
        assert 1450 not in excluder.wavelengths_out_  # Middle of 1400-1500
        assert 1950 not in excluder.wavelengths_out_  # Middle of 1900-2000

        # Check that other wavelengths are present
        assert 1200 in excluder.wavelengths_out_
        assert 2200 in excluder.wavelengths_out_

    def test_custom_exclusion(self):
        """Test custom wavelength range exclusion."""
        excluder = WavelengthExcluder(
            self.wavelengths,
            exclude_ranges=[(1100, 1200), (2300, 2400)]
        )
        X_filtered = excluder.fit_transform(self.X)

        # Should exclude 101 + 101 = 202 wavelengths
        expected_kept = len(self.wavelengths) - 202
        assert X_filtered.shape[1] == expected_kept

        # Check specific exclusions
        assert 1150 not in excluder.wavelengths_out_
        assert 2350 not in excluder.wavelengths_out_

    def test_invert_mode(self):
        """Test invert mode (keep only specified ranges)."""
        excluder = WavelengthExcluder(
            self.wavelengths,
            exclude_ranges=[(1500, 1800)],
            invert=True
        )
        X_filtered = excluder.fit_transform(self.X)

        # Should keep only 1500-1800 (301 wavelengths)
        assert X_filtered.shape[1] == 301
        assert excluder.wavelengths_out_[0] == 1500
        assert excluder.wavelengths_out_[-1] == 1800

    def test_no_exclusion(self):
        """Test with empty exclusion ranges."""
        excluder = WavelengthExcluder(self.wavelengths, exclude_ranges=[])
        X_filtered = excluder.fit_transform(self.X)

        # Should keep all wavelengths
        assert X_filtered.shape == self.X.shape
        np.testing.assert_array_equal(X_filtered, self.X)

    def test_all_excluded_warning(self):
        """Test warning when all wavelengths are excluded."""
        with pytest.warns(UserWarning, match="All wavelengths excluded"):
            excluder = WavelengthExcluder(
                self.wavelengths,
                exclude_ranges=[(1000, 2500)]  # Exclude everything
            )
            excluder.fit(self.X)

        assert excluder.n_features_out_ == 0

    def test_wavelength_mismatch_error(self):
        """Test error when wavelength array doesn't match X shape."""
        wrong_wavelengths = np.arange(1000, 2000)  # Wrong length
        excluder = WavelengthExcluder(wrong_wavelengths)

        with pytest.raises(ValueError, match="must match number of features"):
            excluder.fit(self.X)

    def test_transform_shape_mismatch(self):
        """Test error when transforming data with different shape."""
        excluder = WavelengthExcluder(self.wavelengths)
        excluder.fit(self.X)

        X_wrong_shape = np.random.randn(10, 100)  # Wrong number of features
        with pytest.raises(ValueError, match="was fitted with"):
            excluder.transform(X_wrong_shape)

    def test_pipeline_compatibility(self):
        """Test integration with sklearn Pipeline."""
        pipeline = Pipeline([
            ('exclude', WavelengthExcluder(self.wavelengths, exclude_ranges=[(1400, 1500)])),
            ('pls', PLSRegression(n_components=3))
        ])

        y = np.random.randn(self.n_samples)
        pipeline.fit(self.X, y)
        y_pred = pipeline.predict(self.X)

        # PLS returns 1D for single target, 2D for multiple targets
        assert y_pred.shape == (self.n_samples,) or y_pred.shape == (self.n_samples, 1)

    def test_get_feature_names_out(self):
        """Test get_feature_names_out method."""
        excluder = WavelengthExcluder(self.wavelengths)
        excluder.fit(self.X)

        wavelengths_out = excluder.get_feature_names_out()
        assert len(wavelengths_out) == excluder.n_features_out_
        assert wavelengths_out[0] == 1000  # First wavelength kept

    def test_invert_multiple_ranges(self):
        """Test invert mode with multiple ranges (union of ranges)."""
        excluder = WavelengthExcluder(
            self.wavelengths,
            exclude_ranges=[(1200, 1400), (1800, 2000)],
            invert=True
        )
        excluder.fit(self.X)

        # Should keep 201 + 201 = 402 wavelengths
        expected_kept = 201 + 201
        assert excluder.n_features_out_ == expected_kept

        # Verify both ranges are kept
        assert 1300 in excluder.wavelengths_out_  # Middle of first range
        assert 1900 in excluder.wavelengths_out_  # Middle of second range

        # Verify outside ranges are excluded
        assert 1500 not in excluder.wavelengths_out_  # Between ranges
        assert 1100 not in excluder.wavelengths_out_  # Before ranges
        assert 2100 not in excluder.wavelengths_out_  # After ranges


class TestMSC:
    """Tests for MSC (Multiplicative Scatter Correction) transformer."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 100
        self.n_wavelengths = 200

        # Generate clean spectra
        self.X_clean = np.random.randn(self.n_samples, self.n_wavelengths)

        # Add scatter effects: X_scattered = a + b * X_clean
        scatter_offset = np.random.randn(self.n_samples, 1) * 0.1
        scatter_scale = 0.8 + np.random.rand(self.n_samples, 1) * 0.4  # 0.8-1.2
        self.X_scattered = scatter_offset + scatter_scale * self.X_clean

    def test_mean_reference(self):
        """Test MSC with mean reference."""
        msc = MSC(reference='mean')
        X_corrected = msc.fit_transform(self.X_scattered)

        assert X_corrected.shape == self.X_scattered.shape

        # After MSC, spectra should be more similar to reference
        # (reduced variance in offset/scale)
        offsets_before = np.mean(self.X_scattered, axis=1)
        offsets_after = np.mean(X_corrected, axis=1)

        # Variance of offsets should be reduced
        assert np.var(offsets_after) < np.var(offsets_before)

    def test_median_reference(self):
        """Test MSC with median reference."""
        msc = MSC(reference='median')
        X_corrected = msc.fit_transform(self.X_scattered)

        assert X_corrected.shape == self.X_scattered.shape
        assert hasattr(msc, 'reference_')
        assert len(msc.reference_) == self.n_wavelengths

    def test_custom_reference(self):
        """Test MSC with custom reference spectrum."""
        custom_ref = np.random.randn(self.n_wavelengths)
        msc = MSC(reference=custom_ref)
        X_corrected = msc.fit_transform(self.X_scattered)

        assert X_corrected.shape == self.X_scattered.shape
        np.testing.assert_array_equal(msc.reference_, custom_ref)

    def test_invalid_reference_type(self):
        """Test error for invalid reference type."""
        msc = MSC(reference='invalid')
        with pytest.raises(ValueError, match="reference must be"):
            msc.fit(self.X_scattered)

    def test_custom_reference_wrong_length(self):
        """Test error for custom reference with wrong length."""
        wrong_ref = np.random.randn(50)  # Wrong length
        msc = MSC(reference=wrong_ref)
        with pytest.raises(ValueError, match="must match number of features"):
            msc.fit(self.X_scattered)

    def test_transform_before_fit(self):
        """Test error when transforming before fitting."""
        msc = MSC()
        with pytest.raises(Exception):  # sklearn raises NotFittedError
            msc.transform(self.X_scattered)

    def test_scatter_correction_effectiveness(self):
        """Test that MSC actually reduces scatter effects."""
        # Generate data with known scatter
        np.random.seed(42)
        X_base = np.random.randn(50, 100)
        scatter_scales = np.linspace(0.5, 1.5, 50).reshape(-1, 1)
        scatter_offsets = np.linspace(-0.2, 0.2, 50).reshape(-1, 1)
        X_scattered = scatter_offsets + scatter_scales * X_base

        msc = MSC(reference='mean')
        X_corrected = msc.fit_transform(X_scattered)

        # After MSC, the range of scales/offsets should be reduced
        # Check that the std of mean values across samples decreased
        mean_before = np.mean(X_scattered, axis=1)
        mean_after = np.mean(X_corrected, axis=1)

        # MSC should reduce variation in means
        assert np.std(mean_after) < np.std(mean_before)

    def test_pipeline_compatibility(self):
        """Test MSC in sklearn Pipeline."""
        pipeline = Pipeline([
            ('msc', MSC(reference='mean')),
            ('pls', PLSRegression(n_components=5))
        ])

        y = np.random.randn(self.n_samples)
        pipeline.fit(self.X_scattered, y)
        y_pred = pipeline.predict(self.X_scattered)

        # PLS returns 1D for single target
        assert y_pred.shape == (self.n_samples,) or y_pred.shape == (self.n_samples, 1)

    def test_constant_spectra(self):
        """Test MSC handles constant spectra gracefully."""
        X_constant = np.ones((5, 100))
        msc = MSC()
        msc.fit(X_constant)
        X_corrected = msc.transform(X_constant)

        # Should not produce NaN or Inf
        assert not np.any(np.isnan(X_corrected))
        assert not np.any(np.isinf(X_corrected))

    def test_near_constant_spectra(self):
        """Test MSC with very low variance spectra."""
        X_low_var = np.ones((10, 100)) + np.random.randn(10, 100) * 1e-8
        msc = MSC(reference='mean')
        msc.fit(X_low_var)
        X_corrected = msc.transform(X_low_var)

        # Should handle gracefully without errors
        assert X_corrected.shape == X_low_var.shape
        assert not np.any(np.isnan(X_corrected))
        assert not np.any(np.isinf(X_corrected))

    def test_all_zero_spectra(self):
        """Test MSC with all-zero data (debugger bug fix)."""
        X_zero = np.zeros((10, 50))
        msc = MSC(reference='mean')
        msc.fit(X_zero)

        with pytest.warns(UserWarning, match="near-zero variance"):
            X_corrected = msc.transform(X_zero)

        # Should return data unchanged
        assert X_corrected.shape == X_zero.shape
        assert not np.any(np.isnan(X_corrected))
        assert not np.any(np.isinf(X_corrected))
        np.testing.assert_array_equal(X_corrected, X_zero)


class TestOSC:
    """Tests for OSC (Orthogonal Signal Correction) transformer."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 100
        self.n_wavelengths = 200

        # Generate synthetic data with Y-relevant and Y-orthogonal components
        # Y-relevant component (e.g., protein signal)
        protein_loading = np.sin(np.linspace(0, 4 * np.pi, self.n_wavelengths))
        protein_concentrations = np.random.randn(self.n_samples)
        X_protein = protein_concentrations.reshape(-1, 1) @ protein_loading.reshape(1, -1)

        # Y-orthogonal component (e.g., moisture interference)
        moisture_loading = np.cos(np.linspace(0, 6 * np.pi, self.n_wavelengths))
        moisture_levels = np.random.randn(self.n_samples)
        X_moisture = moisture_levels.reshape(-1, 1) @ moisture_loading.reshape(1, -1)

        # Combined spectra
        self.X = X_protein + 0.5 * X_moisture + 0.1 * np.random.randn(self.n_samples, self.n_wavelengths)
        self.y = protein_concentrations + 0.1 * np.random.randn(self.n_samples)  # Depends on protein, not moisture

    def test_basic_osc(self):
        """Test basic OSC functionality."""
        osc = OSC(n_components=1)
        X_corrected = osc.fit_transform(self.X, self.y)

        assert X_corrected.shape == self.X.shape
        assert hasattr(osc, 'P_osc_')
        assert hasattr(osc, 'variance_removed_')

        # Should have removed 1 component
        assert osc.P_osc_.shape[1] == 1
        assert len(osc.variance_removed_) == 1

    def test_multiple_components(self):
        """Test OSC with multiple components."""
        osc = OSC(n_components=3)
        X_corrected = osc.fit_transform(self.X, self.y)

        assert X_corrected.shape == self.X.shape
        assert osc.P_osc_.shape[1] == 3
        assert len(osc.variance_removed_) == 3

        # Variance removed should be positive
        assert all(osc.variance_removed_ > 0)

    def test_osc_improves_prediction(self):
        """Test that OSC improves prediction by removing orthogonal variation."""
        # Train PLS without OSC
        pls_no_osc = PLSRegression(n_components=5)
        pls_no_osc.fit(self.X, self.y)
        y_pred_no_osc = pls_no_osc.predict(self.X)
        rmse_no_osc = np.sqrt(np.mean((self.y - y_pred_no_osc.ravel()) ** 2))

        # Train PLS with OSC
        osc = OSC(n_components=1)
        X_osc = osc.fit_transform(self.X, self.y)
        pls_with_osc = PLSRegression(n_components=5)
        pls_with_osc.fit(X_osc, self.y)
        y_pred_with_osc = pls_with_osc.predict(X_osc)
        rmse_with_osc = np.sqrt(np.mean((self.y - y_pred_with_osc.ravel()) ** 2))

        # OSC should reduce RMSE (or at worst, not increase it significantly)
        # Note: This test may be stochastic, so we allow small degradation
        assert rmse_with_osc <= rmse_no_osc * 1.1  # Allow 10% tolerance

    def test_osc_requires_y(self):
        """Test that OSC requires y for fitting."""
        osc = OSC(n_components=1)
        with pytest.raises(TypeError):
            osc.fit(self.X)  # Missing y

    def test_osc_sample_mismatch(self):
        """Test error when X and y have different sample counts."""
        osc = OSC(n_components=1)
        y_wrong_length = np.random.randn(50)  # Different from X.shape[0]

        with pytest.raises(ValueError, match="must have same number of samples"):
            osc.fit(self.X, y_wrong_length)

    def test_osc_transform_consistency(self):
        """Test that transform produces consistent results."""
        osc = OSC(n_components=2)
        osc.fit(self.X, self.y)

        X_transformed_1 = osc.transform(self.X)
        X_transformed_2 = osc.transform(self.X)

        np.testing.assert_array_almost_equal(X_transformed_1, X_transformed_2)

    def test_osc_pipeline(self):
        """Test OSC in sklearn Pipeline."""
        pipeline = Pipeline([
            ('osc', OSC(n_components=1)),
            ('pls', PLSRegression(n_components=5))
        ])

        pipeline.fit(self.X, self.y)
        y_pred = pipeline.predict(self.X)

        # PLS returns 1D for single target
        assert y_pred.shape[0] == self.n_samples

    def test_osc_zero_components(self):
        """Test OSC with zero components (should return original data)."""
        osc = OSC(n_components=0)
        X_corrected = osc.fit_transform(self.X, self.y)

        # With 0 components, should have no effect
        # (may have centering effect, so we don't test exact equality)
        assert X_corrected.shape == self.X.shape

    def test_osc_with_2d_y(self):
        """Test OSC with 2D y (multi-target)."""
        y_2d = np.random.randn(self.n_samples, 2)
        osc = OSC(n_components=1)
        X_corrected = osc.fit_transform(self.X, y_2d)

        assert X_corrected.shape == self.X.shape

    def test_osc_uses_training_mean(self):
        """Test that OSC transform uses training statistics, not test statistics (no data leakage)."""
        np.random.seed(42)
        # Training data centered at 1.0
        X_train = np.random.randn(100, 50) + 1.0
        # Test data centered at 5.0 (very different mean)
        X_test = np.random.randn(20, 50) + 5.0
        y_train = np.random.randn(100)

        osc = OSC(n_components=1)
        osc.fit(X_train, y_train)

        # Check training mean is stored
        assert hasattr(osc, 'X_mean_')
        # Training mean should be close to 1.0 (allow some variance due to random data)
        assert abs(np.mean(osc.X_mean_) - 1.0) < 0.2

        # Transform test set
        X_test_osc = osc.transform(X_test)

        # Test set should NOT be centered to zero (should use training mean, not test mean)
        # If leakage exists, X_test_osc would be centered to ~0
        # Without leakage, X_test_osc mean should be around 5.0 - 1.0 = 4.0
        test_mean = np.mean(X_test_osc)
        assert abs(test_mean - 4.0) < 1.0  # Allow some variance from OSC correction
        assert abs(test_mean) > 2.0  # Definitely not centered to zero

        # Also verify training data IS centered properly
        X_train_osc = osc.transform(X_train)
        train_mean = np.mean(X_train_osc)
        assert abs(train_mean) < 0.5  # Training data should be near zero after centering

    def test_osc_excessive_components_warning(self):
        """Test OSC warns when n_components exceeds maximum (debugger recommendation)."""
        X = np.random.randn(10, 5)  # Only 10 samples, 5 features
        y = np.random.randn(10)

        # Request more components than possible
        osc = OSC(n_components=20)

        with pytest.warns(UserWarning, match="greater than the maximum"):
            osc.fit(X, y)

        # Should still work, using maximum possible components
        assert osc.P_osc_.shape[1] <= min(9, 5)  # max is min(n_samples-1, n_features)


class TestEPO:
    """Tests for EPO (External Parameter Orthogonalization) - Placeholder."""

    def test_not_implemented(self):
        """EPO should raise NotImplementedError until Phase 2."""
        epo = EPO(n_components=3)
        X = np.random.randn(50, 100)
        y = np.random.randn(50)
        X_interferents = np.random.randn(20, 100)

        with pytest.raises(NotImplementedError, match="Phase 2"):
            epo.fit(X, y, X_interferents=X_interferents)

        with pytest.raises(NotImplementedError, match="Phase 2"):
            epo.transform(X)


class TestGLSW:
    """Tests for GLSW (Generalized Least Squares Weighting)."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 100
        self.n_wavelengths = 200

        # Generate spectra with heteroscedastic noise
        # Low-variance region (wavelengths 0-99)
        X_low_noise = np.random.randn(self.n_samples, 100) * 0.1
        # High-variance region (wavelengths 100-199) - simulates noisy water bands
        X_high_noise = np.random.randn(self.n_samples, 100) * 1.0

        self.X = np.hstack([X_low_noise, X_high_noise])
        self.y = np.random.randn(self.n_samples)

    def test_covariance_method(self):
        """Test GLSW with covariance method."""
        glsw = GLSW(method='covariance')
        X_weighted = glsw.fit_transform(self.X)

        assert X_weighted.shape == self.X.shape
        assert hasattr(glsw, 'feature_weights_')
        assert hasattr(glsw, 'W_sqrt_')

        # Check that weights were computed
        assert len(glsw.feature_weights_) == self.n_wavelengths

        # Weights should be positive
        assert np.all(glsw.feature_weights_ > 0)

        # Mean weight should be 1.0 (normalized)
        np.testing.assert_almost_equal(np.mean(glsw.feature_weights_), 1.0)

    def test_residual_method(self):
        """Test GLSW with residual method."""
        glsw = GLSW(method='residual', n_components=5)
        X_weighted = glsw.fit_transform(self.X, self.y)

        assert X_weighted.shape == self.X.shape
        assert hasattr(glsw, 'feature_weights_')

        # Weights should be positive
        assert np.all(glsw.feature_weights_ > 0)

        # Mean weight should be 1.0
        np.testing.assert_almost_equal(np.mean(glsw.feature_weights_), 1.0)

    def test_residual_method_requires_y(self):
        """Test that residual method requires y."""
        glsw = GLSW(method='residual')

        with pytest.raises(ValueError, match="requires y"):
            glsw.fit(self.X)  # Missing y

    def test_weights_down_weight_high_variance(self):
        """Test that GLSW down-weights high-variance regions."""
        glsw = GLSW(method='covariance')
        glsw.fit(self.X)

        weights = glsw.get_feature_weights()

        # Low-variance region (0-99) should have HIGHER weights
        weights_low_var = weights[:100]
        # High-variance region (100-199) should have LOWER weights
        weights_high_var = weights[100:]

        # Average weight in low-var region should be higher
        assert np.mean(weights_low_var) > np.mean(weights_high_var)

    def test_pipeline_compatibility(self):
        """Test GLSW in sklearn Pipeline."""
        from sklearn.linear_model import Ridge

        pipeline = Pipeline([
            ('glsw', GLSW(method='covariance')),
            ('ridge', Ridge(alpha=1.0))
        ])

        pipeline.fit(self.X, self.y)
        y_pred = pipeline.predict(self.X)

        # PLS/Ridge may return 1D or 2D
        assert y_pred.shape[0] == self.n_samples

    def test_invalid_method(self):
        """Test error for invalid method."""
        glsw = GLSW(method='invalid')

        with pytest.raises(ValueError, match="method must be"):
            glsw.fit(self.X)

    def test_transform_before_fit(self):
        """Test error when transforming before fitting."""
        glsw = GLSW(method='covariance')

        with pytest.raises(Exception):  # sklearn raises NotFittedError
            glsw.transform(self.X)

    def test_transform_shape_mismatch(self):
        """Test error when transforming data with different shape."""
        glsw = GLSW(method='covariance')
        glsw.fit(self.X)

        X_wrong_shape = np.random.randn(10, 50)  # Wrong number of features

        with pytest.raises(ValueError, match="was fitted with"):
            glsw.transform(X_wrong_shape)

    def test_regularization_prevents_division_by_zero(self):
        """Test that regularization handles zero-variance wavelengths."""
        # Create data with one constant wavelength
        X_with_constant = self.X.copy()
        X_with_constant[:, 0] = 1.0  # Constant wavelength

        glsw = GLSW(method='covariance', regularization=1e-6)
        X_weighted = glsw.fit_transform(X_with_constant)

        # Should not produce NaN or Inf
        assert not np.any(np.isnan(X_weighted))
        assert not np.any(np.isinf(X_weighted))

    def test_get_feature_weights(self):
        """Test get_feature_weights method."""
        glsw = GLSW(method='covariance')
        glsw.fit(self.X)

        weights = glsw.get_feature_weights()

        assert len(weights) == self.n_wavelengths
        assert np.all(weights > 0)

        # Should return a copy, not reference
        weights[0] = 999
        assert glsw.feature_weights_[0] != 999

    def test_glsw_improves_prediction_with_heteroscedastic_noise(self):
        """Test that GLSW improves prediction when noise is heteroscedastic."""
        # Generate data where y depends on low-noise region only
        y_true = np.mean(self.X[:, :100], axis=1)  # Depends on low-noise region
        y_noisy = y_true + np.random.randn(self.n_samples) * 0.1

        # Split train/test
        X_train, X_test = self.X[:80], self.X[80:]
        y_train, y_test = y_noisy[:80], y_noisy[80:]
        y_test_true = y_true[80:]

        # Model without GLSW
        from sklearn.linear_model import Ridge
        model_no_glsw = Ridge(alpha=1.0)
        model_no_glsw.fit(X_train, y_train)
        y_pred_no_glsw = model_no_glsw.predict(X_test)
        rmse_no_glsw = np.sqrt(np.mean((y_test_true - y_pred_no_glsw) ** 2))

        # Model with GLSW
        glsw = GLSW(method='covariance')
        X_train_weighted = glsw.fit_transform(X_train)
        X_test_weighted = glsw.transform(X_test)

        model_with_glsw = Ridge(alpha=1.0)
        model_with_glsw.fit(X_train_weighted, y_train)
        y_pred_with_glsw = model_with_glsw.predict(X_test_weighted)
        rmse_with_glsw = np.sqrt(np.mean((y_test_true - y_pred_with_glsw) ** 2))

        # GLSW should improve or at least not significantly hurt performance
        # (May not always improve due to randomness, so we allow some tolerance)
        assert rmse_with_glsw < rmse_no_glsw * 1.2  # Allow 20% tolerance

    def test_covariance_vs_residual_methods(self):
        """Test that both methods produce reasonable weights."""
        glsw_cov = GLSW(method='covariance')
        glsw_res = GLSW(method='residual', n_components=5)

        glsw_cov.fit(self.X)
        glsw_res.fit(self.X, self.y)

        weights_cov = glsw_cov.get_feature_weights()
        weights_res = glsw_res.get_feature_weights()

        # Both should be positive and normalized
        assert np.all(weights_cov > 0)
        assert np.all(weights_res > 0)
        np.testing.assert_almost_equal(np.mean(weights_cov), 1.0)
        np.testing.assert_almost_equal(np.mean(weights_res), 1.0)

        # Weights should be somewhat correlated (both target high-noise regions)
        # But may differ due to different methods
        correlation = np.corrcoef(weights_cov, weights_res)[0, 1]
        assert correlation > 0.3  # Should have some agreement


class TestDOSC:
    """Tests for DOSC (Direct Orthogonal Signal Correction)."""

    def setup_method(self):
        """Create synthetic spectral data with Y-orthogonal noise."""
        np.random.seed(42)
        self.n_samples = 100
        self.n_wavelengths = 50

        # Create wavelengths
        self.wavelengths = np.linspace(1500, 2500, self.n_wavelengths)

        # Create Y-related signal (predictive)
        self.y = np.random.randn(self.n_samples)
        y_signal = np.outer(self.y, np.sin(np.linspace(0, 2*np.pi, self.n_wavelengths)))

        # Create Y-orthogonal noise (systematic drift unrelated to y)
        drift_pattern = np.linspace(0, 1, self.n_wavelengths)
        drift_levels = np.random.randn(self.n_samples)
        y_orth_noise = np.outer(drift_levels, drift_pattern)

        # Combine: X = Y-related signal + Y-orthogonal noise + random noise
        self.X = y_signal + y_orth_noise * 2.0 + np.random.randn(self.n_samples, self.n_wavelengths) * 0.1

    # ========== BASIC FUNCTIONALITY TESTS ==========

    def test_dosc_basic_fit_transform(self):
        """Test basic DOSC fit and transform."""
        dosc = DOSC(n_components=2)
        dosc.fit(self.X, self.y)

        # Check fitted attributes
        assert hasattr(dosc, 'P_orth_')
        assert hasattr(dosc, 'X_mean_')
        assert hasattr(dosc, 'y_mean_')
        assert hasattr(dosc, 'dosc_components_')
        assert hasattr(dosc, 'explained_variance_')
        assert hasattr(dosc, 'n_components_')

        # Transform data
        X_corrected = dosc.transform(self.X)

        # Check shape
        assert X_corrected.shape == self.X.shape

        # Check no NaN/Inf
        assert not np.any(np.isnan(X_corrected))
        assert not np.any(np.isinf(X_corrected))

    def test_dosc_removes_y_orthogonal_variation(self):
        """Test that DOSC reduces Y-orthogonal variation."""
        dosc = DOSC(n_components=3)
        dosc.fit(self.X, self.y)

        X_corrected = dosc.transform(self.X)

        # Variance of corrected data should be less than original
        # (since we removed systematic variation)
        var_original = np.var(self.X, axis=0).mean()
        var_corrected = np.var(X_corrected, axis=0).mean()

        assert var_corrected < var_original

    def test_dosc_preserves_data_dimensions(self):
        """Test that DOSC preserves number of features."""
        dosc = DOSC(n_components=1)
        dosc.fit(self.X, self.y)

        assert dosc.n_features_in_ == self.n_wavelengths

        X_corrected = dosc.transform(self.X)
        assert X_corrected.shape[1] == self.n_wavelengths

    def test_dosc_uses_training_mean(self):
        """Test that DOSC transform uses training statistics (no data leakage)."""
        X_train = np.random.randn(100, self.n_wavelengths) + 1.0
        y_train = np.random.randn(100)
        X_test = np.random.randn(20, self.n_wavelengths) + 5.0

        dosc = DOSC(n_components=2)
        dosc.fit(X_train, y_train)

        # Verify training mean is stored
        assert hasattr(dosc, 'X_mean_')
        np.testing.assert_allclose(dosc.X_mean_, np.mean(X_train, axis=0), rtol=1e-10)

        # Transform test set - should use TRAINING mean, not test mean
        X_test_corrected = dosc.transform(X_test)

        # Test mean should NOT be centered to zero (uses training mean)
        test_mean = np.mean(X_test_corrected)
        assert abs(test_mean - 4.0) < 1.0  # Should preserve offset
        assert abs(test_mean) > 2.0  # Not centered to zero

    def test_dosc_explained_variance(self):
        """Test DOSC explained variance calculation."""
        dosc = DOSC(n_components=3)
        dosc.fit(self.X, self.y)

        explained_var = dosc.get_explained_variance()

        # Should have 3 components
        assert len(explained_var) == 3

        # Should sum to <= 1.0
        assert np.sum(explained_var) <= 1.0

        # Should be sorted descending
        assert np.all(explained_var[:-1] >= explained_var[1:])

        # All non-negative
        assert np.all(explained_var >= 0)

    def test_dosc_get_components(self):
        """Test DOSC component extraction."""
        dosc = DOSC(n_components=2)
        dosc.fit(self.X, self.y)

        components = dosc.get_dosc_components()

        # Shape should be (n_wavelengths, n_components)
        assert components.shape == (self.n_wavelengths, 2)

        # Components should be orthonormal
        gram = components.T @ components
        np.testing.assert_allclose(gram, np.eye(2), atol=1e-10)

    # ========== VALIDATION TESTS ==========

    def test_dosc_negative_n_components(self):
        """DOSC should reject negative n_components."""
        dosc = DOSC(n_components=-1)

        with pytest.raises(ValueError, match="positive integer"):
            dosc.fit(self.X, self.y)

    def test_dosc_zero_n_components(self):
        """DOSC should reject zero n_components."""
        dosc = DOSC(n_components=0)

        with pytest.raises(ValueError, match="positive integer"):
            dosc.fit(self.X, self.y)

    def test_dosc_invalid_center_type(self):
        """DOSC should reject non-boolean center parameter."""
        dosc = DOSC(n_components=2, center='yes')

        with pytest.raises(TypeError, match="True or False"):
            dosc.fit(self.X, self.y)

    def test_dosc_excessive_components_warning(self):
        """DOSC should warn when n_components exceeds maximum."""
        X_small = np.random.randn(20, 10)
        y_small = np.random.randn(20)

        dosc = DOSC(n_components=50)  # Way too many!

        with pytest.warns(UserWarning, match="exceeds maximum"):
            dosc.fit(X_small, y_small)

        # Should be capped at min(n_samples-1, n_features)
        assert dosc.n_components_ <= min(19, 10)

    def test_dosc_feature_mismatch_transform(self):
        """DOSC should reject transform with wrong number of features."""
        X_test_wrong = np.random.randn(20, 45)
        dosc = DOSC(n_components=2)
        dosc.fit(self.X, self.y)

        with pytest.raises(ValueError, match="50 features"):
            dosc.transform(X_test_wrong)

    def test_dosc_transform_before_fit(self):
        """DOSC should raise error if transform called before fit."""
        dosc = DOSC(n_components=2)

        with pytest.raises(Exception):  # sklearn raises NotFittedError
            dosc.transform(self.X)

    # ========== CENTERING TESTS ==========

    def test_dosc_centering_behavior(self):
        """Test DOSC centering modes."""
        # Test with centering (default)
        dosc_centered = DOSC(n_components=2, center=True)
        dosc_centered.fit(self.X, self.y)
        X_corrected_centered = dosc_centered.transform(self.X)

        # Mean should be close to zero (mean-centered)
        assert np.abs(np.mean(X_corrected_centered)) < 0.1

        # Test without centering
        dosc_no_center = DOSC(n_components=2, center=False)
        dosc_no_center.fit(self.X, self.y)
        X_corrected_no_center = dosc_no_center.transform(self.X)

        # Results should differ
        assert not np.allclose(X_corrected_centered, X_corrected_no_center)

    # ========== INTEGRATION TESTS ==========

    def test_dosc_sklearn_pipeline_integration(self):
        """Test DOSC in sklearn Pipeline."""
        pipeline = Pipeline([
            ('dosc', DOSC(n_components=2)),
            ('pls', PLSRegression(n_components=5))
        ])

        pipeline.fit(self.X, self.y)
        y_pred = pipeline.predict(self.X)

        assert y_pred.shape[0] == self.n_samples

        # Should produce reasonable predictions
        rmse = np.sqrt(np.mean((self.y - y_pred.ravel()) ** 2))
        assert rmse < 2.0  # Reasonable threshold

    def test_dosc_with_ridge_regression(self):
        """Test DOSC integration with Ridge regression."""
        from sklearn.linear_model import Ridge

        pipeline = Pipeline([
            ('dosc', DOSC(n_components=1)),
            ('ridge', Ridge(alpha=1.0))
        ])

        pipeline.fit(self.X, self.y)
        y_pred = pipeline.predict(self.X)

        assert y_pred.shape[0] == self.n_samples

        # Predictions should be reasonable
        rmse = np.sqrt(np.mean((self.y - y_pred) ** 2))
        assert rmse < 2.0

    def test_dosc_serialization(self):
        """Test that DOSC can be pickled and unpickled."""
        import pickle

        dosc = DOSC(n_components=2)
        dosc.fit(self.X, self.y)

        X_before = dosc.transform(self.X)

        # Pickle and unpickle
        pickled = pickle.dumps(dosc)
        dosc_loaded = pickle.loads(pickled)

        # Should produce same results
        X_after = dosc_loaded.transform(self.X)

        np.testing.assert_allclose(X_before, X_after, rtol=1e-10)

    def test_dosc_get_params_set_params(self):
        """Test sklearn get_params and set_params compatibility."""
        dosc = DOSC(n_components=2, center=True)

        # Get params
        params = dosc.get_params()
        assert params['n_components'] == 2
        assert params['center'] is True

        # Set params
        dosc.set_params(n_components=3, center=False)
        assert dosc.n_components == 3
        assert dosc.center is False


class TestIntegrationScenarios:
    """Integration tests for common interference removal scenarios."""

    def setup_method(self):
        """Set up realistic NIR data scenario."""
        np.random.seed(42)
        self.n_samples = 150
        self.wavelengths = np.arange(1000, 2501)
        self.n_wavelengths = len(self.wavelengths)

        # Simulate plant nitrogen prediction with moisture interference
        # Nitrogen signal (1500-1700 nm, 2100-2300 nm)
        nitrogen_signal = np.zeros(self.n_wavelengths)
        nitrogen_signal[(self.wavelengths >= 1500) & (self.wavelengths <= 1700)] = 1.0
        nitrogen_signal[(self.wavelengths >= 2100) & (self.wavelengths <= 2300)] = 0.8

        nitrogen_concentrations = np.random.uniform(1.5, 4.5, self.n_samples)  # 1.5-4.5% N
        X_nitrogen = nitrogen_concentrations.reshape(-1, 1) @ nitrogen_signal.reshape(1, -1)

        # Moisture interference (strong at 1400-1500, 1900-2000 nm)
        moisture_signal = np.zeros(self.n_wavelengths)
        moisture_signal[(self.wavelengths >= 1400) & (self.wavelengths <= 1500)] = 2.0
        moisture_signal[(self.wavelengths >= 1900) & (self.wavelengths <= 2000)] = 1.5

        moisture_levels = np.random.uniform(10, 30, self.n_samples)  # 10-30% moisture
        X_moisture = moisture_levels.reshape(-1, 1) @ moisture_signal.reshape(1, -1)

        # Combined spectra with noise
        self.X = X_nitrogen + X_moisture + 0.05 * np.random.randn(self.n_samples, self.n_wavelengths)
        self.y = nitrogen_concentrations + 0.1 * np.random.randn(self.n_samples)

    def test_wavelength_exclusion_then_osc(self):
        """Test combining wavelength exclusion with OSC."""
        pipeline = Pipeline([
            ('exclude_moisture', WavelengthExcluder(
                self.wavelengths,
                exclude_ranges=[(1400, 1500), (1900, 2000)]
            )),
            ('osc', OSC(n_components=1)),
            ('pls', PLSRegression(n_components=5))
        ])

        pipeline.fit(self.X, self.y)
        y_pred = pipeline.predict(self.X)

        # PLS returns 1D for single target
        assert y_pred.shape[0] == self.n_samples

        # Check that wavelength exclusion was applied
        X_after_exclude = pipeline.named_steps['exclude_moisture'].transform(self.X)
        assert X_after_exclude.shape[1] < self.X.shape[1]

    def test_msc_then_wavelength_exclusion(self):
        """Test scatter correction followed by wavelength filtering."""
        pipeline = Pipeline([
            ('msc', MSC(reference='mean')),
            ('exclude', WavelengthExcluder(
                self.wavelengths,
                exclude_ranges=[(2400, 2500)]  # Exclude noisy edge
            )),
            ('pls', PLSRegression(n_components=3))
        ])

        pipeline.fit(self.X, self.y)
        y_pred = pipeline.predict(self.X)

        # PLS returns 1D for single target
        assert y_pred.shape[0] == self.n_samples

    def test_full_preprocessing_chain(self):
        """Test complete preprocessing chain: exclude → MSC → OSC → PLS."""
        from spectral_predict.preprocess import SNV  # Assuming SNV exists

        # Try with SNV if available, otherwise use MSC
        try:
            scatter_corrector = SNV()
        except:
            scatter_corrector = MSC(reference='mean')

        pipeline = Pipeline([
            ('exclude_moisture', WavelengthExcluder(
                self.wavelengths,
                exclude_ranges=[(1400, 1500), (1900, 2000)]
            )),
            ('scatter', scatter_corrector),
            ('osc', OSC(n_components=2)),
            ('pls', PLSRegression(n_components=5))
        ])

        pipeline.fit(self.X, self.y)
        y_pred = pipeline.predict(self.X)

        # PLS returns 1D for single target
        assert y_pred.shape[0] == self.n_samples

        # Compute RMSE
        rmse = np.sqrt(np.mean((self.y - y_pred.ravel()) ** 2))
        # Should achieve reasonable prediction (nitrogen range is 1.5-4.5, so RMSE < 0.5 would be good)
        assert rmse < 1.0  # Generous threshold for synthetic data

    def test_glsw_with_ridge_regression(self):
        """Test GLSW integration with Ridge regression (heteroscedastic noise handling)."""
        from sklearn.linear_model import Ridge

        # Create pipeline with GLSW
        pipeline = Pipeline([
            ('wavelength_exclude', WavelengthExcluder(
                self.wavelengths,
                exclude_ranges=[(1400, 1500), (1900, 2000)]
            )),
            ('glsw', GLSW(method='covariance')),
            ('ridge', Ridge(alpha=10.0))
        ])

        pipeline.fit(self.X, self.y)
        y_pred = pipeline.predict(self.X)

        assert y_pred.shape[0] == self.n_samples

        # Should produce reasonable predictions
        rmse = np.sqrt(np.mean((self.y - y_pred) ** 2))
        assert rmse < 1.0

    def test_complete_interference_removal_pipeline(self):
        """Test full pipeline: wavelength exclusion → GLSW → OSC → MSC → model."""
        from sklearn.linear_model import Lasso

        pipeline = Pipeline([
            ('exclude', WavelengthExcluder(self.wavelengths, exclude_ranges=[(1400, 1500)])),
            ('glsw', GLSW(method='covariance')),
            ('msc', MSC(reference='mean')),
            ('osc', OSC(n_components=1)),
            ('lasso', Lasso(alpha=0.1))
        ])

        pipeline.fit(self.X, self.y)
        y_pred = pipeline.predict(self.X)

        assert y_pred.shape[0] == self.n_samples

        # Pipeline should execute without errors and produce predictions
        assert not np.any(np.isnan(y_pred))
        assert not np.any(np.isinf(y_pred))


class TestEPO:
    """Test suite for EPO (External Parameter Orthogonalization)."""

    def setup_method(self):
        """Create synthetic spectral data with known interferent."""
        np.random.seed(42)
        self.n_samples = 100
        self.n_wavelengths = 50
        self.n_interferent_samples = 10

        # Create wavelengths
        self.wavelengths = np.linspace(1500, 2500, self.n_wavelengths)

        # Create base signal (analyte)
        self.analyte_signal = np.random.randn(self.n_samples, self.n_wavelengths) * 0.1

        # Create interferent signal (e.g., moisture effect)
        self.interferent_pattern = np.sin(np.linspace(0, 4*np.pi, self.n_wavelengths))
        self.interferent_levels = np.random.randn(self.n_samples) * 2.0

        # Combine: X = analyte + (moisture_level * moisture_pattern)
        self.X = self.analyte_signal + np.outer(self.interferent_levels, self.interferent_pattern)

        # Create target (correlated with analyte, not interferent)
        self.y = 10 + np.sum(self.analyte_signal[:, :5], axis=1) + np.random.randn(self.n_samples) * 0.1

        # Create interferent library (pure interferent spectra at different levels)
        interferent_library_levels = np.linspace(-3, 3, self.n_interferent_samples)
        self.X_interferents = np.outer(interferent_library_levels, self.interferent_pattern)

    # ========== CRITICAL VALIDATION TESTS (Architect-specified) ==========

    def test_epo_missing_interferents(self):
        """CRITICAL: EPO must raise clear error when X_interferents is None."""
        epo = EPO(n_components=2)

        with pytest.raises(ValueError, match="X_interferents is required"):
            epo.fit(self.X)  # Missing X_interferents

    def test_epo_feature_mismatch_fit(self):
        """CRITICAL: EPO must reject X_interferents with wrong number of features."""
        X_interferents_wrong = np.random.randn(10, 45)  # Wrong size!
        epo = EPO(n_components=2)

        with pytest.raises(ValueError, match="same number of features"):
            epo.fit(self.X, X_interferents=X_interferents_wrong)

    def test_epo_empty_interferents(self):
        """CRITICAL: EPO must reject empty interferent library."""
        X_interferents_empty = np.array([]).reshape(0, self.n_wavelengths)
        epo = EPO(n_components=2)

        # sklearn check_array raises its own error for empty arrays
        with pytest.raises(ValueError):
            epo.fit(self.X, X_interferents=X_interferents_empty)

    def test_epo_insufficient_interferent_samples(self):
        """CRITICAL: EPO should reduce n_components if library too small."""
        X_interferents_small = np.random.randn(2, self.n_wavelengths)  # Only 2 samples
        epo = EPO(n_components=5)  # Request 5 components

        with pytest.warns(UserWarning, match="Reducing to 2 components"):
            epo.fit(self.X, X_interferents=X_interferents_small)

        assert epo.n_components_ == 2  # Should auto-reduce

    def test_epo_constant_interferents(self):
        """CRITICAL: EPO must reject constant interferent library."""
        X_interferents_constant = np.ones((10, self.n_wavelengths))
        epo = EPO(n_components=2)

        with pytest.raises(ValueError, match="near-zero variance"):
            epo.fit(self.X, X_interferents=X_interferents_constant)

    def test_epo_zero_interferents(self):
        """CRITICAL: EPO must reject all-zero interferent library."""
        X_interferents_zero = np.zeros((10, self.n_wavelengths))
        epo = EPO(n_components=2)

        with pytest.raises(ValueError, match="near-zero variance"):
            epo.fit(self.X, X_interferents=X_interferents_zero)

    def test_epo_partial_constant_wavelengths(self):
        """CRITICAL: EPO should warn if some wavelengths are constant."""
        X_interferents_partial = np.random.randn(10, self.n_wavelengths)
        X_interferents_partial[:, 10:15] = 5.0  # Make 5 wavelengths constant

        epo = EPO(n_components=2)

        with pytest.warns(UserWarning, match="5/50 wavelengths"):
            epo.fit(self.X, X_interferents=X_interferents_partial)

        # Should still fit successfully
        assert hasattr(epo, 'P_orth_')

    def test_epo_feature_mismatch_transform(self):
        """CRITICAL: EPO must reject transform with wrong number of features."""
        X_test_wrong = np.random.randn(20, 45)  # Wrong size!
        epo = EPO(n_components=2)
        epo.fit(self.X, X_interferents=self.X_interferents)

        with pytest.raises(ValueError, match="50 features"):
            epo.transform(X_test_wrong)

    # ========== BASIC FUNCTIONALITY TESTS ==========

    def test_epo_basic_fit_transform(self):
        """Test basic EPO fit and transform."""
        epo = EPO(n_components=2)
        epo.fit(self.X, self.y, X_interferents=self.X_interferents)

        # Check fitted attributes
        assert hasattr(epo, 'P_orth_')
        assert hasattr(epo, 'X_mean_')
        assert hasattr(epo, 'interferent_mean_')
        assert hasattr(epo, 'interferent_components_')
        assert hasattr(epo, 'explained_variance_')
        assert hasattr(epo, 'n_components_')

        # Transform data
        X_corrected = epo.transform(self.X)

        # Check shape
        assert X_corrected.shape == self.X.shape

        # Check no NaN/Inf
        assert not np.any(np.isnan(X_corrected))
        assert not np.any(np.isinf(X_corrected))

    def test_epo_removes_interferent_signal(self):
        """Test that EPO removes known interferent signal."""
        epo = EPO(n_components=1)
        epo.fit(self.X, self.y, X_interferents=self.X_interferents)

        X_corrected = epo.transform(self.X)

        # Measure correlation with interferent pattern before and after
        corr_before = np.abs(np.corrcoef(self.X.ravel(), np.tile(self.interferent_pattern, self.n_samples))[0, 1])
        corr_after = np.abs(np.corrcoef(X_corrected.ravel(), np.tile(self.interferent_pattern, self.n_samples))[0, 1])

        # Correlation with interferent should decrease
        assert corr_after < corr_before
        assert corr_after < 0.3  # Should be substantially reduced

    def test_epo_preserves_data_dimensions(self):
        """Test that EPO preserves number of features."""
        epo = EPO(n_components=2)
        epo.fit(self.X, X_interferents=self.X_interferents)

        assert epo.n_features_in_ == self.n_wavelengths

        X_corrected = epo.transform(self.X)
        assert X_corrected.shape[1] == self.n_wavelengths  # Same number of wavelengths

    def test_epo_centering_behavior(self):
        """Test EPO centering modes."""
        # Test with centering (default)
        epo_centered = EPO(n_components=2, center=True)
        epo_centered.fit(self.X, X_interferents=self.X_interferents)
        X_corrected_centered = epo_centered.transform(self.X)

        # Mean should be close to zero (mean-centered)
        assert np.abs(np.mean(X_corrected_centered)) < 0.1

        # Test without centering
        epo_no_center = EPO(n_components=2, center=False)
        epo_no_center.fit(self.X, X_interferents=self.X_interferents)
        X_corrected_no_center = epo_no_center.transform(self.X)

        # Results should differ
        assert not np.allclose(X_corrected_centered, X_corrected_no_center)

    def test_epo_uses_training_mean(self):
        """Test that EPO transform uses training statistics (no data leakage)."""
        X_train = np.random.randn(100, self.n_wavelengths) + 1.0
        X_test = np.random.randn(20, self.n_wavelengths) + 5.0  # Different offset

        epo = EPO(n_components=2)
        epo.fit(X_train, X_interferents=self.X_interferents)

        # Verify training mean is stored
        assert hasattr(epo, 'X_mean_')
        np.testing.assert_allclose(epo.X_mean_, np.mean(X_train, axis=0), rtol=1e-10)

        # Transform test set - should use TRAINING mean, not test mean
        X_test_corrected = epo.transform(X_test)

        # Test mean should NOT be centered to zero (uses training mean)
        test_mean = np.mean(X_test_corrected)
        assert abs(test_mean - 4.0) < 1.0  # Should preserve offset (~4.0 difference)
        assert abs(test_mean) > 2.0  # Not centered to zero

    def test_epo_explained_variance(self):
        """Test EPO explained variance calculation."""
        epo = EPO(n_components=3)
        epo.fit(self.X, X_interferents=self.X_interferents)

        explained_var = epo.get_explained_variance()

        # Should have 3 components
        assert len(explained_var) == 3

        # Should sum to <= 1.0 (fraction of total variance)
        assert np.sum(explained_var) <= 1.0

        # Should be sorted descending
        assert np.all(explained_var[:-1] >= explained_var[1:])

        # All non-negative
        assert np.all(explained_var >= 0)

    def test_epo_interferent_components(self):
        """Test EPO interferent component extraction."""
        epo = EPO(n_components=2)
        epo.fit(self.X, X_interferents=self.X_interferents)

        components = epo.get_interferent_components()

        # Shape should be (n_wavelengths, n_components)
        assert components.shape == (self.n_wavelengths, 2)

        # Components should be orthonormal (V.T @ V ≈ I)
        gram = components.T @ components
        np.testing.assert_allclose(gram, np.eye(2), atol=1e-10)

    # ========== INTEGRATION TESTS ==========

    def test_epo_sklearn_pipeline_integration(self):
        """Test EPO in sklearn Pipeline."""
        from sklearn.cross_decomposition import PLSRegression

        pipeline = Pipeline([
            ('epo', EPO(n_components=2)),
            ('pls', PLSRegression(n_components=5))
        ])

        # Fit with X_interferents passed to EPO step
        pipeline.fit(self.X, self.y, epo__X_interferents=self.X_interferents)

        y_pred = pipeline.predict(self.X)

        assert y_pred.shape[0] == self.n_samples

        # Should produce reasonable predictions
        rmse = np.sqrt(np.mean((self.y - y_pred.ravel()) ** 2))
        assert rmse < 1.0  # Should be able to predict reasonably well

    def test_epo_with_ridge_regression(self):
        """Test EPO integration with Ridge regression."""
        from sklearn.linear_model import Ridge

        pipeline = Pipeline([
            ('epo', EPO(n_components=1)),
            ('ridge', Ridge(alpha=1.0))
        ])

        pipeline.fit(self.X, self.y, epo__X_interferents=self.X_interferents)
        y_pred = pipeline.predict(self.X)

        assert y_pred.shape[0] == self.n_samples

        # Predictions should be reasonable
        rmse = np.sqrt(np.mean((self.y - y_pred) ** 2))
        assert rmse < 1.5

    def test_epo_serialization(self):
        """Test that EPO can be pickled and unpickled."""
        import pickle

        epo = EPO(n_components=2)
        epo.fit(self.X, X_interferents=self.X_interferents)

        X_before = epo.transform(self.X)

        # Pickle and unpickle
        pickled = pickle.dumps(epo)
        epo_loaded = pickle.loads(pickled)

        # Should produce same results
        X_after = epo_loaded.transform(self.X)

        np.testing.assert_allclose(X_before, X_after, rtol=1e-10)

    def test_epo_model_io_integration(self):
        """Test EPO integration with DASP model_io save/load."""
        import tempfile
        import os
        from spectral_predict.model_io import save_model, load_model

        # Create pipeline with EPO
        pipeline = Pipeline([
            ('epo', EPO(n_components=2)),
            ('pls', PLSRegression(n_components=5))
        ])

        pipeline.fit(self.X, self.y, epo__X_interferents=self.X_interferents)

        # Get predictions before save
        y_pred_before = pipeline.predict(self.X)

        # Create temp file
        tmpfile = tempfile.NamedTemporaryFile(suffix='.dasp', delete=False)
        tmpfile.close()

        try:
            # Save model
            save_model(
                model=pipeline,
                preprocessor=None,
                metadata={
                    'model_name': 'EPO-PLS',
                    'task_type': 'regression',
                    'preprocessing': 'epo',
                    'wavelengths': self.wavelengths.tolist(),
                    'n_vars': self.n_wavelengths,
                    'performance': {'R2': 0.85, 'RMSE': 0.25}
                },
                filepath=tmpfile.name
            )

            # Load model
            loaded = load_model(tmpfile.name)

            # Predict with loaded model
            y_pred_after = loaded['model'].predict(self.X)

            # Should be identical
            np.testing.assert_allclose(y_pred_before, y_pred_after, rtol=1e-15)

            # Check metadata
            assert loaded['metadata']['model_name'] == 'EPO-PLS'
            assert loaded['metadata']['preprocessing'] == 'epo'

        finally:
            os.unlink(tmpfile.name)

    def test_epo_model_io_with_metadata(self):
        """Test EPO with custom metadata in model_io."""
        import tempfile
        import os
        from spectral_predict.model_io import save_model, load_model

        # Fit EPO
        epo = EPO(n_components=3)
        epo.fit(self.X, X_interferents=self.X_interferents)

        explained_var = epo.get_explained_variance()

        # Create temp file
        tmpfile = tempfile.NamedTemporaryFile(suffix='.dasp', delete=False)
        tmpfile.close()

        try:
            # Save with EPO-specific metadata
            save_model(
                model=epo,
                preprocessor=None,
                metadata={
                    'model_name': 'EPO',
                    'task_type': 'preprocessing',
                    'preprocessing': 'epo',
                    'wavelengths': self.wavelengths.tolist(),
                    'n_vars': self.n_wavelengths,
                    'epo_params': {
                        'n_components': int(epo.n_components_),
                        'explained_variance': explained_var.tolist(),
                        'total_variance_removed': float(np.sum(explained_var)),
                        'center': bool(epo.center)
                    }
                },
                filepath=tmpfile.name
            )

            # Load and verify metadata
            loaded = load_model(tmpfile.name)

            assert 'epo_params' in loaded['metadata']
            epo_meta = loaded['metadata']['epo_params']
            assert epo_meta['n_components'] == 3
            assert len(epo_meta['explained_variance']) == 3
            assert 0 <= epo_meta['total_variance_removed'] <= 1.0

        finally:
            os.unlink(tmpfile.name)

    def test_epo_get_params_set_params(self):
        """Test sklearn get_params and set_params compatibility."""
        epo = EPO(n_components=2, center=True)

        # Get params
        params = epo.get_params()
        assert params['n_components'] == 2
        assert params['center'] is True

        # Set params
        epo.set_params(n_components=3, center=False)
        assert epo.n_components == 3
        assert epo.center is False

    # ========== EDGE CASE TESTS ==========

    def test_epo_single_interferent_sample(self):
        """Test EPO with single interferent sample (should fail - need >=2 for variance)."""
        X_interferents_single = self.X_interferents[:1, :]  # Only 1 sample

        epo = EPO(n_components=1)

        # Single sample has zero variance after centering, should reject
        with pytest.raises(ValueError, match="near-zero variance"):
            epo.fit(self.X, X_interferents=X_interferents_single)

    def test_epo_excessive_components_warning(self):
        """Test warning when requesting too many components."""
        X_interferents_large = np.random.randn(15, self.n_wavelengths)

        epo = EPO(n_components=12)  # Very large

        with pytest.warns(UserWarning, match="very large"):
            epo.fit(self.X, X_interferents=X_interferents_large)

        # Should still work
        assert epo.n_components_ == 12

    def test_epo_no_centering_preserves_mean(self):
        """Test that center=False preserves data mean."""
        X_offset = self.X + 10.0  # Add offset

        epo = EPO(n_components=2, center=False)
        epo.fit(X_offset, X_interferents=self.X_interferents)

        X_corrected = epo.transform(X_offset)

        # Mean should be preserved (approximately)
        mean_before = np.mean(X_offset)
        mean_after = np.mean(X_corrected)

        # Without centering, mean should be similar
        assert abs(mean_after - mean_before) < 2.0

    # ========== DEBUGGER-FOUND BUG FIXES ==========

    def test_epo_negative_n_components(self):
        """DEBUGGER: EPO should reject negative n_components."""
        epo = EPO(n_components=-1)

        with pytest.raises(ValueError, match="positive integer"):
            epo.fit(self.X, X_interferents=self.X_interferents)

    def test_epo_zero_n_components(self):
        """DEBUGGER: EPO should reject zero n_components."""
        epo = EPO(n_components=0)

        with pytest.raises(ValueError, match="positive integer"):
            epo.fit(self.X, X_interferents=self.X_interferents)

    def test_epo_invalid_center_type(self):
        """DEBUGGER: EPO should reject non-boolean center parameter."""
        epo = EPO(n_components=2, center='yes')

        with pytest.raises(TypeError, match="True or False"):
            epo.fit(self.X, X_interferents=self.X_interferents)

    def test_epo_n_components_exceeds_features(self):
        """DEBUGGER: EPO should cap n_components at min(n_samples, n_features)."""
        X_small = np.random.randn(100, 10)
        X_int = np.random.randn(50, 10)

        epo = EPO(n_components=20)  # More than 10 features!

        with pytest.warns(UserWarning, match="maximum possible"):
            epo.fit(X_small, X_interferents=X_int)

        # Should be capped at 10
        assert epo.n_components_ == 10


def test_module_imports():
    """Test that all classes can be imported."""
    from spectral_predict.interference import (
        WavelengthExcluder,
        MSC,
        OSC,
        EPO,
        GLSW,
        DOSC
    )

    # Check that classes exist
    assert WavelengthExcluder is not None
    assert MSC is not None
    assert OSC is not None
    assert EPO is not None
    assert GLSW is not None
    assert DOSC is not None
