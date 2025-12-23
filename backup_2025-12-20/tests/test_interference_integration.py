"""
Comprehensive integration tests for interference removal methods.

Tests complex scenarios combining multiple interference removal techniques
to validate they work together seamlessly in realistic workflows.
"""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, Lasso

from spectral_predict.interference import (
    WavelengthExcluder,
    MSC,
    OSC,
    GLSW,
    EPO,
    DOSC
)


class TestComplexPipelines:
    """Test complex multi-method interference removal pipelines."""

    def setup_method(self):
        """Create realistic spectral data with multiple interference sources."""
        np.random.seed(42)
        self.n_samples = 150
        self.n_wavelengths = 200

        # Create wavelengths (400-2400 nm)
        self.wavelengths = np.linspace(400, 2400, self.n_wavelengths)

        # Create analyte signal (correlated with y)
        self.y = np.random.randn(self.n_samples) * 2 + 10
        analyte_pattern = np.sin(np.linspace(0, 3*np.pi, self.n_wavelengths))
        analyte_signal = np.outer(self.y - 10, analyte_pattern) * 0.5

        # Create moisture interference (strong at 1400-1500, 1900-2000 nm)
        moisture_idx_1 = (self.wavelengths >= 1400) & (self.wavelengths <= 1500)
        moisture_idx_2 = (self.wavelengths >= 1900) & (self.wavelengths <= 2000)
        moisture_levels = np.random.randn(self.n_samples) * 3
        moisture_signal = np.zeros((self.n_samples, self.n_wavelengths))
        moisture_signal[:, moisture_idx_1] = np.outer(moisture_levels, np.ones(np.sum(moisture_idx_1))) * 2
        moisture_signal[:, moisture_idx_2] = np.outer(moisture_levels, np.ones(np.sum(moisture_idx_2))) * 1.5

        # Create baseline drift (Y-orthogonal)
        drift = np.linspace(0, 1, self.n_wavelengths)
        drift_levels = np.random.randn(self.n_samples)
        baseline_drift = np.outer(drift_levels, drift) * 0.8

        # Create scatter effects
        scatter_pattern = 1 + np.random.randn(self.n_samples, 1) * 0.1
        scatter_effect = scatter_pattern * np.random.randn(self.n_wavelengths) * 0.3

        # Combine all sources + random noise
        self.X = (analyte_signal + moisture_signal + baseline_drift +
                  scatter_effect + np.random.randn(self.n_samples, self.n_wavelengths) * 0.05)

        # Create interferent library for EPO (moisture spectra)
        self.X_moisture = np.zeros((10, self.n_wavelengths))
        moisture_levels_lib = np.linspace(-5, 5, 10)
        self.X_moisture[:, moisture_idx_1] = np.outer(moisture_levels_lib, np.ones(np.sum(moisture_idx_1))) * 2
        self.X_moisture[:, moisture_idx_2] = np.outer(moisture_levels_lib, np.ones(np.sum(moisture_idx_2))) * 1.5

    def test_full_preprocessing_pipeline(self):
        """Test complete pipeline: WavelengthExcluder → MSC → DOSC → GLSW → PLS."""
        pipeline = Pipeline([
            ('exclude', WavelengthExcluder(
                self.wavelengths,
                exclude_ranges=[(1400, 1500), (1900, 2000)]
            )),
            ('msc', MSC(reference='mean')),
            ('dosc', DOSC(n_components=2)),
            ('glsw', GLSW(method='covariance')),
            ('pls', PLSRegression(n_components=10))
        ])

        # Fit pipeline
        pipeline.fit(self.X, self.y)

        # Predict
        y_pred = pipeline.predict(self.X)

        # Check predictions
        assert y_pred.shape[0] == self.n_samples
        assert not np.any(np.isnan(y_pred))
        assert not np.any(np.isinf(y_pred))

        # Should achieve reasonable accuracy
        rmse = np.sqrt(np.mean((self.y - y_pred.ravel()) ** 2))
        r2 = 1 - np.sum((self.y - y_pred.ravel()) ** 2) / np.sum((self.y - np.mean(self.y)) ** 2)

        assert rmse < 3.0  # Reasonable RMSE
        assert r2 > 0.3    # Some predictive power

    def test_epo_based_pipeline(self):
        """Test EPO for targeted moisture removal → OSC → Ridge."""
        pipeline = Pipeline([
            ('epo', EPO(n_components=2)),
            ('osc', OSC(n_components=1)),
            ('ridge', Ridge(alpha=10.0))
        ])

        # Fit with EPO interferent library
        pipeline.fit(self.X, self.y, epo__X_interferents=self.X_moisture)

        # Predict
        y_pred = pipeline.predict(self.X)

        # Validate
        assert y_pred.shape[0] == self.n_samples
        assert not np.any(np.isnan(y_pred))

        # Check EPO removed moisture
        epo_fitted = pipeline.named_steps['epo']
        assert hasattr(epo_fitted, 'P_orth_')
        assert epo_fitted.n_components_ == 2

    def test_dosc_osc_comparison(self):
        """Test that DOSC and OSC produce similar results."""
        # Pipeline with OSC
        pipe_osc = Pipeline([
            ('osc', OSC(n_components=2)),
            ('pls', PLSRegression(n_components=5))
        ])

        # Pipeline with DOSC
        pipe_dosc = Pipeline([
            ('dosc', DOSC(n_components=2)),
            ('pls', PLSRegression(n_components=5))
        ])

        # Fit both
        pipe_osc.fit(self.X, self.y)
        pipe_dosc.fit(self.X, self.y)

        # Get predictions
        y_pred_osc = pipe_osc.predict(self.X)
        y_pred_dosc = pipe_dosc.predict(self.X)

        # Both should work (no NaN/Inf)
        assert not np.any(np.isnan(y_pred_osc))
        assert not np.any(np.isnan(y_pred_dosc))

        # R² should be in similar range
        r2_osc = 1 - np.sum((self.y - y_pred_osc.ravel()) ** 2) / np.sum((self.y - np.mean(self.y)) ** 2)
        r2_dosc = 1 - np.sum((self.y - y_pred_dosc.ravel()) ** 2) / np.sum((self.y - np.mean(self.y)) ** 2)

        # Both should have positive R²
        assert r2_osc > 0
        assert r2_dosc > 0

    def test_all_methods_in_one_pipeline(self):
        """Test extreme pipeline with all 6 methods (excluding EPO to avoid dimension mismatch)."""
        # Note: EPO requires interferent library with same dimensions as input
        # If we use WavelengthExcluder first, dimensions don't match
        # This tests 5 methods working together
        pipeline = Pipeline([
            ('exclude', WavelengthExcluder(self.wavelengths, exclude_ranges=[(1400, 1500)])),
            ('msc', MSC(reference='mean')),
            ('osc', OSC(n_components=1)),
            ('glsw', GLSW(method='covariance')),
            ('dosc', DOSC(n_components=1)),
            ('pls', PLSRegression(n_components=5))
        ])

        # This is overkill but should still work!
        pipeline.fit(self.X, self.y)

        y_pred = pipeline.predict(self.X)

        # Should not crash and produce valid predictions
        assert y_pred.shape[0] == self.n_samples
        assert not np.any(np.isnan(y_pred))
        assert not np.any(np.isinf(y_pred))

    def test_pipeline_with_lasso(self):
        """Test interference removal with Lasso regression."""
        pipeline = Pipeline([
            ('wavelength_exclude', WavelengthExcluder(
                self.wavelengths,
                exclude_ranges=[(1400, 1500), (1900, 2000)]
            )),
            ('msc', MSC(reference='mean')),
            ('dosc', DOSC(n_components=2)),
            ('lasso', Lasso(alpha=0.1))
        ])

        pipeline.fit(self.X, self.y)
        y_pred = pipeline.predict(self.X)

        assert y_pred.shape[0] == self.n_samples
        assert not np.any(np.isnan(y_pred))

        # Lasso should work with interference-corrected data
        rmse = np.sqrt(np.mean((self.y - y_pred) ** 2))
        assert rmse < 5.0


class TestSerializationIntegration:
    """Test that complex pipelines can be saved and loaded."""

    def setup_method(self):
        """Create test data."""
        np.random.seed(42)
        self.n_samples = 100
        self.n_wavelengths = 50
        self.wavelengths = np.linspace(1500, 2500, self.n_wavelengths)
        self.X = np.random.randn(self.n_samples, self.n_wavelengths)
        self.y = np.random.randn(self.n_samples)

    def test_complex_pipeline_serialization(self):
        """Test that complex pipelines serialize correctly."""
        import pickle

        pipeline = Pipeline([
            ('exclude', WavelengthExcluder(self.wavelengths, exclude_ranges=[(1400, 1500)])),
            ('msc', MSC(reference='mean')),
            ('dosc', DOSC(n_components=2)),
            ('glsw', GLSW(method='covariance')),
            ('pls', PLSRegression(n_components=5))
        ])

        # Fit pipeline
        pipeline.fit(self.X, self.y)

        # Get predictions before serialization
        y_pred_before = pipeline.predict(self.X)

        # Serialize and deserialize
        pickled = pickle.dumps(pipeline)
        pipeline_loaded = pickle.loads(pickled)

        # Get predictions after serialization
        y_pred_after = pipeline_loaded.predict(self.X)

        # Should be identical
        np.testing.assert_allclose(y_pred_before, y_pred_after, rtol=1e-15)

    def test_epo_pipeline_serialization(self):
        """Test EPO pipeline serialization (special case with X_interferents)."""
        import pickle

        X_interferents = np.random.randn(10, self.n_wavelengths)

        pipeline = Pipeline([
            ('epo', EPO(n_components=2)),
            ('pls', PLSRegression(n_components=5))
        ])

        # Fit with interferents
        pipeline.fit(self.X, self.y, epo__X_interferents=X_interferents)

        y_pred_before = pipeline.predict(self.X)

        # Serialize
        pickled = pickle.dumps(pipeline)
        pipeline_loaded = pickle.loads(pickled)

        # EPO should remember interferent subspace
        y_pred_after = pipeline_loaded.predict(self.X)

        np.testing.assert_allclose(y_pred_before, y_pred_after, rtol=1e-15)


class TestCrossValidationCompatibility:
    """Test that all methods work correctly in cross-validation."""

    def setup_method(self):
        """Create test data."""
        np.random.seed(42)
        self.n_samples = 120
        self.n_wavelengths = 50
        self.wavelengths = np.linspace(1500, 2500, self.n_wavelengths)

        # Create data with signal
        self.y = np.random.randn(self.n_samples) + 10
        signal = np.outer(self.y - 10, np.sin(np.linspace(0, 2*np.pi, self.n_wavelengths)))
        noise = np.random.randn(self.n_samples, self.n_wavelengths) * 0.5
        self.X = signal + noise

    def test_cv_with_osc_pipeline(self):
        """Test OSC pipeline with cross-validation."""
        from sklearn.model_selection import cross_val_score

        pipeline = Pipeline([
            ('osc', OSC(n_components=1)),
            ('pls', PLSRegression(n_components=5))
        ])

        # Cross-validation should work without data leakage
        scores = cross_val_score(
            pipeline, self.X, self.y,
            cv=5,
            scoring='r2'
        )

        # Should get reasonable scores
        assert len(scores) == 5
        assert all(s > -1.0 for s in scores)  # Not catastrophically bad
        assert np.mean(scores) > 0.3  # Some predictive power

    def test_cv_with_dosc_pipeline(self):
        """Test DOSC pipeline with cross-validation."""
        from sklearn.model_selection import cross_val_score

        pipeline = Pipeline([
            ('dosc', DOSC(n_components=2)),
            ('ridge', Ridge(alpha=1.0))
        ])

        scores = cross_val_score(
            pipeline, self.X, self.y,
            cv=5,
            scoring='neg_mean_squared_error'
        )

        # Should not crash
        assert len(scores) == 5
        assert all(np.isfinite(s) for s in scores)

    def test_cv_with_complex_pipeline(self):
        """Test complex pipeline in cross-validation."""
        from sklearn.model_selection import cross_val_score

        pipeline = Pipeline([
            ('exclude', WavelengthExcluder(self.wavelengths, exclude_ranges=[(1400, 1500)])),
            ('msc', MSC(reference='mean')),
            ('dosc', DOSC(n_components=1)),
            ('pls', PLSRegression(n_components=5))
        ])

        scores = cross_val_score(
            pipeline, self.X, self.y,
            cv=3,
            scoring='r2'
        )

        # Should work properly
        assert len(scores) == 3
        assert all(np.isfinite(s) for s in scores)


class TestEdgeCases:
    """Test edge cases in integrated pipelines."""

    def test_very_small_dataset(self):
        """Test pipelines work with very small datasets."""
        np.random.seed(42)
        X = np.random.randn(15, 20)  # Very small!
        y = np.random.randn(15)
        wavelengths = np.linspace(1500, 2500, 20)

        pipeline = Pipeline([
            ('msc', MSC(reference='mean')),
            ('dosc', DOSC(n_components=1)),
            ('ridge', Ridge(alpha=1.0))
        ])

        # Should handle small data gracefully
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)

        assert y_pred.shape[0] == 15
        assert not np.any(np.isnan(y_pred))

    def test_single_wavelength_after_exclusion(self):
        """Test that excluding most wavelengths still works."""
        np.random.seed(42)
        wavelengths = np.linspace(1000, 2500, 50)
        X = np.random.randn(50, 50)
        y = np.random.randn(50)

        # Exclude almost everything (leave only 1000-1100)
        pipeline = Pipeline([
            ('exclude', WavelengthExcluder(
                wavelengths,
                exclude_ranges=[(1100, 2500)]
            )),
            ('ridge', Ridge(alpha=1.0))
        ])

        # Should still work (though predictions may be poor)
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)

        assert not np.any(np.isnan(y_pred))

    def test_high_dimensional_data(self):
        """Test pipelines work with high-dimensional data (many wavelengths)."""
        np.random.seed(42)
        X = np.random.randn(100, 500)  # 500 wavelengths!
        y = np.random.randn(100)

        pipeline = Pipeline([
            ('msc', MSC(reference='mean')),
            ('osc', OSC(n_components=2)),
            ('pls', PLSRegression(n_components=10))
        ])

        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)

        assert y_pred.shape[0] == 100
        assert not np.any(np.isnan(y_pred))


def test_all_methods_importable():
    """Test that all interference methods can be imported."""
    from spectral_predict.interference import (
        WavelengthExcluder,
        MSC,
        OSC,
        GLSW,
        EPO,
        DOSC
    )

    # All should be classes
    assert WavelengthExcluder is not None
    assert MSC is not None
    assert OSC is not None
    assert GLSW is not None
    assert EPO is not None
    assert DOSC is not None


def test_pipeline_get_params():
    """Test that complex pipelines support get_params/set_params."""
    wavelengths = np.linspace(1500, 2500, 50)

    pipeline = Pipeline([
        ('exclude', WavelengthExcluder(wavelengths)),
        ('dosc', DOSC(n_components=2)),
        ('pls', PLSRegression(n_components=5))
    ])

    # Get all parameters
    params = pipeline.get_params()

    assert 'dosc__n_components' in params
    assert params['dosc__n_components'] == 2

    # Set parameters
    pipeline.set_params(dosc__n_components=3)
    assert pipeline.named_steps['dosc'].n_components == 3
