"""
Tests for the Engine API module.

Tests cover:
- Data loading (CSV, Excel, various spectral formats)
- Preprocessing transformations
- Model training
- Predictions
- Model persistence
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TestEngineAPIImport:
    """Test that the engine API can be imported."""

    def test_import_engine_api(self):
        """Test basic import."""
        from spectral_predict_v2.engine.api import EngineAPI
        assert EngineAPI is not None

    def test_import_dataclasses(self):
        """Test dataclass imports."""
        from spectral_predict_v2.engine.api import (
            LoadedData, AnalysisConfig, TrainedModel
        )
        assert LoadedData is not None
        assert AnalysisConfig is not None
        assert TrainedModel is not None


class TestEngineAPIPreprocessing:
    """Test preprocessing functionality."""

    @pytest.fixture
    def engine(self):
        from spectral_predict_v2.engine.api import EngineAPI
        return EngineAPI()

    @pytest.fixture
    def sample_spectra(self):
        """Create sample spectral data."""
        np.random.seed(42)
        n_samples = 20
        n_wavelengths = 100
        X = np.random.randn(n_samples, n_wavelengths) * 0.1 + 1.0
        return X

    def test_snv_preprocessing(self, engine, sample_spectra):
        """Test SNV preprocessing."""
        X_snv = engine.apply_preprocessing(sample_spectra, "snv")
        
        assert X_snv.shape == sample_spectra.shape
        # SNV normalizes each spectrum to zero mean and unit variance
        np.testing.assert_array_almost_equal(
            np.mean(X_snv, axis=1), 
            np.zeros(X_snv.shape[0]), 
            decimal=10
        )

    def test_raw_preprocessing(self, engine, sample_spectra):
        """Test raw (no) preprocessing."""
        X_raw = engine.apply_preprocessing(sample_spectra, "raw")
        
        np.testing.assert_array_equal(X_raw, sample_spectra)

    def test_available_preprocessing_methods(self, engine):
        """Test getting available preprocessing methods."""
        methods = engine.get_available_preprocessing()
        
        assert len(methods) > 0
        method_keys = [m[0] for m in methods]
        assert "raw" in method_keys
        assert "snv" in method_keys


class TestEngineAPIModelTraining:
    """Test model training functionality."""

    @pytest.fixture
    def engine(self):
        from spectral_predict_v2.engine.api import EngineAPI
        return EngineAPI()

    @pytest.fixture
    def training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 50
        n_wavelengths = 100
        
        X = np.random.randn(n_samples, n_wavelengths)
        y = X[:, :5].sum(axis=1) + np.random.randn(n_samples) * 0.5
        
        return X, y

    def test_train_pls_model(self, engine, training_data):
        """Test PLS model training."""
        X, y = training_data
        
        trained = engine.train_model(
            X, y,
            model_type="pls",
            preprocessing="raw",
            n_components=5
        )
        
        assert trained is not None
        assert trained.model is not None
        assert "rmse" in trained.metrics
        assert "r2" in trained.metrics

    def test_available_models(self, engine):
        """Test getting available models."""
        regression_models = engine.get_available_models("regression")
        classification_models = engine.get_available_models("classification")
        
        assert len(regression_models) > 0
        assert len(classification_models) > 0


class TestEngineAPIPrediction:
    """Test prediction functionality."""

    @pytest.fixture
    def engine(self):
        from spectral_predict_v2.engine.api import EngineAPI
        return EngineAPI()

    @pytest.fixture
    def trained_model(self, engine):
        """Create and train a model."""
        np.random.seed(42)
        n_samples = 50
        n_wavelengths = 100
        
        X = np.random.randn(n_samples, n_wavelengths)
        y = X[:, :5].sum(axis=1) + np.random.randn(n_samples) * 0.5
        
        return engine.train_model(X, y, model_type="pls", n_components=5)

    def test_predict(self, engine, trained_model):
        """Test making predictions."""
        np.random.seed(123)
        X_new = np.random.randn(10, 100)
        
        predictions, uncertainty, ad_flags = engine.predict(
            X_new, trained_model, return_uncertainty=True
        )
        
        assert len(predictions) == 10
        assert uncertainty is not None
        assert ad_flags is not None


class TestEngineAPIModelPersistence:
    """Test model save/load functionality."""

    @pytest.fixture
    def engine(self):
        from spectral_predict_v2.engine.api import EngineAPI
        return EngineAPI()

    @pytest.fixture
    def trained_model(self, engine):
        """Create and train a model."""
        np.random.seed(42)
        X = np.random.randn(50, 100)
        y = X[:, :5].sum(axis=1)
        
        model = engine.train_model(X, y, model_type="pls", n_components=5)
        model.config["y_true"] = y
        return model

    def test_save_and_load_model(self, engine, trained_model, tmp_path):
        """Test saving and loading a model."""
        model_path = tmp_path / "test_model.dasp"
        
        engine.save_model(trained_model, str(model_path))
        assert model_path.exists()
        
        loaded = engine.load_model(str(model_path))
        
        assert loaded is not None
        assert loaded.name == trained_model.name
        assert loaded.preprocessing == trained_model.preprocessing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
