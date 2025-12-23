"""Comprehensive tests for model persistence in spectral_predict.model_io module.

This test suite validates model serialization, loading, and prediction functionality
for the DASP spectral analysis package. Tests cover:
- Save/load cycles for all model types
- Preprocessing pipeline persistence
- Prediction accuracy preservation
- Metadata handling
- Ensemble model persistence
- Error handling

Test coverage goal: >90% for model_io.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tempfile
import zipfile
import json
from pathlib import Path
from typing import Tuple, Any

from spectral_predict.model_io import (
    save_model,
    load_model,
    predict_with_model,
    predict_with_uncertainty,
    get_model_info,
    save_ensemble,
    load_ensemble,
)


@pytest.mark.io
class TestSaveLoadCycle:
    """Test save/load roundtrip for different model types."""

    def test_save_load_pls_model(self, tmp_path: Path) -> None:
        """Test PLS model save/load roundtrip."""
        from sklearn.cross_decomposition import PLSRegression

        # Create and fit model
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model = PLSRegression(n_components=5)
        model.fit(X, y)

        # Save model
        filepath = tmp_path / "pls_model.dasp"

        metadata = {
            "model_name": "PLS",
            "task_type": "regression",
            "wavelengths": list(range(1500, 1550)),
            "n_vars": 50,
            "performance": {"R2": 0.95, "RMSE": 0.12},
        }

        save_model(model, None, metadata, str(filepath))

        # Verify file exists and is valid ZIP
        assert filepath.exists()
        assert zipfile.is_zipfile(filepath)

        # Load model
        loaded = load_model(str(filepath))

        # Verify structure
        assert loaded["model"] is not None
        assert loaded["preprocessor"] is None
        assert loaded["metadata"]["model_name"] == "PLS"
        assert loaded["metadata"]["n_vars"] == 50

        # Verify predictions match
        pred_original = model.predict(X[:10])
        pred_loaded = loaded["model"].predict(X[:10])
        np.testing.assert_array_almost_equal(pred_original, pred_loaded)

    def test_save_load_ridge_model(self, tmp_path: Path) -> None:
        """Test Ridge model save/load roundtrip."""
        from sklearn.linear_model import Ridge

        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model = Ridge(alpha=1.0)
        model.fit(X, y)

        filepath = tmp_path / "ridge_model.dasp"

        metadata = {
            "model_name": "Ridge",
            "task_type": "regression",
            "wavelengths": list(range(1500, 1550)),
            "n_vars": 50,
            "params": {"alpha": 1.0},
        }

        save_model(model, None, metadata, str(filepath))
        loaded = load_model(str(filepath))

        assert loaded["metadata"]["model_name"] == "Ridge"
        assert loaded["metadata"]["params"]["alpha"] == 1.0

        # Verify predictions
        pred_original = model.predict(X[:10])
        pred_loaded = loaded["model"].predict(X[:10])
        np.testing.assert_array_almost_equal(pred_original, pred_loaded)

    def test_save_load_randomforest_model(self, tmp_path: Path) -> None:
        """Test RandomForest model save/load roundtrip."""
        from sklearn.ensemble import RandomForestRegressor

        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        filepath = tmp_path / "rf_model.dasp"

        metadata = {
            "model_name": "RandomForest",
            "task_type": "regression",
            "wavelengths": list(range(1500, 1550)),
            "n_vars": 50,
            "params": {"n_estimators": 10},
        }

        save_model(model, None, metadata, str(filepath))
        loaded = load_model(str(filepath))

        assert loaded["metadata"]["model_name"] == "RandomForest"

        # RandomForest predictions should be deterministic after loading
        pred_original = model.predict(X[:10])
        pred_loaded = loaded["model"].predict(X[:10])
        np.testing.assert_array_almost_equal(pred_original, pred_loaded)

    def test_save_load_xgboost_model(self, tmp_path: Path) -> None:
        """Test XGBoost model save/load roundtrip."""
        pytest.importorskip("xgboost")
        from xgboost import XGBRegressor

        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model = XGBRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        filepath = tmp_path / "xgb_model.dasp"

        metadata = {
            "model_name": "XGBoost",
            "task_type": "regression",
            "wavelengths": list(range(1500, 1550)),
            "n_vars": 50,
        }

        save_model(model, None, metadata, str(filepath))
        loaded = load_model(str(filepath))

        assert loaded["metadata"]["model_name"] == "XGBoost"

        # Verify predictions
        pred_original = model.predict(X[:10])
        pred_loaded = loaded["model"].predict(X[:10])
        np.testing.assert_array_almost_equal(pred_original, pred_loaded)

    def test_save_load_lightgbm_model(self, tmp_path: Path) -> None:
        """Test LightGBM model save/load roundtrip."""
        pytest.importorskip("lightgbm")
        from lightgbm import LGBMRegressor

        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model = LGBMRegressor(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X, y)

        filepath = tmp_path / "lgbm_model.dasp"

        metadata = {
            "model_name": "LightGBM",
            "task_type": "regression",
            "wavelengths": list(range(1500, 1550)),
            "n_vars": 50,
        }

        save_model(model, None, metadata, str(filepath))
        loaded = load_model(str(filepath))

        assert loaded["metadata"]["model_name"] == "LightGBM"

        # Verify predictions
        pred_original = model.predict(X[:10])
        pred_loaded = loaded["model"].predict(X[:10])
        np.testing.assert_array_almost_equal(pred_original, pred_loaded)


@pytest.mark.io
class TestPreprocessingPersistence:
    """Test preprocessing pipeline save/load."""

    def test_save_load_with_snv_preprocessor(self, tmp_path: Path) -> None:
        """Test model with SNV preprocessor saved and restored."""
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from spectral_predict.preprocess import SNV

        # Create pipeline with SNV
        pipe = Pipeline([("snv", SNV())])

        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        pipe.fit(X)
        X_processed = pipe.transform(X)

        model = Ridge()
        model.fit(X_processed, y)

        filepath = tmp_path / "model_snv.dasp"

        metadata = {
            "model_name": "Ridge",
            "task_type": "regression",
            "preprocessing": "snv",
            "wavelengths": list(range(1500, 1550)),
            "n_vars": 50,
        }

        save_model(model, pipe, metadata, str(filepath))
        loaded = load_model(str(filepath))

        # Verify preprocessor was saved
        assert loaded["preprocessor"] is not None
        assert loaded["metadata"]["preprocessing"] == "snv"

        # Verify end-to-end prediction works
        X_new = np.random.randn(10, 50)
        X_prep = loaded["preprocessor"].transform(X_new)
        predictions = loaded["model"].predict(X_prep)
        assert predictions.shape == (10,)

    def test_save_load_with_derivative_preprocessor(self, tmp_path: Path) -> None:
        """Test model with Savitzky-Golay derivative pipeline."""
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from spectral_predict.preprocess import SavgolDerivative

        # Create pipeline with 1st derivative
        pipe = Pipeline([("sg1", SavgolDerivative(deriv=1, window=11))])

        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        pipe.fit(X)
        X_processed = pipe.transform(X)

        model = Ridge()
        model.fit(X_processed, y)

        filepath = tmp_path / "model_sg1.dasp"

        metadata = {
            "model_name": "Ridge",
            "task_type": "regression",
            "preprocessing": "sg1",
            "wavelengths": list(range(1500, 1550)),
            "n_vars": 50,
            "window": 11,
            "polyorder": 2,
        }

        save_model(model, pipe, metadata, str(filepath))
        loaded = load_model(str(filepath))

        assert loaded["preprocessor"] is not None
        assert loaded["metadata"]["window"] == 11

        # Verify preprocessing works
        X_new = np.random.randn(10, 50)
        X_prep = loaded["preprocessor"].transform(X_new)
        assert X_prep.shape == (10, 50)

    def test_save_load_composite_pipeline(self, tmp_path: Path) -> None:
        """Test model with SNV + derivative composite pipeline."""
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from spectral_predict.preprocess import SNV, SavgolDerivative

        # Create composite pipeline: SNV then 1st derivative
        pipe = Pipeline([("snv", SNV()), ("sg1", SavgolDerivative(deriv=1, window=11))])

        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        pipe.fit(X)
        X_processed = pipe.transform(X)

        model = Ridge()
        model.fit(X_processed, y)

        filepath = tmp_path / "model_composite.dasp"

        metadata = {
            "model_name": "Ridge",
            "task_type": "regression",
            "preprocessing": "snv_sg1",
            "wavelengths": list(range(1500, 1550)),
            "n_vars": 50,
            "window": 11,
        }

        save_model(model, pipe, metadata, str(filepath))
        loaded = load_model(str(filepath))

        assert loaded["metadata"]["preprocessing"] == "snv_sg1"

        # Verify full pipeline works
        X_new = np.random.randn(10, 50)
        X_prep = loaded["preprocessor"].transform(X_new)
        predictions = loaded["model"].predict(X_prep)
        assert predictions.shape == (10,)


@pytest.mark.io
class TestPredictionTests:
    """Test predictions with saved models."""

    def test_predict_with_saved_model(self, tmp_path: Path) -> None:
        """Test loading model and making predictions."""
        from sklearn.linear_model import Ridge

        # Train model
        X_train = np.random.randn(100, 50)
        y_train = X_train[:, 0] * 2 + X_train[:, 1] * 3
        model = Ridge()
        model.fit(X_train, y_train)

        # Save
        filepath = tmp_path / "model.dasp"
        wavelengths = [float(i) for i in range(1500, 1550)]

        metadata = {
            "model_name": "Ridge",
            "task_type": "regression",
            "wavelengths": wavelengths,
            "n_vars": 50,
        }

        save_model(model, None, metadata, str(filepath))

        # Load
        model_dict = load_model(str(filepath))

        # Predict with DataFrame
        X_new = pd.DataFrame(np.random.randn(10, 50), columns=[str(w) for w in wavelengths])

        predictions = predict_with_model(model_dict, X_new)

        assert predictions.shape == (10,)
        assert not np.any(np.isnan(predictions))

    def test_predict_preserves_results(self, tmp_path: Path) -> None:
        """Test that predictions are identical before and after save/load."""
        from sklearn.linear_model import Ridge

        # Train model
        X_train = np.random.randn(100, 50)
        y_train = np.random.randn(100)
        model = Ridge()
        model.fit(X_train, y_train)

        # Get predictions before save
        X_test = np.random.randn(20, 50)
        pred_before = model.predict(X_test)

        # Save and load
        filepath = tmp_path / "model.dasp"
        metadata = {
            "model_name": "Ridge",
            "task_type": "regression",
            "wavelengths": list(range(1500, 1550)),
            "n_vars": 50,
        }

        save_model(model, None, metadata, str(filepath))
        loaded = load_model(str(filepath))

        # Get predictions after load
        pred_after = loaded["model"].predict(X_test)

        # Should be identical
        np.testing.assert_array_almost_equal(pred_before, pred_after)

    def test_predict_with_different_sample_count(self, tmp_path: Path) -> None:
        """Test predictions work with different number of samples."""
        from sklearn.linear_model import Ridge

        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model = Ridge()
        model.fit(X, y)

        filepath = tmp_path / "model.dasp"
        metadata = {
            "model_name": "Ridge",
            "task_type": "regression",
            "wavelengths": list(range(1500, 1550)),
            "n_vars": 50,
        }

        save_model(model, None, metadata, str(filepath))
        model_dict = load_model(str(filepath))

        # Test with different sample counts
        for n_samples in [1, 5, 20, 100]:
            X_new = np.random.randn(n_samples, 50)
            predictions = predict_with_model(
                model_dict, X_new, validate_wavelengths=False
            )
            assert predictions.shape == (n_samples,)


@pytest.mark.io
class TestMetadataHandling:
    """Test metadata preservation and handling."""

    def test_metadata_preserved(self, tmp_path: Path) -> None:
        """Test that all metadata fields are preserved."""
        from sklearn.linear_model import Ridge

        model = Ridge()
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model.fit(X, y)

        filepath = tmp_path / "model.dasp"

        metadata = {
            "model_name": "Ridge",
            "task_type": "regression",
            "wavelengths": list(range(1500, 1550)),
            "n_vars": 50,
            "performance": {"R2": 0.95, "RMSE": 0.12},
            "params": {"alpha": 1.0},
        }

        save_model(model, None, metadata, str(filepath))
        loaded = load_model(str(filepath))

        # Check all fields preserved
        assert loaded["metadata"]["model_name"] == "Ridge"
        assert loaded["metadata"]["task_type"] == "regression"
        assert loaded["metadata"]["n_vars"] == 50
        assert loaded["metadata"]["performance"]["R2"] == 0.95
        assert loaded["metadata"]["params"]["alpha"] == 1.0

        # Check auto-added fields
        assert "created" in loaded["metadata"]
        assert "dasp_version" in loaded["metadata"]
        assert "model_class" in loaded["metadata"]

    def test_wavelength_list_preserved(self, tmp_path: Path) -> None:
        """Test that exact wavelengths are stored and retrieved."""
        from sklearn.linear_model import Ridge

        model = Ridge()
        X = np.random.randn(100, 100)
        y = np.random.randn(100)
        model.fit(X, y)

        filepath = tmp_path / "model.dasp"

        # Use specific wavelengths
        wavelengths = [float(i) for i in range(1000, 1100)]

        metadata = {
            "model_name": "Ridge",
            "task_type": "regression",
            "wavelengths": wavelengths,
            "n_vars": 100,
        }

        save_model(model, None, metadata, str(filepath))
        loaded = load_model(str(filepath))

        # Wavelengths should be exactly preserved
        assert loaded["metadata"]["wavelengths"] == wavelengths
        assert len(loaded["metadata"]["wavelengths"]) == 100

    def test_invalid_model_file_raises(self, tmp_path: Path) -> None:
        """Test that invalid model files raise appropriate errors."""
        # Test nonexistent file
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent.dasp")

        # Test corrupted ZIP
        bad_file = tmp_path / "corrupted.dasp"
        bad_file.write_text("not a valid zip file")

        with pytest.raises(ValueError, match="not a valid .dasp"):
            load_model(str(bad_file))

        # Test ZIP missing metadata
        zip_no_meta = tmp_path / "no_metadata.dasp"
        with zipfile.ZipFile(zip_no_meta, "w") as zf:
            zf.writestr("model.pkl", b"fake data")

        with pytest.raises(ValueError, match="missing metadata.json"):
            load_model(str(zip_no_meta))


@pytest.mark.io
class TestEnsembleTests:
    """Test ensemble model persistence."""

    def test_save_load_ensemble(self, tmp_path: Path) -> None:
        """Test save/load for ensemble models."""
        pytest.importorskip("spectral_predict.ensemble")
        from spectral_predict.ensemble import RegionAwareWeightedEnsemble
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.linear_model import Ridge

        # Create base models
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        model1 = PLSRegression(n_components=5)
        model1.fit(X, y)

        model2 = Ridge(alpha=1.0)
        model2.fit(X, y)

        # Create ensemble
        ensemble = RegionAwareWeightedEnsemble(
            models=[model1, model2], model_names=["PLS", "Ridge"], n_regions=3
        )

        # Fit ensemble (simplified - just store fitted flag)
        ensemble.weights_ = np.array([0.5, 0.5])

        filepath = tmp_path / "ensemble.dasp"

        metadata = {
            "ensemble_type": "region_weighted",
            "ensemble_name": "Test Ensemble",
            "task_type": "regression",
            "preprocessing": "raw",
            "wavelengths": list(range(1500, 1550)),
            "n_vars": 50,
            "performance": {"R2": 0.96},
        }

        save_ensemble(ensemble, str(filepath), metadata)

        # Load ensemble
        loaded = load_ensemble(str(filepath))

        assert loaded["ensemble"] is not None
        assert loaded["metadata"]["ensemble_type"] == "region_weighted"
        assert loaded["model_names"] == ["PLS", "Ridge"]
        assert len(loaded["ensemble"].models) == 2


@pytest.mark.io
class TestUncertaintyPrediction:
    """Test prediction with uncertainty estimates."""

    def test_predict_with_uncertainty_regression(self, tmp_path: Path) -> None:
        """Test uncertainty prediction for regression models."""
        from sklearn.linear_model import Ridge

        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model = Ridge()
        model.fit(X, y)

        # Calculate CV residuals
        cv_residuals = np.random.randn(100) * 0.1
        cv_predictions = y + cv_residuals
        cv_actuals = y

        filepath = tmp_path / "model.dasp"
        metadata = {
            "model_name": "Ridge",
            "task_type": "regression",
            "wavelengths": list(range(1500, 1550)),
            "n_vars": 50,
            "performance": {"RMSE": 0.1},
        }

        save_model(
            model,
            None,
            metadata,
            str(filepath),
            cv_residuals=cv_residuals,
            cv_predictions=cv_predictions,
            cv_actuals=cv_actuals,
        )

        loaded = load_model(str(filepath))

        # Make predictions with uncertainty
        X_new = np.random.randn(10, 50)
        result = predict_with_uncertainty(loaded, X_new, validate_wavelengths=False)

        assert "predictions" in result
        assert "uncertainty" in result
        assert result["has_uncertainty"]
        assert "rmsecv" in result["uncertainty"]

    def test_predict_with_uncertainty_classification(self, tmp_path: Path) -> None:
        """Test uncertainty prediction for classification models."""
        from sklearn.ensemble import RandomForestClassifier

        X = np.random.randn(100, 50)
        y = np.random.randint(0, 3, 100)  # 3-class problem
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        filepath = tmp_path / "classifier.dasp"
        metadata = {
            "model_name": "RandomForest",
            "task_type": "classification",
            "wavelengths": list(range(1500, 1550)),
            "n_vars": 50,
        }

        save_model(model, None, metadata, str(filepath))
        loaded = load_model(str(filepath))

        # Make predictions with uncertainty
        X_new = np.random.randn(10, 50)
        result = predict_with_uncertainty(loaded, X_new, validate_wavelengths=False)

        assert "predictions" in result
        assert "uncertainty" in result
        if result["has_uncertainty"]:
            assert "probabilities" in result["uncertainty"]
            assert "confidence" in result["uncertainty"]


@pytest.mark.io
class TestGetModelInfo:
    """Test fast model info retrieval without full loading."""

    def test_get_model_info_basic(self, tmp_path: Path) -> None:
        """Test getting model info without loading full model."""
        from sklearn.linear_model import Ridge

        model = Ridge()
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model.fit(X, y)

        filepath = tmp_path / "model.dasp"

        metadata = {
            "model_name": "Ridge",
            "task_type": "regression",
            "wavelengths": list(range(1500, 1550)),
            "n_vars": 50,
            "performance": {"R2": 0.95, "RMSE": 0.12},
        }

        save_model(model, None, metadata, str(filepath))

        # Get info (faster than loading full model)
        info = get_model_info(str(filepath))

        assert info["model_name"] == "Ridge"
        assert info["task_type"] == "regression"
        assert info["n_vars"] == 50
        assert info["performance"]["R2"] == 0.95
        assert "created" in info
        assert "dasp_version" in info

    def test_get_model_info_nonexistent_file(self) -> None:
        """Test get_model_info with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            get_model_info("nonexistent.dasp")


@pytest.mark.io
class TestEdgeCases:
    """Test edge cases for model I/O."""

    def test_filepath_without_dasp_extension(self, tmp_path: Path) -> None:
        """Test that .dasp extension is automatically added."""
        from sklearn.linear_model import Ridge

        model = Ridge()
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model.fit(X, y)

        filepath = tmp_path / "model"  # No extension

        metadata = {
            "model_name": "Ridge",
            "task_type": "regression",
            "wavelengths": list(range(1500, 1550)),
            "n_vars": 50,
        }

        save_model(model, None, metadata, str(filepath))

        # Should create file with .dasp extension
        expected_path = Path(filepath).with_suffix(".dasp")
        assert expected_path.exists()

        # Should be loadable
        loaded = load_model(str(expected_path))
        assert loaded["model"] is not None

    def test_very_large_model(self, tmp_path: Path) -> None:
        """Test saving/loading a large model (many features)."""
        from sklearn.linear_model import Ridge

        # Create model with many features
        n_features = 2000
        model = Ridge()
        X = np.random.randn(100, n_features)
        y = np.random.randn(100)
        model.fit(X, y)

        filepath = tmp_path / "large_model.dasp"

        metadata = {
            "model_name": "Ridge",
            "task_type": "regression",
            "wavelengths": list(range(1000, 1000 + n_features)),
            "n_vars": n_features,
        }

        save_model(model, None, metadata, str(filepath))
        loaded = load_model(str(filepath))

        assert loaded["metadata"]["n_vars"] == n_features

        # Test prediction
        X_new = np.random.randn(5, n_features)
        predictions = predict_with_model(loaded, X_new, validate_wavelengths=False)
        assert predictions.shape == (5,)

    def test_missing_required_metadata_fields(self, tmp_path: Path) -> None:
        """Test that saving without required metadata raises error."""
        from sklearn.linear_model import Ridge

        model = Ridge()
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model.fit(X, y)

        # Missing 'model_name'
        metadata_incomplete = {
            "task_type": "regression",
            "wavelengths": list(range(1500, 1550)),
            "n_vars": 50,
        }

        filepath = tmp_path / "model.dasp"

        with pytest.raises(ValueError, match="missing required fields"):
            save_model(model, None, metadata_incomplete, str(filepath))

    def test_predict_with_wrong_wavelength_count(self, tmp_path: Path) -> None:
        """Test prediction with wrong number of wavelengths."""
        from sklearn.linear_model import Ridge

        model = Ridge()
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model.fit(X, y)

        filepath = tmp_path / "model.dasp"

        metadata = {
            "model_name": "Ridge",
            "task_type": "regression",
            "wavelengths": list(range(1500, 1550)),
            "n_vars": 50,
        }

        save_model(model, None, metadata, str(filepath))
        model_dict = load_model(str(filepath))

        # Create data with wrong number of features
        X_wrong = np.random.randn(10, 30)  # 30 instead of 50

        with pytest.raises(ValueError, match="requires.*wavelengths"):
            predict_with_model(model_dict, X_wrong)


@pytest.mark.io
class TestApplicabilityDomain:
    """Test applicability domain functionality."""

    def test_save_load_with_applicability_domain(self, tmp_path: Path) -> None:
        """Test model with applicability domain data."""
        from sklearn.linear_model import Ridge

        X_train = np.random.randn(100, 50)
        y_train = np.random.randn(100)
        model = Ridge()
        model.fit(X_train, y_train)

        filepath = tmp_path / "model_ad.dasp"

        metadata = {
            "model_name": "Ridge",
            "task_type": "regression",
            "wavelengths": list(range(1500, 1550)),
            "n_vars": 50,
        }

        # Save with training data for applicability domain
        save_model(model, None, metadata, str(filepath), X_train=X_train)

        loaded = load_model(str(filepath))

        # Check applicability domain was saved
        # The flag is in metadata, ad_data and pca_model should be present
        assert "ad_data" in loaded
        assert loaded["ad_data"] is not None
        assert "pca_model" in loaded
        assert loaded["pca_model"] is not None

        # Test prediction with applicability domain
        X_new = np.random.randn(10, 50)
        result = predict_with_uncertainty(loaded, X_new, validate_wavelengths=False)

        if result["has_applicability_domain"]:
            assert "pca_distance" in result["applicability_domain"]
            assert "distance_status" in result["applicability_domain"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
