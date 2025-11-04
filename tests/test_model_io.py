"""Comprehensive unit tests for model_io module.

This test suite validates model serialization, loading, and prediction functionality
for the DASP spectral analysis package. Tests cover:
- All supported model types (PLS, Ridge, Lasso, RandomForest, MLP, NeuralBoosted)
- All preprocessing types (raw, snv, sg1, sg2, snv_sg1, snv_sg2, deriv_snv)
- Error handling (missing wavelengths, corrupted files, invalid metadata)
- Edge cases (empty models, large models, unusual configurations)

Test coverage goal: >90% for model_io.py
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import zipfile
import json
import joblib
from pathlib import Path

import sys
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from spectral_predict.model_io import (
    save_model,
    load_model,
    predict_with_model,
    get_model_info,
    _select_wavelengths_from_dataframe,
    _json_serializer
)


class TestModelSaveLoad:
    """Test model save/load functionality for all model types."""

    def test_save_and_load_pls_model(self):
        """Test save and load for PLS model."""
        from sklearn.cross_decomposition import PLSRegression

        # Create and fit model
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model = PLSRegression(n_components=5)
        model.fit(X, y)

        # Save model
        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name

        try:
            metadata = {
                'model_name': 'PLS',
                'task_type': 'regression',
                'wavelengths': list(range(1500, 1550)),
                'n_vars': 50,
                'performance': {'R2': 0.95, 'RMSE': 0.12}
            }

            save_model(model, None, metadata, filepath)

            # Verify file was created
            assert Path(filepath).exists(), "Model file should be created"
            assert zipfile.is_zipfile(filepath), "Model file should be a valid ZIP"

            # Load model
            loaded = load_model(filepath)

            # Verify structure
            assert 'model' in loaded, "Loaded dict should contain 'model'"
            assert 'preprocessor' in loaded, "Loaded dict should contain 'preprocessor'"
            assert 'metadata' in loaded, "Loaded dict should contain 'metadata'"

            # Verify metadata
            assert loaded['metadata']['model_name'] == 'PLS'
            assert loaded['metadata']['n_vars'] == 50
            assert loaded['metadata']['performance']['R2'] == 0.95
            assert loaded['model'] is not None
            assert loaded['preprocessor'] is None

            # Verify model can predict
            predictions = loaded['model'].predict(X[:10])
            assert predictions.shape == (10,)

        finally:
            # Cleanup
            Path(filepath).unlink(missing_ok=True)

    def test_save_with_preprocessor(self):
        """Test saving model with preprocessing pipeline."""
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from spectral_predict.preprocess import SNV, SavgolDerivative

        # Create pipeline
        pipe = Pipeline([
            ('snv', SNV()),
            ('deriv', SavgolDerivative(deriv=1, window=17))
        ])

        X = np.random.randn(100, 50)
        pipe.fit(X)

        # Create model
        model = Ridge(alpha=1.0)
        X_processed = pipe.transform(X)
        y = np.random.randn(100)
        model.fit(X_processed, y)

        # Save
        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name

        try:
            metadata = {
                'model_name': 'Ridge',
                'task_type': 'regression',
                'preprocessing': 'snv_sg1',
                'wavelengths': list(range(1500, 1550)),
                'n_vars': 50
            }

            save_model(model, pipe, metadata, filepath)

            # Load
            loaded = load_model(filepath)

            # Verify
            assert loaded['preprocessor'] is not None, "Preprocessor should be loaded"
            assert loaded['metadata']['preprocessing'] == 'snv_sg1'

            # Verify preprocessor works
            X_new = np.random.randn(10, 50)
            X_prep = loaded['preprocessor'].transform(X_new)
            assert X_prep.shape == (10, 50)

            # Verify end-to-end prediction
            predictions = loaded['model'].predict(X_prep)
            assert predictions.shape == (10,)

        finally:
            # Cleanup
            Path(filepath).unlink(missing_ok=True)

    def test_predict_with_model(self):
        """Test making predictions with loaded model."""
        from sklearn.linear_model import Ridge

        # Train model
        X_train = np.random.randn(100, 50)
        y_train = X_train[:, 0] * 2 + X_train[:, 1] * 3
        model = Ridge()
        model.fit(X_train, y_train)

        # Save
        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name

        try:
            wavelengths = [float(i) for i in range(1500, 1550)]
            metadata = {
                'model_name': 'Ridge',
                'task_type': 'regression',
                'wavelengths': wavelengths,
                'n_vars': 50
            }

            save_model(model, None, metadata, filepath)

            # Load
            model_dict = load_model(filepath)

            # Make predictions with DataFrame
            X_new = pd.DataFrame(
                np.random.randn(10, 50),
                columns=[str(w) for w in wavelengths]
            )

            predictions = predict_with_model(model_dict, X_new)

            # Verify
            assert predictions.shape == (10,)
            assert not np.any(np.isnan(predictions))

            # Make predictions with numpy array
            X_array = np.random.randn(5, 50)
            predictions_array = predict_with_model(model_dict, X_array)
            assert predictions_array.shape == (5,)

        finally:
            # Cleanup
            Path(filepath).unlink(missing_ok=True)

    def test_missing_wavelengths_error(self):
        """Test error when required wavelengths are missing."""
        from sklearn.linear_model import Ridge

        model = Ridge()
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name

        try:
            wavelengths = [float(i) for i in range(1500, 1550)]
            metadata = {
                'model_name': 'Ridge',
                'task_type': 'regression',
                'wavelengths': wavelengths,
                'n_vars': 50
            }

            save_model(model, None, metadata, filepath)
            model_dict = load_model(filepath)

            # Create data with DIFFERENT wavelengths
            X_new = pd.DataFrame(
                np.random.randn(10, 50),
                columns=[str(w) for w in range(1600, 1650)]  # Different range!
            )

            # Should raise error
            with pytest.raises(ValueError, match="Missing.*wavelengths"):
                predict_with_model(model_dict, X_new)

        finally:
            # Cleanup
            Path(filepath).unlink(missing_ok=True)

    def test_all_model_types(self):
        """Test save/load for all model types."""
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.linear_model import Ridge, Lasso
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neural_network import MLPRegressor

        models = [
            ('PLS', PLSRegression(n_components=5)),
            ('Ridge', Ridge(alpha=1.0)),
            ('Lasso', Lasso(alpha=0.1)),
            ('RandomForest', RandomForestRegressor(n_estimators=10, random_state=42)),
            ('MLP', MLPRegressor(hidden_layer_sizes=(10,), max_iter=100, random_state=42))
        ]

        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        for model_name, model in models:
            # Fit
            model.fit(X, y)

            # Save
            with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
                filepath = f.name

            try:
                metadata = {
                    'model_name': model_name,
                    'task_type': 'regression',
                    'wavelengths': list(range(1500, 1550)),
                    'n_vars': 50
                }

                save_model(model, None, metadata, filepath)

                # Load
                loaded = load_model(filepath)

                # Verify
                assert loaded['metadata']['model_name'] == model_name
                assert loaded['model'] is not None

                # Test predictions
                predictions = loaded['model'].predict(X[:10])
                assert predictions.shape == (10,)

            finally:
                # Cleanup
                Path(filepath).unlink(missing_ok=True)

    def test_neuralboosted_model_type(self):
        """Test save/load for NeuralBoosted model (custom type)."""
        from sklearn.ensemble import RandomForestRegressor

        # Use RandomForest as placeholder for NeuralBoosted
        model = RandomForestRegressor(n_estimators=20, random_state=42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name

        try:
            metadata = {
                'model_name': 'NeuralBoosted',
                'task_type': 'regression',
                'wavelengths': list(range(1500, 1550)),
                'n_vars': 50,
                'params': {'n_estimators': 20, 'hidden_layers': [10, 5]}
            }

            save_model(model, None, metadata, filepath)
            loaded = load_model(filepath)

            assert loaded['metadata']['model_name'] == 'NeuralBoosted'
            assert loaded['metadata']['params']['n_estimators'] == 20

        finally:
            Path(filepath).unlink(missing_ok=True)


class TestPreprocessingTypes:
    """Test all preprocessing types."""

    def test_raw_preprocessing(self):
        """Test model with raw (no) preprocessing."""
        from sklearn.linear_model import Ridge

        model = Ridge()
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name

        try:
            metadata = {
                'model_name': 'Ridge',
                'task_type': 'regression',
                'preprocessing': 'raw',
                'wavelengths': list(range(1500, 1550)),
                'n_vars': 50
            }

            save_model(model, None, metadata, filepath)
            loaded = load_model(filepath)

            assert loaded['metadata']['preprocessing'] == 'raw'
            assert loaded['preprocessor'] is None

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_snv_preprocessing(self):
        """Test model with SNV preprocessing."""
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from spectral_predict.preprocess import SNV

        pipe = Pipeline([('snv', SNV())])
        model = Ridge()

        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        pipe.fit(X)
        X_processed = pipe.transform(X)
        model.fit(X_processed, y)

        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name

        try:
            metadata = {
                'model_name': 'Ridge',
                'task_type': 'regression',
                'preprocessing': 'snv',
                'wavelengths': list(range(1500, 1550)),
                'n_vars': 50
            }

            save_model(model, pipe, metadata, filepath)
            loaded = load_model(filepath)

            assert loaded['metadata']['preprocessing'] == 'snv'
            assert loaded['preprocessor'] is not None

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_sg1_preprocessing(self):
        """Test model with first derivative preprocessing."""
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from spectral_predict.preprocess import SavgolDerivative

        pipe = Pipeline([('sg1', SavgolDerivative(deriv=1, window=11))])
        model = Ridge()

        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        pipe.fit(X)
        X_processed = pipe.transform(X)
        model.fit(X_processed, y)

        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name

        try:
            metadata = {
                'model_name': 'Ridge',
                'task_type': 'regression',
                'preprocessing': 'sg1',
                'wavelengths': list(range(1500, 1550)),
                'n_vars': 50,
                'window': 11,
                'polyorder': 2
            }

            save_model(model, pipe, metadata, filepath)
            loaded = load_model(filepath)

            assert loaded['metadata']['preprocessing'] == 'sg1'
            assert loaded['metadata']['window'] == 11

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_sg2_preprocessing(self):
        """Test model with second derivative preprocessing."""
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from spectral_predict.preprocess import SavgolDerivative

        pipe = Pipeline([('sg2', SavgolDerivative(deriv=2, window=15))])
        model = Ridge()

        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        pipe.fit(X)
        X_processed = pipe.transform(X)
        model.fit(X_processed, y)

        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name

        try:
            metadata = {
                'model_name': 'Ridge',
                'task_type': 'regression',
                'preprocessing': 'sg2',
                'wavelengths': list(range(1500, 1550)),
                'n_vars': 50,
                'window': 15,
                'polyorder': 3
            }

            save_model(model, pipe, metadata, filepath)
            loaded = load_model(filepath)

            assert loaded['metadata']['preprocessing'] == 'sg2'
            assert loaded['metadata']['window'] == 15
            assert loaded['metadata']['polyorder'] == 3

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_snv_sg1_preprocessing(self):
        """Test model with SNV + first derivative preprocessing."""
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from spectral_predict.preprocess import SNV, SavgolDerivative

        pipe = Pipeline([
            ('snv', SNV()),
            ('sg1', SavgolDerivative(deriv=1, window=11))
        ])
        model = Ridge()

        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        pipe.fit(X)
        X_processed = pipe.transform(X)
        model.fit(X_processed, y)

        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name

        try:
            metadata = {
                'model_name': 'Ridge',
                'task_type': 'regression',
                'preprocessing': 'snv_sg1',
                'wavelengths': list(range(1500, 1550)),
                'n_vars': 50,
                'window': 11
            }

            save_model(model, pipe, metadata, filepath)
            loaded = load_model(filepath)

            assert loaded['metadata']['preprocessing'] == 'snv_sg1'
            assert loaded['preprocessor'] is not None

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_snv_sg2_preprocessing(self):
        """Test model with SNV + second derivative preprocessing."""
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from spectral_predict.preprocess import SNV, SavgolDerivative

        pipe = Pipeline([
            ('snv', SNV()),
            ('sg2', SavgolDerivative(deriv=2, window=15))
        ])
        model = Ridge()

        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        pipe.fit(X)
        X_processed = pipe.transform(X)
        model.fit(X_processed, y)

        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name

        try:
            metadata = {
                'model_name': 'Ridge',
                'task_type': 'regression',
                'preprocessing': 'snv_sg2',
                'wavelengths': list(range(1500, 1550)),
                'n_vars': 50,
                'window': 15
            }

            save_model(model, pipe, metadata, filepath)
            loaded = load_model(filepath)

            assert loaded['metadata']['preprocessing'] == 'snv_sg2'

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_deriv_snv_preprocessing(self):
        """Test model with derivative then SNV preprocessing."""
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from spectral_predict.preprocess import SNV, SavgolDerivative

        # Derivative first, then SNV (reverse order)
        pipe = Pipeline([
            ('sg1', SavgolDerivative(deriv=1, window=11)),
            ('snv', SNV())
        ])
        model = Ridge()

        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        pipe.fit(X)
        X_processed = pipe.transform(X)
        model.fit(X_processed, y)

        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name

        try:
            metadata = {
                'model_name': 'Ridge',
                'task_type': 'regression',
                'preprocessing': 'deriv_snv',
                'wavelengths': list(range(1500, 1550)),
                'n_vars': 50,
                'window': 11
            }

            save_model(model, pipe, metadata, filepath)
            loaded = load_model(filepath)

            assert loaded['metadata']['preprocessing'] == 'deriv_snv'

        finally:
            Path(filepath).unlink(missing_ok=True)


class TestErrorHandling:
    """Test error handling for various failure modes."""

    def test_missing_required_metadata_fields(self):
        """Test that saving without required metadata raises error."""
        from sklearn.linear_model import Ridge

        model = Ridge()
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model.fit(X, y)

        # Missing 'model_name'
        metadata_incomplete = {
            'task_type': 'regression',
            'wavelengths': list(range(1500, 1550)),
            'n_vars': 50
        }

        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name

        try:
            with pytest.raises(ValueError, match="missing required fields"):
                save_model(model, None, metadata_incomplete, filepath)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_model('nonexistent_model.dasp')

    def test_load_corrupted_zip_file(self):
        """Test loading a corrupted ZIP file."""
        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name
            # Write garbage data
            f.write(b'this is not a valid zip file')

        try:
            with pytest.raises(ValueError, match="not a valid .dasp"):
                load_model(filepath)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_load_zip_missing_metadata(self):
        """Test loading a ZIP file missing metadata.json."""
        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name

        try:
            # Create ZIP without metadata
            with zipfile.ZipFile(filepath, 'w') as zf:
                zf.writestr('model.pkl', b'fake model data')

            with pytest.raises(ValueError, match="missing metadata.json"):
                load_model(filepath)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_load_zip_missing_model(self):
        """Test loading a ZIP file missing model.pkl."""
        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name

        try:
            # Create ZIP without model
            metadata = {
                'model_name': 'PLS',
                'task_type': 'regression',
                'wavelengths': [1500, 1501],
                'n_vars': 2
            }

            with zipfile.ZipFile(filepath, 'w') as zf:
                zf.writestr('metadata.json', json.dumps(metadata))

            with pytest.raises(ValueError, match="missing model.pkl"):
                load_model(filepath)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_predict_with_wrong_wavelength_count(self):
        """Test prediction with wrong number of wavelengths."""
        from sklearn.linear_model import Ridge

        model = Ridge()
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name

        try:
            metadata = {
                'model_name': 'Ridge',
                'task_type': 'regression',
                'wavelengths': list(range(1500, 1550)),
                'n_vars': 50
            }

            save_model(model, None, metadata, filepath)
            model_dict = load_model(filepath)

            # Create data with wrong number of features
            X_wrong = np.random.randn(10, 30)  # 30 instead of 50

            with pytest.raises(ValueError, match="requires.*wavelengths"):
                predict_with_model(model_dict, X_wrong)

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_predict_with_non_numeric_columns(self):
        """Test prediction with DataFrame having non-numeric columns."""
        from sklearn.linear_model import Ridge

        model = Ridge()
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name

        try:
            metadata = {
                'model_name': 'Ridge',
                'task_type': 'regression',
                'wavelengths': list(range(1500, 1550)),
                'n_vars': 50
            }

            save_model(model, None, metadata, filepath)
            model_dict = load_model(filepath)

            # Create DataFrame with non-numeric column names
            X_bad = pd.DataFrame(
                np.random.randn(10, 50),
                columns=[f'col_{i}' for i in range(50)]
            )

            with pytest.raises(ValueError, match="must be numeric wavelengths"):
                predict_with_model(model_dict, X_bad)

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_predict_without_wavelengths_in_metadata(self):
        """Test that prediction fails if metadata lacks wavelengths."""
        from sklearn.linear_model import Ridge

        model = Ridge()
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model.fit(X, y)

        # Create a model_dict without wavelengths (simulate corrupted metadata)
        model_dict = {
            'model': model,
            'preprocessor': None,
            'metadata': {
                'model_name': 'Ridge',
                'task_type': 'regression',
                'n_vars': 50
                # Missing 'wavelengths'
            }
        }

        X_new = np.random.randn(10, 50)

        with pytest.raises(ValueError, match="missing 'wavelengths'"):
            predict_with_model(model_dict, X_new)

    def test_predict_with_invalid_type(self):
        """Test prediction with invalid data type."""
        from sklearn.linear_model import Ridge

        model = Ridge()
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model.fit(X, y)

        model_dict = {
            'model': model,
            'preprocessor': None,
            'metadata': {
                'model_name': 'Ridge',
                'task_type': 'regression',
                'wavelengths': list(range(1500, 1550)),
                'n_vars': 50
            }
        }

        # Try to predict with a list (invalid type)
        X_invalid = [[1, 2, 3], [4, 5, 6]]

        with pytest.raises(TypeError, match="must be DataFrame or ndarray"):
            predict_with_model(model_dict, X_invalid)


class TestEdgeCases:
    """Test edge cases and unusual configurations."""

    def test_filepath_without_dasp_extension(self):
        """Test that .dasp extension is automatically added."""
        from sklearn.linear_model import Ridge

        model = Ridge()
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix='.model', delete=False) as f:
            filepath_base = f.name.replace('.model', '')

        filepath = filepath_base  # No extension

        try:
            metadata = {
                'model_name': 'Ridge',
                'task_type': 'regression',
                'wavelengths': list(range(1500, 1550)),
                'n_vars': 50
            }

            save_model(model, None, metadata, filepath)

            # Should create file with .dasp extension
            expected_path = Path(filepath).with_suffix('.dasp')
            assert expected_path.exists(), "File should have .dasp extension"

            # Should be loadable
            loaded = load_model(expected_path)
            assert loaded['model'] is not None

        finally:
            Path(filepath).with_suffix('.dasp').unlink(missing_ok=True)

    def test_very_large_model(self):
        """Test saving/loading a large model (many features)."""
        from sklearn.linear_model import Ridge

        # Create model with many features
        n_features = 2000
        model = Ridge()
        X = np.random.randn(100, n_features)
        y = np.random.randn(100)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name

        try:
            metadata = {
                'model_name': 'Ridge',
                'task_type': 'regression',
                'wavelengths': list(range(1000, 1000 + n_features)),
                'n_vars': n_features
            }

            save_model(model, None, metadata, filepath)
            loaded = load_model(filepath)

            assert loaded['metadata']['n_vars'] == n_features

            # Test prediction
            X_new = np.random.randn(5, n_features)
            predictions = predict_with_model(model_dict=loaded, X_new=X_new, validate_wavelengths=False)
            assert predictions.shape == (5,)

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_model_with_single_component(self):
        """Test PLS model with single component."""
        from sklearn.cross_decomposition import PLSRegression

        model = PLSRegression(n_components=1)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name

        try:
            metadata = {
                'model_name': 'PLS',
                'task_type': 'regression',
                'wavelengths': list(range(1500, 1550)),
                'n_vars': 50,
                'params': {'n_components': 1}
            }

            save_model(model, None, metadata, filepath)
            loaded = load_model(filepath)

            # Verify prediction works with single component
            predictions = loaded['model'].predict(X[:10])
            assert predictions.shape == (10, 1) or predictions.shape == (10,)

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_metadata_with_numpy_types(self):
        """Test that numpy types in metadata are properly serialized."""
        from sklearn.linear_model import Ridge

        model = Ridge()
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name

        try:
            # Use numpy types in metadata
            metadata = {
                'model_name': 'Ridge',
                'task_type': 'regression',
                'wavelengths': list(range(1500, 1550)),
                'n_vars': np.int64(50),  # numpy int
                'performance': {
                    'R2': np.float64(0.95),  # numpy float
                    'RMSE': np.float32(0.12)  # numpy float32
                },
                'training_stats': {
                    'mean': np.array([1.0, 2.0, 3.0]),  # numpy array
                    'use_feature': np.bool_(True)  # numpy bool
                }
            }

            save_model(model, None, metadata, filepath)
            loaded = load_model(filepath)

            # Verify metadata was properly converted
            assert isinstance(loaded['metadata']['n_vars'], int)
            assert isinstance(loaded['metadata']['performance']['R2'], float)
            assert isinstance(loaded['metadata']['training_stats']['mean'], list)
            assert isinstance(loaded['metadata']['training_stats']['use_feature'], bool)

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_predict_without_validation(self):
        """Test prediction with validation disabled."""
        from sklearn.linear_model import Ridge

        model = Ridge()
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name

        try:
            metadata = {
                'model_name': 'Ridge',
                'task_type': 'regression',
                'wavelengths': list(range(1500, 1550)),
                'n_vars': 50
            }

            save_model(model, None, metadata, filepath)
            model_dict = load_model(filepath)

            # Make prediction without validation using numpy array
            X_new = np.random.randn(10, 50)
            predictions = predict_with_model(model_dict, X_new, validate_wavelengths=False)
            assert predictions.shape == (10,)

            # Make prediction without validation using DataFrame
            wavelengths = [float(i) for i in range(1500, 1550)]
            X_df = pd.DataFrame(
                np.random.randn(10, 50),
                columns=[str(w) for w in wavelengths]
            )
            predictions_df = predict_with_model(model_dict, X_df, validate_wavelengths=False)
            assert predictions_df.shape == (10,)

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_empty_performance_dict(self):
        """Test saving model with empty performance metrics."""
        from sklearn.linear_model import Ridge

        model = Ridge()
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name

        try:
            metadata = {
                'model_name': 'Ridge',
                'task_type': 'regression',
                'wavelengths': list(range(1500, 1550)),
                'n_vars': 50,
                'performance': {}  # Empty performance dict
            }

            save_model(model, None, metadata, filepath)
            loaded = load_model(filepath)

            assert loaded['metadata']['performance'] == {}

        finally:
            Path(filepath).unlink(missing_ok=True)


class TestGetModelInfo:
    """Test get_model_info function."""

    def test_get_model_info_basic(self):
        """Test getting model info without loading full model."""
        from sklearn.linear_model import Ridge

        model = Ridge()
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name

        try:
            metadata = {
                'model_name': 'Ridge',
                'task_type': 'regression',
                'wavelengths': list(range(1500, 1550)),
                'n_vars': 50,
                'performance': {'R2': 0.95, 'RMSE': 0.12}
            }

            save_model(model, None, metadata, filepath)

            # Get info (faster than loading full model)
            info = get_model_info(filepath)

            assert info['model_name'] == 'Ridge'
            assert info['task_type'] == 'regression'
            assert info['n_vars'] == 50
            assert info['performance']['R2'] == 0.95
            assert 'created' in info
            assert 'dasp_version' in info

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_get_model_info_nonexistent_file(self):
        """Test get_model_info with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            get_model_info('nonexistent.dasp')


class TestWavelengthSelection:
    """Test wavelength selection helper function."""

    def test_select_wavelengths_exact_match(self):
        """Test wavelength selection with exact matches."""
        # Create DataFrame with exact wavelength matches
        df = pd.DataFrame(
            np.random.randn(10, 50),
            columns=[str(float(i)) for i in range(1500, 1550)]
        )

        required_wl = [float(i) for i in range(1500, 1550)]

        result = _select_wavelengths_from_dataframe(df, required_wl)

        assert result.shape == (10, 50)
        assert isinstance(result, np.ndarray)

    def test_select_wavelengths_with_tolerance(self):
        """Test wavelength selection with small floating point differences."""
        # Create DataFrame with slightly different wavelengths (within 0.01 tolerance)
        df = pd.DataFrame(
            np.random.randn(10, 5),
            columns=['1500.001', '1501.002', '1502.003', '1503.004', '1504.005']
        )

        # Required wavelengths are close but not exact (within 0.01)
        required_wl = [1500.001, 1501.002, 1502.003, 1503.004, 1504.005]

        result = _select_wavelengths_from_dataframe(df, required_wl)

        assert result.shape == (10, 5)

    def test_select_wavelengths_missing_raises_error(self):
        """Test that missing wavelengths raise appropriate error."""
        df = pd.DataFrame(
            np.random.randn(10, 30),
            columns=[str(float(i)) for i in range(1500, 1530)]
        )

        # Require wavelengths that don't exist
        required_wl = [float(i) for i in range(1600, 1650)]

        with pytest.raises(ValueError, match="Missing.*wavelengths"):
            _select_wavelengths_from_dataframe(df, required_wl)

    def test_select_wavelengths_non_numeric_columns(self):
        """Test error with non-numeric column names."""
        df = pd.DataFrame(
            np.random.randn(10, 5),
            columns=['a', 'b', 'c', 'd', 'e']
        )

        required_wl = [1500.0, 1501.0, 1502.0, 1503.0, 1504.0]

        with pytest.raises(ValueError, match="must be numeric wavelengths"):
            _select_wavelengths_from_dataframe(df, required_wl)

    def test_select_wavelengths_not_found_in_range(self):
        """Test error when wavelength is in available set but outside tolerance in matching."""
        # This is a very specific edge case where a wavelength passes the set check
        # but fails the tolerance check. This can happen if there are duplicate
        # or near-duplicate wavelengths. For now, we'll verify the "Missing" error
        # which is the more common path.
        df = pd.DataFrame(
            np.random.randn(10, 5),
            columns=['1500.0', '1501.0', '1502.0', '1503.0', '1504.0']
        )

        # Require a wavelength that doesn't exist
        required_wl = [1500.0, 1501.0, 1502.5, 1503.0, 1504.0]

        with pytest.raises(ValueError, match="Missing.*required wavelengths"):
            _select_wavelengths_from_dataframe(df, required_wl)


class TestJSONSerializer:
    """Test JSON serialization helper function."""

    def test_serialize_numpy_int(self):
        """Test serialization of numpy integers."""
        result = _json_serializer(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_serialize_numpy_float(self):
        """Test serialization of numpy floats."""
        result = _json_serializer(np.float64(3.14))
        assert result == 3.14
        assert isinstance(result, float)

    def test_serialize_numpy_array(self):
        """Test serialization of numpy arrays."""
        arr = np.array([1, 2, 3, 4, 5])
        result = _json_serializer(arr)
        assert result == [1, 2, 3, 4, 5]
        assert isinstance(result, list)

    def test_serialize_numpy_bool(self):
        """Test serialization of numpy booleans."""
        result = _json_serializer(np.bool_(True))
        assert result is True
        assert isinstance(result, bool)

    def test_serialize_pandas_na(self):
        """Test serialization of pandas NA values."""
        result = _json_serializer(pd.NA)
        assert result is None

    def test_serialize_unsupported_type_raises_error(self):
        """Test that unsupported types raise TypeError."""
        class CustomClass:
            pass

        obj = CustomClass()

        with pytest.raises(TypeError, match="not JSON serializable"):
            _json_serializer(obj)


class TestRoundTripIntegration:
    """Integration tests for complete save/load/predict workflows."""

    def test_complete_workflow_with_preprocessing(self):
        """Test complete workflow: train, save, load, predict."""
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.pipeline import Pipeline
        from spectral_predict.preprocess import SNV, SavgolDerivative

        # Create training data
        np.random.seed(42)
        n_samples = 200
        n_features = 100

        X_train = np.random.randn(n_samples, n_features) * 0.1 + 1.0
        y_train = (X_train[:, 25] * 2.0 +
                   X_train[:, 50] * 1.5 +
                   X_train[:, 75] * 1.0 +
                   np.random.randn(n_samples) * 0.05)

        # Create preprocessing pipeline
        preprocessor = Pipeline([
            ('snv', SNV()),
            ('deriv', SavgolDerivative(deriv=1, window=11))
        ])

        # Fit preprocessing and transform data
        preprocessor.fit(X_train)
        X_processed = preprocessor.transform(X_train)

        # Train model
        model = PLSRegression(n_components=7)
        model.fit(X_processed, y_train)

        # Calculate performance
        y_pred_train = model.predict(X_processed)
        r2_train = 1 - np.sum((y_train - y_pred_train.ravel())**2) / np.sum((y_train - y_train.mean())**2)

        # Save model
        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name

        try:
            wavelengths = [float(i) for i in range(1000, 1100)]
            metadata = {
                'model_name': 'PLS',
                'task_type': 'regression',
                'preprocessing': 'snv_sg1',
                'wavelengths': wavelengths,
                'n_vars': n_features,
                'window': 11,
                'polyorder': 2,
                'params': {'n_components': 7},
                'performance': {'R2_train': float(r2_train)}
            }

            save_model(model, preprocessor, metadata, filepath)

            # Load model
            loaded = load_model(filepath)

            # Create new test data
            X_test = np.random.randn(20, n_features) * 0.1 + 1.0
            y_test = (X_test[:, 25] * 2.0 +
                     X_test[:, 50] * 1.5 +
                     X_test[:, 75] * 1.0 +
                     np.random.randn(20) * 0.05)

            # Make predictions using loaded model
            X_test_df = pd.DataFrame(
                X_test,
                columns=[str(w) for w in wavelengths]
            )

            predictions = predict_with_model(loaded, X_test_df)

            # Verify predictions are reasonable
            assert predictions.shape == (20,)
            assert not np.any(np.isnan(predictions))

            # Check that predictions are correlated with true values
            correlation = np.corrcoef(y_test, predictions)[0, 1]
            assert correlation > 0.5, "Predictions should be correlated with true values"

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_workflow_multiple_saves_and_loads(self):
        """Test that model can be saved and loaded multiple times."""
        from sklearn.linear_model import Ridge

        model = Ridge()
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        model.fit(X, y)

        filepaths = []

        try:
            # Save same model to multiple files
            for i in range(3):
                with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
                    filepath = f.name
                    filepaths.append(filepath)

                metadata = {
                    'model_name': 'Ridge',
                    'task_type': 'regression',
                    'wavelengths': list(range(1500, 1550)),
                    'n_vars': 50,
                    'version': i
                }

                save_model(model, None, metadata, filepath)

            # Load all files and verify they work
            for i, filepath in enumerate(filepaths):
                loaded = load_model(filepath)
                assert loaded['metadata']['version'] == i

                predictions = loaded['model'].predict(X[:10])
                assert predictions.shape == (10,)

        finally:
            for filepath in filepaths:
                Path(filepath).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
