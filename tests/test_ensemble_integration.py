"""
Integration tests for ensemble save/load with preprocessing.

Tests the full pipeline: create ensemble -> save -> load -> predict
to ensure preprocessing and wavelength subsets are preserved correctly.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor

from src.spectral_predict.ensemble import (
    SimpleAverageEnsemble,
    RegionAwareWeightedEnsemble,
    create_ensemble,
    extract_preprocessor_config
)
from src.spectral_predict.preprocessing_wrapper import PreprocessorConfig
from src.spectral_predict.model_io import save_ensemble, load_ensemble


class TestEnsembleSaveLoad:
    """Test ensemble save/load roundtrip."""

    def test_save_load_simple_average_ensemble(self):
        """Test saving and loading SimpleAverageEnsemble."""
        # Create models
        model1 = PLSRegression(n_components=2)
        model2 = RandomForestRegressor(n_estimators=10, random_state=42)

        # Create and fit on training data
        X_train = np.random.randn(50, 20)
        y_train = np.random.randn(50)

        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)

        # Create ensemble
        ensemble = SimpleAverageEnsemble(
            models=[model1, model2],
            model_names=['PLS-2', 'RF-10']
        )

        # Get predictions before save
        X_test = np.random.randn(10, 20)
        pred_before = ensemble.predict(X_test)

        # Save ensemble
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'ensemble_test.dasp'

            metadata = {
                'ensemble_type': 'simple_average',
                'ensemble_name': 'Test Ensemble',
                'task_type': 'regression',
                'wavelengths': list(range(1500, 1520)),
                'n_vars': 20,
                'preprocessing': 'raw',
                'performance': {'R2': 0.95, 'RMSE': 0.12}
            }

            save_ensemble(ensemble, str(filepath), metadata)

            # Verify file was created
            assert filepath.exists()

            # Load ensemble
            loaded_dict = load_ensemble(str(filepath))

            # Verify loaded components
            assert 'ensemble' in loaded_dict
            assert 'metadata' in loaded_dict
            assert 'model_names' in loaded_dict

            loaded_ensemble = loaded_dict['ensemble']
            assert len(loaded_ensemble.models) == 2
            assert loaded_ensemble.model_names == ['PLS-2', 'RF-10']

            # Get predictions after load
            pred_after = loaded_ensemble.predict(X_test)

            # Predictions should be identical
            np.testing.assert_array_almost_equal(pred_before, pred_after)

    @pytest.mark.skip(reason="TODO: save_ensemble() does not yet persist preprocessor_configs")
    def test_save_load_ensemble_with_preprocessor_configs(self):
        """
        Test save/load ensemble with per-model preprocessing configs.

        NOTE: This test is currently skipped because save_ensemble() doesn't
        yet serialize preprocessor_configs. This is a future enhancement.
        For now, ensembles with preprocessor_configs work in memory but
        cannot be persisted and reloaded while maintaining the configs.
        """
        # Create models
        model1 = PLSRegression(n_components=2)
        model2 = PLSRegression(n_components=3)

        # Training data
        X_train = np.random.randn(50, 20)
        y_train = np.random.randn(50)

        # NOTE: Models should be trained on preprocessed data
        # For this test, we'll train on raw then wrap with configs
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)

        # Create preprocessor configs
        configs = [
            PreprocessorConfig(preprocess_name='raw'),
            PreprocessorConfig(preprocess_name='snv')
        ]

        # Create ensemble with configs
        ensemble = SimpleAverageEnsemble(
            models=[model1, model2],
            model_names=['PLS-Raw', 'PLS-SNV'],
            preprocessor_configs=configs
        )

        # Predictions before save
        X_test = np.random.randn(10, 20)
        pred_before = ensemble.predict(X_test)

        # Save ensemble
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'ensemble_with_configs.dasp'

            metadata = {
                'ensemble_type': 'simple_average',
                'ensemble_name': 'Preprocessed Ensemble',
                'task_type': 'regression',
                'wavelengths': list(range(1500, 1520)),
                'n_vars': 20,
                'preprocessing': 'mixed',
            }

            save_ensemble(ensemble, str(filepath), metadata)

            # Load and verify
            loaded_dict = load_ensemble(str(filepath))
            loaded_ensemble = loaded_dict['ensemble']

            # Predictions after load should match
            pred_after = loaded_ensemble.predict(X_test)
            np.testing.assert_array_almost_equal(pred_before, pred_after)


class TestResultsDFToEnsemble:
    """Test extracting configs from results DataFrame."""

    def test_extract_configs_from_results_df(self):
        """Test extracting preprocessor configs from results DataFrame."""
        # Simulate a results DataFrame like search would produce
        results_df = pd.DataFrame({
            'Model': ['PLS', 'PLS', 'PLS'],
            'Preprocess': ['raw', 'snv', 'snv_sg1'],
            'Deriv': [0, 0, 1],
            'Window': [15, 15, 11],
            'Poly': [2, 2, 2],
            'all_vars': ['N/A', 'N/A', '1500,1510,1520,1530'],
            'R2': [0.90, 0.92, 0.95],
            'RMSE': [0.15, 0.13, 0.10]
        })

        all_wavelengths = list(range(1500, 1550))

        # Extract configs
        configs = []
        for idx, row in results_df.iterrows():
            config = extract_preprocessor_config(row, all_wavelengths)
            configs.append(config)

        # Verify configs
        assert len(configs) == 3

        # Config 0: raw
        assert configs[0].preprocess_name == 'raw'
        assert configs[0].apply_snv is False
        assert configs[0].deriv == 0
        assert configs[0].wavelengths is None

        # Config 1: snv
        assert configs[1].preprocess_name == 'snv'
        assert configs[1].apply_snv is True
        assert configs[1].deriv == 0
        assert configs[1].wavelengths is None

        # Config 2: snv_sg1 with wavelength subset
        assert configs[2].preprocess_name == 'snv_sg1'
        assert configs[2].apply_snv is True
        assert configs[2].deriv == 1
        assert configs[2].wavelengths == [1500.0, 1510.0, 1520.0, 1530.0]

        # Test that configs can transform data
        X = np.random.randn(10, 50)

        X_out0 = configs[0].transform(X)
        assert X_out0.shape == (10, 50)  # Raw: no change

        X_out1 = configs[1].transform(X)
        assert X_out1.shape == (10, 50)  # SNV: same shape

        X_out2 = configs[2].transform(X)
        assert X_out2.shape == (10, 4)  # SNV + deriv + subset to 4 wavelengths


class TestEnsembleAttributeValidation:
    """Test save_ensemble() attribute validation."""

    def test_save_ensemble_validates_models_attribute(self):
        """Test that save_ensemble() validates models attribute."""
        # Create a fake ensemble without models attribute
        class FakeEnsemble:
            pass

        fake_ensemble = FakeEnsemble()

        metadata = {
            'ensemble_type': 'simple_average',
            'task_type': 'regression',
            'wavelengths': [1500, 1510],
            'n_vars': 2
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'fake.dasp'

            with pytest.raises(ValueError, match="models"):
                save_ensemble(fake_ensemble, str(filepath), metadata)

    def test_save_ensemble_auto_generates_model_names(self):
        """Test that save_ensemble() auto-generates missing model_names."""
        # Create ensemble without model_names
        model1 = PLSRegression(n_components=2)
        model2 = PLSRegression(n_components=3)

        X_train = np.random.randn(30, 10)
        y_train = np.random.randn(30)
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)

        ensemble = SimpleAverageEnsemble(
            models=[model1, model2],
            model_names=None  # Will be auto-generated
        )

        # Verify model_names were auto-generated in __init__
        assert ensemble.model_names == ['Model_0', 'Model_1']

        metadata = {
            'ensemble_type': 'simple_average',
            'task_type': 'regression',
            'wavelengths': list(range(1500, 1510)),
            'n_vars': 10
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'auto_names.dasp'

            # Should not raise, model_names exist
            save_ensemble(ensemble, str(filepath), metadata)

            # Load and verify
            loaded_dict = load_ensemble(str(filepath))
            assert loaded_dict['model_names'] == ['Model_0', 'Model_1']


class TestRegionAwareEnsembleWithConfigs:
    """Test RegionAwareWeightedEnsemble with preprocessor configs."""

    def test_region_weighted_ensemble_with_configs(self):
        """Test that RegionAwareWeightedEnsemble works with configs."""
        # Create models
        model1 = PLSRegression(n_components=2)
        model2 = PLSRegression(n_components=3)

        # Training data
        X_train = np.random.randn(100, 15)
        y_train = np.random.randn(100)

        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)

        # Create configs
        configs = [
            PreprocessorConfig(preprocess_name='raw'),
            PreprocessorConfig(preprocess_name='snv')
        ]

        # Create region-aware ensemble with configs
        ensemble = RegionAwareWeightedEnsemble(
            models=[model1, model2],
            model_names=['PLS-2', 'PLS-3'],
            n_regions=5,
            preprocessor_configs=configs
        )

        # Fit ensemble
        ensemble.fit(X_train, y_train)

        # Verify regional_weights_ was computed
        assert ensemble.regional_weights_ is not None
        assert ensemble.regional_weights_.shape == (2, 5)  # 2 models, 5 regions

        # Make predictions
        X_test = np.random.randn(20, 15)
        predictions = ensemble.predict(X_test)

        assert predictions.shape == (20,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
