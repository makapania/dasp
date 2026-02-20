"""
Unit tests for ensemble preprocessing configuration.

Tests the PreprocessorConfig class and extract_preprocessor_config() helper
to ensure they correctly reconstruct preprocessing from stored configuration.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor

from spectral_predict.preprocessing_wrapper import PreprocessorConfig
from spectral_predict.ensemble import (
    extract_preprocessor_config,
    SimpleAverageEnsemble,
    create_ensemble
)


class TestPreprocessorConfig:
    """Test PreprocessorConfig class."""

    def test_raw_preprocessing(self):
        """Test raw (no preprocessing) configuration."""
        config = PreprocessorConfig(preprocess_name='raw')

        X = np.random.randn(10, 20)
        X_out = config.transform(X)

        # Raw should not modify data
        np.testing.assert_array_almost_equal(X, X_out)

    def test_snv_preprocessing(self):
        """Test SNV preprocessing."""
        config = PreprocessorConfig(preprocess_name='snv')

        X = np.random.randn(10, 20) + 5.0  # Add offset
        X_out = config.transform(X)

        # SNV should center and scale each row
        assert X_out.shape == X.shape
        # Check that rows are approximately zero-mean
        row_means = np.mean(X_out, axis=1)
        np.testing.assert_array_almost_equal(row_means, np.zeros(10), decimal=10)
        # Check that standard deviation is close to 1 (may vary slightly depending on ddof)
        row_stds = np.std(X_out, axis=1)
        assert np.all(row_stds > 0.99) and np.all(row_stds < 1.1)

    def test_derivative_preprocessing(self):
        """Test derivative preprocessing."""
        config = PreprocessorConfig(
            preprocess_name='sg1',
            deriv=1,
            window=7,
            polyorder=2
        )

        X = np.random.randn(10, 20)
        X_out = config.transform(X)

        # Derivative should maintain sample count but features may differ
        assert X_out.shape[0] == X.shape[0]
        # With default SG window adjustment, shape should be preserved
        assert X_out.shape[1] == X.shape[1]

    def test_snv_plus_derivative(self):
        """Test combined SNV + derivative preprocessing."""
        config = PreprocessorConfig(
            preprocess_name='snv_sg2',
            deriv=2,
            window=9,
            polyorder=2
        )

        X = np.random.randn(10, 20) + 3.0
        X_out = config.transform(X)

        # Should apply both SNV and second derivative
        assert X_out.shape[0] == X.shape[0]
        assert X_out.shape[1] == X.shape[1]

    def test_wavelength_subsetting(self):
        """Test wavelength subsetting."""
        all_wavelengths = [1500.0, 1510.0, 1520.0, 1530.0, 1540.0, 1550.0]
        selected_wavelengths = [1500.0, 1520.0, 1540.0]  # Every other wavelength

        config = PreprocessorConfig(
            preprocess_name='raw',
            wavelengths=selected_wavelengths,
            all_wavelengths=all_wavelengths
        )

        X = np.random.randn(10, 6)  # 10 samples, 6 wavelengths
        X_out = config.transform(X)

        # Should subset to 3 wavelengths
        assert X_out.shape == (10, 3)
        # Verify correct columns were selected
        np.testing.assert_array_almost_equal(X_out[:, 0], X[:, 0])  # 1500
        np.testing.assert_array_almost_equal(X_out[:, 1], X[:, 2])  # 1520
        np.testing.assert_array_almost_equal(X_out[:, 2], X[:, 4])  # 1540

    def test_derivative_then_subset(self):
        """Test derivative + wavelength subsetting (correct order)."""
        all_wavelengths = list(range(1500, 1560, 2))  # 30 wavelengths
        selected_wavelengths = list(range(1500, 1560, 6))  # Every 3rd wavelength

        config = PreprocessorConfig(
            preprocess_name='sg1',
            deriv=1,
            window=7,
            wavelengths=selected_wavelengths,
            all_wavelengths=all_wavelengths
        )

        X = np.random.randn(10, 30)
        X_out = config.transform(X)

        # Should apply derivative to all 30, then subset to selected wavelengths
        assert X_out.shape[0] == 10
        assert X_out.shape[1] == len(selected_wavelengths)

    def test_get_config_round_trip(self):
        """Test that config can be serialized and restored."""
        config1 = PreprocessorConfig(
            preprocess_name='snv_sg1',
            deriv=1,
            window=11,
            polyorder=2,
            wavelengths=[1500.0, 1520.0],
            all_wavelengths=[1500.0, 1510.0, 1520.0]
        )

        # Get config dict
        config_dict = config1.get_config()

        # Reconstruct from dict
        config2 = PreprocessorConfig.from_config(config_dict)

        # Verify configs are equivalent
        assert config2.preprocess_name == config1.preprocess_name
        assert config2.deriv == config1.deriv
        assert config2.window == config1.window
        assert config2.polyorder == config1.polyorder
        assert config2.wavelengths == config1.wavelengths
        assert config2.all_wavelengths == config1.all_wavelengths


class TestExtractPreprocessorConfig:
    """Test extract_preprocessor_config() helper function."""

    def test_extract_raw_config(self):
        """Test extracting raw preprocessing config."""
        row = pd.Series({
            'Preprocess': 'raw',
            'Deriv': 0,
            'Window': 15,
            'Poly': 2,
            'all_vars': 'N/A'
        })
        all_wavelengths = list(range(1500, 1550))

        config = extract_preprocessor_config(row, all_wavelengths)

        assert config.preprocess_name == 'raw'
        assert config.deriv == 0
        assert config.wavelengths is None

    def test_extract_snv_config(self):
        """Test extracting SNV preprocessing config."""
        row = pd.Series({
            'Preprocess': 'snv',
            'Deriv': 0,
            'Window': 15,
            'Poly': 2,
            'all_vars': 'N/A'
        })
        all_wavelengths = list(range(1500, 1550))

        config = extract_preprocessor_config(row, all_wavelengths)

        assert config.preprocess_name == 'snv'
        assert config.apply_snv is True
        assert config.deriv == 0

    def test_extract_derivative_config(self):
        """Test extracting derivative preprocessing config."""
        row = pd.Series({
            'Preprocess': 'sg1',
            'Deriv': 1,
            'Window': 11,
            'Poly': 2,
            'all_vars': 'N/A'
        })
        all_wavelengths = list(range(1500, 1550))

        config = extract_preprocessor_config(row, all_wavelengths)

        assert config.preprocess_name == 'sg1'
        assert config.deriv == 1
        assert config.window == 11

    def test_extract_with_wavelength_subset(self):
        """Test extracting config with wavelength subset."""
        row = pd.Series({
            'Preprocess': 'snv_sg2',
            'Deriv': 2,
            'Window': 15,
            'Poly': 2,
            'all_vars': '1500,1510,1520,1530'  # Subset of wavelengths
        })
        all_wavelengths = list(range(1500, 1550))

        config = extract_preprocessor_config(row, all_wavelengths)

        assert config.preprocess_name == 'snv_sg2'
        assert config.apply_snv is True
        assert config.deriv == 2
        assert config.wavelengths == [1500.0, 1510.0, 1520.0, 1530.0]
        assert len(config.all_wavelengths) == 50

    def test_extract_handles_missing_all_vars(self):
        """Test that missing all_vars is handled gracefully."""
        row = pd.Series({
            'Preprocess': 'raw',
            'Deriv': 0,
            'Window': 15,
            'Poly': 2
            # all_vars is missing
        })
        all_wavelengths = list(range(1500, 1550))

        config = extract_preprocessor_config(row, all_wavelengths)

        # Should default to None (no subsetting)
        assert config.wavelengths is None


class TestSimpleAverageEnsembleWithConfigs:
    """Test SimpleAverageEnsemble with preprocessor configs."""

    def test_simple_average_with_configs(self):
        """Test that SimpleAverageEnsemble uses preprocessor_configs."""
        # Create two simple models
        model1 = PLSRegression(n_components=2)
        model2 = RandomForestRegressor(n_estimators=10, random_state=42)

        # Create training data
        X_train = np.random.randn(50, 20)
        y_train = np.random.randn(50)

        # Train models on raw data
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)

        # Create configs (one applies SNV, one doesn't)
        config1 = PreprocessorConfig(preprocess_name='snv')
        config2 = PreprocessorConfig(preprocess_name='raw')

        # Create ensemble with configs
        ensemble = SimpleAverageEnsemble(
            models=[model1, model2],
            model_names=['PLS', 'RF'],
            preprocessor_configs=[config1, config2]
        )

        # Make predictions
        X_test = np.random.randn(10, 20)
        predictions = ensemble.predict(X_test)

        # Verify shape
        assert predictions.shape == (10,)

        # Verify that config1 applies SNV by comparing predictions
        X_test_snv = config1.transform(X_test)
        pred1 = model1.predict(X_test_snv)
        pred2 = model2.predict(X_test)
        expected_avg = (pred1 + pred2) / 2

        np.testing.assert_array_almost_equal(predictions, expected_avg)


class TestCreateEnsembleWithConfigs:
    """Test create_ensemble() with preprocessor_configs."""

    def test_create_simple_average_with_configs(self):
        """Test creating simple average ensemble with configs."""
        # Create models
        model1 = PLSRegression(n_components=2)
        model2 = PLSRegression(n_components=3)

        # Create training data
        X_train = np.random.randn(50, 20)
        y_train = np.random.randn(50)

        # Train models
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)

        # Create configs
        configs = [
            PreprocessorConfig(preprocess_name='raw'),
            PreprocessorConfig(preprocess_name='snv')
        ]

        # Create ensemble using factory
        ensemble = create_ensemble(
            models=[model1, model2],
            model_names=['PLS-2', 'PLS-3'],
            X=X_train,
            y=y_train,
            ensemble_type='simple_average',
            preprocessor_configs=configs
        )

        # Verify ensemble was created
        assert isinstance(ensemble, SimpleAverageEnsemble)
        assert ensemble.preprocessor_configs == configs

        # Verify predictions work
        X_test = np.random.randn(10, 20)
        predictions = ensemble.predict(X_test)
        assert predictions.shape == (10,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
