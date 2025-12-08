"""
Comprehensive tests for hyperparameter validation.

Ensures that all hyperparameters defined in model_config.py:
1. Are syntactically valid and work with sklearn models
2. Actually affect model predictions (changing params changes behavior)
3. Work correctly in AUTO search mode
4. Handle incompatible combinations properly

Test coverage includes:
- All model types (PLS, Ridge, Lasso, ElasticNet, RandomForest, LightGBM, XGBoost, CatBoost, MLP, etc.)
- Hyperparameter syntax validation
- Parameter range validation
- Effect on predictions
- Custom hyperparameter grids in AUTO mode
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
from sklearn.model_selection import cross_val_score, KFold

from spectral_predict_v3.core.model_config import (
    get_hyperparameter_grid,
    get_tier_models,
    HYPERPARAMETER_GRIDS
)
from spectral_predict_v3.core.models import get_model
from spectral_predict_v3.core.search import run_auto_search


class TestHyperparameterSyntax:
    """Test that all hyperparameters are syntactically valid."""

    @pytest.fixture
    def sample_data_regression(self):
        """Generate sample regression data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 50

        X = np.random.randn(n_samples, n_features)
        # Create target with some relationship to X
        y = X[:, :5].sum(axis=1) + np.random.randn(n_samples) * 0.1

        return X, y

    @pytest.fixture
    def sample_data_classification(self):
        """Generate sample classification data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 50

        X = np.random.randn(n_samples, n_features)
        # Create binary target
        y = (X[:, :5].sum(axis=1) > 0).astype(int)

        return X, y

    def test_pls_hyperparameters(self, sample_data_regression):
        """Test all PLS hyperparameters work."""
        X, y = sample_data_regression
        grid = get_hyperparameter_grid('PLS')

        for params in grid:
            model = get_model('PLS', 'regression', **params)
            assert model is not None
            model.fit(X, y)
            predictions = model.predict(X)
            assert predictions.shape == y.shape

    def test_ridge_hyperparameters(self, sample_data_regression):
        """Test all Ridge hyperparameters work."""
        X, y = sample_data_regression
        grid = get_hyperparameter_grid('Ridge')

        for params in grid:
            model = get_model('Ridge', 'regression', **params)
            assert model is not None
            model.fit(X, y)
            predictions = model.predict(X)
            assert predictions.shape == y.shape

    def test_lasso_hyperparameters(self, sample_data_regression):
        """Test all Lasso hyperparameters work."""
        X, y = sample_data_regression
        grid = get_hyperparameter_grid('Lasso')

        for params in grid:
            model = get_model('Lasso', 'regression', **params)
            assert model is not None
            model.fit(X, y)
            predictions = model.predict(X)
            assert predictions.shape == y.shape

    def test_elasticnet_hyperparameters(self, sample_data_regression):
        """Test all ElasticNet hyperparameters work."""
        X, y = sample_data_regression
        grid = get_hyperparameter_grid('ElasticNet')

        for params in grid:
            model = get_model('ElasticNet', 'regression', **params)
            assert model is not None
            model.fit(X, y)
            predictions = model.predict(X)
            assert predictions.shape == y.shape

    def test_randomforest_hyperparameters(self, sample_data_regression):
        """Test all RandomForest hyperparameters work."""
        X, y = sample_data_regression
        grid = get_hyperparameter_grid('RandomForest')

        for params in grid:
            model = get_model('RandomForest', 'regression', **params)
            assert model is not None
            model.fit(X, y)
            predictions = model.predict(X)
            assert predictions.shape == y.shape

    def test_lightgbm_hyperparameters(self, sample_data_regression):
        """Test all LightGBM hyperparameters work (if available)."""
        X, y = sample_data_regression
        grid = get_hyperparameter_grid('LightGBM')

        for params in grid:
            model = get_model('LightGBM', 'regression', **params)
            if model is None:
                pytest.skip("LightGBM not available")
            model.fit(X, y)
            predictions = model.predict(X)
            assert predictions.shape == y.shape

    def test_xgboost_hyperparameters(self, sample_data_regression):
        """Test all XGBoost hyperparameters work (if available)."""
        X, y = sample_data_regression
        grid = get_hyperparameter_grid('XGBoost')

        for params in grid:
            model = get_model('XGBoost', 'regression', **params)
            if model is None:
                pytest.skip("XGBoost not available")
            model.fit(X, y)
            predictions = model.predict(X)
            assert predictions.shape == y.shape

    def test_catboost_hyperparameters(self, sample_data_regression):
        """Test all CatBoost hyperparameters work (if available)."""
        X, y = sample_data_regression
        grid = get_hyperparameter_grid('CatBoost')

        for params in grid:
            model = get_model('CatBoost', 'regression', **params)
            if model is None:
                pytest.skip("CatBoost not available")
            model.fit(X, y)
            predictions = model.predict(X)
            assert predictions.shape == y.shape

    def test_mlp_hyperparameters(self, sample_data_regression):
        """Test all MLP hyperparameters work."""
        X, y = sample_data_regression
        grid = get_hyperparameter_grid('MLP')

        for params in grid:
            model = get_model('MLP', 'regression', **params)
            assert model is not None
            model.fit(X, y)
            predictions = model.predict(X)
            assert predictions.shape == y.shape

    def test_plsda_hyperparameters(self, sample_data_classification):
        """Test all PLS-DA hyperparameters work."""
        X, y = sample_data_classification
        grid = get_hyperparameter_grid('PLS-DA')

        for params in grid:
            model = get_model('PLS-DA', 'classification', **params)
            assert model is not None
            model.fit(X, y)
            predictions = model.predict(X)
            assert predictions.shape == y.shape


class TestHyperparameterRanges:
    """Test that hyperparameter values are within valid ranges."""

    def test_positive_alpha_ridge(self):
        """Test that Ridge alpha is positive."""
        grid = get_hyperparameter_grid('Ridge')

        for params in grid:
            assert params['alpha'] > 0, f"Ridge alpha must be positive, got {params['alpha']}"

    def test_positive_alpha_lasso(self):
        """Test that Lasso alpha is positive."""
        grid = get_hyperparameter_grid('Lasso')

        for params in grid:
            assert params['alpha'] > 0, f"Lasso alpha must be positive, got {params['alpha']}"

    def test_elasticnet_l1_ratio_range(self):
        """Test that ElasticNet l1_ratio is between 0 and 1."""
        grid = get_hyperparameter_grid('ElasticNet')

        for params in grid:
            assert 0 <= params['l1_ratio'] <= 1, \
                f"ElasticNet l1_ratio must be in [0,1], got {params['l1_ratio']}"

    def test_pls_n_components_positive(self):
        """Test that PLS n_components is positive."""
        grid = get_hyperparameter_grid('PLS')

        for params in grid:
            assert params['n_components'] >= 1, \
                f"PLS n_components must be >= 1, got {params['n_components']}"

    def test_randomforest_n_estimators_positive(self):
        """Test that RandomForest n_estimators is positive."""
        grid = get_hyperparameter_grid('RandomForest')

        for params in grid:
            assert params['n_estimators'] >= 1, \
                f"RandomForest n_estimators must be >= 1, got {params['n_estimators']}"

    def test_lightgbm_learning_rate_range(self):
        """Test that LightGBM learning_rate is in valid range."""
        grid = get_hyperparameter_grid('LightGBM')

        for params in grid:
            assert 0 < params['learning_rate'] <= 1, \
                f"LightGBM learning_rate must be in (0,1], got {params['learning_rate']}"

    def test_mlp_alpha_nonnegative(self):
        """Test that MLP alpha (L2 regularization) is non-negative."""
        grid = get_hyperparameter_grid('MLP')

        for params in grid:
            assert params['alpha'] >= 0, \
                f"MLP alpha must be >= 0, got {params['alpha']}"


class TestHyperparameterEffects:
    """Test that changing hyperparameters actually affects model behavior."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression data with clear signal."""
        np.random.seed(42)
        n_samples = 200
        n_features = 100

        X = np.random.randn(n_samples, n_features)
        # Strong signal in first 10 features
        true_coef = np.zeros(n_features)
        true_coef[:10] = np.random.randn(10) * 2
        y = X @ true_coef + np.random.randn(n_samples) * 0.5

        return X, y

    def test_pls_components_affect_predictions(self, regression_data):
        """Test that changing PLS n_components changes predictions."""
        X, y = regression_data

        model_2 = get_model('PLS', 'regression', n_components=2)
        model_10 = get_model('PLS', 'regression', n_components=10)

        model_2.fit(X, y)
        model_10.fit(X, y)

        pred_2 = model_2.predict(X)
        pred_10 = model_10.predict(X)

        # Predictions should be different
        assert not np.allclose(pred_2, pred_10), \
            "Changing PLS n_components should affect predictions"

    def test_ridge_alpha_affects_predictions(self, regression_data):
        """Test that changing Ridge alpha changes predictions."""
        X, y = regression_data

        model_low = get_model('Ridge', 'regression', alpha=0.001)
        model_high = get_model('Ridge', 'regression', alpha=100.0)

        model_low.fit(X, y)
        model_high.fit(X, y)

        pred_low = model_low.predict(X)
        pred_high = model_high.predict(X)

        # Predictions should be different
        assert not np.allclose(pred_low, pred_high), \
            "Changing Ridge alpha should affect predictions"

    def test_lasso_alpha_affects_predictions(self, regression_data):
        """Test that changing Lasso alpha changes predictions."""
        X, y = regression_data

        model_low = get_model('Lasso', 'regression', alpha=0.001)
        model_high = get_model('Lasso', 'regression', alpha=1.0)

        model_low.fit(X, y)
        model_high.fit(X, y)

        pred_low = model_low.predict(X)
        pred_high = model_high.predict(X)

        # Predictions should be different
        assert not np.allclose(pred_low, pred_high), \
            "Changing Lasso alpha should affect predictions"

    def test_elasticnet_l1_ratio_affects_predictions(self, regression_data):
        """Test that changing ElasticNet l1_ratio changes predictions."""
        X, y = regression_data

        model_ridge = get_model('ElasticNet', 'regression', alpha=0.1, l1_ratio=0.01)
        model_lasso = get_model('ElasticNet', 'regression', alpha=0.1, l1_ratio=0.99)

        model_ridge.fit(X, y)
        model_lasso.fit(X, y)

        pred_ridge = model_ridge.predict(X)
        pred_lasso = model_lasso.predict(X)

        # Predictions should be different
        assert not np.allclose(pred_ridge, pred_lasso), \
            "Changing ElasticNet l1_ratio should affect predictions"

    def test_randomforest_depth_affects_predictions(self, regression_data):
        """Test that changing RandomForest max_depth changes predictions."""
        X, y = regression_data

        model_shallow = get_model('RandomForest', 'regression',
                                  n_estimators=50, max_depth=3)
        model_deep = get_model('RandomForest', 'regression',
                              n_estimators=50, max_depth=None)

        model_shallow.fit(X, y)
        model_deep.fit(X, y)

        pred_shallow = model_shallow.predict(X)
        pred_deep = model_deep.predict(X)

        # Predictions should be different
        assert not np.allclose(pred_shallow, pred_deep, atol=0.01), \
            "Changing RandomForest max_depth should affect predictions"

    def test_mlp_hidden_layers_affect_predictions(self, regression_data):
        """Test that changing MLP hidden_layer_sizes changes predictions."""
        X, y = regression_data

        model_small = get_model('MLP', 'regression',
                               hidden_layer_sizes=(10,), alpha=0.001, max_iter=1000)
        model_large = get_model('MLP', 'regression',
                               hidden_layer_sizes=(100, 50), alpha=0.001, max_iter=1000)

        model_small.fit(X, y)
        model_large.fit(X, y)

        pred_small = model_small.predict(X)
        pred_large = model_large.predict(X)

        # Predictions should be different
        assert not np.allclose(pred_small, pred_large, atol=0.1), \
            "Changing MLP hidden_layer_sizes should affect predictions"


class TestAutoModeCustomHyperparams:
    """Test that AUTO mode correctly uses custom hyperparameter grids."""

    @pytest.fixture
    def small_dataset(self):
        """Small dataset for quick AUTO search tests."""
        np.random.seed(42)
        n_samples = 50
        n_features = 30

        X = np.random.randn(n_samples, n_features)
        y = X[:, :3].sum(axis=1) + np.random.randn(n_samples) * 0.1

        return X, y

    def test_custom_pls_grid_in_auto(self, small_dataset):
        """Test that custom PLS hyperparameters are used in AUTO mode."""
        X, y = small_dataset

        # Define custom grid with specific n_components
        custom_grid = {
            'PLS': [
                {'n_components': 3},
                {'n_components': 5},
                {'n_components': 7}
            ]
        }

        # Run AUTO search with custom grid
        results = run_auto_search(
            X, y,
            task_type='regression',
            tier='quick',
            folds=3,
            custom_models=['PLS'],
            custom_hyperparam_grids=custom_grid,
            preproc_methods=['raw']  # Only test raw for speed
        )

        # Extract tested n_components
        pls_results = results[results['Model'] == 'PLS']
        tested_components = []
        for params_str in pls_results['Params']:
            if 'n_components=3' in params_str:
                tested_components.append(3)
            elif 'n_components=5' in params_str:
                tested_components.append(5)
            elif 'n_components=7' in params_str:
                tested_components.append(7)

        # Verify only [3, 5, 7] were tested
        assert set(tested_components) == {3, 5, 7}, \
            f"Expected only n_components [3, 5, 7], but tested {set(tested_components)}"

    def test_custom_ridge_grid_in_auto(self, small_dataset):
        """Test that custom Ridge hyperparameters are used in AUTO mode."""
        X, y = small_dataset

        # Define custom grid with specific alphas
        custom_grid = {
            'Ridge': [
                {'alpha': 0.5},
                {'alpha': 2.0}
            ]
        }

        # Run AUTO search with custom grid
        results = run_auto_search(
            X, y,
            task_type='regression',
            tier='quick',
            folds=3,
            custom_models=['Ridge'],
            custom_hyperparam_grids=custom_grid,
            preproc_methods=['raw']
        )

        # Extract tested alphas
        ridge_results = results[results['Model'] == 'Ridge']
        tested_alphas = []
        for params_str in ridge_results['Params']:
            if 'alpha=0.5' in params_str:
                tested_alphas.append(0.5)
            elif 'alpha=2' in params_str:
                tested_alphas.append(2.0)

        # Verify only [0.5, 2.0] were tested
        assert set(tested_alphas) == {0.5, 2.0}, \
            f"Expected only alpha [0.5, 2.0], but tested {set(tested_alphas)}"

    def test_pls_max_lv_parameter(self, small_dataset):
        """Test that pls_max_lv parameter generates correct grid."""
        X, y = small_dataset

        # Test with max_lv=5 (should test all from 1 to 5)
        results = run_auto_search(
            X, y,
            task_type='regression',
            tier='quick',
            folds=3,
            custom_models=['PLS'],
            pls_max_lv=5,
            preproc_methods=['raw']
        )

        # Should have tested n_components from 1 to 5
        pls_results = results[results['Model'] == 'PLS']

        # We should have at least one result per component value
        assert len(pls_results) >= 5, \
            f"Expected at least 5 PLS results (1 per component), got {len(pls_results)}"


class TestModelCompatibility:
    """Test model compatibility and constraint handling."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = X[:, :5].sum(axis=1) + np.random.randn(100) * 0.1
        return X, y

    def test_pls_max_components_constraint(self, sample_data):
        """Test that PLS n_components <= min(n_samples, n_features)."""
        X, y = sample_data
        n_samples, n_features = X.shape

        # n_components should be clipped to valid range
        max_valid = min(n_samples, n_features) - 1

        # Try to create PLS with too many components
        model = get_model('PLS', 'regression', n_components=max_valid + 10)

        # Should still fit (sklearn will clip or error gracefully)
        try:
            model.fit(X, y)
            # If it fits, verify it used valid number of components
            actual_components = model.n_components
            assert actual_components <= max_valid
        except ValueError:
            # sklearn raises ValueError for invalid n_components - this is OK
            pass

    def test_elasticnet_l1_ratio_bounds(self, sample_data):
        """Test that ElasticNet l1_ratio is bounded [0, 1]."""
        X, y = sample_data

        # All defined l1_ratio values should be valid
        grid = get_hyperparameter_grid('ElasticNet')
        for params in grid:
            assert 0 <= params['l1_ratio'] <= 1


class TestCrossValidationCompatibility:
    """Test that models work correctly in cross-validation."""

    @pytest.fixture
    def cv_data(self):
        """Generate data for cross-validation tests."""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = X[:, :5].sum(axis=1) + np.random.randn(100) * 0.1
        return X, y

    def test_pls_cross_validation(self, cv_data):
        """Test PLS works in cross-validation."""
        X, y = cv_data

        for n_comp in [2, 5, 10]:
            model = get_model('PLS', 'regression', n_components=n_comp)
            cv = KFold(n_splits=3, shuffle=True, random_state=42)

            scores = cross_val_score(model, X, y, cv=cv,
                                    scoring='neg_mean_squared_error')

            assert len(scores) == 3
            assert all(np.isfinite(scores))

    def test_ridge_cross_validation(self, cv_data):
        """Test Ridge works in cross-validation."""
        X, y = cv_data

        for alpha in [0.01, 1.0, 100.0]:
            model = get_model('Ridge', 'regression', alpha=alpha)
            cv = KFold(n_splits=3, shuffle=True, random_state=42)

            scores = cross_val_score(model, X, y, cv=cv,
                                    scoring='neg_mean_squared_error')

            assert len(scores) == 3
            assert all(np.isfinite(scores))

    def test_mlp_cross_validation(self, cv_data):
        """Test MLP works in cross-validation."""
        X, y = cv_data

        model = get_model('MLP', 'regression',
                         hidden_layer_sizes=(50,),
                         alpha=0.001,
                         max_iter=500)
        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        scores = cross_val_score(model, X, y, cv=cv,
                                scoring='neg_mean_squared_error')

        assert len(scores) == 3
        assert all(np.isfinite(scores))


def run_tests():
    """Run all tests using pytest."""
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'pytest', __file__, '-v'],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT)
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
