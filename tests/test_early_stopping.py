"""Tests for early stopping support in CV utilities."""

import numpy as np
import pytest
from sklearn.model_selection import KFold
from sklearn.datasets import make_regression, make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Import cv_utils functions
from spectral_predict.cv_utils import (
    cross_validate_with_early_stopping,
    cross_val_predict_with_early_stopping,
    cross_val_score_with_early_stopping,
    is_boosting_model,
    _fit_with_early_stopping,
)

# Import model classes
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


@pytest.fixture
def regression_data():
    """Create synthetic regression data."""
    X, y = make_regression(n_samples=100, n_features=50, noise=0.1, random_state=42)
    return X, y


@pytest.fixture
def classification_data():
    """Create synthetic classification data."""
    X, y = make_classification(n_samples=100, n_features=50, n_classes=2, random_state=42)
    return X, y


class TestIsBoostingModel:
    """Tests for is_boosting_model function."""

    def test_xgboost_regressor_is_boosting(self):
        model = XGBRegressor()
        assert is_boosting_model(model) is True

    def test_xgboost_classifier_is_boosting(self):
        model = XGBClassifier()
        assert is_boosting_model(model) is True

    def test_lightgbm_regressor_is_boosting(self):
        model = LGBMRegressor()
        assert is_boosting_model(model) is True

    def test_lightgbm_classifier_is_boosting(self):
        model = LGBMClassifier()
        assert is_boosting_model(model) is True

    def test_random_forest_not_boosting(self):
        model = RandomForestRegressor()
        assert is_boosting_model(model) is False

    def test_ridge_not_boosting(self):
        model = Ridge()
        assert is_boosting_model(model) is False


class TestCrossValidateWithEarlyStopping:
    """Tests for cross_validate_with_early_stopping function."""

    def test_xgboost_with_early_stopping(self, regression_data):
        """Test that XGBoost uses early stopping correctly."""
        X, y = regression_data
        model = XGBRegressor(n_estimators=100, random_state=42)
        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        results = cross_validate_with_early_stopping(
            model, X, y, cv=cv,
            scoring='neg_root_mean_squared_error',
            early_stopping_rounds=10
        )

        assert 'test_score' in results
        assert len(results['test_score']) == 3
        assert all(score < 0 for score in results['test_score'])  # RMSE is negated

    def test_ridge_without_early_stopping(self, regression_data):
        """Test that Ridge falls back to standard cross_validate."""
        X, y = regression_data
        model = Ridge()
        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        results = cross_validate_with_early_stopping(
            model, X, y, cv=cv,
            scoring='neg_root_mean_squared_error',
            early_stopping_rounds=10
        )

        assert 'test_score' in results
        assert len(results['test_score']) == 3

    def test_early_stopping_disabled(self, regression_data):
        """Test that early stopping can be disabled."""
        X, y = regression_data
        model = XGBRegressor(n_estimators=50, random_state=42)
        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        # With early stopping disabled (rounds=0)
        results = cross_validate_with_early_stopping(
            model, X, y, cv=cv,
            scoring='neg_root_mean_squared_error',
            early_stopping_rounds=0
        )

        assert 'test_score' in results
        assert len(results['test_score']) == 3

    def test_classification_with_early_stopping(self, classification_data):
        """Test early stopping with classification task."""
        X, y = classification_data
        model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        results = cross_validate_with_early_stopping(
            model, X, y, cv=cv,
            scoring='accuracy',
            early_stopping_rounds=10
        )

        assert 'test_score' in results
        assert len(results['test_score']) == 3
        assert all(0 <= score <= 1 for score in results['test_score'])  # Accuracy range


class TestCrossValPredictWithEarlyStopping:
    """Tests for cross_val_predict_with_early_stopping function."""

    def test_xgboost_predict(self, regression_data):
        """Test predictions with early stopping."""
        X, y = regression_data
        model = XGBRegressor(n_estimators=100, random_state=42)
        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        predictions = cross_val_predict_with_early_stopping(
            model, X, y, cv=cv,
            early_stopping_rounds=10
        )

        assert predictions.shape == y.shape

    def test_classification_predict_proba(self, classification_data):
        """Test probability predictions with early stopping."""
        X, y = classification_data
        model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        predictions = cross_val_predict_with_early_stopping(
            model, X, y, cv=cv,
            early_stopping_rounds=10,
            method='predict_proba'
        )

        assert predictions.shape == (len(y), 2)  # Binary classification
        assert all(0 <= p <= 1 for row in predictions for p in row)


class TestCrossValScoreWithEarlyStopping:
    """Tests for cross_val_score_with_early_stopping function."""

    def test_xgboost_score(self, regression_data):
        """Test scores with early stopping."""
        X, y = regression_data
        model = XGBRegressor(n_estimators=100, random_state=42)
        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        scores = cross_val_score_with_early_stopping(
            model, X, y, cv=cv,
            scoring='neg_root_mean_squared_error',
            early_stopping_rounds=10
        )

        assert len(scores) == 3
        assert all(score < 0 for score in scores)


class TestPipelineWithEarlyStopping:
    """Test early stopping with sklearn Pipeline."""

    def test_pipeline_with_scaler(self, regression_data):
        """Test that pipelines with preprocessing work correctly."""
        X, y = regression_data

        # Create a pipeline with scaler and XGBoost
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', XGBRegressor(n_estimators=100, random_state=42))
        ])

        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        results = cross_validate_with_early_stopping(
            pipeline, X, y, cv=cv,
            scoring='neg_root_mean_squared_error',
            early_stopping_rounds=10
        )

        assert 'test_score' in results
        assert len(results['test_score']) == 3


class TestLightGBMEarlyStopping:
    """Test early stopping specifically for LightGBM."""

    def test_lightgbm_regression(self, regression_data):
        """Test LightGBM with early stopping for regression."""
        X, y = regression_data
        model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        results = cross_validate_with_early_stopping(
            model, X, y, cv=cv,
            scoring='neg_root_mean_squared_error',
            early_stopping_rounds=10
        )

        assert 'test_score' in results
        assert len(results['test_score']) == 3

    def test_lightgbm_classification(self, classification_data):
        """Test LightGBM with early stopping for classification."""
        X, y = classification_data
        model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        results = cross_validate_with_early_stopping(
            model, X, y, cv=cv,
            scoring='accuracy',
            early_stopping_rounds=10
        )

        assert 'test_score' in results
        assert len(results['test_score']) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
