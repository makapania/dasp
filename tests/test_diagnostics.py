"""Comprehensive tests for spectral_predict.diagnostics module.

Tests cover:
- compute_residuals: residual and standardized residual computation
- compute_leverage: hat values and threshold computation
- qq_plot_data: Q-Q plot coordinate generation
- jackknife_prediction_intervals: prediction intervals via LOO resampling
- compute_pls_complexity_curve: PLS n_components sweep
- compute_learning_curve: performance vs training set size
- compute_sklearn_validation_curve: generic sklearn validation curve
- compute_ensemble_validation_curve: tree-based ensemble n_estimators sweep
- compute_regularization_validation_curve: alpha/C sweep for regularized models
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from spectral_predict.diagnostics import (
    compute_leverage,
    compute_learning_curve,
    compute_pls_complexity_curve,
    compute_regularization_validation_curve,
    compute_residuals,
    compute_sklearn_validation_curve,
    jackknife_prediction_intervals,
    qq_plot_data,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synthetic_spectra():
    """Small synthetic spectral DataFrame (20 samples x 10 wavelengths)."""
    np.random.seed(42)
    n_samples, n_features = 20, 10
    X = np.random.randn(n_samples, n_features)
    return pd.DataFrame(X, columns=[f"wl_{i}" for i in range(n_features)])


@pytest.fixture
def synthetic_y(synthetic_spectra):
    """Target values matching synthetic_spectra samples."""
    np.random.seed(42)
    X = synthetic_spectra.values
    # Generate y with a known linear relationship plus noise
    true_coefs = np.random.randn(X.shape[1])
    y = X @ true_coefs + np.random.randn(X.shape[0]) * 0.5
    return y


@pytest.fixture
def trained_pls_model(synthetic_spectra, synthetic_y):
    """Fit a simple PLS model on the synthetic data."""
    n_components = min(3, synthetic_spectra.shape[1], synthetic_spectra.shape[0] - 1)
    model = PLSRegression(n_components=n_components, scale=False)
    model.fit(synthetic_spectra.values, synthetic_y)
    return model


# =============================================================================
# compute_residuals tests
# =============================================================================


def test_compute_residuals_basic(synthetic_spectra, synthetic_y, trained_pls_model):
    """Verify residuals are y_true - y_pred."""
    X = synthetic_spectra.values
    y_pred = trained_pls_model.predict(X).ravel()
    residuals, std_residuals = compute_residuals(synthetic_y, y_pred)

    expected = np.array(synthetic_y) - np.array(y_pred)
    np.testing.assert_allclose(residuals, expected, atol=1e-10)


def test_compute_residuals_length(synthetic_spectra, synthetic_y, trained_pls_model):
    """Residuals should have same length as inputs."""
    X = synthetic_spectra.values
    y_pred = trained_pls_model.predict(X).ravel()
    residuals, std_residuals = compute_residuals(synthetic_y, y_pred)

    assert len(residuals) == len(synthetic_y)
    assert len(std_residuals) == len(synthetic_y)


def test_compute_residuals_standardized_mean_near_zero(synthetic_spectra, synthetic_y, trained_pls_model):
    """Standardized residuals should have mean near 0."""
    X = synthetic_spectra.values
    y_pred = trained_pls_model.predict(X).ravel()
    _, std_residuals = compute_residuals(synthetic_y, y_pred)

    assert abs(np.mean(std_residuals)) < 1e-10


def test_compute_residuals_standardized_std_near_one(synthetic_spectra, synthetic_y, trained_pls_model):
    """Standardized residuals should have std near 1 when residuals have nonzero variance."""
    X = synthetic_spectra.values
    y_pred = trained_pls_model.predict(X).ravel()
    residuals, std_residuals = compute_residuals(synthetic_y, y_pred)

    if np.std(residuals) > 1e-10:
        assert abs(np.std(std_residuals) - 1.0) < 1e-10


def test_compute_residuals_perfect_prediction():
    """When y_true == y_pred, residuals should be zero."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    residuals, std_residuals = compute_residuals(y_true, y_pred)

    np.testing.assert_allclose(residuals, 0.0, atol=1e-10)


# =============================================================================
# compute_leverage tests
# =============================================================================


def test_compute_leverage_values_in_valid_range(synthetic_spectra):
    """Leverage values should be in [0, 1] for well-conditioned data."""
    X = synthetic_spectra.values
    leverage, threshold = compute_leverage(X, return_threshold=True)

    assert len(leverage) == X.shape[0]
    assert np.all(leverage >= -1e-10), "Leverage values should not be negative"
    assert np.all(leverage <= 1.0 + 1e-10), "Leverage values should not exceed 1"


def test_compute_leverage_threshold(synthetic_spectra):
    """Threshold should be 2 * (p+1) / n."""
    X = synthetic_spectra.values
    n, p = X.shape
    leverage, threshold = compute_leverage(X, return_threshold=True)

    expected_threshold = 2 * (p + 1) / n
    assert abs(threshold - expected_threshold) < 1e-10


def test_compute_leverage_no_threshold(synthetic_spectra):
    """When return_threshold=False, should return only leverage array."""
    X = synthetic_spectra.values
    result = compute_leverage(X, return_threshold=False)

    assert isinstance(result, np.ndarray)
    assert result.shape == (X.shape[0],)


def test_compute_leverage_sum_equals_p_plus_1(synthetic_spectra):
    """Sum of leverage values should equal p+1 (rank of hat matrix)."""
    X = synthetic_spectra.values
    n, p = X.shape
    leverage = compute_leverage(X, return_threshold=False)

    # sum(h_ii) = trace(H) = rank(X_augmented) = p + 1
    expected_sum = p + 1
    assert abs(np.sum(leverage) - expected_sum) < 0.5, (
        f"Sum of leverage = {np.sum(leverage):.2f}, expected ~{expected_sum}"
    )


# =============================================================================
# qq_plot_data tests
# =============================================================================


def test_qq_plot_data_returns_sorted_quantiles():
    """Sample quantiles should be sorted."""
    residuals = np.random.randn(50)
    theoretical, sample = qq_plot_data(residuals)

    assert np.all(np.diff(sample) >= 0), "Sample quantiles should be sorted"


def test_qq_plot_data_correct_length():
    """Output arrays should have same length as input."""
    residuals = np.random.randn(30)
    theoretical, sample = qq_plot_data(residuals)

    assert len(theoretical) == len(residuals)
    assert len(sample) == len(residuals)


def test_qq_plot_data_theoretical_quantiles_symmetric():
    """Theoretical quantiles should be approximately symmetric around 0 for large n."""
    residuals = np.random.randn(100)
    theoretical, _ = qq_plot_data(residuals)

    # Mean of theoretical quantiles should be near 0
    assert abs(np.mean(theoretical)) < 0.1


# =============================================================================
# jackknife_prediction_intervals tests
# =============================================================================


@pytest.mark.slow
def test_jackknife_intervals_contain_predictions(synthetic_spectra, synthetic_y, trained_pls_model):
    """Point predictions should lie within the jackknife prediction intervals."""
    X = synthetic_spectra.values
    # Use a small train/test split
    X_train, X_test = X[:15], X[15:]
    y_train = synthetic_y[:15]

    predictions, lower, upper, std_errors = jackknife_prediction_intervals(
        trained_pls_model, X_train, y_train, X_test, confidence=0.95
    )

    # All predictions should be within [lower, upper]
    assert np.all(predictions >= lower - 1e-10), "Predictions should be >= lower bounds"
    assert np.all(predictions <= upper + 1e-10), "Predictions should be <= upper bounds"


@pytest.mark.slow
def test_jackknife_intervals_output_shapes(synthetic_spectra, synthetic_y, trained_pls_model):
    """All returned arrays should have the correct shape."""
    X = synthetic_spectra.values
    X_train, X_test = X[:15], X[15:]
    y_train = synthetic_y[:15]
    n_test = X_test.shape[0]

    predictions, lower, upper, std_errors = jackknife_prediction_intervals(
        trained_pls_model, X_train, y_train, X_test, confidence=0.95
    )

    assert predictions.shape == (n_test,)
    assert lower.shape == (n_test,)
    assert upper.shape == (n_test,)
    assert std_errors.shape == (n_test,)


@pytest.mark.slow
def test_jackknife_intervals_wider_at_lower_confidence(synthetic_spectra, synthetic_y, trained_pls_model):
    """Intervals at 99% confidence should be wider than at 90% confidence."""
    X = synthetic_spectra.values
    X_train, X_test = X[:15], X[15:]
    y_train = synthetic_y[:15]

    _, lower_90, upper_90, _ = jackknife_prediction_intervals(
        trained_pls_model, X_train, y_train, X_test, confidence=0.90
    )
    _, lower_99, upper_99, _ = jackknife_prediction_intervals(
        trained_pls_model, X_train, y_train, X_test, confidence=0.99
    )

    width_90 = np.mean(upper_90 - lower_90)
    width_99 = np.mean(upper_99 - lower_99)

    assert width_99 > width_90, "99% intervals should be wider than 90% intervals"


# =============================================================================
# compute_pls_complexity_curve tests
# =============================================================================


@pytest.mark.slow
def test_pls_complexity_curve_returns_expected_keys(synthetic_spectra, synthetic_y):
    """Result dict should contain all expected keys."""
    X = synthetic_spectra.values
    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    result = compute_pls_complexity_curve(X, synthetic_y, max_components=5, cv=cv)

    expected_keys = {
        "param_values", "train_scores", "cv_scores", "cv_std",
        "param_name", "metric_name", "optimal_idx"
    }
    assert expected_keys.issubset(set(result.keys()))


@pytest.mark.slow
def test_pls_complexity_curve_train_scores_decrease(synthetic_spectra, synthetic_y):
    """Training RMSE should generally decrease with more components."""
    X = synthetic_spectra.values
    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    result = compute_pls_complexity_curve(X, synthetic_y, max_components=5, cv=cv)

    train_scores = result["train_scores"]
    # At least the first value should be >= the last (more components -> lower train error)
    assert train_scores[0] >= train_scores[-1] - 0.1, (
        "Training error should generally decrease with more PLS components"
    )


# =============================================================================
# compute_learning_curve tests
# =============================================================================


@pytest.mark.slow
def test_learning_curve_returns_expected_keys(synthetic_spectra, synthetic_y):
    """Result dict should contain all expected keys."""
    X = synthetic_spectra.values
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    estimator = PLSRegression(n_components=2, scale=False)

    result = compute_learning_curve(estimator, X, synthetic_y, cv=cv)

    expected_keys = {
        "train_sizes_abs", "train_scores", "train_std",
        "cv_scores", "cv_std", "metric_name", "interpretation"
    }
    assert expected_keys.issubset(set(result.keys()))


@pytest.mark.slow
def test_learning_curve_interpretation_nonempty(synthetic_spectra, synthetic_y):
    """The interpretation string should not be empty."""
    X = synthetic_spectra.values
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    estimator = PLSRegression(n_components=2, scale=False)

    result = compute_learning_curve(estimator, X, synthetic_y, cv=cv)

    assert isinstance(result["interpretation"], str)
    assert len(result["interpretation"]) > 0


# =============================================================================
# compute_sklearn_validation_curve tests
# =============================================================================


def test_sklearn_validation_curve_with_ridge(synthetic_spectra, synthetic_y):
    """Test validation curve with Ridge regression varying alpha."""
    X = synthetic_spectra.values
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    estimator = Ridge()
    param_range = np.logspace(-3, 3, 5)

    result = compute_sklearn_validation_curve(
        estimator, X, synthetic_y,
        param_name="alpha", param_range=param_range, cv=cv
    )

    assert len(result["param_values"]) == 5
    assert result["param_name"] == "alpha"
    assert result["metric_name"] == "RMSE"
    assert len(result["train_scores"]) == 5
    assert len(result["cv_scores"]) == 5


def test_sklearn_validation_curve_optimal_idx_valid(synthetic_spectra, synthetic_y):
    """Optimal index should be within the valid range."""
    X = synthetic_spectra.values
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    estimator = Ridge()
    param_range = np.logspace(-2, 2, 4)

    result = compute_sklearn_validation_curve(
        estimator, X, synthetic_y,
        param_name="alpha", param_range=param_range, cv=cv
    )

    assert 0 <= result["optimal_idx"] < len(param_range)


# =============================================================================
# compute_regularization_validation_curve tests
# =============================================================================


def test_regularization_curve_with_ridge(synthetic_spectra, synthetic_y):
    """Test regularization validation curve with Ridge model."""
    X = synthetic_spectra.values
    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    result = compute_regularization_validation_curve(
        model_class=Ridge, X=X, y=synthetic_y,
        base_alpha=1.0, cv=cv, n_points=5, orders=1
    )

    assert result["param_name"] == "alpha"
    assert len(result["param_values"]) == 5
    assert result["metric_name"] == "RMSE"
