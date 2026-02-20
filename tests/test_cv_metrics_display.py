"""Test that progress display uses CV metrics."""
import numpy as np
import pandas as pd
import io
import sys

import pytest


def test_progress_shows_cv_metrics_regression():
    """Verify progress callback receives CV metrics for regression."""
    from spectral_predict.search import run_search

    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 100))
    y = pd.Series(X.iloc[:, 0] * 2 + np.random.randn(50) * 0.1)

    progress_updates = []
    def capture(info):
        if 'best_model' in info and info['best_model']:
            progress_updates.append(info['best_model'].copy())

    results_df, _ = run_search(
        X, y, task_type="regression", folds=3,
        models_to_test=["PLS"],
        preprocessing_methods={"raw": True},
        enable_variable_subsets=False,
        progress_callback=capture,
    )

    assert len(progress_updates) > 0, "Should have progress updates"
    best = progress_updates[-1]
    assert 'RMSEcv' in best, "Should have RMSEcv"
    assert 'R2cv' in best, "Should have R2cv"


def test_progress_shows_cv_metrics_classification():
    """Verify progress callback receives CV metrics for classification."""
    from spectral_predict.search import run_search

    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(60, 100))
    y = pd.Series([0] * 30 + [1] * 30)

    progress_updates = []
    def capture(info):
        if 'best_model' in info and info['best_model']:
            progress_updates.append(info['best_model'].copy())

    results_df, _ = run_search(
        X, y, task_type="classification", folds=3,
        models_to_test=["PLS-DA"],
        preprocessing_methods={"raw": True},
        enable_variable_subsets=False,
        progress_callback=capture,
    )

    assert len(progress_updates) > 0, "Should have progress updates"
    best = progress_updates[-1]
    assert 'Accuracycv' in best, "Should have Accuracycv"
    assert 'ROC_AUCcv' in best, "Should have ROC_AUCcv"


def test_console_shows_cv_metrics_regression():
    """Verify console output shows CV metrics for regression."""
    from spectral_predict.search import run_search

    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 100))
    y = pd.Series(X.iloc[:, 0] * 2 + np.random.randn(50) * 0.1)

    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()
    try:
        run_search(
            X, y, task_type="regression", folds=3,
            models_to_test=["PLS"],
            preprocessing_methods={"raw": True},
            enable_variable_subsets=False,
        )
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue()
    # After changes, should show CV metrics
    assert "RMSEcv" in output, f"Console should mention RMSEcv, got: {output[:500]}"
    assert "R²cv" in output, f"Console should mention R²cv, got: {output[:500]}"


def test_console_shows_cv_metrics_classification():
    """Verify console output shows CV metrics for classification."""
    from spectral_predict.search import run_search

    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(60, 100))
    y = pd.Series([0] * 30 + [1] * 30)

    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()
    try:
        run_search(
            X, y, task_type="classification", folds=3,
            models_to_test=["PLS-DA"],
            preprocessing_methods={"raw": True},
            enable_variable_subsets=False,
        )
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue()
    # After changes, should show CV metrics
    assert "AUCcv" in output, f"Console should mention AUCcv, got: {output[:500]}"
    assert "Acccv" in output, f"Console should mention Acccv, got: {output[:500]}"


def test_best_model_tracking_uses_cv_metrics():
    """Verify best model comparison uses CV metrics, not training metrics."""
    from spectral_predict.search import run_search

    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 100))
    y = pd.Series(X.iloc[:, 0] * 2 + np.random.randn(50) * 0.1)

    # Track best models over time
    best_models = []
    def capture(info):
        if 'best_model' in info and info['best_model']:
            best_models.append(info['best_model'].copy())

    # Run with multiple hyperparameters to ensure comparison happens
    results_df, _ = run_search(
        X, y, task_type="regression", folds=3,
        models_to_test=["PLS"],
        preprocessing_methods={"raw": True},
        enable_variable_subsets=False,
        progress_callback=capture,
    )

    assert len(best_models) > 0, "Should have tracked best models"

    # Check that best model has CV metrics
    for model in best_models:
        assert 'RMSEcv' in model, "Best model should have RMSEcv"
        assert 'R2cv' in model, "Best model should have R2cv"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
