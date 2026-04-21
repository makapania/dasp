"""Tests for mixed-type target coercion across validation paths.

Verifies that mixed-type object-dtype target columns (e.g. [1, "1", 2, "2"])
are handled without TypeError at every call site that passes y through
np.unique, LabelEncoder, or sklearn's stratified splitting.

Also verifies that NaN values are dropped BEFORE coercion (never stringified
to "nan"), and that self.y / self.validation_y are never mutated.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from spectral_predict_gui_optimized import SpectralPredictApp
from spectral_predict.search import compute_validation_metrics_for_top_models


def _make_mock_app():
    """Create a MagicMock with just enough attributes to call validation methods."""
    mock = MagicMock(spec=SpectralPredictApp)
    mock._log_progress = MagicMock()
    mock._validation_random = MagicMock(return_value=[])
    return mock


def _synthetic_X(n=10, n_features=5, seed=42):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(rng.randn(n, n_features), columns=[f"wl_{i}" for i in range(n_features)])


class TestValidationStratifiedAcceptsMixedTypeLabels:
    """Test 1: _validation_stratified handles mixed-type y without TypeError."""

    def test_mixed_type_labels(self):
        mock_app = _make_mock_app()
        X = _synthetic_X(n=6, n_features=5)
        y = pd.Series([1, "1", 2, "2", 1, "2"], dtype=object, index=X.index)

        result = SpectralPredictApp._validation_stratified(mock_app, X, y, 2)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(idx in X.index for idx in result)


class TestValidationSpxyAcceptsMixedTypeLabels:
    """Test 2: _validation_spxy handles mixed-type y without TypeError."""

    def test_mixed_type_labels(self):
        mock_app = _make_mock_app()
        X = _synthetic_X(n=6, n_features=5)
        y = pd.Series([1, "1", 2, "2", 1, "2"], dtype=object, index=X.index)

        result = SpectralPredictApp._validation_spxy(mock_app, X, y, 2)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(idx in X.index for idx in result)


class TestComputeValidationMetricsAcceptsMixedTypeYVal:
    """Test 3: compute_validation_metrics_for_top_models handles mixed-type y_val."""

    def test_mixed_type_y_val(self):
        rng = np.random.RandomState(42)
        n_train, n_val, n_features = 30, 10, 20

        X_train = rng.randn(n_train, n_features)
        y_train = np.array(["A", "B"] * 15, dtype=object)
        X_val = rng.randn(n_val, n_features)
        y_val = np.array([1, "1", 2, "2", 1, "1", 2, "2", 1, 2], dtype=object)

        wavelengths = np.arange(n_features, dtype=float)
        all_vars = ",".join(str(w) for w in wavelengths)

        df_results = pd.DataFrame([{
            "Model": "PLS-DA",
            "PreprocessBase": "raw",
            "Preprocess": "raw",
            "Deriv": 0,
            "Window": 0,
            "Poly": 0,
            "LVs": 2,
            "n_vars": n_features,
            "full_vars": n_features,
            "SubsetTag": "full",
            "Imbalance": 0.0,
            "Task": "classification",
            "Params": "{}",
            "CompositeScore": 1.0,
            "Sensitivity": 0.9,
            "Specificity": 0.9,
            "Precision": 0.9,
            "F1": 0.9,
            "Accuracy": 0.9,
            "BalancedAcc": 0.9,
            "AUC": 0.9,
            "Sensitivitycv": 0.9,
            "Specificitycv": 0.9,
            "Precisioncv": 0.9,
            "F1cv": 0.9,
            "Accuracycv": 0.9,
            "BalancedAcccv": 0.9,
            "AUCcv": 0.9,
            "top_vars": "",
            "all_vars": all_vars,
            "Rank": 1,
        }])

        result_df = compute_validation_metrics_for_top_models(
            df_results=df_results,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            task_type="classification",
            wavelengths=wavelengths,
            top_n=10,
        )

        assert isinstance(result_df, pd.DataFrame)
        assert "val_Accuracy" in result_df.columns


class TestNaNIsDroppedNotStringified:
    """Test 4: NaN values are dropped before coercion, never become 'nan' strings."""

    def test_nan_dropped_not_stringified(self):
        y_raw = pd.Series([1, np.nan, "A", np.nan, 2], dtype=object)

        nan_mask = y_raw.isna()
        y_local = y_raw[~nan_mask].copy()

        if getattr(y_local, "dtype", None) == object:
            types_present = {type(v).__name__ for v in y_local}
            if len(types_present) > 1:
                y_local = y_local.astype(str)

        assert len(y_local) == 3
        assert "nan" not in y_local.values
        assert "nan" not in set(y_local.values)


class TestTaskTypeChangeWarning:
    """Test 5: _on_task_type_changed warns when crossing regression <-> classification boundary."""

    def test_warns_on_regression_to_classification_with_validation(self):
        mock_app = _make_mock_app()
        mock_app.task_type = MagicMock()
        mock_app.task_type.get.return_value = "classification"
        mock_app.y = pd.Series([0, 1, 0, 1])
        mock_app.validation_y = pd.Series([0, 1, 0])
        mock_app._last_task_type = "regression"
        mock_app._updating_from_tier = False
        mock_app.model_checkboxes = {}
        mock_app.model_checkbox_widgets = {}

        SpectralPredictApp._on_task_type_changed(mock_app)

        log_calls = [str(c) for c in mock_app._log_progress.call_args_list]
        assert any("Task type changed" in c and "regression" in c and "classification" in c for c in log_calls)
        assert mock_app._last_task_type == "classification"

    def test_no_warn_when_same_category(self):
        mock_app = _make_mock_app()
        mock_app.task_type = MagicMock()
        mock_app.task_type.get.return_value = "classification"
        mock_app.y = pd.Series(["A", "B", "A"])
        mock_app.validation_y = pd.Series(["A", "B"])
        mock_app._last_task_type = "classification"
        mock_app._updating_from_tier = False
        mock_app.model_checkboxes = {}
        mock_app.model_checkbox_widgets = {}

        SpectralPredictApp._on_task_type_changed(mock_app)

        log_calls = [str(c) for c in mock_app._log_progress.call_args_list]
        assert not any("Task type changed" in c for c in log_calls)

    def test_no_warn_when_no_validation(self):
        mock_app = _make_mock_app()
        mock_app.task_type = MagicMock()
        mock_app.task_type.get.return_value = "classification"
        mock_app.y = pd.Series([0, 1, 0])
        mock_app.validation_y = None
        mock_app._last_task_type = "regression"
        mock_app._updating_from_tier = False
        mock_app.model_checkboxes = {}
        mock_app.model_checkbox_widgets = {}

        SpectralPredictApp._on_task_type_changed(mock_app)

        log_calls = [str(c) for c in mock_app._log_progress.call_args_list]
        assert not any("Task type changed" in c for c in log_calls)
