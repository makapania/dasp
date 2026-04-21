"""Tests for refine task-type preservation (Fix A + Fix B1).

Verifies that:
- Saved Task from a results row is honored when loading into Model Development,
  preventing the PLS → PLS-DA regression hotfix.
- The auto-detect fallback (when Task is missing) no longer treats numeric y
  with <10 unique values as classification (the nunique()<10 heuristic removal).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _import_helper():
    from spectral_predict_gui_optimized import _detect_refine_task_type
    return _detect_refine_task_type


class TestDetectRefineTaskType:
    """Test the extracted _detect_refine_task_type helper."""

    @pytest.fixture(autouse=True)
    def _import(self):
        self.fn = _import_helper()

    # ------------------------------------------------------------------
    # Test 1:  Saved regression Task is honored despite few unique y values
    # ------------------------------------------------------------------
    def test_saved_regression_trusted_over_few_unique(self):
        y = pd.Series([100, 200, 300, 400, 500])  # 5 unique numeric values
        config = {"Task": "regression", "Model": "PLS"}
        task, inlier = self.fn(config, y)
        assert task == "regression"
        assert inlier is None

    # ------------------------------------------------------------------
    # Test 2:  Saved classification Task is honored even with many unique y
    # ------------------------------------------------------------------
    def test_saved_classification_trusted_over_many_unique(self):
        rng = np.random.default_rng(42)
        y = pd.Series(rng.uniform(0, 100, size=200))
        config = {"Task": "classification", "Model": "PLS-DA"}
        task, inlier = self.fn(config, y)
        assert task == "classification"
        assert inlier is None

    # ------------------------------------------------------------------
    # Test 3:  Saved one_class Task + inlier label restoration
    # ------------------------------------------------------------------
    def test_saved_one_class_with_inlier_label(self):
        y = pd.Series(["burned", "burned", "unburned"])
        config = {
            "Task": "one_class",
            "Model": "PCA-SIMCA",
            "inlier_class_label": "burned",
        }
        task, inlier = self.fn(config, y)
        assert task == "one_class"
        assert inlier == "burned"

    def test_saved_one_class_with_inlier_class_key(self):
        y = pd.Series(["burned", "burned", "unburned"])
        config = {
            "Task": "one_class",
            "Model": "PCA-SIMCA",
            "inlier_class": "burned",
        }
        task, inlier = self.fn(config, y)
        assert task == "one_class"
        assert inlier == "burned"

    def test_saved_one_class_no_inlier_label(self):
        y = pd.Series(["A", "A", "B"])
        config = {"Task": "one_class", "Model": "PCA-SIMCA"}
        task, inlier = self.fn(config, y)
        assert task == "one_class"
        assert inlier is None

    # ------------------------------------------------------------------
    # Test 4:  No Task key → fallback auto-detect gives regression for
    #          numeric y with many unique values
    # ------------------------------------------------------------------
    def test_no_task_fallback_regression_many_unique(self):
        rng = np.random.default_rng(42)
        y = pd.Series(rng.uniform(0, 100, size=200))
        config = {"Model": "PLS"}
        task, inlier = self.fn(config, y)
        assert task == "regression"
        assert inlier is None

    # ------------------------------------------------------------------
    # Test 5 (Fix B):  No Task key, numeric y with only 5 unique values
    #                  → regression (not classification)
    # ------------------------------------------------------------------
    def test_no_task_fallback_regression_few_unique_numeric(self):
        y = pd.Series([100, 200, 300, 400, 500, 600, 700, 800])
        config = {"Model": "PLS"}
        task, inlier = self.fn(config, y)
        assert task == "regression"
        assert inlier is None

    # ------------------------------------------------------------------
    # Test 6:  No Task key, binary numeric y → classification
    # ------------------------------------------------------------------
    def test_no_task_fallback_classification_binary(self):
        y = pd.Series([0, 1, 0, 1, 1, 0])
        config = {"Model": "PLS-DA"}
        task, inlier = self.fn(config, y)
        assert task == "classification"
        assert inlier is None

    # ------------------------------------------------------------------
    # Test 7:  No Task key, non-numeric y → classification
    # ------------------------------------------------------------------
    def test_no_task_fallback_classification_non_numeric(self):
        y = pd.Series(["cat", "dog", "cat", "dog", "bird"])
        config = {"Model": "PLS-DA"}
        task, inlier = self.fn(config, y)
        assert task == "classification"
        assert inlier is None

    # ------------------------------------------------------------------
    # Test 8:  No Task, no y → regression
    # ------------------------------------------------------------------
    def test_no_task_no_y_defaults_regression(self):
        config = {"Model": "PLS"}
        task, inlier = self.fn(config, None)
        assert task == "regression"
        assert inlier is None

    # ------------------------------------------------------------------
    # Test 9:  Empty string Task treated same as missing
    # ------------------------------------------------------------------
    def test_empty_task_falls_through(self):
        y = pd.Series([100, 200, 300, 400, 500])
        config = {"Task": "", "Model": "PLS"}
        task, inlier = self.fn(config, y)
        assert task == "regression"
        assert inlier is None

    # ------------------------------------------------------------------
    # Test 10:  Burned-temperature scenario (the original user bug)
    # ------------------------------------------------------------------
    def test_burned_temperature_scenario(self):
        y = pd.Series([100, 200, 300, 400, 500, 600, 700, 800] * 3)
        config = {"Task": "regression", "Model": "PLS"}
        task, inlier = self.fn(config, y)
        assert task == "regression"
        assert inlier is None
