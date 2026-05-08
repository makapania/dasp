"""Regression tests for the 2026-05-07 fix-of-fixes:

1. ``predict_with_uncertainty`` no longer silently swallows OC
   decision_function failures. The result dict now carries a
   ``decision_score_error`` key with the error string (or None).

2. The OC validation helper at
   ``compute_validation_metrics_for_top_one_class_models`` now falls
   back to ``preprocess.SAVGOL_POLYORDER_DEFAULTS`` (matching the
   training pipeline) instead of the older ``min(2, window-1)``
   heuristic, which gave the WRONG polyorder for 2nd-derivative grid
   rows (poly=2 used by validation vs poly=3 used by training).
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fix 1: predict_with_uncertainty surfaces decision_score_error
# ---------------------------------------------------------------------------


class _RaisingOC:
    """Synthetic one-class model whose decision_function always raises."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.ones(len(X), dtype=int)

    def decision_function(self, X):
        raise RuntimeError("synthetic decision_function failure for the test")


class _SilentOC:
    """One-class model with neither decision_function nor score_samples."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.ones(len(X), dtype=int)


class TestDecisionScoreErrorSurfacing:
    def _make_model_dict(self, model):
        # Minimal model_dict shape that predict_with_uncertainty accepts for
        # task_type='one_class'. The function calls predict_with_model first
        # (which uses these fields), then accesses internals['X_processed']
        # to extract decision scores.
        wavelengths = np.linspace(1000.0, 2000.0, 5)
        X_train = np.random.RandomState(0).normal(size=(20, 5))
        model.fit(X_train)
        return {
            "model": model,
            "preprocessor": None,
            "metadata": {
                "task_type": "one_class",
                "wavelengths": wavelengths.tolist(),
                "model_name": type(model).__name__,
                "inlier_label": "clean",
                "data_type": "calibration",
            },
        }

    def test_raising_decision_function_populates_error_string(self, caplog):
        from spectral_predict.model_io import predict_with_uncertainty

        model_dict = self._make_model_dict(_RaisingOC())
        X_new = pd.DataFrame(
            np.random.RandomState(1).normal(size=(3, 5)),
            columns=[1000.0, 1250.0, 1500.0, 1750.0, 2000.0],
        )

        with caplog.at_level(logging.WARNING, logger="spectral_predict.model_io"):
            result = predict_with_uncertainty(model_dict, X_new, validate_wavelengths=False)

        assert "decision_score_error" in result
        err = result["decision_score_error"]
        assert err is not None
        assert "RuntimeError" in err
        assert "synthetic decision_function failure" in err
        # logger.warning fired with a useful message
        assert any(
            "OC decision-score extraction failed" in rec.getMessage()
            for rec in caplog.records
        )

    def test_missing_score_methods_populates_error_string(self):
        from spectral_predict.model_io import predict_with_uncertainty

        model_dict = self._make_model_dict(_SilentOC())
        X_new = pd.DataFrame(
            np.random.RandomState(2).normal(size=(3, 5)),
            columns=[1000.0, 1250.0, 1500.0, 1750.0, 2000.0],
        )

        result = predict_with_uncertainty(model_dict, X_new, validate_wavelengths=False)

        assert "decision_score_error" in result
        err = result["decision_score_error"]
        assert err is not None
        assert "_SilentOC" in err
        assert "decision_function" in err
        assert "score_samples" in err

    def test_working_decision_function_leaves_error_none(self):
        from spectral_predict.model_io import predict_with_uncertainty
        from sklearn.ensemble import IsolationForest

        rng = np.random.RandomState(3)
        X_train = rng.normal(size=(40, 5))
        model = IsolationForest(n_estimators=20, random_state=0).fit(X_train)
        wavelengths = np.linspace(1000.0, 2000.0, 5)
        model_dict = {
            "model": model,
            "preprocessor": None,
            "metadata": {
                "task_type": "one_class",
                "wavelengths": wavelengths.tolist(),
                "model_name": "IsolationForest",
                "inlier_label": "clean",
                "data_type": "calibration",
            },
        }
        X_new = pd.DataFrame(
            rng.normal(size=(3, 5)),
            columns=[1000.0, 1250.0, 1500.0, 1750.0, 2000.0],
        )

        result = predict_with_uncertainty(model_dict, X_new, validate_wavelengths=False)

        assert "decision_score_error" in result
        # Either None (clean run) or the structured error flag — never silently absent.
        assert result["decision_score_error"] is None
        assert result["has_uncertainty"] is True


# ---------------------------------------------------------------------------
# Fix 2: validation helper uses SavgolDerivative's polyorder_map for fallback
# ---------------------------------------------------------------------------


class TestValidationHelperPolyorderMatchesTraining:
    def test_polyorder_constant_exported(self):
        from spectral_predict.preprocess import SAVGOL_POLYORDER_DEFAULTS

        # Source of truth — must contain at least the deriv orders the
        # validation helper might encounter from grid-search rows.
        assert SAVGOL_POLYORDER_DEFAULTS[0] == 1
        assert SAVGOL_POLYORDER_DEFAULTS[1] == 2
        assert SAVGOL_POLYORDER_DEFAULTS[2] == 3  # the bug-fix case
        assert SAVGOL_POLYORDER_DEFAULTS[3] == 4
        assert SAVGOL_POLYORDER_DEFAULTS[4] == 5

    def test_savgolderivative_uses_module_constant(self):
        # Belt-and-suspenders: confirm the production transformer reads from
        # the same module constant. If a future refactor reintroduces a
        # local copy, this test fails immediately.
        from spectral_predict.preprocess import SavgolDerivative, SAVGOL_POLYORDER_DEFAULTS

        rng = np.random.RandomState(0)
        X = rng.normal(size=(10, 21))

        # deriv=2 with no explicit polyorder MUST use SAVGOL_POLYORDER_DEFAULTS[2] = 3
        sg = SavgolDerivative(deriv=2, window=11, polyorder=None)
        # Apply via fit/transform — if polyorder=2 had been used, savgol_filter
        # would have produced different values. We don't assert exact values
        # (those are scipy-version-dependent); we assert the equivalence with
        # an explicit polyorder=3 instance.
        sg_explicit = SavgolDerivative(deriv=2, window=11, polyorder=3)
        out_default = sg.fit_transform(X)
        out_explicit = sg_explicit.fit_transform(X)
        np.testing.assert_allclose(out_default, out_explicit)

        # And confirm that polyorder=2 produces a DIFFERENT result, so the
        # default-vs-2 distinction is observable (sanity check that the bug
        # was real).
        sg_old_fallback = SavgolDerivative(deriv=2, window=11, polyorder=2)
        out_old = sg_old_fallback.fit_transform(X)
        assert not np.allclose(out_default, out_old)

    def test_validation_helper_uses_correct_polyorder_for_deriv2(self):
        """The validation helper at compute_validation_metrics_for_top_one_class_models
        used to fall back to min(2, window-1) which gave poly=2 for deriv=2,
        but training uses poly=3. This test pins the corrected behavior by
        running a grid-search-style row with Poly=None and verifying the
        helper produces the SAME val_* metrics as if Poly=3 had been
        explicitly set.
        """
        from spectral_predict.contamination import (
            compute_validation_metrics_for_top_one_class_models,
        )

        rng = np.random.RandomState(0)
        n_train, n_val, n_features = 60, 20, 50
        X_train = rng.normal(size=(n_train, n_features))
        y_train = np.array(["clean"] * 50 + ["dirty"] * 10)
        X_val = rng.normal(size=(n_val, n_features))
        y_val = np.array(["clean"] * 15 + ["dirty"] * 5)
        wavelengths = np.linspace(1000.0, 2000.0, n_features)

        # Build two minimal one-row result frames — one with Poly=None
        # (the grid-search shape), one with Poly=3 (the explicit shape).
        # Both should produce the SAME val_* metrics now.
        # all_vars carries the wavelength list as a comma-separated string;
        # the validation helper requires it to map back to column indices.
        all_vars_str = ",".join(f"{w:.4f}" for w in wavelengths)
        common_row = {
            "Model": "IsolationForest",
            "Preprocess": "deriv2_w11",
            "PreprocessBase": "deriv",
            "Deriv": 2,
            "Window": 11,
            "Variables": "all",
            "all_vars": all_vars_str,
            "Params": "{'n_estimators': 50, 'contamination': 0.05}",
            "n_estimators": 50,
            "contamination": 0.05,
            "n_vars": n_features,
            "BalancedAcccv": 0.5,
            "Sensitivitycv": 0.5,
            "Specificitycv": 0.5,
            "Rank": 1,
        }
        df_none = pd.DataFrame([{**common_row, "Poly": None}])
        df_three = pd.DataFrame([{**common_row, "Poly": 3}])

        out_none = compute_validation_metrics_for_top_one_class_models(
            df_results=df_none,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            inlier_label="clean",
            wavelengths=wavelengths,
            top_n=1,
        )
        out_three = compute_validation_metrics_for_top_one_class_models(
            df_results=df_three,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            inlier_label="clean",
            wavelengths=wavelengths,
            top_n=1,
        )

        # Both runs must produce val_* columns.
        for col in ("val_BalancedAcc", "val_Sensitivity", "val_Specificity"):
            assert col in out_none.columns, f"{col} missing from Poly=None result"
            assert col in out_three.columns, f"{col} missing from Poly=3 result"

        # And the values must match — which means Poly=None correctly
        # resolved to poly=3 (matching SavgolDerivative training default),
        # not poly=2 (the old buggy fallback).
        for col in ("val_BalancedAcc", "val_Sensitivity", "val_Specificity"):
            v_none = out_none[col].iloc[0]
            v_three = out_three[col].iloc[0]
            if pd.isna(v_none) and pd.isna(v_three):
                continue
            assert v_none == pytest.approx(v_three, abs=1e-9), (
                f"{col} differs between Poly=None and Poly=3: "
                f"{v_none} vs {v_three}. The validation helper's polyorder "
                f"fallback no longer matches training's polyorder_map."
            )
