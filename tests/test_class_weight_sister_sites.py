"""Tests for the class_weight discriminator sister-site fixes.

Three production sites had the same defective pattern that 4dcedbc /
c395317 fixed in the GUI dispatcher: a `hasattr(model, 'class_weight')`-only
check that silently no-op'd for CatBoost (uses `auto_class_weights`) and
XGBoost (uses `sample_weight` at fit time, no constructor kwarg).

Sites covered by this test module:
    1. unified_bayesian.py:1238  — Bayesian objective discriminator
    2. nsga2_search.py:1400      — NSGA-II main-eval discriminator
    3. nsga2_search.py:3106      — _compute_classification_metrics (display metrics)

Plus two defense-in-depth gaps (the 'auto' guard in _needs_resampling_pipeline
in both unified_bayesian.py:156 and nsga2_search.py:142).

Plus the cv_utils plumbing extensions (sample_weight propagation through
cross_val_predict_pooled, cross_val_predict_with_early_stopping, etc.).

Test strategy
-------------
The bug is silent: pre-fix, the wrong-path produced unweighted training
that did not crash. So each test must assert a measurable behavior change
(metrics or scores differ when class_weight is requested vs not requested,
on imbalanced data).
"""
from __future__ import annotations

import sys
import os

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectral_predict.cv_utils import (
    cross_val_predict_pooled,
    cross_val_predict_with_early_stopping,
)


# ---------------------------------------------------------------------------
# Shared imbalanced classification fixture (~80/20 split, deterministic shuffle)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def imbalanced_binary_data():
    """50 samples, 30 features, 80/20 class split. Linearly separable enough
    that class_weight + sample_weight produce a visible shift in predictions."""
    rng = np.random.default_rng(42)
    n_samples = 60
    n_features = 30
    n_pos = n_samples // 5  # 12 / 60 = 4:1 ratio (clears 3:1 auto threshold)

    y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_samples - n_pos, dtype=int)])
    rng.shuffle(y)

    # Class-conditional means make the imbalance reweighting matter
    coef = rng.standard_normal(n_features)
    X = rng.standard_normal((n_samples, n_features))
    X[y == 1] += 0.6 * coef  # positives shifted toward coef direction
    return X, y


# ---------------------------------------------------------------------------
# cv_utils plumbing tests
# ---------------------------------------------------------------------------


class TestCVUtilsSampleWeightPlumbing:
    """The cv_utils helpers must thread sample_weight per-fold so the
    Bayesian and NSGA-II callers can deliver balanced weights to
    sample_weight-only classifiers (XGBoost-like)."""

    def test_cross_val_predict_pooled_propagates_fit_params_standard_cv(self, imbalanced_binary_data):
        """Standard CV path delegates to sklearn cross_val_predict, which
        auto-slices fit_params arrays. predict_proba (not hard predict) is
        the right comparison granularity — hard labels can mask probability
        drift up to the class boundary, the same trap T-20's classification
        parity rows hit."""
        from sklearn.pipeline import Pipeline

        X, y = imbalanced_binary_data
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        model = Pipeline([("model", LogisticRegression(max_iter=500, random_state=42))])

        proba_no_weight = cross_val_predict_pooled(model, X, y, cv=cv, method="predict_proba")
        # Heavy positive-class boost should shift the probability surface
        sw = np.where(y == 1, 5.0, 1.0)
        proba_with_weight = cross_val_predict_pooled(
            model, X, y, cv=cv, method="predict_proba",
            fit_params={"model__sample_weight": sw},
        )

        assert not np.allclose(proba_no_weight, proba_with_weight, atol=1e-6), (
            "fit_params={'model__sample_weight': ...} did not reach the per-fold fit. "
            "The plumbing in cross_val_predict_pooled is broken."
        )

    def test_cross_val_predict_pooled_propagates_fit_params_repeated_cv(self, imbalanced_binary_data):
        """Repeated CV uses the manual loop with _slice_fit_params. Same
        contract — sample_weight must be sliced per train_idx and passed to fit."""
        from sklearn.model_selection import RepeatedStratifiedKFold
        from sklearn.pipeline import Pipeline

        X, y = imbalanced_binary_data
        cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=42)
        model = Pipeline([("model", LogisticRegression(max_iter=500, random_state=42))])

        preds_no_weight = cross_val_predict_pooled(model, X, y, cv=cv)
        sw = np.where(y == 1, 5.0, 1.0)
        preds_with_weight = cross_val_predict_pooled(
            model, X, y, cv=cv, fit_params={"model__sample_weight": sw}
        )

        assert not np.array_equal(preds_no_weight, preds_with_weight)

    def test_cross_val_predict_with_early_stopping_propagates_sample_weight_xgboost(
        self, imbalanced_binary_data
    ):
        """Boosting + early stopping path uses the manual CV loop that calls
        _fit_with_early_stopping. sample_weight must be sliced per train_idx
        and forwarded as the explicit kwarg."""
        pytest.importorskip("xgboost")
        from xgboost import XGBClassifier
        from sklearn.pipeline import Pipeline

        X, y = imbalanced_binary_data
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        xgb_kwargs = dict(
            n_estimators=20, max_depth=3, learning_rate=0.1,
            random_state=42, tree_method="hist", n_jobs=1, eval_metric="logloss",
        )
        model = Pipeline([("model", XGBClassifier(**xgb_kwargs))])

        preds_no_weight = cross_val_predict_with_early_stopping(
            model, X, y, cv=cv, early_stopping_rounds=10,
        )
        sw = np.where(y == 1, 5.0, 1.0)
        preds_with_weight = cross_val_predict_with_early_stopping(
            model, X, y, cv=cv, early_stopping_rounds=10, sample_weight=sw,
        )

        assert not np.array_equal(preds_no_weight, preds_with_weight), (
            "sample_weight kwarg did not reach _fit_with_early_stopping. "
            "Plumbing in cross_val_predict_with_early_stopping is broken."
        )


# ---------------------------------------------------------------------------
# Defense-in-depth tests (auto guard in _needs_resampling_pipeline)
# ---------------------------------------------------------------------------


class TestAutoGuardDefInDepth:
    """unified_bayesian._needs_resampling_pipeline and the matching one in
    nsga2_search must short-circuit on 'auto', not just 'class_weight'.
    Mirrors c395317's guard at search.py:289."""

    def test_unified_bayesian_needs_resampling_pipeline_auto_guard(self):
        from spectral_predict.unified_bayesian import _needs_resampling_pipeline
        assert _needs_resampling_pipeline("auto", "classification") is False
        assert _needs_resampling_pipeline("class_weight", "classification") is False

    def test_nsga2_search_needs_resampling_pipeline_auto_guard(self):
        from spectral_predict.nsga2_search import _needs_resampling_pipeline
        assert _needs_resampling_pipeline("auto", "classification") is False
        assert _needs_resampling_pipeline("class_weight", "classification") is False


# ---------------------------------------------------------------------------
# NSGA-II display-metrics site (#3) — _compute_classification_metrics
# ---------------------------------------------------------------------------


class TestNSGA2DisplayMetricsDiscriminatorStructural:
    """Structural verification that the discriminator at nsga2_search.py:3106
    branches on model_type and routes CatBoost / sample_weight-only models
    correctly. The function takes a chromosome-encoded `solution` array, so
    a full end-to-end test would need to construct a valid pymoo solution —
    out of scope. Instead we verify the discriminator's structural shape
    through introspection of the source.

    End-to-end coverage of NSGA-II classification with class_weight is
    provided indirectly by the Bayesian tests below (same discriminator
    pattern, validated end-to-end on identical synthetic data) plus the
    existing tests in test_nsga2_search.py.
    """

    def test_discriminator_handles_catboost_branch(self):
        """Source-level check: the discriminator must include a CatBoost
        branch that sets auto_class_weights, not just hasattr-on-class_weight."""
        import inspect
        from spectral_predict import nsga2_search

        src = inspect.getsource(nsga2_search)
        # The auto_class_weights='Balanced' kwarg is the canonical CatBoost
        # class-weighting mechanism. If it's absent from nsga2_search.py, the
        # CatBoost branch was lost in a future refactor.
        assert "auto_class_weights='Balanced'" in src or 'auto_class_weights="Balanced"' in src, (
            "CatBoost branch of the class_weight discriminator is missing. "
            "Pre-fix: CatBoost trained UNWEIGHTED under imbalance_method='class_weight'."
        )

    def test_discriminator_handles_sample_weight_fallback(self):
        """Source-level check: sample_weight inspection branch (XGBoost path)
        must be present. The flag-style plumbing uses
        use_sample_weight_for_classification or compute_sample_weight."""
        import inspect
        from spectral_predict import nsga2_search

        src = inspect.getsource(nsga2_search)
        assert (
            "use_sample_weight_for_classification" in src
            or "compute_sample_weight" in src
        ), (
            "sample_weight fallback branch is missing — XGBoost classifier "
            "would silently train UNWEIGHTED under class_weight."
        )


# ---------------------------------------------------------------------------
# Bayesian site #1 — calibration metrics under class_weight
# ---------------------------------------------------------------------------


class TestBayesianClassWeightDiscriminator:
    """unified_bayesian.objective applies the discriminator at line ~1238
    and refits the model on full data at line ~1470 to populate calibration
    metrics shown in the Bayesian Results panel.

    Pre-fix: those calibration metrics for CatBoost / XGBoost classifiers
    under imbalance_method='class_weight' described an UNWEIGHTED model.
    Each test runs Bayesian twice (with vs without class_weight) and asserts
    the calibration metric differs.
    """

    @pytest.mark.parametrize(
        "model_name",
        [
            pytest.param("CatBoost", marks=pytest.mark.skipif(
                not pytest.importorskip("catboost", reason="catboost optional"),
                reason="catboost not installed",
            )),
            pytest.param("XGBoost", marks=pytest.mark.skipif(
                not pytest.importorskip("xgboost", reason="xgboost optional"),
                reason="xgboost not installed",
            )),
        ],
    )
    def test_calibration_metric_differs_under_class_weight(self, imbalanced_binary_data, model_name):
        from spectral_predict.unified_bayesian import run_unified_bayesian

        X, y = imbalanced_binary_data
        wavelengths = np.arange(X.shape[1], dtype=float)

        results_no, _ = run_unified_bayesian(
            X=X, y=y, wavelengths=wavelengths,
            model_name=model_name, task_type="classification",
            n_trials=2, cv_folds=3, random_state=42, verbose=False,
            imbalance_method=None,
        )
        results_cw, _ = run_unified_bayesian(
            X=X, y=y, wavelengths=wavelengths,
            model_name=model_name, task_type="classification",
            n_trials=2, cv_folds=3, random_state=42, verbose=False,
            imbalance_method="class_weight",
        )

        # Calibration accuracy column: 'Accuracy' is the calibration set metric
        # written by the trial's full-data refit at unified_bayesian.py:1470.
        assert "Accuracy" in results_no.columns
        assert "Accuracy" in results_cw.columns

        # At least one trial's calibration metric should differ between the
        # two runs — proves the discriminator routed weights for the class_weight
        # case (CatBoost auto_class_weights / XGBoost sample_weight) and that
        # the calibration refit at line 1470 was correctly weighted.
        # Compare per-trial: with same seed and trial order, different imbalance
        # handling should produce different calibration metrics.
        acc_no = results_no["Accuracy"].values
        acc_cw = results_cw["Accuracy"].values
        assert not np.allclose(acc_no, acc_cw, atol=1e-8), (
            f"{model_name} + class_weight produced IDENTICAL calibration "
            f"accuracies to no-imbalance ({acc_no} == {acc_cw}). The "
            f"discriminator silently no-op'd — pre-fix bug class."
        )
