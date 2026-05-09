"""Tests for CV-ANOVA p-value (Eriksson, Trygg & Wold 2008).

Covers compute_cv_anova_pvalue() in scoring.py plus end-to-end column
landing through run_search (grid path) and run_unified_bayesian
(Bayesian path).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import f as f_dist
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict

from spectral_predict.scoring import compute_cv_anova_pvalue


# ---------------------------------------------------------------------------
# Unit tests on the helper directly
# ---------------------------------------------------------------------------


def _make_pls_rmsecv(X, y, n_components, cv=5):
    """Helper: return pooled RMSEcv for a PLS regression fit at n_components."""
    pls = PLSRegression(n_components=n_components, scale=False)
    y_pred_cv = cross_val_predict(pls, X, y, cv=cv).ravel()
    return float(np.sqrt(np.mean((y - y_pred_cv) ** 2)))


def test_high_signal_returns_small_pvalue():
    rng = np.random.default_rng(42)
    n, p = 50, 100
    X = rng.standard_normal((n, p))
    y = X[:, 5] + 0.01 * rng.standard_normal(n)
    rmsecv = _make_pls_rmsecv(X, y, n_components=2, cv=5)
    p_value = compute_cv_anova_pvalue(y_true=y, rmsecv=rmsecv, n_components=2)
    assert 0.0 <= p_value <= 1.0
    assert p_value < 0.001, f"high-signal p-value should be tiny, got {p_value}"


def test_no_signal_returns_large_pvalue():
    rng = np.random.default_rng(7)
    n, p = 60, 100
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)  # uncorrelated with X
    rmsecv = _make_pls_rmsecv(X, y, n_components=2, cv=5)
    p_value = compute_cv_anova_pvalue(y_true=y, rmsecv=rmsecv, n_components=2)
    assert 0.0 <= p_value <= 1.0
    assert p_value > 0.5, f"no-signal p-value should be large, got {p_value}"


def test_press_ge_ssy_clips_to_one():
    # Construct: PRESS = N * RMSEcv**2 = 100, SSY = sum((y-ymean)**2) = 50.
    # PRESS > SSY → F < 0 → clipped to 0 → p = 1.0.
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    # SSY for arange(10) = sum((i - 4.5)**2) = 82.5; RMSEcv chosen to make PRESS > SSY.
    big_rmsecv = float(np.sqrt(2 * 82.5 / len(y)))  # PRESS = 2 * SSY
    p_value = compute_cv_anova_pvalue(y_true=y, rmsecv=big_rmsecv, n_components=2)
    assert p_value == 1.0


def test_over_parametrised_returns_nan():
    y = np.linspace(0, 1, 5)  # N=5
    p_value = compute_cv_anova_pvalue(y_true=y, rmsecv=0.1, n_components=4)  # N - A - 1 = 0
    assert np.isnan(p_value)


def test_zero_components_returns_nan():
    y = np.linspace(0, 1, 30)
    assert np.isnan(compute_cv_anova_pvalue(y_true=y, rmsecv=0.1, n_components=0))
    assert np.isnan(compute_cv_anova_pvalue(y_true=y, rmsecv=0.1, n_components=-1))
    assert np.isnan(compute_cv_anova_pvalue(y_true=y, rmsecv=0.1, n_components=None))


def test_zero_variance_y_returns_nan():
    y = np.full(30, 5.0)  # SSY = 0
    assert np.isnan(compute_cv_anova_pvalue(y_true=y, rmsecv=0.1, n_components=2))


def test_non_finite_inputs_return_nan():
    y = np.linspace(0, 1, 30)
    assert np.isnan(compute_cv_anova_pvalue(y_true=y, rmsecv=np.nan, n_components=2))
    assert np.isnan(compute_cv_anova_pvalue(y_true=y, rmsecv=np.inf, n_components=2))
    assert np.isnan(compute_cv_anova_pvalue(y_true=y, rmsecv=0.0, n_components=2))
    assert np.isnan(compute_cv_anova_pvalue(y_true=y, rmsecv=-0.1, n_components=2))

    y_with_nan = y.copy()
    y_with_nan[0] = np.nan
    assert np.isnan(compute_cv_anova_pvalue(y_true=y_with_nan, rmsecv=0.1, n_components=2))


def test_multi_output_y_returns_nan():
    y_2d = np.zeros((30, 2))
    assert np.isnan(compute_cv_anova_pvalue(y_true=y_2d, rmsecv=0.1, n_components=2))


def test_reference_value_pin():
    # Hand-computed reference: arange(10), RMSEcv = 1.0, A = 2.
    # SSY = sum((i - 4.5)**2) for i in 0..9 = 82.5
    # N = 10, A = 2, df1 = 2, df2 = 7
    # PRESS = N * RMSEcv**2 = 10
    # F = ((SSY - PRESS) / A) / (PRESS / df2) = (72.5 / 2) / (10 / 7)
    #   = 36.25 / 1.42857... = 25.375
    # p = scipy.stats.f.sf(25.375, 2, 7)
    y = np.arange(10).astype(float)
    rmsecv = 1.0
    a = 2
    expected_f = ((82.5 - 10.0) / a) / (10.0 / 7)
    expected_p = float(f_dist.sf(expected_f, a, 7))
    actual_p = compute_cv_anova_pvalue(y_true=y, rmsecv=rmsecv, n_components=a)
    assert abs(actual_p - expected_p) < 1e-9


# ---------------------------------------------------------------------------
# Integration tests through run_search and run_unified_bayesian
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_pls_dataset():
    """Small synthetic dataset with PLS-recoverable signal — fast for end-to-end."""
    rng = np.random.default_rng(2026)
    n, p = 40, 80
    wavelengths = np.linspace(1000.0, 2500.0, p)
    X = rng.standard_normal((n, p))
    # Inject a low-rank signal that PLS can recover.
    y = X[:, 10] * 1.5 + X[:, 30] * 0.8 + 0.05 * rng.standard_normal(n)
    X_df = pd.DataFrame(X, columns=wavelengths.astype(str))
    y_ser = pd.Series(y, name="target")
    return X_df, y_ser, wavelengths


def test_grid_path_lands_cv_anova_column(synthetic_pls_dataset):
    """End-to-end: run_search with PLS-only must populate cv_anova_pvalue."""
    from spectral_predict.search import run_search

    X_df, y_ser, _wavelengths = synthetic_pls_dataset
    results_df, _meta = run_search(
        X_df,
        y_ser,
        task_type="regression",
        folds=5,
        cv_strategy="kfold",
        cv_n_repeats=1,
        models_to_test=["PLS"],
        tpe_preprocess=False,
        enable_variable_subsets=False,
        enable_region_subsets=False,
        max_n_components=5,
        progress_callback=None,
    )
    assert results_df is not None and len(results_df) > 0
    assert "cv_anova_pvalue" in results_df.columns

    pls_rows = results_df[results_df["Model"] == "PLS"]
    assert len(pls_rows) > 0, "expected at least one PLS row"
    pvals = pd.to_numeric(pls_rows["cv_anova_pvalue"], errors="coerce")
    assert pvals.notna().all(), f"all PLS rows should have non-nan p-values, got {pvals.tolist()}"
    assert ((pvals >= 0.0) & (pvals <= 1.0)).all(), f"p-values out of [0,1]: {pvals.tolist()}"


def test_bayesian_path_lands_cv_anova_column(synthetic_pls_dataset):
    """End-to-end: run_unified_bayesian with PLS must populate cv_anova_pvalue."""
    from spectral_predict.unified_bayesian import run_unified_bayesian

    X_df, y_ser, wavelengths = synthetic_pls_dataset
    results_df, _study = run_unified_bayesian(
        X=X_df.to_numpy(dtype=float),
        y=y_ser.to_numpy(dtype=float),
        wavelengths=wavelengths,
        model_name="PLS",
        task_type="regression",
        n_trials=12,
        cv_folds=5,
        cv_strategy="kfold",
        cv_n_repeats=1,
        progress_callback=None,
        verbose=False,
    )
    assert results_df is not None and len(results_df) > 0
    assert "cv_anova_pvalue" in results_df.columns

    pls_rows = results_df[results_df["Model"] == "PLS"]
    assert len(pls_rows) > 0, "expected at least one PLS row"
    pvals = pd.to_numeric(pls_rows["cv_anova_pvalue"], errors="coerce")
    # At least the trials that succeeded should have a non-nan p-value;
    # failed trials may be nan, so don't require all-non-nan.
    valid = pvals.dropna()
    assert len(valid) > 0, "expected at least one PLS Bayesian trial with cv_anova_pvalue"
    assert ((valid >= 0.0) & (valid <= 1.0)).all(), f"p-values out of [0,1]: {valid.tolist()}"


def test_grid_path_non_pls_rows_get_nan(synthetic_pls_dataset):
    """Routing-guard pin: non-PLS regression rows must get nan in cv_anova_pvalue."""
    from spectral_predict.search import run_search

    X_df, y_ser, _wavelengths = synthetic_pls_dataset
    results_df, _meta = run_search(
        X_df,
        y_ser,
        task_type="regression",
        folds=5,
        cv_strategy="kfold",
        cv_n_repeats=1,
        enabled_models=["PLS", "Ridge"],
        models_to_test=["PLS", "Ridge"],
        tpe_preprocess=False,
        enable_variable_subsets=False,
        enable_region_subsets=False,
        max_n_components=5,
        progress_callback=None,
    )
    assert results_df is not None and "cv_anova_pvalue" in results_df.columns

    ridge_rows = results_df[results_df["Model"] == "Ridge"]
    assert len(ridge_rows) > 0, "expected at least one Ridge row"
    ridge_pvals = pd.to_numeric(ridge_rows["cv_anova_pvalue"], errors="coerce")
    assert ridge_pvals.isna().all(), (
        f"non-PLS rows should have cv_anova_pvalue=nan; got {ridge_pvals.tolist()}"
    )

    pls_rows = results_df[results_df["Model"] == "PLS"]
    assert len(pls_rows) > 0, "expected at least one PLS row"
    pls_pvals = pd.to_numeric(pls_rows["cv_anova_pvalue"], errors="coerce")
    assert pls_pvals.notna().all(), (
        f"PLS rows should have non-nan p-values; got {pls_pvals.tolist()}"
    )
