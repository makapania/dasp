"""Bayesian and NSGA-II fit the user's labels like the grid (review round 3).

Numeric labels are fitted as given (PLS-DA regresses on the label values);
XGBoost, which only accepts 0..K-1, is fitted on codes and its predictions are
decoded back so every engine scores in the user's label space. Bayesian
studies whose numeric labels are not 0..K-1 get their own study name.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as skm
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spectral_predict.models import PLSTransformer
from spectral_predict.scoring import classification_metrics


def _data(labels, n_per_class=(20, 18, 16), seed=5, p=30):
    rng = np.random.default_rng(seed)
    codes = np.concatenate([np.full(k, c) for c, k in enumerate(n_per_class[: len(labels)])])
    X = rng.normal(size=(len(codes), p))
    X[:, :4] += 0.8 * codes[:, None]
    X[:, 4] -= 0.7 * (codes == 1)
    y = np.array([labels[c] for c in codes])
    return X, y


# ---------------------------------------------------------------------------
# NSGA-II helpers
# ---------------------------------------------------------------------------


def _solution(n_wavelengths, model_param=2):
    genes = [0, 0, 0, model_param, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # raw, model_types[0]
    return np.array(genes + [1] * n_wavelengths)


def _plsda_raw_cv(X, y, n_components, hp, cv):
    pipe = Pipeline(
        [
            ("pls", PLSTransformer(n_components=n_components, scale=False)),
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=hp.get("lr_C", 1.0),
                    solver=hp.get("lr_solver", "lbfgs"),
                    max_iter=hp.get("lr_max_iter", 1000),
                    random_state=42,
                ),
            ),
        ]
    )
    pred = np.empty_like(y)
    for tr, te in cv.split(X, y):
        pred[te] = clone(pipe).fit(X[tr], y[tr]).predict(X[te])
    return pred


def test_nsga2_plsda_uneven_labels_fit_raw_labels_like_grid():
    from spectral_predict.nsga2_search import (
        _compute_classification_cv_metrics,
        _decode_hyperparameter_genes,
    )

    X, y = _data((1, 2, 100))
    sol = _solution(X.shape[1])
    got = _compute_classification_cv_metrics(X, y, sol, X.shape[1], ["PLS-DA"], cv_folds=3)
    hp = _decode_hyperparameter_genes(0, 0, 0, 0, 0, 0, 0, 0, 0)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    raw_pred = _plsda_raw_cv(X, y, 3, hp, cv)
    expected = classification_metrics(y, raw_pred, classes=np.unique(y))
    assert got["F1cv"] == pytest.approx(expected["F1"])
    assert got["MCCcv"] == pytest.approx(expected["MCC"])
    # the test is sensitive: a fit on 0..K-1 codes gives a different model
    codes = np.searchsorted(np.unique(y), y)
    code_pred = _plsda_raw_cv(X, codes, 3, hp, cv)
    assert not np.array_equal(np.unique(y)[code_pred], raw_pred)


def test_nsga2_problem_fits_numeric_labels_raw_and_xgboost_codes():
    from spectral_predict.nsga2_search import SpectralOptimizationProblem

    X, y = _data((1, 2, 100))
    prob = SpectralOptimizationProblem(X, y, task_type="classification", models=["PLS-DA"])
    np.testing.assert_array_equal(prob.y, y)
    assert prob.label_encoder is None
    np.testing.assert_array_equal(prob.y_xgb, np.searchsorted([1, 2, 100], y))
    Xt, yt = _data(("a", "b"))
    prob_t = SpectralOptimizationProblem(Xt, yt, task_type="classification", models=["PLS-DA"])
    assert prob_t.label_encoder is not None
    np.testing.assert_array_equal(prob_t.y, np.searchsorted(["a", "b"], yt))


def test_nsga2_xgboost_numeric_labels_scored_in_user_labels():
    pytest.importorskip("xgboost")
    from spectral_predict.nsga2_search import (
        _compute_calibration_metrics,
        _compute_classification_cv_metrics,
    )

    X, y = _data((1, 2))
    sol = _solution(X.shape[1], model_param=0)
    cv_m = _compute_classification_cv_metrics(X, y, sol, X.shape[1], ["XGBoost"], cv_folds=3)
    assert np.isfinite(cv_m["F1cv"]) and np.isfinite(cv_m["ROC_AUCcv"])
    cal = _compute_calibration_metrics(X, y, sol, X.shape[1], ["XGBoost"], "classification")
    assert np.isfinite(cal["F1"]) and np.isfinite(cal["Accuracy"])


# ---------------------------------------------------------------------------
# Bayesian
# ---------------------------------------------------------------------------


def _wavelengths(X):
    # Integer wavelengths: all_vars is written with %g, so non-round values
    # would not map back in the validation rebuild (R031, another branch).
    return np.arange(1000.0, 1000.0 + 2 * X.shape[1], 2.0)


def _bayes(X, y, model_name, n_trials=3):
    from spectral_predict.unified_bayesian import run_unified_bayesian

    return run_unified_bayesian(
        X=X,
        y=y,
        wavelengths=_wavelengths(X),
        model_name=model_name,
        task_type="classification",
        n_trials=n_trials,
        cv_folds=3,
        n_top_regions=2,
        enable_sqlite_persistence="never",
        verbose=False,
    )


def test_bayesian_plsda_uneven_labels_rebuilds_like_grid():
    from spectral_predict.search import compute_validation_metrics_for_top_models

    X, y = _data((1, 2, 100))
    df, _ = _bayes(X, y, "PLS-DA")
    assert len(df) > 0
    out = compute_validation_metrics_for_top_models(
        df.copy(),
        X,
        y,
        X,
        y,
        "classification",
        _wavelengths(X),
        top_n=len(df),
    )
    rows = out.dropna(subset=["val_Accuracy"])
    assert len(rows) > 0
    # the grid's rebuild (raw labels) reproduces Bayesian's calibration fit
    for _, r in rows.iterrows():
        assert r["val_Accuracy"] == pytest.approx(r["Accuracy"]), r["Params"]
        assert r["val_F1"] == pytest.approx(r["F1"])
    # sensitivity: the same rows rebuilt on 0..K-1 codes give a different fit
    codes = np.searchsorted(np.unique(y), y)
    out_codes = compute_validation_metrics_for_top_models(
        df.copy(), X, codes, X, codes, "classification", _wavelengths(X), top_n=len(df)
    )
    assert not np.allclose(
        out_codes.dropna(subset=["val_Accuracy"])["val_Accuracy"].to_numpy(),
        rows["val_Accuracy"].to_numpy(),
    )


def test_bayesian_xgboost_numeric_labels_runs_and_reports_user_labels():
    pytest.importorskip("xgboost")
    X, y = _data((1, 2))
    df, _ = _bayes(X, y, "XGBoost", n_trials=2)
    assert len(df) > 0
    r = df.iloc[0]
    assert np.isfinite(r["F1cv"]) and np.isfinite(r["Accuracycv"])
    assert set(r["per_class_metrics"]) == {"1", "2"}


def test_bayesian_label_identity_segment_only_when_needed():
    X, y_codes = _data((0, 1, 2))
    _, y_text = _data(("a", "b", "c"))
    _, y_uneven = _data((1, 2, 100))
    names = {
        "codes": _bayes(X, y_codes, "PLS-DA", n_trials=1)[1].study_name,
        "text": _bayes(X, y_text, "PLS-DA", n_trials=1)[1].study_name,
        "uneven": _bayes(X, y_uneven, "PLS-DA", n_trials=1)[1].study_name,
    }
    # 0..K-1 and text labels: unchanged identity (no segment)
    assert names["codes"] == names["text"]
    # numeric labels that are not 0..K-1: a new study, never resumed with old trials
    assert names["uneven"] != names["codes"]
