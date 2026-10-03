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


def _bayes(X, y, model_name, n_trials=3, **kw):
    from spectral_predict.unified_bayesian import run_unified_bayesian

    kw.setdefault("enable_sqlite_persistence", "never")
    return run_unified_bayesian(
        X=X,
        y=y,
        wavelengths=_wavelengths(X),
        model_name=model_name,
        task_type="classification",
        n_trials=n_trials,
        cv_folds=3,
        n_top_regions=2,
        verbose=False,
        **kw,
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
    _, y_frac = _data((0.1, 0.2, 0.3))
    # fractional labels are still encoded: unchanged identity
    assert _bayes(X, y_frac, "PLS-DA", n_trials=1)[1].study_name == names["codes"]


def test_save_refined_model_uses_only_the_model_development_encoder():
    import inspect

    import spectral_predict_gui_optimized as gui_module

    source = inspect.getsource(gui_module.SpectralPredictApp._save_refined_model)
    # PR #89: the encoder comes from the frozen RefinedState snapshot of the run
    # that produced the model, never from the global search encoder.
    assert "label_encoder_to_save = snap['label_encoder']\n" in source
    assert "or self.label_encoder" not in source


# ---------------------------------------------------------------------------
# Label policy unit tests (scoring.classification_fit_labels)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "y",
    [
        np.array([0, 1, 1, 0], dtype=np.int32),
        np.array([0.0, 1.0, 1.0, 0.0]),
        np.array([False, True, True, False]),
        np.array([0, 2, 1, 2], dtype=np.int64),
    ],
    ids=["int32", "float64", "bool", "int64-3class"],
)
def test_already_coded_labels_give_the_legacy_int64_array(y):
    from sklearn.preprocessing import LabelEncoder

    from spectral_predict.scoring import classification_fit_labels
    from spectral_predict.unified_bayesian import _data_fingerprint

    fit = classification_fit_labels(y)
    legacy = LabelEncoder().fit_transform(y)
    assert fit.policy == "codes"
    assert fit.y_fit.dtype == legacy.dtype
    np.testing.assert_array_equal(fit.y_fit, legacy)
    X = np.ones((len(y), 3))
    wl = np.arange(3.0)
    assert _data_fingerprint(X, fit.y_fit, wl) == _data_fingerprint(X, legacy, wl)


@pytest.mark.parametrize(
    "y,policy",
    [
        (np.array([1, 2, 100, 2]), "raw"),
        (np.array([-1, 1, 1, -1]), "raw"),
        (np.array([1.0, 2.0, 2.0]), "raw"),
        (np.array([0.1, 0.2, 0.2]), "encoded"),
        (np.array(["a", "b", "a"], dtype=object), "encoded"),
    ],
)
def test_label_policy(y, policy):
    from spectral_predict.scoring import classification_fit_labels

    fit = classification_fit_labels(y)
    assert fit.policy == policy
    if policy == "raw":
        np.testing.assert_array_equal(fit.y_fit, y)
        xgb = classification_fit_labels(y, model_name="XGBoost")
        assert xgb.policy == "xgb_codes"
        np.testing.assert_array_equal(xgb.label_classes[xgb.y_fit], y)
    if policy == "encoded":
        np.testing.assert_array_equal(fit.encoder.inverse_transform(fit.y_fit), y)


# ---------------------------------------------------------------------------
# Fractional class labels (encoded, as before) in every engine
# ---------------------------------------------------------------------------


def test_fractional_labels_work_in_grid_bayesian_and_nsga2():
    from spectral_predict.nsga2_search import _compute_classification_cv_metrics
    from spectral_predict.search import run_search

    X, y = _data((0.1, 0.2))
    df_b, _ = _bayes(X, y, "PLS-DA", n_trials=2)
    assert len(df_b) > 0 and np.isfinite(df_b.iloc[0]["Accuracycv"])
    cv_m = _compute_classification_cv_metrics(
        X, y, _solution(X.shape[1]), X.shape[1], ["PLS-DA"], cv_folds=3
    )
    assert np.isfinite(cv_m["F1cv"])
    Xdf = pd.DataFrame(X, columns=_wavelengths(X))
    df_g, enc = run_search(
        Xdf,
        pd.Series(y),
        "classification",
        folds=3,
        models_to_test=["PLS-DA"],
        preprocessing_methods={"raw": True},
        max_n_components=2,
        enable_variable_subsets=False,
        enable_region_subsets=False,
    )
    assert enc is not None and list(enc.classes_) == [0.1, 0.2]
    assert np.isfinite(pd.to_numeric(df_g["Accuracycv"])).all()


# ---------------------------------------------------------------------------
# Binary {-1,1} / bool, multiclass XGBoost decode, XGBoost with class weights
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("labels", [(-1, 1), (False, True)])
def test_nsga2_binary_signed_and_bool_labels_match_codes(labels):
    from spectral_predict.nsga2_search import _compute_classification_cv_metrics

    X, codes = _data((0, 1))
    y = np.array([labels[c] for c in codes])
    sol = _solution(X.shape[1])
    got = _compute_classification_cv_metrics(X, y, sol, X.shape[1], ["PLS-DA"], cv_folds=3)
    ref = _compute_classification_cv_metrics(X, codes, sol, X.shape[1], ["PLS-DA"], cv_folds=3)
    for k in ("F1cv", "MCCcv", "ROC_AUCcv", "Specificitycv"):
        assert got[k] == pytest.approx(ref[k]), k


@pytest.mark.parametrize("imbalance_method", [None, "class_weight"])
def test_xgboost_multiclass_uneven_labels_decoded_in_both_engines(imbalance_method):
    pytest.importorskip("xgboost")
    from spectral_predict.nsga2_search import _compute_classification_cv_metrics

    X, y = _data((1, 2, 100), n_per_class=(20, 14, 8))
    ref_codes = np.searchsorted([1, 2, 100], y)
    sol = _solution(X.shape[1], model_param=0)
    got = _compute_classification_cv_metrics(
        X, y, sol, X.shape[1], ["XGBoost"], cv_folds=3, imbalance_method=imbalance_method
    )
    ref = _compute_classification_cv_metrics(
        X, ref_codes, sol, X.shape[1], ["XGBoost"], cv_folds=3, imbalance_method=imbalance_method
    )
    # same model on codes; scoring in user labels gives identical numbers
    for k in ("F1cv", "MCCcv", "ROC_AUCcv", "LogLosscv"):
        assert got[k] == pytest.approx(ref[k]), k
    df, _ = _bayes(X, y, "XGBoost", n_trials=2, imbalance_method=imbalance_method)
    assert len(df) > 0
    assert set(df.iloc[0]["per_class_metrics"]) == {"1", "2", "100"}


def test_validation_helper_rebuilds_xgboost_rows_on_codes_and_decodes():
    pytest.importorskip("xgboost")
    from spectral_predict.search import compute_validation_metrics_for_top_models

    X, y = _data((1, 2, 100))
    df, _ = _bayes(X, y, "XGBoost", n_trials=2)
    out = compute_validation_metrics_for_top_models(
        df.copy(), X, y, X, y, "classification", _wavelengths(X), top_n=len(df)
    )
    rows = out.dropna(subset=["val_Accuracy"])
    assert len(rows) > 0
    for _, r in rows.iterrows():
        assert r["val_Accuracy"] == pytest.approx(r["Accuracy"])
        assert r["val_F1"] == pytest.approx(r["F1"])


# ---------------------------------------------------------------------------
# Resume across the label policy (persistence on a temporary SQLite file)
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_storage(tmp_path, monkeypatch):
    from spectral_predict import run_state

    url = f"sqlite:///{(tmp_path / 'resume.sqlite3').as_posix()}"
    monkeypatch.setattr(run_state, "get_storage_url", lambda: url)
    return url


@pytest.mark.parametrize("dtype", [np.int32, np.float64, bool])
def test_already_coded_labels_keep_the_pre_policy_fingerprint(sqlite_storage, dtype):
    import optuna
    from sklearn.preprocessing import LabelEncoder

    from spectral_predict.unified_bayesian import DATA_FINGERPRINT_ATTR, _data_fingerprint

    X, codes = _data((0, 1))
    y = codes.astype(dtype)
    _, study = _bayes(X, y, "PLS-DA", n_trials=1, enable_sqlite_persistence="always")
    stored = optuna.load_study(study_name=study.study_name, storage=sqlite_storage).user_attrs
    # what the pre-policy code stamped: LabelEncoder output (int64)
    legacy = _data_fingerprint(X, LabelEncoder().fit_transform(y), _wavelengths(X))
    assert stored[DATA_FINGERPRINT_ATTR] == legacy
    # and the study name has no label segment, so an old study is found and resumed
    _, again = _bayes(X, y, "PLS-DA", n_trials=2, enable_sqlite_persistence="auto")
    assert again.study_name == study.study_name
    assert len([t for t in again.trials if t.state.is_finished()]) >= 2


def test_pre_policy_study_is_reported_as_resume_declined(sqlite_storage):
    from spectral_predict.unified_bayesian import RESUME_DECLINED_KEY

    X, codes = _data((0, 1, 2))
    # Same configuration fitted under the old policy (codes) -> pre-policy name
    _, old = _bayes(X, codes, "PLS-DA", n_trials=1, enable_sqlite_persistence="always")
    y_uneven = np.array([(1, 2, 100)[c] for c in codes])
    events = []
    _, new = _bayes(
        X,
        y_uneven,
        "PLS-DA",
        n_trials=1,
        enable_sqlite_persistence="always",
        progress_callback=events.append,
    )
    assert new.study_name != old.study_name
    notices = [e for e in events if e.get("label_policy_changed")]
    assert len(notices) == 1
    assert notices[0].get(RESUME_DECLINED_KEY) is True
    # hedged: the name alone cannot tell a pre-policy study from a 0..K-1 study
    # of other data, so the notice claims only "different class labels"
    assert "different class labels" in notices[0]["message"]
    assert "re-coded" not in notices[0]["message"]
    assert old.study_name in notices[0]["message"]


def test_xgboost_studies_keep_their_name_and_decode_old_trial_keys(sqlite_storage):
    pytest.importorskip("xgboost")
    X, codes = _data((0, 1, 2))
    y_uneven = np.array([(1, 2, 100)[c] for c in codes])
    # an XGBoost study as the pre-policy code saved it: fitted on 0..K-1 codes,
    # per-class metrics keyed "0", "1", "2"
    _, old = _bayes(X, codes, "XGBoost", n_trials=1, enable_sqlite_persistence="always")
    events = []
    df, new = _bayes(
        X,
        y_uneven,
        "XGBoost",
        n_trials=2,
        enable_sqlite_persistence="always",
        progress_callback=events.append,
    )
    # the XGBoost fit (codes) is unchanged by the policy: same study, resumed
    assert new.study_name == old.study_name
    assert len([t for t in new.trials if t.state.is_finished()]) >= 2
    assert not [e for e in events if e.get("label_policy_changed")]
    # every row, the resumed old one included, is keyed by the user's labels
    for keys in df["per_class_metrics"].dropna():
        assert set(keys) == {"1", "2", "100"}
    assert "F1_Class100" in df.columns and "F1_Class0" not in df.columns
