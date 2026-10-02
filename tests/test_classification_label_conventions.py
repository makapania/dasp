"""Classification label convention (R029) and pooled CV metrics (R030).

Convention: binary positive class = second sorted label; Specificity = TNR of
the first sorted label. Relabelling the classes must not change any metric.
Headline CV metrics come from pooled out-of-fold predictions (LOO included).
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as skm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict

from spectral_predict.scoring import (
    CLASSIFICATION_METRIC_KEYS,
    align_proba_to_classes,
    classification_metrics,
    compute_composite_score,
    compute_specificity,
)

LABEL_PAIRS = [(0, 1), (1, 2), (2, 3), (-1, 1), ("clean", "dirty")]


def _relabel(codes, pair):
    return np.array([pair[c] for c in codes], dtype=object if isinstance(pair[0], str) else None)


# ---------------------------------------------------------------------------
# Unit level: classification_metrics
# ---------------------------------------------------------------------------

_TRUE = np.array([0] * 12 + [1] * 8)
_PRED = np.array([0] * 9 + [1] * 3 + [1] * 5 + [0] * 3)
_P1 = np.linspace(0.05, 0.95, 20)


@pytest.mark.parametrize("pair", LABEL_PAIRS)
def test_metrics_invariant_to_relabelling(pair):
    base = classification_metrics(
        _TRUE, _PRED, classes=[0, 1], y_proba=np.column_stack([1 - _P1, _P1])
    )
    yt, yp = _relabel(_TRUE, pair), _relabel(_PRED, pair)
    got = classification_metrics(
        yt, yp, classes=list(pair), y_proba=np.column_stack([1 - _P1, _P1])
    )
    for k in CLASSIFICATION_METRIC_KEYS:
        assert got[k] == pytest.approx(base[k], nan_ok=True), (pair, k)


@pytest.mark.parametrize("pair", LABEL_PAIRS)
def test_positive_class_is_second_sorted_label(pair):
    yt, yp = _relabel(_TRUE, pair), _relabel(_PRED, pair)
    m = classification_metrics(yt, yp, classes=list(pair))
    pos = sorted(pair)[1]
    assert m["Recall"] == pytest.approx(skm.recall_score(yt, yp, pos_label=pos))
    assert m["Precision"] == pytest.approx(skm.precision_score(yt, yp, pos_label=pos))
    assert m["F1"] == pytest.approx(skm.f1_score(yt, yp, pos_label=pos))
    neg = sorted(pair)[0]
    assert m["Specificity"] == pytest.approx(skm.recall_score(yt, yp, pos_label=neg))
    # the R029 symptom: with {1,2} Specificity silently equalled Recall
    assert m["Recall"] == pytest.approx(5 / 8)
    assert m["Specificity"] == pytest.approx(9 / 12)


def test_specificity_with_labels_on_single_class_subset():
    # A 1-sample fold: without labels= this was a 1x1 matrix -> 0.0
    assert compute_specificity([0], [0], labels=[0, 1]) == 1.0
    assert compute_specificity([1], [1], labels=[0, 1]) == 0.0  # no negatives: 0 by convention


def test_align_proba_to_classes():
    p = np.array([[0.2, 0.8], [0.6, 0.4]])
    out = align_proba_to_classes(p, model_classes=[0, 2], classes=[0, 1, 2])
    np.testing.assert_allclose(out, [[0.2, 0.0, 0.8], [0.6, 0.0, 0.4]])
    with pytest.raises(ValueError):
        align_proba_to_classes(p, model_classes=[0, 5], classes=[0, 1, 2])
    # Without classes_ the column order of a multi-class proba is unknown: raise
    with pytest.raises(ValueError):
        align_proba_to_classes(p, None, [3, 4])
    np.testing.assert_allclose(align_proba_to_classes([[1.0], [1.0]], None, [7]), [[1.0], [1.0]])


def test_unsorted_classes_permute_probability_columns():
    y = np.array([0, 1, 0, 1])
    pred = np.array([0, 1, 0, 1])
    # columns given in the caller's class order [1, 0]
    proba_10 = np.array([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    m = classification_metrics(y, pred, classes=[1, 0], y_proba=proba_10)
    assert m["ROC_AUC"] == pytest.approx(1.0)
    assert m["LogLoss"] == pytest.approx(skm.log_loss(y, proba_10[:, ::-1]))
    with pytest.raises(ValueError):
        classification_metrics(y, pred, classes=[0, 0, 1])


def test_auc_renormalised_when_holdout_lacks_a_class():
    y = np.array([0, 0, 2, 2, 0])
    pred = np.array([0, 2, 2, 2, 0])
    proba = np.array(
        [[0.6, 0.3, 0.1], [0.3, 0.3, 0.4], [0.1, 0.2, 0.7], [0.2, 0.5, 0.3], [0.5, 0.1, 0.4]]
    )
    m = classification_metrics(y, pred, classes=[0, 1, 2], y_proba=proba)
    sub = proba[:, [0, 2]] / proba[:, [0, 2]].sum(axis=1, keepdims=True)
    assert m["ROC_AUC"] == pytest.approx(skm.roc_auc_score(y == 2, sub[:, 1]))
    assert m["LogLoss"] == pytest.approx(skm.log_loss(y, proba, labels=[0, 1, 2]))


def test_probability_rows_not_summing_to_one_are_rejected():
    y = np.array([0, 1, 2, 0, 1, 2])
    pred = np.array([0, 0, 0, 0, 0, 0])
    broadcast = np.ones((6, 3))  # one-column proba broadcast into 3 columns
    m = classification_metrics(y, pred, classes=[0, 1, 2], y_proba=broadcast)
    assert math.isnan(m["LogLoss"]) and math.isnan(m["ROC_AUC"])
    assert m["Accuracy"] == pytest.approx(2 / 6)


# ---------------------------------------------------------------------------
# cv_utils.cross_val_predict_pooled: per-fold probabilities aligned to classes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cv",
    [
        StratifiedKFold(3, shuffle=True, random_state=0),
        pytest.param("repeated", id="repeated"),
    ],
)
@pytest.mark.parametrize("labels", [(0, 1, 2), (1, 2, 100)])
def test_pooled_proba_aligned_when_fold_model_lacks_a_class(cv, labels):
    from sklearn.model_selection import RepeatedStratifiedKFold

    from spectral_predict.cv_utils import cross_val_predict_pooled

    if cv == "repeated":
        cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=0)
    X, codes = _data((10, 10, 10), 2.0)
    y = np.array([labels[c] for c in codes])
    drop = labels[2]

    class _DropsThird(LogisticRegression):
        """Never learns the third class, like a resampler that removed it."""

        def fit(self, X, y, sample_weight=None):
            keep = np.asarray(y) != drop
            return super().fit(np.asarray(X)[keep], np.asarray(y)[keep])

    model = _DropsThird(max_iter=1000)
    proba = cross_val_predict_pooled(model, X, y, cv=cv, method="predict_proba")
    assert proba.shape == (30, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_allclose(proba[:, 2], 0.0)


def test_smote_enn_class_removal_does_not_fake_a_perfect_logloss():
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RepeatedStratifiedKFold

    from spectral_predict.cv_utils import cross_val_predict_pooled
    from spectral_predict.imbalance import build_imbalance_transformer

    # Constant spectra: SMOTE-ENN leaves fold models with fewer classes
    X = np.ones((30, 5))
    y = np.repeat([0, 1, 2], 10)
    pipe = ImbPipeline(
        [
            ("imbalance", build_imbalance_transformer("smote_enn", random_state=0)),
            ("model", RandomForestClassifier(n_estimators=10, random_state=0)),
        ]
    )
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=0)
    pred = cross_val_predict_pooled(pipe, X, y, cv=cv)
    proba = cross_val_predict_pooled(pipe, X, y, cv=cv, method="predict_proba")
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    m = classification_metrics(y, pred, classes=[0, 1, 2], y_proba=proba)
    assert m["Accuracy"] < 0.5
    assert not (m["LogLoss"] < 1e-6)  # was 2e-16 before the alignment fix


# ---------------------------------------------------------------------------
# Ranking robustness
# ---------------------------------------------------------------------------


def _cls_results(acc, f1):
    return pd.DataFrame(
        {
            "Model": ["A", "B", "C"],
            "Accuracycv": acc,
            "F1cv": f1,
            "Accuracy": [1.0, 1.0, 1.0],
            "n_vars": [10, 10, 10],
            "full_vars": [10, 10, 10],
            "top_vars": ["", "", ""],
        }
    )


def test_ranking_survives_undefined_f1():
    df = compute_composite_score(
        _cls_results([0.9, 0.8, 0.95], [np.nan, 0.7, np.nan]), "classification"
    )
    assert list(df["Model"]) == ["C", "A", "B"]
    assert df["CompositeScore"].notna().all()


def test_undefined_primary_metric_ranks_last_and_is_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="spectral_predict.scoring"):
        df = compute_composite_score(
            _cls_results([np.nan, 0.8, 0.95], [0.5, 0.7, 0.9]), "classification"
        )
    assert list(df["Model"]) == ["C", "B", "A"]
    assert df["Rank"].tolist() == [1, 2, 3]
    assert "undefined score" in caplog.text


# ---------------------------------------------------------------------------
# Pooled CV vs sklearn on pooled predictions (R030), through _run_single_config
# ---------------------------------------------------------------------------


def _single_config(X, y, cv, is_binary):
    from spectral_predict.search import _run_single_config

    return _run_single_config(
        X,
        y,
        np.arange(X.shape[1], dtype=float),
        LogisticRegression(max_iter=2000),
        "LogisticRegression",
        {},
        {"name": "raw", "deriv": None, "window": None, "polyorder": None},
        cv,
        "classification",
        is_binary,
        skip_preprocessing=True,
        cv_strategy="loo" if isinstance(cv, LeaveOneOut) else "kfold",
    )


def _expected(X, y, cv):
    model = LogisticRegression(max_iter=2000)
    pred = cross_val_predict(model, X, y, cv=cv)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    classes = np.unique(y)
    binary = len(classes) == 2
    kw = {"pos_label": classes[1]} if binary else {"average": "macro"}
    auc = (
        skm.roc_auc_score(y == classes[1], proba[:, 1])
        if binary
        else skm.roc_auc_score(y, proba, multi_class="ovr", average="macro")
    )
    return {
        "Accuracycv": skm.accuracy_score(y, pred),
        "F1cv": skm.f1_score(y, pred, zero_division=0, **kw),
        "Precisioncv": skm.precision_score(y, pred, zero_division=0, **kw),
        "Recallcv": skm.recall_score(y, pred, zero_division=0, **kw),
        "Kappacv": skm.cohen_kappa_score(y, pred),
        "MCCcv": skm.matthews_corrcoef(y, pred),
        "BalancedAcccv": skm.balanced_accuracy_score(y, pred),
        "ROC_AUCcv": auc,
        "LogLosscv": skm.log_loss(y, proba, labels=classes),
        "Specificitycv": compute_specificity(y, pred, labels=classes),
    }


def _data(n_per_class, sep, seed=0, p=6):
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.full(k, c) for c, k in enumerate(n_per_class)])
    X = rng.normal(size=(len(y), p))
    X[:, 0] += sep * y
    X[:, 1] -= 0.5 * sep * (y == 1)
    return X, y


@pytest.mark.parametrize(
    "n_per_class,sep",
    [
        ((15, 15), 6.0),  # perfectly separable binary
        ((15, 15), 0.8),  # imperfect binary
        ((22, 8), 1.0),  # imbalanced binary
        ((10, 10, 10), 5.0),  # separable multiclass
        ((14, 9, 6), 0.9),  # imperfect, imbalanced multiclass
    ],
)
@pytest.mark.parametrize("cv_kind", ["loo", "kfold"])
def test_pooled_cv_metrics_match_sklearn(n_per_class, sep, cv_kind):
    X, y = _data(n_per_class, sep)
    cv = LeaveOneOut() if cv_kind == "loo" else StratifiedKFold(5, shuffle=True, random_state=0)
    result = _single_config(X, y, cv, len(n_per_class) == 2)
    expected = _expected(X, y, cv)
    for k, v in expected.items():
        assert result[k] == pytest.approx(v, rel=1e-6, abs=1e-9), k
    assert result["BERcv"] == pytest.approx(1 - expected["BalancedAcccv"])


def test_loo_perfect_binary_is_perfect():
    X, y = _data((15, 15), 8.0)
    r = _single_config(X, y, LeaveOneOut(), True)
    for k in (
        "Accuracycv",
        "F1cv",
        "Precisioncv",
        "Recallcv",
        "Specificitycv",
        "MCCcv",
        "Kappacv",
        "BalancedAcccv",
        "ROC_AUCcv",
    ):
        assert r[k] == pytest.approx(1.0), k


def test_fold_returns_probabilities_in_global_class_order():
    from spectral_predict.search import _run_single_fold

    X, y = _data((10, 10, 10), 3.0)
    train = np.flatnonzero(y != 1)  # training fold lacks class 1
    test = np.array([0, 12, 25])
    m = _run_single_fold(
        LogisticRegression(max_iter=500), X, y, train, test, "classification", False
    )
    assert m["y_proba"].shape == (3, 3)
    np.testing.assert_allclose(m["y_proba"][:, 1], 0.0)
    np.testing.assert_allclose(m["y_proba"].sum(axis=1), 1.0)


class _ProbaFailsOnSecondFit(LogisticRegression):
    """predict_proba unavailable for the model of the second CV fold only."""

    n_fits = 0

    def fit(self, X, y, sample_weight=None):
        type(self).n_fits += 1
        self._fit_no = type(self).n_fits
        return super().fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X):
        if getattr(self, "_fit_no", 0) == 2:
            raise AttributeError("no probabilities for this fold")
        return super().predict_proba(X)


def test_one_fold_without_probabilities_makes_pooled_auc_nan():
    from spectral_predict.search import _run_single_config

    X, y = _data((15, 15), 0.8)
    _ProbaFailsOnSecondFit.n_fits = 0
    r = _run_single_config(
        X,
        y,
        np.arange(X.shape[1], dtype=float),
        _ProbaFailsOnSecondFit(max_iter=2000),
        "LogisticRegression",
        {},
        {"name": "raw", "deriv": None, "window": None, "polyorder": None},
        StratifiedKFold(5, shuffle=True, random_state=0),
        "classification",
        True,
        skip_preprocessing=True,
    )
    assert math.isnan(r["ROC_AUCcv"]) and math.isnan(r["LogLosscv"])
    assert np.isfinite(r["F1cv"]) and np.isfinite(r["Accuracycv"])


# ---------------------------------------------------------------------------
# End to end: run_search under every label pair and CV strategy
# ---------------------------------------------------------------------------

_CV_COLS = (
    "Accuracycv",
    "F1cv",
    "Precisioncv",
    "Recallcv",
    "Specificitycv",
    "Kappacv",
    "MCCcv",
    "BalancedAcccv",
    "ROC_AUCcv",
    "LogLosscv",
)


def _search(y_labels, cv_strategy):
    from spectral_predict.search import run_search

    rng = np.random.default_rng(11)
    n, p = 30, 20
    codes = np.array([0] * 17 + [1] * 13)
    X = rng.normal(size=(n, p))
    X[:, :3] += 0.9 * codes[:, None]  # overlapping classes -> imperfect metrics
    Xdf = pd.DataFrame(X, columns=[float(900 + 3 * i) for i in range(p)])
    y = pd.Series(list(_relabel(codes, y_labels)))
    return run_search(
        Xdf,
        y,
        "classification",
        folds=3,
        cv_strategy=cv_strategy,
        cv_n_repeats=2,
        models_to_test=["PLS-DA"],
        preprocessing_methods={"raw": True},
        max_n_components=2,
        enable_variable_subsets=False,
        enable_region_subsets=False,
    )


@pytest.fixture(scope="module")
def _reference_runs():
    return {}


@pytest.mark.parametrize("cv_strategy", ["kfold", "repeated_kfold", "loo"])
@pytest.mark.parametrize("pair", LABEL_PAIRS)
def test_run_search_relabelling_gives_identical_metrics(pair, cv_strategy, _reference_runs):
    df, enc = _search(pair, cv_strategy)
    # label_encoder contract: returned for text labels only
    assert (enc is not None) == isinstance(pair[0], str)
    assert np.isfinite(df["Rank"]).all()
    df = df.sort_values(["Model", "Params"]).reset_index(drop=True)
    got = df[list(_CV_COLS)].to_numpy(dtype=float)
    assert np.all(np.isfinite(got[:, :8]))
    if cv_strategy not in _reference_runs:
        _reference_runs[cv_strategy] = got
    np.testing.assert_allclose(got, _reference_runs[cv_strategy], rtol=1e-12, equal_nan=True)
    # Recall and Specificity are different rates (R029 made them equal for {1,2})
    assert not np.allclose(got[:, 3], got[:, 4])
    # per-class outputs carry the user's labels for numeric targets
    if not isinstance(pair[0], str):
        assert set(df["per_class_metrics"].iloc[0]) == {str(pair[0]), str(pair[1])}
        assert f"F1_Class{pair[1]}" in df.columns


# ---------------------------------------------------------------------------
# Saved model round trip with numeric labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pair", [(1, 2), (2, 3), (-1, 1)])
def test_save_load_predict_round_trip_numeric_labels(tmp_path, pair):
    from spectral_predict.model_io import load_model, predict_with_model, save_model

    X, codes = _data((12, 12), 4.0, p=8)
    y = _relabel(codes, pair)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    wl = [float(1000 + i) for i in range(X.shape[1])]
    meta = {
        "model_name": "LogisticRegression",
        "task_type": "classification",
        "wavelengths": wl,
        "n_vars": len(wl),
        "preprocessing": "raw",
    }
    # run_search returns label_encoder=None for numeric labels, so the saved
    # model carries no encoder and predictions come back in the user's labels.
    path = tmp_path / "m.dasp"
    save_model(model, None, meta, path, label_encoder=None)
    loaded = load_model(path)
    pred = predict_with_model(loaded, pd.DataFrame(X, columns=wl))
    assert set(np.unique(pred)) <= set(pair)
    np.testing.assert_array_equal(pred, model.predict(X))


def test_save_load_predict_round_trip_text_labels(tmp_path):
    from sklearn.preprocessing import LabelEncoder

    from spectral_predict.model_io import load_model, predict_with_model, save_model

    X, codes = _data((12, 12), 4.0, p=8)
    y_text = _relabel(codes, ("clean", "dirty"))
    enc = LabelEncoder().fit(y_text)
    model = LogisticRegression(max_iter=1000).fit(X, enc.transform(y_text))
    wl = [float(1000 + i) for i in range(X.shape[1])]
    meta = {
        "model_name": "LogisticRegression",
        "task_type": "classification",
        "wavelengths": wl,
        "n_vars": len(wl),
        "preprocessing": "raw",
    }
    path = tmp_path / "t.dasp"
    save_model(model, None, meta, path, label_encoder=enc)
    pred = predict_with_model(load_model(path), pd.DataFrame(X, columns=wl))
    np.testing.assert_array_equal(pred, enc.inverse_transform(model.predict(X)))
    assert math.isclose(np.mean(pred == y_text), np.mean(model.predict(X) == codes))


# ---------------------------------------------------------------------------
# Model equivalence: the grid fits the user's own labels, so rebuilding a row
# (validation helper, Model Development helpers) reproduces it (review round 1)
# ---------------------------------------------------------------------------


def _uneven_search(labels, n_per_class, seed=5, **kw):
    from spectral_predict.search import run_search

    rng = np.random.default_rng(seed)
    codes = np.concatenate([np.full(k, c) for c, k in enumerate(n_per_class)])
    X = rng.normal(size=(len(codes), 25))
    X[:, :4] += 0.8 * codes[:, None]
    X[:, 4] -= 0.7 * (codes == 1)
    Xdf = pd.DataFrame(X, columns=[float(1000 + 4 * i) for i in range(X.shape[1])])
    y = pd.Series([labels[c] for c in codes])
    df, enc = run_search(
        Xdf,
        y,
        "classification",
        folds=3,
        models_to_test=["PLS-DA"],
        preprocessing_methods={"raw": True},
        max_n_components=3,
        enable_variable_subsets=False,
        enable_region_subsets=False,
        **kw,
    )
    return Xdf, y, df, enc


@pytest.mark.parametrize("labels", [(1, 2, 100), (2, 3)])
def test_validation_rebuild_reproduces_grid_calibration(labels):
    from spectral_predict.search import compute_validation_metrics_for_top_models

    n_per_class = (22, 20, 18)[: len(labels)]
    Xdf, y, df, _ = _uneven_search(labels, n_per_class)
    X = Xdf.to_numpy()
    out = compute_validation_metrics_for_top_models(
        df.copy(),
        X,
        y.to_numpy(),
        X,
        y.to_numpy(),
        "classification",
        Xdf.columns.values,
        top_n=len(df),
    )
    # Holdout = calibration set: the rebuilt model must give the grid's
    # calibration metrics exactly (same model, same metric definitions).
    for _, r in out.iterrows():
        assert r["val_Accuracy"] == pytest.approx(r["Accuracy"]), r["Params"]
        assert r["val_F1"] == pytest.approx(r["F1"])
        assert r["val_Precision"] == pytest.approx(r["Precision"])
        assert r["val_Recall"] == pytest.approx(r["Recall"])
        assert r["val_ROC_AUC"] == pytest.approx(r["ROC_AUC"])


def test_model_development_rebuild_reproduces_grid_cv_for_uneven_labels():
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from spectral_predict.models import PLSTransformer, parse_row_params, plsda_head_kwargs

    labels = (1, 2, 100)
    Xdf, y, df, enc = _uneven_search(labels, (22, 20, 18))
    assert enc is None  # numeric labels: nothing to decode
    X, yv = Xdf.to_numpy(), y.to_numpy()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # build_cv_splitter("kfold")
    classes = np.unique(yv)
    for _, row in df.iterrows():
        params = parse_row_params(row["Params"])
        pls_kw = {k[len("pls__") :]: v for k, v in params.items() if k.startswith("pls__")}
        pipe = Pipeline(
            [
                ("pls", PLSTransformer(**pls_kw)),
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(**plsda_head_kwargs(params))),
            ]
        )
        # Manual fold loop: sklearn's cross_val_predict(method="predict_proba")
        # label-encodes y before fitting, which changes a PLS-DA model whose
        # labels are unevenly spaced ({1, 2, 100}); Model Development and the
        # grid both fit the raw labels.
        pred = np.empty_like(yv)
        proba = np.zeros((len(yv), len(classes)))
        for tr, te in cv.split(X, yv):
            fitted = clone(pipe).fit(X[tr], yv[tr])
            pred[te] = fitted.predict(X[te])
            proba[te] = align_proba_to_classes(
                fitted.predict_proba(X[te]), fitted.classes_, classes
            )
        m = classification_metrics(yv, pred, classes=classes, y_proba=proba)
        assert row["Accuracycv"] == pytest.approx(m["Accuracy"]), row["Params"]
        assert row["F1cv"] == pytest.approx(m["F1"])
        assert row["ROC_AUCcv"] == pytest.approx(m["ROC_AUC"])
        # per-class outputs keep the user's labels
        assert set(row["per_class_metrics"]) == {str(c) for c in labels}


def test_external_validation_metrics_match_sklearn():
    from spectral_predict.search import (
        _rebuild_model_from_row,
        compute_validation_metrics_for_top_models,
    )

    Xdf, y, df, _ = _uneven_search((2, 3), (25, 20), seed=9)
    X, yv = Xdf.to_numpy(), y.to_numpy()
    val = np.arange(0, 45, 3)
    cal = np.setdiff1d(np.arange(45), val)
    out = compute_validation_metrics_for_top_models(
        df.copy(),
        X[cal],
        yv[cal],
        X[val],
        yv[val],
        "classification",
        Xdf.columns.values,
        top_n=1,
    )
    r = out.dropna(subset=["val_Accuracy"]).iloc[0]
    # Rebuild as the helper does and score with sklearn, positive class = 3
    model = _rebuild_model_from_row(r, "classification")
    model.fit(X[cal], yv[cal])
    pred = model.predict(X[val])
    proba = model.predict_proba(X[val])
    assert r["val_F1"] == pytest.approx(skm.f1_score(yv[val], pred, pos_label=3))
    assert r["val_Recall"] == pytest.approx(skm.recall_score(yv[val], pred, pos_label=3))
    assert r["val_ROC_AUC"] == pytest.approx(skm.roc_auc_score(yv[val] == 3, proba[:, 1]))


def test_unseen_text_holdout_class_counts_as_error():
    from spectral_predict.search import run_search

    Xdf, y, _, _ = _uneven_search(("a", "b"), (20, 20), seed=3)
    rng = np.random.default_rng(0)
    X_val = rng.normal(size=(6, Xdf.shape[1]))
    y_val = np.array(["a", "b", "a", "zzz", "b", "zzz"], dtype=object)
    df2, _ = run_search(
        Xdf,
        y,
        "classification",
        folds=3,
        models_to_test=["PLS-DA"],
        preprocessing_methods={"raw": True},
        max_n_components=2,
        enable_variable_subsets=False,
        enable_region_subsets=False,
        X_validation=X_val,
        y_validation=y_val,
        compute_validation=True,
        validation_top_n=2,
    )
    rows = df2.dropna(subset=["val_Accuracy"])
    assert len(rows) > 0
    # the two "zzz" samples can never be predicted, so accuracy <= 4/6
    assert (rows["val_Accuracy"] <= 4 / 6 + 1e-12).all()
    assert np.isfinite(rows["val_F1"]).all()
