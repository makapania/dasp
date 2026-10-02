"""Booster round selection from the pooled CV curve (R028, R003, R022, R126).

The number of boosting rounds is chosen like the number of PLS latent
variables: every fold is fitted with the maximum round count and no eval_set,
its test predictions at every round count are pooled, and ONE count is chosen
from the pooled curve (``early_stopping_rounds`` = patience). Previously each
fold early-stopped on its own test fold, so a fold's model depended on the
labels it was scored on.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    RepeatedKFold,
    StratifiedKFold,
    cross_val_predict,
)

from spectral_predict.cv_utils import (
    BOOSTING_ROUND_POLICY,
    booster_staged_predict,
    cross_val_boosting_rounds,
    cross_val_predict_with_early_stopping,
    cross_validate_with_early_stopping,
    pool_boosting_predictions,
    select_n_rounds,
    set_booster_rounds,
)


def _regression_data(n=48, p=25, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = X[:, 0] - 0.5 * X[:, 1] + 0.4 * rng.normal(size=n)
    return X, y


def _classification_data(n=48, p=25, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int)
    return X, y


def _boosters(task):
    """(id, estimator) for every booster library that is installed."""
    out = []
    try:
        from xgboost import XGBClassifier, XGBRegressor

        cls = XGBRegressor if task == "regression" else XGBClassifier
        out.append(
            (
                "xgboost",
                cls(
                    n_estimators=60,
                    learning_rate=0.2,
                    max_depth=3,
                    subsample=0.8,
                    random_state=0,
                    n_jobs=1,
                ),
            )
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        cls = LGBMRegressor if task == "regression" else LGBMClassifier
        out.append(
            (
                "lightgbm",
                cls(
                    n_estimators=60,
                    learning_rate=0.2,
                    num_leaves=7,
                    min_child_samples=3,
                    subsample=0.8,
                    subsample_freq=1,
                    random_state=0,
                    verbose=-1,
                    n_jobs=1,
                ),
            )
        )
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor

        cls = CatBoostRegressor if task == "regression" else CatBoostClassifier
        out.append(
            (
                "catboost",
                cls(
                    iterations=60,
                    learning_rate=0.2,
                    depth=3,
                    random_seed=0,
                    verbose=0,
                    thread_count=1,
                    allow_writing_files=False,
                ),
            )
        )
    except ImportError:
        pass
    return out


_CASES = [
    pytest.param(task, model, id=f"{name}-{task}")
    for task in ("regression", "classification")
    for name, model in _boosters(task)
]


def _data(task):
    return _regression_data() if task == "regression" else _classification_data()


def _cv(task):
    if task == "regression":
        return KFold(n_splits=4, shuffle=True, random_state=42)
    return StratifiedKFold(n_splits=4, shuffle=True, random_state=42)


class _FixedSplits:
    """Splitter that replays fixed splits (stratified splits move with the labels)."""

    def __init__(self, splits):
        self._splits = [(np.asarray(tr), np.asarray(te)) for tr, te in splits]

    def split(self, X=None, y=None, groups=None):
        yield from self._splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self._splits)


def _change_test_labels(y, test_idx, task):
    """Change ONLY the labels of one test fold."""
    y2 = y.copy()
    if task == "regression":
        y2[test_idx] = y2[test_idx][::-1] * -3.0 + 7.0
    else:
        y2[test_idx] = 1 - y2[test_idx]
    return y2


# --- The leak itself ---------------------------------------------------------------


@pytest.mark.parametrize(("task", "model"), _CASES)
def test_fold_predictions_do_not_depend_on_test_fold_labels(task, model):
    """Changing only a test fold's labels cannot change that fold's predictions.

    The fold model is fitted on the training rows only, so its predictions at
    EVERY round count are identical. (The single selected count is a pooled CV
    statistic, like the PLS LV count, so it may move; the fold model may not.)
    """
    X, y = _data(task)
    cv = _FixedSplits(_cv(task).split(X, y))
    _, test_idx = next(iter(cv.split(X, y)))
    y2 = _change_test_labels(y, test_idx, task)

    res1 = cross_val_boosting_rounds(model, X, y, cv, patience=10, keep_models=True)
    res2 = cross_val_boosting_rounds(model, X, y2, cv, patience=10, keep_models=True)

    assert np.array_equal(res1.test_indices[0], test_idx)
    staged1 = booster_staged_predict(res1.fold_models[0], X[test_idx])
    staged2 = booster_staged_predict(res2.fold_models[0], X[test_idx])
    np.testing.assert_array_equal(staged1, staged2)


@pytest.mark.parametrize(("task", "model"), _CASES)
def test_grid_fold_predictions_do_not_depend_on_test_fold_labels(task, model):
    """Same invariant through grid search's per-fold worker (search._run_single_fold)."""
    from spectral_predict.search import _run_single_fold

    X, y = _data(task)
    cv = _cv(task)
    train_idx, test_idx = next(iter(cv.split(X, y)))
    y2 = _change_test_labels(y, test_idx, task)
    kwargs = dict(
        pipe=clone(model),
        X=X,
        train_idx=train_idx,
        test_idx=test_idx,
        task_type=task,
        is_binary_classification=task == "classification",
        early_stopping_rounds=10,
    )
    out1 = _run_single_fold(y=y, **kwargs)
    out2 = _run_single_fold(y=y2, **kwargs)
    np.testing.assert_array_equal(out1["staged"], out2["staged"])


# --- Selection rule -----------------------------------------------------------------


def test_select_n_rounds_patience_semantics():
    curve = np.array([5.0, 4.0, 3.0, 3.5, 3.2, 2.0, 1.0])
    # patience 2: best 3.0 at round 3, rounds 4-5 do not improve -> stop, keep 3
    assert select_n_rounds(curve, 2, higher_is_better=False) == 3
    # patience 3 sees round 6 improve
    assert select_n_rounds(curve, 3, higher_is_better=False) == 7
    # no patience: global best
    assert select_n_rounds(curve, None, higher_is_better=False) == 7
    assert select_n_rounds(curve, 0, higher_is_better=False) == 7


def test_select_n_rounds_accuracy_ties_use_logloss():
    acc = np.array([0.8, 0.9, 0.9, 0.9, 0.85])
    logloss = np.array([0.6, 0.5, 0.4, 0.45, 0.3])
    # plain: earliest of the plateau
    assert select_n_rounds(acc, None, higher_is_better=True) == 2
    # tie-break: same accuracy, lower log-loss wins; lower accuracy never wins
    assert select_n_rounds(acc, None, higher_is_better=True, tiebreak=logloss) == 3


def test_select_n_rounds_ignores_nan():
    curve = np.array([np.nan, 2.0, np.nan, 1.0])
    assert select_n_rounds(curve, None, higher_is_better=False) == 4


# --- One count for all folds; the refit reproduces the CV ----------------------------


@pytest.mark.parametrize(("task", "model"), _CASES)
def test_refit_with_selected_rounds_reproduces_cv_predictions(task, model):
    """Fold predictions at the selected count equal a plain CV of a model refit with
    that count (truncation identity), and re-running the selection on the refit
    model returns the same count (Tab 7 / export reproduce the row)."""
    X, y = _data(task)
    cv = _cv(task)
    res = cross_val_boosting_rounds(model, X, y, cv, patience=10)
    assert 1 <= res.n_rounds <= res.max_rounds
    pooled = pool_boosting_predictions(res, len(y), y_dtype=y.dtype)

    refit = clone(model)
    set_booster_rounds(refit, res.n_rounds)
    plain = np.ravel(cross_val_predict(refit, X, y, cv=cv))
    if task == "regression":
        np.testing.assert_allclose(plain, pooled, rtol=0, atol=1e-9)
    else:
        np.testing.assert_array_equal(plain, pooled)

    again = cross_val_boosting_rounds(refit, X, y, cv, patience=10)
    assert again.n_rounds == res.n_rounds


def test_loo_runs_round_selection_without_warning(recwarn):
    """LOO is fine under the pooled-curve policy (one fit per sample); the old
    'early stopping disabled under LOO' guard is gone."""
    from lightgbm import LGBMRegressor

    X, y = _regression_data(n=20)
    preds, n_rounds = cross_val_predict_with_early_stopping(
        LGBMRegressor(n_estimators=30, verbose=-1, min_child_samples=3),
        X,
        y,
        LeaveOneOut(),
        early_stopping_rounds=5,
        return_n_rounds=True,
    )
    assert preds.shape == y.shape and n_rounds is not None
    assert not [w for w in recwarn if "Early stopping disabled" in str(w.message)]


def test_repeated_cv_one_prediction_per_sample():
    from xgboost import XGBClassifier

    X, y = _classification_data()
    cv = RepeatedKFold(n_splits=3, n_repeats=2, random_state=1)
    preds, n_rounds = cross_val_predict_with_early_stopping(
        XGBClassifier(n_estimators=30, n_jobs=1),
        X,
        y,
        cv,
        early_stopping_rounds=5,
        return_n_rounds=True,
    )
    proba = cross_val_predict_with_early_stopping(
        XGBClassifier(n_estimators=30, n_jobs=1),
        X,
        y,
        cv,
        early_stopping_rounds=5,
        method="predict_proba",
    )
    assert preds.shape == y.shape and set(np.unique(preds)) <= {0, 1}
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert n_rounds >= 1


def test_cross_validate_reports_selected_rounds():
    from lightgbm import LGBMRegressor

    X, y = _regression_data()
    out = cross_validate_with_early_stopping(
        LGBMRegressor(n_estimators=40, verbose=-1),
        X,
        y,
        KFold(4, shuffle=True, random_state=0),
        early_stopping_rounds=5,
        return_train_score=True,
    )
    assert 1 <= out["n_rounds_selected"] <= 40
    assert len(out["test_score"]) == 4 and len(out["train_score"]) == 4


# --- Grid search rows (R028 + R126) -----------------------------------------------


def _run_grid(model, task, X, y, **kwargs):
    from spectral_predict.search import run_search

    X_df = pd.DataFrame(X, columns=np.linspace(1000.0, 1100.0, X.shape[1]))
    grids = {
        "LightGBM": dict(
            lightgbm_n_estimators_list=[60],
            lightgbm_learning_rates=[0.2],
            lightgbm_num_leaves_list=[7],
        ),
        "XGBoost": dict(xgb_n_estimators_list=[60], xgb_learning_rates=[0.2], xgb_max_depths=[3]),
    }[model]
    df, _ = run_search(
        X_df,
        pd.Series(y),
        task,
        folds=3,
        tier="quick",
        models_to_test=[model],
        enabled_models=[model],
        preprocessing_methods={"raw": True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        **grids,
        **kwargs,
    )
    return df


def _rounds_param(params_str):
    params = ast.literal_eval(params_str)
    for key in ("n_estimators", "iterations", "model__n_estimators", "model__iterations"):
        if params.get(key) is not None:
            return params[key]
    return None


def test_grid_row_records_selected_rounds_in_params():
    X, y = _regression_data()
    df = _run_grid("LightGBM", "regression", X, y, early_stopping_rounds=10)
    assert len(df) > 0
    for _, row in df.iterrows():
        k = int(row["n_estimators_selected"])
        assert 1 <= k <= 60
        assert _rounds_param(row["Params"]) == k
        assert row["early_stopping_rounds"] == 10


def test_grid_row_without_round_selection_keeps_configured_rounds():
    X, y = _regression_data()
    df = _run_grid("LightGBM", "regression", X, y, early_stopping_rounds=0)
    for _, row in df.iterrows():
        assert pd.isna(row["n_estimators_selected"])
        assert _rounds_param(row["Params"]) == 60


@pytest.mark.parametrize(
    ("model", "task", "imbalance_method"),
    [("XGBoost", "classification", "class_weight"), ("LightGBM", "regression", "binning")],
)
def test_weighted_grid_rows_run_the_round_selection_they_record(model, task, imbalance_method):
    """R126: weighted paths used to skip early stopping yet record 40. Now they run
    the same pooled-curve selection (weights threaded through) and record it."""
    X, y = _data(task)
    df = _run_grid(model, task, X, y, early_stopping_rounds=10, imbalance_method=imbalance_method)
    assert len(df) > 0
    for _, row in df.iterrows():
        assert row["early_stopping_rounds"] == 10
        k = int(row["n_estimators_selected"])
        assert _rounds_param(row["Params"]) == k


# --- Bayesian (R003) -------------------------------------------------------------------


def _run_bayes(model_name, task, es):
    from spectral_predict import run_state, unified_bayesian

    run_state._active_storage_url = None
    run_state._active_run_id = None
    X, y = _data(task)
    return unified_bayesian.run_unified_bayesian(
        X=X,
        y=y,
        wavelengths=np.linspace(900, 1700, X.shape[1]),
        model_name=model_name,
        task_type=task,
        n_trials=3,
        cv_folds=3,
        early_stopping_rounds=es,
        random_state=42,
        verbose=False,
        progress_callback=None,
        enable_sqlite_persistence="never",
    )


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_bayesian_booster_rows_carry_selected_rounds(task):
    from spectral_predict.unified_bayesian import BOOSTING_ROUND_POLICY_ATTR

    df, study = _run_bayes("LightGBM", task, 10)
    assert study.user_attrs.get(BOOSTING_ROUND_POLICY_ATTR) == BOOSTING_ROUND_POLICY
    assert len(df) > 0
    for _, row in df.iterrows():
        k = int(row["n_estimators_selected"])
        assert _rounds_param(row["Params"]) == k


def test_bayesian_study_identity_versions_booster_scoring():
    """Booster studies with round selection get a new name; others keep theirs."""
    from spectral_predict.unified_bayesian import BOOSTING_ROUND_POLICY_ATTR

    _, with_es = _run_bayes("LightGBM", "regression", 10)
    _, without_es = _run_bayes("LightGBM", "regression", 0)
    assert with_es.study_name != without_es.study_name
    assert BOOSTING_ROUND_POLICY_ATTR not in without_es.user_attrs


# --- NSGA-II (R022) --------------------------------------------------------------------


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_nsga2_display_metrics_come_from_the_objective_predictions(task):
    from spectral_predict.nsga2_search import (
        _booster_cv_metrics,
        convert_nsga2_to_v1_format,
        run_nsga2_search,
    )

    X, y = _data(task)
    result = run_nsga2_search(
        X=X,
        y=y,
        task_type=task,
        population_size=6,
        n_generations=2,
        cv_folds=3,
        min_wavelengths=5,
        random_state=42,
        verbose=0,
        models=["LightGBM"],
        early_stopping_rounds=10,
    )
    assert len(result["pareto_front"]) > 0
    for objectives, solution in zip(result["pareto_front"], result["pareto_solutions"]):
        boost = _booster_cv_metrics(
            X,
            y,
            solution,
            result["model_types"],
            task,
            3,
            42,
            early_stopping_rounds=10,
        )
        if task == "regression":
            assert boost["RMSEcv"] == pytest.approx(objectives[0], rel=1e-12, abs=1e-12)
        else:
            assert boost["Accuracycv"] == pytest.approx(1.0 - objectives[0], abs=1e-12)

    df = convert_nsga2_to_v1_format(
        result,
        X.shape[1],
        task,
        folds=3,
        X=X,
        y=y,
        include_best_from_all=False,
    )
    for _, row in df.iterrows():
        k = int(row["n_estimators_selected"])
        assert _rounds_param(row["Params"]) == k
