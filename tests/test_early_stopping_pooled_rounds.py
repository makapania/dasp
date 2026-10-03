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
        # Final model: fitted at the maximum and truncated (redesign).
        assert int(row["n_estimators_fit"]) == 60
        assert row["round_selection_truncated"] is True or row["round_selection_truncated"] == 1


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


# --- Review round 1 follow-ups --------------------------------------------------------


def _weighted_cases():
    from xgboost import XGBClassifier

    return XGBClassifier(n_estimators=40, learning_rate=0.2, max_depth=3, random_state=0, n_jobs=1)


def test_balanced_weights_come_from_the_training_fold_only():
    """Item 1: balanced class weights are computed from each training fold's y, so a
    weighted fold model cannot depend on its test labels either."""
    X, y = _classification_data()
    y = y.copy()
    y[:6] = 2  # imbalanced, three classes
    cv = _FixedSplits(StratifiedKFold(4, shuffle=True, random_state=42).split(X, y))
    _, test_idx = next(iter(cv.split(X, y)))
    y2 = y.copy()
    y2[test_idx] = (y2[test_idx] + 1) % 3

    kw = dict(patience=10, balanced_sample_weight=True, keep_models=True)
    res1 = cross_val_boosting_rounds(_weighted_cases(), X, y, cv, **kw)
    res2 = cross_val_boosting_rounds(_weighted_cases(), X, y2, cv, **kw)
    np.testing.assert_array_equal(
        booster_staged_predict(res1.fold_models[0], X[test_idx]),
        booster_staged_predict(res2.fold_models[0], X[test_idx]),
    )


def test_non_booster_balanced_weights_come_from_the_training_fold_only():
    from sklearn.linear_model import RidgeClassifier

    from spectral_predict.cv_utils import cross_val_predict_pooled

    X, y = _classification_data()
    y = y.copy()
    y[:8] = 1
    cv = _FixedSplits(StratifiedKFold(4, shuffle=True, random_state=0).split(X, y))
    _, test_idx = next(iter(cv.split(X, y)))
    y2 = y.copy()
    y2[test_idx] = 1 - y2[test_idx]
    p1 = cross_val_predict_pooled(
        RidgeClassifier(), X, y, cv, balanced_weight_param="sample_weight"
    )
    p2 = cross_val_predict_pooled(
        RidgeClassifier(), X, y2, cv, balanced_weight_param="sample_weight"
    )
    np.testing.assert_array_equal(p1[test_idx], p2[test_idx])


def test_bayesian_and_nsga2_weighted_cv_use_per_fold_weights(monkeypatch):
    """Item 1: neither search passes weights computed from all of y into its CV."""
    import spectral_predict.cv_utils as cvu
    from spectral_predict import nsga2_search, run_state, unified_bayesian

    calls = []
    real = cvu.cross_val_boosting_rounds

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(unified_bayesian, "cross_val_boosting_rounds", spy)
    monkeypatch.setattr(cvu, "cross_val_boosting_rounds", spy)
    run_state._active_storage_url = None
    run_state._active_run_id = None
    X, y = _classification_data()
    y = y.copy()
    y[:6] = 0
    unified_bayesian.run_unified_bayesian(
        X=X,
        y=y,
        wavelengths=np.linspace(900, 1700, X.shape[1]),
        model_name="XGBoost",
        task_type="classification",
        n_trials=2,
        cv_folds=3,
        early_stopping_rounds=10,
        imbalance_method="class_weight",
        random_state=42,
        verbose=False,
        progress_callback=None,
        enable_sqlite_persistence="never",
    )
    nsga2_search.run_nsga2_search(
        X=X,
        y=y,
        task_type="classification",
        population_size=4,
        n_generations=1,
        cv_folds=3,
        min_wavelengths=5,
        random_state=42,
        verbose=0,
        models=["XGBoost"],
        early_stopping_rounds=10,
        imbalance_method="class_weight",
    )
    assert calls
    for kwargs in calls:
        assert kwargs.get("sample_weight") is None
        assert kwargs.get("balanced_sample_weight") is True


def test_previous_policy_booster_study_is_reported_not_reused(tmp_path, monkeypatch):
    """Item 2: a saved study scored under the old (test-fold) early stopping is found,
    reported as not reused (resume_declined) and left untouched."""
    import optuna

    from spectral_predict import run_state
    from spectral_predict.unified_bayesian import (
        PREVIOUS_POLICY_STUDY_BASE_ATTR,
        RESUME_DECLINED_KEY,
        run_unified_bayesian,
    )

    X, y = _regression_data(n=30)
    opts = dict(
        X=X,
        y=y,
        wavelengths=np.linspace(900, 1700, X.shape[1]),
        model_name="LightGBM",
        task_type="regression",
        cv_folds=3,
        early_stopping_rounds=10,
        random_state=42,
        verbose=False,
    )
    path = tmp_path / "old_policy.sqlite3"
    url = f"sqlite:///{path.as_posix()}"
    monkeypatch.setattr(run_state, "get_storage_url", lambda: url)
    _, template = run_unified_bayesian(**opts, n_trials=0, enable_sqlite_persistence="never")
    old_base = template.user_attrs[PREVIOUS_POLICY_STUDY_BASE_ATTR]
    assert old_base and not template.study_name.startswith(old_base)

    old_name = f"{old_base}_env1_olddigest"
    prior = optuna.create_study(study_name=old_name, storage=url)
    prior.add_trial(optuna.trial.create_trial(value=0.5))
    old_trials = prior.trials
    notes = []

    _, current = run_unified_bayesian(
        **opts, n_trials=1, enable_sqlite_persistence="always", progress_callback=notes.append
    )

    notices = [n for n in notes if n.get("booster_scoring_changed")]
    assert len(notices) == 1
    assert old_name in notices[0]["message"]
    assert notices[0].get(RESUME_DECLINED_KEY) is True
    assert current.study_name != old_name and len(current.trials) == 1
    assert optuna.load_study(study_name=old_name, storage=url).trials == old_trials


def _unsupported_boosters():
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    return [
        pytest.param(
            XGBRegressor(booster="gblinear", n_estimators=20, n_jobs=1), id="xgb-gblinear"
        ),
        pytest.param(XGBRegressor(booster="dart", n_estimators=20, n_jobs=1), id="xgb-dart"),
        pytest.param(
            LGBMRegressor(boosting_type="dart", n_estimators=20, verbose=-1), id="lgbm-dart"
        ),
        pytest.param(
            CatBoostRegressor(
                iterations=20,
                learning_rate=0.1,
                model_shrink_rate=0.1,
                verbose=0,
                allow_writing_files=False,
            ),
            id="catboost-shrink",
        ),
    ]


@pytest.mark.parametrize("model", _unsupported_boosters())
def test_invalid_prefix_configurations_are_not_round_selected(model):
    """Item 3: gblinear, DART and CatBoost shrinkage fall back to the configured count."""
    from spectral_predict.cv_utils import round_selection_unsupported_reason

    X, y = _regression_data()
    assert round_selection_unsupported_reason(model)
    with pytest.raises(ValueError, match="not valid"):
        cross_val_boosting_rounds(model, X, y, KFold(3), patience=5)
    with pytest.warns(UserWarning, match="selection skipped"):
        preds, n_rounds = cross_val_predict_with_early_stopping(
            model, X, y, KFold(3), early_stopping_rounds=5, return_n_rounds=True
        )
    assert n_rounds is None
    np.testing.assert_allclose(preds, cross_val_predict(clone(model), X, y, cv=KFold(3)))


def test_invalid_configuration_grid_row_records_no_selection():
    from lightgbm import LGBMRegressor

    from spectral_predict.search import _run_single_config

    X, y = _regression_data()
    model = LGBMRegressor(boosting_type="dart", n_estimators=20, verbose=-1)
    with pytest.warns(UserWarning, match="selection skipped"):
        row = _run_single_config(
            X,
            y,
            np.linspace(1000.0, 1100.0, X.shape[1]),
            model,
            "LightGBM",
            {},
            {"name": "raw", "deriv": 0, "window": 0, "polyorder": 0},
            KFold(3, shuffle=True, random_state=0),
            "regression",
            False,
            skip_preprocessing=True,
            early_stopping_rounds=10,
        )
    assert row["n_estimators_selected"] is None
    assert row["early_stopping_rounds"] is None
    assert _rounds_param(row["Params"]) == 20


def _full_fit_cases():
    """(id, task, estimator) including CatBoost with automatic round-dependent defaults."""
    out = [
        pytest.param(task, model, id=f"{name}-{task}")
        for task in ("regression", "classification")
        for name, model in _boosters(task)
    ]
    from catboost import CatBoostClassifier, CatBoostRegressor

    out.append(
        pytest.param(
            "regression",
            CatBoostRegressor(
                iterations=60, random_seed=0, verbose=0, thread_count=1, allow_writing_files=False
            ),
            id="catboost-auto-lr-regression",
        )
    )
    out.append(
        pytest.param(
            "classification",
            CatBoostClassifier(
                iterations=60, random_seed=0, verbose=0, thread_count=1, allow_writing_files=False
            ),
            id="catboost-auto-lr-classification",
        )
    )
    return out


@pytest.mark.parametrize(("task", "model"), _full_fit_cases())
def test_truncated_final_model_is_the_full_fit_at_the_selected_round(task, model, tmp_path):
    """Redesign: the final model is the scored configuration fitted on all data at the
    maximum round count and truncated, so its predictions equal that fit's staged
    predictions at k exactly (also after pickling), whatever automatic defaults the
    library chose (CatBoost's learning rate / leaf-estimation iterations)."""
    import pickle

    from spectral_predict.cv_utils import booster_predict_at, truncate_booster

    X, y = _data(task)
    k = 7
    full = clone(model).fit(X, y)
    method = "predict_proba" if task == "classification" else "predict"
    expected = booster_predict_at(full, X, k, method=method)

    truncated = clone(model).fit(X, y)
    truncate_booster(truncated, k)
    got = getattr(truncated, method)(X)
    np.testing.assert_allclose(np.asarray(got).reshape(expected.shape), expected, rtol=0, atol=0)
    params = truncated.get_params()
    assert params.get("n_estimators", params.get("iterations")) == k

    restored = pickle.loads(pickle.dumps(truncated))
    np.testing.assert_allclose(
        np.asarray(getattr(restored, method)(X)).reshape(expected.shape), expected, rtol=0, atol=0
    )


def test_catboost_auto_learning_rate_folds_are_independent_under_resampling():
    """Redesign: each fold chooses CatBoost's automatic rate from its own training data,
    so with a label-dependent resampler (RandomOverSampler) changing only one fold's
    test labels still leaves that fold's model unchanged."""
    from catboost import CatBoostClassifier
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.pipeline import Pipeline as ImbPipeline

    X, y = _classification_data()
    y = y.copy()
    y[:6] = 2
    cv = _FixedSplits(StratifiedKFold(3, shuffle=True, random_state=0).split(X, y))
    splits = list(cv.split(X, y))
    _, test1 = splits[1]
    y2 = y.copy()
    y2[test1] = (y2[test1] + 1) % 3  # changes fold 0's TRAINING class counts, not its test
    pipe = ImbPipeline(
        [
            ("imbalance", RandomOverSampler(random_state=0)),
            (
                "model",
                CatBoostClassifier(
                    iterations=40,
                    random_seed=0,
                    verbose=0,
                    thread_count=1,
                    allow_writing_files=False,
                ),
            ),
        ]
    )
    res1 = cross_val_boosting_rounds(pipe, X, y, cv, patience=10, keep_models=True)
    res2 = cross_val_boosting_rounds(pipe, X, y2, cv, patience=10, keep_models=True)
    # Fold 1's training rows are identical in both runs: its model must be identical,
    # although folds 0 and 2 (whose resampled training sizes changed) differ.
    assert np.array_equal(res1.test_indices[1], test1)
    np.testing.assert_array_equal(
        booster_staged_predict(res1.fold_models[1].steps[-1][1], X[test1]),
        booster_staged_predict(res2.fold_models[1].steps[-1][1], X[test1]),
    )


def test_catboost_link_function_losses_are_staged_on_the_prediction_scale():
    """Item 8: Poisson/Tweedie predictions are exponentiated; staging must match."""
    from catboost import CatBoostRegressor

    X, y = _regression_data()
    y = np.abs(y) * 3
    for loss in ("Poisson", "Tweedie:variance_power=1.5"):
        model = CatBoostRegressor(
            iterations=30,
            learning_rate=0.1,
            loss_function=loss,
            verbose=0,
            thread_count=1,
            allow_writing_files=False,
        )
        model.fit(X, y)
        staged = booster_staged_predict(model, X)
        np.testing.assert_allclose(staged[-1], model.predict(X), rtol=1e-9)
        np.testing.assert_allclose(staged[9], model.predict(X, ntree_end=10), rtol=1e-9)


def test_repeated_cv_selection_uses_the_reported_vote():
    """Item 4: the classification curve is the accuracy of the predictions that would
    be reported at each round count (same majority-vote tie rule)."""
    from sklearn.model_selection import RepeatedStratifiedKFold
    from xgboost import XGBClassifier

    from spectral_predict.cv_utils import booster_predict_at

    X, y = _classification_data()
    y = y.copy()
    y[:10] = 2
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=1)
    res = cross_val_boosting_rounds(
        XGBClassifier(n_estimators=25, max_depth=2, n_jobs=1),
        X,
        y,
        cv,
        patience=None,
        keep_models=True,
    )
    for k in range(1, 26):
        res.fold_predictions = [
            booster_predict_at(m, X[te], k) for m, te in zip(res.fold_models, res.test_indices)
        ]
        pooled = pool_boosting_predictions(res, len(y), y_dtype=y.dtype)
        assert res.curve[k - 1] == pytest.approx(np.mean(pooled == y), abs=1e-12), k


def test_lightgbm_round_aliases_are_normalised():
    """Item 6: a LightGBM round alias overrides n_estimators; read and set it."""
    from lightgbm import LGBMRegressor

    from spectral_predict.cv_utils import booster_max_rounds

    X, y = _regression_data()
    model = LGBMRegressor(n_estimators=50, num_iterations=30, verbose=-1)
    assert booster_max_rounds(model) == 30
    set_booster_rounds(model, 5)
    assert model.fit(X, y).booster_.current_iteration() == 5


def test_eval_only_settings_are_removed_for_cv_and_final_fit():
    """Item 7: early-stopping settings that need an eval set are stripped for the folds
    and the final fit; unrelated callbacks are kept."""
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from xgboost.callback import EarlyStopping, LearningRateScheduler

    from spectral_predict.cv_utils import sanitize_booster, truncate_booster

    X, y = _regression_data()
    lr_schedule = LearningRateScheduler(lambda i: 0.1)
    models = [
        LGBMRegressor(n_estimators=20, early_stopping_round=5, verbose=-1),
        XGBRegressor(
            n_estimators=20,
            early_stopping_rounds=5,
            callbacks=[EarlyStopping(rounds=3), lr_schedule],
        ),
        CatBoostRegressor(
            iterations=20,
            learning_rate=0.1,
            early_stopping_rounds=5,
            use_best_model=True,
            verbose=0,
            allow_writing_files=False,
        ),
    ]
    for model in models:
        res = cross_val_boosting_rounds(model, X, y, KFold(3), patience=5)
        final = sanitize_booster(model)
        final.fit(X, y)
        truncate_booster(final, res.n_rounds)
        if isinstance(model, XGBRegressor):
            callbacks = final.get_params()["callbacks"]
            assert len(callbacks) == 1 and isinstance(callbacks[0], LearningRateScheduler)
            assert final.get_params()["early_stopping_rounds"] is None


def test_select_n_rounds_edge_cases():
    """Item 10: empty curve raises; a NaN tie-break on the incumbent can be beaten."""
    with pytest.raises(ValueError):
        select_n_rounds(np.array([]), None, higher_is_better=False)
    acc = np.array([0.5, 0.8, 0.8])
    tiebreak = np.array([0.7, np.nan, 0.4])
    assert select_n_rounds(acc, None, higher_is_better=True, tiebreak=tiebreak) == 3


def test_nsga2_calibration_and_knee_row_use_selected_rounds():
    """Item 5: calibration metrics describe the stored (selected-count) model, and the
    "best from all evaluations" row gets the same treatment as Pareto rows."""
    from spectral_predict.nsga2_search import (
        _compute_calibration_metrics,
        convert_nsga2_to_v1_format,
        run_nsga2_search,
    )

    X, y = _regression_data()
    result = run_nsga2_search(
        X=X,
        y=y,
        task_type="regression",
        population_size=6,
        n_generations=2,
        cv_folds=3,
        min_wavelengths=5,
        random_state=42,
        verbose=0,
        models=["LightGBM"],
        early_stopping_rounds=10,
        selection_bias=0.0,
    )
    knee = result["knee_solution"]
    knee["objectives"]["error"] = -1.0  # force the best-from-all row into the frame
    result["knee_idx"] = -1
    df = convert_nsga2_to_v1_format(result, X.shape[1], "regression", folds=3, X=X, y=y)
    best = df[df.get("Is_Best_Error", False) == True]  # noqa: E712
    assert len(best) == 1
    assert _rounds_param(best.iloc[0]["Params"]) == int(best.iloc[0]["n_estimators_selected"])
    from spectral_predict.nsga2_search import _booster_cv_metrics

    expected = []
    for solution in result["pareto_solutions"]:
        boost = _booster_cv_metrics(
            X, y, solution, result["model_types"], "regression", 3, 42, early_stopping_rounds=10
        )
        cal = _compute_calibration_metrics(
            X,
            y,
            solution,
            X.shape[1],
            result["model_types"],
            "regression",
            truncate_rounds=boost["n_rounds"],
        )
        expected.append(cal["RMSE"])
    pareto_rows = df[df.get("Is_Best_Error", False) != True]  # noqa: E712
    for rmse in pareto_rows["RMSE"]:
        assert min(abs(rmse - e) for e in expected) < 1e-12


# --- Review round 2 -------------------------------------------------------------------


def test_xgboost_tree_dropout_is_prefix_unsafe():
    """Round 2 #3: dropout under gbtree (rate_drop / one_drop) re-weights earlier trees."""
    from xgboost import XGBRegressor

    from spectral_predict.cv_utils import round_selection_unsupported_reason

    assert round_selection_unsupported_reason(XGBRegressor(rate_drop=0.5))
    assert round_selection_unsupported_reason(XGBRegressor(one_drop=1))
    assert round_selection_unsupported_reason(XGBRegressor(skip_drop=0.5)) is None
    assert round_selection_unsupported_reason(XGBRegressor()) is None


def test_eval_only_settings_removed_when_selection_is_rejected():
    """Round 2 #4: boosters are sanitized whether or not round selection runs."""
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    X, y = _regression_data()
    for model in (
        XGBRegressor(booster="dart", n_estimators=20, early_stopping_rounds=2, n_jobs=1),
        LGBMRegressor(boosting_type="dart", n_estimators=20, early_stopping_round=2, verbose=-1),
    ):
        with pytest.warns(UserWarning, match="selection skipped"):
            preds = cross_val_predict_with_early_stopping(
                model, X, y, KFold(3), early_stopping_rounds=5
            )
        assert preds.shape == y.shape
        with pytest.warns(UserWarning, match="selection skipped"):
            out = cross_validate_with_early_stopping(model, X, y, KFold(3), early_stopping_rounds=5)
        assert len(out["test_score"]) == 3


def test_grid_rejected_selection_with_eval_only_settings_does_not_fail():
    from lightgbm import LGBMRegressor

    from spectral_predict.search import _run_single_config

    X, y = _regression_data()
    model = LGBMRegressor(boosting_type="dart", n_estimators=20, early_stopping_round=2, verbose=-1)
    with pytest.warns(UserWarning, match="selection skipped"):
        row = _run_single_config(
            X,
            y,
            np.linspace(1000.0, 1100.0, X.shape[1]),
            model,
            "LightGBM",
            {},
            {"name": "raw", "deriv": 0, "window": 0, "polyorder": 0},
            KFold(3, shuffle=True, random_state=0),
            "regression",
            False,
            skip_preprocessing=True,
            early_stopping_rounds=10,
        )
    assert np.isfinite(row["RMSEcv"])
    assert row["round_selection_truncated"] is False


def test_validation_rebuild_reproduces_fit_then_truncate(monkeypatch):
    """Rebuilds of a truncated row fit at n_estimators_fit and truncate to the selected
    count (round_truncation_from_row); a CatBoost row with an automatic learning rate
    therefore rebuilds the reported model, not a direct fit at the selected count."""
    from catboost import CatBoostRegressor

    from spectral_predict.cv_utils import (
        round_truncation_from_row,
        set_booster_rounds,
        truncate_booster,
    )

    row = {"n_estimators_selected": 9, "n_estimators_fit": 60, "round_selection_truncated": True}
    assert round_truncation_from_row(row) == (60, 9)
    assert round_truncation_from_row({"n_estimators_selected": 9}) is None
    X, y = _regression_data()
    base = dict(random_seed=0, verbose=0, thread_count=1, allow_writing_files=False)
    reported = CatBoostRegressor(iterations=60, **base).fit(X, y)
    truncate_booster(reported, 9)
    rebuilt = CatBoostRegressor(iterations=9, **base)  # Params carry the selected count
    set_booster_rounds(rebuilt, 60)
    rebuilt.fit(X, y)
    truncate_booster(rebuilt, 9)
    np.testing.assert_array_equal(rebuilt.predict(X), reported.predict(X))
    direct = CatBoostRegressor(iterations=9, **base).fit(X, y)
    assert not np.allclose(direct.predict(X), reported.predict(X))


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_nsga2_rows_record_truncation_and_parseable_params(task):
    """Round 2 #2/#6 and GLM gaps: XGBoost Params parse (no 'missing': nan) and carry
    the selected count; every row records the fit-then-truncate procedure; the
    best-from-all row (classification too) carries the imbalance metadata."""
    from spectral_predict.nsga2_search import convert_nsga2_to_v1_format, run_nsga2_search

    X, y = _data(task)
    imbalance = "class_weight" if task == "classification" else None
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
        models=["XGBoost"],
        early_stopping_rounds=10,
        selection_bias=0.0,
        imbalance_method=imbalance,
    )
    result["knee_solution"]["objectives"]["error"] = -1.0  # force the best-from-all row
    result["knee_idx"] = -1
    df = convert_nsga2_to_v1_format(result, X.shape[1], task, folds=3, X=X, y=y)
    best = df[df.get("Is_Best_Error", False) == True]  # noqa: E712
    assert len(best) == 1
    for _, row in df.iterrows():
        params = ast.literal_eval(row["Params"])
        assert params["n_estimators"] == int(row["n_estimators_selected"])
        assert int(row["n_estimators_fit"]) >= int(row["n_estimators_selected"])
        assert bool(row["round_selection_truncated"])
    assert best.iloc[0]["imbalance_method"] == imbalance


# --- Review round 3 -------------------------------------------------------------------


def _catboost_eval_configured():
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=30,
        learning_rate=0.1,
        od_type="Iter",
        od_wait=5,
        od_pval=None,
        early_stopping_rounds=5,
        verbose=0,
        thread_count=1,
        allow_writing_files=False,
    )


@pytest.mark.parametrize("es", [0, 10], ids=["selection-disabled", "selection-on"])
def test_sanitized_catboost_can_be_cloned_and_fitted(es):
    """Round 3 #3: eval-only keys are removed (not nulled): clone + fit still work."""
    from spectral_predict.cv_utils import sanitize_booster

    X, y = _regression_data()
    model = _catboost_eval_configured()
    clean = sanitize_booster(model)
    for key in ("od_type", "od_wait", "early_stopping_rounds"):
        assert key not in clean.get_params()
    clone(clean).fit(X, y)
    preds = cross_val_predict_with_early_stopping(model, X, y, KFold(3), early_stopping_rounds=es)
    assert preds.shape == y.shape


def test_sanitized_catboost_rejected_selection_can_be_cloned_and_fitted():
    from catboost import CatBoostRegressor

    X, y = _regression_data()
    model = CatBoostRegressor(
        iterations=20,
        learning_rate=0.1,
        model_shrink_rate=0.1,
        od_type="Iter",
        od_wait=5,
        verbose=0,
        thread_count=1,
        allow_writing_files=False,
    )
    with pytest.warns(UserWarning, match="selection skipped"):
        preds = cross_val_predict_with_early_stopping(
            model, X, y, KFold(3), early_stopping_rounds=5
        )
    assert preds.shape == y.shape


def test_truncate_booster_validates_the_round_count():
    """Round 3 #7: k must be a positive integer within the configured rounds, checked
    before anything changes; a LightGBM model that built fewer trees is allowed."""
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    from spectral_predict.cv_utils import truncate_booster

    X, y = _regression_data()
    for model in (
        XGBRegressor(n_estimators=10, n_jobs=1),
        LGBMRegressor(n_estimators=10, verbose=-1),
        CatBoostRegressor(iterations=10, learning_rate=0.1, verbose=0, allow_writing_files=False),
    ):
        model.fit(X, y)
        before = model.predict(X)
        for bad in (0, -1, 11, 2.5, "3", True):
            with pytest.raises(ValueError):
                truncate_booster(model, bad)
        np.testing.assert_array_equal(model.predict(X), before)  # untouched
        truncate_booster(model, 10)  # boundary: all configured rounds
        truncate_booster(model, np.int64(1))  # boundary: one round
    # LightGBM may stop early: fewer trees than configured is fine.
    sparse = LGBMRegressor(n_estimators=50, min_child_samples=100, verbose=-1).fit(X, y)
    assert sparse.booster_.current_iteration() < 50
    truncate_booster(sparse, 40)


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (
            {"round_selection_truncated": True, "n_estimators_fit": 60, "n_estimators_selected": 9},
            (60, 9),
        ),
        (
            {
                "round_selection_truncated": "True",
                "n_estimators_fit": "60",
                "n_estimators_selected": 9.0,
            },
            (60, 9),
        ),
        (
            {
                "round_selection_truncated": 1,
                "n_estimators_fit": 60.0,
                "n_estimators_selected": "9",
            },
            (60, 9),
        ),
        (
            {
                "round_selection_truncated": "False",
                "n_estimators_fit": 60,
                "n_estimators_selected": 9,
            },
            None,
        ),
        (
            {
                "round_selection_truncated": float("nan"),
                "n_estimators_fit": 60,
                "n_estimators_selected": 9,
            },
            None,
        ),
        (
            {
                "round_selection_truncated": True,
                "n_estimators_fit": float("nan"),
                "n_estimators_selected": 9,
            },
            None,
        ),
        (
            {"round_selection_truncated": True, "n_estimators_fit": 60, "n_estimators_selected": 0},
            None,
        ),
        (
            {"round_selection_truncated": True, "n_estimators_fit": 8, "n_estimators_selected": 9},
            None,
        ),
        (
            {
                "round_selection_truncated": True,
                "n_estimators_fit": 60,
                "n_estimators_selected": 9.5,
            },
            None,
        ),
        ({}, None),
    ],
)
def test_round_truncation_metadata_is_parsed_explicitly(row, expected):
    from spectral_predict.cv_utils import round_truncation_from_row

    assert round_truncation_from_row(pd.Series(row) if row else row) == expected


def test_truncation_private_attributes_exist():
    """Version sentinel: truncate_booster relies on these library internals."""
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    X, y = _regression_data()
    xgb_model = XGBRegressor(n_estimators=5, n_jobs=1).fit(X, y)
    assert xgb_model.get_booster() is xgb_model._Booster
    assert xgb_model.get_booster()[:2].num_boosted_rounds() == 2
    lgb_model = LGBMRegressor(n_estimators=5, verbose=-1).fit(X, y)
    assert lgb_model.booster_ is lgb_model._Booster
    cb_model = CatBoostRegressor(
        iterations=5, learning_rate=0.1, verbose=0, allow_writing_files=False
    ).fit(X, y)
    assert isinstance(cb_model._init_params, dict) and "iterations" in cb_model._init_params
    assert callable(cb_model.shrink)


def test_truncated_booster_survives_model_io_round_trip(tmp_path):
    from catboost import CatBoostRegressor

    from spectral_predict.cv_utils import truncate_booster
    from spectral_predict.model_io import load_model, predict_with_model, save_model

    X, y = _regression_data()
    model = CatBoostRegressor(
        iterations=40, random_seed=0, verbose=0, thread_count=1, allow_writing_files=False
    ).fit(X, y)
    truncate_booster(model, 7)
    expected = model.predict(X)
    path = tmp_path / "truncated.dasp"
    save_model(
        model,
        None,
        {
            "model_name": "CatBoost",
            "task_type": "regression",
            "wavelengths": list(range(X.shape[1])),
            "n_vars": X.shape[1],
        },
        path,
    )
    loaded = load_model(path)
    assert loaded["model"].tree_count_ == 7
    np.testing.assert_array_equal(predict_with_model(loaded, X), expected)


def test_fitted_target_transform_wrapper_truncation():
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    from spectral_predict.cv_utils import booster_predict_at, truncate_booster

    X, y = _regression_data()
    y = y - y.min() + 1.0
    wrapped = TransformedTargetRegressor(
        regressor=Pipeline(
            [("scaler", StandardScaler()), ("model", XGBRegressor(n_estimators=30, n_jobs=1))]
        ),
        func=np.log,
        inverse_func=np.exp,
    ).fit(X, y)
    inner = wrapped.regressor_
    expected = np.exp(booster_predict_at(inner.steps[-1][1], inner[:-1].transform(X), 6))
    truncate_booster(wrapped, 6)
    np.testing.assert_allclose(wrapped.predict(X), expected, rtol=1e-12)


def test_grid_row_rebuilds_to_the_reported_model_in_validation():
    """Round 3 #2: a CatBoost grid row whose learning rate was automatic rebuilds (fit at
    n_estimators_fit, truncate) to the reported model: validating on the calibration
    data itself reproduces the row's calibration RMSE."""
    from spectral_predict.search import compute_validation_metrics_for_top_models, run_search

    X, y = _regression_data()
    # Integer wavelengths: all_vars is written with %g (R031, fixed elsewhere), which
    # would otherwise drop columns in the validation rebuild.
    wl = np.arange(1000.0, 1000.0 + X.shape[1])
    df, _ = run_search(
        pd.DataFrame(X, columns=wl),
        pd.Series(y),
        "regression",
        folds=3,
        tier="quick",
        models_to_test=["CatBoost"],
        enabled_models=["CatBoost"],
        preprocessing_methods={"raw": True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        catboost_iterations_list=[40],
        catboost_depths=[3],
        catboost_learning_rates=[None],
        catboost_l2_leaf_reg_list=[3.0],
        catboost_border_count_list=[32],
        catboost_bagging_temperature_list=[1.0],
        catboost_random_strength_list=[1.0],
        early_stopping_rounds=10,
    )
    row = df.iloc[0]
    assert bool(row["round_selection_truncated"])
    assert "learning_rate" not in ast.literal_eval(row["Params"])  # automatic
    out = compute_validation_metrics_for_top_models(
        df.head(1), X, y, X, y, "regression", wl, top_n=1
    )
    assert out.iloc[0]["RMSEP"] == pytest.approx(row["RMSE"], rel=1e-9)


def test_native_model_exports_to_the_same_rounds_and_predictions():
    """Round 3 #2: a native CatBoost model (automatic learning rate) -> its row -> export
    reproduces the selected count and the final model's predictions."""
    from catboost import CatBoostRegressor

    from spectral_predict.code_generator import CodeGenerator, ExportOptions
    from spectral_predict.cv_utils import truncate_booster

    X, y = _regression_data()
    base = dict(
        iterations=40, depth=3, random_state=0, verbose=0, thread_count=1, allow_writing_files=False
    )
    native = CatBoostRegressor(**base)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    res = cross_val_boosting_rounds(native, X, y, cv, patience=10)
    final = clone(native).fit(X, y)
    truncate_booster(final, res.n_rounds)
    row_params = dict(final.get_params())  # what the results row stores (count = k)
    assert "learning_rate" not in row_params
    config = {
        "model_name": "CatBoost",
        "preprocessing": "raw",
        "task_type": "regression",
        "target_name": "target",
        "params": row_params,
        "metrics": {},
        "cv_folds": 5,
        "cv_strategy": "kfold",
        "cv_n_repeats": 5,
        "imbalance_method": None,
        "imbalance_params": {},
        "autoscale": False,
        "variable_indices": None,
        "variable_selection_method": None,
        "trim_derivative_edges": False,
        "inlier_class_label": "",
        "wavelengths": list(range(X.shape[1])),
        "early_stopping_rounds": 10,
        "n_estimators_selected": res.n_rounds,
        "n_estimators_fit": 40,
        "round_selection_truncated": True,
    }
    opts = ExportOptions(
        format="script",
        include_data=True,
        data_X=X,
        data_y=y,
        wavelengths=None,
        include_visualization=False,
    )
    ns: dict = {}
    exec(CodeGenerator(config, opts).generate_script(), ns)
    assert ns["N_BOOST_ROUNDS"] == res.n_rounds
    np.testing.assert_array_equal(ns["model"].predict(X), final.predict(X))


def test_export_reports_calibration_of_the_truncated_model():
    """Round 3 #1: the exported calibration metrics describe the final (truncated) model."""
    from lightgbm import LGBMRegressor  # noqa: F401  (export imports it)

    from spectral_predict.code_generator import CodeGenerator, ExportOptions

    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 20))
    y = rng.normal(size=40)  # noise: the pooled curve picks very few rounds
    params = {
        "n_estimators": 60,
        "learning_rate": 0.3,
        "num_leaves": 7,
        "min_child_samples": 3,
        "random_state": 0,
        "n_jobs": 1,
        "verbosity": -1,
    }
    config = {
        "model_name": "LightGBM",
        "preprocessing": "raw",
        "task_type": "regression",
        "target_name": "target",
        "params": params,
        "metrics": {},
        "cv_folds": 5,
        "cv_strategy": "kfold",
        "cv_n_repeats": 5,
        "imbalance_method": None,
        "imbalance_params": {},
        "autoscale": False,
        "variable_indices": None,
        "variable_selection_method": None,
        "trim_derivative_edges": False,
        "inlier_class_label": "",
        "wavelengths": list(range(20)),
        "early_stopping_rounds": 10,
    }
    for fmt in ("script", "notebook"):
        opts = ExportOptions(
            format=fmt,
            include_data=True,
            data_X=X,
            data_y=y,
            wavelengths=None,
            include_visualization=False,
            colab_ready=False,
        )
        gen = CodeGenerator(config, opts)
        ns: dict = {}
        if fmt == "script":
            exec(gen.generate_script(), ns)
        else:
            for cell in gen.generate_notebook()["cells"]:
                code = "".join(cell["source"])
                if cell["cell_type"] == "code" and "subprocess.check_call" not in code:
                    exec(code, ns)
        assert ns["N_BOOST_ROUNDS"] < 60
        X_used = next(ns[k] for k in ("X_final", "X_processed", "X") if k in ns)
        own = ns["model"].predict(X_used)
        assert ns["cal_rmse"] == pytest.approx(float(np.sqrt(np.mean((y - own) ** 2))), rel=1e-12)


def test_nsga2_malformed_params_row_is_kept_without_truncation():
    """Round 3 #6: one unparseable Params string must not abort the conversion."""
    from spectral_predict.nsga2_search import _record_round_selection

    row = {"Params": "{'n_estimators': 50, 'missing': nan}"}
    with pytest.warns(UserWarning, match="could not record"):
        _record_round_selection(
            row, {"rounds_key": "n_estimators", "n_rounds": 7, "fit_rounds": 50}
        )
    assert row["round_selection_truncated"] is False
    assert row["n_estimators_selected"] is None
    assert row["Params"] == "{'n_estimators': 50, 'missing': nan}"


def test_failed_final_refit_does_not_claim_a_truncated_model(monkeypatch):
    """GLM LOW 8: if the final refit fails, the flags must not claim truncation."""
    import spectral_predict.search as search_mod

    def boom(*args, **kwargs):
        raise RuntimeError("refit failed")

    monkeypatch.setattr(search_mod, "truncate_booster", boom)
    from lightgbm import LGBMRegressor

    X, y = _regression_data()
    row = search_mod._run_single_config(
        X,
        y,
        np.linspace(1000.0, 1100.0, X.shape[1]),
        LGBMRegressor(n_estimators=30, verbose=-1),
        "LightGBM",
        {},
        {"name": "raw", "deriv": 0, "window": 0, "polyorder": 0},
        KFold(3, shuffle=True, random_state=0),
        "regression",
        False,
        skip_preprocessing=True,
        early_stopping_rounds=10,
    )
    assert row["round_selection_truncated"] is False
    assert row["n_estimators_fit"] is None


# --- Review round 4 -------------------------------------------------------------------


def test_round_truncated_wrapper_on_target_transformed_booster():
    """Round 4 #3: booster_max_rounds / set_booster_rounds resolve through a
    TransformedTargetRegressor, so the wrapper fits it at R and truncates to k."""
    from sklearn.compose import TransformedTargetRegressor
    from xgboost import XGBRegressor

    from spectral_predict.cv_utils import (
        booster_max_rounds,
        booster_predict_at,
        round_truncated_from_row,
    )

    X, y = _regression_data()
    y = y - y.min() + 1.0
    ttr = TransformedTargetRegressor(
        regressor=XGBRegressor(n_estimators=9, n_jobs=1, random_state=0),
        func=np.log,
        inverse_func=np.exp,
    )
    assert booster_max_rounds(ttr) == 9
    row = {"round_selection_truncated": True, "n_estimators_fit": 30, "n_estimators_selected": 9}
    wrapped = round_truncated_from_row(ttr, row, "regression").fit(X, y)
    reference = XGBRegressor(n_estimators=30, n_jobs=1, random_state=0).fit(X, np.log(y))
    np.testing.assert_allclose(
        np.log(wrapped.predict(X)), booster_predict_at(reference, X, 9), rtol=1e-6, atol=1e-6
    )


def test_cross_val_boosting_rounds_accepts_a_target_transform_wrapper():
    from sklearn.compose import TransformedTargetRegressor
    from xgboost import XGBRegressor

    from spectral_predict.y_transform import YTransformWrapper

    X, y = _regression_data()
    y = y - y.min() + 1.0
    cv = KFold(3, shuffle=True, random_state=0)
    base = XGBRegressor(n_estimators=30, n_jobs=1, random_state=0)
    via_wrapper = cross_val_boosting_rounds(
        TransformedTargetRegressor(regressor=base, func=np.log, inverse_func=np.exp),
        X,
        y,
        cv,
        patience=5,
    )
    via_transformer = cross_val_boosting_rounds(
        base, X, y, cv, patience=5, target_transformer=YTransformWrapper._get_transformer("log")
    )
    assert via_wrapper.n_rounds == via_transformer.n_rounds
    for a, b in zip(via_wrapper.fold_predictions, via_transformer.fold_predictions):
        np.testing.assert_allclose(a, b, rtol=1e-12)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float("inf"), None),
        (float("-inf"), None),
        ("inf", None),
        (1e400, None),
        ("1e3", 1000),
        (7.0, 7),
        (np.float32(3.0), 3),
        (0.5, None),
        (None, None),
        (True, None),
    ],
)
def test_parse_count_is_total(value, expected):
    from spectral_predict.cv_utils import parse_count_cell

    assert parse_count_cell(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.0, True),
        (0.0, False),
        (np.float64(1.0), True),
        (2.0, False),
        ("TRUE", True),
        ("0", False),
        (float("nan"), False),
        (np.bool_(True), True),
    ],
)
def test_parse_bool_accepts_integral_floats(value, expected):
    from spectral_predict.cv_utils import parse_bool_cell

    assert parse_bool_cell(value) is expected


@pytest.mark.parametrize(
    ("flag", "fit", "selected", "expected"),
    [
        (True, 40, 9, (40, 9)),
        ("False", 40, 9, (None, None)),
        (1.0, 40, 9, (40, 9)),
        (True, 8, 9, (None, None)),
        (True, float("inf"), 9, (None, None)),
        (True, 40, 0, (None, None)),
    ],
)
def test_code_generator_parses_truncation_metadata_like_the_rebuild(flag, fit, selected, expected):
    from spectral_predict.code_generator import CodeGenerator

    gen = CodeGenerator(
        {
            "model_name": "XGBoost",
            "task_type": "regression",
            "params": {},
            "round_selection_truncated": flag,
            "n_estimators_fit": fit,
            "n_estimators_selected": selected,
        }
    )
    assert (gen.n_estimators_fit, gen.n_estimators_selected) == expected


def _wrapper_cases():
    from catboost import CatBoostClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier, XGBRegressor

    return [
        pytest.param(
            "regression",
            XGBRegressor(n_estimators=5, n_jobs=1, random_state=0),
            id="xgb-regression",
        ),
        pytest.param(
            "binary", XGBClassifier(n_estimators=5, n_jobs=1, random_state=0), id="xgb-binary"
        ),
        pytest.param(
            "multiclass",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        CatBoostClassifier(
                            iterations=5,
                            random_seed=0,
                            verbose=0,
                            thread_count=1,
                            allow_writing_files=False,
                        ),
                    ),
                ]
            ),
            id="catboost-multiclass-pipeline",
        ),
    ]


@pytest.mark.parametrize(("kind", "estimator"), _wrapper_cases())
def test_round_truncated_wrappers_clone_pickle_and_predict(kind, estimator):
    """Round 4 #5: wrapper unit tests (clone, pickle, predict_proba, multiclass,
    Pipeline(scaler, booster)) against a reference fitted at R and truncated to k."""
    import pickle

    from spectral_predict.cv_utils import (
        RoundTruncatedClassifier,
        round_truncated_from_row,
        set_booster_rounds,
        truncate_booster,
    )

    X, y = _regression_data()
    task = "regression"
    if kind == "binary":
        y, task = (y > np.median(y)).astype(int), "classification"
    elif kind == "multiclass":
        y, task = np.digitize(y, np.quantile(y, [1 / 3, 2 / 3])), "classification"
    row = {"round_selection_truncated": True, "n_estimators_fit": 25, "n_estimators_selected": 6}
    wrapped = round_truncated_from_row(estimator, row, task)
    assert isinstance(wrapped, RoundTruncatedClassifier) == (task == "classification")
    fitted = clone(wrapped).fit(X, y)

    reference = clone(estimator)
    set_booster_rounds(reference, 25)
    reference.fit(X, y)
    truncate_booster(reference, 6)
    method = "predict_proba" if task == "classification" else "predict"
    expected = getattr(reference, method)(X)
    np.testing.assert_allclose(getattr(fitted, method)(X), expected, rtol=0, atol=1e-12)
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_allclose(getattr(restored, method)(X), expected, rtol=0, atol=1e-12)
    if task == "classification":
        np.testing.assert_array_equal(fitted.classes_, np.unique(y))
        np.testing.assert_array_equal(np.ravel(fitted.predict(X)), np.ravel(reference.predict(X)))


@pytest.mark.parametrize("library", ["xgboost", "lightgbm"])
def test_truncated_xgboost_lightgbm_model_io_round_trip(library, tmp_path):
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    from spectral_predict.cv_utils import truncate_booster
    from spectral_predict.model_io import load_model, predict_with_model, save_model

    X, y = _regression_data()
    model = (
        XGBRegressor(n_estimators=40, n_jobs=1, random_state=0)
        if library == "xgboost"
        else LGBMRegressor(n_estimators=40, verbose=-1, random_state=0)
    ).fit(X, y)
    truncate_booster(model, 7)
    expected = model.predict(X)
    path = tmp_path / f"{library}.dasp"
    save_model(
        model,
        None,
        {
            "model_name": library,
            "task_type": "regression",
            "wavelengths": list(range(X.shape[1])),
            "n_vars": X.shape[1],
        },
        path,
    )
    loaded = load_model(path)
    assert loaded["model"].get_params()["n_estimators"] == 7
    np.testing.assert_array_equal(predict_with_model(loaded, X), expected)
