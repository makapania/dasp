"""Export-side CV parity for boosters: one round count from the pooled CV curve.

Pins the contract that the exported notebook/script CV reproduces the in-app
booster CV (``cv_utils.cross_val_boosting_rounds``, used by grid search, the
Bayesian and NSGA-II objectives and Model Development).

Background
----------
Until 2026-10 both the app and the export early-stopped every booster fold on
its own held-out test fold (``eval_set=[(X_test, y_test)]``), so each fold chose
its tree count from the labels it was then scored on (review findings R028,
R003, R022). The fix treats the round count like the number of PLS latent
variables: every fold is fitted with the maximum round count and no eval_set,
the test predictions at every round count are pooled across folds, and ONE
count is chosen from the pooled curve (``early_stopping_rounds`` is the
patience of that scan). This file used to pin the leaky convention; it now
pins the corrected one.

What this pins
--------------
1. The generator threads ``early_stopping_rounds`` into ``EARLY_STOPPING_ROUNDS``.
2. The exported CV picks the same round count and produces per-sample
   predictions identical to ``cross_val_boosting_rounds`` (classification,
   regression, and XGBoost with class-weight sample weights).
3. The exported final model is fitted with the selected round count.
4. ``early_stopping_rounds`` None/0 falls through to a plain fit of the
   configured round count.
5. No exported code passes an eval_set.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold

from spectral_predict.code_generator import CodeGenerator, ExportOptions
from spectral_predict.cv_utils import cross_val_boosting_rounds, pool_boosting_predictions


def _make_data(seed: int = 42):
    """Three-class data sized like the user's collagen-cat case (41×20)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((41, 20))
    y = np.concatenate([np.zeros(21, int), np.ones(6, int), 2 * np.ones(14, int)])
    rng.shuffle(y)
    X[:, 0] += y  # some signal so the curve is not flat
    return X, y


def _make_regression_data(seed: int = 7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((40, 20))
    y = X[:, 0] - 0.5 * X[:, 1] + 0.3 * rng.standard_normal(40)
    return X, y


def _lightgbm_params() -> dict:
    return {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "num_leaves": 50,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "min_child_samples": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": 1,
        "verbosity": -1,
        "bagging_freq": 1,
    }


def _build_model_config(
    early_stopping_rounds,
    params=None,
    model_name="LightGBM",
    task_type="classification",
    imbalance_method=None,
) -> dict:
    return {
        "model_name": model_name,
        "preprocessing": "raw",
        "task_type": task_type,
        "target_name": "target",
        "params": params or _lightgbm_params(),
        "metrics": {},
        "cv_folds": 5,
        "cv_strategy": "kfold",
        "cv_n_repeats": 5,
        "imbalance_method": imbalance_method,
        "imbalance_params": {},
        "autoscale": False,
        "variable_indices": None,
        "variable_selection_method": None,
        "trim_derivative_edges": False,
        "inlier_class_label": "",
        "wavelengths": list(range(20)),
        "early_stopping_rounds": early_stopping_rounds,
    }


def _exec_generated(model_config, X, y) -> dict:
    """Generate the notebook, exec every code cell, return the namespace."""
    opts = ExportOptions(
        format="notebook",
        include_data=True,
        data_X=X,
        data_y=y,
        wavelengths=None,
        colab_ready=False,
        include_visualization=False,
    )
    nb = CodeGenerator(model_config, opts).generate_notebook()
    ns: dict = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        code = "".join(cell["source"])
        # Skip the install-deps cell — venv has everything; subprocess pip is slow.
        if "subprocess.check_call" in code:
            continue
        exec(code, ns)
    return ns


def _notebook_code(model_config, X, y) -> str:
    opts = ExportOptions(
        format="notebook",
        include_data=True,
        data_X=X,
        data_y=y,
        wavelengths=None,
        colab_ready=False,
        include_visualization=False,
    )
    nb = CodeGenerator(model_config, opts).generate_notebook()
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def _inapp(model, X, y, cv, esr, **kwargs):
    res = cross_val_boosting_rounds(model, X, y, cv, patience=esr, **kwargs)
    return res, pool_boosting_predictions(res, len(y), y_dtype=np.asarray(y).dtype)


def test_lightgbm_classification_export_matches_inapp():
    """Export CV with early_stopping_rounds=40 picks the in-app round count and
    reproduces the in-app per-sample predictions."""
    from lightgbm import LGBMClassifier

    X, y = _make_data()
    params = _lightgbm_params()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    res, inapp_preds = _inapp(LGBMClassifier(**params), X, y, cv, 40)

    ns = _exec_generated(_build_model_config(40, params), X, y)

    assert ns["N_BOOST_ROUNDS"] == res.n_rounds
    assert np.array_equal(inapp_preds, ns["all_y_pred_arr"]), (
        f"Export CV diverges from in-app CV.\n"
        f"  in-app preds:  {inapp_preds.tolist()}\n"
        f"  export preds:  {ns['all_y_pred_arr'].tolist()}"
    )
    # The exported final model is fitted with the selected round count.
    assert ns["model"].get_params()["n_estimators"] == res.n_rounds
    # The exported final model is the full fit at the maximum, truncated (as in-app).
    from spectral_predict.cv_utils import truncate_booster

    inapp_final = LGBMClassifier(**params).fit(X, y)
    truncate_booster(inapp_final, res.n_rounds)
    np.testing.assert_array_equal(ns["model"].predict_proba(X), inapp_final.predict_proba(X))


def test_lightgbm_regression_export_matches_inapp():
    from lightgbm import LGBMRegressor

    X, y = _make_regression_data()
    params = _lightgbm_params()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    res, inapp_preds = _inapp(LGBMRegressor(**params), X, y, cv, 20)

    ns = _exec_generated(_build_model_config(20, params, task_type="regression"), X, y)

    assert ns["N_BOOST_ROUNDS"] == res.n_rounds
    np.testing.assert_allclose(ns["all_y_pred_arr"], inapp_preds, rtol=0, atol=1e-10)


def test_xgboost_class_weight_export_matches_inapp():
    """XGBoost + class_weight: balanced sample weights per training fold in both."""
    from xgboost import XGBClassifier

    X, y = _make_data()
    params = {
        "n_estimators": 80,
        "learning_rate": 0.1,
        "max_depth": 3,
        "random_state": 42,
        "n_jobs": 1,
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    res, inapp_preds = _inapp(XGBClassifier(**params), X, y, cv, 15, balanced_sample_weight=True)

    ns = _exec_generated(
        _build_model_config(15, params, model_name="XGBoost", imbalance_method="class_weight"),
        X,
        y,
    )

    assert ns["N_BOOST_ROUNDS"] == res.n_rounds
    assert np.array_equal(inapp_preds, ns["all_y_pred_arr"])


def test_lightgbm_no_early_stop_export_matches_plain_fit():
    """early_stopping_rounds=None must fall through to plain .fit() of the
    configured round count."""
    from lightgbm import LGBMClassifier

    X, y = _make_data()
    params = _lightgbm_params()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    plain = np.empty_like(y)
    for tr, te in cv.split(X, y):
        m = clone(LGBMClassifier(**params)).fit(X[tr], y[tr])
        plain[te] = m.predict(X[te])

    ns = _exec_generated(_build_model_config(None, params), X, y)

    assert ns["N_BOOST_ROUNDS"] is None
    assert np.array_equal(plain, ns["all_y_pred_arr"])


def test_zero_early_stop_treated_as_disabled():
    """early_stopping_rounds=0 must behave like None (disabled)."""
    X, y = _make_data()
    params = _lightgbm_params()

    plain_preds = _exec_generated(_build_model_config(None, params), X, y)["all_y_pred_arr"]
    zero_preds = _exec_generated(_build_model_config(0, params), X, y)["all_y_pred_arr"]

    assert np.array_equal(plain_preds, zero_preds)


@pytest.mark.parametrize("esr", [None, 40])
def test_emitted_constant_reflects_threaded_value(esr):
    """The notebook source must contain the threaded EARLY_STOPPING_ROUNDS
    value so reviewers can see what setting produced the reported numbers."""
    X, y = _make_data()
    code = _notebook_code(_build_model_config(esr), X, y)
    expected = 0 if esr is None else esr
    assert f"EARLY_STOPPING_ROUNDS = {expected}" in code


@pytest.mark.parametrize("imbalance_method", [None, "class_weight", "smote"])
def test_export_never_passes_an_eval_set(imbalance_method):
    """No exported CV path may early-stop on the test fold."""
    X, y = _make_data()
    code = _notebook_code(_build_model_config(40, imbalance_method=imbalance_method), X, y)
    assert "eval_set=" not in code
    assert "_choose_boosting_rounds(" in code


def test_export_fits_each_booster_fold_once(monkeypatch):
    """One fit per fold (at the maximum round count) plus the final fit: the CV loop
    reports the round-selection fits' predictions instead of refitting each fold."""
    from lightgbm import LGBMClassifier

    calls = []
    real_fit = LGBMClassifier.fit

    def counting_fit(self, *args, **kwargs):
        calls.append(1)
        return real_fit(self, *args, **kwargs)

    monkeypatch.setattr(LGBMClassifier, "fit", counting_fit)
    X, y = _make_data()
    _exec_generated(_build_model_config(40), X, y)
    assert len(calls) == 5 + 1


def test_export_regression_fits_each_booster_fold_once(monkeypatch):
    from lightgbm import LGBMRegressor

    calls = []
    real_fit = LGBMRegressor.fit

    def counting_fit(self, *args, **kwargs):
        calls.append(1)
        return real_fit(self, *args, **kwargs)

    monkeypatch.setattr(LGBMRegressor, "fit", counting_fit)
    X, y = _make_regression_data()
    _exec_generated(_build_model_config(20, task_type="regression"), X, y)
    assert len(calls) == 5 + 1


def test_export_without_cv_reproduces_the_truncated_row():
    """A row fitted at n_estimators_fit and truncated is reproduced by an export
    without cross-validation: fit at the recorded maximum, truncate to the count."""
    from catboost import CatBoostRegressor

    from spectral_predict.cv_utils import truncate_booster

    X, y = _make_regression_data()
    params = {
        "iterations": 9,
        "depth": 3,
        "random_state": 0,
        "verbose": 0,
        "thread_count": 1,
        "allow_writing_files": False,
    }
    config = _build_model_config(None, params, model_name="CatBoost", task_type="regression")
    config.update(n_estimators_selected=9, n_estimators_fit=40, round_selection_truncated=True)
    opts = ExportOptions(
        format="script",
        include_data=True,
        data_X=X,
        data_y=y,
        wavelengths=None,
        include_visualization=False,
        include_cross_validation=False,
    )
    script = CodeGenerator(config, opts).generate_script()
    # Pre-existing gap: the regression final-model template prints CCC with
    # _lins_ccc, which only the CV section defines.
    ns: dict = {"_lins_ccc": lambda a, b: float("nan")}
    exec(script, ns)
    assert "'iterations': 40" in script  # fitted at the recorded maximum ...
    assert ns["model"].tree_count_ == 9  # ... and truncated to the selected count
    exported = dict(ns["model"].get_params())
    exported["iterations"] = 40
    reference = CatBoostRegressor(**exported).fit(X, y)
    truncate_booster(reference, 9)
    np.testing.assert_array_equal(ns["model"].predict(X), reference.predict(X))


def test_export_lightgbm_alias_resolves_effective_round_count():
    """Round 2 #7: an overriding LightGBM alias is resolved before aliases are dropped."""
    from lightgbm import LGBMRegressor

    X, y = _make_regression_data()
    params = {
        "n_estimators": 100,
        "num_iterations": 7,
        "boosting_type": "dart",
        "random_state": 0,
        "n_jobs": 1,
        "verbosity": -1,
    }
    ns = _exec_generated(_build_model_config(None, params, task_type="regression"), X, y)
    native = LGBMRegressor(**params).fit(X, y)
    assert ns["model"].booster_.current_iteration() == 7
    np.testing.assert_allclose(ns["model"].predict(X), native.predict(X))
