"""Tab 7 (Model Development) refit of a booster row: one round count from the pooled CV curve.

Drives ``_run_refined_model_thread`` synchronously. Grid search stores the CV-selected
round count in Params (R028/R003). Tab 7 re-runs the same selection with that count as
the maximum, which returns the same count (the selection is idempotent), so the refit
reproduces the row's R2cv and the saved model carries the selected count. No fold is
ever fitted with an eval_set.
"""

from __future__ import annotations

import ast
import contextlib
import io
import traceback

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.gui


def _data():
    rng = np.random.default_rng(3)
    n, p = 45, 120
    X = rng.normal(size=(n, p))
    y = X[:, 5] - 0.5 * X[:, 40] + 0.3 * rng.normal(size=n)
    X_df = pd.DataFrame(X, columns=np.linspace(1000.0, 1238.0, p))
    X_df.index = [f"s{i}" for i in range(n)]
    return X_df, pd.Series(y, index=X_df.index)


def _grid_row(X_df, y):
    from spectral_predict.search import run_search

    df, _ = run_search(
        X_df,
        y,
        "regression",
        folds=3,
        tier="quick",
        models_to_test=["LightGBM"],
        enabled_models=["LightGBM"],
        preprocessing_methods={"raw": True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        lightgbm_n_estimators_list=[80],
        lightgbm_learning_rates=[0.2],
        lightgbm_num_leaves_list=[7],
        early_stopping_rounds=10,
    )
    return df.iloc[0].to_dict()


def _refit(app, X_df, y, row):
    app.X_original = X_df
    app.X = X_df
    app.y = y
    app.active_indices = None
    app.excluded_spectra = set()
    app.validation_enabled.set(False)
    app.validation_indices = []
    app.use_autoscale.set(False)
    app.selected_model_config = dict(row)
    app._original_wavelength_order = [float(c) for c in X_df.columns]
    app.refine_task_type.set("regression")
    app.refine_model_type.set("LightGBM")
    app.refine_preprocess.set("raw")
    app.refine_folds.set(3)
    app.refine_cv_strategy.set("kfold")
    app.model_loaded_from_results = True
    app.refine_hyperparams_modified = False
    app.refined_model = None

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        app._run_refined_model_thread()
    app.root.update()
    out = buf.getvalue()
    assert app.refined_model is not None, out[-4000:]
    return app.refined_model, out


def _n_estimators(model):
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    return est.get_params()["n_estimators"]


def test_tab7_refit_reproduces_grid_booster_row(gui_app):
    X_df, y = _data()
    row = _grid_row(X_df, y)
    k = int(row["n_estimators_selected"])
    assert ast.literal_eval(row["Params"])["n_estimators"] == k

    model, out = _refit(gui_app, X_df, y, row)

    # Same maximum as the grid row (n_estimators_fit), same count; the final model is
    # the full fit truncated to it.
    assert _n_estimators(model) == k
    assert f"boosting rounds: {k} of {int(row['n_estimators_fit'])}" in out
    assert gui_app.refined_performance["r2_mean"] == pytest.approx(row["R2cv"], abs=1e-9)


def test_tab7_refit_of_old_row_selects_rounds_without_test_fold(gui_app):
    """A pre-fix row (Params hold the maximum) gets the honest pooled selection."""
    from lightgbm import LGBMRegressor

    from spectral_predict.cv_utils import build_cv_splitter, cross_val_boosting_rounds

    X_df, y = _data()
    row = _grid_row(X_df, y)
    params = ast.literal_eval(row["Params"])
    params["n_estimators"] = 80
    row["Params"] = str(params)
    row["n_estimators_selected"] = None

    model, _ = _refit(gui_app, X_df, y, row)

    expected = cross_val_boosting_rounds(
        LGBMRegressor(**params),
        X_df.to_numpy(),
        y.to_numpy(),
        build_cv_splitter("kfold", 3, "regression", random_state=42),
        patience=10,
    ).n_rounds
    assert _n_estimators(model) == expected


def test_tab7_fits_each_booster_fold_once(gui_app, monkeypatch):
    """Tab 7 reports the round-selection fits' predictions: one fit per fold plus the
    final fit, not a second fit of every fold."""
    from lightgbm import LGBMRegressor

    X_df, y = _data()
    row = _grid_row(X_df, y)
    calls = []
    real_fit = LGBMRegressor.fit

    def counting_fit(self, *args, **kwargs):
        # The validation-curve diagnostic refits the model on purpose; count only
        # the CV and final fits.
        names = {frame.name for frame in traceback.extract_stack()}
        if "_compute_validation_curve" not in names:
            calls.append(1)
        return real_fit(self, *args, **kwargs)

    monkeypatch.setattr(LGBMRegressor, "fit", counting_fit)
    _refit(gui_app, X_df, y, row)
    assert len(calls) == 3 + 1


def test_tab7_rejected_selection_with_y_transform_does_not_crash(gui_app):
    """Round 2 #5 / GLM: DART (selection rejected) plus a target transform wraps the
    pipeline in a TransformedTargetRegressor; Tab 7 must not reach into .steps."""
    X_df, y = _data()
    y = y - y.min() + 1.0  # positive for Log
    row = {
        "Model": "LightGBM",
        "Task": "regression",
        "Preprocess": "raw",
        "Deriv": 0,
        "Window": 17,
        "LVs": None,
        "early_stopping_rounds": 10,
        "Params": str(
            {"n_estimators": 20, "boosting_type": "dart", "verbosity": -1, "random_state": 0}
        ),
    }
    gui_app.refine_y_transform.set("Log")
    try:
        model, out = _refit(gui_app, X_df, y, row)
    finally:
        gui_app.refine_y_transform.set("None")
    assert "selection skipped" in out


def test_tab7_log_transform_booster_saves_the_cv_model_in_original_units(gui_app, tmp_path):
    """Round 4 #2: with a log Y-transform and round selection, the saved model is the
    transformed-scale booster fitted at the maximum round count and truncated to the
    selected count (what the CV curve was read from), wrapped so predict_with_model
    returns original units."""
    from unittest.mock import patch

    from sklearn.compose import TransformedTargetRegressor
    from xgboost import XGBRegressor

    from spectral_predict.cv_utils import booster_predict_at
    from spectral_predict.model_io import load_model, predict_with_model

    X_df, y = _data()
    y = y - y.min() + 1.0  # positive for log
    row = {
        "Model": "XGBoost",
        "Task": "regression",
        "Preprocess": "raw",
        "Deriv": 0,
        "Window": 17,
        "LVs": None,
        "early_stopping_rounds": 10,
        "Params": str(
            {
                "n_estimators": 40,
                "learning_rate": 0.2,
                "max_depth": 3,
                "random_state": 0,
                "n_jobs": 1,
            }
        ),
    }
    gui_app.refine_model_type.set("XGBoost")
    gui_app.refine_y_transform.set("Log")
    try:
        app = gui_app
        app.X_original = X_df
        app.X = X_df
        app.y = y
        app.active_indices = None
        app.excluded_spectra = set()
        app.validation_enabled.set(False)
        app.validation_indices = []
        app.use_autoscale.set(False)
        app.selected_model_config = dict(row)
        app._original_wavelength_order = [float(c) for c in X_df.columns]
        app.refine_task_type.set("regression")
        app.refine_preprocess.set("raw")
        app.refine_folds.set(3)
        app.refine_cv_strategy.set("kfold")
        app.model_loaded_from_results = True
        app.refine_hyperparams_modified = False
        app.refined_model = None
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            app._run_refined_model_thread()
        app.root.update()
        assert app.refined_model is not None, buf.getvalue()[-4000:]
    finally:
        gui_app.refine_y_transform.set("None")

    saved = gui_app.refined_model
    assert isinstance(saved, TransformedTargetRegressor)
    k = gui_app.refined_config["n_estimators_selected"]
    assert gui_app.refined_config["n_estimators_fit"] == 40 and 1 <= k <= 40
    booster = saved.regressor_
    booster = booster.steps[-1][1] if hasattr(booster, "steps") else booster
    assert booster.get_booster().num_boosted_rounds() == k

    # The CV curve's model at k, on the transformed scale: XGBoost fitted on log(y) at
    # the maximum round count, read at round k.
    params = dict(booster.get_params())
    params["n_estimators"] = 40
    X = X_df.to_numpy()
    reference = XGBRegressor(**params).fit(X, np.log(y.to_numpy()))
    on_log_scale = booster_predict_at(reference, X, k)
    np.testing.assert_allclose(np.log(saved.predict(X)), on_log_scale, rtol=1e-6, atol=1e-6)

    path = tmp_path / "yt_booster.dasp"
    with (
        patch("tkinter.filedialog.asksaveasfilename", return_value=str(path)),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        gui_app._save_refined_model()
    loaded = load_model(path)
    pred = np.asarray(predict_with_model(loaded, X_df), dtype=float).ravel()
    np.testing.assert_allclose(pred, np.exp(on_log_scale), rtol=1e-6, atol=1e-6)
