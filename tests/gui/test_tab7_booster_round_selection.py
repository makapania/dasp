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

    assert _n_estimators(model) == k
    assert f"boosting rounds: {k} of {k}" in out
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
