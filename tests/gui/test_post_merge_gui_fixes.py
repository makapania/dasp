"""Post-merge review fixes in the GUI.

* Ensemble reconstruction (``_reconstruct_models_from_results``) filtered every
  ``model__*`` key out of a row's Params, so models reconstructed from Bayesian rows
  (which store estimator params under ``model__``) trained with default
  hyperparameters. PLS ``n_components`` above 10 was also clipped by ``get_model``.
* The same function built the PLS-DA head with ``random_state=42`` and no
  ``class_weight``, ignoring the row's ``lr__random_state`` / ``lr__class_weight``.
* ``logger`` was undefined in the GUI module: Tab 7 refit raised NameError when the task
  radio disagreed with the saved result's Task.
* The learning-curve error callback referenced the except-bound ``e`` from a deferred
  lambda, which raises NameError once the except block has ended.

Search-time reference models and rows come from ``tests/test_t51_supervised_bundles.py``
(``round_trip_case``), whose real-run test proves they match real Bayesian rows.
"""

from __future__ import annotations

import contextlib
import io
import logging
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spectral_predict import unified_bayesian as ub
from spectral_predict.models import PLSTransformer, estimator_params_from_row, get_model
from tests.test_t51_supervised_bundles import (
    ROUND_TRIPS,
    _round_trip_data,
    assert_estimator_carries,
    predictions,
    round_trip_case,
)

pytestmark = pytest.mark.gui


def _reconstruct(app, row: dict, X: pd.DataFrame, y: np.ndarray, task: str):
    top_models_df = pd.DataFrame(
        [{"Poly": 2, "Deriv": 0, "Window": 17, "Preprocess": "raw", **row}]
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        reconstructed = app._reconstruct_models_from_results(top_models_df, X, y, task)
    assert len(reconstructed) == 1, buf.getvalue()[-4000:]
    return reconstructed[0][0]


def _assert_row_params_applied(fitted, row_params: dict) -> None:
    """Every estimator param the row stores reaches the reconstructed estimator."""
    inner = fitted.pipeline  # the GUI preprocessing wrapper holds the model pipeline
    if hasattr(inner, "named_steps") and "lr" in inner.named_steps:  # PLS-DA
        params = inner.get_params(deep=True)
        expected = {k: v for k, v in row_params.items() if k.startswith(("pls__", "lr__"))}
    else:
        final = inner.steps[-1][1] if hasattr(inner, "steps") else inner
        params = final.get_params()
        expected = estimator_params_from_row(row_params)
    assert expected
    for key, value in expected.items():
        assert params[key] == value, (key, params.get(key), value)


@pytest.mark.parametrize("name", sorted(ROUND_TRIPS))
def test_ensemble_reconstruction_matches_bayesian_search_model(gui_app, name):
    case = round_trip_case(name)
    X = case["X"]

    fitted = _reconstruct(gui_app, case["row"], X, case["y"], case["task"])

    assert_estimator_carries(fitted.pipeline, case)
    _assert_row_params_applied(fitted, case["row_params"])
    np.testing.assert_allclose(
        predictions(fitted, case["X_test"], case["task"]),
        predictions(case["reference"], case["X_test"], case["task"]),
        rtol=1e-6,
        atol=1e-8,
    )


def _captured_case(reference: Pipeline, model: str, data_task: str) -> dict:
    X, y, X_test = _round_trip_data(data_task)
    row_params = ub._capture_serializable_params(reference)
    reference.fit(X.values, y)
    return {
        "X": X,
        "y": y,
        "X_test": X_test,
        "row_params": row_params,
        "reference": reference,
        "row": {"Model": model, "Params": str(row_params)},
    }


EXTRA_CASES = {
    "Ridge": lambda: _captured_case(
        Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=37.0, random_state=42))]),
        "Ridge",
        "regression",
    ),
    "PLS-12-components": lambda: _captured_case(
        Pipeline([("model", PLSRegression(n_components=12, scale=False))]), "PLS", "regression"
    ),
}


@pytest.mark.parametrize("name", sorted(EXTRA_CASES))
def test_ensemble_reconstruction_matches_bayesian_linear_models(gui_app, name):
    case = EXTRA_CASES[name]()

    fitted = _reconstruct(gui_app, case["row"], case["X"], case["y"], "regression")

    _assert_row_params_applied(fitted, case["row_params"])
    np.testing.assert_allclose(
        np.ravel(fitted.predict(case["X_test"])),
        np.ravel(case["reference"].predict(case["X_test"])),
        rtol=1e-6,
        atol=1e-8,
    )


def test_ensemble_reconstruction_grid_row_with_bare_params_still_applies(gui_app):
    X, y, X_test = _round_trip_data("regression")
    grid_params = {"n_estimators": 30, "max_features": 0.3, "max_depth": 5, "random_state": 42}
    reference = get_model("RandomForest", "regression").set_params(**grid_params).fit(X.values, y)

    fitted = _reconstruct(
        gui_app, {"Model": "RandomForest", "Params": str(grid_params)}, X, y, "regression"
    )

    model = (
        fitted.pipeline.named_steps["model"]
        if hasattr(fitted.pipeline, "named_steps")
        else fitted.pipeline
    )
    for key, value in grid_params.items():
        assert model.get_params()[key] == value, key
    np.testing.assert_allclose(fitted.predict(X_test), reference.predict(X_test), rtol=1e-6)


def test_ensemble_reconstruction_plsda_head_keeps_seed_and_class_weight(gui_app):
    X, _, X_test = _round_trip_data("binary")
    rng = np.random.default_rng(12)
    signal = X.values[:, 0] - 0.7 * X.values[:, 3] + 0.8 * rng.standard_normal(len(X))
    y = (signal > np.quantile(signal, 0.75)).astype(int)  # imbalanced 3:1

    def head(random_state, class_weight):
        return Pipeline(
            [
                ("pls", PLSTransformer(n_components=3, scale=False)),
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        C=0.5,
                        solver="saga",
                        max_iter=30,
                        random_state=random_state,
                        class_weight=class_weight,
                    ),
                ),
            ]
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # saga with few iterations does not converge
        reference = head(7, "balanced")
        row_params = ub._capture_serializable_params(reference)
        reference.fit(X.values, y)
        seed_42 = head(42, "balanced").fit(X.values, y)
        ref_proba = reference.predict_proba(X_test)
        assert not np.allclose(ref_proba, seed_42.predict_proba(X_test), rtol=1e-6, atol=1e-8)
        assert row_params["lr__random_state"] == 7
        assert row_params["lr__class_weight"] == "balanced"

        fitted = _reconstruct(
            gui_app, {"Model": "PLS-DA", "Params": str(row_params)}, X, y, "classification"
        )

    lr = fitted.pipeline.named_steps["lr"]
    assert lr.random_state == 7
    assert lr.class_weight == "balanced"
    np.testing.assert_allclose(fitted.predict_proba(X_test), ref_proba, rtol=1e-6, atol=1e-8)


# --- NameErrors -----------------------------------------------------------------------


def test_tab7_refit_task_mismatch_warns_instead_of_name_error(gui_app, caplog):
    """Radio says classification, saved result says regression: warn and use the saved Task."""
    X, y, _ = _round_trip_data("regression")
    X_df = X.copy()
    X_df.index = [f"s{i}" for i in range(len(X_df))]
    app = gui_app
    app.X_original = X_df
    app.X = X_df
    app.y = pd.Series(y, index=X_df.index)
    app.active_indices = None
    app.excluded_spectra = set()
    app.validation_enabled.set(False)
    app.validation_indices = []
    app.use_autoscale.set(False)
    app.selected_model_config = {
        "Model": "PLS",
        "Task": "regression",
        "Params": str({"n_components": 3}),
        "LVs": 3,
        "Preprocess": "raw",
        "Deriv": 0,
        "Window": 17,
    }
    app._original_wavelength_order = [float(c) for c in X_df.columns]
    app.refine_task_type.set("classification")
    app.refine_model_type.set("PLS")
    app.refine_preprocess.set("raw")
    app.refine_folds.set(3)
    app.refine_cv_strategy.set("kfold")
    app.model_loaded_from_results = True
    app.refine_hyperparams_modified = False
    app.refined_model = None

    buf = io.StringIO()
    with caplog.at_level(logging.WARNING), contextlib.redirect_stdout(buf):
        app._run_refined_model_thread()
    app.root.update()

    assert "NameError" not in buf.getvalue(), buf.getvalue()[-4000:]
    assert app.refined_model is not None, buf.getvalue()[-4000:]
    assert any("using saved Task" in r.getMessage() for r in caplog.records)
    assert app.refine_task_type.get() == "regression"


def test_learning_curve_error_callback_receives_message(gui_app, monkeypatch):
    app = gui_app
    received = []
    scheduled = []
    monkeypatch.setattr(app, "_on_learning_curve_error", received.append)
    monkeypatch.setattr(app.root, "after", lambda ms, func=None, *args: scheduled.append(func))
    monkeypatch.setattr(app, "refined_config", None, raising=False)  # .get() raises

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        app._run_learning_curve_thread()
    assert len(scheduled) == 1
    scheduled[0]()  # runs after the except block ended, like Tk's event loop

    assert len(received) == 1
    assert isinstance(received[0], str) and received[0]
