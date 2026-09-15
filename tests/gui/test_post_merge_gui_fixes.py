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
from spectral_predict.models import (
    PLSTransformer,
    build_model,
    estimator_params_from_row,
    get_model,
)
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


def _inner(fitted):
    """The model pipeline, unwrapping a GUI preprocessing wrapper if there is one."""
    return getattr(fitted, "pipeline", fitted)


def _assert_row_params_applied(fitted, row_params: dict) -> None:
    """Every estimator param the row stores reaches the reconstructed estimator."""
    inner = _inner(fitted)
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

    assert_estimator_carries(_inner(fitted), case)
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

    inner = _inner(fitted)
    model = inner.named_steps["model"] if hasattr(inner, "named_steps") else inner
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

    lr = _inner(fitted).named_steps["lr"]
    assert lr.random_state == 7
    assert lr.class_weight == "balanced"
    np.testing.assert_allclose(fitted.predict_proba(X_test), ref_proba, rtol=1e-6, atol=1e-8)


# --- Row preprocessing: Autoscale, derivatives, baseline, smoothing, subsets ----------

WL = [1000 + 2 * i for i in range(30)]


def _bayesian_preprocessed(X: np.ndarray, config: dict, baseline_method=None) -> np.ndarray:
    """The matrix the Bayesian objective fits on (``apply_preprocessing``)."""
    return ub.apply_preprocessing(
        X, {"deriv": 0, "window": 0, "polyorder": 0, **config}, baseline_method=baseline_method
    )


PREP_CASES = {
    # name: (model, estimator pipeline, preprocessing config, baseline, row preprocessing cols)
    "PLS-autoscale": (
        "PLS",
        lambda: Pipeline([("model", PLSRegression(n_components=4, scale=False))]),
        {"name": "raw", "apply_autoscale": True},
        None,
        {"Preprocess": "raw+autoscale", "PreprocessBase": "raw", "Autoscale": True},
    ),
    "SVR-autoscale": (
        "SVR",
        lambda: Pipeline([("model", build_model("SVR", {"C": 5.0, "gamma": 0.01}))]),
        {"name": "raw", "apply_autoscale": True},
        None,
        {"Preprocess": "raw+autoscale", "PreprocessBase": "raw", "Autoscale": True},
    ),
    "MLP-autoscale": (
        "MLP",
        lambda: Pipeline(
            [
                (
                    "model",
                    build_model(
                        "MLP", {"hidden_layer_sizes": (16,), "alpha": 1e-3, "max_iter": 200}
                    ),
                )
            ]
        ),
        {"name": "snv", "apply_autoscale": True},
        None,
        {"Preprocess": "snv+autoscale", "PreprocessBase": "snv", "Autoscale": True},
    ),
    "PLS-deriv": (
        "PLS",
        lambda: Pipeline([("model", PLSRegression(n_components=4, scale=False))]),
        {"name": "deriv1", "deriv": 1, "window": 11, "polyorder": 2},
        None,
        {"Preprocess": "deriv", "PreprocessBase": "deriv", "Deriv": 1, "Window": 11, "Poly": 2},
    ),
    "PLS-baseline-smoothing-autoscale": (
        "PLS",
        lambda: Pipeline([("model", PLSRegression(n_components=4, scale=False))]),
        {
            "name": "snv_deriv1",
            "deriv": 1,
            "window": 11,
            "polyorder": 2,
            "apply_baseline": True,
            "apply_smoothing": True,
            "apply_autoscale": True,
        },
        "polynomial",
        {
            "Preprocess": "polynomial+sg0+snv_deriv+autoscale",
            "PreprocessBase": "snv_deriv",
            "Deriv": 1,
            "Window": 11,
            "Poly": 2,
            "Autoscale": True,
            "baseline_method": "polynomial",
            "smoothing": True,
            "smoothing_window": 17,
            "smoothing_polyorder": 2,
        },
    ),
}


@pytest.mark.parametrize("name", sorted(PREP_CASES))
def test_ensemble_reconstruction_honours_row_preprocessing(gui_app, name):
    model_name, make_reference, config, baseline, prep_cols = PREP_CASES[name]
    X, y, X_test = _round_trip_data("regression")
    reference = make_reference()
    row_params = ub._capture_serializable_params(reference)
    # Train and test are preprocessed together so the stateful autoscale step sees the
    # training rows only: fit it on train, then apply it to test.
    X_all = _bayesian_preprocessed(
        np.vstack([X.values, X_test]), {**config, "apply_autoscale": False}, baseline
    )
    X_tr, X_te = X_all[: len(X)], X_all[len(X) :]
    if config.get("apply_autoscale"):
        scaler = StandardScaler().fit(X_tr)
        X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
    reference.fit(X_tr, y)

    fitted = _reconstruct(
        gui_app, {"Model": model_name, "Params": str(row_params), **prep_cols}, X, y, "regression"
    )

    _assert_row_params_applied(fitted, row_params)
    inner = _inner(fitted)
    names = [n for n, _ in inner.steps] if hasattr(inner, "steps") else []
    assert "scaler" not in names or not config.get("apply_autoscale"), names
    np.testing.assert_allclose(
        np.ravel(fitted.predict(X_test)), np.ravel(reference.predict(X_te)), rtol=1e-6, atol=1e-8
    )


def test_ensemble_reconstruction_subsets_after_preprocessing(gui_app):
    X, y, X_test = _round_trip_data("regression")
    subset = [2, 3, 5, 8, 13, 21, 22, 23]
    all_vars = ", ".join(f"{WL[i]:.1f}" for i in subset)
    reference = Pipeline([("model", PLSRegression(n_components=3, scale=False))])
    row_params = ub._capture_serializable_params(reference)
    prep = {"name": "snv_deriv1", "deriv": 1, "window": 11, "polyorder": 2}
    reference.fit(_bayesian_preprocessed(X.values, prep)[:, subset], y)

    fitted = _reconstruct(
        gui_app,
        {
            "Model": "PLS",
            "Params": str(row_params),
            "Preprocess": "snv_deriv",
            "Deriv": 1,
            "Window": 11,
            "Poly": 2,
            "all_vars": all_vars,
        },
        X,
        y,
        "regression",
    )

    np.testing.assert_allclose(
        np.ravel(fitted.predict(X_test)),
        np.ravel(reference.predict(_bayesian_preprocessed(X_test, prep)[:, subset])),
        rtol=1e-6,
        atol=1e-8,
    )


def test_ensemble_reconstruction_accepts_dict_params_cell(gui_app):
    """In-memory result rows can hold Params as a dict rather than str(dict)."""
    X, y, X_test = _round_trip_data("regression")
    params = {"n_estimators": 25, "max_features": 0.3, "max_depth": 4, "random_state": 42}
    reference = get_model("RandomForest", "regression").set_params(**params).fit(X.values, y)
    top_models_df = pd.DataFrame(
        [
            {
                "Model": "RandomForest",
                "Params": None,
                "Preprocess": "raw",
                "Deriv": None,
                "Window": None,
                "Poly": None,
            }
        ]
    )
    top_models_df.at[0, "Params"] = params  # a dict cell, as in-memory rows can hold

    with contextlib.redirect_stdout(io.StringIO()):
        reconstructed = gui_app._reconstruct_models_from_results(top_models_df, X, y, "regression")

    fitted = reconstructed[0][0]
    inner = _inner(fitted)
    model = inner.named_steps["model"] if hasattr(inner, "named_steps") else inner
    assert model.n_estimators == 25 and model.max_features == 0.3
    np.testing.assert_allclose(fitted.predict(X_test), reference.predict(X_test), rtol=1e-6)


def test_ensemble_refit_reaches_catboost_inside_gui_wrapper(tmp_path, monkeypatch):
    """GUI wrappers return shallow get_params(deep=True); the refit must still reach CatBoost."""
    catboost = pytest.importorskip("catboost")
    from spectral_predict.ensemble import RegionAwareWeightedEnsemble
    from spectral_predict_gui_optimized import WavelengthSubsetWrapper

    X, y, _ = _round_trip_data("regression")
    cols = list(X.columns[:12])
    fit_dir = tmp_path / "fit"
    fit_dir.mkdir()
    monkeypatch.chdir(fit_dir)
    legacy = catboost.CatBoostRegressor(iterations=10, depth=2, random_state=0, verbose=False)
    wrapped = WavelengthSubsetWrapper(
        Pipeline([("scaler", StandardScaler()), ("model", legacy)]), cols
    )
    wrapped.fit(X, y)

    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "catboost_info").write_text("not a directory", encoding="utf-8")
    monkeypatch.chdir(blocked)

    ensemble = RegionAwareWeightedEnsemble(
        models=[wrapped, Ridge().fit(X, y)], model_names=["CatBoost", "Ridge"], n_regions=2, cv=3
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*failed during OOF prediction.*")
        ensemble.fit(X, y)

    assert np.all(np.isfinite(ensemble.predict(X)))
    assert (blocked / "catboost_info").is_file()


# --- Chromosome rows and missing derivative windows: parity with validation rebuild ----


def _validation_rmsep(row: dict, X: pd.DataFrame, y: np.ndarray, X_val: np.ndarray, y_val):
    """RMSEP from the public validation rebuild for a single results row."""
    from spectral_predict.search import compute_validation_metrics_for_top_models

    df = pd.DataFrame([{"CompositeScore": 0.0, "Task": "regression", **row}])
    with contextlib.redirect_stdout(io.StringIO()):
        out = compute_validation_metrics_for_top_models(
            df, X.values, y, X_val, y_val, "regression", np.array(WL, dtype=float), top_n=1
        )
    return float(out.loc[0, "RMSEP"])


def _val_split(dtype=np.float64):
    X, y, _ = _round_trip_data("regression")
    rng = np.random.default_rng(5)
    X_val = rng.standard_normal((20, X.shape[1]))
    y_val = X_val[:, 0] - 0.7 * X_val[:, 3] + 0.4 * X_val[:, 7]
    if dtype == np.float32:
        # Count-like float32 spectra around 1e6, as the SPC reader returns: float32
        # rounding changes derivatives unless the rebuild converts to float64 first.
        X = (1e6 + 1e3 * X.cumsum(axis=1)).astype(np.float32)
        X_val = (1e6 + 1e3 * np.cumsum(X_val, axis=1)).astype(np.float32)
    return X, y, X_val, y_val


CHROMOSOME_ROWS = {
    # Exhaustive-search row shape (search.py _build_preprocessing_configs): the base name
    # carries the derivative order and window, which build_preprocessing_pipeline rejects.
    "exhaustive-snv_deriv1_w11": {
        "Model": "PLS",
        "Params": str({"n_components": 4, "scale": False}),
        "LVs": 4,
        "Preprocess": "snv_deriv",
        "PreprocessBase": "snv_deriv1_w11",
        "preprocess_chromosome": "[6, 3, 0]",
        "Deriv": 1,
        "Window": 11,
        "Poly": 2,
        "Autoscale": False,
    },
    # 3-gene chromosome with autoscale on: no per-model scaler for the SVR.
    "exhaustive-autoscale-svr": {
        "Model": "SVR",
        "Params": str({"C": 5.0, "gamma": 0.01}),
        "Preprocess": "deriv+autoscale",
        "PreprocessBase": "deriv1_w13",
        "preprocess_chromosome": "[2, 4, 1]",
        "Deriv": 1,
        "Window": 13,
        "Poly": 2,
        "Autoscale": True,
    },
    # A chromosome row whose Preprocess is not caught by GA_SUFFIXES or any known name.
    "chromosome-unrecognised-name": {
        "Model": "Ridge",
        "Params": str({"alpha": 3.0}),
        "Preprocess": "deriv2_w9",
        "preprocess_chromosome": "[3, 2]",
    },
}


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("name", sorted(CHROMOSOME_ROWS))
def test_ensemble_reconstruction_decodes_preprocess_chromosome(gui_app, name, dtype):
    row = CHROMOSOME_ROWS[name]
    X, y, X_val, y_val = _val_split(dtype)

    fitted = _reconstruct(gui_app, row, X, y, "regression")
    rmse = float(np.sqrt(np.mean((np.ravel(fitted.predict(X_val)) - y_val) ** 2)))

    inner = _inner(fitted)
    names = [n for n, _ in inner.steps]
    if row.get("Autoscale"):
        assert "autoscale" in names and "scaler" not in names, names
    np.testing.assert_allclose(rmse, _validation_rmsep(row, X, y, X_val, y_val), rtol=1e-9)

    # Clonable for per-fold refits.
    from sklearn.base import clone

    refit = clone(fitted).fit(X, y)
    np.testing.assert_allclose(refit.predict(X_val), fitted.predict(X_val), rtol=1e-9)
    refit_rmse = float(np.sqrt(np.mean((np.ravel(refit.predict(X_val)) - y_val) ** 2)))
    np.testing.assert_allclose(refit_rmse, _validation_rmsep(row, X, y, X_val, y_val), rtol=1e-9)


def test_ensemble_reconstruction_defaults_missing_derivative_window(gui_app):
    """A snv_deriv row with Window=NaN rebuilt with window 15 before ae15e64; keep that."""
    row = {
        "Model": "Ridge",
        "Params": str({"alpha": 2.0}),
        "Preprocess": "snv_deriv",
        "Deriv": 1,
        "Window": np.nan,
        "Poly": 2,
    }
    X, y, X_val, y_val = _val_split()

    fitted = _reconstruct(gui_app, row, X, y, "regression")

    savgol = _inner(fitted).named_steps["savgol"]
    assert (savgol.deriv, savgol.window, savgol.polyorder) == (1, 15, 2)
    rmse = float(np.sqrt(np.mean((np.ravel(fitted.predict(X_val)) - y_val) ** 2)))
    np.testing.assert_allclose(rmse, _validation_rmsep(row, X, y, X_val, y_val), rtol=1e-9)


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
