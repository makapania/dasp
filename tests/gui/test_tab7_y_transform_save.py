"""Tab 7 Y-transform refit -> save -> load -> predict parity (R048, R001, R020, R014/R019).

Drives ``_run_refined_model_thread`` synchronously on the session Tk app, then
``_save_refined_model`` to a temp .dasp, reloads it and predicts on the raw spectra.

Before the fix:
- every non-early-stopping Y-transform refit crashed on ``pipe.steps`` of a
  TransformedTargetRegressor (R048), and 'Box-Cox' was rejected by ``wrap()``;
- had it not crashed, the TTR save branch dropped the full-spectrum preprocessor
  (R001) and saved the inner scaler a second time as the preprocessor (R020);
- with early stopping, CV transformed fold targets but the saved model was a raw-y
  fit labelled with the transform (R014/R019), and the label came from the live
  widget at save time.

Each case checks that the loaded .dasp reproduces the in-memory refit model's
predictions and its calibration RMSE (which ``_run_refined_model_thread`` computes
directly from the fitted training pipeline).
"""

from __future__ import annotations

import contextlib
import io
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spectral_predict.model_io import load_model, predict_with_model
from spectral_predict.y_transform import YTransformWrapper, normalize_y_transform_method

pytestmark = pytest.mark.gui

TRANSFORMS = ["Log", "Log1p", "Sqrt", "Box-Cox", "Yeo-Johnson"]
N_SAMPLES = 36


@pytest.fixture(autouse=True)
def _reset_y_transform_widget(gui_app):
    """The session app is shared: never leak a selected transform into later tests."""
    yield
    gui_app.refine_y_transform.set("None")


N_WL = 60


def _spectra(seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    wl = np.linspace(1100.0, 1690.0, N_WL)
    conc = rng.uniform(0.5, 3.0, N_SAMPLES)
    band = np.exp(-0.5 * ((wl - 1400.0) / 40.0) ** 2)
    base = 0.4 + 0.1 * np.sin(wl / 90.0)
    X = base + np.outer(conc, band) * 0.3 + rng.normal(0, 0.004, (N_SAMPLES, N_WL))
    # Strictly positive, right-skewed target so every transform is valid.
    y = np.exp(0.6 * conc) + rng.normal(0, 0.05, N_SAMPLES)
    ids = [f"s{i}" for i in range(N_SAMPLES)]
    X_df = pd.DataFrame(X, columns=[f"{w:.1f}" for w in wl], index=ids)
    return X_df, pd.Series(y, index=ids, name="target")


def _subset_wavelengths(X_df: pd.DataFrame) -> list[float]:
    all_wl = [float(c) for c in X_df.columns]
    # Non-contiguous subset away from the edges: derivative context must come from
    # the full spectrum, so a missing full-spectrum preprocessor changes predictions.
    return all_wl[10:25] + all_wl[35:45]


def _row(model: str, early_stopping: bool) -> dict:
    if model == "PLS":
        params, extra = {"n_components": 3}, {"LVs": 3}
    elif model == "Ridge":
        params, extra = {"alpha": 0.01}, {}
    elif model == "XGBoost":
        # Full row (as a real Results row carries): Tab 7 fills unspecified XGBoost
        # params with GUI defaults that the code export does not know about.
        params = {
            "n_estimators": 40,
            "max_depth": 2,
            "learning_rate": 0.2,
            "random_state": 0,
            "subsample": 0.8,
            "colsample_bytree": 0.6,
            "reg_lambda": 1.5,
            "reg_alpha": 0.2,
            "min_child_weight": 1,
            "gamma": 0.0,
        }
        extra = {}
    else:  # pragma: no cover - guard against typos in parametrisation
        raise ValueError(model)
    row = {
        "Model": model,
        "Task": "regression",
        "Params": str(params),
        "Preprocess": "deriv",
        "Deriv": 1,
        "Window": 7,
        "Poly": 2,
        **extra,
    }
    if early_stopping:
        row["early_stopping_rounds"] = 5
    return row


def _refit(app, model: str, y_transform: str, subset: bool, early_stopping: bool = False):
    X_df, y = _spectra()
    app.X_original = X_df
    app.X = X_df
    app.y = y
    app.active_indices = None
    app.excluded_spectra = set()
    app.validation_enabled.set(False)
    app.validation_indices = []
    app.use_autoscale.set(False)
    app.selected_model_config = _row(model, early_stopping)
    wl = _subset_wavelengths(X_df) if subset else [float(c) for c in X_df.columns]
    app._original_wavelength_order = wl
    app.refine_task_type.set("regression")
    app.refine_model_type.set(model)
    app.refine_preprocess.set("sg1")
    app.refine_window.set(7)
    app.refine_window_custom.set("")
    app.refine_folds.set(3)
    app.refine_cv_strategy.set("kfold")
    app.refine_y_transform.set(y_transform)
    app.model_loaded_from_results = True
    app.refine_hyperparams_modified = False
    app.refined_model = None

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        app._run_refined_model_thread()
    app.root.update()
    assert app.refined_model is not None, buf.getvalue()[-4000:]
    return X_df, y


def _save_and_load(app, tmp_path, name: str = "m.dasp") -> dict:
    path = tmp_path / name
    with (
        patch("tkinter.filedialog.asksaveasfilename", return_value=str(path)),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        app._save_refined_model()
    assert path.exists(), "save did not write a .dasp"
    return load_model(path)


def _assert_roundtrip_parity(app, loaded: dict, X_df: pd.DataFrame, y: pd.Series):
    pred_loaded = np.asarray(predict_with_model(loaded, X_df), dtype=float).ravel()
    pred_mem = np.asarray(app.refined_model.predict(app.refined_X_train), dtype=float).ravel()
    np.testing.assert_allclose(pred_loaded, pred_mem, rtol=1e-9, atol=1e-10)
    # Independent anchor: cal_rmse was computed from the fitted training pipeline.
    rmse = float(np.sqrt(np.mean((y.values - pred_loaded) ** 2)))
    assert rmse == pytest.approx(app.refined_performance["cal_rmse"], rel=1e-9)
    return pred_loaded


@pytest.mark.parametrize("subset", [False, True], ids=["full", "subset"])
@pytest.mark.parametrize("model", ["PLS", "Ridge"])
@pytest.mark.parametrize("y_transform", TRANSFORMS)
def test_ytransform_refit_save_load_predict_parity(gui_app, tmp_path, y_transform, model, subset):
    X_df, y = _refit(gui_app, model, y_transform, subset)
    assert isinstance(gui_app.refined_model, TransformedTargetRegressor)

    # The widget moving after training must not change what the file claims (R014).
    gui_app.refine_y_transform.set("None")
    loaded = _save_and_load(gui_app, tmp_path)

    assert loaded["metadata"]["y_transform"] == normalize_y_transform_method(y_transform)
    saved_model = loaded["model"]
    assert isinstance(saved_model, TransformedTargetRegressor)

    # R001: the fitted full-spectrum spectral preprocessor is saved.
    prep = loaded["preprocessor"]
    assert prep is not None
    prep_steps = [type(s).__name__ for _, s in prep.steps]
    # R020: the per-subset scaler lives only inside the saved model, never in the
    # preprocessor (it would be applied twice, or to the full-width spectrum).
    assert "StandardScaler" not in prep_steps
    inner = saved_model.regressor_
    if model == "Ridge":
        assert isinstance(inner, Pipeline)
        assert isinstance(inner.named_steps["scaler"], StandardScaler)
    else:
        assert not isinstance(inner, Pipeline)

    _assert_roundtrip_parity(gui_app, loaded, X_df, y)


@pytest.mark.parametrize(
    "y_transform,subset",
    [(t, True) for t in TRANSFORMS] + [("Log", False)],
)
def test_ytransform_early_stopping_final_model_uses_transform(
    gui_app, tmp_path, y_transform, subset
):
    """R014/R019: the saved booster is fitted on the transformed full-calibration y."""
    X_df, y = _refit(gui_app, "XGBoost", y_transform, subset, early_stopping=True)
    model = gui_app.refined_model
    assert isinstance(model, TransformedTargetRegressor)

    # Rebuild the expected final model by hand: transform fitted on all calibration y,
    # booster fitted on transformed y, predictions inverse-transformed.
    X_work = gui_app.refined_X_train
    transformer = YTransformWrapper._get_transformer(y_transform)
    y_t = transformer.fit_transform(y.values.reshape(-1, 1)).ravel()
    booster = clone(model.regressor_).fit(X_work, y_t)
    expected = transformer.inverse_transform(booster.predict(X_work).reshape(-1, 1)).ravel()
    np.testing.assert_allclose(model.predict(X_work), expected, rtol=1e-5, atol=1e-6)

    gui_app.refine_y_transform.set("None")
    loaded = _save_and_load(gui_app, tmp_path)
    assert loaded["metadata"]["y_transform"] == normalize_y_transform_method(y_transform)
    _assert_roundtrip_parity(gui_app, loaded, X_df, y)


def test_no_transform_records_none_and_keeps_plain_model(gui_app, tmp_path):
    X_df, y = _refit(gui_app, "PLS", "None", subset=True)
    assert not isinstance(gui_app.refined_model, TransformedTargetRegressor)
    gui_app.refine_y_transform.set("Log")  # widget moved after training
    loaded = _save_and_load(gui_app, tmp_path)
    assert loaded["metadata"]["y_transform"] == "none"
    _assert_roundtrip_parity(gui_app, loaded, X_df, y)


# --- Review round 1: consumers of a Y-transformed refit -----------------------------


def _exec_export(app) -> dict:
    """Generate the embedded-data notebook for the current refit and execute it."""
    from spectral_predict.code_generator import CodeGenerator, ExportOptions

    cfg = app._build_export_model_config()
    opts = ExportOptions(
        format="notebook",
        include_data=True,
        data_X=app.refined_X_train,
        data_y=app.refined_y_train,
        wavelengths=app.refined_wavelengths,
        colab_ready=False,
        include_visualization=False,
    )
    ns: dict = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in CodeGenerator(cfg, opts).generate_notebook()["cells"]:
            code = "".join(cell["source"])
            if cell["cell_type"] != "code" or "subprocess.check_call" in code:
                continue
            exec(code, ns)
    return ns


@pytest.mark.parametrize(
    "model,y_transform,early_stopping",
    [("PLS", "Log", False), ("Ridge", "Box-Cox", False), ("XGBoost", "Log", True)],
)
def test_exported_refinement_preserves_y_transform(gui_app, model, y_transform, early_stopping):
    """Exported CV and final model reproduce the transformed in-app refit."""
    _refit(gui_app, model, y_transform, subset=True, early_stopping=early_stopping)
    cfg = gui_app._build_export_model_config()
    assert cfg["y_transform"] == normalize_y_transform_method(y_transform)

    ns = _exec_export(gui_app)

    inapp_cv = np.empty(len(gui_app.refined_y_pred))
    inapp_cv[gui_app.refined_cv_indices] = gui_app.refined_y_pred
    np.testing.assert_allclose(ns["all_y_pred_arr"], inapp_cv, rtol=1e-6, atol=1e-8)

    X_work = gui_app.refined_X_train
    np.testing.assert_allclose(
        ns["model"].predict(X_work), gui_app.refined_model.predict(X_work), rtol=1e-6, atol=1e-8
    )


@pytest.mark.parametrize("model,param", [("PLS", "n_components"), ("Ridge", "alpha")])
def test_complexity_curve_uses_frozen_y_transform(gui_app, model, param):
    """The Model Complexity curve is computed on the transformed-target model."""
    from sklearn.model_selection import validation_curve

    from spectral_predict.cv_utils import build_cv_splitter

    _refit(gui_app, model, "Log", subset=True)
    curve = gui_app.complexity_curve_data
    assert curve is not None and curve["param_name"] == param

    # Independent reference: clones of the SAVED transformed model, varied on the
    # same parameter and scored in original units.
    saved = gui_app.refined_model
    inner = saved.regressor
    name = f"regressor__model__{param}" if isinstance(inner, Pipeline) else f"regressor__{param}"
    cv = build_cv_splitter(
        strategy="kfold", n_folds=3, task_type="regression", n_repeats=5, random_state=42
    )
    X_work, y = gui_app.refined_X_train, gui_app.refined_y_train
    _, cv_raw = validation_curve(
        clone(saved),
        X_work,
        y,
        param_name=name,
        param_range=curve["param_values"],
        cv=cv,
        scoring="neg_root_mean_squared_error",
    )
    np.testing.assert_allclose(curve["cv_scores"], -cv_raw.mean(axis=1), rtol=1e-6)


@pytest.mark.parametrize(
    "model_name,estimator,param,base",
    [
        ("RandomForest", "rf", "n_estimators", 100),
        ("Ridge", "ridge", "alpha", 0.37),
    ],
)
def test_ttr_complexity_curve_includes_selected_value(gui_app, model_name, estimator, param, base):
    """Codex round 2: the fitted value is on the grid and is the one marked selected."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    y = np.exp(0.3 * X[:, 0]) + 0.5
    inner = (
        RandomForestRegressor(n_estimators=base, max_depth=3, random_state=0)
        if estimator == "rf"
        else Ridge(alpha=base)
    )
    ttr = YTransformWrapper.wrap(Pipeline([("model", inner)]), "Log")
    with contextlib.redirect_stdout(io.StringIO()):
        curve = gui_app._compute_wrapped_validation_curve(
            model_name, ttr, X, y, KFold(3, shuffle=True, random_state=0), "regression"
        )
    assert curve["param_name"] == param
    assert curve["param_values"][curve["selected_idx"]] == base


def test_classification_ignores_y_transform_widget(gui_app, tmp_path):
    X_df, y = _spectra()
    labels = pd.Series(np.where(y.values > np.median(y.values), "hi", "lo"), index=y.index)
    gui_app.X_original = X_df
    gui_app.X = X_df
    gui_app.y = labels
    gui_app.active_indices = None
    gui_app.excluded_spectra = set()
    gui_app.validation_enabled.set(False)
    gui_app.validation_indices = []
    gui_app.use_autoscale.set(False)
    gui_app.selected_model_config = {
        "Model": "PLS-DA",
        "Task": "classification",
        "Params": str({"n_components": 2}),
        "LVs": 2,
        "Preprocess": "raw",
        "Deriv": 0,
        "Window": 17,
    }
    gui_app._original_wavelength_order = [float(c) for c in X_df.columns]
    gui_app.refine_task_type.set("classification")
    gui_app.refine_model_type.set("PLS-DA")
    gui_app.refine_preprocess.set("raw")
    gui_app.refine_folds.set(3)
    gui_app.refine_cv_strategy.set("kfold")
    gui_app.refine_y_transform.set("Log")
    gui_app.model_loaded_from_results = True
    gui_app.refine_hyperparams_modified = False
    gui_app.refined_model = None
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            gui_app._run_refined_model_thread()
        gui_app.root.update()
    finally:
        gui_app.refine_y_transform.set("None")
    assert gui_app.refined_model is not None, buf.getvalue()[-3000:]

    assert not isinstance(gui_app.refined_model, TransformedTargetRegressor)
    assert gui_app.refined_config["y_transform"] == "none"
    loaded = _save_and_load(gui_app, tmp_path)
    assert loaded["metadata"]["y_transform"] == "none"
