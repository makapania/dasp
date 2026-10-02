"""R010/R064: a bias or nonlinear correction is saved only with the model it was fitted for.

Before the fix ``_save_refined_model`` embedded whatever ``nonlinear_correction_data`` /
``bias_correction_data`` was lying around: a polynomial computed for an earlier model,
or a regression correction after switching to classification (which
``predict_with_model`` then applied to class labels).
"""

from __future__ import annotations

import contextlib
import io
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from spectral_predict.model_io import predict_with_model
from tests.gui.test_tab7_y_transform_save import _refit, _row, _save_and_load, _spectra

pytestmark = pytest.mark.gui


@pytest.fixture
def correction_on(gui_app):
    gui_app.apply_bias_correction.set(True)
    gui_app.save_correction_with_model.set(True)
    gui_app.use_nonlinear_correction.set(True)
    gui_app.nonlinear_correction_method.set("Polynomial (3)")
    yield gui_app
    gui_app.apply_bias_correction.set(False)
    gui_app.use_nonlinear_correction.set(False)
    gui_app.nonlinear_correction_method.set("Polynomial (2)")


def _compute_nonlinear(app) -> dict:
    app._compute_nonlinear_correction()
    assert app.nonlinear_correction_data is not None
    return app.nonlinear_correction_data


def test_nonlinear_correction_from_run_a_not_saved_with_run_b(correction_on, tmp_path):
    app = correction_on
    _refit(app, "PLS", "None", subset=True)
    stale = _compute_nonlinear(app)

    _refit(app, "Ridge", "None", subset=False)  # run B: a different model
    loaded = _save_and_load(app, tmp_path)

    saved = loaded["bias_correction"]
    assert saved != stale
    # Falls back to run B's own linear correction (computed after run B finished).
    assert saved is not None and saved["method"] == "linear"
    assert saved == app.bias_correction_data
    np.testing.assert_allclose(saved["bias"], app.bias_correction_data["bias"])


def test_correction_computed_after_run_is_saved(correction_on, tmp_path):
    """The intended workflow still works: run, compute the correction, save."""
    app = correction_on
    _refit(app, "PLS", "None", subset=True)
    _refit(app, "Ridge", "None", subset=True)
    current = _compute_nonlinear(app)

    loaded = _save_and_load(app, tmp_path)
    assert loaded["bias_correction"] == current
    assert loaded["bias_correction"]["method"] == "nonlinear"


def test_regression_correction_not_saved_with_classifier(correction_on, tmp_path):
    app = correction_on
    _refit(app, "PLS", "None", subset=True)
    _compute_nonlinear(app)
    assert app.bias_correction_data is not None

    X_df, y = _spectra()
    labels = pd.Series(np.where(y.values > np.median(y.values), "hi", "lo"), index=y.index)
    app.X_original = X_df
    app.X = X_df
    app.y = labels
    app.selected_model_config = {
        "Model": "PLS-DA",
        "Task": "classification",
        "Params": str({"n_components": 2}),
        "LVs": 2,
        "Preprocess": "raw",
        "Deriv": 0,
        "Window": 17,
    }
    app._original_wavelength_order = [float(c) for c in X_df.columns]
    app.refine_task_type.set("classification")
    app.refine_model_type.set("PLS-DA")
    app.refine_preprocess.set("raw")
    app.refined_model = None
    with contextlib.redirect_stdout(io.StringIO()):
        app._run_refined_model_thread()
    app.root.update()
    assert app.refined_config["task_type"] == "classification"

    assert app.bias_correction_data is None
    assert app.nonlinear_correction_data is None
    loaded = _save_and_load(app, tmp_path)
    assert loaded["bias_correction"] is None
    assert loaded["metadata"]["has_bias_correction"] is False


def test_nonlinear_correction_rejects_model_change_during_compute(correction_on, monkeypatch):
    """A refit that finishes while Compute runs must not adopt the old correction."""
    import spectral_predict.bias_correction as bc

    app = correction_on
    _refit(app, "PLS", "None", subset=True)
    real = bc.compute_nonlinear_correction

    def _model_b_finishes_mid_compute(*args, **kwargs):
        result = real(*args, **kwargs)
        # What the refit worker does when model B's CV predictions are stored.
        app._bind_corrections_to_new_model()
        return result

    monkeypatch.setattr(bc, "compute_nonlinear_correction", _model_b_finishes_mid_compute)
    app._compute_nonlinear_correction()

    assert app.nonlinear_correction_data is None
    assert app._nonlinear_correction_token is None
    assert app._correction_to_save() is None


@pytest.fixture
def deferred_refits(gui_app, monkeypatch):
    """Launch refits without running them; the test runs each worker when it wants.

    Yields the list of launched ``(target, args)``; always clears the active-refit
    state afterwards so later tests on the shared app are unaffected.
    """
    import threading

    launched: list = []

    class _DeferredThread:
        def __init__(self, target=None, args=(), daemon=None, **_kw):
            launched.append((target, args))

        def start(self):
            pass  # the run is "in progress" until the test runs the worker

    monkeypatch.setattr(threading, "Thread", _DeferredThread)
    # Widget-level parameter validation is not under test here.
    monkeypatch.setattr(gui_app, "_validate_refinement_parameters", lambda: True)
    try:
        yield launched
    finally:
        gui_app._end_refit(gui_app._refit_generation)
        gui_app._restore_refit_buttons_after_abort()


REFIT_DEPENDENT = (
    "refine_save_button",
    "refine_save_button_results",
    "export_code_button",
    "bc_compute_button",
)


def test_compute_and_save_disabled_while_refit_runs(gui_app, deferred_refits):
    _refit(gui_app, "PLS", "None", subset=True)
    gui_app._run_refined_model()
    assert deferred_refits, "refit thread was not launched"
    for name in REFIT_DEPENDENT:
        assert str(getattr(gui_app, name).cget("state")) == "disabled", name


def test_export_refused_and_compute_ignored_while_refit_runs(
    correction_on, deferred_refits, monkeypatch
):
    """GLM/Codex round 3: method-level busy guards, not only disabled buttons."""
    import tkinter as tk

    import spectral_predict.bias_correction as bc

    app = correction_on
    _refit(app, "PLS", "None", subset=True)
    app._run_refined_model()
    assert app._refit_active

    opened = []
    monkeypatch.setattr(tk, "Toplevel", lambda *a, **k: opened.append(1))
    with patch("tkinter.messagebox.showwarning") as warn:
        app._export_for_publication()
    assert warn.called and not opened

    def _must_not_run(*_a, **_k):
        raise AssertionError("correction computed while a refit is running")

    monkeypatch.setattr(bc, "compute_nonlinear_correction", _must_not_run)
    monkeypatch.setattr(bc, "compute_bias_slope", _must_not_run)
    app._compute_nonlinear_correction()
    app._update_bias_correction_ui()


def test_export_config_comes_from_one_snapshot(gui_app):
    _refit(gui_app, "PLS", "None", subset=True)
    snap = gui_app._refined_state_snapshot()
    _refit(gui_app, "Ridge", "None", subset=False)  # another model is published

    cfg = gui_app._build_export_model_config(snap)
    assert cfg["model_name"] == "PLS"
    assert cfg["wavelengths"] == snap["wavelengths"]
    assert gui_app._build_export_model_config()["model_name"] == "Ridge"


class _LaunchError(RuntimeError):
    pass


@pytest.mark.parametrize("where", ["constructor", "start"])
def test_refit_launch_failure_releases_busy(gui_app, monkeypatch, where):
    """test_refit_constructor_failure_releases_busy / test_refit_start_failure_releases_busy."""
    import threading

    _refit(gui_app, "PLS", "None", subset=True)

    class _FailingThread:
        def __init__(self, *a, **k):
            if where == "constructor":
                raise _LaunchError("can't start new thread")

        def start(self):
            raise _LaunchError("can't start new thread")

    monkeypatch.setattr(threading, "Thread", _FailingThread)
    monkeypatch.setattr(gui_app, "_validate_refinement_parameters", lambda: True)
    with patch("tkinter.messagebox.showerror") as err:
        gui_app._run_refined_model()
    assert err.called
    assert gui_app._refit_active is False
    assert str(gui_app.refine_run_button.cget("state")) == "normal"
    # The previous model is still complete, so it can still be saved/exported.
    for name in REFIT_DEPENDENT:
        assert str(getattr(gui_app, name).cget("state")) == "normal", name


@pytest.mark.parametrize("exit_path", ["no_inlier", "too_few_folds"])
def test_failed_one_class_refit_preserves_previous_save_state(
    gui_app, tmp_path, monkeypatch, exit_path
):
    """Codex round 3: an early one-class exit must not touch the previous save state."""
    import spectral_predict.contamination as contamination

    X_a, y_a = _refit(gui_app, "PLS", "None", subset=True)  # model A
    before = gui_app._refined_state_snapshot()

    # One-class attempt on data with a DIFFERENT wavelength axis.
    rng = np.random.default_rng(1)
    wl_b = np.linspace(1200.0, 1600.0, 40)
    X_b = pd.DataFrame(
        0.5 + 0.01 * rng.normal(size=(30, 40)),
        columns=[f"{w:.1f}" for w in wl_b],
        index=[f"b{i}" for i in range(30)],
    )
    y_b = pd.Series(np.where(np.arange(30) % 2 == 0, "good", "bad"), index=X_b.index)
    gui_app.X_original = X_b
    gui_app.X = X_b
    gui_app.y = y_b
    gui_app._original_wavelength_order = [float(w) for w in X_b.columns]
    gui_app.selected_model_config = {
        "Model": "OneClassSVM",
        "Task": "one_class",
        "Params": str({"nu": 0.1}),
        "Preprocess": "raw",
        "Deriv": 0,
        "Window": 17,
    }
    gui_app.refine_task_type.set("one_class")
    gui_app.refine_model_type.set("OneClassSVM")
    gui_app.refine_preprocess.set("raw")
    if exit_path == "no_inlier":
        gui_app.inlier_class_label.set("absent-class")
    else:
        gui_app.inlier_class_label.set("good")
        monkeypatch.setattr(contamination, "run_one_class_cv", lambda *a, **k: {"skipped": True})
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            gui_app._run_refined_model_thread()
        gui_app.root.update()
    finally:
        gui_app.inlier_class_label.set("")
        gui_app.refine_task_type.set("regression")

    after = gui_app._refined_state_snapshot()
    assert after["token"] is before["token"]
    assert after["model"] is before["model"]
    assert after["full_wavelengths"] == before["full_wavelengths"]
    assert after["config"] == before["config"]
    # Save stays available for the intact previous model ...
    assert str(gui_app.refine_save_button.cget("state")) == "normal"

    # ... and the saved file is model A: A's axis, A's predictions.
    loaded = _save_and_load(gui_app, tmp_path)
    assert loaded["metadata"]["full_wavelengths"] == [float(c) for c in X_a.columns]
    pred = np.asarray(predict_with_model(loaded, X_a), dtype=float).ravel()
    expected = np.asarray(before["model"].predict(before["X_train"]), dtype=float).ravel()
    np.testing.assert_allclose(pred, expected, rtol=1e-9)


def test_refit_tab_return_cannot_overlap_or_save_stale_correction(
    correction_on, deferred_refits, tmp_path
):
    """Codex round 2: a second refit must not start, and save must not mix runs."""
    app = correction_on
    _refit(app, "PLS", "None", subset=True)  # model A
    stale = _compute_nonlinear(app)

    app._run_refined_model()  # refit B starts (worker deferred)
    assert len(deferred_refits) == 1
    # Leaving and re-entering Model Development (or loading a Results row) re-enables
    # Run while B is still running. A second click must be refused.
    for name in ("refine_run_button", "refine_run_button_selection"):
        getattr(app, name).config(state="normal")
    app._run_refined_model()
    assert len(deferred_refits) == 1, "a second refit started while one was running"

    # Save is refused while B is running (no file, nothing half-published).
    blocked = tmp_path / "during.dasp"
    with patch("tkinter.filedialog.asksaveasfilename", return_value=str(blocked)):
        app._save_refined_model()
    assert not blocked.exists()

    # B (a different model) completes on this thread.
    app.selected_model_config = _row("Ridge", False)
    app.refine_model_type.set("Ridge")
    target, args = deferred_refits[0]
    with contextlib.redirect_stdout(io.StringIO()):
        target(*args)
    app.root.update()
    assert app._refit_active is False
    assert type(app.refined_model).__name__ == "Pipeline"  # scaler + Ridge

    loaded = _save_and_load(app, tmp_path, "b.dasp")
    assert loaded["bias_correction"] != stale
    assert loaded["bias_correction"]["method"] == "linear"
    assert type(loaded["model"]).__name__ == "Pipeline"


def test_save_aborts_if_model_changes_during_save_dialog(correction_on, tmp_path):
    app = correction_on
    _refit(app, "PLS", "None", subset=True)
    path = tmp_path / "swapped.dasp"

    def _dialog_while_model_swaps(**_kw):
        app._bind_corrections_to_new_model()  # another model is published meanwhile
        return str(path)

    with patch("tkinter.filedialog.asksaveasfilename", side_effect=_dialog_while_model_swaps):
        app._save_refined_model()
    assert not path.exists()


def test_linear_correction_rejects_model_change_during_compute(correction_on, monkeypatch):
    import spectral_predict.bias_correction as bc

    app = correction_on
    _refit(app, "PLS", "None", subset=True)
    real = bc.compute_bias_slope

    def _model_changes_mid_compute(*args, **kwargs):
        result = real(*args, **kwargs)
        app._bind_corrections_to_new_model()
        return result

    monkeypatch.setattr(bc, "compute_bias_slope", _model_changes_mid_compute)
    app._update_bias_correction_ui()
    assert app.bias_correction_data is None
    assert app._correction_to_save() is None


def test_error_after_valid_model_keeps_save_enabled(gui_app):
    _refit(gui_app, "PLS", "None", subset=True)
    buttons = ("refine_save_button", "refine_save_button_results", "export_code_button")
    gui_app._update_refined_results("failed before replacing the model", is_error=True)
    for name in buttons:
        assert str(getattr(gui_app, name).cget("state")) == "normal", name

    gui_app._refined_model_token = None  # failure in the middle of a model swap
    gui_app._update_refined_results("failed mid-swap", is_error=True)
    for name in buttons:
        assert str(getattr(gui_app, name).cget("state")) == "disabled", name
