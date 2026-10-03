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

    # Refit B (a different model) starts; its inputs are frozen at launch.
    app.selected_model_config = _row("Ridge", False)
    app.refine_model_type.set("Ridge")
    app._run_refined_model()  # worker deferred
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

    # The widgets move while B runs: the run must not see it.
    app.refine_model_type.set("PLS")
    # B completes on this thread.
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


# --- Round 4: one immutable published state; Save/Export from it only ------------------


def _state_fields(app) -> dict:
    from spectral_predict_gui_optimized import _REFINED_STATE_FIELDS

    st = app._refined_state
    return {name: st.get(name) for name in _REFINED_STATE_FIELDS}


def test_export_snapshot_preserves_training_params_and_preprocessing(
    gui_app, tmp_path, monkeypatch
):
    """Export/Save describe the trained model A, not Results row B selected later."""
    import spectral_predict.cv_utils as cvu

    _refit(gui_app, "Ridge", "None", subset=True)  # A: alpha 0.01, sg1 (Poly 2), no autoscale
    config_a = dict(gui_app.refined_config)

    # Select row B with different settings, toggle autoscale, and let B's run fail.
    row_b = _row("Ridge", early_stopping=True)
    row_b.update(Params=str({"alpha": 9.0}), Deriv=2, Poly=3, imbalance_method="binning")
    gui_app.selected_model_config = row_b
    gui_app.use_autoscale.set(True)

    def _boom(*_a, **_k):
        raise RuntimeError("run B fails")

    monkeypatch.setattr(cvu, "build_cv_splitter", _boom)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            gui_app._run_refined_model_thread()
        gui_app.root.update()
        assert gui_app.refined_config == config_a  # B did not publish

        cfg = gui_app._build_export_model_config()
        assert cfg["params"] == {"alpha": 0.01}
        assert cfg["deriv_order"] == 1
        assert cfg["polyorder"] == 2
        assert cfg["imbalance_method"] is None
        assert cfg["early_stopping_rounds"] is None
        assert cfg["autoscale"] is False

        loaded = _save_and_load(gui_app, tmp_path)
        meta = loaded["metadata"]
        assert meta["params"] == str({"alpha": 0.01})
        assert meta["polyorder"] == 2
        assert meta["imbalance_method"] is None
        assert meta["autoscale"] is False
    finally:
        gui_app.use_autoscale.set(False)


def test_plot_reader_during_publish(gui_app):
    """A finished worker's publication cannot be seen half-way by any reader."""
    import dataclasses

    _refit(gui_app, "PLS", "None", subset=True)  # A
    st_a = gui_app._refined_state
    n = len(st_a.y_true)
    state_b = _state_fields(gui_app)
    state_b.update(y_true=np.arange(n - 5, dtype=float), y_pred=np.arange(n - 5, dtype=float))

    # Worker side: publication is queued for the Tk thread, so a Tk-thread reader
    # (plots) running before the next event still sees A completely.
    gui_app._publish_refined_state_on_tk_thread(state_b)
    assert gui_app._refined_state is st_a
    assert len(gui_app.refined_y_true) == len(gui_app.refined_y_pred) == n
    gui_app._plot_refined_predictions()

    # A reader that captured the state keeps one complete run; the object is frozen.
    gui_app.root.update()
    assert gui_app._refined_state is not st_a
    assert len(st_a.y_true) == len(st_a.y_pred) == n
    assert len(gui_app.refined_y_true) == len(gui_app.refined_y_pred) == n - 5
    with pytest.raises(dataclasses.FrozenInstanceError):
        st_a.y_true = None


def test_learning_curve_captures_one_refined_state(gui_app, monkeypatch):
    """The learning-curve worker uses one run's data AND model, even if B publishes."""
    from sklearn.linear_model import Ridge

    import spectral_predict.cv_utils as cvu
    import spectral_predict.diagnostics as diagnostics

    _refit(gui_app, "PLS", "None", subset=True)  # A
    model_a, X_a = gui_app.refined_model, gui_app.refined_X_train
    state_b = _state_fields(gui_app)
    state_b.update(model=Ridge(alpha=1.0), X_train=np.asarray(X_a) * 2.0)
    real_splitter = cvu.build_cv_splitter

    def _splitter_while_b_publishes(*a, **k):
        gui_app._publish_refined_state(state_b)  # B lands mid-operation
        return real_splitter(*a, **k)

    captured = {}

    def _capture(estimator, X, y, cv, **_k):
        captured.update(estimator=estimator, X=X)
        return {}

    monkeypatch.setattr(cvu, "build_cv_splitter", _splitter_while_b_publishes)
    monkeypatch.setattr(diagnostics, "compute_learning_curve", _capture)
    monkeypatch.setattr(gui_app, "_plot_learning_curve", lambda: None)
    gui_app._run_learning_curve_thread()
    gui_app.root.update()

    assert type(captured["estimator"]) is type(model_a)
    assert captured["X"] is X_a


def test_open_export_dialog_rejects_failed_publish(gui_app, monkeypatch):
    """An Export dialog left open must not export a state without a valid token."""
    import tkinter as tk

    import spectral_predict.code_generator as code_generator

    captured, toplevels = {}, []
    real_button, real_toplevel = tk.Button, tk.Toplevel

    def _button(*a, **k):
        if "EXPORT" in str(k.get("text", "")):
            captured["do_export"] = k["command"]
        return real_button(*a, **k)

    def _toplevel(*a, **k):
        win = real_toplevel(*a, **k)
        toplevels.append(win)
        return win

    monkeypatch.setattr(tk, "Button", _button)
    monkeypatch.setattr(tk, "Toplevel", _toplevel)
    _refit(gui_app, "PLS", "None", subset=True)
    try:
        gui_app._export_for_publication()
        assert "do_export" in captured
        gui_app._refined_model_token = None  # the publication failed / is incomplete
        with (
            patch("tkinter.filedialog.asksaveasfilename") as dialog,
            patch.object(code_generator, "CodeGenerator") as generator,
        ):
            captured["do_export"]()
        assert not dialog.called and not generator.called
        assert gui_app._refined_state_snapshot() is None
    finally:
        for win in toplevels:
            win.destroy()


def test_loading_results_row_refused_while_refit_runs(gui_app):
    sentinel = object()
    gui_app.loaded_model_config = sentinel
    gui_app._refit_active = True
    try:
        with patch("tkinter.messagebox.showwarning") as warn:
            gui_app._load_model_for_refinement({"Model": "Ridge", "Task": "regression"})
        assert warn.called
        assert gui_app.loaded_model_config is sentinel
    finally:
        gui_app._refit_active = False
        gui_app.loaded_model_config = None


# --- Round 5: the worker and Save/Export never read the live selection or widgets -----


def test_double_click_during_refit_keeps_selection_and_run_uses_a(
    correction_on, deferred_refits, tmp_path
):
    """Real Results double-click path while A runs: refused; A's row is what trains/saves."""
    app = correction_on
    _refit(app, "Ridge", "None", subset=True)  # loads data / widgets
    row_a = _row("Ridge", early_stopping=False)
    row_a.update(Params=str({"alpha": 0.01}), optuna_params={"alpha": 0.01}, is_coupled=False)
    app.selected_model_config = row_a
    app._run_refined_model()  # A launched, worker deferred
    assert app._refit_active

    row_b = dict(row_a, Params=str({"alpha": 9.0}), Rank=2)
    row_b.pop("optuna_params")
    app.results_display_df = pd.DataFrame([row_b], index=[0])
    app.results_tree.insert("", "end", iid="0", values=())
    app.results_tree.selection_set("0")
    try:
        with patch("tkinter.messagebox.showwarning") as warn:
            app._on_result_double_click(None)
        assert warn.called
        assert app.selected_model_config is row_a  # untouched

        # Even a selection change by any other path cannot reach the running worker.
        app.selected_model_config = row_b
        target, args = deferred_refits[0]
        with contextlib.redirect_stdout(io.StringIO()):
            target(*args)
        app.root.update()
    finally:
        app.results_tree.delete("0")
        app.results_display_df = None

    inner = app.refined_model.named_steps["model"]
    assert inner.alpha == pytest.approx(0.01)
    assert app.refined_config["optuna_params"] == {"alpha": 0.01}
    loaded = _save_and_load(app, tmp_path)
    meta = loaded["metadata"]
    assert meta["params"] == str({"alpha": 0.01})
    assert meta["optuna_params"] == {"alpha": 0.01}
    assert meta["is_coupled_result"] is True  # set because A carried optuna_params
    assert app._build_export_model_config()["params"] == {"alpha": 0.01}


def test_saved_metadata_describes_the_run_not_later_widgets(gui_app, tmp_path):
    """Toggling validation / data type / x unit after the refit does not change the file."""
    gui_app.current_data_type.set("reflectance")
    gui_app.current_x_unit.set("nm")
    _refit(gui_app, "PLS", "None", subset=True)  # validation disabled in _refit
    try:
        gui_app.current_data_type.set("absorbance")
        gui_app.current_x_unit.set("cm-1")
        gui_app.validation_enabled.set(True)
        gui_app.validation_indices = ["s1", "s2"]
        loaded = _save_and_load(gui_app, tmp_path)
    finally:
        gui_app.validation_enabled.set(False)
        gui_app.validation_indices = []
        gui_app.current_data_type.set("reflectance")
        gui_app.current_x_unit.set("nm")
    meta = loaded["metadata"]
    assert meta["data_type"] == "reflectance"
    assert meta["x_unit"] == "nm"
    assert meta["validation_set_enabled"] is False
    assert meta["validation_size"] == 0
    assert meta["validation_algorithm"] is None


def test_plot_click_after_newer_publish_offers_plotted_runs_specimen(gui_app, monkeypatch):
    """Codex round 5: a click on A's plot must not offer B's specimen for exclusion."""
    import types

    import matplotlib.backend_bases as backend_bases
    from matplotlib.axes import Axes

    _refit(gui_app, "PLS", "None", subset=True)  # A
    st_a = gui_app._refined_state
    callbacks = []
    real_connect = backend_bases.FigureCanvasBase.mpl_connect

    def _record(canvas, name, func):
        callbacks.append((name, func))
        return real_connect(canvas, name, func)

    monkeypatch.setattr(backend_bases.FigureCanvasBase, "mpl_connect", _record)
    offered = []
    monkeypatch.setattr(
        gui_app, "_show_exclude_button", lambda _f, label, *a: offered.append(label)
    )
    monkeypatch.setattr(gui_app, "_create_or_update_annotation", lambda *a, **k: None)
    gui_app._plot_regression_predictions()
    click = [f for n, f in callbacks if n == "button_press_event"][-1]
    ax = next(
        c.cell_contents
        for c in click.__closure__
        if isinstance(getattr(c, "cell_contents", None), Axes)
    )

    # B is published (same number of CV predictions, different specimens/order).
    state_b = _state_fields(gui_app)
    state_b.update(
        specimen_ids=[f"B{i}" for i in range(len(st_a.y_true))],
        cv_indices=np.asarray(st_a.cv_indices)[::-1].copy(),
    )
    gui_app._publish_refined_state(state_b)

    k = 3
    click(types.SimpleNamespace(inaxes=ax, xdata=st_a.y_true[k], ydata=st_a.y_pred[k], button=1))
    assert offered == [st_a.specimen_ids[k]]


def test_shap_captures_one_refined_state(gui_app, monkeypatch):
    import spectral_predict_gui_optimized as gui_mod

    if not gui_mod.HAS_SHAP:
        pytest.skip("shap not installed")
    _refit(gui_app, "PLS", "None", subset=True)  # A
    st_a = gui_app._refined_state
    state_b = _state_fields(gui_app)
    state_b.update(X_cv=np.zeros((5, 3)), model=None)

    class _FakeLinear:
        def __init__(self, model, X):
            self.X = X

        def shap_values(self, X):
            return np.zeros_like(np.asarray(X, dtype=float))

    monkeypatch.setattr(gui_mod.shap, "LinearExplainer", _FakeLinear)
    # root.update() inside the computation runs a queued publication of B.
    monkeypatch.setattr(gui_app.root, "update", lambda: gui_app._publish_refined_state(state_b))
    monkeypatch.setattr(gui_app, "_plot_shap_summary", lambda: None)
    gui_app._compute_shap_values()

    assert gui_app._shap_state is st_a
    assert gui_app.shap_values.shape == np.asarray(st_a.X_cv).shape


def test_one_class_save_and_export_ignore_switched_row(gui_app, tmp_path):
    """One-class path: Save/Export keep the trained row's settings after a row switch."""
    rng = np.random.default_rng(2)
    wl = np.linspace(1200.0, 1600.0, 40)
    X = pd.DataFrame(
        0.5 + 0.02 * rng.normal(size=(30, 40)),
        columns=[f"{w:.1f}" for w in wl],
        index=[f"o{i}" for i in range(30)],
    )
    y = pd.Series(np.where(np.arange(30) % 3 == 0, "bad", "good"), index=X.index)
    row_a = {
        "Model": "OneClassSVM",
        "Task": "one_class",
        "Params": str({"nu": 0.1}),
        "Preprocess": "raw",
        "Deriv": 0,
        "Window": 17,
        "Poly": 2,
    }
    gui_app.X_original = X
    gui_app.X = X
    gui_app.y = y
    gui_app.active_indices = None
    gui_app.excluded_spectra = set()
    gui_app.validation_enabled.set(False)
    gui_app.validation_indices = []
    gui_app.use_autoscale.set(False)
    gui_app.selected_model_config = row_a
    gui_app._original_wavelength_order = [float(w) for w in X.columns]
    gui_app.refine_task_type.set("one_class")
    gui_app.refine_model_type.set("OneClassSVM")
    gui_app.refine_preprocess.set("raw")
    gui_app.refine_folds.set(3)
    gui_app.refine_cv_strategy.set("kfold")
    gui_app.inlier_class_label.set("good")
    gui_app.model_loaded_from_results = True
    gui_app.refine_hyperparams_modified = False
    gui_app.refined_model = None
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            gui_app._run_refined_model_thread()
        gui_app.root.update()
        assert gui_app.refined_config["task_type"] == "one_class", buf.getvalue()[-3000:]

        # Switch to another row, change the inlier label and autoscale afterwards.
        gui_app.selected_model_config = dict(row_a, Params=str({"nu": 0.4}), Poly=3)
        gui_app.inlier_class_label.set("bad")
        gui_app.use_autoscale.set(True)

        loaded = _save_and_load(gui_app, tmp_path)
        cfg = gui_app._build_export_model_config()
    finally:
        gui_app.inlier_class_label.set("")
        gui_app.use_autoscale.set(False)
        gui_app.refine_task_type.set("regression")
    meta = loaded["metadata"]
    assert meta["params"] == str({"nu": 0.1})
    assert meta["polyorder"] == 2
    assert meta["autoscale"] is False
    assert meta["inlier_class_label"] == "good"
    assert cfg["params"] == {"nu": 0.1}
    assert cfg["polyorder"] == 2
    assert cfg["inlier_class_label"] == "good"


# --- Round 6 ---------------------------------------------------------------------------


def _set_wl_spec(app, text: str) -> None:
    app.refine_wl_spec.delete("1.0", "end")
    app.refine_wl_spec.insert("1.0", text)


def test_worker_ignores_wavelength_order_changed_mid_run(gui_app, deferred_refits):
    """DeepSeek round 6: the spec parser used the live _original_wavelength_order."""
    X_df, _ = _refit(gui_app, "PLS", "None", subset=True)
    all_wl = [float(c) for c in X_df.columns]
    in_range = [w for w in all_wl if 1200.0 <= w <= 1400.0]
    try:
        _set_wl_spec(gui_app, "1200-1400")
        gui_app._original_wavelength_order = []  # empty: the worker parses the spec text
        gui_app._run_refined_model()  # inputs frozen, worker deferred
        # Mid-run, a variable-selection order appears (reversed, and only 5 wavelengths).
        gui_app._original_wavelength_order = list(reversed(in_range))[:5]
        target, args = deferred_refits[0]
        with contextlib.redirect_stdout(io.StringIO()):
            target(*args)
        gui_app.root.update()
        assert gui_app.refined_wavelengths == in_range
    finally:
        _set_wl_spec(gui_app, "")
        gui_app._original_wavelength_order = None


def test_plot_click_without_run_specimen_ids_offers_nothing(gui_app, monkeypatch):
    """No pairing of a run's CV indices with the live self.y.index (round 6)."""
    import types

    import matplotlib.backend_bases as backend_bases
    from matplotlib.axes import Axes

    _refit(gui_app, "PLS", "None", subset=True)
    state = _state_fields(gui_app)
    state.update(specimen_ids=None)
    gui_app._publish_refined_state(state)
    st = gui_app._refined_state

    callbacks = []
    real_connect = backend_bases.FigureCanvasBase.mpl_connect

    def _record(canvas, name, func):
        callbacks.append((name, func))
        return real_connect(canvas, name, func)

    monkeypatch.setattr(backend_bases.FigureCanvasBase, "mpl_connect", _record)
    offered = []
    monkeypatch.setattr(
        gui_app, "_show_exclude_button", lambda _f, label, *a: offered.append(label)
    )
    monkeypatch.setattr(gui_app, "_create_or_update_annotation", lambda *a, **k: None)
    gui_app._plot_regression_predictions()
    click = [f for n, f in callbacks if n == "button_press_event"][-1]
    ax = next(
        c.cell_contents
        for c in click.__closure__
        if isinstance(getattr(c, "cell_contents", None), Axes)
    )
    click(types.SimpleNamespace(inaxes=ax, xdata=st.y_true[2], ydata=st.y_pred[2], button=1))
    assert offered == []


def test_captured_frames_are_frozen_shallow_copies(gui_app):
    """Round 6: captured frames share data (no memory cost) but ignore in-place edits."""
    X_df, _ = _refit(gui_app, "PLS", "None", subset=True)
    inputs = gui_app._capture_refit_inputs()
    captured = inputs["X"]
    assert captured is not gui_app.X
    assert np.shares_memory(captured.to_numpy(), gui_app.X.to_numpy())

    original_columns = list(captured.columns)
    gui_app.X.columns = [f"w{i}" for i in range(gui_app.X.shape[1])]  # in-place edit
    gui_app.X.iloc[0, 0] = 1e9
    try:
        assert list(captured.columns) == original_columns
        assert captured.iloc[0, 0] != 1e9
    finally:
        gui_app.X = X_df
