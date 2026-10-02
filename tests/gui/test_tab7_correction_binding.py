"""R010/R064: a bias or nonlinear correction is saved only with the model it was fitted for.

Before the fix ``_save_refined_model`` embedded whatever ``nonlinear_correction_data`` /
``bias_correction_data`` was lying around: a polynomial computed for an earlier model,
or a regression correction after switching to classification (which
``predict_with_model`` then applied to class labels).
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
import pytest

from tests.gui.test_tab7_y_transform_save import _refit, _save_and_load, _spectra

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


def test_compute_and_save_disabled_while_refit_runs(gui_app, monkeypatch):
    import threading

    _refit(gui_app, "PLS", "None", subset=True)
    started = []

    class _DeferredThread:
        def __init__(self, target=None, daemon=None, **_kw):
            started.append(target)

        def start(self):
            pass  # the run is "in progress" for the rest of the test

    monkeypatch.setattr(threading, "Thread", _DeferredThread)
    # Widget-level parameter validation is not under test here.
    monkeypatch.setattr(gui_app, "_validate_refinement_parameters", lambda: True)
    try:
        gui_app._run_refined_model()
        assert started, "refit thread was not launched"
        for name in ("refine_save_button", "refine_save_button_results", "bc_compute_button"):
            assert str(getattr(gui_app, name).cget("state")) == "disabled", name
    finally:
        gui_app._restore_refit_buttons_after_abort()
