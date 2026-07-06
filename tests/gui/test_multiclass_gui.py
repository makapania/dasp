"""GUI contract tests for the T-31 multi-class SIMCA task type (Phase D).

These exercise the real Tkinter app headless (root withdrawn) via the shared
session_app fixture. They assert the 5th task radio + its control panel toggle
correctly and never disturb the single-Y (regression/classification/one-class)
paths.
"""

from __future__ import annotations

import tkinter as tk

import pytest

from spectral_predict.model_registry import MULTICLASS_ENGINES


def _is_managed(widget) -> bool:
    """True if the widget is currently placed by a geometry manager."""
    try:
        return bool(widget.winfo_manager())
    except tk.TclError:
        return False


# ---------------------------------------------------------------------------
# D1 — radio + controls
# ---------------------------------------------------------------------------

def test_multiclass_task_vars_exist(gui_app):
    """The multiclass control variables are created with spec defaults."""
    app = gui_app
    assert hasattr(app, "mc_alpha")
    assert abs(app.mc_alpha.get() - 0.05) < 1e-9
    # n_components default is the novelty-oriented 0.99 per-class variance frac
    assert str(app.mc_n_components.get()) == "0.99"
    assert app.mc_min_class_samples.get() == 10


def test_multiclass_engine_vars_cover_registry(gui_app):
    """One engine BooleanVar exists per MULTICLASS_ENGINES entry; pca-simca on."""
    app = gui_app
    engine_vars = app.mc_engine_vars  # dict engine_name -> BooleanVar
    assert set(engine_vars) == set(MULTICLASS_ENGINES)
    assert engine_vars["pca-simca"].get() is True


def test_multiclass_radio_selectable(gui_app):
    """The task_type var accepts the multiclass_simca value (radio wired)."""
    app = gui_app
    app.task_type.set("multiclass_simca")
    assert app.task_type.get() == "multiclass_simca"


def test_multiclass_controls_shown_and_others_hidden(gui_app):
    """Selecting multiclass shows its panel and hides one-class/standard panels."""
    app = gui_app
    app.task_type.set("multiclass_simca")
    app._on_task_type_changed()

    assert _is_managed(app.mc_hyperparams_frame), "multiclass control panel not shown"
    assert _is_managed(app.mc_models_frame), "engine picker not shown"
    # one-class controls hidden
    assert not _is_managed(app.inlier_class_frame)
    assert not _is_managed(app.oc_hyperparams_frame)
    # standard model panel hidden
    assert not _is_managed(app.standard_models_frame)


def test_switching_away_restores_standard_path(gui_app):
    """Leaving multiclass restores the standard regression panel byte-for-byte."""
    app = gui_app
    app.task_type.set("multiclass_simca")
    app._on_task_type_changed()
    app.task_type.set("regression")
    app._on_task_type_changed()

    assert not _is_managed(app.mc_hyperparams_frame)
    assert not _is_managed(app.mc_models_frame)
    assert _is_managed(app.standard_models_frame)


# ---------------------------------------------------------------------------
# D2 — decision-matrix + Wold view rendering
# ---------------------------------------------------------------------------

def _make_view():
    import numpy as np
    import pandas as pd
    from spectral_predict.search import build_multiclass_decision_view

    rng = np.random.default_rng(1)
    blocks, labels = [], []
    for k in range(3):
        blocks.append(rng.normal(k * 4.0, 1.0, size=(30, 30)))
        labels += [f"C{k}"] * 30
    X = pd.DataFrame(np.vstack(blocks), columns=[f"w{j}" for j in range(30)])
    y = pd.Series(labels)
    cfg = {
        "method": "raw", "name": "raw", "deriv": None, "window": None,
        "polyorder": None, "baseline_method": None, "baseline_params": None,
        "smoothing": False, "smoothing_window": 17, "smoothing_polyorder": 2,
    }
    return build_multiclass_decision_view(
        X, y, engine="pca-simca", preprocess_cfg=cfg, n_components=0.99,
    )


def test_decision_matrix_dataframe_columns(gui_app):
    app = gui_app
    view = _make_view()
    dm = app._multiclass_decision_matrix_dataframe(view)
    assert list(dm.columns[:3]) == ["Sample", "TrueClass", "Decision"]
    assert "Accepted" in dm.columns
    for c in view["classes"]:
        assert f"p({c})" in dm.columns
    assert len(dm) == len(view["labels"])


def test_run_analysis_accepts_multiclass_engine_selection(gui_app):
    """Regression: the pre-run 'select a model' guard must recognise the
    multi-class engine picker (mc_engine_vars), not just the standard/one-class
    model checkboxes. Previously it always warned 'Please select at least one
    model' for multiclass no matter how many engines were checked."""
    import threading
    from unittest.mock import patch

    import numpy as np
    import pandas as pd

    app = gui_app
    rng = np.random.default_rng(0)
    blocks, labels = [], []
    for k in range(3):
        blocks.append(rng.normal(k * 4.0, 1.0, size=(12, 20)))
        labels += [f"C{k}"] * 12
    app.X = pd.DataFrame(np.vstack(blocks))
    app.X_original = app.X.copy()
    app.y = pd.Series(labels)
    app.task_type.set("multiclass_simca")
    app._on_task_type_changed()
    app.mc_engine_vars["pca-simca"].set(True)

    warned = {}
    started = {"v": False}

    class _FakeThread:
        def __init__(self, target, args, daemon):
            self._t, self._a = target, args

        def start(self):
            self._t(*self._a)

    def _stub_worker(*_a, **_k):
        started["v"] = True

    with patch("tkinter.messagebox.showwarning",
               side_effect=lambda title, msg: warned.setdefault("m", (title, msg))), \
         patch.object(app, "_run_analysis_thread", _stub_worker), \
         patch("threading.Thread", _FakeThread):
        app._run_analysis()

    assert warned.get("m") is None, f"unexpected warning: {warned.get('m')}"
    assert started["v"] is True


def test_multiclass_leaderboard_renders(gui_app):
    """Regression: the leaderboard filter bar did int(NaN) on the non-numeric
    multiclass LVs column ("auto" / per-class dict string), crashing the whole
    results-table render. It must populate without error."""
    import numpy as np
    import pandas as pd
    from spectral_predict.search import run_multiclass_simca_search

    app = gui_app
    rng = np.random.default_rng(0)
    blocks, labels = [], []
    for k in range(4):
        blocks.append(rng.normal(k * 4.0, 1.0, size=(25, 30)))
        labels += [f"C{k}"] * 25
    X = pd.DataFrame(np.vstack(blocks))
    y = pd.Series(labels)
    res = run_multiclass_simca_search(
        X, y, engines=["pca-simca"], preprocessing_methods=["raw", "snv"],
        n_components=0.99, varsel_paths=["none"], cv_splits=3,
    )
    if "Select" not in res.columns:
        res.insert(0, "Select", False)
    # ComplexityScore must be computed (not NaN'd by the str-int TypeError)
    assert res["ComplexityScore"].notna().all()
    app._populate_results_table(res)
    app.root.update_idletasks()
    assert len(app.results_tree.get_children()) == len(res)


def test_show_decision_view_opens_window(gui_app):
    import tkinter as tk

    app = gui_app
    view = _make_view()
    before = [w for w in app.root.winfo_children() if isinstance(w, tk.Toplevel)]
    app._show_multiclass_decision_view(view)
    app.root.update_idletasks()
    after = [w for w in app.root.winfo_children() if isinstance(w, tk.Toplevel)]
    assert len(after) == len(before) + 1
    # clean up the window we opened
    after[-1].destroy()


# ---------------------------------------------------------------------------
# Run-a-selected-leaderboard-result (double-click) + Save Model (.dasp)
# ---------------------------------------------------------------------------

def _mc_row(preprocess="raw", engine="pca-simca", deriv=None, window=None, poly=None):
    """A synthetic multi-class leaderboard row (the subset of columns the
    run-selected handler reconstructs a config from)."""
    return {
        "Task": "multiclass_simca",
        "Model": engine,
        "engine_family": engine,
        "varsel_path": "none",
        "SubsetTag": "none",
        "Alpha": 0.05,
        "Preprocess": preprocess,
        "Deriv": deriv,
        "Window": window,
        "Poly": poly,
        "reason": "",
    }


class _SyncThread:
    """Thread stand-in that runs the target synchronously on .start()."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._t, self._a, self._k = target, args, (kwargs or {})

    def start(self):
        if self._t is not None:
            self._t(*self._a, **self._k)


def test_double_click_multiclass_routes_to_selected_result(gui_app):
    """A double-click on a multi-class leaderboard row must route to the
    dedicated run-selected handler, NOT the regression/classification Refine
    tab (get_model('pca-simca') would fail there)."""
    import pandas as pd
    from unittest.mock import patch

    app = gui_app
    app.results_display_df = pd.DataFrame([_mc_row()])
    app.task_type.set("multiclass_simca")
    orig_sel = app.results_tree.selection
    app.results_tree.selection = lambda: ("0",)
    try:
        with patch.object(app, "_run_selected_multiclass_result") as mc, \
             patch.object(app, "_load_model_for_refinement") as refine:
            app._on_result_double_click(None)
        assert mc.called, "multiclass row did not route to run-selected handler"
        assert not refine.called, "multiclass row leaked into the Refine path"
    finally:
        app.results_tree.selection = orig_sel


def test_double_click_regression_untouched(gui_app):
    """A double-click on a NON-multiclass row must still load into the Refine
    tab exactly as before (single-Y path byte-identical)."""
    import pandas as pd
    from unittest.mock import patch

    app = gui_app
    app.results_display_df = pd.DataFrame(
        [{"Task": "regression", "Model": "PLS", "R2cv": 0.9, "Rank": 1, "n_vars": 50}]
    )
    app.task_type.set("regression")
    orig_sel = app.results_tree.selection
    app.results_tree.selection = lambda: ("0",)
    try:
        with patch.object(app, "_run_selected_multiclass_result") as mc, \
             patch.object(app, "_load_model_for_refinement") as refine, \
             patch.object(app.notebook, "select"), \
             patch.object(app.model_dev_notebook, "select"):
            app._on_result_double_click(None)
        assert refine.called, "regression row no longer loads into Refine"
        assert not mc.called, "regression row wrongly hit the multiclass handler"
    finally:
        app.results_tree.selection = orig_sel


def test_double_click_regression_row_with_multiclass_radio_untouched(gui_app):
    """Codex MEDIUM: a regression row must NOT route to the multiclass handler
    just because the live task_type radio is set to multiclass (results table
    still holds the old single-Y rows after a radio flip). Routing keys on the
    ROW's Task, not the radio."""
    import pandas as pd
    from unittest.mock import patch

    app = gui_app
    app.results_display_df = pd.DataFrame(
        [{"Task": "regression", "Model": "PLS", "R2cv": 0.9, "Rank": 1, "n_vars": 50}]
    )
    app.task_type.set("multiclass_simca")  # radio moved on, but row is regression
    orig_sel = app.results_tree.selection
    app.results_tree.selection = lambda: ("0",)
    try:
        with patch.object(app, "_run_selected_multiclass_result") as mc, \
             patch.object(app, "_load_model_for_refinement") as refine, \
             patch.object(app.notebook, "select"), \
             patch.object(app.model_dev_notebook, "select"):
            app._on_result_double_click(None)
        assert refine.called, "regression row stolen by multiclass handler on radio flip"
        assert not mc.called
    finally:
        app.results_tree.selection = orig_sel
        app.task_type.set("regression")


def test_run_selected_multiclass_builds_view(gui_app):
    """The run-selected handler rebuilds the row's config on a worker thread and
    shows its decision view."""
    import numpy as np
    import pandas as pd
    from unittest.mock import patch

    app = gui_app
    rng = np.random.default_rng(0)
    blocks, labels = [], []
    for k in range(3):
        blocks.append(rng.normal(k * 4.0, 1.0, size=(30, 30)))
        labels += [f"C{k}"] * 30
    X = pd.DataFrame(np.vstack(blocks), columns=[f"{j}" for j in range(30)])
    y = pd.Series(labels)
    app._mc_export_data = (X, y)
    app._mc_run_config = {
        "n_components": 0.99, "min_class_samples": 10, "n_select": None,
        "baseline_method": None, "baseline_params": None, "smoothing": False,
        "smoothing_window": 17, "smoothing_polyorder": 2, "alpha": 0.05,
    }
    captured = {}
    with patch("threading.Thread", _SyncThread), \
         patch.object(app, "_show_multiclass_decision_view",
                      side_effect=lambda v: captured.setdefault("v", v)):
        app._run_selected_multiclass_result(_mc_row())
        app.root.update()  # flush the root.after that schedules the view
    v = captured.get("v")
    assert v is not None, "decision view was never shown"
    assert v["config"]["engine"] == "pca-simca"
    assert not v.get("reason"), v.get("reason")


def test_run_selected_requires_stashed_config(gui_app):
    """Guard: without the atomic _mc_export_data/_mc_run_config stash the handler
    must warn and NOT reconstruct a wrong (default-baseline) pipeline."""
    from unittest.mock import patch

    app = gui_app
    for attr in ("_mc_export_data", "_mc_run_config"):
        if hasattr(app, attr):
            delattr(app, attr)
    warned, shown = {}, {}
    with patch("tkinter.messagebox.showwarning",
               side_effect=lambda *a, **k: warned.setdefault("w", a)), \
         patch.object(app, "_show_multiclass_decision_view",
                      side_effect=lambda v: shown.setdefault("v", v)):
        app._run_selected_multiclass_result(_mc_row())
    assert warned.get("w") is not None
    assert shown.get("v") is None


def test_run_selected_rejects_unknown_varsel_path(gui_app):
    """An unrecognized varsel_path must warn (not silently coerce to None) and
    build no view."""
    import numpy as np
    import pandas as pd
    from unittest.mock import patch

    app = gui_app
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(30, 20)))
    y = pd.Series(["C0"] * 15 + ["C1"] * 15)
    app._mc_export_data = (X, y)
    app._mc_run_config = {"n_components": 0.99, "min_class_samples": 10, "n_select": None,
                          "baseline_method": None, "baseline_params": None, "smoothing": False,
                          "smoothing_window": 17, "smoothing_polyorder": 2, "alpha": 0.05}
    app._mc_worker_running = False
    row = _mc_row()
    row["varsel_path"] = "not_a_real_path"
    warned, shown = {}, {}
    with patch("tkinter.messagebox.showwarning",
               side_effect=lambda *a, **k: warned.setdefault("w", a)), \
         patch.object(app, "_show_multiclass_decision_view",
                      side_effect=lambda v: shown.setdefault("v", v)):
        app._run_selected_multiclass_result(row)
    assert warned.get("w") is not None
    assert shown.get("v") is None


@pytest.mark.parametrize(
    "cfg",
    [
        {"method": "raw", "name": "raw", "deriv": None, "window": None},
        {"method": "snv", "name": "snv", "deriv": None, "window": None},
        {"method": "deriv", "name": "deriv1_w7", "deriv": 1, "window": 7},
        {"method": "snv_deriv", "name": "snv_deriv1_w7", "deriv": 1, "window": 7},
    ],
)
@pytest.mark.parametrize("col_kind", ["float", "str"])
def test_save_multiclass_model_roundtrip(gui_app, tmp_path, cfg, col_kind):
    """Fitting + saving the selected config's model produces a loadable .dasp
    whose predict_with_model reproduces the in-sample decision matrix when fed
    the RAW spectra (preprocessing is applied on load, incl. SG edge mask).

    Covers numeric-STRING wavelength columns (Codex HIGH): CSV/Excel headers
    often load as strings, which must be stored as floats so predict's
    float-based wavelength matching does not treat them as missing."""
    import numpy as np
    import pandas as pd
    from spectral_predict.model_io import load_model, predict_with_model
    from spectral_predict.search import build_multiclass_decision_view

    app = gui_app
    rng = np.random.default_rng(3)
    blocks, labels = [], []
    for k in range(3):
        blocks.append(rng.normal(k * 5.0, 1.0, size=(40, 30)))
        labels += [f"C{k}"] * 40
    columns = ([float(j) for j in range(30)] if col_kind == "float"
               else [str(float(j)) for j in range(30)])
    X = pd.DataFrame(np.vstack(blocks), columns=columns)
    y = pd.Series(labels)
    full_cfg = {
        **cfg, "polyorder": None, "baseline_method": None, "baseline_params": None,
        "smoothing": False, "smoothing_window": 17, "smoothing_polyorder": 2,
    }
    view = build_multiclass_decision_view(
        X, y, engine="pca-simca", preprocess_cfg=full_cfg, n_components=0.99
    )
    assert not view.get("reason"), view.get("reason")

    path = tmp_path / "mc.dasp"
    app._fit_and_save_multiclass_model(view["config"], X, y, str(path))
    assert path.exists()

    loaded = load_model(str(path))
    out = predict_with_model(loaded, X, validate_wavelengths=True)
    np.testing.assert_array_equal(out["decision_matrix"], view["accept"])
    np.testing.assert_array_equal(
        np.asarray(out["summary_label"], dtype=object),
        np.asarray(view["labels"], dtype=object),
    )
