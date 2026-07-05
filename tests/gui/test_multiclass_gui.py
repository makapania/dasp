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
