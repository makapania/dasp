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
