"""Multi-class SIMCA sweep parity tests (T-31 Task 6).

Verify the multi-class alpha / n_components sweep state vars + collectors
mirror the one-class SIMCA sweep (_collect_simca_overrides). Skips gracefully
when no display is available (headless CI without Tk).
"""

from __future__ import annotations

import pytest

tk = pytest.importorskip("tkinter")

from spectral_predict_gui_optimized import SpectralPredictApp


@pytest.fixture(scope="module")
def _mc_app():
    # A single Tk root for the whole module: creating a fresh Tk() per test
    # exhausts Tkinter menu resources on Windows and the later tk.Tk() call
    # raises TclError (mirrors the shared session_app fixture in conftest).
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    a = SpectralPredictApp(root)
    yield a
    try:
        root.destroy()
    except tk.TclError:
        pass


@pytest.fixture
def app(_mc_app):
    # Reset the sweep vars to their construction defaults so tests don't bleed.
    _mc_app.mc_alpha_001.set(False)
    _mc_app.mc_alpha_005.set(True)
    _mc_app.mc_alpha_custom.set("")
    _mc_app.mc_ncomp_3.set(False)
    _mc_app.mc_ncomp_5.set(False)
    _mc_app.mc_ncomp_7.set(False)
    _mc_app.mc_ncomp_095.set(False)
    _mc_app.mc_ncomp_099.set(True)
    _mc_app.mc_ncomp_custom.set("")
    _mc_app.mc_ncomp_per_class_cv.set(False)
    return _mc_app


def test_mc_sweep_collectors_return_defaults(app):
    app.mc_ncomp_099.set(True)
    assert app._collect_mc_ncomp_list() == [0.99]
    app.mc_alpha_005.set(True)
    assert app._collect_mc_alpha_list() == [0.05]


def test_mc_ncomp_collector_mixes_int_and_fraction(app):
    app.mc_ncomp_5.set(True)
    app.mc_ncomp_095.set(True)
    got = sorted(app._collect_mc_ncomp_list(), key=lambda x: (isinstance(x, float), x))
    assert 5 in got and 0.95 in got


def test_mc_alpha_collector_falls_back_to_default(app):
    app.mc_alpha_001.set(False)
    app.mc_alpha_005.set(False)
    app.mc_alpha_custom.set("")
    assert app._collect_mc_alpha_list() == [0.05]


def test_mc_ncomp_per_class_cv_toggle(app):
    app.mc_ncomp_099.set(False)
    app.mc_ncomp_per_class_cv.set(True)
    assert app._collect_mc_ncomp_list() == ["per_class_cv"]


# ---------------------------------------------------------------------------
# Task 7 — hyperparameters relocated from Import page to 4A Model Config
# ---------------------------------------------------------------------------

def test_import_page_has_no_mc_panel(app):
    # The old import-page frame must be gone.
    assert not hasattr(app, "mc_hyperparams_frame")


def test_mc_model_config_card_exists(app):
    assert hasattr(app, "mc_model_config_frame")


def test_task_type_toggles_mc_card(app):
    app.task_type.set("multiclass_simca")
    app._on_task_type_changed()
    assert app.mc_model_config_frame.winfo_manager() != ""   # mapped
    app.task_type.set("regression")
    app._on_task_type_changed()
    assert app.mc_model_config_frame.winfo_manager() == ""    # unmapped


# ---------------------------------------------------------------------------
# Task 8 — multi-class variable selection in tab 4B (grouped set + Top-N reuse)
# ---------------------------------------------------------------------------

def test_mc_varsel_card_visible_and_grouped(app):
    app.task_type.set("multiclass_simca")
    app._on_task_type_changed()
    # The Advanced Variable Selection card stays visible in multiclass mode.
    assert app.varsel_card_outer.winfo_manager() != ""
    # Grouped multiclass picker exists and is shown; standard frame hidden.
    assert hasattr(app, "mc_varsel_group_frame")
    assert app.mc_varsel_group_frame.winfo_manager() != ""
    assert app.varsel_frame.winfo_manager() == ""
    # Offered method keys present (corrected backend-resolvable set).
    for k in ("importance", "cars", "spa", "uve", "wold_modeling"):
        assert k in app.mc_varsel_vars
    # Excluded no-op / redundant methods must NOT be offered.
    for k in ("cars_tree", "uve_cars_tree", "vcpa", "vcpa-iriv", "ga"):
        assert k not in app.mc_varsel_vars
    # Two labeled groups exist.
    assert set(app.mc_varsel_groups) == {
        "SIMCA-native (novelty-safe)",
        "Discrimination-based (confirm novelty on a true external class)",
    }
    # Restore standard frame when leaving multiclass mode.
    app.task_type.set("regression")
    app._on_task_type_changed()
    assert app.mc_varsel_group_frame.winfo_manager() == ""
    assert app.varsel_frame.winfo_manager() != ""


def test_mc_reuses_topn_counts(app):
    # The shared Top-N vars exist and are the size-sweep source for multiclass.
    for v in ("var_10", "var_50", "var_100"):
        assert hasattr(app, v)
