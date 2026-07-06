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


# ---------------------------------------------------------------------------
# Task 9 — swept sizes + varsel-method collectors feed the search call
# ---------------------------------------------------------------------------

def test_mc_sizes_collector_returns_checked(app):
    for v in ("var_10", "var_20", "var_50", "var_100",
              "var_250", "var_500", "var_1000"):
        getattr(app, v).set(False)
    app.var_20.set(True)
    app.var_250.set(True)
    assert app._collect_mc_sizes() == [20, 250]


def test_mc_sizes_collector_falls_back_to_default(app):
    for v in ("var_10", "var_20", "var_50", "var_100",
              "var_250", "var_500", "var_1000"):
        getattr(app, v).set(False)
    assert app._collect_mc_sizes() == [100]


def test_mc_varsel_paths_collector_returns_checked(app):
    for v in app.mc_varsel_vars.values():
        v.set(False)
    app.mc_varsel_vars["importance"].set(True)
    assert app._collect_mc_varsel_paths() == ["importance"]


def test_mc_varsel_paths_collector_falls_back_to_none(app):
    for v in app.mc_varsel_vars.values():
        v.set(False)
    assert app._collect_mc_varsel_paths() == ["none"]


# ---------------------------------------------------------------------------
# Task 10 — a completed multi-class run populates the leaderboard only; it
# does NOT auto-open the decision-view window (parity with other methods).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Task 11 — SPXY holdout selection is disallowed for multi-class class modeling
# (its d_y distance term is undefined for a categorical class label).
# ---------------------------------------------------------------------------

def test_spxy_disabled_for_multiclass(app):
    app.task_type.set("multiclass_simca")
    app._on_task_type_changed()
    assert app._validation_algo_allowed("SPXY") is False
    assert app._validation_algo_allowed("Kennard-Stone") is True
    assert app._validation_algo_allowed("Random") is True
    # Non-multiclass task types allow SPXY again.
    app.task_type.set("regression")
    app._on_task_type_changed()
    assert app._validation_algo_allowed("SPXY") is True


def test_task_type_toggle_resets_spxy_selection(app):
    # A stale SPXY selection is reset to Kennard-Stone when switching to
    # multiclass, and the SPXY radio is disabled; leaving multiclass re-enables.
    app.task_type.set("regression")
    app._on_task_type_changed()
    app.validation_algorithm.set("SPXY")
    app.task_type.set("multiclass_simca")
    app._on_task_type_changed()
    assert app.validation_algorithm.get() == "Kennard-Stone"
    assert "disabled" in app.validation_spxy_radio.state()
    app.task_type.set("regression")
    app._on_task_type_changed()
    assert "disabled" not in app.validation_spxy_radio.state()


def test_decision_view_header_states_config(app):
    # The one-line config header must name the engine, alpha, per-class PCA
    # size, and variable-selection path + size so the decision view is
    # self-describing. .get() tolerates a dict or a pandas row.
    header = app._multiclass_decision_header({
        "engine_family": "pca-simca",
        "Alpha": 0.05,
        "NComponents": 0.99,
        "varsel_path": "importance",
        "n_vars": 50,
    })
    assert "pca-simca" in header
    assert "0.05" in header
    assert "0.99" in header
    assert "importance" in header
    assert "50" in header


def test_decision_view_header_tolerates_missing_keys(app):
    # A missing key must not raise — the label renders with whatever is present.
    header = app._multiclass_decision_header({"engine_family": "ocsvm"})
    assert "ocsvm" in header


def test_no_auto_decision_popup(app, tmp_path):
    import pandas as pd

    called = {"n": 0}
    app._show_multiclass_decision_view = lambda *a, **k: called.__setitem__(
        "n", called["n"] + 1
    )
    # Stub the populate so flushing the Tk event queue below can't crash on the
    # minimal leaderboard; we only care that the popup opener is never invoked.
    app._populate_results_table = lambda *a, **k: None

    # A valid decision view (no 'reason') — the case that USED to auto-open.
    app._mc_decision_view = {"decision_matrix": [], "classes": []}
    # Minimal leaderboard so _finalize_multiclass_run_ui runs its full body.
    app.results_df = pd.DataFrame({"Select": [False], "rank": [1]})
    app.output_dir.set(str(tmp_path))
    app.target_column.set("class")

    app._finalize_multiclass_run_ui()
    # The old auto-open scheduled the window via root.after(0, ...); flush the
    # Tk event queue so any such callback would actually fire and be counted.
    app.root.update()

    assert called["n"] == 0
    # The silent CSV export is a file artifact, not a popup — it must still run.
    assert list(tmp_path.glob("multiclass_results_*.csv"))
