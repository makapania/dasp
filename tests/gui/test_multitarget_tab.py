"""GUI tests for the T-17 Multi-Target sub-tab.

Covers the F6-gui feature: multi-select target widget + ``selected_targets``,
JOINT/INDEPENDENT labelling, per-target + joint-Q² results grid, and the
Grid-engine lock (grey-out Bayesian/NSGA-II + force ``optimization_method='grid'``
when >1 target is selected). Single-Y UI is untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _load_multitarget_data(app, n=40, p=24, n_targets=3, seed=0):
    """Set synthetic spectral X + a numeric multi-target reference on the app."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    idx = [f"S{i}" for i in range(n)]
    cols = [f"w{j}" for j in range(p)]
    X_df = pd.DataFrame(X, index=idx, columns=cols)

    # Correlated numeric targets so JOINT modelling is meaningful.
    base = X[:, :3].sum(axis=1)
    ref = pd.DataFrame(index=idx)
    for t in range(n_targets):
        ref[f"prop_{t}"] = base * (t + 1) + rng.normal(scale=0.1, size=n)
    ref["spec_file"] = idx  # non-numeric-ish column to be excluded

    app.X = X_df
    app.X_original = X_df.copy()
    app.ref = ref
    app.combined_metadata_df = None
    app.spectral_file_column.set("spec_file")
    return X_df, ref


@pytest.mark.gui
class TestMultiTargetTab:
    def test_subtab_and_state_exist(self, gui_app):
        assert hasattr(gui_app, "selected_targets")
        assert gui_app.selected_targets == []
        assert hasattr(gui_app, "multitarget_listbox")
        assert hasattr(gui_app, "multitarget_tree")
        # Model picker carries JOINT/INDEPENDENT tags.
        assert "PLS" in gui_app.multitarget_model_vars
        assert "LightGBM" in gui_app.multitarget_model_vars

    def test_refresh_lists_numeric_targets_only(self, gui_app):
        _load_multitarget_data(gui_app)
        gui_app._refresh_multitarget_columns()
        listed = list(gui_app.multitarget_listbox.get(0, "end"))
        assert set(listed) == {"prop_0", "prop_1", "prop_2"}
        assert "spec_file" not in listed

    def test_multiselect_locks_1d_engines_and_forces_grid(self, gui_app):
        _load_multitarget_data(gui_app)
        gui_app._refresh_multitarget_columns()

        # Pretend the user had Bayesian selected.
        gui_app.optimization_method.set("unified")
        gui_app.multitarget_listbox.selection_clear(0, "end")
        gui_app.multitarget_listbox.selection_set(0, 2)  # all 3 targets
        gui_app._on_multitarget_selection_changed()

        assert len(gui_app.selected_targets) == 3
        # Grid forced; 1-D-only engines disabled.
        assert gui_app.optimization_method.get() == "grid"
        assert str(gui_app.opt_radio_unified.cget("state")) == "disabled"
        assert str(gui_app.opt_radio_nsga2.cget("state")) == "disabled"
        assert str(gui_app.opt_radio_grid.cget("state")) == "normal"

    def test_single_target_reenables_engines(self, gui_app):
        _load_multitarget_data(gui_app)
        gui_app._refresh_multitarget_columns()
        gui_app.multitarget_listbox.selection_clear(0, "end")
        gui_app.multitarget_listbox.selection_set(0)  # one target only
        gui_app._on_multitarget_selection_changed()
        assert len(gui_app.selected_targets) == 1
        assert str(gui_app.opt_radio_unified.cget("state")) == "normal"
        assert str(gui_app.opt_radio_nsga2.cget("state")) == "normal"

    def test_run_populates_joint_and_per_target_columns(self, gui_app, monkeypatch):
        _load_multitarget_data(gui_app)
        gui_app._refresh_multitarget_columns()
        gui_app.multitarget_listbox.selection_set(0, 2)
        gui_app._on_multitarget_selection_changed()

        # PLS (JOINT) + Ridge (INDEPENDENT).
        for name, var in gui_app.multitarget_model_vars.items():
            var.set(name in ("PLS", "Ridge"))

        # The full inherited grid expands to many cells (preprocessing x varsel x
        # HP grid); that expansion is the backend's concern (covered by
        # tests/test_multitarget_grid.py). Here we isolate the GUI populate path:
        # stub the backend to a controlled 2-row output (1 PLS JOINT + 1 Ridge
        # INDEPENDENT) and verify the worker->queue->poller->_populate plumbing.
        from spectral_predict.multitarget_search import (
            MultiTargetResult, MultiTargetSearchOutput,
        )

        targets = ["prop_0", "prop_1", "prop_2"]

        def _per_target():
            return [{"target": t, "r2": 0.9, "rmse": 0.1, "rpd": 3.0,
                     "rer": 5.0, "ccc": 0.85, "bias": 0.0} for t in targets]

        def _fake_grid(X, Y, **kwargs):
            return MultiTargetSearchOutput(
                results=[
                    MultiTargetResult(
                        model_name="PLS", mode="JOINT", params={}, joint_q2=0.8,
                        metrics={"per_target": _per_target(), "q2": np.array([0.8, 0.8, 0.8])},
                        precise_note="", scale_y=True, mechanism="x",
                    ),
                    MultiTargetResult(
                        model_name="Ridge", mode="INDEPENDENT", params={}, joint_q2=0.7,
                        metrics={"per_target": _per_target(), "q2": np.array([0.7, 0.7, 0.7])},
                        precise_note="independent", scale_y=False, mechanism="n separate fits",
                    ),
                ],
                target_names=targets, correlation={}, n_targets=3, skipped=[],
            )

        import spectral_predict.multitarget_grid as mg
        monkeypatch.setattr(mg, "run_multitarget_grid_search", _fake_grid)

        gui_app._run_multitarget_search()

        # The search runs on a daemon worker thread; the worker never calls
        # root.after (Tcl rejects cross-thread registration), it enqueues events
        # that a main-thread poller (scheduled via root.after) drains. Pump the
        # Tk event loop so that poller fires and _multitarget_done runs (which
        # stores _multitarget_last_output + populates the tree) before asserting.
        import time
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                gui_app.root.update()
            except Exception:
                pass
            if gui_app._multitarget_last_output is not None:
                break
            thread = getattr(gui_app, "_multitarget_thread", None)
            if thread is not None and not thread.is_alive() and gui_app._multitarget_queue.empty():
                break  # worker finished without producing output
            time.sleep(0.02)

        out = gui_app._multitarget_last_output
        assert out is not None
        assert out.n_targets == 3
        modes = {r.model_name: r.mode for r in out.results}
        assert modes["PLS"] == "JOINT"
        assert modes["Ridge"] == "INDEPENDENT"

        # Results grid has joint_q2 + per-target metric columns for each target.
        cols = list(gui_app.multitarget_tree["columns"])
        assert "joint_q2" in cols
        for t in ("prop_0", "prop_1", "prop_2"):
            for key in ("r2", "rmse", "rpd", "rer", "ccc", "bias"):
                assert f"{t}__{key}" in cols
        assert len(gui_app.multitarget_tree.get_children()) == 2

    def test_run_refuses_single_target(self, gui_app):
        _load_multitarget_data(gui_app)
        gui_app._refresh_multitarget_columns()
        gui_app.multitarget_listbox.selection_clear(0, "end")
        gui_app.multitarget_listbox.selection_set(0)
        gui_app._on_multitarget_selection_changed()
        gui_app._multitarget_last_output = None
        gui_app._run_multitarget_search()  # should warn + no-op (dialogs suppressed)
        assert gui_app._multitarget_last_output is None


@pytest.mark.gui
class TestMultiTargetGridDispatch:
    def test_dispatch_passes_inherited_config_and_uses_separate_controller(self, gui_app, monkeypatch):
        _load_multitarget_data(gui_app)
        gui_app._refresh_multitarget_columns()
        gui_app.multitarget_listbox.selection_clear(0, "end")
        gui_app.multitarget_listbox.selection_set(0, 2)
        gui_app._on_multitarget_selection_changed()
        gui_app.multitarget_model_vars["PLS"].set(True)

        captured = {}

        def _fake_grid(X, Y, **kwargs):
            captured.update(kwargs)
            from spectral_predict.multitarget_search import MultiTargetSearchOutput
            return MultiTargetSearchOutput(results=[], target_names=kwargs["target_names"],
                                           correlation={}, n_targets=Y.shape[1], skipped=["uve"])

        import spectral_predict.multitarget_grid as mg
        monkeypatch.setattr(mg, "run_multitarget_grid_search", _fake_grid)
        # Run the worker body synchronously for the test.
        gui_app._run_multitarget_search_thread(
            gui_app._collect_multitarget_config()
        )
        assert "PLS" in captured["model_names"]
        assert "preprocessing_methods" in captured
        assert "model_grid_overrides" in captured
        assert captured["optimization_method"] == "grid"
        # A SEPARATE controller instance is used, not the single-Y one.
        assert gui_app._multitarget_controller is not gui_app.search_controller

    def test_cancel_stops_multitarget_controller(self, gui_app):
        from spectral_predict.search_controller import SearchController
        gui_app._multitarget_controller = SearchController()
        gui_app._cancel_multitarget_search()
        assert gui_app._multitarget_controller.is_ended() or gui_app._multitarget_controller._stop_requested


@pytest.mark.gui
class TestMultiTargetPlsTolParity:
    """FIX 4: multi-target ``_collect_pls_overrides`` must read ``pls_tol_1e5``
    (single-Y path does at :27954) and match the single-Y empty-fallback [1e-6]."""

    def _reset_tol_vars(self, app):
        for attr, val in (("pls_tol_1e7", False), ("pls_tol_1e6", False),
                          ("pls_tol_1e5", False)):
            getattr(app, attr).set(val)
        app.pls_tol_custom.set("")

    def test_tol_1e5_reaches_multitarget_grid(self, gui_app):
        self._reset_tol_vars(gui_app)
        gui_app.pls_tol_1e5.set(True)  # only the 1e-5 box checked
        try:
            out = gui_app._collect_pls_overrides()
        finally:
            self._reset_tol_vars(gui_app)
            gui_app.pls_tol_1e6.set(True)  # restore default
        tols = out["pls_tol_list"]
        assert tols is not None, "1e-5 selection was dropped (pls_tol_list is None)"
        assert 1e-5 in tols

    def test_empty_selection_falls_back_to_1e6(self, gui_app):
        self._reset_tol_vars(gui_app)  # nothing checked, no custom
        try:
            out = gui_app._collect_pls_overrides()
        finally:
            self._reset_tol_vars(gui_app)
            gui_app.pls_tol_1e6.set(True)  # restore default
        # Single-Y path defaults to [1e-6] when no tol box is checked (:27970).
        assert out["pls_tol_list"] == [1e-6]


@pytest.mark.gui
def test_leaderboard_shows_preprocess_varsel_nvars_columns(gui_app):
    from spectral_predict.multitarget_search import MultiTargetResult, MultiTargetSearchOutput

    res = MultiTargetResult(
        model_name="PLS", mode="JOINT", params={"n_components": 5}, joint_q2=0.8,
        metrics={"per_target": [{"target": "a", "r2": 0.8, "rmse": 1.0, "rpd": 2.0,
                                 "rer": 3.0, "ccc": 0.9, "bias": 0.0}],
                 "q2": np.array([0.8])},
        precise_note="", scale_y=True, mechanism="x",
        preprocessing="snv", varsel_method="ipls_forward", varsel_tag="fwd", n_variables=25,
    )
    out = MultiTargetSearchOutput(results=[res], target_names=["a"], correlation={},
                                  n_targets=1, skipped=["uve"])
    gui_app._populate_multitarget_results(out)
    cols = list(gui_app.multitarget_tree["columns"])
    assert "preprocessing" in cols
    assert "varsel" in cols
    assert "nvars" in cols
    row = gui_app.multitarget_tree.get_children()[0]
    values = gui_app.multitarget_tree.item(row, "values")
    assert "snv" in values
    assert "ipls_forward" in values
    assert "25" in values
