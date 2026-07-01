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

    def test_run_populates_joint_and_per_target_columns(self, gui_app):
        _load_multitarget_data(gui_app)
        gui_app._refresh_multitarget_columns()
        gui_app.multitarget_listbox.selection_set(0, 2)
        gui_app._on_multitarget_selection_changed()

        # PLS (JOINT) + Ridge (INDEPENDENT).
        for name, var in gui_app.multitarget_model_vars.items():
            var.set(name in ("PLS", "Ridge"))

        gui_app._run_multitarget_search()

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
