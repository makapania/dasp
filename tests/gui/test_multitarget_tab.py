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

    def test_inner_subnotebook_has_three_tabs(self, gui_app):
        """T-17 W1-1: the multi-target tab is restructured into an inner
        ttk.Notebook with Setup / Progress / Results sub-tabs."""
        assert hasattr(gui_app, "multitarget_subnotebook"), \
            "inner sub-notebook attribute missing"
        nb = gui_app.multitarget_subnotebook
        tabs = [nb.tab(i, "text") for i in nb.tabs()]
        # The three sub-tabs exist; whitespace styling differences tolerated.
        assert any("Setup" in t for t in tabs), f"Setup sub-tab missing: {tabs}"
        assert any("Progress" in t for t in tabs), f"Progress sub-tab missing: {tabs}"
        assert any("Results" in t for t in tabs), f"Results sub-tab missing: {tabs}"

    def test_progress_surface_widgets_exist(self, gui_app):
        """T-17 W1-1: the Progress sub-tab owns a determinate bar + info +
        ETA + best-model + capped log + Pause/Resume/Stop buttons."""
        for attr in (
            "multitarget_progress_bar", "multitarget_progress_info",
            "multitarget_time_estimate", "multitarget_best_model_info",
            "multitarget_progress_log", "multitarget_pause_btn",
            "multitarget_resume_btn", "multitarget_stop_btn",
            "multitarget_analysis_start_time",
        ):
            assert hasattr(gui_app, attr), f"missing progress widget: {attr}"
        # Pause/Resume/Start disabled at idle (mirrors single-Y state machine).
        assert str(gui_app.multitarget_pause_btn.cget("state")) == "disabled"
        assert str(gui_app.multitarget_resume_btn.cget("state")) == "disabled"
        assert str(gui_app.multitarget_stop_btn.cget("state")) == "disabled"

    def test_refresh_lists_numeric_targets_only(self, gui_app):
        _load_multitarget_data(gui_app)
        gui_app._refresh_multitarget_columns()
        listed = list(gui_app.multitarget_listbox.get(0, "end"))
        assert set(listed) == {"prop_0", "prop_1", "prop_2"}
        assert "spec_file" not in listed

    def test_leaderboard_column_sort(self, gui_app):
        """Clicking a header sorts the multi-target leaderboard (numeric + toggle)."""
        from types import SimpleNamespace

        def _res(model, jq2):
            return SimpleNamespace(
                preprocessing="SNV", varsel_method="uve", n_variables=10,
                params={"n_components": 5}, model_name=model, mode="INDEPENDENT",
                joint_q2=jq2,
                metrics={"per_target": [{"target": "prop_0", "r2": jq2,
                                         "rmse": 1.0, "rpd": 2.0, "rer": 3.0}]},
            )

        # Deliberately out-of-order joint_q2 so a correct sort is observable.
        output = SimpleNamespace(
            target_names=["prop_0"],
            results=[_res("PLS", 0.30), _res("Ridge", 0.90), _res("LightGBM", 0.60)],
        )
        gui_app._populate_multitarget_results(output)
        tree = gui_app.multitarget_tree

        def _col(column):
            return [tree.set(iid, column) for iid in tree.get_children("")]

        # Ascending on Joint Q² (numeric).
        gui_app._sort_multitarget_tree("joint_q2")
        assert _col("joint_q2") == ["0.3000", "0.6000", "0.9000"]
        # Same header again toggles to descending.
        gui_app._sort_multitarget_tree("joint_q2")
        assert _col("joint_q2") == ["0.9000", "0.6000", "0.3000"]
        # Text column sorts case-insensitively.
        gui_app._sort_multitarget_tree("model")
        assert _col("model") == ["LightGBM", "PLS", "Ridge"]

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

    def test_collect_config_forwards_uve_params(self, gui_app):
        _load_multitarget_data(gui_app)
        gui_app._refresh_multitarget_columns()
        gui_app.multitarget_listbox.selection_clear(0, "end")
        gui_app.multitarget_listbox.selection_set(0, 2)
        gui_app._on_multitarget_selection_changed()
        gui_app.multitarget_model_vars["PLS"].set(True)
        try:
            gui_app.uve_cutoff_multiplier.set(1.3)
            gui_app.uve_n_components.set("7")
            cfg = gui_app._collect_multitarget_config()
            assert cfg is not None
            assert cfg["uve_cutoff_multiplier"] == 1.3
            assert cfg["uve_n_components"] == 7
            gui_app.uve_n_components.set("")
            cfg2 = gui_app._collect_multitarget_config()
            assert cfg2 is not None
            assert cfg2["uve_n_components"] is None
        finally:
            gui_app.uve_cutoff_multiplier.set(1.0)
            gui_app.uve_n_components.set("")

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


def _pump_until(app, predicate, timeout=10.0):
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            app.root.update()
        except Exception:
            pass
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.mark.gui
class TestMultiTargetRunLifecycle:
    """FIX 1: guard concurrent runs (reject 2nd + disable Run button), track
    ``_multitarget_after_id``, and use a FRESH per-run queue so a prior run's
    stale terminal message can't bleed into a new run."""

    def _prime(self, app):
        _load_multitarget_data(app)
        app._refresh_multitarget_columns()
        app.multitarget_listbox.selection_clear(0, "end")
        app.multitarget_listbox.selection_set(0, 2)
        app._on_multitarget_selection_changed()
        for name, var in app.multitarget_model_vars.items():
            var.set(name == "PLS")
        app._multitarget_last_output = None

    def _blocking_backend(self, monkeypatch, started, release):
        def _grid(X, Y, **kwargs):
            started.set()
            release.wait(10)
            from spectral_predict.multitarget_search import MultiTargetSearchOutput
            return MultiTargetSearchOutput(
                results=[], target_names=kwargs["target_names"],
                correlation={}, n_targets=Y.shape[1], skipped=[])
        import spectral_predict.multitarget_grid as mg
        monkeypatch.setattr(mg, "run_multitarget_grid_search", _grid)

    def test_second_run_rejected_and_button_disabled(self, gui_app, monkeypatch):
        import threading
        self._prime(gui_app)
        started, release = threading.Event(), threading.Event()
        self._blocking_backend(monkeypatch, started, release)
        thread_a = None
        try:
            gui_app._run_multitarget_search()  # run A
            assert started.wait(5), "worker A never started"
            thread_a = gui_app._multitarget_thread
            assert thread_a.is_alive()

            # Second start while A is alive must be rejected — no new worker.
            gui_app._run_multitarget_search()
            assert gui_app._multitarget_thread is thread_a, "a second worker was started"

            # Run button disabled while active; poll after-id tracked.
            assert str(gui_app.multitarget_run_button.cget("state")) == "disabled"
            assert getattr(gui_app, "_multitarget_after_id", None) is not None
        finally:
            release.set()
            if thread_a is not None:
                _pump_until(gui_app, lambda: not thread_a.is_alive()
                            and gui_app._multitarget_queue.empty())
        # Poller runs _multitarget_done on the main thread and re-enables Run.
        _pump_until(gui_app, lambda: gui_app._multitarget_last_output is not None)
        assert str(gui_app.multitarget_run_button.cget("state")) == "normal"

    def test_fresh_queue_per_run(self, gui_app, monkeypatch):
        import threading
        self._prime(gui_app)
        started, release = threading.Event(), threading.Event()
        self._blocking_backend(monkeypatch, started, release)
        stale = gui_app._multitarget_queue
        stale.put(("done", "STALE"))  # leftover from a hypothetical prior run
        thread_a = None
        try:
            gui_app._run_multitarget_search()
            assert started.wait(5)
            thread_a = gui_app._multitarget_thread
            # A fresh queue is installed per run, so the stale message can never
            # be drained into this run.
            assert gui_app._multitarget_queue is not stale
        finally:
            release.set()
            if thread_a is not None:
                _pump_until(gui_app, lambda: not thread_a.is_alive()
                            and gui_app._multitarget_queue.empty())
        _pump_until(gui_app, lambda: gui_app._multitarget_last_output is not None)
        # The stale 'done' never overwrote the real run's output.
        assert gui_app._multitarget_last_output != "STALE"


@pytest.mark.gui
class TestMultiTargetTerminalRace:
    """FIX 2: the poller must not stop until a terminal ('done'/'failed') event
    has actually been handled. Otherwise a final tuple enqueued between the
    empty-check and the stop is dropped and the UI stays stuck on 'Running…'."""

    def _dead_thread(self):
        import threading
        t = threading.Thread(target=lambda: None)
        t.start()
        t.join()
        return t

    def _rescheduled(self, spy, app):
        # NB: bound methods create a fresh wrapper per attribute access, so
        # compare by == (same __self__ + __func__), never `is`.
        return any(len(c.args) >= 2 and c.args[1] == app._poll_multitarget_queue
                   for c in spy.call_args_list)

    def test_poller_reschedules_until_terminal_seen(self, gui_app, monkeypatch):
        import queue as _q
        from unittest import mock
        # Worker already dead + queue currently empty, but NO terminal event has
        # been handled yet — models the race where the worker set is_alive()->False
        # and enqueued 'done' just after this drain. The poller must keep polling.
        gui_app._multitarget_thread = self._dead_thread()
        gui_app._multitarget_queue = _q.Queue()
        gui_app._multitarget_terminal_seen = False
        spy = mock.Mock(return_value="afterid")
        monkeypatch.setattr(gui_app.root, "after", spy)
        gui_app._poll_multitarget_queue()
        assert self._rescheduled(spy, gui_app), \
            "poller stopped before any terminal event was handled (dropped-'done' race)"

    def test_poller_stops_once_terminal_seen(self, gui_app, monkeypatch):
        import queue as _q
        from unittest import mock
        gui_app._multitarget_thread = self._dead_thread()
        gui_app._multitarget_queue = _q.Queue()
        gui_app._multitarget_terminal_seen = True  # terminal already handled
        spy = mock.Mock(return_value="afterid")
        monkeypatch.setattr(gui_app.root, "after", spy)
        gui_app._poll_multitarget_queue()
        assert not self._rescheduled(spy, gui_app), \
            "poller kept rescheduling after the run had terminated cleanly"


@pytest.mark.gui
class TestMultiTargetPollerErrorSurfacing:
    """FIX 3: the poller must not silently swallow an exception raised by a
    terminal handler (_multitarget_done/_failed) — it should surface it."""

    def test_terminal_handler_error_is_surfaced(self, gui_app, monkeypatch):
        import queue as _q
        gui_app._multitarget_thread = None
        gui_app._multitarget_queue = _q.Queue()
        gui_app._multitarget_terminal_seen = False

        def _boom(output):
            raise RuntimeError("populate blew up")

        monkeypatch.setattr(gui_app, "_populate_multitarget_results", _boom)
        gui_app.multitarget_status_label.config(text="Running…")
        gui_app._multitarget_queue.put(("done", object()))
        gui_app._poll_multitarget_queue()  # messagebox.showerror suppressed by fixture
        assert "Fail" in gui_app.multitarget_status_label.cget("text"), \
            "terminal-handler error was silently swallowed (status stuck)"


@pytest.mark.gui
class TestMultiTargetProgressSurface:
    """T-17 W1-1 / W1-3: the poller now consumes the full progress payload
    (current/total/best_model) into the Progress sub-tab widgets instead of
    discarding everything but the message string. Mirrors single-Y
    ``_progress_callback_impl`` / ``_log_progress``."""

    def test_progress_payload_updates_bar_info_and_best_model(self, gui_app):
        """A fake ``('progress', payload)`` tuple drives the bar + info +
        best-model line — the load-bearing fix for ``_poll_multitarget_queue``
        (was: discarded everything but ``message``)."""
        from datetime import datetime, timedelta
        gui_app._multitarget_analysis_start_time = datetime.now() - timedelta(seconds=10)
        payload = {
            "message": "Multi-target cell 3/10",
            "current": 3, "total": 10,
            "best_model": {
                "Model": "PLS", "Preprocess": "snv", "Deriv": None,
                "RMSEcv": 0.12, "R2cv": 0.87, "top_vars": "42",
            },
        }
        gui_app._multitarget_queue.put(("progress", payload))
        gui_app._poll_multitarget_queue()

        # Status label still shows the message.
        assert gui_app.multitarget_status_label.cget("text") == "Multi-target cell 3/10"
        # Bar + info now reflect the new payload.
        assert float(gui_app.multitarget_progress_bar.cget("value")) == 3.0
        assert float(gui_app.multitarget_progress_bar.cget("maximum")) == 10.0
        assert "3/10" in gui_app.multitarget_progress_info.cget("text")
        # Best-model line shows PLS + snv + the CV metrics.
        best_txt = gui_app.multitarget_best_model_info.cget("text")
        assert "PLS" in best_txt and "snv" in best_txt
        assert "0.1200" in best_txt and "0.8700" in best_txt
        # ETA label is populated (start time was set).
        assert gui_app.multitarget_time_estimate.cget("text") != ""

    def test_progress_payload_zero_total_does_not_divide_by_zero(self, gui_app):
        """A 0/0 payload (e.g. the W1-3 ``[LOG] Run log:`` line emitted before
        any cell runs) must not crash the poller or stomp the bar."""
        # Reset the best-model line (session-scoped app may carry state from
        # earlier tests).
        gui_app.multitarget_best_model_info.config(text="(none yet)")
        gui_app._multitarget_queue.put(("progress", {
            "message": "[LOG] Run log: /tmp/x.log", "current": 0, "total": 0,
            "best_model": None,
        }))
        gui_app._poll_multitarget_queue()  # must not raise
        assert "[LOG]" in gui_app.multitarget_status_label.cget("text")
        # Best-model line untouched (None payload).
        assert gui_app.multitarget_best_model_info.cget("text") == "(none yet)"

    def test_progress_log_appends_and_caps(self, gui_app):
        """``_multitarget_log_progress`` writes to the log Text widget AND caps
        at 2000 lines (mirrors single-Y ``_append_progress``)."""
        gui_app.multitarget_progress_log.delete("1.0", "end")
        for i in range(50):
            gui_app._multitarget_log_progress(f"line {i}")
        contents = gui_app.multitarget_progress_log.get("1.0", "end")
        assert "line 0" in contents and "line 49" in contents
        # Cap: force well past 2000 lines and confirm the head is dropped.
        gui_app.multitarget_progress_log.delete("1.0", "end")
        for i in range(2500):
            gui_app._multitarget_log_progress(f"line {i}")
        # The oldest lines must have been evicted; only the tail survives.
        surviving = gui_app.multitarget_progress_log.get("1.0", "end-1c").splitlines()
        assert len(surviving) <= 2001  # 2000 lines + trailing-empty from index()
        assert "line 0" not in surviving
        assert "line 2499" in surviving


@pytest.mark.gui
class TestMultiTargetPauseResumeWiring:
    """T-17 W1-1: Pause/Resume/Stop buttons drive the multi-target controller
    ONLY — never ``self.search_controller`` (zero-regression guard)."""

    def test_pause_button_drives_multitarget_controller_only(self, gui_app):
        from spectral_predict.search_controller import SearchController
        gui_app._multitarget_controller = SearchController()
        # Provide a real single-Y controller sentinel so we can prove it's
        # untouched by the multi-target pause path (zero-regression guard).
        single_y_sentinel = SearchController()
        original_single_y = gui_app.search_controller
        gui_app.search_controller = single_y_sentinel
        try:
            gui_app._pause_multitarget_search()
            # Multi-target controller is paused.
            assert gui_app._multitarget_controller.is_paused
            # Single-Y controller is untouched (zero-regression guard).
            assert not single_y_sentinel.is_paused
            # Button state machine: pausing → both Pause and Resume disabled.
            assert str(gui_app.multitarget_pause_btn.cget("state")) == "disabled"
            assert str(gui_app.multitarget_resume_btn.cget("state")) == "disabled"
            assert str(gui_app.multitarget_stop_btn.cget("state")) == "normal"
        finally:
            gui_app.search_controller = original_single_y

    def test_resume_button_drives_multitarget_controller_only(self, gui_app):
        from spectral_predict.search_controller import SearchController
        import threading
        # A LIVE worker thread is required for resume to actually flip state —
        # use a blocking thread so it's still alive when resume is called.
        release = threading.Event()
        def _block():
            release.wait(5)
        t = threading.Thread(target=_block)
        t.start()
        gui_app._multitarget_thread = t
        try:
            gui_app._multitarget_controller = SearchController()
            gui_app._multitarget_controller.pause()
            assert gui_app._multitarget_controller.is_paused
            gui_app._resume_multitarget_search()
            # Multi-target controller resumed.
            assert not gui_app._multitarget_controller.is_paused
            # Button state machine: running → Pause enabled, Resume disabled.
            assert str(gui_app.multitarget_pause_btn.cget("state")) == "normal"
            assert str(gui_app.multitarget_resume_btn.cget("state")) == "disabled"
        finally:
            release.set()
            t.join(timeout=2)

    def test_stop_button_drives_multitarget_controller_only(self, gui_app):
        from spectral_predict.search_controller import SearchController
        single_y_ctrl = SearchController()
        gui_app._multitarget_controller = SearchController()
        # Stash a sentinel single-Y controller so we can prove it's untouched.
        original_single_y = gui_app.search_controller
        gui_app.search_controller = single_y_ctrl
        try:
            gui_app._cancel_multitarget_search()
            assert gui_app._multitarget_controller.is_ended()
            # The single-Y controller is COMPLETELY unaffected.
            assert not single_y_ctrl.is_ended()
            assert not single_y_ctrl.is_paused
        finally:
            gui_app.search_controller = original_single_y

    def test_update_multitarget_buttons_state_machine(self, gui_app):
        """All four states of ``_update_multitarget_buttons`` produce the
        expected enabled/disabled trio (mirrors single-Y
        ``_update_search_buttons``)."""
        for state, expected in {
            "idle":     ("disabled", "disabled", "disabled"),
            "running":  ("normal",   "disabled", "normal"),
            "pausing":  ("disabled", "disabled", "normal"),
            "paused":   ("disabled", "normal",   "normal"),
        }.items():
            gui_app._update_multitarget_buttons(state)
            got = (
                str(gui_app.multitarget_pause_btn.cget("state")),
                str(gui_app.multitarget_resume_btn.cget("state")),
                str(gui_app.multitarget_stop_btn.cget("state")),
            )
            assert got == expected, f"state={state!r} got={got!r} expected={expected!r}"


@pytest.mark.gui
def test_multitarget_disk_log_mirror_invokes_log_event(gui_app, monkeypatch):
    """T-17 W1-3: ``_multitarget_log_progress`` mirrors to disk via
    ``run_logging.log_event`` so logs survive process death (matching the
    single-Y ``_log_progress`` disk mirror)."""
    import spectral_predict.run_logging as rl
    calls = []
    monkeypatch.setattr(rl, "log_event", lambda msg: calls.append(msg))
    gui_app._multitarget_log_progress("disk-mirror test line")
    assert "disk-mirror test line" in calls, "log_event was not invoked"


@pytest.mark.gui
def test_effective_n_notice_reports_incomplete_row_drop(gui_app):
    """FIX 4: complete-case CV across all selected targets silently shrinks N when
    one target is missing on some rows. The config must carry an effective-N notice
    that reports the correct dropped count."""
    _load_multitarget_data(gui_app, n=40, n_targets=2)
    # Inject NaNs into ONE target on 5 rows — those rows are incomplete-case.
    col = gui_app.ref.columns.get_loc("prop_1")
    gui_app.ref.iloc[:5, col] = np.nan
    gui_app._refresh_multitarget_columns()
    gui_app.multitarget_listbox.selection_clear(0, "end")
    gui_app.multitarget_listbox.selection_set(0, 1)  # prop_0 + prop_1
    gui_app._on_multitarget_selection_changed()
    gui_app.multitarget_model_vars["PLS"].set(True)
    try:
        cfg = gui_app._collect_multitarget_config()
        assert cfg is not None
        notice = cfg["effective_n_notice"]
        assert "5 dropped" in notice, f"wrong drop count in notice: {notice!r}"
        # 40 rows - 5 incomplete = 35 complete cases actually used.
        assert cfg["Y"].shape[0] == 35
    finally:
        gui_app.multitarget_listbox.selection_clear(0, "end")
        gui_app.multitarget_listbox.selection_set(0)
        gui_app._on_multitarget_selection_changed()


@pytest.mark.gui
class TestMultiTargetValidationLock:
    """FIX 5: SPXY + Stratified partitioning key off ONE target's y distribution,
    so they are single-Y ONLY. When >1 target is selected both radios grey out and
    the algorithm is forced to Kennard-Stone; dropping back to <=1 restores the
    user's prior choice (validation_algorithm is SHARED with the single-Y path)."""

    def _restore(self, app):
        app.multitarget_listbox.selection_clear(0, "end")
        app.multitarget_listbox.selection_set(0)
        app._on_multitarget_selection_changed()
        app._pre_multitarget_val_algo = None
        app.validation_algorithm.set("SPXY")

    def test_multiselect_disables_radios_and_forces_kennard_stone(self, gui_app):
        _load_multitarget_data(gui_app)
        gui_app._refresh_multitarget_columns()
        gui_app._pre_multitarget_val_algo = None
        gui_app.validation_algorithm.set("SPXY")
        gui_app.multitarget_listbox.selection_clear(0, "end")
        gui_app.multitarget_listbox.selection_set(0, 2)  # all 3 targets
        gui_app._on_multitarget_selection_changed()
        try:
            assert gui_app.validation_algorithm.get() == "Kennard-Stone"
            assert str(gui_app._val_radio_spxy.cget("state")) == "disabled"
            assert str(gui_app._val_radio_stratified.cget("state")) == "disabled"
        finally:
            self._restore(gui_app)

    def test_single_target_restores_prior_algo_and_reenables(self, gui_app):
        _load_multitarget_data(gui_app)
        gui_app._refresh_multitarget_columns()
        gui_app._pre_multitarget_val_algo = None
        gui_app.validation_algorithm.set("Stratified")
        gui_app.multitarget_listbox.selection_clear(0, "end")
        gui_app.multitarget_listbox.selection_set(0, 2)
        gui_app._on_multitarget_selection_changed()
        assert gui_app.validation_algorithm.get() == "Kennard-Stone"  # saved + forced
        # Drop to a single target — the prior algorithm is restored, radios re-enabled.
        gui_app.multitarget_listbox.selection_clear(0, "end")
        gui_app.multitarget_listbox.selection_set(0)
        gui_app._on_multitarget_selection_changed()
        try:
            assert gui_app.validation_algorithm.get() == "Stratified"
            assert str(gui_app._val_radio_spxy.cget("state")) == "normal"
            assert str(gui_app._val_radio_stratified.cget("state")) == "normal"
        finally:
            self._restore(gui_app)


@pytest.mark.gui
class TestMultiTargetCouplingSelector:
    """Coupling-mode selector (Independent / Joint / Both), default Independent,
    forwarded to the grid backend as ``coupling_mode``."""

    def test_coupling_state_and_default(self, gui_app):
        assert hasattr(gui_app, "multitarget_coupling")
        assert gui_app.multitarget_coupling.get() == "independent"

    def test_collect_config_forwards_coupling_mode(self, gui_app):
        _load_multitarget_data(gui_app)
        gui_app._refresh_multitarget_columns()
        gui_app.multitarget_listbox.selection_clear(0, "end")
        gui_app.multitarget_listbox.selection_set(0, 2)
        gui_app._on_multitarget_selection_changed()
        gui_app.multitarget_model_vars["PLS"].set(True)
        try:
            cfg = gui_app._collect_multitarget_config()
            assert cfg is not None
            assert cfg["coupling_mode"] == "independent"
            gui_app.multitarget_coupling.set("both")
            cfg2 = gui_app._collect_multitarget_config()
            assert cfg2["coupling_mode"] == "both"
        finally:
            gui_app.multitarget_coupling.set("independent")

    def test_coupling_mode_reaches_backend_kwargs(self, gui_app, monkeypatch):
        _load_multitarget_data(gui_app)
        gui_app._refresh_multitarget_columns()
        gui_app.multitarget_listbox.selection_clear(0, "end")
        gui_app.multitarget_listbox.selection_set(0, 2)
        gui_app._on_multitarget_selection_changed()
        gui_app.multitarget_model_vars["PLS"].set(True)
        gui_app.multitarget_coupling.set("joint")

        captured = {}

        def _fake_grid(X, Y, **kwargs):
            captured.update(kwargs)
            from spectral_predict.multitarget_search import MultiTargetSearchOutput
            return MultiTargetSearchOutput(
                results=[], target_names=kwargs["target_names"],
                correlation={}, n_targets=Y.shape[1], skipped=[])

        import spectral_predict.multitarget_grid as mg
        monkeypatch.setattr(mg, "run_multitarget_grid_search", _fake_grid)
        try:
            gui_app._run_multitarget_search_thread(gui_app._collect_multitarget_config())
        finally:
            gui_app.multitarget_coupling.set("independent")
        assert captured["coupling_mode"] == "joint"


@pytest.mark.gui
def test_cell_lower_bound_excludes_non_joint_capable_in_joint_mode(gui_app):
    """The pre-run cell-count heads-up must weight each model's HP configs by its
    coupling capability (mirroring _expand_model_modes): in JOINT mode an
    independent-only model (Ridge) emits 0 cells and must NOT be counted. The old
    flat ``n_cfg`` (total configs, incl. Ridge) over-counted."""
    from spectral_predict.models import get_model_grids

    grids = get_model_grids(
        task_type="regression", n_features=24, max_n_components=5,
        tier="standard", enabled_models=["PLS", "Ridge"],
    )
    # Precondition: both models actually contribute configs (else non-discriminating).
    assert len(grids["PLS"]) >= 1
    assert len(grids["Ridge"]) >= 1

    n_pp = 3
    got = gui_app._multitarget_cell_lower_bound(grids, n_pp, "joint")

    # PLS is joint-capable (weight 1); Ridge has no joint variant (weight 0), so
    # the estimate equals ONLY the PLS contribution.
    expected = n_pp * len(grids["PLS"])
    assert got == expected

    # Discriminating: the old flat weighting counted every config (incl. Ridge),
    # which would over-estimate here.
    flat_old = n_pp * sum(len(v) for v in grids.values())
    assert got < flat_old


@pytest.mark.gui
def test_tab_change_refreshes_target_list(gui_app):
    """FIX 1: the target list is primed only at tab creation + the Refresh button,
    so a data load never updated it. Showing the Multi-Target sub-tab fires
    <<NotebookTabChanged>> on config_notebook, which must refresh the list."""
    _load_multitarget_data(gui_app)
    # Simulate freshly-loaded data whose columns have not yet reached the listbox.
    gui_app.multitarget_listbox.delete(0, "end")
    assert list(gui_app.multitarget_listbox.get(0, "end")) == []
    gui_app.config_notebook.event_generate("<<NotebookTabChanged>>")
    gui_app.root.update()
    listed = list(gui_app.multitarget_listbox.get(0, "end"))
    assert set(listed) == {"prop_0", "prop_1", "prop_2"}


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
