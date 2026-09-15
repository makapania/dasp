"""Round 13 fixes for PR #79 (Codex review of round 12, 5c6d73c).

1. A run that held nothing out is compared too: creating a holdout before
   resuming changes its training rows, so it asks first.
2. A failed split restore changes no validation state (it used to clear
   validation_X first, which silently skipped validation metrics later).
Nits: a blank detail box of a switched-off option doesn't block a launch, and
the pending-run Delete message doesn't offer a retry once something was deleted.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import spectral_predict_gui_optimized as gui_module
from spectral_predict.run_state import DiscardResult, RunMetadata

from tests.gui.test_resume_data_mismatch_keeps_run import _dataset, resumed  # noqa: F401
from tests.gui.test_resume_round12 import resumed_with_split, validation_state  # noqa: F401
from tests.gui.test_resume_run_completion import _regression_data, worker_env  # noqa: F401


def _meta_with_split(indices, run_id="run1234"):
    return RunMetadata(
        run_id=run_id, storage_path="s", storage_url="", label=None,
        dataset_fingerprint=None, model_names=[], n_trials_per_model=None,
        started_iso="2026-09-15T00:00:00", validation_indices=list(indices),
    )


@pytest.mark.parametrize(
    "answer, expect_result, expect_holdout",
    [(True, "ok", set()), (False, "fresh", {4, 5}), (None, None, {4, 5})],
)
def test_run_without_holdout_asks_before_a_new_one_is_used(
    gui_app, validation_state, answer, expect_result, expect_holdout
):
    X, y = _dataset(1)
    gui_app.X, gui_app.y = X, y
    gui_app.validation_enabled.set(True)
    gui_app.validation_indices = {4, 5}
    gui_app.validation_X, gui_app.validation_y = X.loc[[4, 5]], y.loc[[4, 5]]

    with patch("tkinter.messagebox.askyesnocancel", return_value=answer) as ask:
        result = gui_app._reconcile_resume_validation_split(_meta_with_split([]))

    assert ask.called and "held nothing out" in ask.call_args[0][1]
    assert result == expect_result
    assert set(gui_app.validation_indices or []) == expect_holdout
    if expect_result == "ok":
        assert gui_app.validation_X is None and gui_app.validation_y is None


def test_failed_restore_leaves_the_current_validation_set_intact(gui_app, validation_state):
    X, y = _dataset(1)
    gui_app.X, gui_app.y = X, y
    gui_app.validation_enabled.set(True)
    gui_app.validation_indices = {4, 5}
    gui_app.validation_X, gui_app.validation_y = X.loc[[4, 5]], y.loc[[4, 5]]

    with patch("tkinter.messagebox.askyesnocancel", return_value=True), \
         patch("tkinter.messagebox.showerror") as err:
        result = gui_app._reconcile_resume_validation_split(_meta_with_split([98, 99]))

    assert result is None
    assert err.called
    assert gui_app.validation_indices == {4, 5}
    assert gui_app.validation_X is not None and len(gui_app.validation_X) == 2
    assert gui_app.validation_y is not None and len(gui_app.validation_y) == 2


def test_blank_detail_of_a_switched_off_option_does_not_block(gui_app, worker_env, monkeypatch):
    rs = worker_env
    gui_app.X, gui_app.y = _regression_data()
    real = gui_module._launch_settings_snapshot

    def without_smoothing_window(app):
        snap = real(app)
        snap.pop("smoothing_window")  # blank box
        return snap

    monkeypatch.setattr(gui_module, "_launch_settings_snapshot", without_smoothing_window)
    gui_app.bayes_enable_smoothing.set(False)
    with patch("tkinter.messagebox.showerror") as err:
        assert gui_app._confirm_resume_before_launch(["PLS"], "quick") is True
    assert not err.called
    assert rs.get_active_run_id() == gui_app._pending_bayesian_run_id

    rs._reset_for_tests()
    gui_app.bayes_enable_smoothing.set(True)
    with patch("tkinter.messagebox.showerror") as err:
        assert gui_app._confirm_resume_before_launch(["PLS"], "quick") is False
    assert "smoothing_window" in err.call_args[0][1]


def test_pending_delete_message_says_fresh_when_something_was_deleted(
    gui_app, resumed, monkeypatch
):
    rs, meta, store, started, (X, y) = resumed
    monkeypatch.setattr(rs, "resume_run", lambda run_id: None)  # not resuming yet
    monkeypatch.setattr(rs, "is_resuming", lambda: False)
    monkeypatch.setattr(
        rs, "discard_incomplete_run",
        lambda run_id: DiscardResult(sidecar_deleted=True, storage_deleted=False,
                                     errors=["storage locked"]),
    )
    gui_app.X, gui_app.y = X, y
    with patch("tkinter.messagebox.askyesnocancel", return_value=False), \
         patch("tkinter.messagebox.showerror") as err:
        assert gui_app._confirm_resume_before_launch(["PLS"], "quick") is False
    assert "no longer be resumed" in err.call_args[0][1]
