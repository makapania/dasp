"""QW4 in the GUI: KS / SPXY / DUPLEX pick the calibration set, and a saved
holdout is restored by its sample IDs, never re-selected.

A run saved before QW4 held out the samples Kennard-Stone picked first (the
extremes). Resuming it must load exactly that holdout even though "Create
Validation Set" now picks a different one; the crash-resume calibration
identity check depends on it.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import spectral_predict.sample_selection as sample_selection
from spectral_predict.sample_selection import kennard_stone

from tests.gui.test_dataset_state import (  # noqa: F401 -- fixtures and helpers
    _gui_record,
    _spectra,
    _split,
    _start_and_resume,
    clean_state,
)
from tests.gui.test_resume_run_completion import worker_env  # noqa: F401 -- fixture

IDS = [f"S{i}" for i in range(1, 31)]


@pytest.fixture
def holdout_app(clean_state):
    app = clean_state
    saved = (app.validation_algorithm.get(), app.validation_percentage.get())
    X, y = _spectra(IDS, seed=4)
    assert app._install_dataset(X, y, None, None, replot=False)
    app.validation_percentage.set(20.0)  # 30 samples -> 6 held out
    yield app
    app.validation_algorithm.set(saved[0])
    app.validation_percentage.set(saved[1])


def _create(app, algorithm):
    app.validation_algorithm.set(algorithm)
    with (
        patch("tkinter.messagebox.showerror") as err,
        patch("tkinter.messagebox.showwarning") as warn,
        patch("tkinter.messagebox.showinfo"),
    ):
        app._create_validation_set()
    assert not err.called and not warn.called
    return set(app.validation_indices)


@pytest.mark.parametrize("algorithm", ["Kennard-Stone", "SPXY"])
def test_create_validation_set_calibrates_on_the_extremes(holdout_app, algorithm):
    app = holdout_app
    holdout = _create(app, algorithm)
    y = app.y
    cal = [s for s in y.index if s not in holdout]

    assert len(holdout) == 6 and app.validation_enabled.get()
    # _spectra encodes y in X, so the y extremes are the spectral extremes too.
    assert y.idxmin() in cal and y.idxmax() in cal
    assert y[list(holdout)].min() > y[cal].min()
    assert y[list(holdout)].max() < y[cal].max()
    assert set(app.validation_X.index) == holdout


def test_kennard_stone_holdout_is_complement_of_ks_calibration(holdout_app):
    app = holdout_app
    holdout = _create(app, "Kennard-Stone")
    cal_pos = kennard_stone(app.X.to_numpy(float), 24)
    assert holdout == set(app.X.index) - set(app.X.index[cal_pos])


def test_duplex_option_creates_a_holdout(holdout_app):
    app = holdout_app
    holdout = _create(app, "DUPLEX")
    assert len(holdout) == 6
    cal_pos, val_pos = sample_selection.duplex(app.X.to_numpy(float), n_cal=24)
    assert holdout == set(app.X.index[val_pos])


def _refuse(*args, **kwargs):
    raise AssertionError("a saved holdout must be restored by ID, not re-selected")


def test_resume_restores_a_pre_qw4_holdout_exactly(holdout_app, worker_env):
    app, rs = holdout_app, worker_env
    # What a pre-QW4 build held out: the samples Kennard-Stone picks first.
    old_holdout = set(app.X.index[kennard_stone(app.X.to_numpy(float), 6)])
    _split(app, old_holdout)
    meta = _start_and_resume(rs, app.X, app.y, **_gui_record(app))

    # A new session: the user re-creates the split with the corrected direction.
    new_holdout = _create(app, "Kennard-Stone")
    assert new_holdout != old_holdout

    with (
        patch.object(sample_selection, "split_calibration_holdout", _refuse),
        patch.object(sample_selection, "kennard_stone", _refuse),
        patch.object(sample_selection, "spxy", _refuse),
        patch.object(sample_selection, "duplex", _refuse),
        patch.object(type(app), "_holdout_labels_by_selection", staticmethod(_refuse)),
        patch("tkinter.messagebox.askyesnocancel", return_value=True) as ask,
    ):
        assert app._confirm_resume_before_launch(["PLS"], "quick") is True
        assert app._verify_resume_calibration_identity(meta) == "ok"

    assert ask.call_args[0][0] == "Validation set differs from the interrupted run"
    assert set(app.validation_indices) == old_holdout
    assert set(app.validation_X.index) == old_holdout
    rs._reset_for_tests()


def test_resume_with_matching_saved_holdout_asks_nothing(holdout_app, worker_env):
    app, rs = holdout_app, worker_env
    holdout = _create(app, "SPXY")
    _start_and_resume(rs, app.X, app.y, **_gui_record(app))
    with (
        patch.object(sample_selection, "split_calibration_holdout", _refuse),
        patch("tkinter.messagebox.askyesnocancel") as ask,
    ):
        assert app._confirm_resume_before_launch(["PLS"], "quick") is True
    assert not ask.called
    assert set(app.validation_indices) == holdout
    rs._reset_for_tests()
