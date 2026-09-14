"""Regression tests for the Model Development CV-strategy row (folds/repeats widgets).

Re-setting the default 'kfold' strategy (as _load_default_parameters and model
loading do) used to raise TclError "isn't packed", because widgets were packed
relative to siblings that had been hidden.
"""

from __future__ import annotations

import pytest

TRANSITIONS = [
    ["kfold", "kfold"],
    ["loo", "kfold"],
    ["loo", "repeated_kfold"],
    ["repeated_kfold", "kfold", "kfold"],
    ["kfold", "loo", "loo", "repeated_kfold", "repeated_kfold", "kfold"],
]

EXPECTED_VISIBLE = {
    "kfold": ["refine_folds_label", "refine_folds_spinbox"],
    "repeated_kfold": [
        "refine_folds_label",
        "refine_folds_spinbox",
        "refine_repeats_label",
        "refine_repeats_spinbox",
    ],
    "loo": [],
}


@pytest.fixture
def callback_errors(gui_app):
    root = gui_app.root
    errors: list[BaseException] = []
    original = root.report_callback_exception
    root.report_callback_exception = lambda exc, val, tb: errors.append(val)
    yield errors
    root.report_callback_exception = original


@pytest.mark.parametrize("sequence", TRANSITIONS, ids=lambda s: "->".join(s))
def test_cv_strategy_transitions_raise_no_callback_errors(gui_app, callback_errors, sequence):
    for strategy in sequence:
        gui_app.refine_cv_strategy.set(strategy)
    gui_app.refine_cv_strategy.set("kfold")

    assert callback_errors == []


@pytest.mark.parametrize("final", ["kfold", "repeated_kfold", "loo"])
def test_cv_strategy_shows_widgets_in_order_before_hint(gui_app, callback_errors, final):
    for strategy in ["loo", "repeated_kfold", "kfold", final, final]:
        gui_app.refine_cv_strategy.set(strategy)

    packed = gui_app.refine_cv_hint.master.pack_slaves()
    expected = [getattr(gui_app, name) for name in EXPECTED_VISIBLE[final]]
    visible = [w for w in packed if w in expected]
    hint_pos = packed.index(gui_app.refine_cv_hint)

    assert callback_errors == []
    assert visible == expected
    assert all(packed.index(w) < hint_pos for w in visible)

    gui_app.refine_cv_strategy.set("kfold")
