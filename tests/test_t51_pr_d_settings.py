"""T-51 PR D: extra-axes settings, legacy snapshots and the dimension advisory (no Tk).

The GUI wiring and resume flows are in ``tests/gui/test_t51_pr_d_gui.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from spectral_predict import run_gui_settings as rgs
from spectral_predict.search_spaces import BUNDLES


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _FakeGUI:
    """Every PR D var at a chosen value."""

    def __init__(self, **values):
        for name in rgs.extra_axes_settings():
            setattr(self, name, _Var(rgs.LEGACY_DEFAULTS[name]))
        for name, value in values.items():
            setattr(self, name, _Var(value))


def _axis(bundle_id: str) -> str:
    return f"{rgs.EXTRA_AXES_VAR_PREFIX}{bundle_id}"


# --- names, capture, legacy defaults ------------------------------------------------


def test_every_bundle_has_a_captured_setting_with_a_legacy_default():
    names = rgs.extra_axes_setting_names()
    assert names == tuple(_axis(b) for b in BUNDLES)
    for name in rgs.extra_axes_settings():
        assert name in rgs.CAPTURABLE_SETTINGS
        assert name in rgs.LEGACY_DEFAULTS
    assert rgs.LEGACY_DEFAULTS[rgs.N_STARTUP_TRIALS_SETTING] == ""
    assert all(rgs.LEGACY_DEFAULTS[n] is False for n in names)


@pytest.mark.parametrize(
    "text, expected",
    [("", None), ("  ", None), (None, None), ("30", 30), (" 30", 30), ("1", 1)],
)
def test_parse_n_startup_trials_accepts(text, expected):
    assert rgs.parse_n_startup_trials(text) == expected


@pytest.mark.parametrize("text", ["abc", "0", "-1", "20.5", "1e2"])
def test_parse_n_startup_trials_rejects(text):
    with pytest.raises(ValueError, match="whole number"):
        rgs.parse_n_startup_trials(text)


def test_normalize_fills_only_missing_new_keys():
    saved = {"folds": 5, _axis("xgb_sampling"): True}
    normalized = rgs.normalize_saved_settings(saved)
    assert normalized[_axis("xgb_sampling")] is True  # saved value wins
    assert normalized[_axis("rf_features")] is False
    assert normalized[rgs.N_STARTUP_TRIALS_SETTING] == ""
    assert normalized["folds"] == 5
    assert "use_pls" not in normalized, "only PR D keys have legacy defaults"
    assert saved == {"folds": 5, _axis("xgb_sampling"): True}, "input not mutated"


@pytest.mark.parametrize("empty", [None, {}])
def test_empty_snapshots_stay_no_ops(empty):
    assert rgs.normalize_saved_settings(empty) == empty
    current = rgs.capture_gui_settings(_FakeGUI(**{_axis("xgb_sampling"): True}))
    assert rgs.diff_gui_settings(empty, current) == []
    report = rgs.restore_gui_settings(_FakeGUI(), empty)
    assert report.total_restored == 0


# --- diff: an old snapshot compares as the legacy defaults -----------------------------


def test_old_snapshot_against_defaults_has_no_difference():
    current = rgs.capture_gui_settings(_FakeGUI(folds=5))
    assert rgs.diff_gui_settings({"folds": 5}, current) == []


def test_old_snapshot_against_a_ticked_bundle_is_a_difference():
    current = rgs.capture_gui_settings(_FakeGUI(folds=5, **{_axis("xgb_sampling"): True}))
    assert rgs.diff_gui_settings({"folds": 5}, current) == [(_axis("xgb_sampling"), False, True)]


def test_old_snapshot_against_a_changed_startup_is_a_difference():
    current = rgs.capture_gui_settings(_FakeGUI(**{rgs.N_STARTUP_TRIALS_SETTING: "40"}))
    assert rgs.diff_gui_settings({"folds": 5}, current) == [
        (rgs.N_STARTUP_TRIALS_SETTING, "", "40")
    ]


def test_new_snapshot_startup_change_is_a_difference():
    saved = rgs.capture_gui_settings(_FakeGUI(**{rgs.N_STARTUP_TRIALS_SETTING: "30"}))
    current = rgs.capture_gui_settings(_FakeGUI(**{rgs.N_STARTUP_TRIALS_SETTING: "40"}))
    assert rgs.diff_gui_settings(saved, current) == [(rgs.N_STARTUP_TRIALS_SETTING, "30", "40")]


# --- restore: partial patches never fill defaults (the review's restore loop) ---------


def test_partial_restore_writes_only_the_given_keys():
    gui = _FakeGUI(**{_axis("xgb_sampling"): True, rgs.N_STARTUP_TRIALS_SETTING: "40"})
    rgs.restore_gui_settings(gui, {rgs.N_STARTUP_TRIALS_SETTING: "30"})
    assert getattr(gui, _axis("xgb_sampling")).get() is True
    assert all(
        getattr(gui, name).get() == (name == _axis("xgb_sampling"))
        for name in rgs.extra_axes_setting_names()
    )
    assert getattr(gui, rgs.N_STARTUP_TRIALS_SETTING).get() == "30"


def test_partial_patch_converges_in_one_pass():
    """Saved: bundle on, startup 30. Now: bundle on, startup 40 -> one restore settles it."""
    saved = rgs.capture_gui_settings(
        _FakeGUI(**{_axis("xgb_sampling"): True, rgs.N_STARTUP_TRIALS_SETTING: "30"})
    )
    gui = _FakeGUI(**{_axis("xgb_sampling"): True, rgs.N_STARTUP_TRIALS_SETTING: "40"})
    diffs = rgs.diff_gui_settings(saved, rgs.capture_gui_settings(gui))
    rgs.restore_gui_settings(gui, {name: value for name, value, _ in diffs})
    assert rgs.diff_gui_settings(saved, rgs.capture_gui_settings(gui)) == []


def test_reverse_partial_patch_keeps_matching_startup():
    """Restoring a bundle leaves a non-default startup that already matches alone."""
    saved = rgs.capture_gui_settings(
        _FakeGUI(**{_axis("lgbm_child"): True, rgs.N_STARTUP_TRIALS_SETTING: "35"})
    )
    gui = _FakeGUI(**{rgs.N_STARTUP_TRIALS_SETTING: "35"})
    diffs = rgs.diff_gui_settings(saved, rgs.capture_gui_settings(gui))
    assert [d[0] for d in diffs] == [_axis("lgbm_child")]
    rgs.restore_gui_settings(gui, {name: value for name, value, _ in diffs})
    assert getattr(gui, rgs.N_STARTUP_TRIALS_SETTING).get() == "35"
    assert rgs.diff_gui_settings(saved, rgs.capture_gui_settings(gui)) == []


def test_full_restore_of_old_snapshot_resets_new_controls():
    gui = _FakeGUI(**{_axis("xgb_sampling"): True, rgs.N_STARTUP_TRIALS_SETTING: "40"})
    rgs.restore_gui_settings(gui, rgs.normalize_saved_settings({"folds": 5}))
    assert getattr(gui, _axis("xgb_sampling")).get() is False
    assert getattr(gui, rgs.N_STARTUP_TRIALS_SETTING).get() == ""


def test_summary_lists_extra_axes_and_startup():
    settings = {_axis("xgb_sampling"): True, rgs.N_STARTUP_TRIALS_SETTING: " 30 "}
    summary = rgs.summarize_gui_settings(settings)
    assert "extra-axes=xgb_sampling" in summary
    assert "startup-trials=30" in summary
    assert "extra-axes" not in rgs.summarize_gui_settings({"folds": 5})


# --- advisory -----------------------------------------------------------------------


def test_advisory_counts_base_shared_and_bundle_axes():
    from spectral_predict.extra_axes_advisory import model_dimensions

    plain = model_dimensions("XGBoost", "regression", ())
    opened = model_dimensions("XGBoost", "regression", ("xgb_sampling",))
    assert opened.extra == len(BUNDLES["xgb_sampling"].axes) > 0
    assert opened.total == plain.total + opened.extra
    assert opened.bundles == ("xgb_sampling",)
    # Bundles for another model or task add nothing.
    assert model_dimensions("PLS", "regression", ("xgb_sampling",)).extra == 0
    assert model_dimensions("XGBoost", "regression", ("lof_metric",)).extra == 0


def test_advisory_shared_axes_follow_the_bayesian_options():
    from spectral_predict.extra_axes_advisory import SHARED_SUBSET_AXES, shared_axis_names

    plain = shared_axis_names(False, False, False)
    assert set(SHARED_SUBSET_AXES) <= plain
    everything = shared_axis_names(True, True, True)
    assert everything - plain == {"apply_baseline", "apply_smoothing", "apply_autoscale"}


@pytest.mark.parametrize(
    "model, task, bundles, options",
    [
        ("XGBoost", "regression", ("xgb_sampling",), {"autoscale": True}),
        ("PLS", "regression", (), {}),
        ("IsolationForest", "one_class", ("if_max_samples",), {}),
    ],
)
def test_advisory_dimension_matches_a_real_study(model, task, bundles, options):
    """Drift guard: every param a real study records is counted, and no more.

    The advisory's shared subset axes are a constant because the objective suggests
    them inline; a real study's params catch a new or renamed shared axis.
    """
    optuna = pytest.importorskip("optuna")
    from spectral_predict.extra_axes_advisory import SHARED_SUBSET_AXES, model_dimensions
    from spectral_predict.unified_bayesian import run_unified_bayesian

    rng = np.random.default_rng(0)
    y = rng.uniform(0, 10, 40)
    X = np.outer(y, np.linspace(0.5, 1.5, 30)) + rng.normal(0, 0.2, (40, 30))
    kwargs = {}
    if task == "one_class":
        y = np.where(y > 3.0, 1, 0)
        kwargs["inlier_class_label"] = 1
    _, study = run_unified_bayesian(
        X, y, np.linspace(1000, 2500, 30), model, task_type=task, n_trials=6, cv_folds=3,
        random_state=0, verbose=False, enable_sqlite_persistence="never",
        enabled_extra_axes=bundles, enable_autoscale=options.get("autoscale", False),
        **kwargs,
    )
    seen = set()
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            seen |= set(trial.params)
    assert set(SHARED_SUBSET_AXES) <= seen
    dims = model_dimensions(model, task, bundles, **options)
    assert len(seen) <= dims.total, "the study searched a name the advisory does not count"
