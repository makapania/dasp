"""T-43 regression tests for resume auto-restore of GUI settings.

Covers:
- Round-trip: capture from a fake-GUI → serialize via RunMetadata.to_dict
  → deserialize via from_dict → restore onto a fresh fake-GUI.
- Backward-compat: a sidecar without ``gui_settings`` deserializes fine and
  ``restore_gui_settings(None)`` is a no-op.
- Forgiving: stale keys (sidecar from a future build) are skipped without
  crashing; whitelisted keys without a matching Tk var are skipped.
- End-to-end through start_run: sidecar JSON contains the captured settings.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


# --- minimal Tk-var stand-ins ----------------------------------------------
#
# We don't want a real Tk root for unit tests (slow, requires display in CI).
# These tiny shims expose the ``.get()`` / ``.set()`` surface the
# capture/restore helpers actually use.


class _FakeVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class _FakeGUI:
    """Just enough surface area for capture_gui_settings / restore_gui_settings."""

    def __init__(self, **vars_):
        for name, value in vars_.items():
            setattr(self, name, _FakeVar(value))


# --- fixtures --------------------------------------------------------------


@pytest.fixture
def fresh_state(tmp_path, monkeypatch):
    """Fresh run_state module with a tmp user-data dir."""
    if sys.platform == "win32":
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    else:
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    sys.modules.pop("spectral_predict.run_state", None)
    sys.modules.pop("spectral_predict.resource_paths", None)
    sys.modules.pop("spectral_predict.run_gui_settings", None)
    rp = importlib.import_module("spectral_predict.resource_paths")
    rs = importlib.import_module("spectral_predict.run_state")
    rgs = importlib.import_module("spectral_predict.run_gui_settings")
    rs._reset_for_tests()

    yield rs, rp, rgs, tmp_path

    rs._reset_for_tests()


@pytest.fixture
def populated_gui():
    """A fake-GUI with a representative slice of analysis-defining Tk vars."""
    return _FakeGUI(
        # preprocessing
        use_raw=False,
        use_snv=True,
        use_sg1=False,
        use_sg2=True,
        use_autoscale=True,
        use_msc=False,
        # baseline
        enable_baseline=True,
        baseline_method="airpls",
        baseline_poly_degree=3,
        # smoothing
        enable_smoothing=False,
        smoothing_window=15,
        smoothing_polyorder=2,
        # SG windows
        window_7=True,
        window_11=False,
        window_17=True,
        window_23=False,
        window_31=False,
        # variable selection
        varsel_importance=True,
        varsel_uve=True,
        varsel_cars=False,
        # models
        use_pls=True,
        use_ridge=False,
        use_lightgbm=True,
        use_xgboost=True,
        # tier / optimization
        model_tier="standard",
        optimization_method="unified",
        n_unified_trials=200,
        bayesian_persistence_mode="auto",
        # task / CV
        task_type="regression",
        folds=5,
        cv_strategy="kfold",
        cv_n_repeats=3,
        max_n_components=15,
    )


# --- capture / round-trip --------------------------------------------------


def test_capture_returns_only_whitelisted_present_vars(fresh_state, populated_gui):
    _, _, rgs, _ = fresh_state
    captured = rgs.capture_gui_settings(populated_gui)

    assert captured["use_snv"] is True
    assert captured["baseline_method"] == "airpls"
    assert captured["n_unified_trials"] == 200
    # Settings absent from the fake-GUI are absent from the snapshot rather
    # than recorded as None — keeps the sidecar minimal.
    assert "use_neuralboosted" not in captured


def test_capture_skips_non_whitelisted_attrs(fresh_state):
    _, _, rgs, _ = fresh_state
    gui = _FakeGUI(use_snv=True, pca_color_var="Y Value", filter_model_var="All")
    captured = rgs.capture_gui_settings(gui)

    assert captured == {"use_snv": True}


def test_round_trip_preserves_settings(fresh_state, populated_gui):
    rs, _, rgs, _ = fresh_state
    captured = rgs.capture_gui_settings(populated_gui)

    meta = rs.RunMetadata(
        run_id="abc", storage_path="/tmp/x.sqlite3", storage_url="",
        label=None, dataset_fingerprint=None, model_names=[],
        n_trials_per_model=None, started_iso="2026-05-02T00:00:00",
        gui_settings=captured,
    )
    revived = rs.RunMetadata.from_dict(meta.to_dict())
    assert revived.gui_settings == captured

    # Restore onto a blank fake-GUI mirrors the original.
    blank = _FakeGUI(
        **{name: None for name in captured},
    )
    report = rgs.restore_gui_settings(blank, revived.gui_settings)
    assert report.errors == []
    assert report.skipped_unknown == []
    assert report.total_restored == len(captured)
    for name, value in captured.items():
        assert getattr(blank, name).get() == value


# --- backward-compat -------------------------------------------------------


def test_legacy_sidecar_without_gui_settings_loads(fresh_state):
    rs, _, _, _ = fresh_state
    legacy = {
        "run_id": "abc",
        "storage_path": "/tmp/x.sqlite3",
        "storage_url": "",
        "label": "old",
        "dataset_fingerprint": "ff",
        "model_names": ["pls"],
        "n_trials_per_model": 50,
        "started_iso": "2026-04-01T12:00:00",
        "bayesian_persistence_mode": "never",
        # no `gui_settings` key
    }
    meta = rs.RunMetadata.from_dict(legacy)
    assert meta.gui_settings is None


def test_restore_with_none_settings_is_noop(fresh_state):
    _, _, rgs, _ = fresh_state
    gui = _FakeGUI(use_snv=False)
    report = rgs.restore_gui_settings(gui, None)

    assert report.total_restored == 0
    assert report.errors == []
    assert gui.use_snv.get() is False  # untouched


def test_restore_with_empty_dict_is_noop(fresh_state):
    _, _, rgs, _ = fresh_state
    gui = _FakeGUI(use_snv=False)
    report = rgs.restore_gui_settings(gui, {})

    assert report.total_restored == 0
    assert gui.use_snv.get() is False


# --- forgiving on drift ----------------------------------------------------


def test_restore_skips_unknown_keys_without_crashing(fresh_state):
    _, _, rgs, _ = fresh_state
    gui = _FakeGUI(use_snv=False)
    settings = {
        "use_snv": True,
        "future_setting_does_not_exist": 42,  # newer build
        "pca_color_var": "X",  # not whitelisted (display state)
    }

    report = rgs.restore_gui_settings(gui, settings)

    assert "use_snv" in report.restored
    assert "future_setting_does_not_exist" in report.skipped_unknown
    assert "pca_color_var" in report.skipped_unknown
    assert report.errors == []
    assert gui.use_snv.get() is True


def test_restore_skips_whitelisted_but_missing_var(fresh_state):
    _, _, rgs, _ = fresh_state
    gui = _FakeGUI(use_snv=False)  # missing use_xgboost
    settings = {"use_snv": True, "use_xgboost": True}

    report = rgs.restore_gui_settings(gui, settings)

    assert report.restored == ["use_snv"]
    assert report.skipped_no_var == ["use_xgboost"]
    assert gui.use_snv.get() is True


def test_restore_collects_set_errors_without_crashing(fresh_state):
    _, _, rgs, _ = fresh_state

    class _BrokenVar:
        def get(self):
            return None

        def set(self, value):
            raise RuntimeError("type coercion failed")

    gui = _FakeGUI(use_snv=False)
    gui.smoothing_window = _BrokenVar()
    report = rgs.restore_gui_settings(
        gui, {"use_snv": True, "smoothing_window": "bad"}
    )

    assert "use_snv" in report.restored
    assert any(e.startswith("smoothing_window") for e in report.errors)
    # Other settings still applied — error in one var doesn't block others.
    assert gui.use_snv.get() is True


def test_from_dict_drops_unknown_top_level_field(fresh_state):
    """A future build adds a new RunMetadata field; current code mustn't crash."""
    rs, _, _, _ = fresh_state
    future = {
        "run_id": "abc",
        "storage_path": "/tmp/x.sqlite3",
        "storage_url": "",
        "label": "x",
        "dataset_fingerprint": None,
        "model_names": [],
        "n_trials_per_model": None,
        "started_iso": "2026-05-02T00:00:00",
        "bayesian_persistence_mode": "never",
        "gui_settings": None,
        "some_future_field": {"nested": True},  # newer build
    }
    meta = rs.RunMetadata.from_dict(future)
    assert meta.run_id == "abc"


# --- end-to-end through start_run -----------------------------------------


def test_start_run_persists_gui_settings_to_sidecar(fresh_state, populated_gui):
    rs, rp, rgs, _ = fresh_state
    captured = rgs.capture_gui_settings(populated_gui)

    meta = rs.start_run(
        label="standard",
        dataset_fingerprint="abc",
        model_names=["pls"],
        n_trials_per_model=200,
        bayesian_persistence_mode="auto",
        gui_settings=captured,
    )

    sidecar = rp.get_user_optuna_dir() / "active_run.json"
    written = json.loads(sidecar.read_text(encoding="utf-8"))
    assert written["gui_settings"]["use_snv"] is True
    assert written["gui_settings"]["baseline_method"] == "airpls"
    assert written["gui_settings"]["n_unified_trials"] == 200
    # Round-trip through find_incomplete_run.
    rs._reset_for_tests()
    revived = rs.find_incomplete_run()
    assert revived is not None
    assert revived.gui_settings == captured


def test_start_run_without_gui_settings_omits_field(fresh_state):
    """Headless callers (tests, future CLI) shouldn't be forced to pass settings."""
    rs, rp, _, _ = fresh_state
    rs.start_run(label="x", model_names=["m"])

    sidecar = rp.get_user_optuna_dir() / "active_run.json"
    written = json.loads(sidecar.read_text(encoding="utf-8"))
    assert written.get("gui_settings") is None


# --- summary ---------------------------------------------------------------


def test_summarize_handles_empty_settings(fresh_state):
    _, _, rgs, _ = fresh_state
    assert rgs.summarize_gui_settings(None) == ""
    assert rgs.summarize_gui_settings({}) == ""


def test_from_dict_coerces_malformed_gui_settings_to_none(fresh_state):
    """A corrupted sidecar storing gui_settings as a string would crash
    downstream in restore_gui_settings when `.items()` is called on the
    string. The from_dict type-guard coerces non-dict values to None so
    the resume flow degrades to "no auto-restore" instead of raising."""
    rs, _, _, _ = fresh_state
    bad = {
        "run_id": "abc",
        "storage_path": "/tmp/x.sqlite3",
        "storage_url": "",
        "label": None,
        "dataset_fingerprint": None,
        "model_names": [],
        "n_trials_per_model": None,
        "started_iso": "2026-05-02T00:00:00",
        "bayesian_persistence_mode": "never",
        "gui_settings": "not-a-dict-somehow",
    }
    meta = rs.RunMetadata.from_dict(bad)
    assert meta.gui_settings is None


def test_restore_detects_silent_tk_int_var_corruption(fresh_state):
    """Tcl quirk: tk.IntVar.set("abc") does NOT raise — Tcl stores the
    string and the TclError surfaces on the next .get(). The set/get
    round-trip verify in restore_gui_settings must catch the silent
    poisoning and report it as an error rather than counting it as
    successfully restored."""
    _, _, rgs, _ = fresh_state

    class _PoisonableIntVar:
        """Mimics tk.IntVar's set-succeeds, get-raises pattern."""
        def __init__(self):
            self._value = 0

        def set(self, value):
            self._value = value  # store anything; no raise

        def get(self):
            if not isinstance(self._value, int):
                raise RuntimeError("expected integer")
            return self._value

    gui = _FakeGUI(use_snv=False)
    gui.smoothing_window = _PoisonableIntVar()
    report = rgs.restore_gui_settings(
        gui, {"use_snv": True, "smoothing_window": "abc"}
    )

    assert "use_snv" in report.restored
    assert "smoothing_window" not in report.restored
    assert any("smoothing_window" in e for e in report.errors)


def test_restore_detects_value_mismatch_after_set(fresh_state):
    """Some Tk var subclasses round/truncate the set value. The get-back
    verify catches that drift instead of falsely reporting "restored"."""
    _, _, rgs, _ = fresh_state

    class _RoundingDoubleVar:
        def __init__(self):
            self._value = 0.0

        def set(self, value):
            # Simulate floor-to-int round-trip drift.
            self._value = int(float(value))

        def get(self):
            return self._value

    gui = _FakeGUI(use_snv=False)
    gui.uve_cutoff_multiplier = _RoundingDoubleVar()
    report = rgs.restore_gui_settings(
        gui, {"uve_cutoff_multiplier": 1.25, "use_snv": True}
    )

    assert "use_snv" in report.restored
    assert any("uve_cutoff_multiplier" in e for e in report.errors)


def test_capture_skips_var_with_raising_get(fresh_state):
    """A real Tk var can raise on .get() if the Tcl interpreter is in a bad
    state (e.g., destroyed widget). Snapshot must omit the key, not crash."""
    _, _, rgs, _ = fresh_state

    class _BrokenGetVar:
        def get(self):
            raise RuntimeError("Tcl interpreter is gone")

        def set(self, value):
            pass

    gui = _FakeGUI(use_snv=True)
    gui.smoothing_window = _BrokenGetVar()
    captured = rgs.capture_gui_settings(gui)

    assert captured["use_snv"] is True
    assert "smoothing_window" not in captured


def test_restore_then_override_persistence_mode_order(fresh_state):
    """Document the order-dependent contract used by `_check_for_incomplete_run`:
    restore first (which would set bayesian_persistence_mode back to 'auto'),
    then force 'always'. If a future refactor inverts the order the resume
    would silently ignore the SQLite URL."""
    _, _, rgs, _ = fresh_state
    gui = _FakeGUI(bayesian_persistence_mode="auto", use_snv=True)
    settings = {"bayesian_persistence_mode": "auto", "use_snv": False}

    rgs.restore_gui_settings(gui, settings)
    assert gui.bayesian_persistence_mode.get() == "auto"

    # GUI's `_check_for_incomplete_run` then overrides to 'always' — this
    # final write must happen AFTER restore, not before.
    gui.bayesian_persistence_mode.set("always")
    assert gui.bayesian_persistence_mode.get() == "always"


def test_external_validation_controls_in_whitelist(fresh_state):
    """The external validation partition (SPXY / KS / Stratified / Random /
    Manual) is decided BEFORE the search, not post-hoc. Trials in the
    resumed SQLite were trained on a specific cal split; if the user
    re-creates a different validation set on resume (different algorithm,
    different %), samples used by the resumed trials' calibration could
    end up in the "new" validation set — silent leakage. Capturing the
    five validation Tk vars is sufficient for deterministic algorithms;
    non-deterministic ones need T-49 (indices persistence)."""
    _, _, rgs, _ = fresh_state
    required = {
        "validation_enabled",
        "validation_percentage",
        "validation_algorithm",
        "show_validation_metrics",
        "validation_top_n",
    }
    assert required.issubset(set(rgs.CAPTURABLE_SETTINGS))


def test_bayesian_specific_controls_in_whitelist(fresh_state):
    """The Bayesian search path reads bayes_* Tk vars (bayes_enable_baseline,
    bayes_baseline_method, bayes_enable_smoothing, bayes_region_test_*,
    bayes_enable_uve) at analysis time. Without them in the whitelist, a
    resumed Bayesian run silently uses defaults despite the resume banner."""
    _, _, rgs, _ = fresh_state
    required = {
        "bayes_enable_baseline",
        "bayes_baseline_method",
        "bayes_enable_smoothing",
        "bayes_region_test_all",
        "bayes_region_test_pairwise",
        "bayes_enable_uve",
    }
    assert required.issubset(set(rgs.CAPTURABLE_SETTINGS))


def test_bayesian_specific_controls_round_trip(fresh_state):
    """End-to-end: capture + restore preserves the bayes_* vars exactly."""
    _, _, rgs, _ = fresh_state
    gui = _FakeGUI(
        bayes_enable_baseline=True,
        bayes_baseline_method="airpls",
        bayes_enable_smoothing=True,
        bayes_region_test_all=True,
        bayes_region_test_pairwise=False,
        bayes_enable_uve=False,
        # Plus a non-bayes setting to confirm both paths captured.
        use_snv=True,
    )
    captured = rgs.capture_gui_settings(gui)
    assert captured["bayes_enable_baseline"] is True
    assert captured["bayes_baseline_method"] == "airpls"
    assert captured["bayes_enable_uve"] is False

    blank = _FakeGUI(
        bayes_enable_baseline=False,
        bayes_baseline_method="als",
        bayes_enable_smoothing=False,
        bayes_region_test_all=False,
        bayes_region_test_pairwise=True,
        bayes_enable_uve=True,
        use_snv=False,
    )
    rgs.restore_gui_settings(blank, captured)
    assert blank.bayes_enable_baseline.get() is True
    assert blank.bayes_baseline_method.get() == "airpls"
    assert blank.bayes_enable_uve.get() is False


def test_target_column_in_whitelist(fresh_state):
    """T-resume-y-variable-persist: the Y-column selector defaults to the
    file's first column on load, so a resumed classification run whose Y
    column wasn't first in the file would silently restart against the
    wrong target. Capturing target_column closes the gap."""
    _, _, rgs, _ = fresh_state
    assert "target_column" in rgs.CAPTURABLE_SETTINGS


def test_target_column_round_trips_when_present_in_data(fresh_state):
    """Capture target_column='ProteinPct' from one fake-GUI; restore onto
    a fresh fake-GUI whose `_get_available_target_columns` includes that
    name; assert the value lands."""
    _, _, rgs, _ = fresh_state

    source = _FakeGUI(target_column="ProteinPct", use_snv=True)
    captured = rgs.capture_gui_settings(source)
    assert captured["target_column"] == "ProteinPct"

    class _GUIWithColumns(_FakeGUI):
        def _get_available_target_columns(self):
            return ["MoisturePct", "ProteinPct", "FatPct"]

    blank = _GUIWithColumns(target_column="MoisturePct", use_snv=False)
    report = rgs.restore_gui_settings(blank, captured)

    assert "target_column" in report.restored
    assert report.errors == []
    assert blank.target_column.get() == "ProteinPct"


def test_target_column_falls_back_when_missing_from_current_data(fresh_state):
    """Stale-column scenario: the captured column name is no longer in the
    loaded dataset (renamed, different file, reordered). The Tk var stays
    untouched and the failure is surfaced via `errors` so the resume
    banner tells the user to pick the target manually instead of leaving
    them staring at a stuck combobox."""
    _, _, rgs, _ = fresh_state

    class _GUIWithColumns(_FakeGUI):
        def _get_available_target_columns(self):
            return ["MoisturePct", "FatPct"]  # no ProteinPct

    blank = _GUIWithColumns(target_column="MoisturePct", use_snv=False)
    report = rgs.restore_gui_settings(
        blank, {"target_column": "ProteinPct", "use_snv": True}
    )

    assert "target_column" not in report.restored
    assert "use_snv" in report.restored  # other settings still applied
    assert any("target_column" in e and "ProteinPct" in e for e in report.errors)
    # Tk var not mutated — the on-load default ("MoisturePct") stands so
    # the Combobox is left in a consistent, user-pickable state.
    assert blank.target_column.get() == "MoisturePct"


def test_target_column_skips_validation_when_helper_absent(fresh_state):
    """Headless callers (tests, future CLI) and partially-initialized GUIs
    may not expose `_get_available_target_columns`. The validation hook is
    duck-typed — when the helper is missing, fall through to the standard
    set/readback path so the restore still works in those contexts."""
    _, _, rgs, _ = fresh_state

    blank = _FakeGUI(target_column="", use_snv=False)
    assert not hasattr(blank, "_get_available_target_columns")

    report = rgs.restore_gui_settings(
        blank, {"target_column": "AnyColumn", "use_snv": True}
    )

    assert "target_column" in report.restored
    assert report.errors == []
    assert blank.target_column.get() == "AnyColumn"


def test_target_column_handles_helper_raising(fresh_state):
    """Defense-in-depth: the GUI's `_get_available_target_columns` reads
    `self.combined_metadata_df.columns` / `self.ref.columns`. If those
    attributes are in an unexpected partial-init state, the helper could
    raise. Surface as an error rather than letting the resume crash."""
    _, _, rgs, _ = fresh_state

    class _GUIWithBrokenHelper(_FakeGUI):
        def _get_available_target_columns(self):
            raise RuntimeError("ref columns not yet wired up")

    blank = _GUIWithBrokenHelper(target_column="MoisturePct", use_snv=False)
    report = rgs.restore_gui_settings(
        blank, {"target_column": "ProteinPct", "use_snv": True}
    )

    assert "target_column" not in report.restored
    assert "use_snv" in report.restored
    assert any("target_column" in e for e in report.errors)
    assert blank.target_column.get() == "MoisturePct"  # unmutated


def test_summarize_models_line_excludes_preprocessing_toggles(fresh_state):
    """The Models line of the resume banner must list only model names.
    Preprocessing toggles (use_snv, use_sg1, etc.) share the `use_*`
    prefix but represent preprocessing, not models — globbing every
    `use_*` key into the model list misleads the user about how many
    models the resumed run was searching."""
    _, _, rgs, _ = fresh_state
    settings = {
        "use_pls": True,
        "use_ridge": True,
        "use_lightgbm": False,
        "use_snv": True,    # preprocessing — must NOT appear in Models
        "use_sg1": True,    # preprocessing — must NOT appear in Models
        "use_autoscale": True,  # preprocessing — must NOT appear in Models
        "optimization_method": "unified",
        "n_unified_trials": 50,
        "model_tier": "standard",
        "task_type": "regression",
        "cv_strategy": "kfold",
        "folds": 5,
        "cv_n_repeats": 3,
    }
    summary = rgs.summarize_gui_settings(settings)

    # Extract the Models line.
    models_line = next(
        (line for line in summary.splitlines() if line.startswith("Models")),
        "",
    )
    assert "pls" in models_line
    assert "ridge" in models_line
    # Preprocessing toggles must NOT appear in the Models line.
    assert "snv" not in models_line
    assert "sg1" not in models_line
    assert "autoscale" not in models_line
    # The count must reflect actual models, not the use_* glob.
    assert "Models (2)" in models_line


def test_restore_report_fully_succeeded_property(fresh_state):
    """`fully_succeeded` is True iff at least one setting was restored and
    no `.set()` calls failed. Mirrors DiscardResult.fully_succeeded."""
    _, _, rgs, _ = fresh_state

    gui = _FakeGUI(use_snv=False, use_pls=False)
    report = rgs.restore_gui_settings(
        gui, {"use_snv": True, "use_pls": True}
    )
    assert report.fully_succeeded is True

    # Empty restore (None settings) is NOT a success — nothing happened.
    report_empty = rgs.restore_gui_settings(gui, None)
    assert report_empty.fully_succeeded is False

    # Errors poison the result regardless of partial success.
    class _BrokenSet:
        def get(self):
            return None
        def set(self, v):
            raise RuntimeError("nope")

    gui2 = _FakeGUI(use_snv=False)
    gui2.use_pls = _BrokenSet()
    mixed = rgs.restore_gui_settings(
        gui2, {"use_snv": True, "use_pls": True}
    )
    assert mixed.restored == ["use_snv"]
    assert mixed.errors  # populated
    assert mixed.fully_succeeded is False


def test_summarize_includes_key_facts(fresh_state, populated_gui):
    _, _, rgs, _ = fresh_state
    captured = rgs.capture_gui_settings(populated_gui)
    summary = rgs.summarize_gui_settings(captured)

    assert "unified" in summary
    assert "200 trials" in summary
    assert "standard" in summary  # tier
    assert "regression" in summary  # task
    assert "kfold" in summary
    assert "pls" in summary  # model name with use_ prefix stripped
    assert "snv" in summary
    assert "autoscale" in summary
    assert "baseline=airpls" in summary


def test_validation_indices_round_trip_int(fresh_state):
    """Int-typed DataFrame indices (default range index) round-trip
    through start_run -> sidecar JSON -> from_dict, sorted."""
    rs, rp, _, _ = fresh_state
    rs.start_run(
        label="x",
        model_names=["pls"],
        validation_indices=[42, 7, 100, 3],
    )
    sidecar = rp.get_user_optuna_dir() / "active_run.json"
    written = json.loads(sidecar.read_text(encoding="utf-8"))
    assert written["validation_indices"] == [3, 7, 42, 100]

    rs._reset_for_tests()
    revived = rs.find_incomplete_run()
    assert revived is not None
    assert revived.validation_indices == [3, 7, 42, 100]


def test_validation_indices_round_trip_string_labels(fresh_state):
    """String DataFrame labels (sample IDs) preserve insertion order."""
    rs, rp, _, _ = fresh_state
    rs.start_run(
        label="x",
        model_names=["pls"],
        validation_indices=["sample_05", "sample_01", "sample_12"],
    )
    sidecar = rp.get_user_optuna_dir() / "active_run.json"
    written = json.loads(sidecar.read_text(encoding="utf-8"))
    # Insertion order preserved (no sort because str labels).
    assert written["validation_indices"] == [
        "sample_05", "sample_01", "sample_12",
    ]


def test_validation_indices_legacy_sidecar_loads_with_none(fresh_state):
    """Older sidecars without the field deserialize cleanly."""
    rs, _, _, _ = fresh_state
    legacy = {
        "run_id": "abc",
        "storage_path": "/tmp/x.sqlite3",
        "storage_url": "",
        "label": None,
        "dataset_fingerprint": None,
        "model_names": [],
        "n_trials_per_model": None,
        "started_iso": "2026-04-01T12:00:00",
        "bayesian_persistence_mode": "never",
        # no validation_indices key
    }
    meta = rs.RunMetadata.from_dict(legacy)
    assert meta.validation_indices is None


def test_validation_indices_malformed_coerces_to_none(fresh_state):
    """Corrupted sidecar storing validation_indices as garbage shouldn't
    crash the resume path — coerce to None, log a warning, continue."""
    rs, _, _, _ = fresh_state
    bad = {
        "run_id": "abc",
        "storage_path": "/tmp/x.sqlite3",
        "storage_url": "",
        "label": None,
        "dataset_fingerprint": None,
        "model_names": [],
        "n_trials_per_model": None,
        "started_iso": "2026-04-01T12:00:00",
        "bayesian_persistence_mode": "never",
        "validation_indices": "not-a-list",
    }
    meta = rs.RunMetadata.from_dict(bad)
    assert meta.validation_indices is None

    bad2 = dict(bad)
    bad2["validation_indices"] = [1, 2, [3, 4]]  # nested list element
    meta2 = rs.RunMetadata.from_dict(bad2)
    assert meta2.validation_indices is None


def test_validation_indices_empty_list_treated_as_none(fresh_state):
    """An empty list is equivalent to "no validation captured." start_run
    should normalize to None so the sidecar doesn't carry empty arrays."""
    rs, _, _, _ = fresh_state
    meta = rs.start_run(
        label="x", model_names=["pls"], validation_indices=[],
    )
    assert meta.validation_indices is None


def test_apply_pending_validation_indices_slices_by_label():
    """The GUI helper resolves pending indices against the current
    DataFrame and populates validation_X / validation_y."""
    import pandas as pd

    # Stand-in for the GUI: just enough surface area for the helper.
    class _FakeApp:
        def __init__(self):
            self.X = pd.DataFrame(
                {"f1": range(10), "f2": range(10, 20)},
                index=[f"s{i}" for i in range(10)],
            )
            self.y = pd.Series(range(10), index=[f"s{i}" for i in range(10)])
            self.validation_X = None
            self.validation_y = None
            self.validation_indices = set()
            self._pending_validation_indices = ["s2", "s5", "s8"]
            self._logs = []

        def _log_progress(self, msg):
            self._logs.append(msg)

    # Bind the real method into the fake.
    from spectral_predict_gui_optimized import (
        SpectralPredictApp,
    )
    app = _FakeApp()
    SpectralPredictApp._apply_pending_validation_indices(app)

    assert app.validation_indices == {"s2", "s5", "s8"}
    assert list(app.validation_X.index) == ["s2", "s5", "s8"]
    assert list(app.validation_y.index) == ["s2", "s5", "s8"]
    assert app._pending_validation_indices is None  # cleared


def test_apply_pending_validation_indices_skip_when_user_already_set():
    """If the user clicked Create Validation Set manually before Run
    Analysis, respect that — don't clobber with the resumed indices."""
    import pandas as pd

    class _FakeApp:
        def __init__(self):
            self.X = pd.DataFrame({"f1": range(10)},
                                   index=[f"s{i}" for i in range(10)])
            self.y = pd.Series(range(10), index=[f"s{i}" for i in range(10)])
            # User manually created a different validation set.
            self.validation_X = pd.DataFrame({"f1": [3, 4]}, index=["s3", "s4"])
            self.validation_y = pd.Series([3, 4], index=["s3", "s4"])
            self.validation_indices = {"s3", "s4"}
            self._pending_validation_indices = ["s2", "s5", "s8"]
            self._logs = []

        def _log_progress(self, msg):
            self._logs.append(msg)

    from spectral_predict_gui_optimized import SpectralPredictApp
    app = _FakeApp()
    SpectralPredictApp._apply_pending_validation_indices(app)

    # User's choice preserved.
    assert app.validation_indices == {"s3", "s4"}
    assert list(app.validation_X.index) == ["s3", "s4"]
    assert app._pending_validation_indices is None  # cleared
    assert any("ignoring" in m for m in app._logs)


def test_apply_pending_validation_indices_skip_on_missing_label():
    """If a captured label isn't in the current data's index, skip safely
    — shouldn't happen post-fingerprint-check, but defense in depth."""
    import pandas as pd

    class _FakeApp:
        def __init__(self):
            self.X = pd.DataFrame({"f1": range(5)},
                                   index=[f"s{i}" for i in range(5)])
            self.y = pd.Series(range(5), index=[f"s{i}" for i in range(5)])
            self.validation_X = None
            self.validation_y = None
            self.validation_indices = set()
            self._pending_validation_indices = ["s2", "s99"]  # s99 missing
            self._logs = []

        def _log_progress(self, msg):
            self._logs.append(msg)

    from spectral_predict_gui_optimized import SpectralPredictApp
    app = _FakeApp()
    SpectralPredictApp._apply_pending_validation_indices(app)

    assert app.validation_X is None  # untouched
    assert app.validation_indices == set()
    assert app._pending_validation_indices is None  # cleared
    assert any("not present" in m for m in app._logs)


def test_apply_pending_validation_indices_skip_on_label_only_in_X():
    """Defense-in-depth against partial mutation: if X and y indices
    disagree on a captured label, skip cleanly rather than half-mutate
    state (which would silently disable validation metrics downstream
    when validation_y ends up None or stale)."""
    import pandas as pd

    class _FakeApp:
        def __init__(self):
            self.X = pd.DataFrame({"f1": range(5)},
                                   index=[f"s{i}" for i in range(5)])
            self.y = pd.Series(range(4), index=[f"s{i}" for i in range(4)])
            self.validation_X = None
            self.validation_y = None
            self.validation_indices = set()
            self._pending_validation_indices = ["s2", "s4"]
            self._logs = []

        def _log_progress(self, msg):
            self._logs.append(msg)

    from spectral_predict_gui_optimized import SpectralPredictApp
    app = _FakeApp()
    SpectralPredictApp._apply_pending_validation_indices(app)

    assert app.validation_X is None
    assert app.validation_y is None
    assert app.validation_indices == set()
    assert app._pending_validation_indices is None
    assert any("not present" in m for m in app._logs)


def test_gui_settings_survive_through_full_sidecar_round_trip(fresh_state, populated_gui):
    """End-to-end: capture settings, persist via start_run, simulate process
    death (reset module state), find_incomplete_run reads the sidecar, the
    revived RunMetadata.gui_settings round-trips byte-for-byte. This is the
    interaction surface where T-41's auto-migration would have to preserve
    sidecar state — but the sidecar is written once at start_run and never
    re-written by migration, so the round-trip survives unchanged."""
    rs, rp, rgs, _ = fresh_state
    captured = rgs.capture_gui_settings(populated_gui)

    rs.start_run(
        label="overnight",
        dataset_fingerprint="abc",
        model_names=["pls", "ridge", "lightgbm"],
        n_trials_per_model=200,
        bayesian_persistence_mode="auto",  # the auto-migrate-eligible mode
        gui_settings=captured,
    )

    # Process death: clear in-memory module state but keep the sidecar.
    rs._reset_for_tests()

    revived = rs.find_incomplete_run()
    assert revived is not None
    assert revived.gui_settings == captured
    assert revived.bayesian_persistence_mode == "auto"
