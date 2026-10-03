"""GUI callback tests for fix/contaminant-maths (QW3, QW6, R024, R025, R075, R113, R114).

Covers the Contaminant Analysis "Apply & Validate" page (EPO output is a
spectrum, Restore, Export) and the Interference "Application" page (all six
methods run; OSC/DOSC need aligned reference values), plus the controls QW6
greys out.

Numerical thresholds hold for the controlled synthetic data built here (separable
Gaussian bands, small noise); they are not general guarantees.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

N_WL = 120
WAVELENGTHS = np.linspace(1000.0, 2190.0, N_WL)
_GRID = np.arange(N_WL)


def _band(centre: float, width: float = 5.0) -> np.ndarray:
    return np.exp(-0.5 * ((_GRID - centre) / width) ** 2)


ANALYTE = _band(30)
CONTAM = _band(85)
BASELINE = 0.5 + 0.2 * np.sin(_GRID / 25)


def _spectra(rng, n: int, contaminated: bool = False) -> np.ndarray:
    X = BASELINE + np.outer(rng.uniform(0.5, 1.5, n), ANALYTE) + rng.normal(0, 0.002, (n, N_WL))
    if contaminated:
        X = X + np.outer(rng.uniform(0.6, 1.4, n), CONTAM)
    return X


@pytest.fixture(autouse=True)
def _close_popups_and_figures(gui_app):
    """Close the before/after popups and pyplot figures each correction opens."""
    yield
    import tkinter as tk

    import matplotlib.pyplot as plt

    for child in gui_app.root.winfo_children():
        if isinstance(child, tk.Toplevel):
            child.destroy()
    plt.close("all")


@pytest.fixture
def contam_app(gui_harness):
    """Session app with clean data, one contaminant group and a main dataset."""
    app = gui_harness.app
    rng = np.random.default_rng(2026)
    app.contam_clean_data = _spectra(rng, 40)
    app.contam_groups = {"Glyptal": _spectra(rng, 40, contaminated=True)}
    app.contam_wavelengths = WAVELENGTHS
    app.contam_results = {"exclusion_regions": [(WAVELENGTHS[80], WAVELENGTHS[90])]}
    main = np.vstack([_spectra(rng, 20), _spectra(rng, 20, contaminated=True)])
    app.X = pd.DataFrame(main, index=[f"S{i}" for i in range(40)], columns=WAVELENGTHS)
    # Shared session app: clear state earlier tests may have left.
    app.X_before_contam_correction = None
    app._contam_X_written = None
    app._contam_X_fingerprint = None
    app.contam_corrected_X = None
    app.contam_correction_method.set("EPO Projection")
    app.contam_epo_components.set("auto")
    app.validation_indices = set()
    app.validation_X = None
    app.validation_y = None
    yield gui_harness
    app.validation_indices = set()
    app.validation_X = None
    app.validation_y = None
    app.contam_epo_components.set("auto")


@pytest.mark.gui
class TestContaminantCorrection:
    def test_opls_filter_is_not_a_correction_choice(self, contam_app):
        """Item 2: the OPLS-DA filter keeps the contaminant; it is not offered."""
        app = contam_app.app
        app.contam_correction_method.set("OPLS-DA Filter")
        app.contam_apply_source.set("Main Dataset")
        X_before = app.X.copy()
        with patch("tkinter.messagebox.showerror") as err:
            app._contam_apply_correction()
        assert err.called
        pd.testing.assert_frame_equal(app.X, X_before)
        assert not hasattr(app, "_apply_opls_filter")

    def test_epo_on_main_dataset_returns_spectra_without_contaminant(self, contam_app):
        app = contam_app.app
        app.contam_apply_source.set("Main Dataset")
        X_before = app.X.copy()

        contam_app.invoke_method("_contam_apply_correction")

        epo = app.contam_epo_transformer
        assert epo.n_components_ == 1
        np.testing.assert_allclose(app.X.to_numpy(), X_before.to_numpy() @ epo.P_orth_)
        # Original scale (old output was mean-centred, i.e. ~0): the 20 clean rows
        # keep most of their level. They lose a little because EPO also removes the
        # baseline's own projection on the contaminant direction; the contaminated
        # rows lose more because their band is removed.
        assert app.X.to_numpy()[:20].mean() / X_before.to_numpy()[:20].mean() > 0.8
        # The contaminant band is gone.
        c = CONTAM / np.linalg.norm(CONTAM)
        assert np.max(np.abs(app.X.to_numpy() @ c)) < 0.05 * np.max(np.abs(X_before.to_numpy() @ c))
        assert list(app.X.columns) == list(X_before.columns)
        assert str(app.contam_restore_btn.cget("state")) == "normal"

    def test_restore_undoes_main_dataset_correction(self, contam_app):
        app = contam_app.app
        app.contam_apply_source.set("Main Dataset")
        X_before = app.X.copy()

        contam_app.invoke_method("_contam_apply_correction")
        contam_app.invoke_method("_contam_apply_correction")  # stacked corrections
        contam_app.invoke_method("_contam_restore_main_dataset")

        pd.testing.assert_frame_equal(app.X, X_before)
        assert app.X_before_contam_correction is None
        assert str(app.contam_restore_btn.cget("state")) == "disabled"

    def test_restore_refuses_after_dataset_changed(self, contam_app):
        """A new dataset loaded after the correction must not be overwritten."""
        app = contam_app.app
        app.contam_apply_source.set("Main Dataset")
        contam_app.invoke_method("_contam_apply_correction")

        new_data = app.X.copy() + 1.0
        app.X = new_data
        with patch("tkinter.messagebox.showwarning") as warn:
            app._contam_restore_main_dataset()
        assert warn.called
        assert app.X is new_data

    def test_new_dataset_gets_its_own_backup(self, contam_app):
        app = contam_app.app
        app.contam_apply_source.set("Main Dataset")
        contam_app.invoke_method("_contam_apply_correction")

        second = app.X.copy() * 2.0
        app.X = second
        contam_app.invoke_method("_contam_apply_correction")
        contam_app.invoke_method("_contam_restore_main_dataset")
        pd.testing.assert_frame_equal(app.X, second)


@pytest.mark.gui
class TestRoundTwoGuards:
    """Review round 1: holdout resync, in-place edits, components control, caution."""

    def _with_holdout(self, app):
        ids = list(app.X.index[:6])
        app.validation_indices = set(ids)
        app.validation_X = app.X.loc[ids]
        app.validation_y = pd.Series(np.arange(6.0), index=ids)
        return ids

    def test_apply_and_restore_resynchronize_validation_spectra(self, contam_app):
        app = contam_app.app
        app.contam_apply_source.set("Main Dataset")
        ids = self._with_holdout(app)
        original = app.X.copy()

        contam_app.invoke_method("_contam_apply_correction")
        pd.testing.assert_frame_equal(app.validation_X, app.X.loc[ids])
        assert not np.allclose(app.validation_X.to_numpy(), original.loc[ids].to_numpy())

        contam_app.invoke_method("_contam_restore_main_dataset")
        pd.testing.assert_frame_equal(app.X, original)
        pd.testing.assert_frame_equal(app.validation_X, original.loc[ids])

    def test_in_place_edit_blocks_restore(self, contam_app):
        app = contam_app.app
        app.contam_apply_source.set("Main Dataset")
        contam_app.invoke_method("_contam_apply_correction")
        app.X.iloc[0, 0] += 1.0  # same object, edited in place
        edited = app.X.copy()
        with patch("tkinter.messagebox.showwarning") as warn:
            app._contam_restore_main_dataset()
        assert warn.called
        pd.testing.assert_frame_equal(app.X, edited)

    def test_auto_finds_nothing_explains_and_offers_override(self, contam_app):
        app = contam_app.app
        rng = np.random.default_rng(99)
        app.contam_groups = {"Same": _spectra(rng, 40)}  # no contaminant
        app.contam_apply_source.set("Main Dataset")
        X_before = app.X.copy()
        with (
            patch("tkinter.messagebox.showinfo") as info,
            patch("tkinter.messagebox.showerror") as err,
        ):
            app._contam_apply_correction()
        assert not err.called
        assert info.called and "EPO directions to remove" in info.call_args[0][1]
        pd.testing.assert_frame_equal(app.X, X_before)

        app.contam_epo_components.set("1")
        contam_app.invoke_method("_contam_apply_correction")
        assert app.contam_epo_transformer.n_components_ == 1

    def test_auto_count_is_advisory_cancel_changes_nothing(self, contam_app):
        app = contam_app.app
        app.contam_apply_source.set("Main Dataset")
        X_before = app.X.copy()
        with patch("tkinter.messagebox.askyesnocancel", return_value=None) as ask:
            app._contam_apply_correction()
        assert ask.called and "suggests removing 1" in ask.call_args[0][1]
        pd.testing.assert_frame_equal(app.X, X_before)
        assert "Cancelled" in app.contam_apply_status_label.cget("text")

    def test_auto_count_choose_a_number(self, contam_app):
        app = contam_app.app
        app.contam_apply_source.set("Contaminant Groups")
        with (
            patch("tkinter.messagebox.askyesnocancel", return_value=False),
            patch("tkinter.simpledialog.askinteger", return_value=1) as pick,
        ):
            app._contam_apply_correction()
        assert pick.called
        assert app.contam_epo_transformer.n_total_components == 1

    def test_missing_holdout_id_clears_the_holdout(self, contam_app):
        """Codex round 2: a holdout that cannot be rebuilt must not stay usable."""
        app = contam_app.app
        app.contam_apply_source.set("Main Dataset")
        ids = list(app.X.index[:4]) + ["NOT_IN_X"]
        app.validation_indices = set(ids)
        app.validation_X = pd.concat(
            [app.X.loc[ids[:4]], app.X.loc[ids[:1]].rename(index={ids[0]: "NOT_IN_X"})]
        )
        app.validation_y = pd.Series(np.arange(5.0), index=ids)
        app.validation_enabled.set(True)
        with patch("tkinter.messagebox.showinfo") as info:
            app._contam_apply_correction()
        assert app.validation_X is None and app.validation_y is None
        assert not app.validation_indices
        assert app.validation_enabled.get() is False
        assert "CLEARED" in info.call_args[0][1]

    def test_detection_with_single_spectrum_group_reports_note(self, contam_app):
        """GLM round 2: Tab 13C aborted entirely, with an API-worded message."""
        app = contam_app.app
        rng = np.random.default_rng(5)
        app.contam_groups = {
            "Glyptal": _spectra(rng, 20, contaminated=True),
            "Single": _spectra(rng, 1, contaminated=True),
        }
        app.contam_method.set("Estimated EPO")
        with (
            patch("tkinter.messagebox.showinfo") as info,
            patch("tkinter.messagebox.showerror") as err,
        ):
            app._contam_run_automated_detection()
        assert not err.called, err.call_args
        assert "EPO directions to remove" in info.call_args[0][1]

    def test_auto_count_yes_applies_the_suggestion(self, contam_app):
        app = contam_app.app
        app.contam_apply_source.set("Main Dataset")
        X_before = app.X.copy()
        with patch("tkinter.messagebox.askyesnocancel", return_value=True) as ask:
            app._contam_apply_correction()
        assert ask.called
        P = app.contam_epo_transformer.P_orth_
        np.testing.assert_allclose(app.X.to_numpy(), X_before.to_numpy() @ P)

    def test_auto_count_no_then_number_uses_the_refit_transformer(self, contam_app):
        app = contam_app.app
        app.contam_apply_source.set("Main Dataset")
        X_before = app.X.copy()
        with patch("tkinter.messagebox.askyesnocancel", return_value=False), \
             patch("tkinter.simpledialog.askinteger", return_value=1):
            app._contam_apply_correction()
        epo = app.contam_epo_transformer
        assert epo.n_total_components == 1 and epo.n_components_ == 1
        np.testing.assert_allclose(app.X.to_numpy(), X_before.to_numpy() @ epo.P_orth_)

    def test_auto_count_no_then_cancel_preserves_existing_correction(self, contam_app):
        app = contam_app.app
        app.contam_apply_source.set("Main Dataset")
        contam_app.invoke_method("_contam_apply_correction")  # Yes (fixture default)
        X_after_first = app.X.copy()
        transformer = app.contam_epo_transformer
        backup = app.X_before_contam_correction.copy()
        export_cache = app.contam_corrected_X.copy()
        with patch("tkinter.messagebox.askyesnocancel", return_value=False), \
             patch("tkinter.simpledialog.askinteger", return_value=None):
            app._contam_apply_correction()
        pd.testing.assert_frame_equal(app.X, X_after_first)
        assert app.contam_epo_transformer is transformer
        pd.testing.assert_frame_equal(app.X_before_contam_correction, backup)
        pd.testing.assert_frame_equal(app.contam_corrected_X, export_cache)
        assert "Cancelled" in app.contam_apply_status_label.cget("text")

    def test_non_advisory_mode_applies_without_asking(self, contam_app, monkeypatch):
        app = contam_app.app
        monkeypatch.setattr(type(app), "_CONTAM_AUTO_COUNT_ADVISORY", False)
        app.contam_apply_source.set("Main Dataset")
        X_before = app.X.copy()
        with patch("tkinter.messagebox.askyesnocancel") as ask:
            app._contam_apply_correction()
        assert not ask.called
        np.testing.assert_allclose(
            app.X.to_numpy(), X_before.to_numpy() @ app.contam_epo_transformer.P_orth_)

    def test_skewed_group_adds_a_dialog_warning(self, contam_app):
        app = contam_app.app
        rng = np.random.default_rng(8)
        z = rng.lognormal(size=40)
        skewed = BASELINE + np.outer(0.5 + z, ANALYTE) + rng.normal(0, 0.002, (40, N_WL))
        app.contam_clean_data = skewed
        app.contam_apply_source.set("Contaminant Groups")
        app.contam_epo_components.set("1")
        app._contam_apply_correction()
        # The skew lives along the analyte direction: point the suggestion there.
        epo = app.contam_epo_transformer
        epo.interferent_components_ = (ANALYTE / np.linalg.norm(ANALYTE))[:, None]
        assert "skewed in: clean" in app._contam_skew_warning(epo)

    def test_restore_missing_holdout_id_clears_validation(self, contam_app):
        app = contam_app.app
        app.contam_apply_source.set("Main Dataset")
        contam_app.invoke_method("_contam_apply_correction")
        ids = list(app.X.index[:3]) + ["NOT_IN_X"]
        app.validation_indices = set(ids)
        app.validation_X = pd.concat([app.X.loc[ids[:3]], app.X.loc[ids[:1]].rename(
            index={ids[0]: "NOT_IN_X"})])
        app.validation_y = pd.Series(np.arange(4.0), index=ids)
        app.validation_enabled.set(True)
        contam_app.invoke_method("_contam_restore_main_dataset")
        assert app.validation_X is None and app.validation_y is None
        assert not app.validation_indices
        assert app.validation_enabled.get() is False

    def test_epo_success_message_carries_the_caution(self, contam_app):
        app = contam_app.app
        app.contam_apply_source.set("Contaminant Groups")
        with patch("tkinter.messagebox.showinfo") as info:
            app._contam_apply_correction()
        assert "Caution" in info.call_args[0][1]
        caution = app.contam_epo_caution_label.cget("text")
        assert "differ ONLY by the contaminant" in caution
        assert "Groups of 2-3 spectra need a manual count" in caution
        assert "one of several groups" in caution


@pytest.mark.gui
class TestExportCorrectedSpectra:
    @pytest.mark.parametrize("suffix", [".csv", ".xlsx", ".npy"])
    def test_export_writes_readable_file(self, contam_app, tmp_path, suffix):
        app = contam_app.app
        app.contam_apply_source.set("Contaminant Groups")
        contam_app.invoke_method("_contam_apply_correction")
        expected = app.contam_corrected_X
        assert expected.shape == (40, N_WL)

        out = tmp_path / f"corrected{suffix}"
        with patch("tkinter.filedialog.asksaveasfilename", return_value=str(out)):
            contam_app.invoke_method("_contam_export_corrected_spectra")

        assert out.exists()
        if suffix == ".csv":
            back = pd.read_csv(out, index_col=0)
        elif suffix == ".xlsx":
            back = pd.read_excel(out, index_col=0)
        else:
            back = pd.DataFrame(np.load(out))
        np.testing.assert_allclose(back.to_numpy(dtype=float), expected.to_numpy(), rtol=1e-9)
        if suffix != ".npy":
            assert list(back.index) == list(expected.index)
        assert "exported" in app.contam_export_status_label.cget("text")

    def test_export_without_corrected_data_writes_nothing(self, contam_app, tmp_path):
        app = contam_app.app
        app.contam_corrected_X = None
        app.contam_export_status_label.config(text="")
        out = tmp_path / "nothing.csv"
        with (
            patch("tkinter.filedialog.asksaveasfilename", return_value=str(out)),
            patch("tkinter.messagebox.showwarning") as warn,
        ):
            app._contam_export_corrected_spectra()
        assert warn.called
        assert not out.exists()
        assert "exported" not in app.contam_export_status_label.cget("text")

    def test_export_failure_is_not_reported_as_success(self, contam_app, tmp_path):
        app = contam_app.app
        app.contam_apply_source.set("Contaminant Groups")
        contam_app.invoke_method("_contam_apply_correction")
        bad = tmp_path / "missing_dir" / "out.csv"
        with (
            patch("tkinter.filedialog.asksaveasfilename", return_value=str(bad)),
            patch("tkinter.messagebox.showerror") as err,
        ):
            app._contam_export_corrected_spectra()
        assert err.called
        assert "failed" in app.contam_export_status_label.cget("text")


# ---------------------------------------------------------------------------
# Interference "Application" page: all six methods
# ---------------------------------------------------------------------------


def _app_data(rng, n=60):
    y = rng.uniform(0, 1, n)
    X = (
        BASELINE
        + np.outer(y, ANALYTE)
        + np.outer(3 * rng.normal(0, 1, n), CONTAM)
        + rng.normal(0, 0.002, (n, N_WL))
    )
    return X, y


@pytest.fixture
def interference_app(gui_harness):
    app = gui_harness.app
    rng = np.random.default_rng(7)
    X, y = _app_data(rng)
    app.app_spectra = {
        "wavelengths": WAVELENGTHS,
        "X": X,
        "y": y,
        "n_spectra": len(X),
        "source": "test",
    }
    app.app_spectra_corrected = None
    app.interferent_libraries["moisture"] = {
        "wavelengths": WAVELENGTHS,
        "X": np.outer(np.linspace(0.5, 2.0, 5), CONTAM),
        "metadata": {"n_samples": 5, "n_wavelengths": N_WL},
    }
    return gui_harness


def _run_method(harness, method):
    app = harness.app
    app.app_method.set(method)
    app._populate_app_method_settings(method)
    if method == "Wavelength Exclusion":
        app.app_wl_exclude_entry.delete(0, "end")
        app.app_wl_exclude_entry.insert(0, "1500-1600")
    if method == "EPO":
        app.app_epo_library_combo.set("moisture")
        app.app_epo_library_type.set("differences")  # the fixture library is pure CONTAM
    with (
        patch("tkinter.messagebox.showerror") as err,
        patch("tkinter.messagebox.showwarning") as warn,
    ):
        harness.invoke_method("_app_apply_correction")
    return err, warn


@pytest.mark.gui
class TestInterferenceApplication:
    @pytest.mark.parametrize(
        "method", ["Wavelength Exclusion", "MSC", "OSC", "EPO", "DOSC", "GLSW"]
    )
    def test_every_method_runs(self, interference_app, method):
        """R075: Wavelength Exclusion, OSC and DOSC used to always crash."""
        err, warn = _run_method(interference_app, method)
        assert not err.called, err.call_args
        assert not warn.called, warn.call_args
        result = interference_app.app.app_spectra_corrected
        assert result is not None and result["method"] == method
        X = interference_app.app.app_spectra["X"]
        assert result["X"].shape[0] == X.shape[0]
        assert result["X"].shape[1] == len(result["wavelengths"])
        assert np.all(np.isfinite(result["X"]))

    def test_wavelength_exclusion_drops_the_range(self, interference_app):
        _run_method(interference_app, "Wavelength Exclusion")
        kept = interference_app.app.app_spectra_corrected["wavelengths"]
        assert not np.any((kept >= 1500) & (kept <= 1600))
        assert len(kept) < N_WL

    @pytest.mark.parametrize("method", ["OSC", "DOSC"])
    def test_osc_dosc_remove_y_orthogonal_band_and_keep_analyte(self, interference_app, method):
        """Controlled synthetic case: the large y-independent band goes, the
        y-related band stays (R025: old OSC removed the y-predictive direction)."""
        app = interference_app.app
        _run_method(interference_app, method)
        X = app.app_spectra["X"]
        y = app.app_spectra["y"]
        Xc = app.app_spectra_corrected["X"]
        a = ANALYTE / np.linalg.norm(ANALYTE)
        c = CONTAM / np.linalg.norm(CONTAM)
        assert np.polyfit(y, Xc @ a, 1)[0] / np.polyfit(y, X @ a, 1)[0] > 0.9
        assert np.var(Xc @ c) < 0.05 * np.var(X @ c)

    @pytest.mark.parametrize("method", ["OSC", "DOSC"])
    def test_osc_dosc_without_reference_values_explain_and_stop(self, interference_app, method):
        app = interference_app.app
        app.app_spectra["y"] = None
        err, warn = _run_method(interference_app, method)
        assert warn.called
        assert "reference" in warn.call_args[0][1].lower()
        assert app.app_spectra_corrected is None

    def test_epo_output_is_a_spectrum(self, interference_app):
        _run_method(interference_app, "EPO")
        X = interference_app.app.app_spectra["X"]
        Xc = interference_app.app.app_spectra_corrected["X"]
        c = CONTAM / np.linalg.norm(CONTAM)
        assert np.max(np.abs(Xc @ c)) < 1e-8
        assert Xc.mean() / X.mean() > 0.8

    def test_load_from_import_carries_aligned_reference_values(self, gui_harness):
        app = gui_harness.app
        rng = np.random.default_rng(3)
        X, y = _app_data(rng, 30)
        ids = [f"S{i}" for i in range(30)]
        app.X = pd.DataFrame(X, index=ids, columns=WAVELENGTHS)
        # y in a different order: alignment must be by sample label.
        app.y = pd.Series(y, index=ids).iloc[::-1]
        gui_harness.invoke_method("_app_load_from_import")

        np.testing.assert_allclose(app.app_spectra["X"], X)
        np.testing.assert_allclose(app.app_spectra["y"], y)
        np.testing.assert_allclose(app.app_spectra["wavelengths"], WAVELENGTHS)

    def test_load_from_import_with_text_labels_has_no_reference(self, gui_harness):
        app = gui_harness.app
        rng = np.random.default_rng(4)
        X, _ = _app_data(rng, 12)
        ids = [f"S{i}" for i in range(12)]
        app.X = pd.DataFrame(X, index=ids, columns=WAVELENGTHS)
        app.y = pd.Series(["bone", "tooth"] * 6, index=ids)
        gui_harness.invoke_method("_app_load_from_import")
        assert app.app_spectra["y"] is None


# ---------------------------------------------------------------------------
# QW6: controls that do nothing are disabled and say so
# ---------------------------------------------------------------------------


@pytest.mark.gui
class TestDeadControlsDisabled:
    def test_method_configuration_controls_are_disabled(self, gui_app):
        for attr in (
            "epo_enable_checkbox",
            "dosc_enable_checkbox",
            "glsw_enable_checkbox",
            "glsw_apply_to_analysis_checkbox",
        ):
            widget = getattr(gui_app, attr)
            assert widget.instate(["disabled"]), attr
            assert (
                "not applied during analysis" in str(widget.cget("text")).lower()
                or "not available" in str(widget.cget("text")).lower()
            ), attr
        assert (
            "not applied during analysis" in gui_app.interference_config_banner.cget("text").lower()
        )

    def test_glsw_apply_to_analysis_defaults_off(self, gui_app):
        assert gui_app.advanced_interference_settings["glsw"]["apply_to_analysis"].get() is False

    def test_use_msc_var_removed(self, gui_app):
        assert not hasattr(gui_app, "use_msc")

    def test_one_class_disables_preprocessing_importance_dropdown(self, gui_app):
        original = gui_app.task_type.get()
        try:
            gui_app.task_type.set("one_class")
            assert gui_app.smart_importance_combo.instate(["disabled"])
            assert "one-class" in gui_app.importance_desc_label.cget("text")
            gui_app.task_type.set("regression")
            assert not gui_app.smart_importance_combo.instate(["disabled"])
        finally:
            gui_app.task_type.set(original)
