"""QW2 / R128 / R091 on the Calibration Transfer tab.

- the default method is per-wavelength slope/bias ('tsr');
- radio labels and the guide no longer advertise CTAI / NS-PFCE / standard-free transfer;
- slope/bias fits on Kennard-Stone-selected standards (or all), never "the first n rows";
- the agreement R² is computed on the fitting rows only;
- JYPLS-inv without measured y refuses to build instead of substituting zeros.
"""

from __future__ import annotations

import tkinter as tk
from unittest.mock import patch

import numpy as np
import pytest

from spectral_predict.sample_selection import kennard_stone


def _all_widget_texts(widget) -> list[str]:
    texts = []
    try:
        texts.append(str(widget.cget("text")))
    except tk.TclError:
        pass
    for child in widget.winfo_children():
        texts.extend(_all_widget_texts(child))
    return texts


def _paired_standards(n: int = 30, p: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    wl = np.linspace(1000.0, 2000.0, p)
    X_primary = 0.5 + 0.1 * rng.standard_normal((n, 1)) + 0.05 * rng.standard_normal((n, p))
    X_satellite = 0.9 * X_primary + 0.03
    return wl, X_primary, X_satellite


_CT_ATTRS = (
    "current_primary_data",
    "current_satellite_data",
    "ct_transfer_model",
    "current_transfer_model",
    "ct_primary_X",
    "ct_primary_y",
    "ct_satellite_X",
    "ct_satellite_y",
    "ct_wavelengths_common",
    "ct_X_primary_common",
    "ct_X_satellite_common",
    "ct_loaded_model_data_type",
)
_CT_VARS = (
    "ct_method_var",
    "ct_tsr_n_samples_var",
    "ct_roi_mode_var",
    "ct_roi_start_var",
    "ct_roi_end_var",
    "ct_primary_data_type",
    "ct_satellite_data_type",
    "ct_load_model_path_var",
)
_CT_TEXTS = ("ct_transfer_info_text", "ct_loaded_model_info_text")
_CT_LABELS = ("ct_mode_a_transfer_status_label", "ct_mode_b_transfer_status_label")


def _text_get(widget) -> tuple[str, str]:
    return str(widget.cget("state")), widget.get("1.0", "end-1c")


def _text_set(widget, saved: tuple[str, str]) -> None:
    state, content = saved
    widget.config(state="normal")
    widget.delete("1.0", tk.END)
    widget.insert("1.0", content)
    widget.config(state=state)


@pytest.fixture
def ct_app(gui_app):
    """gui_app with paired standards loaded; all CT state it touches is restored after."""
    app = gui_app
    missing = object()
    attrs = {name: getattr(app, name, missing) for name in _CT_ATTRS}
    tk_vars = {name: getattr(app, name).get() for name in _CT_VARS}
    texts = {name: _text_get(getattr(app, name)) for name in _CT_TEXTS}
    save_state = str(app.ct_save_tm_button.cget("state"))
    labels = {
        name: (str(getattr(app, name).cget("text")), str(getattr(app, name).cget("foreground")))
        for name in _CT_LABELS
    }

    wl, Xp, Xs = _paired_standards()
    app.current_primary_data = (wl, Xp)
    app.current_satellite_data = (wl, Xs)
    app.ct_roi_mode_var.set("full")
    app.ct_primary_data_type.set("absorbance")
    app.ct_satellite_data_type.set("absorbance")
    app.ct_transfer_model = None
    app.current_transfer_model = None
    yield app, Xp, Xs

    for name, value in attrs.items():
        if value is missing:
            if hasattr(app, name):
                delattr(app, name)
        else:
            setattr(app, name, value)
    for name, value in tk_vars.items():
        getattr(app, name).set(value)
    for name, value in texts.items():
        _text_set(getattr(app, name), value)
    app.ct_save_tm_button.config(state=save_state)
    for name, (text, foreground) in labels.items():
        getattr(app, name).config(text=text, foreground=foreground)
    for widget in app.ct_transfer_plot_frame.winfo_children():
        widget.destroy()


def test_default_method_is_slope_bias(gui_app):
    """A fresh app selects 'tsr' (fixtures restore the var after each CT test)."""
    assert gui_app.ct_method_var.get() == "tsr"
    assert gui_app.ct_tsr_n_samples_var.get() == "All"


def test_tab_text_has_no_stale_method_claims(gui_app):
    texts = "\n".join(_all_widget_texts(gui_app.tab10))
    for stale in (
        "Feature-based",
        "<10 standards",
        "No transfer standards available",
        "Cross-Transfer Adaptive Interpolation",
        "NS-PFCE",
        "CTAI",
    ):
        assert stale not in texts, stale
    for label in ("Slope/bias per wavelength", "PC-DS", "Iterative ridge DS"):
        assert label in texts, label


def test_tooltips_drop_false_claims():
    from spectral_predict_gui_optimized import TOOLTIP_CONTENT

    tips = TOOLTIP_CONTENT["calibration_transfer"]
    joined = " ".join(tips.values())
    assert "Adaptive Integration" not in joined
    assert "Null-Space Projection" not in joined
    assert "Kennard-Stone" in tips["param_tsr_samples"]
    assert "not trimmed scores regression" in tips["method_TSR"]


def test_slope_bias_subset_uses_kennard_stone_standards(ct_app):
    app, Xp, Xs = ct_app
    app.ct_method_var.set("tsr")
    app.ct_tsr_n_samples_var.set("8")
    with patch.object(app, "_plot_transfer_quality"):
        app._build_transfer_model_new()

    tm = app.ct_transfer_model
    assert tm is not None and tm.method == "tsr"
    idx = np.asarray(tm.params["transfer_indices"])
    np.testing.assert_array_equal(idx, kennard_stone(Xp, n_samples=8))
    assert set(idx.tolist()) != set(range(8)), "must not be the first n rows"
    assert tm.params["standard_selection"] == "kennard-stone"


def test_slope_bias_all_uses_every_pair(ct_app):
    app, Xp, Xs = ct_app
    app.ct_method_var.set("tsr")
    app.ct_tsr_n_samples_var.set("All")
    with patch.object(app, "_plot_transfer_quality"):
        app._build_transfer_model_new()

    tm = app.ct_transfer_model
    np.testing.assert_array_equal(tm.params["transfer_indices"], np.arange(len(Xp)))
    np.testing.assert_allclose(tm.params["slope"], 1 / 0.9)


def test_slope_bias_too_many_standards_refused(ct_app):
    app, Xp, _ = ct_app
    app.ct_method_var.set("tsr")
    app.ct_tsr_n_samples_var.set(str(len(Xp) + 5))
    with patch.object(app, "_plot_transfer_quality"), patch("tkinter.messagebox.showerror") as err:
        app._build_transfer_model_new()
    assert app.ct_transfer_model is None
    assert err.called


def test_agreement_r2_uses_only_fitting_rows():
    from spectral_predict_gui_optimized import ct_spectral_agreement_on_fit_rows

    rng = np.random.default_rng(4)
    Xp = rng.standard_normal((20, 10))
    X_out = Xp.copy()
    unused = np.arange(10, 20)
    X_out[unused] += 100.0  # rows the slope/bias fit never saw
    params = {"transfer_indices": np.arange(10)}

    r2_tsr, rows = ct_spectral_agreement_on_fit_rows(Xp, X_out, "tsr", params)
    np.testing.assert_array_equal(rows, np.arange(10))
    assert r2_tsr == pytest.approx(1.0)

    # Methods fitted on every loaded row report on every row.
    r2_ds, rows_ds = ct_spectral_agreement_on_fit_rows(Xp, X_out, "ds", {})
    assert len(rows_ds) == 20
    assert r2_ds < 0.5


def test_quality_plot_renders_for_slope_bias_subset(ct_app):
    """The real plot path runs (its errors are only printed, so catch the print)."""
    app, _, _ = ct_app
    app.ct_method_var.set("tsr")
    app.ct_tsr_n_samples_var.set("8")
    with patch("builtins.print") as fake_print:
        app._build_transfer_model_new()
    printed = " ".join(str(c) for c in fake_print.call_args_list)
    assert "Error creating transfer quality plots" not in printed
    assert app.ct_transfer_plot_frame.winfo_children()


def test_quality_plot_with_region_of_interest(ct_app):
    """An ROI model is fitted on clipped columns; the plot must clip too (round 1, item 9)."""
    from spectral_predict_gui_optimized import ct_region_arrays

    app, Xp, _ = ct_app
    wl = app.current_primary_data[0]
    app.ct_method_var.set("tsr")
    app.ct_tsr_n_samples_var.set("All")
    app.ct_roi_mode_var.set("roi")
    app.ct_roi_start_var.set(f"{wl[10]:.3f}")
    app.ct_roi_end_var.set(f"{wl[40]:.3f}")
    with patch("builtins.print") as fake_print:
        app._build_transfer_model_new()

    tm = app.ct_transfer_model
    assert tm is not None
    n_roi = tm.meta["region_of_interest"]["n_wavelengths_region"]
    assert len(tm.params["slope"]) == n_roi < Xp.shape[1]
    printed = " ".join(str(c) for c in fake_print.call_args_list)
    assert "Error creating transfer quality plots" not in printed
    assert app.ct_transfer_plot_frame.winfo_children()

    X_pri, X_sat, wl_roi = ct_region_arrays(
        app.ct_X_primary_common, app.ct_X_satellite_common, app.ct_wavelengths_common, tm.meta
    )
    assert X_pri.shape[1] == X_sat.shape[1] == len(wl_roi) == n_roi


@pytest.mark.parametrize("n_roi", [1, 2, 4])
def test_quality_plot_with_narrow_region_keeps_raw_and_scatter(ct_app, n_roi):
    """A region too narrow for the SG derivative must not hide the other plots (round 2)."""
    from spectral_predict_gui_optimized import ct_derivative_window

    app, Xp, _ = ct_app
    wl = app.current_primary_data[0]
    app.ct_method_var.set("tsr")
    app.ct_tsr_n_samples_var.set("All")
    app.ct_roi_mode_var.set("roi")
    app.ct_roi_start_var.set(f"{wl[10] - 1.0:.3f}")
    app.ct_roi_end_var.set(f"{wl[10 + n_roi - 1] + 1.0:.3f}")
    with patch("builtins.print") as fake_print:
        app._build_transfer_model_new()

    tm = app.ct_transfer_model
    assert tm is not None
    assert tm.meta["region_of_interest"]["n_wavelengths_region"] == n_roi
    printed = " ".join(str(c) for c in fake_print.call_args_list)
    assert "Error creating transfer quality plots" not in printed

    children = app.ct_transfer_plot_frame.winfo_children()
    notebooks = [w for w in children if w.winfo_class() == "TNotebook"]
    assert len(notebooks) == 1
    tab_texts = [notebooks[0].tab(t, "text") for t in notebooks[0].tabs()]
    assert tab_texts[0] == "Raw Spectra"
    if ct_derivative_window(n_roi) is None:
        assert tab_texts == ["Raw Spectra", "Derivatives"]
    else:
        assert tab_texts == ["Raw Spectra", "1st Derivative", "2nd Derivative"]
    # notebook + scatter canvas + its export button: the scatter was drawn
    assert len(children) >= 3


@pytest.mark.parametrize(
    ("n", "expected"),
    [(1, None), (2, None), (3, 3), (4, 3), (5, 5), (6, 5), (12, 11), (200, 11)],
)
def test_derivative_window(n, expected):
    from spectral_predict_gui_optimized import ct_derivative_window

    assert ct_derivative_window(n) == expected


def test_region_arrays_without_roi_are_unchanged():
    from spectral_predict_gui_optimized import ct_region_arrays

    X = np.ones((3, 5))
    wl = np.arange(5.0)
    out = ct_region_arrays(X, X, wl, {"note": "no roi"})
    assert out[0] is X and out[2] is wl
    assert ct_region_arrays(X, X, wl, None)[0] is X


@pytest.mark.parametrize("fmt", ["pkl", "json"])
def test_loaded_model_info_uses_display_name(ct_app, tmp_path, fmt):
    """Load Existing Transfer Model shows PC-DS, not the raw 'ctai' key (round 1, item 2)."""
    import pickle

    from spectral_predict.calibration_transfer import (
        TransferModel,
        estimate_ctai,
        save_transfer_model,
    )

    app, Xp, Xs = ct_app
    wl = app.current_primary_data[0]
    tm = TransferModel("p", "s", "ctai", wl, estimate_ctai(Xp, Xs, n_components=3), meta={})
    if fmt == "pkl":
        path = tmp_path / "tm.pkl"
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": tm,
                    "method": "ctai",
                    "primary_id": "p",
                    "satellite_id": "s",
                    "wavelengths_common": wl,
                },
                f,
            )
    else:
        path = str(save_transfer_model(tm, tmp_path, name="tm")) + ".json"
    app.ct_load_model_path_var.set(str(path))
    app._load_existing_transfer_model()

    shown = app.ct_loaded_model_info_text.get("1.0", "end-1c")
    assert "PC-DS" in shown
    assert "Method: ctai" not in shown


def test_agreement_title_says_not_a_validation():
    from spectral_predict_gui_optimized import CT_AGREEMENT_TITLE

    assert "includes fitting data" in CT_AGREEMENT_TITLE
    assert "not a validation" in CT_AGREEMENT_TITLE


def test_jypls_without_reference_values_refuses(ct_app):
    """The dormant JYPLS-inv path must not build with placeholder zeros (R091)."""
    app, _, _ = ct_app
    app.ct_primary_X = None
    app.ct_primary_y = None
    app.ct_method_var.set("jypls-inv")
    with patch.object(app, "_plot_transfer_quality"), patch("tkinter.messagebox.showerror") as err:
        app._build_transfer_model_new()
    assert app.ct_transfer_model is None
    assert err.called
    assert "reference value" in str(err.call_args)


def test_jypls_radio_still_disabled(gui_app):
    def find_radio(widget):
        for child in widget.winfo_children():
            try:
                if (
                    child.winfo_class() == "TRadiobutton"
                    and str(child.cget("value")) == "jypls-inv"
                ):
                    return child
            except tk.TclError:
                pass
            found = find_radio(child)
            if found is not None:
                return found
        return None

    radio = find_radio(gui_app.tab10)
    assert radio is not None
    assert "disabled" in str(radio.cget("state"))
