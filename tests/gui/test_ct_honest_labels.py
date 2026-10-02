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


@pytest.fixture
def ct_app(gui_app):
    """gui_app with paired standards loaded and CT tab state restored afterwards."""
    app = gui_app
    saved = (
        app.ct_method_var.get(),
        app.ct_tsr_n_samples_var.get(),
        app.ct_roi_mode_var.get(),
        app.current_primary_data,
        app.current_satellite_data,
        app.ct_transfer_model,
        app.ct_primary_X,
        app.ct_primary_y,
        app.ct_satellite_X,
        app.ct_satellite_y,
    )
    wl, Xp, Xs = _paired_standards()
    app.current_primary_data = (wl, Xp)
    app.current_satellite_data = (wl, Xs)
    app.ct_roi_mode_var.set("full")
    app.ct_primary_data_type.set("absorbance")
    app.ct_satellite_data_type.set("absorbance")
    app.ct_transfer_model = None
    yield app, Xp, Xs
    (
        method,
        n_std,
        roi,
        app.current_primary_data,
        app.current_satellite_data,
        app.ct_transfer_model,
        app.ct_primary_X,
        app.ct_primary_y,
        app.ct_satellite_X,
        app.ct_satellite_y,
    ) = saved
    app.ct_method_var.set(method)
    app.ct_tsr_n_samples_var.set(n_std)
    app.ct_roi_mode_var.set(roi)


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
