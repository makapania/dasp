"""F2 part 2 on the Calibration Transfer tab: pairing, validation panel, planning.

Commit-scope groups:
- pairing the A1 standards by sample ID (never row order; SPYDER found 0/40 correct
  by row order), pairs-table CSV, group column, visible satellite resampling;
- the A1b Validate & Compare panel (evaluate_transfer board, joins, value scale,
  fit-selected through the existing save path, physical_match end to end);
- the Plan Transfer Standards dialog (Kennard-Stone).
"""

from __future__ import annotations

import tkinter as tk
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


def _paired_standards_df(n: int = 12, p: int = 80, seed: int = 0):
    """Primary/satellite DataFrames for the same specimens, satellite shuffled."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(1000.0, 2000.0, p)
    X_primary = 0.5 + 0.1 * rng.standard_normal((n, 1)) + 0.05 * rng.standard_normal((n, p))
    X_satellite = 0.9 * X_primary + 0.03
    ids = [f"bone_{i + 1:02d}" for i in range(n)]
    primary = pd.DataFrame(X_primary, index=ids, columns=wl)
    order = rng.permutation(n)
    shuffled_ids = [ids[i] for i in order]
    satellite = pd.DataFrame(X_satellite[order], index=shuffled_ids, columns=wl)
    return wl, primary, satellite


_CT_STATE_ATTRS = (
    "current_primary_data",
    "current_satellite_data",
    "current_primary_df",
    "current_satellite_df",
    "ct_eval_model_dict",
    "_ct_validate_running",
    "_ct_eval_context",
    "current_primary_ids_positional",
    "current_satellite_ids_positional",
    "ct_primary_header_joins",
    "ct_satellite_header_joins",
    "ct_paired",
    "ct_pair_groups",
    "ct_pair_groups_table_keys",
    "ct_pairs_rename",
    "ct_pairs_table_active",
    "ct_pairs_table_named_primary",
    "ct_pairs_table_named_satellite",
    "ct_pairs_collisions",
    "ct_pair_resample_note",
    "ct_pairs_table_unresolved",
    "ct_last_evaluation",
    "ct_transfer_model",
    "current_transfer_model",
    "ct_wavelengths_common",
    "ct_X_primary_common",
    "ct_X_satellite_common",
    "ct_pairs_table_path_var",
    "ct_pair_group_entry_var",
    "_ct_input_generation",
    "_ct_run_seq",
    "_ct_active_run_id",
    "ct_primary_reflectance_scale",
    "ct_satellite_reflectance_scale",
    "new_satellite_data_predict",
    "current_prediction_model_dict",
    "current_prediction_model",
    "ct_pred_value_scale",
    "ct_pred_y_pred",
)


@pytest.fixture
def ct_app(gui_app):
    """gui_app with standards loaded as the A1 loaders leave them; state restored."""
    app = gui_app
    missing = object()
    attrs = {name: getattr(app, name, missing) for name in _CT_STATE_ATTRS}
    tree_items = list(app.ct_pairs_tree.get_children())
    status_text = str(app.ct_pairs_status_label.cget("text"))
    status_fg = str(app.ct_pairs_status_label.cget("foreground"))
    save_state = str(app.ct_save_tm_button.cget("state"))

    wl, primary, satellite = _paired_standards_df()
    app.current_primary_data = (wl, primary.to_numpy())
    app.current_satellite_data = (wl, satellite.to_numpy())
    app.current_primary_df = primary
    app.current_satellite_df = satellite
    app.current_primary_ids_positional = False
    app.current_satellite_ids_positional = False
    app.ct_paired = None
    app.ct_pair_groups = None
    app.ct_pairs_rename = None
    app.ct_last_evaluation = None
    app.ct_transfer_model = None
    app.ct_eval_model_dict = None
    app._ct_validate_running = False
    app._ct_eval_context = None
    app.ct_eval_space_var.set("auto (detected)")
    app.ct_eval_joins_var.set("auto")
    app.ct_primary_data_type.set("reflectance")
    app.ct_satellite_data_type.set("reflectance")
    app.ct_roi_mode_var.set("full")
    yield app, wl, primary, satellite

    for name, value in attrs.items():
        if value is missing:
            if hasattr(app, name):
                delattr(app, name)
        else:
            setattr(app, name, value)
    for item in list(app.ct_pairs_tree.get_children()):
        app.ct_pairs_tree.delete(item)
    app.ct_pairs_status_label.config(text=status_text, foreground=status_fg)
    app.ct_save_tm_button.config(state=save_state)
    for widget in app.ct_transfer_plot_frame.winfo_children():
        widget.destroy()


# ---------------------------------------------------------------------------
# Commit 1: pairing by sample ID
# ---------------------------------------------------------------------------


def test_pair_by_id_survives_shuffled_satellite(ct_app):
    """Row order paired 0/40 on a real set; ID pairing must be order-immune."""
    app, wl, primary, satellite = ct_app
    with patch.object(app, "_refresh_ct_pairs_tree"):
        paired = app._pair_ct_standards()

    assert paired is not None
    assert paired.ids == list(primary.index)
    # Same specimens in the same order as the primary: satellite row j really is
    # 0.9 * primary row j + 0.03 (the shuffle would break row-order pairing).
    np.testing.assert_allclose(paired.X_satellite, 0.9 * paired.X_primary + 0.03)
    assert app.ct_paired is paired


def test_pairing_reports_unmatched_ids(ct_app):
    app, wl, primary, satellite = ct_app
    satellite_extra = pd.concat(
        [
            satellite,
            pd.DataFrame(0.4 + 0.0 * satellite.iloc[:1].to_numpy(), index=["ghost"], columns=wl),
        ]
    )
    app.current_satellite_df = satellite_extra.drop(index="bone_01")
    paired = app._pair_ct_standards()

    assert paired is not None
    assert "bone_01" in paired.unmatched_primary
    assert "ghost" in paired.unmatched_satellite
    status = str(app.ct_pairs_status_label.cget("text"))
    assert "1 primary-only" in status and "1 satellite-only" in status
    tree_text = "\n".join(
        " ".join(str(v) for v in app.ct_pairs_tree.item(item, "values"))
        for item in app.ct_pairs_tree.get_children()
    )
    assert "(no match)" in tree_text


def test_duplicate_ids_refused(ct_app):
    app, wl, primary, satellite = ct_app
    sat = satellite.copy()
    sat.index = [sat.index[0]] * len(sat)  # every row the same ID
    app.current_satellite_df = sat
    with patch("tkinter.messagebox.showerror") as err:
        assert app._pair_ct_standards() is None
    assert err.called
    assert app.ct_paired is None


def test_pairs_table_renames_and_seeds_groups(ct_app, tmp_path):
    """Different file names between instruments pair through the table."""
    app, wl, primary, satellite = ct_app
    # Satellite files carry their own names; the table is the ground truth for
    # which satellite file is which primary specimen (satellite rows are shuffled,
    # so each row keeps its own specimen's LAB name, not a positional one).
    sat = satellite.copy()
    sat.index = [f"LAB-B{int(i.split('_')[1]):02d}.asd" for i in satellite.index]
    app.current_satellite_df = sat
    table = pd.DataFrame(
        {
            "primary": list(primary.index),
            "satellite": [f"LAB-B{int(i.split('_')[1]):02d}.asd" for i in primary.index],
            "group": ["unitA"] * 6 + ["unitB"] * (len(primary) - 6),
        }
    )
    table_path = tmp_path / "pairs.csv"
    table.to_csv(table_path, index=False)
    app.ct_pairs_table_path_var.set(str(table_path))

    app._load_ct_pairs_table()
    assert app.ct_pairs_rename
    paired = app._pair_ct_standards()
    assert paired is not None
    assert paired.ids == list(primary.index)
    np.testing.assert_allclose(paired.X_satellite, 0.9 * paired.X_primary + 0.03)
    assert app.ct_pair_groups["bone_01"] == "unitA"
    assert app.ct_pair_groups["bone_12"] == "unitB"


def test_group_set_on_selected_rows(ct_app):
    app, _, _, _ = ct_app
    app._pair_ct_standards()
    items = app.ct_pairs_tree.get_children()
    app.ct_pairs_tree.selection_set(items[:3])
    app.ct_pair_group_entry_var.set("session2")
    app._ct_set_group_on_selected()
    assert app.ct_pair_groups is not None
    assert len(app.ct_pair_groups) == 3
    assert app.ct_pair_groups["bone_01"] == "session2"
    fresh = app.ct_pairs_tree.get_children()  # the refresh replaced the items
    row = app.ct_pairs_tree.item(fresh[0], "values")
    assert row[2] == "session2"


def test_differing_grids_resample_visibly(ct_app):
    app, wl, primary, _ = ct_app
    # Smooth in wavelength, so linear interpolation between the coarse satellite
    # points recovers the primary-grid values (noise would not interpolate).
    smooth = 0.4 + 0.05 * np.sin(wl[None, :] / 120.0 + np.arange(primary.shape[0])[:, None])
    primary = primary.copy()
    primary.loc[:, :] = smooth
    wl_coarse = wl[::2]
    satellite = pd.DataFrame((0.9 * smooth + 0.03)[:, ::2], index=primary.index, columns=wl_coarse)
    app.current_primary_df = primary
    app.current_satellite_df = satellite
    paired = app._pair_ct_standards()

    assert paired is not None
    # Onto the primary grid restricted to the overlap (the coarser satellite
    # stops one primary step short of 2000 nm; no extrapolation).
    expected_wl = wl[wl <= wl_coarse[-1]]
    assert paired.wavelengths.shape == expected_wl.shape
    note = app.ct_pair_resample_note
    assert note and "resampled" in note and "step" in note
    assert note in str(app.ct_pairs_status_label.cget("text"))
    np.testing.assert_allclose(paired.X_satellite, 0.9 * paired.X_primary + 0.03, atol=1e-3)


def test_too_small_overlap_refuses(ct_app):
    app, wl, primary, _ = ct_app
    satellite = pd.DataFrame(
        0.9 * primary.to_numpy() + 0.03,
        index=primary.index,
        columns=np.linspace(2000.0, 2100.0, primary.shape[1]),
    )
    app.current_satellite_df = satellite
    with patch("tkinter.messagebox.showerror") as err:
        assert app._pair_ct_standards() is None
    assert err.called


def test_positional_ids_are_flagged(ct_app):
    app, wl, primary, satellite = ct_app
    app.current_primary_ids_positional = True
    with patch.object(app, "_refresh_ct_pairs_tree"):
        app._pair_ct_standards()
    assert "positional" in str(app.ct_pairs_status_label.cget("text"))


def test_directory_load_keeps_sample_ids(gui_app, tmp_path):
    """A legacy-CSV folder load keeps the file stems as sample IDs."""
    app = gui_app
    rng = np.random.default_rng(1)
    wl = np.arange(1000.0, 1010.0)
    for stem in ("a", "b", "c"):
        row = 0.4 + 0.01 * rng.standard_normal(len(wl))
        pd.DataFrame(np.vstack([wl, row])).to_csv(
            tmp_path / f"{stem}.csv", index=False, header=False
        )
    path = str(tmp_path)
    app.ct_primary_data_path_var.set(path)
    with patch.object(app, "_update_data_info"), patch.object(app, "play_sound"):
        app._load_primary_spectra()

    df = app.current_primary_df
    assert list(df.index) == ["a", "b", "c"]
    assert not app.current_primary_ids_positional
    assert list(df.columns) == [float(w) for w in wl]


def test_single_file_load_gets_positional_ids(gui_app, tmp_path):
    app = gui_app
    wl = np.arange(1000.0, 1010.0)
    X = 0.4 + 0.01 * np.random.default_rng(2).standard_normal((4, len(wl)))
    csv_path = tmp_path / "spectra.csv"
    pd.DataFrame(np.vstack([wl, X])).to_csv(csv_path, index=False, header=False)
    app.ct_primary_data_path_var.set(str(csv_path))
    with patch.object(app, "_update_data_info"), patch.object(app, "play_sound"):
        app._load_primary_spectra()

    assert app.current_primary_ids_positional
    assert list(app.current_primary_df.index) == [f"row_{i + 1}" for i in range(4)]


def test_build_uses_paired_rows_not_row_order(ct_app):
    """A slope/bias build on shuffled standards recovers the true 1/0.9 slope."""
    app, _, _, satellite = ct_app
    app.ct_method_var.set("tsr")
    app.ct_tsr_n_samples_var.set("All")
    with patch.object(app, "_plot_transfer_quality"):
        app._build_transfer_model_new()

    tm = app.ct_transfer_model
    assert tm is not None and tm.method == "tsr"
    np.testing.assert_allclose(tm.params["slope"], 1 / 0.9, rtol=1e-6)
    assert tm.meta["pairing"]["by"] == "sample_id"


def test_dead_load_ct_paired_spectra_is_gone(gui_app):
    assert not hasattr(gui_app, "_load_ct_paired_spectra")


# ---------------------------------------------------------------------------
# Commit 2: the Validate & Compare panel
# ---------------------------------------------------------------------------


class _InlineThread:
    """threading.Thread stand-in that runs the worker synchronously."""

    def __init__(self, target=None, daemon=False, **kwargs):
        self._target = target

    def start(self):
        self._target()


@pytest.fixture
def inline_workers(ct_app):
    """Workers run synchronously; root.after fires the callback immediately."""
    app = ct_app[0]
    with (
        patch.object(
            type(app.root),
            "after",
            lambda self, ms, fn=None: fn() if callable(fn) else None,
        ),
        patch("threading.Thread", _InlineThread),
    ):
        yield ct_app


def _board_labels(app):
    return [
        str(app.ct_leaderboard_tree.item(item, "values")[0])
        for item in app.ct_leaderboard_tree.get_children()
    ]


def test_validate_board_has_physical_rows(inline_workers):
    app, wl, primary, satellite = inline_workers
    with patch("tkinter.messagebox.showerror") as err:
        app._run_ct_validate()
    assert not err.called, err.call_args_list

    labels = _board_labels(app)
    assert any("Physical match" in label for label in labels)
    assert any("No correction" in label for label in labels)
    assert app.ct_last_evaluation is not None
    assert app._ct_eval_context is not None
    # No reference values (decision 1): no reference row can exist
    assert not any("reference" in label.lower() for label in labels)
    assert "RMSD_vs_primary" in app.ct_leaderboard_tree.cget("columns")
    diag = app.ct_diagnostics_text.get("1.0", "end-1c")
    assert "Ranked by" in diag and "Joins used" in diag
    assert str(app.ct_fit_selected_button.cget("state")) == "normal"


def test_validate_with_groups_ranks_by_grouped_columns(inline_workers):
    app, _, _, _ = inline_workers
    groups = {f"bone_{i + 1:02d}": ("unitA" if i < 6 else "unitB") for i in range(12)}
    app.ct_pair_groups = groups
    with patch("tkinter.messagebox.showerror") as err:
        app._run_ct_validate()
    assert not err.called, err.call_args_list

    columns = app.ct_leaderboard_tree.cget("columns")
    assert "RMSD_vs_primary_grouped" in columns
    assert "Slope_vs_primary_grouped" in columns
    # No prediction model loaded: the grouped rank column is the spectral one
    assert app.ct_last_evaluation.score_column == "spectral_RMSE_grouped"
    header = str(app.ct_leaderboard_header_label.cget("text"))
    assert "spectral_RMSE_grouped" in header


def test_type_mismatch_refuses_before_running(inline_workers):
    app, _, _, _ = inline_workers
    app.ct_satellite_data_type.set("absorbance")
    started = []
    with patch("threading.Thread", side_effect=lambda *a, **k: started.append(1)):
        with patch("tkinter.messagebox.showerror") as err:
            app._run_ct_validate()
    assert err.called
    assert not started
    assert app.ct_last_evaluation is None


def test_joins_override_reaches_evaluate(inline_workers):
    app, _, _, _ = inline_workers
    captured = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop here")

    import spectral_predict.transfer_evaluation as te

    with patch.object(te, "evaluate_transfer", fake_evaluate):
        with patch("tkinter.messagebox.showerror"):
            app._run_ct_validate()
    assert captured["joins"] == "auto"
    assert captured["ids"] == list(app.ct_paired.ids)
    assert np.allclose(captured["wavelengths"], app.ct_paired.wavelengths)

    # header joins come from the per-instrument attribute captured at load time
    app.ct_primary_header_joins = (1000.0, 1800.0)
    app.ct_eval_joins_var.set("ASD file header")
    with patch.object(te, "evaluate_transfer", fake_evaluate):
        with patch("tkinter.messagebox.showerror"):
            app._run_ct_validate()
    assert captured["joins"] == (1000.0, 1800.0)

    app.ct_eval_joins_var.set("custom")
    app.ct_eval_joins_custom_var.set("1005, 1810")
    with patch.object(te, "evaluate_transfer", fake_evaluate):
        with patch("tkinter.messagebox.showerror"):
            app._run_ct_validate()
    assert captured["joins"] == (1005.0, 1810.0)

    app.ct_eval_joins_var.set("none")
    with patch.object(te, "evaluate_transfer", fake_evaluate):
        with patch("tkinter.messagebox.showerror"):
            app._run_ct_validate()
    assert captured["joins"] is None


def test_percent_reflectance_divided_before_evaluate(inline_workers):
    app, wl, primary, satellite = inline_workers
    app.ct_primary_reflectance_scale = 100.0
    app.ct_satellite_reflectance_scale = 100.0
    captured = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop here")

    import spectral_predict.transfer_evaluation as te

    contexts = []
    with (
        patch.object(te, "evaluate_transfer", fake_evaluate),
        patch.object(
            app, "_on_ct_validate_done", lambda payload, context: contexts.append(context)
        ),
    ):
        with patch("tkinter.messagebox.showerror"):
            app._run_ct_validate()
    # the arrays are pairing-ordered (primary order), each divided by its scale
    np.testing.assert_allclose(captured["X_primary"], app.ct_paired.X_primary / 100.0)
    np.testing.assert_allclose(captured["X_satellite"], app.ct_paired.X_satellite / 100.0)
    assert "divided by 100" in contexts[0]["scale_note"]


def _physical_standards(n=14, p=160, seed=5):
    """0-1 reflectance pairs with band structure; satellite bands deeper + blurred."""
    from scipy.ndimage import gaussian_filter1d

    wl = np.linspace(1000.0, 2200.0, p)
    t = np.linspace(0.0, 1.0, n)
    level = 0.60 + 0.04 * np.sin(2 * np.pi * t)
    band1 = 0.10 * np.exp(-(((wl - 1450.0) / 70.0) ** 2))
    band2 = 0.06 * np.exp(-(((wl - 1950.0) / 90.0) ** 2))
    rng = np.random.default_rng(seed)
    A_p = np.zeros((n, p))
    for i in range(n):
        depth = t[i]
        spectrum = level[i] - depth * band1 - (1 - depth) * band2
        A_p[i] = -np.log10(np.clip(spectrum, 1e-3, None))
    A_p += 0.002 * rng.standard_normal(A_p.shape)
    # satellite: blur + deepen, i.e. the physical match must undo both
    A_s = gaussian_filter1d(A_p * 1.25, sigma=1.1, axis=1)
    X_p = 10**-A_p
    X_s = 10**-A_s
    ids = [f"std{i + 1:02d}" for i in range(n)]
    return (
        wl,
        pd.DataFrame(X_p, index=ids, columns=wl),
        pd.DataFrame(X_s, index=ids, columns=wl),
    )


@pytest.fixture
def physical_app(gui_app):
    """gui_app loaded with physical-match-shaped standards, ready to validate."""
    app = gui_app
    saved = {
        name: getattr(app, name, None)
        for name in (
            "ct_paired",
            "ct_pair_groups",
            "ct_last_evaluation",
            "ct_transfer_model",
            "current_transfer_model",
            "ct_eval_model_dict",
            "_ct_eval_context",
            "_ct_validate_running",
            "ct_primary_reflectance_scale",
            "ct_satellite_reflectance_scale",
            "new_satellite_data_predict",
            "new_satellite_data_export",
            "current_prediction_model_dict",
            "current_prediction_model",
            "ct_pred_value_scale",
            "ct_pred_data_type",
            "ct_pred_data_converted",
            "ct_pred_source_data_type",
            "ct_pred_y_pred",
            "ct_export_value_scale",
            "ct_export_data_type",
            "ct_export_data_converted",
            "transformed_spectra",
            "transformed_spectra_type",
        )
    }
    wl, primary, satellite = _physical_standards()
    app.current_primary_df = primary
    app.current_satellite_df = satellite
    app.ct_primary_data_type.set("reflectance")
    app.ct_satellite_data_type.set("reflectance")
    app.ct_primary_reflectance_scale = 1.0
    app.ct_satellite_reflectance_scale = 1.0
    app.ct_pair_groups = None
    app.ct_pairs_rename = None
    app.ct_transfer_model = None
    app.current_transfer_model = None
    app.ct_eval_model_dict = None
    app.ct_last_evaluation = None
    app._ct_eval_context = None
    app._ct_validate_running = False
    yield app, wl, primary, satellite
    for name, value in saved.items():
        setattr(app, name, value)
    for widget in app.ct_transfer_plot_frame.winfo_children():
        widget.destroy()


def _run_inline(app, fn):
    """Run a worker method with the thread and after() inlined."""
    with (
        patch.object(
            type(app.root),
            "after",
            lambda self, ms, fn=None: fn() if callable(fn) else None,
        ),
        patch("threading.Thread", _InlineThread),
    ):
        with patch("tkinter.messagebox.showerror") as err:
            fn()
    assert not err.called, err.call_args_list


def _select_physical_row(app):
    target = next(
        item
        for item in app.ct_leaderboard_tree.get_children()
        if "Physical match" in str(app.ct_leaderboard_tree.item(item, "values")[0])
    )
    app.ct_leaderboard_tree.selection_set(target)


def test_fit_selected_physical_match_through_existing_paths(physical_app, tmp_path):
    """Requirement B: fit, quality plot, save button, save -> load, apply."""
    from spectral_predict.calibration_transfer import (
        apply_transfer_dispatch,
        load_transfer_model,
        resample_to_grid,
        save_transfer_model,
    )

    app, wl, primary, satellite = physical_app
    app._pair_ct_standards()
    _run_inline(app, app._run_ct_validate)

    labels = _board_labels(app)
    assert any("Physical match" in label for label in labels)
    _select_physical_row(app)
    _run_inline(app, app._fit_ct_selected)

    tm = app.ct_transfer_model
    assert tm is not None and tm.method == "physical_match"
    assert tm.meta["value_scale"] == 1.0  # fitted on 0-1 reflectance
    assert str(app.ct_save_tm_button.cget("state")) == "normal"
    assert "Physical match" in app.ct_transfer_info_text.get("1.0", "end-1c")
    # the quality plot rendered for physical_match (no "unsupported" bail-out)
    assert app.ct_transfer_plot_frame.winfo_children()

    # save -> load through the GUI's load-existing path
    prefix = save_transfer_model(tm, tmp_path, name="phys")
    app.ct_load_model_path_var.set(str(prefix) + ".json")
    app._load_existing_transfer_model()
    loaded = app.current_transfer_model
    assert loaded is not None and loaded.method == "physical_match"
    # dispatch gets the wavelengths it needs from the model itself
    X_transferred = apply_transfer_dispatch(satellite.to_numpy(), loaded)
    assert X_transferred.shape == satellite.shape

    # different-grid satellite data is resampled onto wavelengths_common
    X_common = resample_to_grid(
        satellite.to_numpy(), satellite.columns.to_numpy(), loaded.wavelengths_common
    )
    np.testing.assert_allclose(apply_transfer_dispatch(X_common, loaded), X_transferred)


def test_percent_scale_consistent_between_fit_and_apply(physical_app):
    """Requirement A: fit on percent standards, apply to percent satellite data."""
    from spectral_predict.calibration_transfer import apply_transfer_dispatch

    app, wl, primary, satellite = physical_app
    # percent-scale loads: the panel divides to 0-1 for fit and records it
    app.ct_primary_reflectance_scale = 100.0
    app.ct_satellite_reflectance_scale = 100.0
    app._pair_ct_standards()
    _run_inline(app, app._run_ct_validate)
    _select_physical_row(app)
    _run_inline(app, app._fit_ct_selected)
    tm = app.ct_transfer_model
    assert tm is not None

    X_new_percent = 100.0 * satellite.to_numpy()  # percent-scale new satellite data
    # without conversion, physical match refuses percent data outright
    with pytest.raises(ValueError):
        apply_transfer_dispatch(X_new_percent, tm)

    # the GUI funnel converts to the fit scale: the result is the fraction-scale
    # computation (the scale the Validate board scored the model on)
    out = app._apply_transfer_model(X_new_percent, tm, value_scale=100.0)
    expected_fraction = apply_transfer_dispatch(X_new_percent / 100.0, tm)
    np.testing.assert_allclose(out, expected_fraction)
    # Mode B asks for the incoming scale: the fraction computation x100
    out_incoming = app._apply_transfer_model(
        X_new_percent, tm, value_scale=100.0, output_scale="incoming"
    )
    np.testing.assert_allclose(out_incoming, expected_fraction * 100.0)
    # without the meta the GUI cannot convert, and physical match refuses
    # percent data — exactly the apply-time failure the meta prevents
    tm.meta.pop("value_scale")
    with pytest.raises(ValueError):
        app._apply_transfer_model(X_new_percent, tm, value_scale=100.0)


def test_physical_match_mode_a_prediction_end_to_end(physical_app, tmp_path):
    """Requirement B: Mode A predicts through a panel-fitted physical match."""
    from sklearn.cross_decomposition import PLSRegression
    from spectral_predict import model_io
    from spectral_predict.calibration_transfer import apply_transfer_dispatch

    app, wl, primary, satellite = physical_app
    app._pair_ct_standards()
    _run_inline(app, app._run_ct_validate)
    _select_physical_row(app)
    _run_inline(app, app._fit_ct_selected)
    tm = app.ct_transfer_model
    assert tm is not None

    # a regression model trained on the primary (fraction reflectance)
    y = np.linspace(1.0, 9.0, primary.shape[0])
    pls = PLSRegression(n_components=2).fit(primary.to_numpy(), y)
    model_path = tmp_path / "primary.dasp"
    model_io.save_model(
        pls,
        None,
        {
            "model_name": "PLS",
            "task_type": "regression",
            "wavelengths": [float(w) for w in wl],
            "n_vars": len(wl),
            "target_name": "y",
        },
        str(model_path),
    )
    app.current_prediction_model_dict = model_io.load_model(str(model_path))
    app.current_prediction_model = app.current_prediction_model_dict["model"]

    # new satellite data in PERCENT reflectance, on the transfer's grid
    X_new_percent = 100.0 * satellite.to_numpy()
    app.new_satellite_data_predict = (wl, X_new_percent)
    app.ct_pred_data_type.set("reflectance")
    app.ct_pred_value_scale = 100.0
    app.ct_pred_data_converted = False
    app.ct_pred_source_data_type = None

    with patch("tkinter.messagebox.showwarning"):
        _run_inline(app, app._run_prediction_workflow)

    expected = pls.predict(apply_transfer_dispatch(X_new_percent / 100.0, tm)).ravel()
    got = np.asarray(app.ct_pred_y_pred, dtype=float)
    np.testing.assert_allclose(got, expected, rtol=1e-6)
    assert len(app.ct_predictions_tree.get_children()) == len(y)


def test_physical_match_mode_b_export_end_to_end(physical_app):
    """Requirement B: Mode B transforms and exports in the primary's scale.

    The primary loaded as fraction reflectance here, so the export is fraction
    even though the new satellite data came in as percent.
    """
    from spectral_predict.calibration_transfer import apply_transfer_dispatch

    app, wl, primary, satellite = physical_app
    app._pair_ct_standards()
    _run_inline(app, app._run_ct_validate)
    _select_physical_row(app)
    _run_inline(app, app._fit_ct_selected)
    tm = app.ct_transfer_model

    X_new_percent = 100.0 * satellite.to_numpy()
    app.new_satellite_data_export = (wl, X_new_percent)
    app.ct_export_data_type.set("reflectance")
    app.ct_export_value_scale = 100.0
    app.ct_export_data_converted = False

    _run_inline(app, app._transform_spectra)

    wl_out, X_out = app.transformed_spectra
    assert len(wl_out) == len(tm.wavelengths_common)
    expected = apply_transfer_dispatch(X_new_percent / 100.0, tm)
    np.testing.assert_allclose(X_out, expected)
    assert "fraction" in app.ct_transform_stats_text.get("1.0", "end-1c")
    assert app.transformed_spectra_type["value_scale"] == 1.0


def test_eval_model_loader_rejects_non_regression(ct_app, tmp_path):
    app, *_ = ct_app
    app.ct_eval_model_path_var.set(str(tmp_path / "model.dasp"))
    with patch.object(
        app,
        "_load_model_file_for_ct",
        return_value={"metadata": {"task_type": "classification"}},
    ):
        with patch("tkinter.messagebox.showerror") as err:
            app._load_ct_eval_model()
    assert err.called
    assert app.ct_eval_model_dict is None


# ---------------------------------------------------------------------------
# Review fixes: conversion/reload invalidation + input generation (B, C)
# ---------------------------------------------------------------------------


def _dummy_transfer_model(wl):
    from spectral_predict.calibration_transfer import TransferModel

    return TransferModel(
        "p", "s", "tsr", np.asarray(wl, dtype=float),
        {"slope": np.ones(len(wl)), "bias": np.zeros(len(wl))}, {})


def test_convert_updates_pairing_data_and_invalidates(ct_app):
    """B: Convert must not leave the pairing DataFrames with the old values."""
    app, wl, primary, satellite = ct_app
    app._pair_ct_standards()
    assert app.ct_paired is not None
    stale = _dummy_transfer_model(wl)
    app.ct_transfer_model = stale
    app.current_transfer_model = stale
    app.ct_save_tm_button.config(state='normal')
    gen = getattr(app, "_ct_input_generation", 0)

    with patch.object(app, "_update_data_info"):
        app._ct_primary_convert()
        app._ct_satellite_convert()

    # both DataFrames carry the converted values; IDs and grid untouched
    np.testing.assert_allclose(
        app.current_primary_df.to_numpy(), np.log10(1.0 / primary.to_numpy()),
        rtol=1e-12)
    np.testing.assert_allclose(
        app.current_satellite_df.to_numpy(), np.log10(1.0 / satellite.to_numpy()),
        rtol=1e-12)
    assert list(app.current_primary_df.index) == list(primary.index)
    assert list(app.current_primary_df.columns) == list(primary.columns)
    # nothing derived from the unconverted values survives
    assert app.ct_paired is None
    assert app.ct_last_evaluation is None and app._ct_eval_context is None
    assert app.ct_transfer_model is None and app.current_transfer_model is None
    assert str(app.ct_save_tm_button.cget("state")) == "disabled"
    assert getattr(app, "_ct_input_generation", 0) >= gen + 2


def test_reload_clears_derived_state(inline_workers):
    """C: reloading standards must clear pairing, board, diagnostics and model."""
    app, wl, primary, satellite = inline_workers
    app._pair_ct_standards()
    with patch("tkinter.messagebox.showerror") as err:
        app._run_ct_validate()
    assert not err.called
    assert app.ct_leaderboard_tree.get_children()
    assert app.ct_diagnostics_text.get("1.0", "end-1c")

    stale = _dummy_transfer_model(wl)
    app.ct_transfer_model = stale
    app.current_transfer_model = stale
    app.ct_save_tm_button.config(state='normal')
    app.ct_pair_groups = {"bone_01": "unitA"}

    app._ct_store_loaded_standards(
        'primary', 'folder', wl, primary.to_numpy(), None)

    assert app.ct_paired is None
    assert app.ct_last_evaluation is None and app._ct_eval_context is None
    assert app.ct_pair_groups is None and app.ct_pairs_rename is None
    assert app.ct_transfer_model is None and app.current_transfer_model is None
    assert app.ct_X_primary_common is None and app.ct_wavelengths_common is None
    assert str(app.ct_save_tm_button.cget("state")) == "disabled"
    assert not app.ct_pairs_tree.get_children()
    assert not app.ct_leaderboard_tree.get_children()
    assert app.ct_diagnostics_text.get("1.0", "end-1c") == ""
    assert str(app.ct_fit_selected_button.cget("state")) == "disabled"

    # a transfer model loaded from a FILE is not panel state and survives
    loaded = _dummy_transfer_model(wl)
    app.current_transfer_model = loaded
    app._ct_store_loaded_standards(
        'satellite', 'folder', wl, satellite.to_numpy(), None)
    assert app.current_transfer_model is loaded


def test_reload_reapplies_pairs_table(ct_app, tmp_path):
    """C (choice): a loaded pairs table is re-applied to the reloaded files."""
    app, wl, primary, satellite = ct_app
    sat_names = [f"LAB-B{int(i.split('_')[1]):02d}" for i in satellite.index]
    sat = satellite.copy()
    sat.index = sat_names
    app.current_satellite_df = sat
    table = pd.DataFrame(
        {"primary": list(primary.index), "satellite": sat_names})
    table_path = tmp_path / "pairs.csv"
    table.to_csv(table_path, index=False)
    app.ct_pairs_table_path_var.set(str(table_path))
    app._load_ct_pairs_table()
    assert len(app.ct_pairs_rename) == len(sat_names)

    # reload only half the satellite files: the table is re-checked against
    # what is loaded now (the vanished rows are re-reported, not kept silently)
    keep = list(sat.index[:6])
    app._last_dir_ids = keep
    try:
        with patch("tkinter.messagebox.showwarning") as warn:
            app._ct_store_loaded_standards(
                'satellite', str(tmp_path), wl, sat.loc[keep].to_numpy(), None)
        assert warn.called
    finally:
        app._last_dir_ids = None
    assert sorted(app.ct_pairs_rename) == sorted(keep)
    assert "6 pairs" in str(app.ct_pairs_status_label.cget("text"))


def test_stale_validate_result_discarded(inline_workers):
    """C: the input generation discards results computed on replaced inputs."""
    app, wl, primary, satellite = inline_workers
    contexts = []
    with (
        patch.object(app, "_on_ct_validate_done",
                     lambda payload, context: contexts.append((payload, context))),
        patch("tkinter.messagebox.showerror"),
    ):
        app._run_ct_validate()
    payload, context = contexts[0]
    assert context["generation"] == getattr(app, "_ct_input_generation", 0)

    app._ct_store_loaded_standards('primary', 'folder', wl, primary.to_numpy(), None)
    assert context["generation"] != app._ct_input_generation
    assert app._ct_active_run_id is None  # the reload abandoned the run (R4-3)
    status_before = str(app.ct_validate_status_label.cget("text"))
    app._on_ct_validate_done(payload, context)

    assert app.ct_last_evaluation is None  # the stale board was not installed
    assert app._ct_validate_running is False
    # the abandoned run's callback leaves the status untouched (R4-3)
    assert str(app.ct_validate_status_label.cget("text")) == status_before


def test_stale_fit_result_discarded(ct_app):
    """C: a fit finishing after the inputs changed installs nothing."""
    app, *_ = ct_app
    app._ct_input_generation = 5
    app.ct_transfer_model = None
    app.current_transfer_model = None

    app._on_ct_fit_done((True, None, ""), "No correction",
                        {"generation": 4,
                         "run_id": getattr(app, "_ct_active_run_id", 0)})

    assert app.ct_transfer_model is None and app.current_transfer_model is None
    assert app._ct_validate_running is False
    assert "discarded" in str(app.ct_validate_status_label.cget("text"))


# ---------------------------------------------------------------------------
# Review fix A: prediction-model scale (percent-trained .dasp models)
# ---------------------------------------------------------------------------


def _regression_model_on(tmp_path, X, y, wl, name="PLS"):
    """A fitted, saved and loaded regression model trained on X."""
    from sklearn.cross_decomposition import PLSRegression
    from spectral_predict import model_io

    pls = PLSRegression(n_components=2).fit(X, y)
    path = tmp_path / f"{name}.dasp"
    model_io.save_model(
        pls, None,
        {
            "model_name": name,
            "task_type": "regression",
            "wavelengths": [float(w) for w in wl],
            "n_vars": len(wl),
            "target_name": "y",
        },
        str(path),
    )
    return model_io.load_model(str(path))


def test_board_scales_percent_trained_model_back_up(inline_workers, tmp_path):
    """A: a percent-trained model must see percent-scale spectra on the board."""
    app, wl, primary, satellite = inline_workers
    # load both instruments as percent reflectance (values 0-100)
    primary_pct, satellite_pct = primary * 100.0, satellite * 100.0
    app.current_primary_df = primary_pct
    app.current_satellite_df = satellite_pct
    app.current_primary_data = (wl, primary_pct.to_numpy())
    app.current_satellite_data = (wl, satellite_pct.to_numpy())
    app.ct_paired = None
    app.ct_primary_reflectance_scale = 100.0
    app.ct_satellite_reflectance_scale = 100.0
    y = np.linspace(1.0, 9.0, primary.shape[0])
    model = _regression_model_on(tmp_path, primary_pct.to_numpy(), y, wl)
    app.ct_eval_model_dict = model
    with patch("tkinter.messagebox.showerror") as err:
        app._run_ct_validate()
    assert not err.called, err.call_args_list

    board = app.ct_last_evaluation.leaderboard
    got = float(board.loc[board["method"] == "none", "RMSD_vs_primary"].iloc[0])
    pls = model["model"]
    # the model was trained on the primary's loaded (percent) data, so both the
    # satellite predictions and the primary reference rows are percent-scale;
    # the paired arrays are in primary order (the satellite df is shuffled) and
    # already carry the loaded percent values
    pred_sat = pls.predict(app.ct_paired.X_satellite).ravel()
    pred_primary = pls.predict(app.ct_paired.X_primary).ravel()
    expected = float(np.sqrt(np.mean((pred_sat - pred_primary) ** 2)))
    assert got == pytest.approx(expected, rel=1e-8)


def test_mode_a_scales_transferred_spectra_to_primary_scale(physical_app, tmp_path):
    """A: Mode A hands a percent-trained model percent-scale transferred spectra."""
    from spectral_predict.calibration_transfer import apply_transfer_dispatch

    app, wl, primary, satellite = physical_app
    # both instruments loaded as percent reflectance (values 0-100); the panel
    # fits on the divided 0-1 spectra
    primary_pct, satellite_pct = primary * 100.0, satellite * 100.0
    app.current_primary_df = primary_pct
    app.current_satellite_df = satellite_pct
    app.ct_paired = None
    app.ct_primary_reflectance_scale = 100.0
    app.ct_satellite_reflectance_scale = 100.0
    app._pair_ct_standards()
    _run_inline(app, app._run_ct_validate)
    _select_physical_row(app)
    _run_inline(app, app._fit_ct_selected)
    tm = app.ct_transfer_model
    assert tm is not None
    assert tm.meta["value_scale"] == 1.0  # the fit scale
    assert tm.meta["primary_value_scale"] == 100.0  # the primary's loaded scale

    # a model trained on the primary's loaded (percent) data
    y = np.linspace(1.0, 9.0, primary.shape[0])
    model = _regression_model_on(tmp_path, primary_pct.to_numpy(), y, wl)
    app.current_prediction_model_dict = model
    app.current_prediction_model = model["model"]

    X_new_percent = satellite_pct.to_numpy()
    app.new_satellite_data_predict = (wl, X_new_percent)
    app.ct_pred_data_type.set("reflectance")
    app.ct_pred_value_scale = 100.0
    app.ct_pred_data_converted = False
    app.ct_pred_source_data_type = None

    with patch("tkinter.messagebox.showwarning"):
        _run_inline(app, app._run_prediction_workflow)

    pls = model["model"]
    expected = pls.predict(
        apply_transfer_dispatch(X_new_percent / 100.0, tm) * 100.0).ravel()
    np.testing.assert_allclose(np.asarray(app.ct_pred_y_pred), expected, rtol=1e-6)


def test_mode_b_exports_in_primary_scale_with_note(physical_app):
    """A: Mode B writes transferred spectra in the primary's loaded scale."""
    from spectral_predict.calibration_transfer import apply_transfer_dispatch

    app, wl, primary, satellite = physical_app
    primary_pct, satellite_pct = primary * 100.0, satellite * 100.0
    app.current_primary_df = primary_pct
    app.current_satellite_df = satellite_pct
    app.ct_paired = None
    app.ct_primary_reflectance_scale = 100.0
    app.ct_satellite_reflectance_scale = 100.0
    app._pair_ct_standards()
    _run_inline(app, app._run_ct_validate)
    _select_physical_row(app)
    _run_inline(app, app._fit_ct_selected)
    tm = app.ct_transfer_model

    # the NEW satellite data loads as fraction reflectance while the transfer's
    # primary was percent: the export must still look like primary-instrument data
    X_new_fraction = satellite.to_numpy()
    app.new_satellite_data_export = (wl, X_new_fraction)
    app.ct_export_data_type.set("reflectance")
    app.ct_export_value_scale = 1.0
    app.ct_export_data_converted = False

    _run_inline(app, app._transform_spectra)

    wl_out, X_out = app.transformed_spectra
    assert len(wl_out) == len(tm.wavelengths_common)
    expected = apply_transfer_dispatch(X_new_fraction, tm) * 100.0
    np.testing.assert_allclose(X_out, expected)
    stats = app.ct_transform_stats_text.get("1.0", "end-1c")
    assert "percent" in stats
    assert app.transformed_spectra_type["value_scale"] == 100.0


def test_old_transfer_file_without_scale_keys_behaves_as_before(physical_app, tmp_path):
    """A: transfer files recording neither scale key get no conversion at all."""
    from spectral_predict.calibration_transfer import apply_transfer_dispatch

    app, wl, primary, satellite = physical_app
    app._pair_ct_standards()
    _run_inline(app, app._run_ct_validate)
    _select_physical_row(app)
    _run_inline(app, app._fit_ct_selected)
    tm = app.ct_transfer_model
    tm.meta.pop("value_scale", None)
    tm.meta.pop("primary_value_scale", None)

    # apply: fraction in -> fraction out, no conversion anywhere
    X = satellite.to_numpy()
    np.testing.assert_allclose(
        app._apply_transfer_model(X, tm, value_scale=1.0),
        apply_transfer_dispatch(X, tm))

    # Mode A: the model is fed exactly what the apply produced
    y = np.linspace(1.0, 9.0, primary.shape[0])
    model = _regression_model_on(tmp_path, primary.to_numpy(), y, wl)
    app.current_prediction_model_dict = model
    app.current_prediction_model = model["model"]
    app.new_satellite_data_predict = (wl, X)
    app.ct_pred_data_type.set("reflectance")
    app.ct_pred_value_scale = 1.0
    app.ct_pred_data_converted = False
    app.ct_pred_source_data_type = None
    with patch("tkinter.messagebox.showwarning"):
        _run_inline(app, app._run_prediction_workflow)
    expected = model["model"].predict(apply_transfer_dispatch(X, tm)).ravel()
    np.testing.assert_allclose(np.asarray(app.ct_pred_y_pred), expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Review fix E: fitting "No correction" must clear the active transfer
# ---------------------------------------------------------------------------


def test_no_correction_fit_clears_active_transfer(physical_app):
    """E: a previously fitted transfer must not stay active after 'No correction'."""
    app, wl, primary, satellite = physical_app
    app._pair_ct_standards()
    _run_inline(app, app._run_ct_validate)
    _select_physical_row(app)
    _run_inline(app, app._fit_ct_selected)
    assert app.ct_transfer_model is not None
    assert str(app.ct_save_tm_button.cget("state")) == "normal"

    target = next(
        item
        for item in app.ct_leaderboard_tree.get_children()
        if "No correction" in str(app.ct_leaderboard_tree.item(item, "values")[0])
    )
    app.ct_leaderboard_tree.selection_set(target)
    _run_inline(app, app._fit_ct_selected)

    assert app.ct_transfer_model is None
    assert app.current_transfer_model is None
    assert str(app.ct_save_tm_button.cget("state")) == "disabled"
    assert "No transfer model active" in str(
        app.ct_mode_a_transfer_status_label.cget("text"))
    assert "No transfer model active" in str(
        app.ct_mode_b_transfer_status_label.cget("text"))
    # Mode A refuses rather than silently applying the stale transfer
    app.new_satellite_data_predict = (wl, satellite.to_numpy())
    app.current_prediction_model_dict = {"metadata": {}}
    app.current_prediction_model = None
    with patch("tkinter.messagebox.showerror") as err:
        app._run_prediction_workflow()
    assert err.called
    assert "transfer model" in str(err.call_args)


# ---------------------------------------------------------------------------
# Review fix D: refuse to extrapolate new satellite grids (Mode A and Mode B)
# ---------------------------------------------------------------------------


def _fitted_physical_transfer(app):
    app._pair_ct_standards()
    _run_inline(app, app._run_ct_validate)
    _select_physical_row(app)
    _run_inline(app, app._fit_ct_selected)
    return app.ct_transfer_model


def test_mode_a_refuses_uncovered_satellite_grid(physical_app, tmp_path):
    """D: new spectra narrower than the fitted grid must be refused, not extrapolated."""
    from spectral_predict.calibration_transfer import resample_to_grid

    app, wl, primary, satellite = physical_app
    tm = _fitted_physical_transfer(app)
    assert tm is not None

    y = np.linspace(1.0, 9.0, primary.shape[0])
    model = _regression_model_on(tmp_path, primary.to_numpy(), y, wl)
    app.current_prediction_model_dict = model
    app.current_prediction_model = model["model"]
    inside = slice(8, -8)  # strictly inside the fitted range
    narrow_wl = wl[inside]
    X_narrow = resample_to_grid(satellite.to_numpy(), wl, narrow_wl)
    app.new_satellite_data_predict = (narrow_wl, X_narrow)
    app.ct_pred_data_type.set("reflectance")
    app.ct_pred_value_scale = 1.0
    app.ct_pred_data_converted = False
    app.ct_pred_source_data_type = None
    app.ct_pred_y_pred = None

    with patch("tkinter.messagebox.showerror") as err:
        app._run_prediction_workflow()
    assert err.called
    msg = str(err.call_args)
    assert f"{narrow_wl.min():.1f}" in msg  # the loaded range is named
    assert f"{wl.min():.1f}" in msg  # the fitted range is named
    assert app.ct_pred_y_pred is None


def test_mode_a_allows_covering_wider_grid(physical_app, tmp_path):
    """D: a wider grid that covers the fitted range interpolates and predicts."""
    from spectral_predict.calibration_transfer import (
        apply_transfer_dispatch, resample_to_grid)

    app, wl, primary, satellite = physical_app
    tm = _fitted_physical_transfer(app)

    wide_wl = np.linspace(wl[0] - 40.0, wl[-1] + 40.0, wl.size + 40)
    X_wide = resample_to_grid(satellite.to_numpy(), wl, wide_wl)
    y = np.linspace(1.0, 9.0, primary.shape[0])
    model = _regression_model_on(tmp_path, primary.to_numpy(), y, wl)
    app.current_prediction_model_dict = model
    app.current_prediction_model = model["model"]
    app.new_satellite_data_predict = (wide_wl, X_wide)
    app.ct_pred_data_type.set("reflectance")
    app.ct_pred_value_scale = 1.0
    app.ct_pred_data_converted = False
    app.ct_pred_source_data_type = None

    with patch("tkinter.messagebox.showwarning"):
        _run_inline(app, app._run_prediction_workflow)

    X_on_grid = resample_to_grid(X_wide, wide_wl, tm.wavelengths_common)
    expected = model["model"].predict(apply_transfer_dispatch(X_on_grid, tm)).ravel()
    np.testing.assert_allclose(np.asarray(app.ct_pred_y_pred), expected, rtol=1e-6)


def test_mode_b_refuses_uncovered_satellite_grid(physical_app):
    """D: Mode B must not extrapolate a narrower grid either."""
    from spectral_predict.calibration_transfer import resample_to_grid

    app, wl, primary, satellite = physical_app
    tm = _fitted_physical_transfer(app)

    inside = slice(8, -8)
    narrow_wl = wl[inside]
    X_narrow = resample_to_grid(satellite.to_numpy(), wl, narrow_wl)
    app.new_satellite_data_export = (narrow_wl, X_narrow)
    app.ct_export_data_type.set("reflectance")
    app.ct_export_value_scale = 1.0
    app.ct_export_data_converted = False
    app.transformed_spectra = None

    with patch("tkinter.messagebox.showerror") as err:
        app._transform_spectra()
    assert err.called
    msg = str(err.call_args)
    assert f"{narrow_wl.min():.1f}" in msg and f"{wl.min():.1f}" in msg
    assert app.transformed_spectra is None


# ---------------------------------------------------------------------------
# Review fixes F (pairs-table file extensions) and G (descending grids)
# ---------------------------------------------------------------------------


def test_pairs_table_names_with_extensions_match_stems(ct_app, tmp_path):
    """F: 'sample1.asd' in the table must match the loaded stem 'sample1'."""
    app, wl, primary, satellite = ct_app
    # satellite files carry their own stems; each row keeps its own specimen
    sat = satellite.copy()
    sat.index = [f"s{int(i.split('_')[1]):02d}" for i in satellite.index]
    app.current_satellite_df = sat
    table = pd.DataFrame({
        "primary": list(primary.index),
        "satellite": [f"S{int(i.split('_')[1]):02d}.ASD" for i in primary.index],
    })
    table_path = tmp_path / "pairs.csv"
    table.to_csv(table_path, index=False)
    app.ct_pairs_table_path_var.set(str(table_path))

    with patch("tkinter.messagebox.showerror") as err:
        app._load_ct_pairs_table()
    assert not err.called, err.call_args_list
    assert len(app.ct_pairs_rename) == len(sat)
    paired = app._pair_ct_standards()
    assert paired is not None
    assert paired.ids == list(primary.index)
    np.testing.assert_allclose(paired.X_satellite, 0.9 * paired.X_primary + 0.03)


def test_descending_grids_pair_on_sorted_columns(ct_app):
    """G: descending wavelength grids pair as if sorted ascending."""
    app, wl, primary, satellite = ct_app
    app.current_primary_df = primary[primary.columns[::-1]]
    app.current_satellite_df = satellite[satellite.columns[::-1]]
    paired = app._pair_ct_standards()
    assert paired is not None
    assert np.all(np.diff(np.asarray(paired.wavelengths)) > 0)  # ascending
    np.testing.assert_allclose(paired.X_satellite, 0.9 * paired.X_primary + 0.03)

    # mixed directions previously produced an empty overlap and a refusal
    app.current_primary_df = primary  # ascending again
    paired = app._pair_ct_standards()
    assert paired is not None
    assert np.all(np.diff(np.asarray(paired.wavelengths)) > 0)
    np.testing.assert_allclose(paired.X_satellite, 0.9 * paired.X_primary + 0.03)


# ---------------------------------------------------------------------------
# Review fix H: ROI apply must convert the whole row, not splice mixed scales
# ---------------------------------------------------------------------------


def test_roi_apply_converts_whole_spectrum_not_just_region(ct_app):
    """H: the ROI splice works in one scale for the whole spectrum."""
    app, wl, primary, satellite = ct_app
    app.ct_primary_data_type.set("reflectance")
    app.ct_satellite_data_type.set("reflectance")
    app.ct_primary_reflectance_scale = 1.0
    app.ct_satellite_reflectance_scale = 1.0
    app.ct_method_var.set("tsr")
    app.ct_tsr_n_samples_var.set("All")
    app.ct_roi_mode_var.set("roi")
    app.ct_roi_start_var.set(f"{wl[10]:.1f}")
    app.ct_roi_end_var.set(f"{wl[60]:.1f}")
    with patch.object(app, "_plot_transfer_quality"), \
         patch("tkinter.messagebox.showinfo"):
        app._build_transfer_model_new()
    tm = app.ct_transfer_model
    assert tm is not None
    assert tm.meta["region_of_interest"]["enabled"]
    assert tm.meta["value_scale"] == 1.0  # fitted on fraction reflectance

    from spectral_predict.calibration_transfer import apply_transfer_dispatch

    idx = np.array(tm.meta["region_of_interest"]["indices"])
    inside = np.zeros(wl.size, bool)
    inside[idx] = True
    X_pct = satellite.to_numpy() * 100.0  # percent-scale incoming data

    # Mode A asks for the fit scale: the WHOLE spectrum comes back in it
    out_fit = app._apply_transfer_with_roi(X_pct, tm, value_scale=100.0)
    np.testing.assert_allclose(out_fit[:, ~inside], X_pct[:, ~inside] / 100.0)
    np.testing.assert_allclose(
        out_fit[:, inside],
        apply_transfer_dispatch(X_pct[:, idx] / 100.0, tm))

    # Mode B asks for the incoming scale: the whole spectrum comes back in it
    out_in = app._apply_transfer_with_roi(
        X_pct, tm, value_scale=100.0, output_scale="incoming")
    np.testing.assert_allclose(out_in[:, ~inside], X_pct[:, ~inside])
    np.testing.assert_allclose(
        out_in[:, inside],
        apply_transfer_dispatch(X_pct[:, idx] / 100.0, tm) * 100.0)


# ---------------------------------------------------------------------------
# Commit 3: Plan Transfer Standards (Kennard-Stone)
# ---------------------------------------------------------------------------


def test_plan_standards_dialog_kennard_stone(ct_app, tmp_path):
    from spectral_predict.sample_selection import kennard_stone

    app, wl, primary, satellite = ct_app
    try:
        app._open_ct_plan_standards()
        assert app._ct_plan_dialog is not None
        app.ct_plan_source_var.set("primary")
        app.ct_plan_n_var.set("5")
        _run_inline(app, app._run_ct_plan_standards)

        expected = [primary.index[i] for i in kennard_stone(primary.to_numpy(), n_samples=5)]
        rows = [app.ct_plan_tree.item(item, "values") for item in app.ct_plan_tree.get_children()]
        assert [r[1] for r in rows] == expected
        assert [r[0] for r in rows] == ["1", "2", "3", "4", "5"]
        assert not app._ct_plan_running

        out_path = tmp_path / "standards.csv"
        with patch("tkinter.filedialog.asksaveasfilename", return_value=str(out_path)):
            app._export_ct_plan_standards()
        text = out_path.read_text(encoding="utf-8").strip().splitlines()
        assert text[0] == "order,sample_id"
        assert text[1] == f"1,{expected[0]}"
    finally:
        app._close_ct_plan_standards()
    assert app._ct_plan_dialog is None


def test_plan_standards_refuses_bad_count(ct_app):
    app, *_ = ct_app
    app._open_ct_plan_standards()
    try:
        app.ct_plan_source_var.set("primary")
        app.ct_plan_n_var.set("0")
        with patch("tkinter.messagebox.showwarning") as warn:
            app._run_ct_plan_standards()
        assert warn.called
        app.ct_plan_n_var.set("99")
        with patch("tkinter.messagebox.showwarning") as warn:
            app._run_ct_plan_standards()
        assert warn.called
    finally:
        app._close_ct_plan_standards()


# ---------------------------------------------------------------------------
# Review round 2 (2026-10-17)
# ---------------------------------------------------------------------------


def _fit_percent_transfer(app, primary, satellite, row_prefix):
    """Panel-fit a transfer on percent-loaded standards; return the model.

    Both instruments are loaded as percent reflectance (the panel divides to
    0-1 for the fit), so the model records ``value_scale=1.0`` (the fit scale)
    and ``primary_value_scale=100.0`` (the primary's loaded scale).
    """
    app.current_primary_df = primary * 100.0
    app.current_satellite_df = satellite * 100.0
    app.ct_paired = None
    app.ct_primary_reflectance_scale = 100.0
    app.ct_satellite_reflectance_scale = 100.0
    app._pair_ct_standards()
    _run_inline(app, app._run_ct_validate)
    target = next(
        item
        for item in app.ct_leaderboard_tree.get_children()
        if str(app.ct_leaderboard_tree.item(item, "values")[0]).startswith(row_prefix)
    )
    app.ct_leaderboard_tree.selection_set(target)
    _run_inline(app, app._fit_ct_selected)
    tm = app.ct_transfer_model
    assert tm is not None
    assert tm.meta["value_scale"] == 1.0
    assert tm.meta["primary_value_scale"] == 100.0
    return tm


def _provision_equalize_ui(app):
    """Create the Section F widgets/vars _file_equalize_batch reads.

    The File Equalize helpers exist but their frame is not wired into the tab
    (no call site creates the vars), so a test of the export path provisions
    exactly the state a wired UI would hold. Returns a restore() callable.
    """
    names = ("ct_eq_tm_source_var", "ct_eq_input_dir_var", "ct_eq_output_dir_var",
             "ct_eq_file_org_var", "ct_eq_format_var", "ct_eq_status_text")
    saved = {name: getattr(app, name, None) for name in names
             if hasattr(app, name)}
    app.ct_eq_tm_source_var = tk.StringVar(value="current")
    app.ct_eq_input_dir_var = tk.StringVar()
    app.ct_eq_output_dir_var = tk.StringVar()
    app.ct_eq_file_org_var = tk.StringVar(value="single")
    app.ct_eq_format_var = tk.StringVar(value="csv")
    app.ct_eq_status_text = tk.Text(app.root)

    def restore():
        app.ct_eq_status_text.destroy()
        for name in names:
            if hasattr(app, name):
                delattr(app, name)
        for name, value in saved.items():
            setattr(app, name, value)

    return restore


@pytest.mark.parametrize("row_prefix", ["Physical match", "DS"])
def test_file_equalize_exports_in_primary_scale(physical_app, tmp_path, row_prefix):
    """R2-1: File Equalize detects the folder's scale and exports primary-scale."""
    from spectral_predict.calibration_transfer import apply_transfer_dispatch

    app, wl, primary, satellite = physical_app
    tm = _fit_percent_transfer(app, primary, satellite, row_prefix)

    # a folder of percent-reflectance satellite CSVs (one spectrum per file)
    X_pct = (satellite * 100.0).to_numpy()
    in_dir = tmp_path / "satellite_pct"
    in_dir.mkdir()
    header = ",".join(repr(float(w)) for w in wl)
    for i, sid in enumerate(satellite.index):
        (in_dir / f"{sid}.csv").write_text(
            header + "\n" + ",".join(repr(float(v)) for v in X_pct[i]) + "\n",
            encoding="utf-8")
    out_dir = tmp_path / "equalized"
    out_dir.mkdir()

    restore = _provision_equalize_ui(app)
    try:
        app.ct_eq_input_dir_var.set(str(in_dir))
        app.ct_eq_output_dir_var.set(str(out_dir))
        with (patch("tkinter.messagebox.showinfo"),
              patch("tkinter.messagebox.showerror") as err):
            app._file_equalize_batch()
        assert not err.called, err.call_args_list

        (out_file,) = list(out_dir.glob("*transformed_spectra*.csv"))
        exported = pd.read_csv(out_file, index_col=0)
        assert exported.shape == X_pct.shape
        # percent in -> converted to the fit scale -> transferred -> exported
        # in the primary's loaded (percent) scale, like Mode B
        expected = apply_transfer_dispatch(X_pct / 100.0, tm) * 100.0
        np.testing.assert_allclose(exported.to_numpy(), expected, rtol=1e-4)
        status = app.ct_eq_status_text.get("1.0", "end-1c")
        assert "percent (0-100) reflectance" in status
    finally:
        restore()


@pytest.mark.parametrize("row_prefix", ["Physical match", "DS"])
def test_transfer_chain_converts_scale_and_exports_primary_scale(
        physical_app, row_prefix):
    """R2-1: the comparison-tab transfer chain routes through the same funnel."""
    from spectral_predict.calibration_transfer import apply_transfer_dispatch

    app, wl, primary, satellite = physical_app
    tm = _fit_percent_transfer(app, primary, satellite, row_prefix)

    data_pct = satellite * 100.0
    saved = {
        "comparison_data": app.comparison_data,
        "comparison_value_scale": getattr(app, "comparison_value_scale", None),
        "transfer_models": list(app.transfer_models),
        "type": app.comparison_data_type.get(),
    }
    try:
        app.comparison_data = data_pct
        app.comparison_data_type.set("reflectance")
        app.comparison_value_scale = 100.0
        app.apply_transfer.set(True)
        app.transfer_models = [{"path": "p", "model": tm, "description": row_prefix}]

        with patch("tkinter.messagebox.showerror") as err:
            out = app._apply_transfer_chain(data_pct)
        assert not err.called, err.call_args_list

        expected = apply_transfer_dispatch(data_pct.to_numpy() / 100.0, tm) * 100.0
        np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-6)
    finally:
        app.apply_transfer.set(False)
        app.transfer_models = saved["transfer_models"]
        app.comparison_data = saved["comparison_data"]
        app.comparison_value_scale = saved["comparison_value_scale"]
        app.comparison_data_type.set(saved["type"])


def test_legacy_transfer_scale_warning(physical_app):
    """R2-2: files without scale keys keep old behaviour but are flagged."""
    from spectral_predict.calibration_transfer import apply_transfer_dispatch

    app, wl, primary, satellite = physical_app
    app._pair_ct_standards()
    _run_inline(app, app._run_ct_validate)
    _select_physical_row(app)
    _run_inline(app, app._fit_ct_selected)
    tm = app.ct_transfer_model
    tm.meta.pop("value_scale", None)
    tm.meta.pop("primary_value_scale", None)

    # reflectance apply: warned (logged by the helper); absorbance: nothing to say
    note = app._ct_legacy_scale_note(tm, 100.0)
    assert note and "predates value-scale recording" in note
    assert app._ct_legacy_scale_note(tm, None) is None
    tm.meta["value_scale"] = 1.0
    assert app._ct_legacy_scale_note(tm, 1.0) is None
    tm.meta.pop("value_scale", None)

    # Mode B shows the same line in its stats and keeps the old behaviour:
    # no conversion anywhere, whatever the loaded scale
    X = satellite.to_numpy()
    app.new_satellite_data_export = (wl, X)
    app.ct_export_data_type.set("reflectance")
    app.ct_export_value_scale = 1.0
    app.ct_export_data_converted = False
    _run_inline(app, app._transform_spectra)
    stats = app.ct_transform_stats_text.get("1.0", "end-1c")
    assert "predates value-scale recording" in stats
    _wl_out, X_out = app.transformed_spectra
    np.testing.assert_allclose(X_out, apply_transfer_dispatch(X, tm))


def test_pairs_table_reports_folded_id_collisions(ct_app, tmp_path):
    """R2-3: two loaded files folding to one name are reported, not last-wins."""
    app, wl, primary, satellite = ct_app
    sat = satellite.copy()
    names = [f"LAB-B-{i + 1:02d}" for i in range(len(sat))]
    names[2] = "LAB.B.02"  # folds to the same key as "LAB-B-02"
    sat.index = names
    app.current_satellite_df = sat
    table = pd.DataFrame({
        "primary": ["bone_01", "bone_02", "bone_03"],
        "satellite": ["LAB-B-01.asd", "LAB-B-02.asd", "LAB-B-04.asd"],
    })
    table_path = tmp_path / "pairs.csv"
    table.to_csv(table_path, index=False)
    app.ct_pairs_table_path_var.set(str(table_path))

    with patch("tkinter.messagebox.showwarning") as warn:
        app._load_ct_pairs_table()
    # the ambiguous row is reported as a problem, not matched by luck of order
    warned = " ".join(str(c) for c in warn.call_args_list)
    assert "ambiguous" in warned
    assert "LAB-B-02" in warned
    # the collision itself: status label + a "!" row in the pairs view
    assert "1 naming conflict" in str(app.ct_pairs_status_label.cget("text"))
    rows = [str(app.ct_pairs_tree.item(item, "values")[0])
            for item in app.ct_pairs_tree.get_children()]
    conflict_rows = [r for r in rows if r.startswith("!")]
    assert len(conflict_rows) == 1
    assert "LAB-B-02" in conflict_rows[0] and "LAB.B.02" in conflict_rows[0]
    # the unambiguous rows still rename; the ambiguous one does not
    assert app.ct_pairs_rename == {"LAB-B-01": "bone_01", "LAB-B-04": "bone_03"}
    paired = app._pair_ct_standards()
    assert "bone_02" in paired.unmatched_primary


def test_pairs_table_reports_duplicate_row_mappings(ct_app, tmp_path):
    """R2-3: two table rows naming the same file keep the first, report both."""
    app, wl, primary, satellite = ct_app
    sat = satellite.copy()
    sat.index = [f"LAB-B-{i + 1:02d}" for i in range(len(sat))]
    app.current_satellite_df = sat

    table_path = tmp_path / "pairs.csv"
    pd.DataFrame({
        "primary": ["bone_01", "bone_02"],
        "satellite": ["LAB-B-01", "LAB-B-01"],  # same satellite file twice
    }).to_csv(table_path, index=False)
    app.ct_pairs_table_path_var.set(str(table_path))
    with patch("tkinter.messagebox.showwarning") as warn:
        app._load_ct_pairs_table()
    assert not warn.called  # every row matched a loaded file
    assert app.ct_pairs_rename == {"LAB-B-01": "bone_01"}  # first row kept
    assert "1 naming conflict" in str(app.ct_pairs_status_label.cget("text"))
    rows = [str(app.ct_pairs_tree.item(item, "values")[0])
            for item in app.ct_pairs_tree.get_children()]
    assert any("both map the satellite file 'LAB-B-01'" in r for r in rows)

    # same story on the primary side: two rows claiming one primary file
    pd.DataFrame({
        "primary": ["bone_01", "bone_01.asd"],
        "satellite": ["LAB-B-01", "LAB-B-02"],
    }).to_csv(table_path, index=False)
    with patch("tkinter.messagebox.showwarning"):
        app._load_ct_pairs_table()
    assert app.ct_pairs_rename == {"LAB-B-01": "bone_01"}
    rows = [str(app.ct_pairs_tree.item(item, "values")[0])
            for item in app.ct_pairs_tree.get_children()]
    assert any("both map the primary file 'bone_01'" in r for r in rows)


def test_plan_done_after_dialog_closed_does_not_touch_dead_widgets(gui_app):
    """R2-4: closing the plan dialog mid-run must not raise on destroyed widgets."""
    app = gui_app
    app._open_ct_plan_standards()
    dialog = app._ct_plan_dialog
    try:
        dialog.destroy()  # closed without the handler that clears the reference
        assert not dialog.winfo_exists()
        with patch("tkinter.messagebox.showerror") as err:
            app._on_ct_plan_done((True, ["s1", "s2"], ""))
            app._on_ct_plan_done((False, None, "boom"))
        assert not err.called
        # and the cleared-reference case
        app._ct_plan_dialog = None
        with patch("tkinter.messagebox.showerror") as err:
            app._on_ct_plan_done((True, ["s1"], ""))
        assert not err.called
        assert not app._ct_plan_running
    finally:
        if app._ct_plan_dialog is not None and app._ct_plan_dialog.winfo_exists():
            app._ct_plan_dialog.destroy()
        app._ct_plan_dialog = None


# ---------------------------------------------------------------------------
# Review round 3
# ---------------------------------------------------------------------------


def _captured_validate_run(app):
    """Run Validate inline, returning its (payload, context) before delivery."""
    captured = []
    with (patch.object(app, "_on_ct_validate_done",
                       lambda payload, context: captured.append((payload, context))),
          patch("tkinter.messagebox.showerror")):
        app._run_ct_validate()
    assert captured
    return captured[0]


def test_pairs_table_reload_discards_inflight_validate(inline_workers, tmp_path):
    """R3-1: loading a different pairs table invalidates a run in flight."""
    app, wl, primary, satellite = inline_workers
    sat_names = [f"LAB-B{int(i.split('_')[1]):02d}" for i in satellite.index]
    sat = satellite.copy()
    sat.index = sat_names
    app.current_satellite_df = sat
    table_full = tmp_path / "full.csv"
    pd.DataFrame(
        {"primary": list(primary.index), "satellite": sat_names}
    ).to_csv(table_full, index=False)
    app.ct_pairs_table_path_var.set(str(table_full))
    app._load_ct_pairs_table()
    app._pair_ct_standards()
    with patch("tkinter.messagebox.showerror") as err:
        app._run_ct_validate()
    assert not err.called
    assert app.ct_last_evaluation is not None

    payload, context = _captured_validate_run(app)

    # meanwhile the user loads a table that pairs only half the files
    table_half = tmp_path / "half.csv"
    pd.DataFrame({
        "primary": list(primary.index)[:6],
        "satellite": sat_names[:6],
    }).to_csv(table_half, index=False)
    app.ct_pairs_table_path_var.set(str(table_half))
    with patch("tkinter.messagebox.showwarning"):
        app._load_ct_pairs_table()

    assert app.ct_last_evaluation is None  # the board belonged to the old pairs
    # the invalidation abandoned the in-flight run: no active run id, the
    # busy state reset once, and its own message on the status line
    assert app._ct_active_run_id is None
    status_before = str(app.ct_validate_status_label.cget("text"))
    assert "cleared" in status_before
    app._on_ct_validate_done(payload, context)
    assert app.ct_last_evaluation is None  # the stale run is not installed
    # the abandoned run's callback changes nothing (R4-3)
    assert str(app.ct_validate_status_label.cget("text")) == status_before


def test_group_edit_discards_inflight_validate(inline_workers):
    """R3-1: a group edit invalidates a run started before the edit."""
    app, wl, primary, satellite = inline_workers
    app._pair_ct_standards()
    with patch("tkinter.messagebox.showerror") as err:
        app._run_ct_validate()
    assert not err.called
    assert app.ct_last_evaluation is not None

    payload, context = _captured_validate_run(app)

    app.ct_pair_group_entry_var.set("unitZ")
    app.ct_pairs_tree.selection_set(app.ct_pairs_tree.get_children()[0])
    app._ct_set_group_on_selected()

    assert app.ct_last_evaluation is None
    assert app._ct_active_run_id is None  # the in-flight run was abandoned
    app._on_ct_validate_done(payload, context)
    assert app.ct_last_evaluation is None
    # the abandoned run's callback leaves the invalidation's message in place
    # (the group edit itself reports on the pairs status label)
    status = str(app.ct_validate_status_label.cget("text"))
    assert status == "Groups changed — leaderboard and fitted transfer " \
                     "cleared. Run Validate & Compare again."


def test_eval_settings_change_discards_board(inline_workers):
    """R3-1: changing Space or Detector joins invalidates the board."""
    app, wl, primary, satellite = inline_workers
    app._pair_ct_standards()
    with patch("tkinter.messagebox.showerror") as err:
        app._run_ct_validate()
    assert not err.called
    assert app.ct_last_evaluation is not None

    app.ct_eval_space_var.set("reflectance")
    app._on_ct_eval_settings_changed()
    assert app.ct_last_evaluation is None and app._ct_eval_context is None
    assert not app.ct_leaderboard_tree.get_children()
    assert "cleared" in str(app.ct_validate_status_label.cget("text"))

    # the settings widgets are wired to the same handler
    assert app.ct_eval_space_combo.bind("<<ComboboxSelected>>")
    assert app.ct_eval_joins_combo.bind("<<ComboboxSelected>>")

    with patch("tkinter.messagebox.showerror") as err:
        app._run_ct_validate()
    assert not err.called
    assert app.ct_last_evaluation is not None
    app.ct_eval_joins_var.set("none")
    app._on_ct_eval_settings_changed()
    assert app.ct_last_evaluation is None


def test_custom_joins_edit_discards_board(inline_workers):
    """R4-2: editing the Custom (nm) joins entry invalidates the board.

    Fit uses the Validate context frozen at launch; an unnoticed entry edit
    would leave an old board fit-ready under different joins.
    """
    app, wl, primary, satellite = inline_workers
    app._pair_ct_standards()
    with patch("tkinter.messagebox.showerror") as err:
        app._run_ct_validate()
    assert not err.called
    assert app.ct_last_evaluation is not None
    app.ct_leaderboard_tree.selection_set(
        app.ct_leaderboard_tree.get_children()[0])

    # a user edits the custom-joins entry (the entry writes this variable)
    app.ct_eval_joins_custom_var.set("1100, 1700")

    assert app.ct_last_evaluation is None and app._ct_eval_context is None
    assert not app.ct_leaderboard_tree.get_children()
    import spectral_predict.transfer_evaluation as te

    fitted = []
    with (patch.object(te, "fit_transfer",
                       lambda *a, **k: fitted.append(1)),
          patch("tkinter.messagebox.showwarning") as warn):
        app._fit_ct_selected()
    assert not fitted  # Fit refused: no board survives the joins edit
    assert warn.called
    assert "Validate & Compare" in str(warn.call_args)


def test_eval_model_load_discards_board(inline_workers, tmp_path):
    """R3-1: loading a different prediction model invalidates the board."""
    app, wl, primary, satellite = inline_workers
    app._pair_ct_standards()
    with patch("tkinter.messagebox.showerror") as err:
        app._run_ct_validate()
    assert not err.called
    assert app.ct_last_evaluation is not None

    y = np.linspace(1.0, 9.0, primary.shape[0])
    model = _regression_model_on(tmp_path, primary.to_numpy(), y, wl)
    app.ct_eval_model_path_var.set(str(tmp_path / "PLS.dasp"))
    with patch("tkinter.messagebox.showinfo"):
        app._load_ct_eval_model()

    assert app.ct_eval_model_dict is not None
    assert app.ct_last_evaluation is None and app._ct_eval_context is None


def test_stale_callback_does_not_unlock_newer_run(inline_workers):
    """R3-2: an old callback must leave the newer run's busy state alone."""
    app, wl, primary, satellite = inline_workers
    app._pair_ct_standards()
    payload1, ctx1 = _captured_validate_run(app)

    # a newer run is in flight when run 1's callback finally arrives
    app._ct_active_run_id = ctx1["run_id"] + 1
    app._ct_validate_running = True
    app._set_ct_validate_busy(True, "Running evaluate_transfer ...")
    app._on_ct_validate_done(payload1, ctx1)

    assert app._ct_validate_running is True
    assert str(app.ct_validate_button.cget("state")) == "disabled"
    assert app.ct_last_evaluation is None

    # the newer run's callback still installs and unlocks normally
    app._on_ct_validate_done(payload1, dict(ctx1, run_id=app._ct_active_run_id))
    assert app.ct_last_evaluation is payload1[1]
    assert app._ct_validate_running is False
    assert str(app.ct_validate_button.cget("state")) == "normal"


def test_stale_fit_callback_does_not_unlock_newer_run(ct_app):
    """R3-2: a stale fit callback must not re-enable controls mid-run."""
    app, *_ = ct_app
    app._ct_active_run_id = 3
    app._ct_validate_running = True
    app._set_ct_validate_busy(True, "Fitting DS ...")

    app._on_ct_fit_done((True, None, ""), "No correction",
                        {"generation": getattr(app, "_ct_input_generation", 0),
                         "run_id": 2})

    assert app._ct_validate_running is True
    assert str(app.ct_validate_button.cget("state")) == "disabled"
    assert str(app.ct_fit_selected_button.cget("state")) == "disabled"


def test_invalidation_abandons_inflight_run(inline_workers):
    """R4-3: invalidation during a run resets the busy state once, and the
    abandoned run's later callback touches nothing."""
    app, wl, primary, satellite = inline_workers
    app._pair_ct_standards()
    with patch("tkinter.messagebox.showerror") as err:
        app._run_ct_validate()
    assert not err.called
    assert app.ct_last_evaluation is not None  # a board is up

    payload, ctx = _captured_validate_run(app)  # a second run is in flight
    assert app._ct_validate_running is True
    assert str(app.ct_validate_button.cget("state")) == "disabled"

    # a real settings change invalidates while that run is in flight
    app.ct_eval_space_var.set("reflectance")
    app.ct_eval_space_combo.event_generate("<<ComboboxSelected>>", when="now")

    assert app._ct_active_run_id is None  # the run was abandoned
    assert app._ct_validate_running is False
    assert str(app.ct_validate_button.cget("state")) == "normal"  # can rerun

    status_before = str(app.ct_validate_status_label.cget("text"))
    assert "cleared" in status_before
    app._on_ct_validate_done(payload, ctx)
    assert str(app.ct_validate_status_label.cget("text")) == status_before
    assert app.ct_last_evaluation is None
    assert app._ct_validate_running is False


def test_run1_callback_after_relaunch_leaves_run2_busy(inline_workers):
    """R4-3: an abandoned run's callback must not unlock the run after it."""
    app, wl, primary, satellite = inline_workers
    app._pair_ct_standards()
    payload1, ctx1 = _captured_validate_run(app)  # run 1 in flight

    # the settings change invalidates run 1; a new run launches right away
    app.ct_eval_joins_var.set("none")
    app.ct_eval_joins_combo.event_generate("<<ComboboxSelected>>", when="now")
    payload2, ctx2 = _captured_validate_run(app)  # run 2 in flight
    assert ctx2["run_id"] != ctx1["run_id"]
    assert app._ct_validate_running is True
    assert str(app.ct_validate_button.cget("state")) == "disabled"

    app._on_ct_validate_done(payload1, ctx1)  # run 1 finally reports
    assert app._ct_validate_running is True   # run 2 still owns the panel
    assert str(app.ct_validate_button.cget("state")) == "disabled"
    assert app.ct_last_evaluation is None

    app._on_ct_validate_done(payload2, ctx2)  # the live run completes
    assert app.ct_last_evaluation is payload2[1]
    assert app._ct_validate_running is False
    assert str(app.ct_validate_button.cget("state")) == "normal"


def test_partly_matched_table_does_not_pair_unintended_ids(ct_app, tmp_path):
    """R3-3: with a table active, only its rows pair — no silent ID fallback.

    The table says bone_03 -> LAB-missing, but LAB-missing is not loaded and
    a satellite file literally named bone_03 exists. That file must stay
    unmatched; the table, not name luck, defines the pairs.
    """
    app, wl, primary, satellite = ct_app
    # satellite files carry per-specimen names (rows stay shuffled, so each
    # name belongs to its own row's specimen)
    sat = satellite.copy()
    sat.index = [f"LAB-{pid}" for pid in satellite.index]
    # the trap: one extra satellite file named exactly like primary bone_03
    trap = sat.iloc[[0]].copy()
    trap.index = ["bone_03"]
    sat = pd.concat([sat, trap])
    app.current_satellite_df = sat
    rows = {
        "primary": [pid for pid in primary.index if pid != "bone_03"],
        "satellite": [f"LAB-{pid}" for pid in primary.index
                      if pid != "bone_03"],
    }
    rows["primary"].append("bone_03")
    rows["satellite"].append("LAB-missing")  # not loaded -> row unresolved
    table_path = tmp_path / "pairs.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False)
    app.ct_pairs_table_path_var.set(str(table_path))

    with patch("tkinter.messagebox.showwarning") as warn:
        app._load_ct_pairs_table()
    assert warn.called  # the unresolved row is reported, not silently dropped

    paired = app._pair_ct_standards()
    assert paired is not None
    assert len(paired.ids) == 11
    assert "bone_03" in paired.unmatched_primary  # its table row is unresolved
    assert "bone_03" in paired.unmatched_satellite  # the stray file is NOT paired
    # the pairs really are the table's rows, in its order
    np.testing.assert_allclose(paired.X_satellite, 0.9 * paired.X_primary + 0.03)
    status = str(app.ct_pairs_status_label.cget("text"))
    assert "11 rows paired" in status
    assert "1 rows unresolved" in status
    # the table NAMES bone_03 (via its unresolved row), so only its unloaded
    # satellite's real counterpart LAB-bone_03 and the trap file are unnamed
    assert "2 files not in the table" in status


def test_not_in_table_count_excludes_files_named_by_unresolved_rows(
        ct_app, tmp_path):
    """R4-5: 'files not in the table' counts only files no table row names.

    bone_03's row is unresolved (its satellite never loaded) but the table
    does name bone_03 — it must not be counted as 'not in the table'.
    """
    app, wl, primary, satellite = ct_app
    prim = primary.iloc[:4].copy()  # bone_01..bone_04
    sat = satellite.iloc[:4].copy()
    sat.index = [f"LAB-{pid}" for pid in prim.index]
    app.current_primary_df = prim
    app.current_satellite_df = sat
    table_path = tmp_path / "pairs.csv"
    pd.DataFrame({
        "primary": ["bone_01", "bone_02", "bone_03"],
        "satellite": ["LAB-bone_01", "LAB-bone_02", "LAB-missing"],
    }).to_csv(table_path, index=False)
    app.ct_pairs_table_path_var.set(str(table_path))
    with patch("tkinter.messagebox.showwarning") as warn:
        app._load_ct_pairs_table()
    assert warn.called  # the bone_03 -> LAB-missing row is unresolved

    paired = app._pair_ct_standards()
    assert paired is not None
    assert len(paired.ids) == 2
    assert sorted(paired.unmatched_primary) == ["bone_03", "bone_04"]
    assert sorted(paired.unmatched_satellite) == ["LAB-bone_03", "LAB-bone_04"]
    status = str(app.ct_pairs_status_label.cget("text"))
    assert "1 rows unresolved" in status
    # bone_04, LAB-bone_03 and LAB-bone_04 are unnamed; bone_03 IS named
    assert "3 files not in the table" in status


def test_clearing_pairs_table_returns_to_id_matching(ct_app, tmp_path):
    """R3-3: clearing the table returns pairing to plain ID matching."""
    app, wl, primary, satellite = ct_app
    table_path = tmp_path / "pairs.csv"
    pd.DataFrame({  # an identity table: the same IDs on both instruments
        "primary": list(primary.index),
        "satellite": list(primary.index),
    }).to_csv(table_path, index=False)
    app.ct_pairs_table_path_var.set(str(table_path))
    app._load_ct_pairs_table()
    assert app.ct_pairs_rename
    assert app.ct_pairs_table_active is True

    app._clear_ct_pairs_table()
    assert app.ct_pairs_rename is None
    assert app.ct_pairs_table_active is False
    assert app.ct_pairs_table_path_var.get() == ""

    paired = app._pair_ct_standards()  # no table: every shared ID pairs
    assert paired is not None
    assert len(paired.ids) == len(primary.index)


def test_zero_resolution_table_after_reload_refuses_not_id_fallback(
        ct_app, tmp_path):
    """R4-1: a table whose rows resolve nothing after a reload keeps
    defining the pairs.

    The reload renames the satellite files to the primary IDs, so ID
    matching would happily pair all of them — but the table path is still
    set, and the table is the ground truth. Pairing must refuse with a
    clear status; only Clear Table returns to ID matching.
    """
    app, wl, primary, satellite = ct_app
    sat = satellite.copy()
    sat_names = [f"LAB-{pid}" for pid in satellite.index]
    sat.index = sat_names
    app.current_satellite_df = sat
    table_path = tmp_path / "pairs.csv"
    pd.DataFrame(
        {"primary": list(primary.index), "satellite": sat_names}
    ).to_csv(table_path, index=False)
    app.ct_pairs_table_path_var.set(str(table_path))
    app._load_ct_pairs_table()
    assert app.ct_pairs_table_active is True
    assert len(app.ct_pairs_rename) == len(sat_names)
    assert app._pair_ct_standards() is not None

    # reload the satellite as files named exactly like the primary: no table
    # row resolves any more, but shared-ID matching would pair all 12
    shared = satellite.copy()
    shared.index = list(primary.index)
    app._last_dir_ids = list(primary.index)
    try:
        with (patch("tkinter.messagebox.showwarning") as warn,
              patch("tkinter.messagebox.showerror") as err):
            app._ct_store_loaded_standards(
                'satellite', str(tmp_path), wl, shared.to_numpy(), None)
    finally:
        app._last_dir_ids = None
    assert warn.called  # every row is reported unresolved
    assert app.ct_pairs_table_active is True
    assert app.ct_pairs_rename is None

    with patch("tkinter.messagebox.showerror") as err:
        paired = app._pair_ct_standards()
    assert paired is None
    assert app.ct_paired is None  # no ID-matched pairs behind the refusal
    assert err.called
    status = str(app.ct_pairs_status_label.cget("text"))
    assert "no rows resolve" in status
    assert "Clear Table" in status

    # Clear Table is the way back to plain ID matching
    app._clear_ct_pairs_table()
    paired = app._pair_ct_standards()
    assert paired is not None
    assert len(paired.ids) == len(primary.index)


def test_clear_table_keeps_manual_groups_drops_table_groups(ct_app, tmp_path):
    """R4-4: Clear Table removes table-derived group labels, keeps manual
    edits — including a manual override of a table-grouped row."""
    app, wl, primary, satellite = ct_app
    sat = satellite.copy()
    sat.index = [f"LAB-{pid}" for pid in satellite.index]
    app.current_satellite_df = sat
    table_path = tmp_path / "pairs.csv"
    pd.DataFrame({
        "primary": list(primary.index),
        "satellite": [f"LAB-{pid}" for pid in primary.index],
        "group": ["unitA", "unitB"] + [None] * 10,
    }).to_csv(table_path, index=False)
    app.ct_pairs_table_path_var.set(str(table_path))
    app._load_ct_pairs_table()
    assert app.ct_pair_groups == {"bone_01": "unitA", "bone_02": "unitB"}
    assert set(app.ct_pair_groups_table_keys) == {"bone_01", "bone_02"}

    app._pair_ct_standards()
    # a manual label on a fresh row, and a manual override of a table row
    # (each edit refreshes the tree, so the item ids are re-read every time)
    app.ct_pair_group_entry_var.set("manualM")
    app.ct_pairs_tree.selection_set((app.ct_pairs_tree.get_children()[2],))  # bone_03
    app._ct_set_group_on_selected()
    app.ct_pair_group_entry_var.set("manualA")
    app.ct_pairs_tree.selection_set((app.ct_pairs_tree.get_children()[0],))  # bone_01
    app._ct_set_group_on_selected()
    assert app.ct_pair_groups["bone_01"] == "manualA"

    app._clear_ct_pairs_table()
    assert app.ct_pair_groups == {"bone_01": "manualA", "bone_03": "manualM"}
    assert app.ct_pair_groups_table_keys is None


class _chain_state:
    """Point the comparison tab at a transfer chain over percent data."""

    def __init__(self, app, models):
        self.app = app
        self.models = models
        self.saved = {
            "comparison_data": app.comparison_data,
            "comparison_value_scale": getattr(app, "comparison_value_scale", None),
            "transfer_models": list(app.transfer_models),
            "type": app.comparison_data_type.get(),
        }

    def __enter__(self):
        self.app.comparison_data_type.set("reflectance")
        self.app.comparison_value_scale = 100.0
        self.app.apply_transfer.set(True)
        self.app.transfer_models = self.models
        return self.app

    def __exit__(self, *exc):
        app = self.app
        app.apply_transfer.set(False)
        app.transfer_models = self.saved["transfer_models"]
        app.comparison_data = self.saved["comparison_data"]
        app.comparison_value_scale = self.saved["comparison_value_scale"]
        app.comparison_data_type.set(self.saved["type"])
        return False


def _legacy_percent_transfer(app, wl, primary, satellite):
    """A transfer file that predates scale recording: DS fitted on percent."""
    from spectral_predict.calibration_transfer import TransferModel, estimate_ds

    A = estimate_ds((primary * 100.0).to_numpy(), (satellite * 100.0).to_numpy(),
                    lam=1e-6)
    return TransferModel("p", "s", "ds", np.asarray(wl, dtype=float),
                         {"A": A}, {})


def test_chain_new_then_legacy_feeds_legacy_primary_scale(physical_app):
    """R3-4: a legacy step after a new one receives the primary's loaded scale."""
    from spectral_predict.calibration_transfer import apply_transfer_dispatch

    app, wl, primary, satellite = physical_app
    tm_new = _fit_percent_transfer(app, primary, satellite, "DS")
    legacy = _legacy_percent_transfer(app, wl, primary, satellite)
    X_pct = satellite * 100.0

    with _chain_state(app, [
            {"path": "new", "model": tm_new, "description": "new (0-1 fit)"},
            {"path": "legacy", "model": legacy, "description": "legacy (percent)"}]):
        with patch("tkinter.messagebox.showerror") as err:
            out = app._apply_transfer_chain(X_pct)
        assert not err.called, err.call_args_list

    expected = apply_transfer_dispatch(
        apply_transfer_dispatch(X_pct.to_numpy() / 100.0, tm_new) * 100.0, legacy)
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-5)


def test_chain_legacy_then_new_keeps_conversion(physical_app):
    """R3-4: a legacy step first keeps its old behaviour; the new one funnels."""
    from spectral_predict.calibration_transfer import apply_transfer_dispatch

    app, wl, primary, satellite = physical_app
    tm_new = _fit_percent_transfer(app, primary, satellite, "DS")
    legacy = _legacy_percent_transfer(app, wl, primary, satellite)
    X_pct = satellite * 100.0

    with _chain_state(app, [
            {"path": "legacy", "model": legacy, "description": "legacy (percent)"},
            {"path": "new", "model": tm_new, "description": "new (0-1 fit)"}]):
        with patch("tkinter.messagebox.showerror") as err:
            out = app._apply_transfer_chain(X_pct)
        assert not err.called, err.call_args_list

    expected = apply_transfer_dispatch(
        apply_transfer_dispatch(X_pct.to_numpy(), legacy) / 100.0, tm_new) * 100.0
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-5)


def test_chain_two_new_steps_convert_once_at_end(physical_app):
    """R3-4: two recorded steps stay in the fit scale; one conversion at the end."""
    from spectral_predict.calibration_transfer import apply_transfer_dispatch

    app, wl, primary, satellite = physical_app
    tm1 = _fit_percent_transfer(app, primary, satellite, "DS")
    tm2 = _fit_percent_transfer(app, primary, satellite, "Physical match")
    X_pct = satellite * 100.0

    with _chain_state(app, [
            {"path": "a", "model": tm1, "description": "new 1"},
            {"path": "b", "model": tm2, "description": "new 2"}]):
        with patch("tkinter.messagebox.showerror") as err:
            out = app._apply_transfer_chain(X_pct)
        assert not err.called, err.call_args_list

    expected = apply_transfer_dispatch(
        apply_transfer_dispatch(X_pct.to_numpy() / 100.0, tm1), tm2) * 100.0
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-5)


def _descending_csv(tmp_path, name, wl, X_sat, n_samples=4):
    """Column-oriented CSV whose wavelength column runs descending.

    The ends sit just past the fitted range by float rounding, so the
    coverage gate passes only through its min/max comparison.
    """
    from spectral_predict.calibration_transfer import resample_to_grid

    coarse = np.linspace(wl[-1] + 1e-7, wl[0] - 1e-7, 60)
    values = resample_to_grid(X_sat[:n_samples], wl, coarse)  # columns follow coarse
    lines = ["wavelength," + ",".join(f"s{i + 1}" for i in range(n_samples))]
    for j, w in enumerate(coarse):
        lines.append(repr(float(w))
                     + "," + ",".join(repr(float(v)) for v in values[:, j]))
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return coarse, values


def test_mode_a_load_handler_accepts_descending_grid(physical_app, tmp_path):
    """R3-5: a descending grid that covers the range is resampled, not refused."""
    from spectral_predict.calibration_transfer import resample_to_grid as rgrid

    app, wl, primary, satellite = physical_app
    tm = _fitted_physical_transfer(app)
    coarse, values = _descending_csv(tmp_path, "descending.csv", wl,
                                     satellite.to_numpy())
    app.ct_pred_satellite_path_var.set(str(tmp_path / "descending.csv"))
    app.new_satellite_data_predict = None

    with (patch("spectral_predict.io.read_combined_csv", side_effect=ValueError),
          patch("tkinter.messagebox.askyesno", return_value=True) as ask,
          patch("tkinter.messagebox.showerror") as err):
        app._load_new_satellite_data_predict()
    assert err.called is False, err.call_args_list
    assert ask.called

    stored_wl, stored_X = app.new_satellite_data_predict
    np.testing.assert_allclose(stored_wl, tm.wavelengths_common)
    expected = rgrid(values, coarse, tm.wavelengths_common)
    np.testing.assert_allclose(stored_X, expected, rtol=1e-6)


def test_mode_b_load_handler_accepts_descending_grid(physical_app, tmp_path):
    """R3-5: Mode B's load handler resamples a descending grid too."""
    from spectral_predict.calibration_transfer import resample_to_grid as rgrid

    app, wl, primary, satellite = physical_app
    tm = _fitted_physical_transfer(app)
    # legacy layout the export loader understands: row 0 = wavelengths
    coarse = np.linspace(wl[-1] + 1e-7, wl[0] - 1e-7, 60)
    values = rgrid(satellite.to_numpy()[:4], wl, coarse)
    lines = [",".join(repr(float(w)) for w in coarse)]
    lines += [",".join(repr(float(v)) for v in row) for row in values]
    csv_path = tmp_path / "descending_export.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    app.ct_export_satellite_path_var.set(str(csv_path))
    app.new_satellite_data_export = None

    with (patch("tkinter.messagebox.askyesno", return_value=True) as ask,
          patch("tkinter.messagebox.showerror") as err):
        app._load_new_satellite_data_export()
    assert err.called is False, err.call_args_list
    assert ask.called

    stored_wl, stored_X = app.new_satellite_data_export
    np.testing.assert_allclose(stored_wl, tm.wavelengths_common)
    expected = rgrid(values, coarse, tm.wavelengths_common)
    np.testing.assert_allclose(stored_X, expected, rtol=1e-6)


def test_pairing_keeps_collision_count_in_status(ct_app, tmp_path):
    """R3-6: pairing must not overwrite the table's naming-conflict warning."""
    app, wl, primary, satellite = ct_app
    sat = satellite.copy()
    sat.index = [f"LAB-B-{i + 1:02d}" for i in range(len(sat))]
    app.current_satellite_df = sat
    table_path = tmp_path / "pairs.csv"
    pd.DataFrame({
        "primary": ["bone_01", "bone_02", "bone_03"],
        "satellite": ["LAB-B-01", "LAB-B-02", "LAB-B-02"],  # row 3 conflicts
    }).to_csv(table_path, index=False)
    app.ct_pairs_table_path_var.set(str(table_path))
    with patch("tkinter.messagebox.showwarning") as warn:
        app._load_ct_pairs_table()
    assert not warn.called
    assert "1 naming conflict" in str(app.ct_pairs_status_label.cget("text"))

    paired = app._pair_ct_standards()
    assert paired is not None
    status = str(app.ct_pairs_status_label.cget("text"))
    assert "1 naming conflict" in status
    assert str(app.ct_pairs_status_label.cget("foreground")) == str(
        app.colors["warning"])


def _provision_load_and_predict_dir(app):
    """Create the widgets/vars _load_and_predict_ct touches (its frame is not
    wired into the tab either — same state a wired UI would hold)."""
    saved_var = getattr(app, "ct_new_satellite_dir_var", None)
    had_text = hasattr(app, "ct_prediction_text")
    saved_text = getattr(app, "ct_prediction_text", None) if had_text else None
    app.ct_new_satellite_dir_var = tk.StringVar()
    app.ct_prediction_text = tk.Text(app.root)

    def restore():
        app.ct_prediction_text.destroy()
        if saved_var is None:
            delattr(app, "ct_new_satellite_dir_var")
        else:
            app.ct_new_satellite_dir_var = saved_var
        if had_text:
            app.ct_prediction_text = saved_text
        else:
            delattr(app, "ct_prediction_text")

    return restore


def _percent_satellite_dir(tmp_path, wl, satellite):
    """Directory of percent-reflectance satellite CSVs (header = wavelengths)."""
    X_pct = (satellite * 100.0).to_numpy()
    d = tmp_path / "satellite_pct"
    d.mkdir()
    header = ",".join(repr(float(w)) for w in wl)
    for i, sid in enumerate(satellite.index):
        (d / f"{sid}.csv").write_text(
            header + "\n" + ",".join(repr(float(v)) for v in X_pct[i]) + "\n",
            encoding="utf-8")
    return d, X_pct


def test_load_and_predict_routes_through_scale_funnel(physical_app, tmp_path):
    """R3-7: the older apply-and-predict path feeds the model primary-scale."""
    from spectral_predict.calibration_transfer import apply_transfer_dispatch

    app, wl, primary, satellite = physical_app
    tm = _fit_percent_transfer(app, primary, satellite, "DS")
    app.ct_pred_transfer_model = tm
    y = np.linspace(1.0, 9.0, primary.shape[0])
    model = _regression_model_on(tmp_path, (primary * 100.0).to_numpy(), y, wl)
    app.ct_primary_model_dict = model
    d, X_pct = _percent_satellite_dir(tmp_path, wl, satellite)
    restore_dir_var = _provision_load_and_predict_dir(app)
    app.ct_new_satellite_dir_var.set(str(d))
    app.ct_pred_y_pred = None

    try:
        with (patch("tkinter.messagebox.showinfo"),
              patch("tkinter.messagebox.showerror") as err,
              patch.object(app, "_plot_ct_predictions")):
            app._load_and_predict_ct()
        assert not err.called, err.call_args_list

        expected = model["model"].predict(
            apply_transfer_dispatch(X_pct / 100.0, tm) * 100.0).ravel()
        np.testing.assert_allclose(np.asarray(app.ct_pred_y_pred), expected,
                                   rtol=1e-6)
    finally:
        restore_dir_var()


def test_load_and_predict_warns_on_legacy_transfer_file(physical_app, tmp_path,
                                                        capsys):
    """R3-7: a legacy transfer file is warned about on this path too."""
    from spectral_predict.calibration_transfer import apply_transfer_dispatch

    app, wl, primary, satellite = physical_app
    tm = _fit_percent_transfer(app, primary, satellite, "DS")
    tm.meta.pop("value_scale", None)
    tm.meta.pop("primary_value_scale", None)
    app.ct_pred_transfer_model = tm
    y = np.linspace(1.0, 9.0, primary.shape[0])
    model = _regression_model_on(tmp_path, (primary * 100.0).to_numpy(), y, wl)
    app.ct_primary_model_dict = model
    d, X_pct = _percent_satellite_dir(tmp_path, wl, satellite)
    restore_dir_var = _provision_load_and_predict_dir(app)
    app.ct_new_satellite_dir_var.set(str(d))
    app.ct_pred_y_pred = None

    try:
        with (patch("tkinter.messagebox.showinfo"),
              patch("tkinter.messagebox.showerror") as err,
              patch.object(app, "_plot_ct_predictions")):
            app._load_and_predict_ct()
        assert not err.called, err.call_args_list
        assert "predates value-scale recording" in capsys.readouterr().out

        # old behaviour kept: no conversion anywhere for a legacy file
        expected = model["model"].predict(
            apply_transfer_dispatch(X_pct, tm)).ravel()
        np.testing.assert_allclose(np.asarray(app.ct_pred_y_pred), expected,
                                   rtol=1e-6)
    finally:
        restore_dir_var()

