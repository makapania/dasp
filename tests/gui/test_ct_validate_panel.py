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
    "current_primary_ids_positional",
    "current_satellite_ids_positional",
    "ct_primary_header_joins",
    "ct_satellite_header_joins",
    "ct_paired",
    "ct_pair_groups",
    "ct_pairs_rename",
    "ct_pair_resample_note",
    "ct_last_evaluation",
    "ct_transfer_model",
    "current_transfer_model",
    "ct_wavelengths_common",
    "ct_X_primary_common",
    "ct_X_satellite_common",
    "ct_pairs_table_path_var",
    "ct_pair_group_entry_var",
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
