"""QW7: results-table column widths and Treeview row height follow the display scale."""

from __future__ import annotations

import sys
import tkinter.font as tkfont
from tkinter import ttk

import pandas as pd
import pytest

import spectral_predict_gui_optimized as gui

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="scale is pinned to 1.0 on macOS")


class _FakeRoot:
    def __init__(self, dpi: float) -> None:
        self._dpi = dpi

    def winfo_fpixels(self, _distance: str) -> float:
        return self._dpi


@pytest.fixture
def scaled_150(gui_app):
    """Run the session app at a simulated 150% scale, then restore it exactly.

    Restores the module scale state and the shared Results table: its columns and
    their widths, the stored default widths and the displayed/loaded DataFrames.
    """
    app = gui_app
    tree = app.results_tree
    saved_scale = (gui._UI_SCALE, dict(gui.SPACING), dict(gui.SIDEBAR_CONFIG))
    saved_columns = tuple(tree["columns"])
    saved_widths = {col: tree.column(col, "width") for col in saved_columns}
    saved_attrs = {
        name: getattr(app, name, None)
        for name in ("_results_default_col_widths", "results_display_df", "results_df")
    }
    had_default_widths = hasattr(app, "_results_default_col_widths")

    gui._apply_ui_scale(_FakeRoot(144.0))
    yield app

    gui._UI_SCALE = saved_scale[0]
    gui.SPACING.clear()
    gui.SPACING.update(saved_scale[1])
    gui.SIDEBAR_CONFIG.clear()
    gui.SIDEBAR_CONFIG.update(saved_scale[2])
    app._apply_theme(app.current_theme_name.get())
    children = tree.get_children()
    if children:
        tree.delete(*children)
    tree["columns"] = saved_columns
    for col, width in saved_widths.items():
        tree.column(col, width=width)
    for name, value in saved_attrs.items():
        setattr(app, name, value)
    if not had_default_widths and hasattr(app, "_results_default_col_widths"):
        del app._results_default_col_widths


def _small_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Select": [False, False],
            "Rank": [1, 2],
            "Model": ["PLS", "Ridge"],
            "Preprocess": ["raw", "snv"],
            "RMSEcv": [1.23456e-05, 2.5e-05],
            "RMSE_Q1": [-1.23456e-05, 0.5],
            "R2cv": [0.99, 1.23456e10],  # '+' exponent: wider than '-' at equal length
            "n_vars": [10, 20],
        }
    )


def test_results_columns_are_scaled(scaled_150):
    app = scaled_150
    app._populate_results_table(_small_results())
    app.root.update_idletasks()
    assert app.results_tree.column("Model", "width") == gui._px(120) == 180
    assert app.results_tree.column("RMSEcv", "width") >= gui._px(80) == 120


def test_float_columns_fit_their_text_in_the_row_font(scaled_150):
    """Wiring check in the session app: each float cell's text fits its column.

    Only _UI_SCALE is simulated here; Tk's own scaling stays at the session's. Real
    125/150/200% measurements are in tests/test_gui_dpi_fonts.py
    (test_results_columns_fit_their_text_at_scale).
    """
    app = scaled_150
    app._populate_results_table(_small_results())
    app.root.update_idletasks()
    row_font = tkfont.nametofont("TkDefaultFont", root=app.root)
    tk_cell_padding = 2 * gui._px(4)
    for col in ("RMSEcv", "RMSE_Q1", "R2cv"):
        width = app.results_tree.column(col, "width")
        for iid in app.results_tree.get_children():
            text = app.results_tree.set(iid, col)
            assert row_font.measure(text) + tk_cell_padding <= width, (col, text, width)


def test_treeview_rowheight_follows_row_font(scaled_150):
    app = scaled_150
    app._apply_theme(app.current_theme_name.get())
    linespace = tkfont.nametofont("TkDefaultFont", root=app.root).metrics("linespace")
    rowheight = int(ttk.Style(app.root).lookup("Treeview", "rowheight"))
    assert rowheight == linespace + gui._px(2)
    assert rowheight > linespace
