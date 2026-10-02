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

    def winfo_screenwidth(self) -> int:
        return 2560

    def winfo_screenheight(self) -> int:
        return 1440


@pytest.fixture
def scaled_150(gui_app):
    """Run the session app at a simulated 150% scale, then restore it exactly."""
    saved = (gui._UI_SCALE, gui._SCREEN_SIZE, dict(gui.SPACING), dict(gui.SIDEBAR_CONFIG))
    gui._apply_ui_scale(_FakeRoot(144.0))
    yield gui_app
    gui._UI_SCALE, gui._SCREEN_SIZE = saved[0], saved[1]
    gui.SPACING.clear()
    gui.SPACING.update(saved[2])
    gui.SIDEBAR_CONFIG.clear()
    gui.SIDEBAR_CONFIG.update(saved[3])
    gui_app._apply_theme(gui_app.current_theme_name.get())
    gui_app.results_df = None
    children = gui_app.results_tree.get_children()
    if children:
        gui_app.results_tree.delete(*children)


def _small_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Select": [False, False],
            "Rank": [1, 2],
            "Model": ["PLS", "Ridge"],
            "Preprocess": ["raw", "snv"],
            "RMSEcv": [1.23456e-05, 2.5e-05],
            "R2cv": [0.99, 0.98],
            "n_vars": [10, 20],
        }
    )


def test_results_columns_are_scaled(scaled_150):
    app = scaled_150
    app._populate_results_table(_small_results())
    app.root.update_idletasks()
    assert app.results_tree.column("RMSEcv", "width") == gui._px(80) == 120
    assert app.results_tree.column("Model", "width") == gui._px(120) == 180


def test_treeview_rowheight_follows_row_font(scaled_150):
    app = scaled_150
    app._apply_theme(app.current_theme_name.get())
    linespace = tkfont.nametofont("TkDefaultFont", root=app.root).metrics("linespace")
    rowheight = int(ttk.Style(app.root).lookup("Treeview", "rowheight"))
    assert rowheight == linespace + gui._px(2)
    assert rowheight > linespace
