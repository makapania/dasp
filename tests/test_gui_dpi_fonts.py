"""High-DPI scaling helpers and named fonts in the GUI module (QW7)."""

from __future__ import annotations

import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

import numpy as np
import pandas as pd
import pytest

import spectral_predict_gui_optimized as gui

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="scale is pinned to 1.0 on macOS")


class _FakeRoot:
    """Stand-in for a Tk root that reports a fixed dpi and screen size."""

    def __init__(self, dpi: float, screen: tuple[int, int] = (2560, 1440)) -> None:
        self._dpi = dpi
        self._screen = screen

    def winfo_fpixels(self, _distance: str) -> float:
        return self._dpi

    def winfo_screenwidth(self) -> int:
        return self._screen[0]

    def winfo_screenheight(self) -> int:
        return self._screen[1]


@pytest.fixture
def restore_scale():
    """Snapshot the module scale state and put it back exactly as it was."""
    saved = (gui._UI_SCALE, dict(gui.SPACING), dict(gui.SIDEBAR_CONFIG))
    yield
    gui._UI_SCALE = saved[0]
    gui.SPACING.clear()
    gui.SPACING.update(saved[1])
    gui.SIDEBAR_CONFIG.clear()
    gui.SIDEBAR_CONFIG.update(saved[2])


@pytest.fixture
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # no display
        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    yield root
    root.destroy()


def test_scale_is_idempotent_and_rescales_constants(restore_scale):
    assert gui._apply_ui_scale(_FakeRoot(120.0)) == 1.25
    assert gui._px(70) == 88
    assert gui.SPACING["xl"] == 30
    assert gui.SIDEBAR_CONFIG["expanded_width"] == 275
    # A second app in the same process must not compound the scaling.
    gui._apply_ui_scale(_FakeRoot(120.0))
    assert gui.SPACING["xl"] == 30
    gui._apply_ui_scale(_FakeRoot(96.0))
    assert gui.SPACING == gui._BASE_SPACING
    assert gui._px(70) == 70


def test_px_geometry_scales_dialog_sizes(restore_scale):
    gui._apply_ui_scale(_FakeRoot(96.0))
    assert gui._px_geometry("350x200") == "350x200"
    gui._apply_ui_scale(_FakeRoot(144.0))
    assert gui._px_geometry("350x200") == "525x300"


class _FakeOwner(_FakeRoot):
    """Owner window centred on a 1920x1080 screen (no Win32 handle, so the fallback runs)."""

    def __init__(self) -> None:
        super().__init__(144.0, screen=(1920, 1080))

    def winfo_rootx(self) -> int:
        return 460

    def winfo_rooty(self) -> int:
        return 240

    def winfo_width(self) -> int:
        return 1000

    def winfo_height(self) -> int:
        return 600


def test_px_geometry_without_owner_is_size_only(restore_scale):
    gui._apply_ui_scale(_FakeRoot(144.0))
    assert gui._px_geometry("520x720") == "780x1080"


def test_px_geometry_clamps_and_places_inside_the_work_area(restore_scale):
    # Peak calculator at 150% on a 1920x1080 panel: 780x1080 would not fit.
    gui._apply_ui_scale(_FakeRoot(144.0))
    geometry = gui._px_geometry("520x720", _FakeOwner())
    size, x, y = geometry.split("+")
    width, height = (int(v) for v in size.split("x"))
    assert width == 780
    assert height == 1080 - gui._px(40)  # room for the title bar
    assert 0 <= int(x) and int(x) + width <= 1920
    assert 0 <= int(y) and int(y) + height + gui._px(40) <= 1080


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 monitor API")
def test_monitor_work_area_on_windows(tk_root):
    tk_root.deiconify()
    tk_root.update_idletasks()
    left, top, right, bottom = gui._monitor_work_area(tk_root)
    assert right - left > 0 and bottom - top > 0
    tk_root.withdraw()


def test_scale_never_shrinks_below_one(restore_scale):
    assert gui._apply_ui_scale(_FakeRoot(72.0)) == 1.0
    assert gui.SIDEBAR_CONFIG == gui._BASE_SIDEBAR_CONFIG


def test_named_fonts_use_an_installed_family(tk_root):
    fonts = gui._init_named_fonts(tk_root)
    assert set(fonts) == {"body", "small", "strong", "heading", "title", "mono"}
    for font in fonts.values():
        family = font.cget("family")
        # The bug this replaces: a nested family tuple parsed as "Segoe UI Arial".
        assert " Arial" not in family
        assert font.actual("family").lower() == family.lower()
    if sys.platform == "win32" and "Segoe UI" in tkfont.families(tk_root):
        assert fonts["body"].actual("family") == "Segoe UI"
    # Re-initialising on the same root reuses the fonts instead of raising.
    again = gui._init_named_fonts(tk_root)
    assert again["body"] is fonts["body"]


def test_ttk_style_resolves_to_named_font(tk_root):
    fonts = gui._init_named_fonts(tk_root)
    style = ttk.Style(tk_root)
    style.configure("QW7Probe.TLabel", font=fonts["body"])
    spec = style.lookup("QW7Probe.TLabel", "font")
    assert spec == "DaspBody"
    assert tkfont.Font(root=tk_root, font=spec).actual("family") == fonts["body"].cget("family")


def test_float_column_width_is_the_widest_formatted_value(tk_root):
    """Width comes from real measurements in the row font, whatever its face or scale."""
    font = tkfont.nametofont("TkDefaultFont", root=tk_root)
    values = pd.Series([0.5, 1.23456e-05, 12.0, np.nan])
    assert gui._float_column_text_width(values, font) == font.measure("1.23456e-05")


def test_float_column_width_handles_empty_and_large_columns(tk_root):
    font = tkfont.nametofont("TkDefaultFont", root=tk_root)
    assert gui._float_column_text_width(pd.Series([np.nan, np.inf]), font) == 0
    # The widest value sits past the first 500 rows; the extremes still catch it.
    values = pd.Series([0.5] * 2000 + [-1.23456e-05])
    assert gui._float_column_text_width(values, font) == font.measure("-1.23456e-05")


def test_dpi_awareness_is_noop_off_windows(monkeypatch):
    monkeypatch.setattr(gui.sys, "platform", "linux")
    assert gui._enable_windows_dpi_awareness() is False
