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
    """Owner window on a 1920x1080 screen (no Win32 handle, so the screen fallback runs)."""

    def __init__(self, state: str = "normal") -> None:
        super().__init__(144.0, screen=(1920, 1080))
        self._state = state

    def winfo_ismapped(self) -> bool:
        return self._state != "withdrawn"

    def winfo_toplevel(self) -> "_FakeOwner":
        return self

    def state(self) -> str:
        return self._state

    def winfo_rootx(self) -> int:
        return -32000 if self._state == "iconic" else 100

    def winfo_rooty(self) -> int:
        return -32000 if self._state == "iconic" else 100

    def winfo_width(self) -> int:
        return 160 if self._state == "iconic" else 600

    def winfo_height(self) -> int:
        return 30 if self._state == "iconic" else 400


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


def test_px_geometry_centres_on_a_normal_owner(restore_scale):
    gui._apply_ui_scale(_FakeRoot(96.0))
    assert gui._px_geometry("200x100", _FakeOwner()) == "200x100+300+250"


@pytest.mark.parametrize("state", ["iconic", "withdrawn"])
def test_px_geometry_centres_on_the_work_area_when_owner_is_not_shown(restore_scale, state):
    # An iconified owner sits at -32000,-32000; centring on it would put the dialog
    # off-screen (then clamped into the top-left corner).
    gui._apply_ui_scale(_FakeRoot(96.0))
    assert gui._px_geometry("200x100", _FakeOwner(state)) == "200x100+860+490"


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 monitor API")
def test_monitor_work_area_uses_win32_not_the_fallback(tk_root, monkeypatch):
    """The Win32 path must answer; a silent fallback to the Tk screen size fails this."""
    import ctypes
    from ctypes import wintypes

    fallbacks = []
    monkeypatch.setattr(gui.logger, "debug", lambda msg, *a, **k: fallbacks.append(msg % a))
    tk_root.geometry("200x100+100+100")  # on the primary monitor
    tk_root.deiconify()
    tk_root.update()
    area = gui._monitor_work_area(tk_root)
    tk_root.withdraw()
    assert not fallbacks, fallbacks
    # Independent API: the primary monitor's work area (SPI_GETWORKAREA = 0x30).
    rect = wintypes.RECT()
    assert ctypes.windll.user32.SystemParametersInfoW(0x30, 0, ctypes.byref(rect), 0)
    assert area == (rect.left, rect.top, rect.right, rect.bottom)


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


# Review counterexamples: widest values after row 500 that are not the min, max or
# smallest magnitude; and equal-length strings of different width ('+' vs '-').
_COUNTEREXAMPLES = {
    "late_rows": [0.5] * 500 + [1e-12, 1e5, 1.23456e-5],
    "equal_length": [1.23456e-05, 1.23456e10, 0.5],
    "signed": [0.5] * 2000 + [-1.23456e-05],
}


def _brute_force_width(values: list[float], font: tkfont.Font) -> int:
    return max(font.measure(f"{v:.6g}") for v in values if np.isfinite(v))


@pytest.mark.parametrize("name", sorted(_COUNTEREXAMPLES))
def test_float_column_width_is_the_true_widest_value(tk_root, name):
    font = tkfont.nametofont("TkDefaultFont", root=tk_root)
    values = _COUNTEREXAMPLES[name]
    expected = _brute_force_width(values, font)
    assert gui._float_column_text_width(pd.Series(values), font) == expected


def test_plus_exponent_is_wider_than_minus_exponent(tk_root):
    font = tkfont.nametofont("TkDefaultFont", root=tk_root)
    assert font.measure("1.23456e+10") > font.measure("1.23456e-05")
    width = gui._float_column_text_width(pd.Series([1.23456e-05, 1.23456e10]), font)
    assert width == font.measure("1.23456e+10")


def test_float_column_width_of_an_empty_column_is_zero(tk_root):
    font = tkfont.nametofont("TkDefaultFont", root=tk_root)
    assert gui._float_column_text_width(pd.Series([np.nan, np.inf]), font) == 0


@pytest.fixture(params=[1.25, 1.5, 2.0], ids=lambda s: f"{int(s * 100)}pct")
def scaled_tk_root(request, restore_scale):
    """A fresh root whose `tk scaling` and _UI_SCALE both match the given display scale.

    Fonts measured on this root are at that scale, so text widths are real
    measurements rather than a 96-dpi width multiplied by the scale.
    """
    scale = request.param
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    root.tk.call("tk", "scaling", scale * 96 / 72)
    gui._apply_ui_scale(_FakeRoot(96.0 * scale))
    yield root
    root.destroy()


@pytest.mark.parametrize("name", sorted(_COUNTEREXAMPLES))
def test_results_columns_fit_their_text_at_scale(scaled_tk_root, name):
    """Every float cell, measured at 125/150/200%, fits its Results column with padding."""
    row_font = tkfont.Font(root=scaled_tk_root, font="TkDefaultFont")
    values = _COUNTEREXAMPLES[name]
    tk_cell_padding = 2 * gui._px(4)
    for col in ("RMSEcv", "RMSE_Q1", "F1_Class0", "R2cv"):
        width = gui._results_column_width(col, pd.Series(values), row_font)
        for v in values:
            text = f"{v:.6g}"
            assert row_font.measure(text) + tk_cell_padding <= width, (col, text, width)


def test_dpi_awareness_is_noop_off_windows(monkeypatch):
    monkeypatch.setattr(gui.sys, "platform", "linux")
    assert gui._enable_windows_dpi_awareness() is False
