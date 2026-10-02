"""High-DPI scaling helpers and named fonts in the GUI module (QW7)."""

from __future__ import annotations

import subprocess
import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

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
    saved = (gui._UI_SCALE, gui._SCREEN_SIZE, dict(gui.SPACING), dict(gui.SIDEBAR_CONFIG))
    yield
    gui._UI_SCALE, gui._SCREEN_SIZE = saved[0], saved[1]
    gui.SPACING.clear()
    gui.SPACING.update(saved[2])
    gui.SIDEBAR_CONFIG.clear()
    gui.SIDEBAR_CONFIG.update(saved[3])


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


def test_px_geometry_clamps_to_screen(restore_scale):
    # Peak calculator at 150% on a 1920x1080 panel: 780x1080 would not fit.
    gui._apply_ui_scale(_FakeRoot(144.0, screen=(1920, 1080)))
    assert gui._px_geometry("520x720") == "780x972"


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


_MEASURE_AT_96_DPI = """
import tkinter as tk, tkinter.font as tkfont
r = tk.Tk(); r.withdraw()
r.tk.call("tk", "scaling", 96 / 72)
print(tkfont.nametofont("TkDefaultFont", root=r).measure("1.23456e-05"))
r.destroy()
"""


@pytest.fixture(scope="module")
def sci_text_px_96() -> int:
    """Width of '1.23456e-05' in the Treeview row font at 96 dpi.

    Measured in a fresh, DPI-unaware subprocess. In this process the numbers can be
    off: pyplot's TkAgg figure manager declares per-monitor DPI awareness when no Tk
    mainloop is running, and some GUI tests create pyplot figures, after which Tk
    reports physical-pixel widths.
    """
    out = subprocess.run(
        [sys.executable, "-c", _MEASURE_AT_96_DPI],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if out.returncode != 0:
        pytest.skip(f"Tk unavailable in subprocess: {out.stderr.strip()[-200:]}")
    return int(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("scale", [1.25, 1.5, 1.75, 2.0])
def test_scaled_result_column_fits_scientific_notation(sci_text_px_96, restore_scale, scale):
    """The default 80 px results column must still hold '1.23456e-05' once scaled.

    Text width grows linearly with the display scale (points follow `tk scaling`), so
    the 96-dpi width times the scale stands in for a real high-dpi measurement.
    """
    gui._apply_ui_scale(_FakeRoot(96.0 * scale))
    text_px_96 = sci_text_px_96
    cell_padding = gui._px(8)
    assert gui._px(80) >= text_px_96 * scale + cell_padding


def test_dpi_awareness_is_noop_off_windows(monkeypatch):
    monkeypatch.setattr(gui.sys, "platform", "linux")
    assert gui._enable_windows_dpi_awareness() is False
