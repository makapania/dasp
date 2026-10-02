"""High-DPI scaling helpers and named fonts in the GUI module (QW7)."""

from __future__ import annotations

import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

import pytest

import spectral_predict_gui_optimized as gui


class _FakeRoot:
    """Stand-in for a Tk root that reports a fixed dpi."""

    def __init__(self, dpi: float) -> None:
        self._dpi = dpi

    def winfo_fpixels(self, _distance: str) -> float:
        return self._dpi


@pytest.fixture
def restore_scale():
    """Put the module scale back to 96 dpi so other tests see the base constants."""
    yield
    gui._apply_ui_scale(_FakeRoot(96.0))


@pytest.fixture
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # no display
        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.mark.skipif(sys.platform == "darwin", reason="scale is pinned to 1.0 on macOS")
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


def test_dpi_awareness_is_noop_off_windows(monkeypatch):
    monkeypatch.setattr(gui.sys, "platform", "linux")
    assert gui._enable_windows_dpi_awareness() is False
